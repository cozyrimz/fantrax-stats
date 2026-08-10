#!/usr/bin/env python3
"""Re-score every season under a proposed weight set and compare the outcome.

A proposal is a JSON file describing changes to the recovered weights:

    {
      "name": "trim-volume",
      "description": "why this exists",
      "scale": {"F": 1.15},
      "multiply": {"all": {"BR": 0.5}, "D": {"CLR": 0.6}},
      "set": {"F": {"G": 12}}
    }

"scale" multiplies every category a position scores, which moves that position
against the others without reordering players inside it. Under "multiply" and
"set", the key "all" applies to every position that scores the category. Because
scoring is linear, replaying weights over the real stat totals gives the exact
points every player would have finished with.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

import diagnose
import league_config as lc

POSITION_ORDER = ["D", "M", "F", "G"]


def apply_proposal(
    weights: Dict[str, Dict[str, float]], proposal: Dict
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    updated = copy.deepcopy(weights)
    changes: List[str] = []

    def targets(scope: str) -> List[str]:
        return list(updated.keys()) if scope == "all" else [scope]

    for scope, table in (proposal.get("multiply") or {}).items():
        for position in targets(scope):
            for category, factor in table.items():
                if category in updated.get(position, {}):
                    before = updated[position][category]
                    after = round(before * factor, 4)
                    if before != after:
                        updated[position][category] = after
                        changes.append("%s %s %g -> %g" % (position, category, before, after))

    for scope, table in (proposal.get("set") or {}).items():
        for position in targets(scope):
            if position not in updated:
                continue
            for category, value in table.items():
                if category not in updated[position] and scope == "all":
                    continue
                before = updated[position].get(category)
                if before != value:
                    updated[position][category] = value
                    changes.append(
                        "%s %s %s -> %g" % (position, category, "unset" if before is None else "%g" % before, value)
                    )

    # A single factor across every category a position scores, applied last so it
    # multiplies the finished table. It shifts that position's standing against
    # the others without reordering the players within it, which is the cleanest
    # way to correct positional imbalance. Applied any earlier, a category set to
    # an absolute value afterwards would escape the factor.
    for position, factor in (proposal.get("scale") or {}).items():
        if position not in updated or factor == 1:
            continue
        for category in list(updated[position]):
            updated[position][category] = round(updated[position][category] * factor, 4)
        changes.append("%s all categories x%g" % (position, factor))

    return updated, changes


def compare(before: Dict, after: Dict) -> pd.DataFrame:
    rows = [
        ("Gini of value over replacement", before["overall"]["giniOfValue"], after["overall"]["giniOfValue"], "lower"),
        ("Gini of points among rostered", before["overall"]["giniOfPointsRostered"], after["overall"]["giniOfPointsRostered"], "lower"),
        ("Top 10 share of top 50 points", before["overall"]["top10ShareOfTop50"], after["overall"]["top10ShareOfTop50"], "lower"),
        ("Best player vs rank 50", before["overall"]["topToFiftiethRatio"], after["overall"]["topToFiftiethRatio"], "lower"),
    ]
    if before.get("churn") and after.get("churn"):
        rows.append(
            (
                "Top 20 share of weekly top-30 slots",
                before["churn"]["eliteShareOfWeeklyTop"],
                after["churn"]["eliteShareOfWeeklyTop"],
                "lower",
            )
        )
        rows.append(
            (
                "Distinct players in weekly top 30",
                before["churn"]["distinctPlayersInWeeklyTop"],
                after["churn"]["distinctPlayersInWeeklyTop"],
                "higher",
            )
        )
    for pos in POSITION_ORDER:
        if pos not in before["positions"] or pos not in after["positions"]:
            continue
        rows.append(
            (
                "%s elite multiple of replacement" % pos,
                before["positions"][pos]["eliteMultiple"],
                after["positions"][pos]["eliteMultiple"],
                "lower",
            )
        )
        rows.append(
            (
                "%s usable surplus over demand" % pos,
                before["positions"][pos]["usableSurplus"],
                after["positions"][pos]["usableSurplus"],
                "higher",
            )
        )
    for pos in POSITION_ORDER:
        rows.append(
            (
                "%s in top 50" % pos,
                before["topPositionMix"].get(pos, 0),
                after["topPositionMix"].get(pos, 0),
                "balance",
            )
        )

    frame = pd.DataFrame(rows, columns=["metric", "current", "proposed", "better"])
    frame["change"] = frame.apply(
        lambda r: None
        if r["current"] is None or r["proposed"] is None
        else round(r["proposed"] - r["current"], 3),
        axis=1,
    )
    return frame


def movers(before_frame: pd.DataFrame, after_frame: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    before_ranked = before_frame.sort_values("points", ascending=False).reset_index(drop=True)
    after_ranked = after_frame.sort_values("points", ascending=False).reset_index(drop=True)

    before_rank = {row["playerId"]: i + 1 for i, row in before_ranked.iterrows()}
    after_rank = {row["playerId"]: i + 1 for i, row in after_ranked.iterrows()}

    rows = []
    for _, row in after_ranked.iterrows():
        pid = row["playerId"]
        if pid not in before_rank:
            continue
        rows.append(
            {
                "name": row["name"],
                "position": row["position"],
                "archetype": row.get("archetype", ""),
                "rankBefore": before_rank[pid],
                "rankAfter": after_rank[pid],
                "move": before_rank[pid] - after_rank[pid],
                "pointsBefore": round(
                    float(before_frame.loc[before_frame["playerId"] == pid, "points"].iloc[0]), 1
                ),
                "pointsAfter": round(float(row["points"]), 1),
            }
        )

    frame = pd.DataFrame(rows)
    risers = frame.sort_values("move", ascending=False).head(limit)
    fallers = frame.sort_values("move").head(limit)
    return risers, fallers


def run(output_dir: Path, proposal: Dict, season: str = None, with_churn: bool = True) -> Dict:
    weights = diagnose.load_weights(output_dir)
    totals = diagnose.attach_archetypes(diagnose.load_totals(output_dir, season), output_dir)

    proposed_weights, changes = apply_proposal(weights, proposal)

    before, before_frame, _ = diagnose.compute_metrics(totals, weights, label="current")
    after, after_frame, _ = diagnose.compute_metrics(totals, proposed_weights, label=proposal.get("name", "proposed"))

    if with_churn:
        games = diagnose.load_games(output_dir, season)
        before["churn"] = diagnose.churn_metrics(games, weights)
        after["churn"] = diagnose.churn_metrics(games, proposed_weights)

    comparison = compare(before, after)
    risers, fallers = movers(before_frame, after_frame)

    return {
        "proposal": proposal,
        "changes": changes,
        "weights": proposed_weights,
        "before": before,
        "after": after,
        "comparison": comparison,
        "risers": risers,
        "fallers": fallers,
    }


def snap_step(value: float) -> float:
    """The increment a weight of this size should be expressed in."""
    magnitude = abs(value)
    return 0.01 if magnitude < 0.2 else 0.05 if magnitude < 2 else 0.5 if magnitude < 10 else 1.0


def snap(value: float) -> float:
    """Round a weight to a value that is sensible to type into league settings."""
    return round(round(value / snap_step(value)) * snap_step(value), 2)


def to_clean_proposal(
    current: Dict[str, Dict[str, float]],
    proposed: Dict[str, Dict[str, float]],
    name: str,
    description: str = "",
    totals: pd.DataFrame = None,
    coverage: float = 0.97,
    protected: Iterable[str] = (),
) -> Dict:
    """Restate a tuned proposal as round numbers a human can type into Fantrax.

    Tuning happens with multipliers and position scalars, which produce values
    like 0.1495 across every category a position scores. League settings have to
    be edited by hand, so weights are snapped to sensible steps and, when season
    totals are supplied, only the changes that actually move points are kept:
    changes are ranked by how many points they shift and the tail that
    contributes the last few percent is dropped.

    Dropping is decided within a position, never across positions: the busiest
    position would otherwise use up the whole budget and a thinner one such as
    goalkeeper would lose every change it needs. Positions listed in `protected`
    keep all of their changes, which is required wherever a single factor is
    applied across a position, since applying it to only some categories would
    reorder the players inside that position instead of moving the position as a
    whole.
    """

    keep_all = set(protected)
    table: Dict[str, Dict[str, float]] = {}
    for position in POSITION_ORDER:
        counts = None
        if totals is not None:
            subset = totals[totals["position"] == position]
            counts = subset.sum(numeric_only=True)

        candidates = []
        for category, value in sorted((proposed.get(position) or {}).items()):
            before = current.get(position, {}).get(category)
            if before is None or before == value:
                continue
            cleaned = snap(value)
            if cleaned == before:
                continue
            volume = float(counts.get(category, 0.0)) if counts is not None else 1.0
            candidates.append((abs(cleaned - before) * volume, category, cleaned))

        if totals is not None and candidates and position not in keep_all:
            candidates.sort(reverse=True)
            budget = coverage * sum(impact for impact, _, _ in candidates)
            running = 0.0
            kept = []
            for entry in candidates:
                if running >= budget:
                    break
                running += entry[0]
                kept.append(entry)
            candidates = kept

        for _, category, cleaned in candidates:
            table.setdefault(position, {})[category] = cleaned

    return {"name": name, "description": description, "scale": {}, "multiply": {}, "set": table}


def print_result(result: Dict) -> None:
    proposal = result["proposal"]
    print("Proposal: %s" % proposal.get("name", "unnamed"))
    if proposal.get("description"):
        print("  %s" % proposal["description"])
    print("  %d weight changes" % len(result["changes"]))
    for change in result["changes"][:20]:
        print("    %s" % change)
    if len(result["changes"]) > 20:
        print("    ... %d more" % (len(result["changes"]) - 20))

    print("")
    print(result["comparison"].to_string(index=False))
    print("")

    before_mix = result["before"].get("topArchetypeMix") or {}
    after_mix = result["after"].get("topArchetypeMix") or {}
    if before_mix or after_mix:
        print("Player types in the top 50:")
        for archetype in sorted(set(before_mix) | set(after_mix)):
            was, now = before_mix.get(archetype, 0), after_mix.get(archetype, 0)
            if was or now:
                print("  %-32s %2d -> %2d  (%+d)" % (archetype, was, now, now - was))
        print("")
    print("Biggest risers:")
    print(result["risers"][["name", "position", "rankBefore", "rankAfter", "move"]].to_string(index=False))
    print("")
    print("Biggest fallers:")
    print(result["fallers"][["name", "position", "rankBefore", "rankAfter", "move"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--season", help="Season label; defaults to all seasons pooled")
    parser.add_argument(
        "--emit-clean",
        type=Path,
        help="Rewrite the proposal as round, directly enterable weights and verify it still works",
    )
    args = parser.parse_args()

    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    result = run(args.output_dir, proposal, args.season)
    print_result(result)

    if args.emit_clean:
        current = diagnose.load_weights(args.output_dir)
        clean = to_clean_proposal(
            current,
            result["weights"],
            proposal.get("name", "recommended"),
            proposal.get("description", ""),
            totals=diagnose.load_totals(args.output_dir, args.season),
        )
        args.emit_clean.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
        verified = run(args.output_dir, clean, args.season)
        print("")
        print("Rounded to %d enterable weights, re-checked:" % sum(len(v) for v in clean["set"].values()))
        print(verified["comparison"].to_string(index=False))

    name = proposal.get("name", args.proposal.stem)
    (args.output_dir / ("simulation_%s.json" % name)).write_text(
        json.dumps(
            {
                "proposal": proposal,
                "changes": result["changes"],
                "weights": result["weights"],
                "before": result["before"],
                "after": result["after"],
                "comparison": result["comparison"].to_dict(orient="records"),
                "risers": result["risers"].to_dict(orient="records"),
                "fallers": result["fallers"].to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print("")
    print("Wrote %s" % (args.output_dir / ("simulation_%s.json" % name)))


if __name__ == "__main__":
    main()
