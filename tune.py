#!/usr/bin/env python3
"""Tune a scoring proposal to hit positional balance, then round it for entry.

The two problems separate cleanly. Cutting the volume categories decides *which*
players within a position have value, and a single multiplier per position
decides *how many* of that position reach the top of the league. Because a
per-position multiplier cannot reorder the players inside that position, the
multipliers can be solved for directly: pick the rank each position deserves
given how many of them the league must start, then scale each position so the
player at that rank is worth the same everywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import diagnose
import league_config as lc
import simulate

POSITION_ORDER = ["D", "M", "F", "G"]


def marginal_ranks(top_n: int = 50) -> Dict[str, int]:
    """The rank in each position that a balanced top `top_n` would reach."""
    demand = lc.BASELINE_STARTERS
    total = sum(demand.values())
    return {pos: max(1, round(top_n * count / total)) for pos, count in demand.items()}


def solve_scalars(
    output_dir: Path, base_proposal: Dict, season: str = None, top_n: int = 50
) -> Dict[str, float]:
    """Find the per-position multiplier that equalises the marginal starter."""
    current = diagnose.load_weights(output_dir)
    trimmed, _ = simulate.apply_proposal(current, base_proposal)
    totals = diagnose.attach_archetypes(diagnose.load_totals(output_dir, season), output_dir)
    _, working, _ = diagnose.compute_metrics(totals, trimmed)

    # A pooled frame holds one elite tier per season, so the rank that marks the
    # edge of that tier scales with the number of seasons, exactly as slot counts
    # do in diagnose.
    ranks = marginal_ranks(top_n * diagnose.season_count(working))
    marginal: Dict[str, float] = {}
    for position, rank in ranks.items():
        subset = working[working["position"] == position].sort_values("points", ascending=False)
        if len(subset) < rank:
            continue
        marginal[position] = float(subset.iloc[rank - 1]["points"])

    usable = {p: v for p, v in marginal.items() if v > 0}
    if not usable:
        raise SystemExit("Cannot solve scalars: no position has positive points at its marginal rank")

    # Anchor on midfielders so the overall scale of the league stays recognisable.
    anchor = usable.get("M") or sum(usable.values()) / len(usable)
    return {p: round(anchor / v, 3) for p, v in usable.items()}


def concentrate_scale(
    totals: pd.DataFrame,
    weights: Dict[str, Dict[str, float]],
    scalars: Dict[str, float],
    ranks: Dict[str, int],
    max_edits: int,
    bounds: Tuple[float, float] = (0.3, 3.0),
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """Deliver each position's scalar through a few categories instead of all of them.

    Scaling every category a position scores is the clean way to move that
    position, but it means retyping fifty weights per position. The same shift in
    a position's total can be had by moving only some categories harder, at the
    cost of reordering the players inside the position, since players differ in
    how much of their scoring comes from those categories.

    The categories chosen are the ones that are both large and evenly spread, so
    the reordering is as small as possible. That is the same property the volume
    trims exploit: passes, recoveries and duels are worth a lot in aggregate and
    barely distinguish players, which makes them the cheapest place to take
    points from or add them to.
    """
    concentrated = {p: dict(table) for p, table in weights.items()}
    rows = []

    for position, factor in scalars.items():
        if factor == 1.0 or position not in weights:
            continue
        subset = totals[totals["position"] == position]
        table = weights[position]
        scored = [c for c in table if c in subset.columns]
        if subset.empty or not scored:
            continue

        contribution = pd.DataFrame(
            {c: subset[c].astype(float) * table[c] for c in scored}, index=subset.index
        )
        # The total has to include the penalties, or the factor would be solved
        # against an inflated baseline. Only positively weighted categories are
        # candidates to carry the change, since raising a penalty to move a
        # position up makes no sense.
        player_total = contribution.sum(axis=1)
        categories = [c for c in scored if table[c] > 0]
        active = player_total > 0
        if not active.any() or not categories:
            continue

        share = contribution.loc[active, categories].div(player_total[active], axis=0)
        # Evenly spread means every player draws a similar fraction of their
        # points from the category, so moving it barely changes the order.
        spread = share.std() / share.mean().replace(0, np.nan)
        volume = contribution[categories].sum()
        ranked = sorted(
            (c for c in categories if volume[c] > 0),
            key=lambda c: (spread.get(c, np.inf), -volume[c]),
        )

        # A uniform scalar moves the whole position by the same proportion, so it
        # is enough to reproduce its effect on the one player who decides the
        # position's share of the elite tier: the player at the marginal rank.
        # Matching the position's aggregate instead would overshoot at the top,
        # because the steady categories accumulate with games played and the best
        # players play the most.
        rank = ranks.get(position)
        ordered = player_total.sort_values(ascending=False)
        if not rank or len(ordered) < rank:
            continue
        marginal_player = ordered.index[rank - 1]
        target = float(ordered.iloc[rank - 1]) * factor

        # Weights are picked straight off the grid of values a human would type,
        # rather than solved as a fraction and rounded afterwards. When one
        # category carries a whole position, rounding it after the fact throws
        # away enough of the change to undo the balance it was solved for.
        chosen: Dict[str, float] = {}
        moved = pd.Series(0.0, index=subset.index)
        tolerance = 0.002 * abs(target)

        for category in ranked[:max_edits]:
            residual = target - float(player_total[marginal_player] + moved[marginal_player])
            if abs(residual) <= tolerance:
                break
            per_unit = float(subset.loc[marginal_player, category])
            if per_unit <= 0:
                continue
            original = table[category]
            ideal = original + residual / per_unit
            low, high = original * bounds[0], original * bounds[1]
            step = simulate.snap_step(max(min(ideal, high), low))
            picked = round(round(max(min(ideal, high), low) / step) * step, 2)
            if picked <= 0 or picked == original:
                continue
            chosen[category] = picked
            moved = moved + subset[category].astype(float) * (picked - original)

        if not chosen:
            rows.append((position, factor, np.nan, np.nan, "none available"))
            continue

        for category, picked in chosen.items():
            concentrated[position][category] = picked

        updated = player_total + moved
        delivered = float(updated.sort_values(ascending=False).iloc[rank - 1]) / float(
            ordered.iloc[rank - 1]
        )
        order_kept = float(player_total[active].corr(updated[active], method="spearman"))
        rows.append(
            (
                position,
                factor,
                round(delivered, 3),
                round(order_kept, 4),
                ", ".join(
                    "%s %g->%g" % (c, table[c], v) for c, v in chosen.items()
                ),
            )
        )

    report = pd.DataFrame(
        rows,
        columns=["position", "wanted", "deliveredAtMargin", "orderKept", "carriedBy"],
    )
    return concentrated, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Where shared artifacts live (weights, game logs); defaults to --output-dir",
    )
    parser.add_argument("--base", type=Path, required=True, help="Proposal holding the category trims")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the tuned proposal")
    parser.add_argument("--season", help="Season label; defaults to all seasons pooled")
    parser.add_argument(
        "--name",
        default="recommended",
        help="Name for the tuned proposal, kept distinct from the base it was solved from",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=2.0,
        help=(
            "Clamp on how far a position may be scaled. A binding clamp leaves the "
            "mix unbalanced by construction, so it is set wide and warns when hit"
        ),
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.85,
        help="Keep the changes responsible for this share of the points moved, dropping the tail",
    )
    parser.add_argument(
        "--max-edits",
        type=int,
        default=10,
        help=(
            "Deliver each position's scalar through at most this many categories "
            "instead of every category it scores, to cut down manual entry. Zero "
            "scales every category the position scores"
        ),
    )
    parser.add_argument(
        "--max-lever",
        type=float,
        default=1.33,
        help=(
            "Limit on how far any single category may be moved when carrying a "
            "position scalar. Tighter keeps the order of players within the "
            "position closer to the full scalar, at the cost of more categories"
        ),
    )
    parser.add_argument(
        "--scale-bias",
        default="{}",
        help=(
            "JSON map of extra per-position multipliers applied after the scalar "
            "solve, e.g. '{\"F\": 1.15, \"D\": 0.98}' to lift forwards slightly "
            "past strict slot balance"
        ),
    )
    args = parser.parse_args()

    try:
        scale_bias = json.loads(args.scale_bias)
    except json.JSONDecodeError as exc:
        raise SystemExit("Invalid --scale-bias JSON: %s" % exc) from exc

    data_dir = args.data_dir or args.output_dir
    base = json.loads(args.base.read_text(encoding="utf-8"))
    current = diagnose.load_weights(data_dir)
    totals = diagnose.attach_archetypes(
        diagnose.load_totals(data_dir, args.season), data_dir
    )
    tier = diagnose.TOP_N * diagnose.season_count(totals)

    # Round the trims for entry before anything is solved against them. Solving
    # first and rounding afterwards silently discards trims the solution depended
    # on, so the proposal that gets written no longer behaves like the one that
    # was verified.
    trim_only, _ = simulate.apply_proposal(current, {**base, "scale": {}})
    trims = simulate.to_clean_proposal(
        current,
        trim_only,
        args.name,
        base.get("description", ""),
        totals=totals,
        coverage=args.coverage,
    )
    trims["scale"] = base.get("scale") or {}
    print(
        "Trims rounded for entry: %d weights across %s"
        % (sum(len(v) for v in trims["set"].values()), ", ".join(sorted(trims["set"])))
    )

    solved = solve_scalars(data_dir, trims, args.season)

    # The solve measures the marginal player under the base proposal, which may
    # already scale a position. A solved factor is therefore relative to that
    # baseline and has to compose with it rather than replace it.
    base_scale = trims["scale"]
    composed = {p: base_scale.get(p, 1.0) * v for p, v in solved.items()}
    for position, factor in base_scale.items():
        composed.setdefault(position, factor)
    for position, factor in scale_bias.items():
        composed[position] = composed.get(position, 1.0) * factor
    if scale_bias:
        print("Scale bias applied: %s -> %s" % (scale_bias, {p: round(composed[p], 3) for p in composed}))

    floor, ceiling = 1 / args.max_scale, args.max_scale
    clamped = {p: round(min(ceiling, max(floor, v)), 3) for p, v in composed.items()}
    binding = [p for p, v in composed.items() if not floor <= v <= ceiling]
    print("Solved position scalars: %s" % clamped)
    if binding:
        print(
            "  Warning: %s hit the --max-scale limit of %g (wanted %s), so the mix "
            "below will fall short of balance."
            % (
                ", ".join(sorted(binding)),
                args.max_scale,
                ", ".join("%s %.3f" % (p, composed[p]) for p in sorted(binding)),
            )
        )

    tuned = dict(trims)
    tuned["scale"] = clamped
    result = simulate.run(data_dir, tuned, args.season)
    print("Top %d by position with a full position scalar: %s" % (tier, result["after"]["topPositionMix"]))
    print("Balanced target: %s" % marginal_ranks(tier))

    proposal = dict(trims)
    if args.max_edits:
        trimmed, _ = simulate.apply_proposal(current, {**trims, "scale": {}})
        concentrated, concentration = concentrate_scale(
            totals,
            trimmed,
            clamped,
            marginal_ranks(tier),
            args.max_edits,
            bounds=(1 / args.max_lever, args.max_lever),
        )
        levers: Dict[str, Dict[str, float]] = {}
        for position, table in concentrated.items():
            for category, value in table.items():
                if trimmed.get(position, {}).get(category) != value:
                    levers.setdefault(position, {})[category] = value
        merged = {p: dict(t) for p, t in trims["set"].items()}
        for position, table in levers.items():
            merged.setdefault(position, {}).update(table)
        proposal["set"] = merged
        proposal["scale"] = {}
        print("")
        print("Position scalars carried by a few categories instead of every one:")
        print(concentration.to_string(index=False))
    else:
        proposal["scale"] = clamped

    args.out.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")

    verified = simulate.run(data_dir, proposal, args.season)
    print("")
    print(
        "Wrote %s with %d enterable weights"
        % (args.out, sum(len(v) for v in proposal["set"].values()))
    )
    print(verified["comparison"].to_string(index=False))
    print("Top %d by position as written: %s" % (tier, verified["after"]["topPositionMix"]))


if __name__ == "__main__":
    main()
