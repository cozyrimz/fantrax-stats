#!/usr/bin/env python3
"""Measure how evenly the league's scoring spreads value.

All metrics are computed from season stat totals and a weight matrix, so the
same functions serve both the current scoring and any proposed alternative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import fantrax_data as fx
import league_config as lc

# A player at this fraction of the median starter's output is close enough to
# start, so counting them measures how deep the usable pool really is.
USABLE_THRESHOLD = 0.80

TOP_N = 50


def load_weights(output_dir: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads((output_dir / "weights.json").read_text(encoding="utf-8"))
    return {pos: info["weights"] for pos, info in data["positions"].items()}


def load_games(output_dir: Path, season: str = None) -> pd.DataFrame:
    if season:
        path = output_dir / ("games_%s.parquet" % season)
        if not path.exists():
            raise SystemExit("No games for season %s at %s" % (season, path))
        return pd.read_parquet(path)
    paths = sorted(output_dir.glob("games_*.parquet"))
    if not paths:
        raise SystemExit("No games_*.parquet in %s — run build_dataset.py first" % output_dir)
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def load_totals(output_dir: Path, season: str = None) -> pd.DataFrame:
    if season:
        path = output_dir / ("season_totals_%s.csv" % season)
        if not path.exists():
            raise SystemExit("No totals for season %s at %s" % (season, path))
        return pd.read_csv(path)
    paths = sorted(output_dir.glob("season_totals_*.csv"))
    if not paths:
        raise SystemExit("No season_totals_*.csv in %s — run build_dataset.py first" % output_dir)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def attach_archetypes(totals: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    path = output_dir / "archetypes.csv"
    if not path.exists():
        totals["archetype"] = "Unclassified"
        totals["archetypeFull"] = totals["position"] + " Unclassified"
        return totals
    archetypes = pd.read_csv(path)
    merged = totals.merge(
        archetypes[["playerId", "season", "archetype", "archetypeFull", "usage"]],
        on=["playerId", "season"],
        how="left",
    )
    merged["archetype"] = merged["archetype"].fillna("Fringe")
    merged["archetypeFull"] = merged["archetypeFull"].fillna(merged["position"] + " Fringe")
    return merged


def score_totals(totals: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.Series:
    """Recompute season fantasy points from stat totals and a weight matrix.

    Scoring is linear in the per-game stats, so applying weights to season totals
    gives exactly the same result as scoring each game and summing.
    """
    points = pd.Series(0.0, index=totals.index)
    for position, table in weights.items():
        mask = totals["position"] == position
        if not mask.any():
            continue
        categories = [c for c in table if c in totals.columns]
        matrix = totals.loc[mask, categories].to_numpy(dtype=float)
        vector = np.array([table[c] for c in categories])
        points.loc[mask] = matrix @ vector
    return points


def season_count(frame: pd.DataFrame) -> int:
    """Number of separate seasons in the frame.

    Slot counts come from the league settings and apply to one season. When
    several seasons are pooled, each is its own market of that size, so every
    slot count scales with the number of seasons present. Without this, pooling
    three seasons would compare a thousand player-seasons against a single
    season's worth of slots and put replacement level far too high.
    """
    if "season" not in frame.columns:
        return 1
    return max(1, int(frame["season"].nunique()))


def allocate_starters(frame: pd.DataFrame, points: pd.Series) -> Dict[str, int]:
    """Work out how many players at each position the league actually starts.

    Managers must field the positional minimums, then fill the remaining flex
    slots with the best players available regardless of position, subject to the
    maximum they may start. The resulting counts are league-wide demand.
    """
    ranked = frame.assign(points=points).sort_values("points", ascending=False)
    seasons = season_count(frame)
    demand = {pos: lc.TEAMS * lc.MIN_ACTIVE[pos] * seasons for pos in lc.MIN_ACTIVE}
    ceiling = {pos: lc.TEAMS * lc.MAX_ACTIVE[pos] * seasons for pos in lc.MAX_ACTIVE}

    taken = {pos: 0 for pos in demand}
    used_rows = set()
    for pos in demand:
        subset = ranked[ranked["position"] == pos].head(demand[pos])
        taken[pos] = len(subset)
        used_rows.update(subset.index)

    remaining = lc.TEAMS * lc.STARTERS * seasons - sum(taken.values())
    for idx, row in ranked.iterrows():
        if remaining <= 0:
            break
        if idx in used_rows:
            continue
        pos = row["position"]
        if pos not in taken or taken[pos] >= ceiling[pos]:
            continue
        taken[pos] += 1
        used_rows.add(idx)
        remaining -= 1

    return taken


def replacement_levels(frame: pd.DataFrame, points: pd.Series, demand: Dict[str, int]) -> Dict[str, float]:
    """Points of the best player at each position who is not a starter."""
    levels = {}
    ranked = frame.assign(points=points)
    for pos, count in demand.items():
        subset = ranked[ranked["position"] == pos].sort_values("points", ascending=False)
        if subset.empty:
            levels[pos] = 0.0
        elif len(subset) > count:
            levels[pos] = float(subset.iloc[count]["points"])
        else:
            levels[pos] = float(subset.iloc[-1]["points"])
    return levels


def gini(values: np.ndarray) -> float:
    """Inequality of a non-negative distribution: 0 is perfectly flat, 1 maximally concentrated."""
    clipped = np.clip(np.asarray(values, dtype=float), 0, None)
    if clipped.size == 0 or clipped.sum() == 0:
        return 0.0
    ordered = np.sort(clipped)
    n = ordered.size
    index = np.arange(1, n + 1)
    return float((2 * index - n - 1).dot(ordered) / (n * ordered.sum()))


def category_contributions(
    frame: pd.DataFrame, weights: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Points each player earned from each category."""
    categories = sorted({c for table in weights.values() for c in table if c in frame.columns})
    contributions = pd.DataFrame(0.0, index=frame.index, columns=categories)
    for position, table in weights.items():
        mask = frame["position"] == position
        if not mask.any():
            continue
        for category in categories:
            weight = table.get(category, 0.0)
            if weight:
                contributions.loc[mask, category] = frame.loc[mask, category] * weight
    return contributions


def category_profile(
    frame: pd.DataFrame, contributions: pd.DataFrame, points: pd.Series, position: str
) -> pd.DataFrame:
    """How much each category drives scoring, and whether it rewards volume.

    A category with a large share of points that varies little between players is
    paying for time on the pitch rather than for anything a manager can target.
    """
    mask = frame["position"] == position
    if not mask.any():
        return pd.DataFrame()

    subset = contributions[mask]
    positive = subset.clip(lower=0)
    total_positive = positive.sum(axis=1).replace(0, np.nan)
    share = positive.div(total_positive, axis=0)

    games = frame.loc[mask, "GP"]
    rows = []
    for category in subset.columns:
        values = subset[category]
        if (values == 0).all():
            continue
        mean_share = float(share[category].mean())
        per_game = values / games.replace(0, np.nan)
        rows.append(
            {
                "category": category,
                "totalPoints": float(values.sum()),
                "meanShare": mean_share,
                "shareOfAllPoints": float(values.sum() / positive.sum().sum()) if positive.sum().sum() else 0.0,
                "spread": float(per_game.std() / per_game.mean()) if per_game.mean() else 0.0,
                "corrWithGames": float(values.corr(games)) if values.std() else 0.0,
            }
        )

    profile = pd.DataFrame(rows).sort_values("totalPoints", ascending=False)
    # Volume categories pay a lot, vary little between players, and track games
    # played. That combination is what lets high-minute grinders out-earn
    # specialists.
    profile["volumeScore"] = (
        profile["shareOfAllPoints"].abs()
        * profile["corrWithGames"].clip(lower=0)
        / profile["spread"].replace(0, np.nan)
    ).fillna(0.0)
    return profile.reset_index(drop=True)


def parse_game_dates(games: pd.DataFrame) -> pd.Series:
    """Turn the game log's "May 4" style dates into real timestamps.

    The log omits the year, so it is inferred from the season: August onwards
    belongs to the first calendar year, January onwards to the second.
    """
    start_year = games["season"].str.slice(0, 4).astype(int)
    parsed = pd.to_datetime(games["date"] + " " + start_year.astype(str), format="%b %d %Y", errors="coerce")
    rolled = pd.to_datetime(
        games["date"] + " " + (start_year + 1).astype(str), format="%b %d %Y", errors="coerce"
    )
    # Months before August fall in the second half of the season.
    return parsed.where(parsed.dt.month >= 8, rolled)


def churn_metrics(
    games: pd.DataFrame, weights: Dict[str, Dict[str, float]], top_n: int = 30, elite_n: int = 20
) -> Dict[str, float]:
    """How much the weekly scoring leaderboard rotates.

    If the same names fill the best scores every week, nobody on waivers can
    help and the pool is effectively static. Measuring the weekly leaderboard
    directly captures that far better than a season-long distribution does.
    """
    working = games.copy()
    working["points"] = score_totals(working, weights)
    working["week"] = parse_game_dates(working).dt.to_period("W")
    working = working.dropna(subset=["week"])
    if working.empty:
        return {}

    season_totals_points = working.groupby("playerId")["points"].sum().sort_values(ascending=False)
    elite = set(season_totals_points.head(elite_n).index)

    appearances: Dict[str, int] = {}
    elite_slots = 0
    slots = 0
    for _, block in working.groupby(["season", "week"]):
        best = block.nlargest(min(top_n, len(block)), "points")
        for player_id in best["playerId"]:
            appearances[player_id] = appearances.get(player_id, 0) + 1
            slots += 1
            if player_id in elite:
                elite_slots += 1

    if not slots:
        return {}

    counts = np.array(sorted(appearances.values()))
    return {
        "distinctPlayersInWeeklyTop": len(appearances),
        "weeklyTopSlots": slots,
        "churnRatio": round(len(appearances) / slots, 3),
        "eliteShareOfWeeklyTop": round(elite_slots / slots, 3),
        "giniOfWeeklyTopAppearances": round(gini(counts), 3),
    }


def compute_metrics(
    frame: pd.DataFrame,
    weights: Dict[str, Dict[str, float]],
    label: str = "current",
) -> Dict[str, object]:
    points = score_totals(frame, weights)
    working = frame.assign(points=points)

    demand = allocate_starters(frame, points)
    levels = replacement_levels(frame, points, demand)

    working["replacement"] = working["position"].map(levels)
    working["vor"] = working["points"] - working["replacement"]

    ranked = working.sort_values("points", ascending=False).reset_index(drop=True)
    rostered = ranked.head(min(lc.ROSTERED_POOL * season_count(frame), len(ranked)))

    positions: Dict[str, object] = {}
    for pos in lc.MIN_ACTIVE:
        subset = ranked[ranked["position"] == pos].sort_values("points", ascending=False)
        if subset.empty:
            continue
        count = demand[pos]
        starters = subset.head(count)
        starter_median = float(starters["points"].median()) if count else 0.0
        bar = USABLE_THRESHOLD * starter_median
        usable = int((subset["points"] >= bar).sum())
        best_free_agent = subset[~subset["playerId"].isin(rostered["playerId"])]

        positions[pos] = {
            "players": int(len(subset)),
            "starterDemand": count,
            # Replacement level is only meaningful if the pool extends well past
            # league-wide demand. A partial export makes the worst player in the
            # sample look like the replacement.
            "poolAdequate": bool(len(subset) >= count * 1.5),
            "replacementPoints": round(levels[pos], 1),
            "starterMedianPoints": round(starter_median, 1),
            "topPoints": round(float(subset.iloc[0]["points"]), 1),
            "eliteOverReplacement": round(float(starters.head(12)["points"].mean() - levels[pos]), 1)
            if count
            else 0.0,
            "eliteMultiple": round(float(starters.head(12)["points"].mean() / levels[pos]), 2)
            if levels[pos] > 0
            else None,
            "usablePlayers": usable,
            "usableSurplus": usable - count,
            "giniOfValue": round(gini(subset["vor"].to_numpy()), 3),
            "bestFreeAgentPoints": round(float(best_free_agent.iloc[0]["points"]), 1)
            if not best_free_agent.empty
            else None,
            "waiverGap": round(
                float(1 - best_free_agent.iloc[0]["points"] / starters.iloc[-1]["points"]), 3
            )
            if not best_free_agent.empty and count and starters.iloc[-1]["points"] > 0
            else None,
        }

    # Scaled the same way as slot counts, so the elite tier means the same thing
    # whether one season or several are in the frame.
    top = ranked.head(min(TOP_N * season_count(frame), len(ranked)))
    contributions = category_contributions(working, weights)

    metrics: Dict[str, object] = {
        "label": label,
        "players": int(len(ranked)),
        "overall": {
            "giniOfValue": round(gini(working["vor"].to_numpy()), 3),
            "giniOfPointsRostered": round(gini(rostered["points"].to_numpy()), 3),
            "top10ShareOfTop50": round(
                float(top.head(10)["points"].sum() / top["points"].sum()), 3
            )
            if len(top)
            else 0.0,
            "rank1": round(float(ranked.iloc[0]["points"]), 1),
            "rank10": round(float(ranked.iloc[9]["points"]), 1) if len(ranked) > 9 else None,
            "rank25": round(float(ranked.iloc[24]["points"]), 1) if len(ranked) > 24 else None,
            "rank50": round(float(ranked.iloc[49]["points"]), 1) if len(ranked) > 49 else None,
            "rank100": round(float(ranked.iloc[99]["points"]), 1) if len(ranked) > 99 else None,
            "topToFiftiethRatio": round(
                float(ranked.iloc[0]["points"] / ranked.iloc[49]["points"]), 2
            )
            if len(ranked) > 49 and ranked.iloc[49]["points"] > 0
            else None,
        },
        "positions": positions,
        "poolAdequate": all(info["poolAdequate"] for info in positions.values()),
        "topPositionMix": top["position"].value_counts().to_dict(),
        "topArchetypeMix": top["archetypeFull"].value_counts().to_dict()
        if "archetypeFull" in top.columns
        else {},
        "starterDemand": demand,
    }

    return metrics, working, contributions


def volume_report(
    working: pd.DataFrame, contributions: pd.DataFrame, weights: Dict[str, Dict[str, float]]
) -> Dict[str, pd.DataFrame]:
    points = working["points"]
    return {
        pos: category_profile(working, contributions, points, pos)
        for pos in lc.MIN_ACTIVE
        if (working["position"] == pos).any()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Where shared artifacts live (weights, game logs); defaults to --output-dir",
    )
    parser.add_argument("--season", help="Season label; defaults to all seasons pooled")
    args = parser.parse_args()

    data_dir = args.data_dir or args.output_dir
    weights = load_weights(data_dir)
    totals = attach_archetypes(load_totals(data_dir, args.season), data_dir)

    metrics, working, contributions = compute_metrics(totals, weights, label="current")
    profiles = volume_report(working, contributions, weights)
    metrics["churn"] = churn_metrics(load_games(data_dir, args.season), weights)

    season_tag = args.season or "all"
    (args.output_dir / ("metrics_%s.json" % season_tag)).write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    for pos, profile in profiles.items():
        profile.to_csv(args.output_dir / ("category_profile_%s_%s.csv" % (season_tag, pos)), index=False)
    working.to_csv(args.output_dir / ("player_values_%s.csv" % season_tag), index=False)

    print("Starter demand: %s" % metrics["starterDemand"])
    print("Top-50 position mix: %s" % metrics["topPositionMix"])
    print(
        "Curve: #1 %.0f  #10 %s  #25 %s  #50 %s  #100 %s"
        % (
            metrics["overall"]["rank1"],
            metrics["overall"]["rank10"],
            metrics["overall"]["rank25"],
            metrics["overall"]["rank50"],
            metrics["overall"]["rank100"],
        )
    )
    print("Gini of value over replacement: %.3f" % metrics["overall"]["giniOfValue"])
    if metrics["churn"]:
        print(
            "Weekly leaderboard: %d distinct players fill %d top-30 slots (churn %.3f), "
            "season's top 20 hold %.0f%% of them"
            % (
                metrics["churn"]["distinctPlayersInWeeklyTop"],
                metrics["churn"]["weeklyTopSlots"],
                metrics["churn"]["churnRatio"],
                100 * metrics["churn"]["eliteShareOfWeeklyTop"],
            )
        )
    for pos, info in metrics["positions"].items():
        print(
            "  %s: demand %d, replacement %.0f, elite/replacement %s, usable %d (surplus %+d), waiver gap %s"
            % (
                pos,
                info["starterDemand"],
                info["replacementPoints"],
                info["eliteMultiple"],
                info["usablePlayers"],
                info["usableSurplus"],
                info["waiverGap"],
            )
        )
    print("Wrote metrics and category profiles to %s" % args.output_dir)


if __name__ == "__main__":
    main()
