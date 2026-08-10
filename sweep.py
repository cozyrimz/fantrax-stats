#!/usr/bin/env python3
"""Trace the trade-off between flattening the curve and diversifying player types.

Cutting the high-volume categories breaks up the dominance of centre backs and
holding midfielders, but those same categories act as a floor that every starter
collects, so cutting them also widens the gap between the best players and the
rest. This sweeps the depth of the cut so the balance point can be chosen from
evidence rather than guessed.

The sweep runs in both directions, because raising those categories is the only
move a linear scoring system has for narrowing the gap, and it re-solves the
position scalars at every point. Without that re-solve the trace would mostly
show positions drifting in and out of the elite tier, which is a separate problem
with its own fix, rather than the top-heaviness the sweep is meant to measure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

import diagnose
import simulate

# The categories that pay for time on the pitch, with how hard each is cut
# relative to the sweep factor.
VOLUME_CATEGORIES = {"BR": 1.0, "DW": 1.0, "AP": 1.0, "LBA": 0.6}
DEFENDER_CATEGORIES = {"CLR": 1.0, "AER": 0.6}


def proposal_for(trim: float, forward_scale: float, defender_scale: float) -> Dict:
    """Build a proposal that cuts volume categories to `trim` of their value."""

    def factor(strength: float) -> float:
        return round(1 - (1 - trim) * strength, 3)

    return {
        "name": "trim %.2f" % trim,
        "scale": {"F": forward_scale, "D": defender_scale},
        "multiply": {
            "all": {c: factor(s) for c, s in VOLUME_CATEGORIES.items()},
            "D": {c: factor(s) for c, s in DEFENDER_CATEGORIES.items()},
        },
        "set": {},
    }


def archetype_concentration(metrics: Dict) -> float:
    """Share of the top 50 held by its single most common player type."""
    mix = metrics.get("topArchetypeMix") or {}
    total = sum(mix.values())
    return max(mix.values()) / total if total else 0.0


def elite_multiple(metrics: Dict) -> float:
    """How many times replacement level the best player at a position is, averaged."""
    positions = metrics.get("positions") or {}
    values = [p["eliteMultiple"] for p in positions.values() if p.get("eliteMultiple")]
    return round(sum(values) / len(values), 3) if values else 0.0


def trace(
    output_dir: Path,
    season: str = None,
    forward_scale: float = 1.0,
    defender_scale: float = 1.0,
    with_churn: bool = False,
    rebalance: bool = True,
    levels: List[float] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for trim in levels or [0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.4]:
        proposal = proposal_for(trim, forward_scale, defender_scale)
        if rebalance:
            # Import here so the module stays usable if tune is being edited.
            import tune

            solved = tune.solve_scalars(output_dir, proposal, season)
            base_scale = proposal["scale"]
            proposal["scale"] = {
                p: round(base_scale.get(p, 1.0) * v, 3) for p, v in solved.items()
            }
        result = simulate.run(output_dir, proposal, season, with_churn=with_churn)
        after = result["after"]
        rows.append(
            {
                "volumeKept": trim,
                "gini": after["overall"]["giniOfValue"],
                "giniRostered": after["overall"]["giniOfPointsRostered"],
                "eliteMultiple": elite_multiple(after),
                "topVs50": after["overall"]["topToFiftiethRatio"],
                "top10Share": after["overall"]["top10ShareOfTop50"],
                "biggestTypeShare": round(archetype_concentration(after), 3),
                "D": after["topPositionMix"].get("D", 0),
                "M": after["topPositionMix"].get("M", 0),
                "F": after["topPositionMix"].get("F", 0),
                "G": after["topPositionMix"].get("G", 0),
                "eliteWeeklyShare": (after.get("churn") or {}).get("eliteShareOfWeeklyTop"),
                "distinctWeekly": (after.get("churn") or {}).get("distinctPlayersInWeeklyTop"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--season", help="Season label; defaults to all seasons pooled")
    parser.add_argument("--forward-scale", type=float, default=1.0)
    parser.add_argument("--defender-scale", type=float, default=1.0)
    parser.add_argument(
        "--no-rebalance",
        action="store_true",
        help="Leave positions where the trim puts them instead of re-solving the scalars",
    )
    args = parser.parse_args()

    frame = trace(
        args.output_dir,
        args.season,
        args.forward_scale,
        args.defender_scale,
        with_churn=True,
        rebalance=not args.no_rebalance,
    )
    print("Volume categories set to this fraction of their current weight:")
    print(frame.to_string(index=False))
    print("")
    print(
        "Reading it: lower eliteMultiple, gini and topVs50 mean a flatter curve; "
        "lower biggestTypeShare means the elite tier is spread across more player "
        "types; higher distinctWeekly means more players are worth starting in any "
        "given week, which is what drives waiver activity."
    )


if __name__ == "__main__":
    main()
