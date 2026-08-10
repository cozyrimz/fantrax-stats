#!/usr/bin/env python3
"""Check a proposal against every season separately before trusting it.

A weight set tuned on one season can look excellent on that season and do
nothing on another, either because the change was fitted to which players
happened to be collected or because one season was unusual. Replaying the same
weights season by season shows which effects are real and which are not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

import diagnose
import simulate

POSITION_ORDER = ["D", "M", "F", "G"]


def seasons_available(output_dir: Path) -> List[str]:
    return sorted(p.stem.replace("season_totals_", "") for p in output_dir.glob("season_totals_*.csv"))


def check(output_dir: Path, proposal: Dict, seasons: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for season in seasons:
        result = simulate.run(output_dir, proposal, season)
        before, after = result["before"], result["after"]
        row = {
            "season": season,
            "players": before["players"],
            "giniBefore": before["overall"]["giniOfValue"],
            "giniAfter": after["overall"]["giniOfValue"],
        }
        for churn_key, label in (("eliteShareOfWeeklyTop", "eliteWeekly"),):
            row[label + "Before"] = (before.get("churn") or {}).get(churn_key)
            row[label + "After"] = (after.get("churn") or {}).get(churn_key)
        for position in POSITION_ORDER:
            row[position] = "%d>%d" % (
                before["topPositionMix"].get(position, 0),
                after["topPositionMix"].get(position, 0),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def direction_summary(frame: pd.DataFrame) -> List[str]:
    """State only the effects that point the same way in every season."""
    notes: List[str] = []
    checks = [
        ("Inequality of value", "giniBefore", "giniAfter", "lower"),
        ("Share of weekly top thirty held by the season's best twenty", "eliteWeeklyBefore", "eliteWeeklyAfter", "lower"),
    ]
    for label, before_col, after_col, better in checks:
        if before_col not in frame or frame[before_col].isna().any():
            continue
        deltas = frame[after_col] - frame[before_col]
        improved = (deltas < 0) if better == "lower" else (deltas > 0)
        if improved.all():
            notes.append("%s improves in every season (%s)." % (label, _deltas(deltas)))
        elif (~improved).all():
            notes.append("%s worsens in every season (%s)." % (label, _deltas(deltas)))
        else:
            notes.append("%s is inconsistent across seasons (%s), so it should not be claimed." % (label, _deltas(deltas)))
    return notes


def _deltas(series: pd.Series) -> str:
    return ", ".join("%+.3f" % v for v in series)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--proposal", type=Path, default=Path("proposals/recommended.json"))
    args = parser.parse_args()

    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    seasons = seasons_available(args.output_dir)
    if len(seasons) < 2:
        print("Only %d season available, so nothing can be cross-checked yet." % len(seasons))

    frame = check(args.output_dir, proposal, seasons)
    print("Same weights replayed on each season (top-50 counts shown as before>after):")
    print(frame.to_string(index=False))
    print("")
    for note in direction_summary(frame):
        print("- %s" % note)

    frame.to_csv(args.output_dir / "validation.csv", index=False)


if __name__ == "__main__":
    main()
