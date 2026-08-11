#!/usr/bin/env python3
"""Run the analysis pipeline, one season at a time.

Each season is its own player pool with its own top fifty, replacement level and
starter demand. Pooling several seasons ranks the same player multiple times and
balances against three seasons' worth of slots, which is useful for checking
whether a weight set generalises but is the wrong frame for choosing what to
enter into league settings.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import validate

# Lift forwards slightly past strict slot balance and trim defenders a touch,
# after raising goals, shots on target and assists in the forward base trim.
FORWARD_SCALE_BIAS = '{"F": 1.15, "D": 0.98}'


SHARED_STEPS = [
    ("Building datasets from cache", ["build_dataset.py"]),
    ("Recovering scoring weights", ["recover_weights.py"]),
    ("Clustering player archetypes", ["archetypes.py"]),
]


def season_steps(season: str, season_dir: Path, proposal: Path, data_dir: Path) -> list[tuple[str, list[str]]]:
    shared = ["--data-dir", str(data_dir)]
    return [
        (
            "Measuring balance (%s)" % season,
            ["diagnose.py", "--season", season, "--output-dir", str(season_dir)] + shared,
        ),
        (
            "Solving the recommendation (%s)" % season,
            [
                "tune.py",
                "--base",
                "proposals/base-trim.json",
                "--season",
                season,
                "--name",
                "recommended-%s" % season,
                "--out",
                str(proposal),
                "--scale-bias",
                FORWARD_SCALE_BIAS,
            ]
            + shared,
        ),
        (
            "Writing report and charts (%s)" % season,
            [
                "report.py",
                "--season",
                season,
                "--output-dir",
                str(season_dir),
                "--recommend",
                str(proposal),
            ]
            + shared,
        ),
    ]


def run_step(title: str, command: list[str], cwd: Path) -> None:
    print("\n=== %s ===" % title)
    result = subprocess.run([sys.executable] + command, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit("Step failed: %s" % " ".join(command))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--proposals-dir", type=Path, default=Path("proposals"))
    parser.add_argument(
        "--season",
        help="Run one season only; defaults to every season in the cache",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    seasons = [args.season] if args.season else validate.seasons_available(args.output_dir)
    if not seasons:
        raise SystemExit("No season totals found under %s" % args.output_dir)

    for title, command in SHARED_STEPS:
        run_step(title, command, root)

    for season in seasons:
        season_dir = args.output_dir / "seasons" / season
        season_dir.mkdir(parents=True, exist_ok=True)
        proposal = args.proposals_dir / ("recommended-%s.json" % season)
        for title, command in season_steps(season, season_dir, proposal, args.output_dir):
            run_step(title, command, root)

    latest = seasons[-1]
    run_step(
        "Solving the recommendation (all seasons)",
        [
            "tune.py",
            "--base",
            "proposals/base-trim.json",
            "--name",
            "recommended",
            "--out",
            str(args.proposals_dir / "recommended.json"),
            "--data-dir",
            str(args.output_dir),
            "--scale-bias",
            FORWARD_SCALE_BIAS,
        ],
        root,
    )
    print("\n=== Recommendations ===")
    print("Pooled (all seasons): recommended.json")
    print("Latest season only: recommended-%s.json" % latest)

    run_step(
        "Solving the reduce-team-dependency recommendation (all seasons)",
        [
            "tune.py",
            "--base",
            "proposals/base-trim-reduce-team-dependency.json",
            "--name",
            "recommended-reduce-team-dependency",
            "--out",
            str(args.proposals_dir / "recommended-reduce-team-dependency.json"),
            "--data-dir",
            str(args.output_dir),
            "--scale-bias",
            FORWARD_SCALE_BIAS,
            "--block-team-carriers",
        ],
        root,
    )
    print("Reduce team dependency: recommended-reduce-team-dependency.json")

    run_step(
        "Generating the scoring lab canvas",
        ["build_canvas.py", "--presets", str(args.proposals_dir), "--output-dir", str(args.output_dir)],
        root,
    )

    run_step(
        "Generating the public scoring lab (GitHub Pages)",
        ["build_web.py", "--presets", str(args.proposals_dir), "--output-dir", str(args.output_dir)],
        root,
    )

    print("\nPipeline complete for: %s" % ", ".join(seasons))
    print("Per-season reports: %s/seasons/<season>/report.md" % args.output_dir)


if __name__ == "__main__":
    main()
