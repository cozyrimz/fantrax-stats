#!/usr/bin/env python3
"""Recover the league's exact scoring weights from the per-game data.

Every game row satisfies FPts = sum(weight[category] * stat[category]) for the
player's position, so the weights are the solution of a linear system. With
enough rows the solution is exact, which both recovers the scoring config and
validates that the stat parsing is correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import fantrax_data as fx

# Singular values below this are treated as zero when detecting directions of
# the weight space that the data cannot pin down.
RANK_TOL = 1e-7

# The league's configured weights are entered in the Fantrax UI in quarter-point
# steps, so snapping removes floating point noise from the solve.
WEIGHT_STEP = 0.25
SNAP_TOL = 1e-4


def load_all_games(output_dir: Path) -> pd.DataFrame:
    paths = sorted(output_dir.glob("games_*.parquet"))
    if not paths:
        raise SystemExit("No games_*.parquet in %s — run build_dataset.py first" % output_dir)
    frames = [pd.read_parquet(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    stats = [c for c in combined.columns if c not in fx.ID_COLUMNS and c != "FPts"]
    combined[stats] = combined[stats].fillna(0.0)
    return combined


def snap(value: float) -> float:
    stepped = round(value / WEIGHT_STEP) * WEIGHT_STEP
    return stepped if abs(stepped - value) < SNAP_TOL else round(value, 6)


def solve_position(games: pd.DataFrame, position: str) -> Dict[str, object]:
    subset = games[games["position"] == position]
    if subset.empty:
        return {}

    categories = fx.position_categories(games, position)
    design = subset[categories].to_numpy(dtype=float)
    points = subset["FPts"].to_numpy(dtype=float)

    weights, _residuals, rank, _sv = np.linalg.lstsq(design, points, rcond=None)
    predicted = design @ weights
    max_error = float(np.abs(predicted - points).max())

    # Directions the data cannot resolve: right singular vectors with a zero
    # singular value. A category with weight in such a direction is only known
    # in combination with others.
    _u, singular, vt = np.linalg.svd(design, full_matrices=False)
    null_basis = vt[singular <= max(singular[0], 1.0) * RANK_TOL] if singular.size else np.empty((0, len(categories)))
    ambiguous = []
    if null_basis.size:
        exposure = np.abs(null_basis).max(axis=0)
        ambiguous = [categories[i] for i in range(len(categories)) if exposure[i] > 1e-8]

    snapped = {cat: snap(float(w)) for cat, w in zip(categories, weights)}
    snapped_error = float(
        np.abs(design @ np.array([snapped[c] for c in categories]) - points).max()
    )

    return {
        "weights": snapped,
        "rows": int(design.shape[0]),
        "categories": len(categories),
        "rank": int(rank),
        "maxResidual": round(max_error, 8),
        "maxResidualSnapped": round(snapped_error, 8),
        "ambiguous": ambiguous,
    }


def unidentified_categories(schemas: Dict[str, List[str]], position: str, solved: List[str]) -> List[str]:
    """Categories in a position's scoring table that never occurred in the data."""
    return sorted(set(schemas.get(position, [])) - set(solved))


def resolve_ambiguous(positions: Dict[str, Dict[str, object]]) -> List[str]:
    """Split ambiguous weight groups using a position where they are identified.

    Categories that only ever occur together (a red card always accompanies a
    second yellow) are individually unobservable, and the least-norm solve
    spreads their combined value evenly. When another position identifies the
    same categories separately and their total agrees, that position's split is
    the correct one and preserves the exact reconstruction.
    """
    notes = []
    for position, data in positions.items():
        group = list(data.get("ambiguous") or [])
        if not group:
            continue
        weights = data["weights"]
        total = sum(weights[c] for c in group)

        for donor, donor_data in positions.items():
            if donor == position:
                continue
            donor_weights = donor_data["weights"]
            if any(c not in donor_weights for c in group):
                continue
            if set(group) & set(donor_data.get("ambiguous") or []):
                continue
            if abs(sum(donor_weights[c] for c in group) - total) > 1e-6:
                continue
            for category in group:
                weights[category] = donor_weights[category]
            data["resolvedFrom"] = donor
            notes.append(
                "%s: split %s using %s (total %.2f preserved)"
                % (position, "/".join(group), donor, total)
            )
            break
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Maximum allowed reconstruction error before failing",
    )
    args = parser.parse_args()

    games = load_all_games(args.output_dir)
    categories_meta = json.loads((args.output_dir / "categories.json").read_text(encoding="utf-8"))
    schemas = categories_meta.get("positionSchemas", {})

    result: Dict[str, object] = {
        "note": (
            "Weights are solved per position from per-game data and reproduce Fantrax FPts "
            "exactly. Categories listed under neverOccurred have no recorded event in the "
            "data, so their weight cannot be observed and also cannot affect any score. "
            "Categories under ambiguous only ever occur alongside another category, so only "
            "their combined effect is observable; changing them individually requires "
            "reading the value from the league scoring config."
        ),
        "positions": {},
    }
    failures = []

    positions: Dict[str, Dict[str, object]] = {}
    for position in fx.POSITIONS:
        solved = solve_position(games, position)
        if not solved:
            print("%s: no rows" % position)
            continue
        solved["neverOccurred"] = unidentified_categories(
            schemas, position, list(solved["weights"].keys())
        )
        positions[position] = solved

    for note in resolve_ambiguous(positions):
        print("resolved %s" % note)

    result["positions"] = positions

    for position, solved in positions.items():
        subset = games[games["position"] == position]
        categories = list(solved["weights"].keys())
        final = np.array([solved["weights"][c] for c in categories])
        error = np.abs(subset[categories].to_numpy(dtype=float) @ final - subset["FPts"].to_numpy(dtype=float)).max()
        solved["maxResidualFinal"] = round(float(error), 8)
        missing = solved["neverOccurred"]

        print(
            "%s: %d rows, %d categories, rank %d, max residual %.2e"
            % (
                position,
                solved["rows"],
                solved["categories"],
                solved["rank"],
                solved["maxResidualFinal"],
            )
        )
        if missing:
            print("  never occurred (no impact on any score): %s" % ", ".join(missing))
        if solved["ambiguous"] and "resolvedFrom" not in solved:
            print("  not uniquely identified: %s" % ", ".join(solved["ambiguous"]))
        if solved["maxResidualFinal"] > args.tolerance:
            failures.append(position)

    weights_path = args.output_dir / "weights.json"
    weights_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    matrix = pd.DataFrame(
        {pos: data["weights"] for pos, data in result["positions"].items()}
    ).sort_index()
    matrix.index.name = "category"
    matrix["name"] = [categories_meta["names"].get(c, c) for c in matrix.index]
    matrix.to_csv(args.output_dir / "weights.csv")

    print("Wrote %s and %s" % (weights_path, args.output_dir / "weights.csv"))

    if failures:
        raise SystemExit(
            "Reconstruction failed for %s — scoring weights are not exactly recoverable"
            % ", ".join(failures)
        )
    print("Validation passed: recovered weights reproduce Fantrax FPts exactly.")


if __name__ == "__main__":
    main()
