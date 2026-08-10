#!/usr/bin/env python3
"""Build analysis datasets from the cached Fantrax game logs.

Writes, per season, a per-game stat matrix (parquet) and season totals (csv),
plus a categories.json describing every scoring category.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import fantrax_data as fx


def build(cache_dir: Path, output_dir: Path, season_ids: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_categories: Dict[str, str] = {}
    index = fx.load_player_index(output_dir)
    if not index:
        print("No player_index.json — run fetch_player_index.py for real player names")

    for season_id in season_ids:
        label = fx.SEASONS.get(season_id, season_id)
        games, categories = fx.load_games(cache_dir, season_id, index)
        all_categories.update(categories)

        totals = fx.season_totals(games)

        games_path = output_dir / ("games_%s.parquet" % label)
        totals_path = output_dir / ("season_totals_%s.csv" % label)
        games.to_parquet(games_path, index=False)
        totals.to_csv(totals_path, index=False)

        by_position = totals["position"].value_counts().to_dict()
        print(
            "%s: %d players, %d player-games, positions %s"
            % (label, len(totals), len(games), by_position)
        )
        print("  -> %s" % games_path)
        print("  -> %s" % totals_path)

    schemas = fx.load_position_schemas(cache_dir, season_ids)
    (output_dir / "categories.json").write_text(
        json.dumps({"names": all_categories, "positionSchemas": schemas}, indent=2),
        encoding="utf-8",
    )
    print("%d scoring categories -> %s" % (len(all_categories), output_dir / "categories.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--season",
        action="append",
        dest="seasons",
        help="Season label or id. Defaults to every season present in the cache.",
    )
    args = parser.parse_args()

    if args.seasons:
        season_ids = [fx.resolve_season(s)[0] for s in args.seasons]
    else:
        season_ids = sorted(p.name for p in args.cache_dir.iterdir() if p.is_dir())
        if not season_ids:
            raise SystemExit("No season directories found in %s" % args.cache_dir)

    build(args.cache_dir, args.output_dir, season_ids)


if __name__ == "__main__":
    main()
