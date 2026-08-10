#!/usr/bin/env python3
"""Shared loading and parsing of cached Fantrax game-log responses.

The cache is written by export_stats.py as cache/{seasonId}/{playerId}.json, each
file being the raw `getPlayerProfile` payload for one player and season.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd

SEASONS: Dict[str, str] = {
    "923": "2023-24",
    "924": "2024-25",
    "925": "2025-26",
}

SEASON_BY_LABEL: Dict[str, str] = {v: k for k, v in SEASONS.items()}

# Descriptive columns in the game-log table that are not scoring categories.
META_KEYS = {"date", "team", "opponent", "score"}
FPTS_KEY = "fpts"

# Identity columns carried alongside the stat matrix.
ID_COLUMNS = [
    "playerId",
    "name",
    "position",
    "status",
    "season",
    "gameNo",
    "date",
    "team",
    "opp",
    "score",
]

POSITIONS = ["D", "M", "F", "G"]


def resolve_season(arg: str) -> Tuple[str, str]:
    if arg in SEASONS:
        return arg, SEASONS[arg]
    if arg in SEASON_BY_LABEL:
        return SEASON_BY_LABEL[arg], arg
    raise SystemExit("Unknown season %r. Use a seasonId (923) or label (2024-25)." % arg)


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return 0.0


def _league_field(profile: Dict[str, Any], field: str) -> str:
    for item in (profile.get("miscData") or {}).get("leagueData") or []:
        if item.get("name") == field:
            return str(item.get("value") or "")
    return ""


def parse_game_log(profile: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Return (columns, rows) from a getPlayerProfile GAME_LOG_FANTASY payload."""
    section = (profile.get("sectionContent") or {}).get("GAME_LOG_FANTASY") or {}
    tables = section.get("tables") or []
    if not tables:
        return [], []

    table = tables[0]
    columns = [
        {
            "key": h.get("key", "col_%d" % i),
            "short": h.get("shortName") or h.get("name") or "col_%d" % i,
            "name": h.get("name") or h.get("shortName") or "col_%d" % i,
        }
        for i, h in enumerate(table.get("header", {}).get("cells") or [])
    ]

    rows = []
    for row in table.get("rows") or []:
        cells = row.get("cells") or []
        record = {}
        for i, col in enumerate(columns):
            if i >= len(cells):
                break
            record[col["key"]] = cells[i].get("content", "")
        rows.append(record)

    return columns, rows


def iter_profiles(cache_dir: Path, season_id: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    season_dir = cache_dir / season_id
    if not season_dir.is_dir():
        raise SystemExit("No cache directory for season %s: %s" % (season_id, season_dir))
    for path in sorted(season_dir.glob("*.json")):
        try:
            yield path.stem, json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue


def load_player_index(output_dir: Path) -> Dict[str, Dict[str, str]]:
    """Names, teams and current roster status, keyed by player id.

    The profile payload has no display name and its status field reflects the
    season being viewed, so both come from the Players page index instead.
    """
    path = output_dir / "player_index.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def display_name(player_id: str, profile: Dict[str, Any], index: Dict[str, Dict[str, str]]) -> str:
    entry = index.get(player_id)
    if entry and entry.get("name"):
        return entry["name"]
    slug = (profile.get("miscData") or {}).get("urlName")
    if slug:
        return " ".join(part.capitalize() for part in str(slug).split("-"))
    return player_id


def load_games(
    cache_dir: Path, season_id: str, index: Dict[str, Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Build a per-game stat matrix for one season.

    Returns a DataFrame with one row per player-game (identity columns, FPts, then
    one column per scoring category keyed by its short name) and a mapping of
    short name to full category name.
    """
    season_label = SEASONS.get(season_id, season_id)
    index = index or {}
    records: List[Dict[str, Any]] = []
    category_names: Dict[str, str] = {}

    for player_id, profile in iter_profiles(cache_dir, season_id):
        columns, rows = parse_game_log(profile)
        if not columns or not rows:
            continue

        entry = index.get(player_id) or {}
        name = display_name(player_id, profile, index)
        position = _league_field(profile, "Eligible") or entry.get("position", "")
        status = entry.get("status") or _league_field(profile, "Status/Team")

        stat_cols = [c for c in columns if c["key"] not in META_KEYS and c["key"] != FPTS_KEY]
        for col in stat_cols:
            category_names[col["short"]] = col["name"]

        by_key = {c["key"]: c for c in columns}
        for game_no, row in enumerate(rows, start=1):
            record: Dict[str, Any] = {
                "playerId": player_id,
                "name": name,
                "position": position,
                "status": status,
                "season": season_label,
                "gameNo": game_no,
                "date": row.get("date", ""),
                "team": row.get("team", ""),
                "opp": row.get("opponent", ""),
                "score": row.get("score", ""),
                "FPts": to_float(row.get(FPTS_KEY)),
            }
            for col in stat_cols:
                record[col["short"]] = to_float(row.get(col["key"]))
            records.append(record)

    if not records:
        raise SystemExit("No game rows found in cache for season %s" % season_id)

    frame = pd.DataFrame.from_records(records)
    stat_columns = [c for c in frame.columns if c not in ID_COLUMNS and c != "FPts"]
    frame[stat_columns] = frame[stat_columns].fillna(0.0)
    ordered = ID_COLUMNS + ["FPts"] + sorted(stat_columns)
    return frame[ordered], category_names


def stat_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c not in ID_COLUMNS and c != "FPts"]


def season_totals(games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a per-game matrix into one row per player."""
    stats = stat_columns(games)
    grouped = games.groupby(["playerId", "name", "position", "status", "season"], as_index=False)
    totals = grouped[["FPts"] + stats].sum()
    counts = grouped.size().rename(columns={"size": "GP"})
    totals = totals.merge(counts, on=["playerId", "name", "position", "status", "season"])
    totals["FPtsPerGame"] = (totals["FPts"] / totals["GP"]).round(4)
    ordered = ["playerId", "name", "position", "status", "season", "GP", "FPts", "FPtsPerGame"] + stats
    return totals[ordered].sort_values("FPts", ascending=False).reset_index(drop=True)


def load_position_schemas(cache_dir: Path, season_ids: List[str]) -> Dict[str, List[str]]:
    """Categories present in each position's game-log header, whether or not they occurred.

    Used to report categories whose weight cannot be identified from the data
    because no player ever recorded a non-zero value.
    """
    schemas: Dict[str, set] = {}
    for season_id in season_ids:
        for _player_id, profile in iter_profiles(cache_dir, season_id):
            columns, rows = parse_game_log(profile)
            if not columns or not rows:
                continue
            position = _league_field(profile, "Eligible")
            shorts = {
                c["short"] for c in columns if c["key"] not in META_KEYS and c["key"] != FPTS_KEY
            }
            schemas.setdefault(position, set()).update(shorts)
    return {pos: sorted(cats) for pos, cats in schemas.items()}


def position_categories(games: pd.DataFrame, position: str) -> List[str]:
    """Categories that are actually part of a position's scoring table.

    Keepers and outfielders have different game-log schemas, so after the union
    fill a category is only relevant to a position if that position ever records
    a non-zero value for it.
    """
    subset = games[games["position"] == position]
    stats = stat_columns(games)
    return [c for c in stats if (subset[c] != 0).any()]
