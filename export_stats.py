#!/usr/bin/env python3
"""Export Fantrax EPL player stats via the private getPlayerProfile API."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

LEAGUE_ID = "4p1r6fkvmr872kn2"
API_VERSION = "185.1.9"
FXPA_URL = "https://www.fantrax.com/fxpa/req"
LEAGUE_INFO_URL = "https://www.fantrax.com/fxea/general/getLeagueInfo"

SEASONS: dict[str, str] = {
    "923": "2023-24",
    "924": "2024-25",
    "925": "2025-26",
}

SEASON_BY_LABEL: dict[str, str] = {v: k for k, v in SEASONS.items()}

FIXED_COLUMNS = {"date", "team", "opponent", "score"}


def load_session(cookies_path: Path) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; fantrax-stats-export/1.0)",
            "Content-Type": "application/json",
        }
    )

    raw = cookies_path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        for item in json.loads(raw):
            session.cookies.set(
                item["name"],
                item["value"],
                domain=item.get("domain"),
                path=item.get("path", "/"),
            )
    elif raw.startswith("{"):
        data = json.loads(raw)
        cookie_str = data.get("Cookie") or data.get("cookie")
        if not cookie_str:
            raise SystemExit('JSON file must be a cookie array or {"Cookie": "..."}')
        session.headers["Cookie"] = cookie_str
    else:
        session.headers["Cookie"] = raw

    return session


def fxpa_request(session: requests.Session, league_id: str, method: str, data: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "msgs": [{"method": method, "data": data}],
        "uiv": 3,
        "dt": 0,
        "at": 0,
        "tz": "America/Chicago",
        "v": API_VERSION,
    }
    resp = session.post(FXPA_URL, params={"leagueId": league_id}, json=payload, timeout=60)
    resp.raise_for_status()
    body = resp.json()

    if "pageError" in body:
        err = body["pageError"]
        code = err.get("code", "UNKNOWN")
        text = err.get("text") or err.get("title") or str(err)
        raise RuntimeError(f"Fantrax API error ({code}): {text}")

    responses = body.get("responses") or []
    if not responses:
        raise RuntimeError(f"Empty response for method {method}")

    response_data = responses[0]
    if "pageError" in response_data:
        err = response_data["pageError"]
        code = err.get("code", "UNKNOWN")
        text = err.get("text") or err.get("title") or str(err)
        raise RuntimeError(f"Fantrax API error ({code}): {text}")

    return response_data.get("data") or {}


def is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "too quickly" in msg or "slow down" in msg


class Throttle:
    """Self-tuning pause between profile requests.

    Fantrax throttles player-profile views specifically, and the threshold moves
    around, so a fixed delay either wastes hours or walks into a block. This
    slows down whenever it is told off and creeps back toward the base rate once
    requests are flowing again.
    """

    GROWTH = 1.6
    DECAY = 0.85
    SUCCESSES_BEFORE_DECAY = 20

    def __init__(self, base: float, ceiling: float = 60.0) -> None:
        self.base = base
        self.ceiling = ceiling
        self.current = base
        self.streak = 0

    def wait(self) -> None:
        time.sleep(self.current)

    def on_success(self) -> None:
        self.streak += 1
        if self.streak >= self.SUCCESSES_BEFORE_DECAY and self.current > self.base:
            self.current = max(self.base, self.current * self.DECAY)
            self.streak = 0

    def on_rate_limit(self) -> None:
        self.streak = 0
        self.current = min(self.ceiling, self.current * self.GROWTH)


# Waits after successive rate-limit rejections for a single player. The block
# outlasts short pauses, so this climbs into the minutes.
RETRY_WAITS = [30, 60, 120, 240, 480, 600, 600, 600]


def fetch_player_game_log_with_retry(
    session: requests.Session,
    league_id: str,
    player_id: str,
    season_id: str,
    throttle: Throttle,
) -> dict[str, Any]:
    for attempt, wait in enumerate(RETRY_WAITS):
        try:
            data = fetch_player_game_log(session, league_id, player_id, season_id)
            throttle.on_success()
            return data
        except RuntimeError as exc:
            if not is_rate_limited(exc):
                raise
            throttle.on_rate_limit()
            if attempt == len(RETRY_WAITS) - 1:
                raise
            print(
                f"  rate limited on {player_id}, waiting {wait}s "
                f"(retry {attempt + 1}/{len(RETRY_WAITS)}, pacing now {throttle.current:.1f}s)",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {player_id} after {len(RETRY_WAITS)} retries")


def fetch_league_players(session: requests.Session, league_id: str) -> dict[str, dict[str, Any]]:
    resp = session.get(LEAGUE_INFO_URL, params={"leagueId": league_id}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    player_info = data.get("playerInfo") or data.get("playerInfoMap") or {}
    if not player_info:
        raise RuntimeError("getLeagueInfo returned no playerInfo — check cookies and league access")
    return player_info


def fetch_player_game_log(
    session: requests.Session,
    league_id: str,
    player_id: str,
    season_id: str,
) -> dict[str, Any]:
    return fxpa_request(
        session,
        league_id,
        "getPlayerProfile",
        {
            "playerId": player_id,
            "tab": "GAME_LOG_FANTASY",
            "seasonId": season_id,
        },
    )


def parse_game_log_table(data: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    section = (data.get("sectionContent") or {}).get("GAME_LOG_FANTASY") or {}
    tables = section.get("tables") or []
    if not tables:
        return [], []

    table = tables[0]
    headers = table.get("header", {}).get("cells") or []
    columns = [
        {
            "key": h.get("key", f"col_{i}"),
            "shortName": h.get("shortName") or h.get("name") or f"col_{i}",
            "name": h.get("name") or h.get("shortName") or f"col_{i}",
        }
        for i, h in enumerate(headers)
    ]

    games: list[dict[str, Any]] = []
    for row in table.get("rows") or []:
        cells = row.get("cells") or []
        record: dict[str, Any] = {}
        for i, col in enumerate(columns):
            if i >= len(cells):
                break
            cell = cells[i]
            record[col["key"]] = cell.get("content", "")
        games.append(record)

    return columns, games


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return 0.0


def aggregate_season_totals(columns: list[dict[str, str]], games: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {"GP": len(games)}
    stat_columns = [c for c in columns if c["key"] not in FIXED_COLUMNS]

    for col in stat_columns:
        key = col["key"]
        short = col["shortName"]
        totals[short] = sum(to_float(g.get(key)) for g in games)

    return totals


def player_display_name(player_id: str, profile_data: dict[str, Any], player_info: dict[str, Any] | None) -> str:
    if player_info and player_info.get("name"):
        return str(player_info["name"])
    league_data = (profile_data.get("miscData") or {}).get("leagueData") or []
    for item in league_data:
        if item.get("name") == "Player":
            return str(item.get("value") or player_id)
    return player_id


def player_position(profile_data: dict[str, Any], player_info: dict[str, Any] | None) -> str:
    if player_info:
        pos = player_info.get("posShortNames") or player_info.get("position")
        if pos:
            return str(pos)
    league_data = (profile_data.get("miscData") or {}).get("leagueData") or []
    for item in league_data:
        if item.get("name") == "Eligible":
            return str(item.get("value") or "")
    return ""


def export_season(
    session: requests.Session,
    league_id: str,
    season_id: str,
    season_label: str,
    player_info: dict[str, dict[str, Any]],
    cache_dir: Path,
    delay: float,
    refresh: bool,
    limit: int | None,
    output_path: Path,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    player_ids = sorted(player_info.keys())
    if limit:
        player_ids = player_ids[:limit]

    all_short_names: set[str] = set()
    rows_out: list[dict[str, Any]] = []
    throttle = Throttle(delay)

    # Players whose fetch fails outright are retried after the main pass rather
    # than dropped, so a temporary block does not leave permanent holes.
    pending = list(player_ids)
    failed: list[str] = []

    for pass_no in (1, 2):
        if not pending:
            break
        if pass_no == 2:
            throttle = Throttle(delay * 2, ceiling=120.0)
            print(
                f"Retrying {len(pending)} players that failed, pacing at {throttle.current:.0f}s",
                file=sys.stderr,
            )

        for idx, player_id in enumerate(pending, start=1):
            cache_file = cache_dir / f"{player_id}.json"

            if cache_file.exists() and not refresh:
                continue

            try:
                profile_data = fetch_player_game_log_with_retry(
                    session, league_id, player_id, season_id, throttle
                )
            except Exception as exc:
                print(f"[{idx}/{len(pending)}] {player_id}: ERROR {exc}", file=sys.stderr)
                failed.append(player_id)
                throttle.wait()
                continue

            cache_file.write_text(json.dumps(profile_data), encoding="utf-8")
            throttle.wait()

            if idx % 25 == 0:
                print(
                    f"[{idx}/{len(pending)}] fetched (pacing {throttle.current:.1f}s)",
                    file=sys.stderr,
                )

        pending, failed = failed, []

    if pending:
        print(f"Still missing {len(pending)} players after retry: {pending[:10]}", file=sys.stderr)

    for idx, player_id in enumerate(player_ids, start=1):
        cache_file = cache_dir / f"{player_id}.json"
        if not cache_file.exists():
            continue
        profile_data = json.loads(cache_file.read_text(encoding="utf-8"))

        columns, games = parse_game_log_table(profile_data)
        if not columns:
            continue

        totals = aggregate_season_totals(columns, games)
        all_short_names.update(totals.keys())

        info = player_info.get(player_id)
        info_dict = info if isinstance(info, dict) else None
        row: dict[str, Any] = {
            "playerId": player_id,
            "name": player_display_name(player_id, profile_data, info_dict),
            "position": player_position(profile_data, info_dict),
            "season": season_label,
        }
        row.update(totals)
        rows_out.append(row)

    priority = ["FPts"]
    fieldnames = ["playerId", "name", "position", "season", "GP"] + priority + sorted(
        s for s in all_short_names if s not in {"GP", "FPts"}
    )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    print(f"Wrote {len(rows_out)} players -> {output_path}")


def resolve_season_ids(season_args: list[str]) -> list[tuple[str, str]]:
    resolved: list[tuple[str, str]] = []
    for arg in season_args:
        if arg in SEASONS:
            resolved.append((arg, SEASONS[arg]))
        elif arg in SEASON_BY_LABEL:
            sid = SEASON_BY_LABEL[arg]
            resolved.append((sid, arg))
        else:
            raise SystemExit(f"Unknown season {arg!r}. Use seasonId (923) or label (2024-25).")
    return resolved


def cmd_test(session: requests.Session, league_id: str, season_id: str, player_id: str) -> None:
    data = fetch_player_game_log(session, league_id, player_id, season_id)
    columns, games = parse_game_log_table(data)
    totals = aggregate_season_totals(columns, games)
    print(json.dumps({"playerId": player_id, "seasonId": season_id, "games": len(games), "totals": totals}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Fantrax player stats (season totals from game logs)")
    parser.add_argument("--league-id", default=LEAGUE_ID)
    parser.add_argument(
        "--season",
        action="append",
        dest="seasons",
        help="Season label (2024-25) or Fantrax seasonId (924). Repeat for multiple seasons.",
    )
    parser.add_argument(
        "--recent",
        type=int,
        metavar="N",
        help="Export the N most recent seasons (e.g. 3 -> 2023-24, 2024-25, 2025-26)",
    )
    parser.add_argument("--cookies", type=Path, default=Path("cookies.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Base seconds between API calls; adapts upward when rate limited",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore cached player JSON")
    parser.add_argument("--limit", type=int, help="Only process first N players (for testing)")
    parser.add_argument("--test", nargs="?", const="03vlj", metavar="PLAYER_ID", help="Test one player")
    args = parser.parse_args()

    if not args.cookies.exists():
        raise SystemExit(f"Missing cookies file: {args.cookies}")

    session = load_session(args.cookies)

    if args.test is not None:
        player_id = args.test
        season_list = resolve_season_ids(args.seasons or ["924"])
        cmd_test(session, args.league_id, season_list[0][0], player_id)
        return

    if args.recent:
        ordered = sorted(SEASONS.items(), key=lambda x: x[0])
        # Newest first, so a partial run still yields the most relevant season.
        season_list = list(reversed(ordered[-args.recent :]))
    elif args.seasons:
        season_list = resolve_season_ids(args.seasons)
    else:
        season_list = [("924", "2024-25")]

    player_info = fetch_league_players(session, args.league_id)
    print(f"Found {len(player_info)} players in league pool", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for season_id, season_label in season_list:
        cache_dir = args.cache_dir / season_id
        output_path = args.output_dir / f"players_{season_label.replace('/', '-')}_stats.csv"
        print(f"\n=== {season_label} (seasonId {season_id}) ===", file=sys.stderr)
        export_season(
            session=session,
            league_id=args.league_id,
            season_id=season_id,
            season_label=season_label,
            player_info=player_info,
            cache_dir=cache_dir,
            delay=args.delay,
            refresh=args.refresh,
            limit=args.limit,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
