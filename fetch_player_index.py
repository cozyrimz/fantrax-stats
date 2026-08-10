#!/usr/bin/env python3
"""Fetch the league's player index: names, teams, positions and roster status.

The per-player profile payload does not carry a display name, and roster status
has to come from the current season rather than a historical one. The Players
page endpoint returns both for every player a page at a time, and unlike the
profile endpoint it is not rate limited.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

from export_stats import LEAGUE_ID, SEASONS, fxpa_request, load_session

PAGE_SIZE = 100


def fetch_index(session, league_id: str, season_id: str, delay: float = 1.0) -> Dict[str, Dict]:
    players: Dict[str, Dict] = {}
    page = 1
    total_pages = 1

    while page <= total_pages:
        data = fxpa_request(
            session,
            league_id,
            "getPlayerStats",
            {
                "seasonOrProjection": "SEASON_%s_YEAR_TO_DATE" % season_id,
                "timeframeTypeCode": "YEAR_TO_DATE",
                "view": "STATS",
                "pageNumber": str(page),
                "statusOrTeamFilter": "ALL",
                "maxResultsPerPage": str(PAGE_SIZE),
            },
        )

        paging = data.get("paginatedResultSet") or {}
        total_pages = int(paging.get("totalNumPages", 1))

        headers = [
            c.get("shortName") or c.get("name")
            for c in (data.get("tableHeader") or {}).get("cells") or []
        ]
        status_index = headers.index("Sta") if "Sta" in headers else None

        for row in data.get("statsTable") or []:
            scorer = row.get("scorer") or {}
            player_id = scorer.get("scorerId")
            if not player_id:
                continue
            cells = row.get("cells") or []
            players[player_id] = {
                "name": scorer.get("name") or player_id,
                "team": scorer.get("teamShortName") or "",
                "position": scorer.get("posShortNames") or "",
                "status": cells[status_index].get("content", "")
                if status_index is not None and status_index < len(cells)
                else "",
            }

        print("page %d/%d -> %d players" % (page, total_pages, len(players)))
        page += 1
        if page <= total_pages:
            time.sleep(delay)

    return players


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-id", default=LEAGUE_ID)
    parser.add_argument("--cookies", type=Path, default=Path("cookies.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--season",
        default=max(SEASONS),
        help="Season to read roster status from; defaults to the most recent",
    )
    args = parser.parse_args()

    session = load_session(args.cookies)
    players = fetch_index(session, args.league_id, args.season)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "player_index.json"
    path.write_text(json.dumps(players, indent=2, sort_keys=True), encoding="utf-8")
    print("Wrote %d players -> %s" % (len(players), path))


if __name__ == "__main__":
    main()
