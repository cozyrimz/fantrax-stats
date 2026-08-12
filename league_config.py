#!/usr/bin/env python3
"""League roster and lineup configuration.

Mirrors the Fantrax league settings, which determine positional scarcity and
therefore replacement level.
"""

from __future__ import annotations

from typing import Dict

TEAMS = 12
ROSTER_MAX = 14
STARTERS = 11
BENCH_MAX = 3

# Minimum and maximum players that may be started at each position.
MIN_ACTIVE: Dict[str, int] = {"D": 3, "M": 3, "F": 1, "G": 1}
MAX_ACTIVE: Dict[str, int] = {"D": 5, "M": 5, "F": 3, "G": 1}

# Maximum players of each position that may be held on the roster at all.
MAX_ROSTERED: Dict[str, int] = {"D": 7, "M": 7, "F": 5, "G": 2}

# Starting slots beyond the guaranteed minimums, which a manager may fill with
# any of D, M or F.
FLEX_SLOTS = STARTERS - sum(MIN_ACTIVE.values())

# Baseline assumption for replacement level: the flex slots are spread evenly
# across the three outfield positions, giving 1 G / 4 D / 4 M / 2 F per team.
BASELINE_STARTERS: Dict[str, int] = {"G": 1, "D": 4, "M": 4, "F": 2}

# Benchmark for per-game point rates on the value curve (full EPL season).
SEASON_GAMES = 38

# Total players rostered across the league.
ROSTERED_POOL = TEAMS * ROSTER_MAX


def replacement_rank(position: str, starters: Dict[str, int] = None) -> int:
    """Rank within a position at which a player is only replacement level.

    The Nth best player at a position is exactly as valuable as the worst
    starter at that position when N equals league-wide demand for the slot.
    """
    table = starters or BASELINE_STARTERS
    return TEAMS * table[position]
