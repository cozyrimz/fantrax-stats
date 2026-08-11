#!/usr/bin/env python3
"""Generate the interactive scoring lab canvas with the analysis data embedded.

A canvas cannot fetch data, so the player stat matrix is written straight into
the file. Only the categories worth adjusting carry their own column; everything
else is folded into a per-player base score, which keeps the payload small while
still reproducing exact points for any weight the user picks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

import diagnose
import simulate
import validate
import league_config as lc

CANVAS_DIR = Path(
    "/Users/sarimshah/.cursor/projects/Users-sarimshah-Code-fantrax-stats/canvases"
)
CANVAS_NAME = "scoring-lab.canvas.tsx"

# Categories always exposed as levers, whatever their share of points.
ALWAYS_LEVERS = ["G", "A", "CS", "SOT", "KP"]

# How many sliders to fill in by point impact. The recommendation's own
# categories are always exposed on top of these, so it stays fully adjustable.
MAX_LEVERS = 20

# Players kept per position: comfortably past league-wide starter demand so
# replacement level and waiver depth stay meaningful inside the canvas.
KEEP_PER_POSITION = 110

# Presets kept on disk for per-season reports but hidden from the scoring lab UI.
PRESET_EXCLUDE = frozenset({"identity", "recommended-2023-24", "recommended-2024-25"})

# Button order in the preset row; anything else sorts alphabetically after these.
PRESET_ORDER = [
    "recommended",
    "recommended-hand-tuned",
    "recommended-reduce-team-dependency",
    "recommended-2025-26",
    "flatten",
    "lift-scarcity",
]

PRESET_TIPS = {
    "recommended": "Trim volume stats and rebalance the top-fifty mix across all three seasons, with a forward scoring boost.",
    "recommended-hand-tuned": "Manual tuning from current scoring: more defensive actions and creator stats; keepers unchanged.",
    "recommended-reduce-team-dependency": "Same rebalance, but cuts clean sheets and other team-result stats in favour of individual actions.",
    "recommended-2025-26": "Original recommendation tuned on 2025-26 only.",
    "flatten": "Doubles volume categories as a control — the opposite of the recommendation.",
    "lift-scarcity": "Raises goals, assists, big chances, and shots on target only.",
    "trim-volume": "Cuts recoveries, duels, passes, and clearances without rebalancing positions.",
}


def choose_levers(
    working: pd.DataFrame,
    contributions: pd.DataFrame,
    weights: Dict[str, Dict[str, float]],
    required: Iterable[str] = (),
) -> List[str]:
    """Pick the sliders to expose, biggest movers of points first.

    Anything a preset changes has to be exposed, or that preset would appear in
    the canvas as a partial version of itself and land on different numbers than
    the written analysis.
    """
    impact = contributions.abs().sum().sort_values(ascending=False)
    levers = [c for c in ALWAYS_LEVERS if c in impact.index]
    for category in required:
        if category in impact.index and category not in levers:
            levers.append(category)
    for category in impact.index:
        if len(levers) >= MAX_LEVERS:
            break
        if category not in levers:
            levers.append(category)
    return levers


def categories_used_by(proposal: Dict) -> List[str]:
    """Every category a proposal changes."""
    used: List[str] = []
    for key in ("multiply", "set"):
        for table in (proposal.get(key) or {}).values():
            for category in table:
                if category not in used:
                    used.append(category)
    return used


def build_payload(output_dir: Path, required: Iterable[str] = ()) -> Dict:
    weights = diagnose.load_weights(output_dir)
    totals = diagnose.attach_archetypes(diagnose.load_totals(output_dir, None), output_dir)
    totals = totals.assign(points=diagnose.score_totals(totals, weights))

    # Levers are chosen from the full pool so every season's recommendation
    # category can appear as a slider when present in the data.
    _, working_all, contributions = diagnose.compute_metrics(totals, weights)
    levers = choose_levers(working_all, contributions, weights, required)

    kept = []
    for season in sorted(totals["season"].unique()):
        season_frame = totals[totals["season"] == season]
        for position in lc.MIN_ACTIVE:
            subset = season_frame[season_frame["position"] == position].sort_values(
                "points", ascending=False
            )
            kept.append(subset.head(KEEP_PER_POSITION))
    frame = pd.concat(kept).reset_index(drop=True)

    # Points from every category that is not adjustable, so the canvas only needs
    # the lever columns to recompute a player's exact total.
    lever_points = pd.Series(0.0, index=frame.index)
    for position, table in weights.items():
        mask = frame["position"] == position
        if not mask.any():
            continue
        for category in levers:
            weight = table.get(category)
            if weight and category in frame.columns:
                lever_points.loc[mask] += frame.loc[mask, category] * weight
    base = frame["points"] - lever_points

    players = []
    for i, row in frame.iterrows():
        players.append(
            {
                "n": row["name"],
                "p": row["position"],
                "a": row.get("archetype", ""),
                "y": row["season"],
                "g": int(row["GP"]),
                "b": round(float(base.loc[i]), 1),
                "s": [
                    round(float(row[c]), 1) if c in frame.columns else 0.0 for c in levers
                ],
            }
        )

    names = json.loads((output_dir / "categories.json").read_text(encoding="utf-8"))["names"]
    seasons = sorted(totals["season"].unique().tolist())

    return {
        "levers": [{"c": c, "name": names.get(c, c)} for c in levers],
        "weights": {
            pos: {c: table.get(c, 0.0) for c in levers if c in table}
            for pos, table in weights.items()
        },
        "players": players,
        "seasons": seasons,
        "seasonCounts": {
            season: int((frame["season"] == season).sum()) for season in seasons
        },
        "league": {
            "teams": lc.TEAMS,
            "starters": lc.STARTERS,
            "roster": lc.ROSTER_MAX,
            "minActive": lc.MIN_ACTIVE,
            "maxActive": lc.MAX_ACTIVE,
            "baseline": lc.BASELINE_STARTERS,
        },
        "playerCount": int(len(frame)),
    }


TEMPLATE = r'''/**
 * Scoring lab for a custom Fantrax EPL league.
 *
 * Adjust any scoring weight and watch the value curve, positional mix and
 * waiver depth respond. Points are recomputed exactly: each player carries the
 * points they earned from categories that are not adjustable here, plus their
 * season totals in every category that is.
 */
import {
  BarChart,
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  LineChart,
  Pill,
  Row,
  Spacer,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type Lever = { c: string; name: string };
type Player = { n: string; p: string; a: string; y: string; g: number; b: number; s: number[] };

const DATA = __DATA__ as {
  levers: Lever[];
  weights: Record<string, Record<string, number>>;
  players: Player[];
  seasons: string[];
  seasonCounts: Record<string, number>;
  league: {
    teams: number;
    starters: number;
    roster: number;
    minActive: Record<string, number>;
    maxActive: Record<string, number>;
    baseline: Record<string, number>;
  };
  playerCount: number;
};

const POSITIONS = ["D", "M", "F", "G"];
const POSITION_NAMES: Record<string, string> = {
  D: "Defenders",
  M: "Midfielders",
  F: "Forwards",
  G: "Keepers",
};

const PRESETS: Record<string, Record<string, Record<string, number>>> = __PRESETS__;
const PRESET_TIPS: Record<string, string> = __PRESET_TIPS__;
const PRESET_ORDER: string[] = __PRESET_ORDER__;

function scaleWeights(
  multipliers: Record<string, Record<string, number>>,
): Record<string, Record<string, number>> {
  const out: Record<string, Record<string, number>> = {};
  for (const pos of Object.keys(DATA.weights)) {
    out[pos] = {};
    for (const [cat, weight] of Object.entries(DATA.weights[pos])) {
      const factor = multipliers[pos]?.[cat] ?? multipliers.all?.[cat] ?? 1;
      out[pos][cat] = Math.round(weight * factor * 1000) / 1000;
    }
  }
  return out;
}

function scorePlayers(weights: Record<string, Record<string, number>>, season?: string) {
  return DATA.players
    .map((player, index) => {
      const table = weights[player.p] ?? {};
      let points = player.b;
      DATA.levers.forEach((lever, i) => {
        const weight = table[lever.c];
        if (weight) points += weight * player.s[i];
      });
      return { index, player, points };
    })
    .filter((row) => !season || row.player.y === season);
}

type Scored = ReturnType<typeof scorePlayers>[number];

/** League-wide starting slots the league actually fills at each position. */
function starterDemand(scored: Scored[]): Record<string, number> {
  const { teams, starters, minActive, maxActive } = DATA.league;
  const taken: Record<string, number> = {};
  const used = new Set<number>();
  const ranked = [...scored].sort((a, b) => b.points - a.points);

  for (const pos of POSITIONS) {
    const need = teams * (minActive[pos] ?? 0);
    const pool = ranked.filter((row) => row.player.p === pos).slice(0, need);
    taken[pos] = pool.length;
    pool.forEach((row) => used.add(row.index));
  }

  let remaining = teams * starters - Object.values(taken).reduce((a, b) => a + b, 0);
  for (const row of ranked) {
    if (remaining <= 0) break;
    if (used.has(row.index)) continue;
    const pos = row.player.p;
    const ceiling = teams * (maxActive[pos] ?? 0);
    if (taken[pos] >= ceiling) continue;
    taken[pos] += 1;
    used.add(row.index);
    remaining -= 1;
  }
  return taken;
}

function gini(values: number[]): number {
  const clipped = values.map((v) => Math.max(0, v)).sort((a, b) => a - b);
  const total = clipped.reduce((a, b) => a + b, 0);
  if (!clipped.length || total === 0) return 0;
  let weighted = 0;
  clipped.forEach((value, i) => {
    weighted += (2 * (i + 1) - clipped.length - 1) * value;
  });
  return weighted / (clipped.length * total);
}

function analyse(weights: Record<string, Record<string, number>>, season?: string) {
  const scored = scorePlayers(weights, season);
  const ranked = [...scored].sort((a, b) => b.points - a.points);
  const demand = starterDemand(scored);

  const replacement: Record<string, number> = {};
  const byPosition: Record<string, Scored[]> = {};
  for (const pos of POSITIONS) {
    const pool = ranked.filter((row) => row.player.p === pos);
    byPosition[pos] = pool;
    const count = demand[pos] ?? 0;
    replacement[pos] = pool.length > count ? pool[count].points : (pool[pool.length - 1]?.points ?? 0);
  }

  const vor = ranked.map((row) => row.points - (replacement[row.player.p] ?? 0));
  const top50 = ranked.slice(0, 50);
  const top150 = ranked.slice(0, 150);

  const positions = POSITIONS.map((pos) => {
    const pool = byPosition[pos];
    const count = demand[pos] ?? 0;
    const starters = pool.slice(0, count);
    const median = starters.length
      ? starters[Math.floor(starters.length / 2)].points
      : 0;
    const usable = pool.filter((row) => row.points >= 0.8 * median).length;
    const elite = starters.slice(0, 12);
    const eliteMean = elite.length
      ? elite.reduce((a, b) => a + b.points, 0) / elite.length
      : 0;
    return {
      pos,
      players: pool.length,
      demand: count,
      replacement: replacement[pos] ?? 0,
      median,
      usable,
      surplus: usable - count,
      eliteMultiple: replacement[pos] > 0 ? eliteMean / replacement[pos] : null,
      inTop50: top50.filter((row) => row.player.p === pos).length,
      inTop150: top150.filter((row) => row.player.p === pos).length,
    };
  });

  return {
    ranked,
    byPosition,
    demand,
    positions,
    gini: gini(vor),
    top50,
    ratio: ranked.length > 49 && ranked[49].points > 0 ? ranked[0].points / ranked[49].points : null,
    top10Share:
      top50.length > 0
        ? top50.slice(0, 10).reduce((a, b) => a + b.points, 0) /
          top50.reduce((a, b) => a + b.points, 0)
        : 0,
  };
}

function weightStep(value: number): number {
  const m = Math.abs(value);
  if (m < 0.2) return 0.01;
  if (m < 2) return 0.05;
  if (m < 10) return 0.5;
  return 1;
}

function snapWeight(value: number): number {
  const step = weightStep(value);
  return Math.round(Math.round(value / step) * step * 100) / 100;
}

function proposedWeight(base: number, factor: number): number {
  return snapWeight(base * factor);
}

function WeightSlider({
  lever,
  position,
  factor,
  onChange,
}: {
  lever: Lever;
  position: string;
  factor: number;
  onChange: (value: number) => void;
}) {
  const theme = useHostTheme();
  const base = DATA.weights[position]?.[lever.c];
  if (base === undefined) return null;
  const current = proposedWeight(base, factor);
  const changed = Math.abs(current - snapWeight(base)) > 0.001;
  const step = weightStep(current || base);
  const sliderVal = Math.min(1.75, Math.max(0.25, factor));

  const applyWeight = (raw: number) => {
    const proposed = snapWeight(raw);
    onChange(base ? proposed / base : 1);
  };

  return (
    <Row gap={10} align="center">
      <Text
        size="small"
        weight="medium"
        style={{ width: 52, flexShrink: 0, fontVariantNumeric: "tabular-nums" }}
      >
        {lever.c}
      </Text>
      <input
        type="range"
        min={0.25}
        max={1.75}
        step={0.05}
        value={sliderVal}
        onChange={(event: { target: { value: string } }) => onChange(Number(event.target.value))}
        style={{ flex: 1, minWidth: 0, accentColor: theme.accent.primary }}
        aria-label={`${lever.name} multiplier`}
      />
      <Row gap={6} align="center" style={{ flexShrink: 0 }}>
        <Text size="small" tone="tertiary" style={{ fontVariantNumeric: "tabular-nums" }}>
          {base}
        </Text>
        <Text size="small" tone="tertiary">
          →
        </Text>
        <input
          type="number"
          defaultValue={current}
          key={`${position}-${lever.c}-${current}`}
          step={step}
          onBlur={(event: { currentTarget: { value: string } }) => {
            applyWeight(Number(event.currentTarget.value));
          }}
          onKeyDown={(event: { key: string; preventDefault: () => void; currentTarget: HTMLInputElement }) => {
            if (event.key === "Enter") {
              event.preventDefault();
              applyWeight(Number(event.currentTarget.value));
              return;
            }
            if (event.key === "ArrowUp" || event.key === "ArrowDown") {
              event.preventDefault();
              const delta = event.key === "ArrowUp" ? step : -step;
              applyWeight(Number(event.currentTarget.value) + delta);
            }
          }}
          style={{
            width: "4.25rem",
            fontVariantNumeric: "tabular-nums",
            textAlign: "right",
            padding: "4px 6px",
            borderRadius: 4,
            border: `1px solid ${changed ? theme.accent.primary : theme.border.primary}`,
            background: theme.background.secondary,
            color: theme.text.primary,
          }}
          aria-label={`${lever.name} proposed weight`}
        />
      </Row>
    </Row>
  );
}

export default function ScoringLab() {
  const [multipliers, setMultipliers] = useCanvasState<Record<string, Record<string, number>>>(
    "scoring-multipliers",
    {},
  );
  const [position, setPosition] = useCanvasState<string>("lab-position", "D");
  const [listSeason, setListSeason] = useCanvasState<string>(
    "lab-list-season",
    DATA.seasons[DATA.seasons.length - 1] ?? "",
  );

  const weights = scaleWeights(multipliers);
  const current = analyse(weights, listSeason);
  const baseline = analyse(scaleWeights({}), listSeason);
  const seasonDetails = DATA.seasons.flatMap((season) =>
    analyse(weights, season).positions.map((row) => ({ season, ...row })),
  );

  const setFactor = (pos: string, cat: string, value: number) => {
    const base = DATA.weights[pos]?.[cat];
    if (base !== undefined) {
      const proposed = snapWeight(base * value);
      value = base ? proposed / base : 1;
    }
    setMultipliers((prev) => {
      const next = { ...prev, [pos]: { ...(prev[pos] ?? {}) } };
      if (Math.abs(value - 1) < 0.001) {
        delete next[pos][cat];
        if (!Object.keys(next[pos]).length) delete next[pos];
      } else {
        next[pos][cat] = Math.round(value * 1000) / 1000;
      }
      return next;
    });
  };

  const applyPreset = (name: string) => {
    setMultipliers(name === "current" ? {} : PRESETS[name] ?? {});
  };

  const ranks = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110];
  const curveSeries = POSITIONS.map((pos) => ({
    name: POSITION_NAMES[pos],
    data: ranks.map((rank) => {
      const pool = current.byPosition[pos];
      return pool && pool.length >= rank ? Math.round(pool[rank - 1].points) : 0;
    }),
  }));

  const delta = (value: number, base: number, betterLower = true) => {
    const change = value - base;
    if (Math.abs(change) < 0.0005) return "no change";
    const better = betterLower ? change < 0 : change > 0;
    return `${change > 0 ? "+" : ""}${change.toFixed(3)} ${better ? "better" : "worse"}`;
  };

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1180 }}>
      <Stack gap={6}>
        <H1>Scoring lab</H1>
        <Text tone="secondary">
          {DATA.playerCount} player-seasons across {DATA.seasons.join(", ")}, scored with the
          league's recovered weights. Move a slider to see how the value curve, positional mix and
          waiver depth change. Every number is a real recomputed season total, not an estimate.
        </Text>
      </Stack>

      <Grid columns={5} gap={16}>
        <Stat
          value={current.gini.toFixed(3)}
          label={`Value inequality (Gini) · ${delta(current.gini, baseline.gini)}`}
          tone={current.gini < baseline.gini ? "success" : undefined}
        />
        <Stat
          value={current.ratio ? `${current.ratio.toFixed(2)}x` : "n/a"}
          label={`Best player vs rank 50${
            current.ratio && baseline.ratio ? ` · ${delta(current.ratio, baseline.ratio)}` : ""
          }`}
        />
        <Stat
          value={`${Math.round(100 * current.top10Share)}%`}
          label="Top 10 share of top 50 points"
        />
        <Stat
          value={current.positions.map((p) => p.inTop50).join(" / ")}
          label="Top 50 mix (D / M / F / G)"
        />
        <Stat
          value={current.positions.map((p) => p.inTop150).join(" / ")}
          label="Top 150 mix (D / M / F / G)"
          title="Positional mix among the top 150 scorers in the selected season."
        />
      </Grid>

      <Row gap={8} align="center" wrap>
        <Text size="small" tone="secondary">
          Presets
        </Text>
        <Pill
          onClick={() => applyPreset("current")}
          active={Object.keys(multipliers).length === 0}
          title={PRESET_TIPS.current}
        >
          Current scoring
        </Pill>
        {PRESET_ORDER.map((name) => (
          <span key={name}>
            <Pill onClick={() => applyPreset(name)} title={PRESET_TIPS[name]}>
              {name}
            </Pill>
          </span>
        ))}
        <Spacer />
        <Button variant="ghost" onClick={() => setMultipliers({})}>
          Reset all
        </Button>
      </Row>

      <Card>
        <CardHeader trailing={<Text size="small" tone="tertiary">current → proposed</Text>}>
          Category weights
        </CardHeader>
        <CardBody>
          <Stack gap={12}>
            <Row gap={8} wrap>
              {POSITIONS.map((pos) => (
                <span key={pos}>
                  <Pill active={pos === position} onClick={() => setPosition(pos)}>
                    {POSITION_NAMES[pos]}
                  </Pill>
                </span>
              ))}
            </Row>
            <Divider />
            <Stack gap={8}>
              {DATA.levers.map((lever) => (
                <div key={`${position}-${lever.c}`}>
                  <WeightSlider
                    lever={lever}
                    position={position}
                    factor={multipliers[position]?.[lever.c] ?? 1}
                    onChange={(value) => setFactor(position, lever.c, value)}
                  />
                </div>
              ))}
            </Stack>
          </Stack>
        </CardBody>
      </Card>

      <Stack gap={8}>
        <H2>Value curve by position</H2>
        <Text size="small" tone="secondary">
          Season fantasy points of the nth best player at each position. A flatter line means the
          gap between a star and a replaceable starter is smaller, so waiver pickups matter.
        </Text>
        <LineChart
          categories={ranks.map((r) => `#${r}`)}
          series={curveSeries}
          height={300}
          valueSuffix=" pts"
        />
        <Text size="small" tone="tertiary">
          Source: Fantrax game logs for {listSeason} · x axis is rank within position, y axis is
          season fantasy points
        </Text>
      </Stack>

      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H2>Supply against demand</H2>
          <Text size="small" tone="secondary">
            Starting slots the twelve teams must fill each week, against how many players score
            within 80 percent of a median starter. Bars taller than demand mean startable players
            are sitting on waivers.
          </Text>
          <BarChart
            categories={POSITIONS.map((p) => POSITION_NAMES[p])}
            series={[
              { name: "Starting slots", data: current.positions.map((p) => p.demand) },
              { name: "Usable players", data: current.positions.map((p) => p.usable) },
            ]}
            height={260}
            showValues
          />
          <Text size="small" tone="tertiary">
            Source: recomputed season totals · league settings 1 keeper, 3 to 5 defenders, 3 to 5
            midfielders, 1 to 3 forwards
          </Text>
        </Stack>

        <Stack gap={8}>
          <H2>Positional balance</H2>
          <Text size="small" tone="secondary">
            Share of the top 50 scorers held by each position, against the share of starting slots
            that position represents. Equal bars mean the scoring matches what the league fields.
          </Text>
          <BarChart
            categories={POSITIONS.map((p) => POSITION_NAMES[p])}
            series={[
              {
                name: "Share of top 50",
                data: current.positions.map((p) => Math.round((100 * p.inTop50) / 50)),
              },
              {
                name: "Share of starting slots",
                data: current.positions.map(
                  (p) =>
                    Math.round(
                      (100 * (DATA.league.teams * (DATA.league.baseline[p.pos] ?? 0))) /
                        (DATA.league.teams * DATA.league.starters),
                    ),
                ),
              },
            ]}
            height={260}
            valueSuffix="%"
            showValues
          />
          <Text size="small" tone="tertiary">
            Source: recomputed season totals · shares in percent
          </Text>
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H2>Position detail</H2>
        <Text size="small" tone="secondary">
          Each season is recomputed separately under the current sliders. Starter demand and the
          top fifty are always for one season at a time, not pooled across years.
        </Text>
        <Table
          headers={[
            "Season",
            "Position",
            "Players",
            "Starting slots",
            "Replacement level",
            "Median starter",
            "Elite multiple",
            "Usable",
            "Surplus",
            "In top 50",
          ]}
          columnAlign={[
            "left",
            "left",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
          ]}
          rows={seasonDetails.map((row) => [
            row.season,
            POSITION_NAMES[row.pos],
            row.players,
            row.demand,
            Math.round(row.replacement),
            Math.round(row.median),
            row.eliteMultiple ? `${row.eliteMultiple.toFixed(2)}x` : "n/a",
            row.usable,
            `${row.surplus > 0 ? "+" : ""}${row.surplus}`,
            row.inTop50,
          ])}
        />
      </Stack>

      <Stack gap={8}>
        <Row gap={8} align="center" wrap>
          <H2>Top 50 under the current sliders</H2>
          <Spacer />
          {DATA.seasons.map((season) => (
            <span key={season}>
              <Pill active={season === listSeason} onClick={() => setListSeason(season)}>
                {season}
              </Pill>
            </span>
          ))}
        </Row>
        <Text size="small" tone="secondary">
          {DATA.seasonCounts[listSeason] ?? 0} players in {listSeason}. Move compares each player's
          rank against the current scoring for that season.
        </Text>
        <Table
          headers={["#", "Player", "Position", "Type", "Games", "Points", "Move"]}
          columnAlign={["right", "left", "left", "left", "right", "right", "right"]}
          stickyHeader
          rows={current.ranked.slice(0, 50).map((row, i) => {
            const before = baseline.ranked.findIndex((other) => other.index === row.index) + 1;
            const move = before - (i + 1);
            return [
              i + 1,
              row.player.n,
              row.player.p,
              row.player.a,
              row.player.g,
              Math.round(row.points),
              move === 0 ? "-" : `${move > 0 ? "+" : ""}${move}`,
            ];
          })}
        />
      </Stack>

      <Callout tone="neutral" title="How points are recomputed">
        Scoring is linear in the match statistics, so a player's season total is the sum of every
        category weight times the volume they produced. Categories not exposed as sliders are held
        at their current weight and folded into each player's base score, which is why totals here
        match Fantrax exactly when every slider sits at its original value.
      </Callout>
    </Stack>
  );
}
'''


def collect_presets(
    output_dir: Path, payload: Dict, presets_dir: Path
) -> tuple[
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, str],
    List[str],
    List[str],
]:
    """Turn proposal files into slider multiplier presets the UI can apply."""
    current = diagnose.load_weights(output_dir)
    lever_categories = {lever["c"] for lever in payload["levers"]}
    presets: Dict[str, Dict[str, Dict[str, float]]] = {}
    tips: Dict[str, str] = {
        "current": "League weights as entered in Fantrax today.",
    }
    skipped: List[str] = []
    for path in sorted(presets_dir.glob("*.json")):
        proposal = json.loads(path.read_text(encoding="utf-8"))
        name = proposal.get("name", path.stem)
        if name in PRESET_EXCLUDE:
            continue
        proposed, _ = simulate.apply_proposal(current, proposal)
        table: Dict[str, Dict[str, float]] = {}
        covered = True
        for position, weights in proposed.items():
            for category, value in weights.items():
                base = current.get(position, {}).get(category, 0.0)
                if not base or value == base:
                    continue
                if category not in lever_categories:
                    covered = False
                    continue
                table.setdefault(position, {})[category] = round(value / base, 4)
        if table and covered:
            presets[name] = table
            if name in PRESET_TIPS:
                tips[name] = PRESET_TIPS[name]
            elif proposal.get("description"):
                tips[name] = proposal["description"]
        elif table:
            skipped.append(name)
    order = [name for name in PRESET_ORDER if name in presets]
    order.extend(sorted(name for name in presets if name not in order))
    return presets, tips, order, skipped


def resolve_required_levers(output_dir: Path, presets_dir: Path) -> List[str]:
    """Categories both recommended presets may change, so each is fully adjustable."""
    paths: List[Path] = []
    headline = presets_dir / "recommended.json"
    if headline.exists():
        paths.append(headline)
    team_reduce = presets_dir / "recommended-reduce-team-dependency.json"
    if team_reduce.exists() and team_reduce not in paths:
        paths.append(team_reduce)
    hand_tuned = presets_dir / "recommended-hand-tuned.json"
    if hand_tuned.exists() and hand_tuned not in paths:
        paths.append(hand_tuned)
    seasons = validate.seasons_available(output_dir)
    if seasons:
        latest = presets_dir / ("recommended-%s.json" % seasons[-1])
        if latest.exists() and latest not in paths:
            paths.append(latest)
    if not paths:
        return []
    used: List[str] = []
    for path in paths:
        for category in categories_used_by(json.loads(path.read_text(encoding="utf-8"))):
            if category not in used:
                used.append(category)
    return used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--presets", type=Path, default=Path("proposals"))
    parser.add_argument(
        "--max-players",
        type=int,
        help="Trim the embedded data, for cheaply type-checking the generated code",
    )
    parser.add_argument("--name", default=CANVAS_NAME)
    args = parser.parse_args()

    required = resolve_required_levers(args.output_dir, args.presets)
    payload = build_payload(args.output_dir, required)
    if args.max_players:
        payload["players"] = payload["players"][: args.max_players]

    presets, preset_tips, preset_order, skipped = collect_presets(
        args.output_dir, payload, args.presets
    )

    source = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    source = source.replace("__PRESETS__", json.dumps(presets, indent=2))
    source = source.replace("__PRESET_TIPS__", json.dumps(preset_tips, indent=2))
    source = source.replace("__PRESET_ORDER__", json.dumps(preset_order))

    CANVAS_DIR.mkdir(parents=True, exist_ok=True)
    target = CANVAS_DIR / args.name
    target.write_text(source, encoding="utf-8")

    print(
        "Wrote %s (%d players, %d levers, %d presets, %.0f KB)"
        % (
            target,
            len(payload["players"]),
            len(payload["levers"]),
            len(presets),
            len(source) / 1024,
        )
    )
    if skipped:
        print(
            "  Left out %s: they change categories that are not sliders here, so a "
            "preset would not reproduce the written numbers." % ", ".join(sorted(skipped))
        )


if __name__ == "__main__":
    main()
