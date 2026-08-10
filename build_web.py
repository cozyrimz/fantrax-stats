#!/usr/bin/env python3
"""Generate a self-contained scoring lab for GitHub Pages.

Uses the same embedded dataset and presets as the Cursor canvas, but emits a
single index.html that runs in any browser with no build step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_canvas import build_payload, collect_presets, resolve_required_levers

DOCS_DIR = Path(__file__).parent / "docs"

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Points Planner</title>
  <meta name="description" content="Explore Fantrax scoring weight changes and see how they affect player value, positional balance, and waiver depth." />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #fafafa;
      --surface: #ffffff;
      --text: #1a1a1a;
      --muted: #666;
      --border: #e0e0e0;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --success: #15803d;
      --font: system-ui, -apple-system, "Segoe UI", sans-serif;
      --mono: ui-monospace, "SF Mono", Menlo, monospace;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111;
        --surface: #1c1c1c;
        --text: #f0f0f0;
        --muted: #aaa;
        --border: #333;
        --accent: #60a5fa;
        --accent-soft: #1e293b;
        --success: #4ade80;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 24px 20px 48px; }
    h1 { font-size: 1.75rem; font-weight: 650; margin: 0 0 6px; }
    h2 { font-size: 1.1rem; font-weight: 600; margin: 0; }
    .sub { color: var(--muted); font-size: 0.92rem; margin: 0 0 12px; }
    .lead { font-size: 1.05rem; margin: 0 0 14px; max-width: 52rem; }
    .intro-list {
      margin: 0 0 16px;
      padding-left: 1.2rem;
      color: var(--muted);
      font-size: 0.88rem;
      max-width: 52rem;
    }
    .intro-list li { margin-bottom: 6px; }
    .intro-data { font-size: 0.82rem; color: var(--muted); margin: 0 0 20px; }
    .tip {
      display: inline-block;
      cursor: help;
      border-bottom: 1px dotted var(--muted);
    }
    .slider-row .tip {
      font-family: var(--mono);
      font-size: 0.82rem;
      border-bottom-style: dashed;
    }
    th .tip { border-bottom-color: currentColor; font-weight: 600; }
    #float-tip {
      position: fixed;
      z-index: 9999;
      max-width: 280px;
      padding: 8px 10px;
      border-radius: 6px;
      background: var(--text);
      color: var(--bg);
      font-size: 0.75rem;
      font-weight: 400;
      line-height: 1.35;
      pointer-events: none;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.18);
    }
    #float-tip[hidden] { display: none !important; }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .stat {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
    }
    .stat .val { font-size: 1.35rem; font-weight: 650; font-variant-numeric: tabular-nums; }
    .stat .lbl { font-size: 0.78rem; color: var(--muted); margin-top: 4px; }
    .stat.good .val { color: var(--success); }
    .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .pill {
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      border-radius: 999px;
      padding: 5px 12px;
      font-size: 0.82rem;
      cursor: pointer;
    }
    .pill.active { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
    .pill:hover { border-color: var(--accent); }
    .btn {
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--muted);
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 0.82rem;
      cursor: pointer;
    }
    .btn:hover { color: var(--text); border-color: var(--text); }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 20px;
      overflow: visible;
    }
    .card-hd {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      font-weight: 600;
      font-size: 0.9rem;
    }
    .card-bd { padding: 16px; }
    .slider-row {
      display: grid;
      grid-template-columns: 52px 1fr 96px;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
      font-size: 0.82rem;
    }
    .slider-row code { font-family: var(--mono); }
    .slider-row input[type=range] { width: 100%; accent-color: var(--accent); }
    .slider-row .wt { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
    .slider-row.changed .wt { color: var(--text); }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
    @media (max-width: 800px) { .grid2 { grid-template-columns: 1fr; } }
    .chart-box { height: 280px; position: relative; margin-bottom: 6px; }
    .caption { font-size: 0.75rem; color: var(--muted); margin: 0 0 16px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }
    th { font-weight: 600; color: var(--muted); font-size: 0.75rem; position: sticky; top: 0; background: var(--surface); }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    .table-wrap { overflow: auto; max-height: 520px; border: 1px solid var(--border); border-radius: 8px; }
    .callout {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      font-size: 0.88rem;
    }
    .callout strong { display: block; margin-bottom: 4px; }
    .section { margin-bottom: 24px; }
    .spacer { flex: 1; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Points Planner</h1>
    <p class="lead">Explore scoring changes for your Fantrax league before you edit anything in league settings.</p>
    <ul class="intro-list">
      <li>Move the category sliders to see how each stat's weight shifts player value, positional balance, and waiver depth.</li>
      <li>Presets load the analysis recommendations; you can tweak from there and compare against current scoring.</li>
      <li>Charts and the top-fifty list follow the season you pick; the position detail table shows every season side by side.</li>
      <li>All points are recomputed from real per-game logs — when sliders are at their defaults, totals match Fantrax exactly.</li>
    </ul>
    <p class="intro-data" id="intro-data"></p>

    <div class="stats" id="stats"></div>

    <div class="row section">
      <span style="font-size:0.82rem;color:var(--muted)">Presets</span>
      <button class="pill active" data-preset="current">Current scoring</button>
      <span id="preset-buttons"></span>
      <span class="spacer"></span>
      <button class="btn" id="reset-all">Reset all</button>
    </div>

    <div class="card section">
      <div class="card-hd">Category weights <span style="float:right;font-weight:400;color:var(--muted)">weight → adjusted</span></div>
      <div class="card-bd">
        <div class="row" style="margin-bottom:12px" id="pos-tabs"></div>
        <div id="sliders"></div>
      </div>
    </div>

    <div class="section">
      <h2>Value curve by position</h2>
      <p class="caption">Season fantasy points of the nth best player at each position. A flatter line means waiver pickups matter more.</p>
      <div class="chart-box"><canvas id="curve-chart"></canvas></div>
      <p class="caption" id="curve-source"></p>
    </div>

    <div class="grid2">
      <div>
        <h2>Supply against demand</h2>
        <p class="caption">Starting slots vs players scoring within 80% of a median starter.</p>
        <div class="chart-box"><canvas id="supply-chart"></canvas></div>
      </div>
      <div>
        <h2>Positional balance</h2>
        <p class="caption">Share of the top 50 scorers by position vs share of starting slots.</p>
        <div class="chart-box"><canvas id="balance-chart"></canvas></div>
      </div>
    </div>

    <div class="section">
      <h2>Position detail</h2>
      <p class="caption">Each season recomputed separately under the current sliders.</p>
      <div class="table-wrap">
        <table id="detail-table"><thead></thead><tbody></tbody></table>
      </div>
    </div>

    <div class="section">
      <div class="row" style="margin-bottom:8px">
        <h2>Top 50 under the current sliders</h2>
        <span class="spacer"></span>
        <span id="season-tabs"></span>
      </div>
      <p class="caption" id="list-caption"></p>
      <div class="table-wrap">
        <table id="top-table"><thead></thead><tbody></tbody></table>
      </div>
    </div>

    <div class="callout section">
      <strong>How points are recomputed</strong>
      Scoring is linear in the match statistics. Categories not exposed as sliders are folded into each player's base score, so totals match Fantrax exactly when every slider sits at its original value.
    </div>
  </div>
  <div id="float-tip" role="tooltip" hidden></div>

  <script>
    const DATA = __DATA__;
    const PRESETS = __PRESETS__;
    const POSITIONS = ["D", "M", "F", "G"];
    const POS_NAMES = { D: "Defenders", M: "Midfielders", F: "Forwards", G: "Keepers" };
    const POS_COLORS = { D: "#1f77b4", M: "#2ca02c", F: "#d62728", G: "#9467bd" };
    const RANKS = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110];

    const STAT_TIPS = {
      gini: "How unevenly value is spread above replacement level. 0 means everyone looks the same; higher means a few players dominate.",
      ratio: "Season points of the #1 player divided by the #50 player in the selected season. Lower means the elite tier is less far ahead.",
      top10: "What fraction of all points scored by the top fifty players is concentrated in just the top ten. Lower means the elite tier is less top-heavy.",
      mix: "How many of the top fifty scorers are defenders, midfielders, forwards, and keepers. Compare to your lineup slots (roughly 36% / 36% / 18% / 10%).",
    };

    const DETAIL_TIPS = {
      "Starting slots": "How many players at this position all twelve teams must start each week, given your lineup minimums and flex rules.",
      "Replacement": "Points of the best player at this position who is not a weekly starter — the waiver-wire benchmark.",
      "Median starter": "Typical points among players who actually start at this position. Usable players are those scoring at least 80% of this.",
      "Elite mult.": "Average points of the top twelve starters divided by replacement level. Lower means less gap between stars and waiver fodder.",
      "Usable": "Players scoring at least 80% of a median starter — roughly startable in a pinch.",
      "Surplus": "Usable players minus starting slots. Positive means more startable options than roster spots demand.",
      "In top 50": "How many of this season's top fifty scorers play this position under the current weights.",
      "Move": "Rank change versus current Fantrax scoring in the selected season. Plus means the player rose under your weights.",
    };

    let multipliers = loadJson("scoring-multipliers", {});
    let position = localStorage.getItem("lab-position") || "D";
    let listSeason = localStorage.getItem("lab-list-season") || DATA.seasons[DATA.seasons.length - 1];

    let charts = {};

    function loadJson(key, fallback) {
      try { return JSON.parse(localStorage.getItem(key)) || fallback; } catch { return fallback; }
    }
    function saveState() {
      localStorage.setItem("scoring-multipliers", JSON.stringify(multipliers));
      localStorage.setItem("lab-position", position);
      localStorage.setItem("lab-list-season", listSeason);
    }

    function scaleWeights(m) {
      const out = {};
      for (const pos of Object.keys(DATA.weights)) {
        out[pos] = {};
        for (const [cat, weight] of Object.entries(DATA.weights[pos])) {
          const factor = (m[pos] && m[pos][cat]) || (m.all && m.all[cat]) || 1;
          out[pos][cat] = Math.round(weight * factor * 1000) / 1000;
        }
      }
      return out;
    }

    function scorePlayers(weights, season) {
      return DATA.players.map((player, index) => {
        const table = weights[player.p] || {};
        let points = player.b;
        DATA.levers.forEach((lever, i) => {
          const w = table[lever.c];
          if (w) points += w * player.s[i];
        });
        return { index, player, points };
      }).filter(row => !season || row.player.y === season);
    }

    function starterDemand(scored) {
      const { teams, starters, minActive, maxActive } = DATA.league;
      const taken = {};
      const used = new Set();
      const ranked = [...scored].sort((a, b) => b.points - a.points);
      for (const pos of POSITIONS) {
        const need = teams * (minActive[pos] || 0);
        const pool = ranked.filter(r => r.player.p === pos).slice(0, need);
        taken[pos] = pool.length;
        pool.forEach(r => used.add(r.index));
      }
      let remaining = teams * starters - Object.values(taken).reduce((a, b) => a + b, 0);
      for (const row of ranked) {
        if (remaining <= 0) break;
        if (used.has(row.index)) continue;
        const pos = row.player.p;
        const ceiling = teams * (maxActive[pos] || 0);
        if (taken[pos] >= ceiling) continue;
        taken[pos] += 1;
        used.add(row.index);
        remaining -= 1;
      }
      return taken;
    }

    function gini(values) {
      const clipped = values.map(v => Math.max(0, v)).sort((a, b) => a - b);
      const total = clipped.reduce((a, b) => a + b, 0);
      if (!clipped.length || total === 0) return 0;
      let weighted = 0;
      clipped.forEach((v, i) => { weighted += (2 * (i + 1) - clipped.length - 1) * v; });
      return weighted / (clipped.length * total);
    }

    function analyse(weights, season) {
      const scored = scorePlayers(weights, season);
      const ranked = [...scored].sort((a, b) => b.points - a.points);
      const demand = starterDemand(scored);
      const replacement = {};
      const byPosition = {};
      for (const pos of POSITIONS) {
        const pool = ranked.filter(r => r.player.p === pos);
        byPosition[pos] = pool;
        const count = demand[pos] || 0;
        replacement[pos] = pool.length > count ? pool[count].points : (pool[pool.length - 1]?.points ?? 0);
      }
      const vor = ranked.map(r => r.points - (replacement[r.player.p] ?? 0));
      const top50 = ranked.slice(0, 50);
      const positions = POSITIONS.map(pos => {
        const pool = byPosition[pos];
        const count = demand[pos] || 0;
        const starters = pool.slice(0, count);
        const median = starters.length ? starters[Math.floor(starters.length / 2)].points : 0;
        const usable = pool.filter(r => r.points >= 0.8 * median).length;
        const elite = starters.slice(0, 12);
        const eliteMean = elite.length ? elite.reduce((a, b) => a + b.points, 0) / elite.length : 0;
        return {
          pos, players: pool.length, demand: count,
          replacement: replacement[pos] ?? 0, median, usable,
          surplus: usable - count,
          eliteMultiple: replacement[pos] > 0 ? eliteMean / replacement[pos] : null,
          inTop50: top50.filter(r => r.player.p === pos).length,
        };
      });
      return {
        ranked, byPosition, demand, positions,
        gini: gini(vor),
        top50,
        ratio: ranked.length > 49 && ranked[49].points > 0 ? ranked[0].points / ranked[49].points : null,
        top10Share: top50.length ? top50.slice(0, 10).reduce((a, b) => a + b.points, 0) / top50.reduce((a, b) => a + b.points, 0) : 0,
      };
    }

    function delta(value, base, betterLower = true) {
      const change = value - base;
      if (Math.abs(change) < 0.0005) return "no change";
      const better = betterLower ? change < 0 : change > 0;
      return `${change > 0 ? "+" : ""}${change.toFixed(3)} ${better ? "better" : "worse"}`;
    }

    function esc(s) {
      return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/"/g,"&quot;");
    }

    function escAttr(s) {
      return esc(s).replace(/'/g, "&#39;");
    }

    function tip(label, hint) {
      if (!hint) return esc(label);
      return `<span class="tip" tabindex="0" title="${escAttr(hint)}" data-tip="${escAttr(hint)}">${esc(label)}</span>`;
    }

    function thTip(label, hint, num) {
      return `<th class="${num ? "num" : ""}">${tip(label, hint)}</th>`;
    }

    const floatTip = document.getElementById("float-tip");
    let floatTipTarget = null;

    function showFloatTip(el) {
      const text = el.getAttribute("data-tip");
      if (!text) return hideFloatTip();
      floatTipTarget = el;
      floatTip.textContent = text;
      floatTip.hidden = false;
      const margin = 8;
      const anchor = el.getBoundingClientRect();
      floatTip.style.left = "0px";
      floatTip.style.top = "0px";
      const box = floatTip.getBoundingClientRect();
      let left = anchor.right + margin;
      let top = anchor.top + (anchor.height - box.height) / 2;
      if (left + box.width > window.innerWidth - margin) {
        left = anchor.left - margin - box.width;
      }
      if (top + box.height > window.innerHeight - margin) {
        top = window.innerHeight - margin - box.height;
      }
      if (top < margin) top = margin;
      floatTip.style.left = `${Math.max(margin, left)}px`;
      floatTip.style.top = `${top}px`;
    }

    function hideFloatTip() {
      floatTipTarget = null;
      floatTip.hidden = true;
    }

    document.addEventListener(
      "pointerover",
      (e) => {
        const el = e.target.closest("[data-tip]");
        if (el) showFloatTip(el);
      },
      true,
    );

    document.addEventListener(
      "pointerout",
      (e) => {
        const el = e.target.closest("[data-tip]");
        if (!el) return;
        const next = e.relatedTarget;
        if (next && el.contains(next)) return;
        hideFloatTip();
      },
      true,
    );

    document.addEventListener("focusin", (e) => {
      const el = e.target.closest("[data-tip]");
      if (el) showFloatTip(el);
    });

    document.addEventListener("focusout", (e) => {
      const el = e.target.closest("[data-tip]");
      if (el && floatTipTarget === el) hideFloatTip();
    });

    window.addEventListener("scroll", hideFloatTip, true);
    window.addEventListener("resize", () => {
      if (floatTipTarget) showFloatTip(floatTipTarget);
    });

    function render() {
      const weights = scaleWeights(multipliers);
      const current = analyse(weights, listSeason);
      const baseline = analyse(scaleWeights({}), listSeason);
      const seasonDetails = DATA.seasons.flatMap(season =>
        analyse(weights, season).positions.map(row => ({ season, ...row }))
      );

      document.getElementById("intro-data").textContent =
        `${DATA.playerCount} player-seasons in the dataset (${DATA.seasons.join(", ")}). Headline stats and charts use ${listSeason} unless noted.`;

      const giniDelta = delta(current.gini, baseline.gini);
      document.getElementById("stats").innerHTML = `
        <div class="stat ${current.gini < baseline.gini ? "good" : ""}">
          <div class="val">${current.gini.toFixed(3)}</div>
          <div class="lbl">${tip("Value inequality (Gini)", STAT_TIPS.gini)} · ${giniDelta}</div>
        </div>
        <div class="stat">
          <div class="val">${current.ratio ? current.ratio.toFixed(2) + "x" : "n/a"}</div>
          <div class="lbl">${tip("Best player vs rank 50", STAT_TIPS.ratio)}${current.ratio && baseline.ratio ? " · " + delta(current.ratio, baseline.ratio) : ""}</div>
        </div>
        <div class="stat">
          <div class="val">${Math.round(100 * current.top10Share)}%</div>
          <div class="lbl">${tip("Top 10 share of top 50 points", STAT_TIPS.top10)}</div>
        </div>
        <div class="stat">
          <div class="val">${current.positions.map(p => p.inTop50).join(" / ")}</div>
          <div class="lbl">${tip("Top 50 mix (D / M / F / G)", STAT_TIPS.mix)}</div>
        </div>`;

      document.getElementById("pos-tabs").innerHTML = POSITIONS.map(pos =>
        `<button class="pill ${pos === position ? "active" : ""}" data-pos="${pos}">${POS_NAMES[pos]}</button>`
      ).join("");

      document.getElementById("sliders").innerHTML = DATA.levers.map((lever, i) => {
        const base = DATA.weights[position]?.[lever.c];
        if (base === undefined) return "";
        const factor = (multipliers[position] && multipliers[position][lever.c]) || 1;
        const cur = Math.round(base * factor * 1000) / 1000;
        const changed = Math.abs(factor - 1) > 0.001;
        return `<div class="slider-row ${changed ? "changed" : ""}">
          ${tip(lever.c, lever.name)}
          <input type="range" min="0.1" max="3" step="0.05" value="${factor}"
            title="${escAttr(lever.name)}"
            data-pos="${position}" data-cat="${lever.c}" />
          <span class="wt">${base} → ${cur}</span>
        </div>`;
      }).join("");

      document.getElementById("curve-source").textContent =
        `Source: Fantrax game logs for ${listSeason}`;

      document.getElementById("season-tabs").innerHTML = DATA.seasons.map(season =>
        `<button class="pill ${season === listSeason ? "active" : ""}" data-season="${season}">${season}</button>`
      ).join("");

      document.getElementById("list-caption").textContent =
        `${DATA.seasonCounts[listSeason] || 0} players in ${listSeason}. Move compares rank against current scoring for that season.`;

      const detailHead = [
        ["Season", null, false],
        ["Position", null, false],
        ["Players", "Players with enough minutes to appear in this season's pool.", false],
        ["Starting slots", DETAIL_TIPS["Starting slots"], true],
        ["Replacement", DETAIL_TIPS["Replacement"], true],
        ["Median starter", DETAIL_TIPS["Median starter"], true],
        ["Elite mult.", DETAIL_TIPS["Elite mult."], true],
        ["Usable", DETAIL_TIPS["Usable"], true],
        ["Surplus", DETAIL_TIPS["Surplus"], true],
        ["In top 50", DETAIL_TIPS["In top 50"], true],
      ];
      document.querySelector("#detail-table thead").innerHTML =
        "<tr>" + detailHead.map(([label, hint, num]) => thTip(label, hint, num)).join("") + "</tr>";
      document.querySelector("#detail-table tbody").innerHTML = seasonDetails.map(row => `<tr>
        <td>${esc(row.season)}</td><td>${POS_NAMES[row.pos]}</td>
        <td class="num">${row.players}</td><td class="num">${row.demand}</td>
        <td class="num">${Math.round(row.replacement)}</td><td class="num">${Math.round(row.median)}</td>
        <td class="num">${row.eliteMultiple ? row.eliteMultiple.toFixed(2) + "x" : "n/a"}</td>
        <td class="num">${row.usable}</td>
        <td class="num">${row.surplus > 0 ? "+" : ""}${row.surplus}</td>
        <td class="num">${row.inTop50}</td></tr>`).join("");

      const topHead = [
        ["#", null, true],
        ["Player", null, false],
        ["Position", null, false],
        ["Type", "Playing style cluster derived from per-game stat rates (e.g. creator, stopper).", false],
        ["Games", "Matches played in this season.", true],
        ["Points", "Season total under the current slider weights.", true],
        ["Move", DETAIL_TIPS["Move"], true],
      ];
      document.querySelector("#top-table thead").innerHTML =
        "<tr>" + topHead.map(([label, hint, num]) => thTip(label, hint, num)).join("") + "</tr>";
      document.querySelector("#top-table tbody").innerHTML = current.ranked.slice(0, 50).map((row, i) => {
        const before = baseline.ranked.findIndex(o => o.index === row.index) + 1;
        const move = before - (i + 1);
        const moveStr = move === 0 ? "-" : `${move > 0 ? "+" : ""}${move}`;
        return `<tr>
          <td class="num">${i + 1}</td><td>${esc(row.player.n)}</td><td>${row.player.p}</td>
          <td>${esc(row.player.a)}</td><td class="num">${row.player.g}</td>
          <td class="num">${Math.round(row.points)}</td><td class="num">${moveStr}</td></tr>`;
      }).join("");

      renderCharts(current);
      saveState();
    }

    function renderCharts(current) {
      const labels = RANKS.map(r => "#" + r);
      const curveDatasets = POSITIONS.map(pos => ({
        label: POS_NAMES[pos],
        data: RANKS.map(rank => {
          const pool = current.byPosition[pos];
          return pool && pool.length >= rank ? Math.round(pool[rank - 1].points) : 0;
        }),
        borderColor: POS_COLORS[pos],
        backgroundColor: POS_COLORS[pos],
        tension: 0.2,
        pointRadius: 3,
      }));

      const slotShare = POSITIONS.map(p =>
        Math.round(100 * (DATA.league.teams * (DATA.league.baseline[p] || 0)) / (DATA.league.teams * DATA.league.starters))
      );

      upsertChart("curve-chart", "line", {
        data: { labels, datasets: curveDatasets },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { position: "bottom" } },
          scales: {
            y: { title: { display: true, text: "Season fantasy points" } },
            x: { title: { display: true, text: "Rank within position" } },
          },
        },
      });

      upsertChart("supply-chart", "bar", {
        data: {
          labels: POSITIONS.map(p => POS_NAMES[p]),
          datasets: [
            { label: "Starting slots", data: current.positions.map(p => p.demand), backgroundColor: "#8884" },
            { label: "Usable players", data: current.positions.map(p => p.usable), backgroundColor: "#2563eb99" },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { position: "bottom" } },
          scales: { y: { beginAtZero: true } },
        },
      });

      upsertChart("balance-chart", "bar", {
        data: {
          labels: POSITIONS.map(p => POS_NAMES[p]),
          datasets: [
            { label: "Share of top 50", data: current.positions.map(p => Math.round(100 * p.inTop50 / 50)), backgroundColor: "#2563eb99" },
            { label: "Share of starting slots", data: slotShare, backgroundColor: "#8884" },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { position: "bottom" } },
          scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: "Percent" } } },
        },
      });
    }

    function upsertChart(id, type, config) {
      const canvas = document.getElementById(id);
      if (charts[id]) { charts[id].destroy(); }
      charts[id] = new Chart(canvas, { type, ...config });
    }

    function applyPreset(name) {
      multipliers = name === "current" ? {} : (PRESETS[name] ? JSON.parse(JSON.stringify(PRESETS[name])) : {});
      document.querySelectorAll("[data-preset]").forEach(el => {
        el.classList.toggle("active", el.dataset.preset === name);
      });
      render();
    }

    document.getElementById("preset-buttons").innerHTML = Object.keys(PRESETS).map(name =>
      `<button class="pill" data-preset="${esc(name)}">${esc(name)}</button>`
    ).join("");

    document.addEventListener("click", e => {
      const preset = e.target.closest("[data-preset]");
      if (preset) { applyPreset(preset.dataset.preset); return; }
      const pos = e.target.closest("[data-pos]");
      if (pos && !pos.dataset.cat) { position = pos.dataset.pos; render(); return; }
      const season = e.target.closest("[data-season]");
      if (season) { listSeason = season.dataset.season; render(); return; }
    });

    document.getElementById("reset-all").addEventListener("click", () => applyPreset("current"));

    document.getElementById("sliders").addEventListener("input", e => {
      const t = e.target;
      if (t.matches("input[type=range]")) {
        const pos = t.dataset.pos, cat = t.dataset.cat, val = Number(t.value);
        multipliers = { ...multipliers, [pos]: { ...(multipliers[pos] || {}), [cat]: val } };
        document.querySelectorAll("[data-preset]").forEach(el => el.classList.remove("active"));
        render();
      }
    });

    if (Object.keys(multipliers).length) {
      document.querySelectorAll("[data-preset]").forEach(el => el.classList.remove("active"));
      render();
    } else {
      applyPreset("current");
    }
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--presets", type=Path, default=Path("proposals"))
    parser.add_argument("--out", type=Path, default=DOCS_DIR / "index.html")
    args = parser.parse_args()

    required = resolve_required_levers(args.output_dir, args.presets)
    payload = build_payload(args.output_dir, required)
    presets, skipped = collect_presets(args.output_dir, payload, args.presets)

    html = HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    html = html.replace("__PRESETS__", json.dumps(presets, separators=(",", ":")))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    (args.out.parent / ".nojekyll").touch()

    print(
        "Wrote %s (%d players, %d levers, %d presets, %.0f KB)"
        % (args.out, len(payload["players"]), len(payload["levers"]), len(presets), len(html) / 1024)
    )
    if skipped:
        print("  Presets omitted from web UI: %s" % ", ".join(sorted(skipped)))


if __name__ == "__main__":
    main()
