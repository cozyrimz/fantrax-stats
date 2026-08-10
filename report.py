#!/usr/bin/env python3
"""Render the scoring balance diagnosis as charts and a written report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import diagnose
import fantrax_data as fx
import league_config as lc
import simulate
import sweep
import validate

CHART_DIR = "charts"
POSITION_ORDER = ["D", "M", "F", "G"]
POSITION_NAMES = {"D": "Defenders", "M": "Midfielders", "F": "Forwards", "G": "Keepers"}


def chart_value_curves(working: pd.DataFrame, metrics: Dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for pos in POSITION_ORDER:
        subset = working[working["position"] == pos].sort_values("points", ascending=False)
        if subset.empty:
            continue
        ax.plot(range(1, len(subset) + 1), subset["points"].to_numpy(), label=POSITION_NAMES[pos])
        info = metrics["positions"].get(pos)
        if info:
            ax.axvline(info["starterDemand"], linestyle=":", linewidth=0.9, alpha=0.5)

    ax.set_xlabel("Rank within position")
    ax.set_ylabel("Season fantasy points")
    ax.set_title("Value curve by position (dotted lines mark league-wide starter demand)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_category_decomposition(
    working: pd.DataFrame, contributions: pd.DataFrame, path: Path, top_n: int = 50
) -> None:
    ranked = working.sort_values("points", ascending=False).head(top_n)
    subset = contributions.loc[ranked.index]

    groups = {
        "Goals and shooting": ["G", "SOT", "PKG", "FKG", "BCM", "PKM"],
        "Creation": ["A", "A2", "KP", "BCC", "AC", "CF", "ABS", "AOG", "APL", "AR", "APKG", "ASOP", "AFKG", "AHW"],
        "Passing volume": ["AP", "LBA"],
        "Defensive actions": ["TkW", "Int", "IntB", "CLR", "CLO", "BS", "BC", "TLM"],
        "Duels and recoveries": ["AER", "AERL", "DW", "DL", "BR", "CoS", "FS", "DIS"],
        "Clean sheets and keeping": ["CS", "Sv", "SvIB", "PKS", "HCS", "Sm", "Pu", "GA", "GAO"],
        "Discipline and errors": ["YC", "SYC", "RC", "FC", "HB", "DP", "ErS", "ErG", "OG", "Tu", "PLC", "CC", "SB"],
    }

    totals = {}
    for label, members in groups.items():
        present = [c for c in members if c in subset.columns]
        totals[label] = subset[present].sum().sum() if present else 0.0

    labels = list(totals.keys())
    values = [totals[k] for k in labels]
    order = np.argsort(values)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([labels[i] for i in order][::-1], [values[i] for i in order][::-1])
    ax.set_xlabel("Total fantasy points earned by the top %d players" % top_n)
    ax.set_title("Where the top %d players' points come from" % top_n)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_archetype_mix(working: pd.DataFrame, path: Path, top_n: int = 50) -> None:
    if "archetypeFull" not in working.columns:
        return
    ranked = working.sort_values("points", ascending=False)
    top = ranked.head(top_n)["archetypeFull"].value_counts()
    everyone = ranked["archetypeFull"].value_counts()

    labels = list(top.index)
    share_top = [top[label] / top.sum() for label in labels]
    share_all = [everyone.get(label, 0) / everyone.sum() for label in labels]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(labels) + 2)))
    ax.barh(y - 0.2, share_top, height=0.4, label="Share of top %d" % top_n)
    ax.barh(y + 0.2, share_all, height=0.4, label="Share of all players")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Share of players")
    ax.set_title("Which player types reach the top of the scoring")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_waiver_depth(metrics: Dict, path: Path) -> None:
    positions = [p for p in POSITION_ORDER if p in metrics["positions"]]
    demand = [metrics["positions"][p]["starterDemand"] for p in positions]
    usable = [metrics["positions"][p]["usablePlayers"] for p in positions]

    x = np.arange(len(positions))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, demand, width=0.4, label="Starting slots the league must fill")
    ax.bar(x + 0.2, usable, width=0.4, label="Players within 80% of a median starter")
    ax.set_xticks(x)
    ax.set_xticklabels([POSITION_NAMES[p] for p in positions])
    ax.set_ylabel("Players")
    ax.set_title("Usable supply against demand (bars above demand mean live waivers)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_tradeoff(frame: pd.DataFrame, path: Path) -> None:
    """Show the two things that pull against each other as volume is reweighted.

    Positions are re-balanced at every point on the sweep, so the positional mix
    is deliberately flat here and is not worth plotting. What moves is how far
    ahead the elite are over a season, against how varied and how changeable the
    league is week to week.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

    kept = 100 * frame["volumeKept"]
    left.plot(kept, frame["eliteMultiple"], marker="o", color="#c1440e")
    left.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
    left.annotate(
        "replacement level", (kept.iloc[0], 1.0), textcoords="offset points", xytext=(4, 5), fontsize=8, color="#666666"
    )
    # Anchored at replacement level rather than zoomed to the data, because the
    # claim being made is that this barely moves.
    left.set_ylim(0.9, max(2.9, frame["eliteMultiple"].max() * 1.1))
    left.set_xlabel("Volume categories set to this % of current weight")
    left.set_ylabel("Best player as a multiple of replacement")
    left.set_title("Season top-heaviness barely responds")
    left.grid(alpha=0.3)

    right.plot(
        kept,
        frame["biggestTypeShare"],
        marker="o",
        color="#1f77b4",
        label="Elite tier held by its most common player type",
    )
    right.plot(
        kept,
        frame["eliteWeeklyShare"],
        marker="s",
        color="#2ca02c",
        label="Weekly top 30 held by the season's best 20",
    )
    right.set_xlabel("Volume categories set to this % of current weight")
    right.set_ylabel("Share")
    right.set_title("Variety and weekly churn respond strongly")
    right.legend(fontsize=8)
    right.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def volume_table(profiles: Dict[str, pd.DataFrame], position: str, limit: int = 12) -> str:
    profile = profiles.get(position)
    if profile is None or profile.empty:
        return "_No data._"

    ranked = profile.sort_values("totalPoints", ascending=False).head(limit)
    lines = [
        "| Category | Points | Share of positive points | Spread between players | Tracks games played |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in ranked.iterrows():
        lines.append(
            "| %s | %.0f | %.1f%% | %.2f | %.2f |"
            % (
                row["category"],
                row["totalPoints"],
                100 * row["shareOfAllPoints"],
                row["spread"],
                row["corrWithGames"],
            )
        )
    return "\n".join(lines)


def build_report(
    metrics: Dict,
    working: pd.DataFrame,
    profiles: Dict[str, pd.DataFrame],
    weights: Dict[str, Dict[str, float]],
    season_tag: str,
    chart_paths: Dict[str, str],
    recommendation: Dict = None,
    reliability: tuple = None,
    tradeoff: pd.DataFrame = None,
) -> str:
    overall = metrics["overall"]
    ranked = working.sort_values("points", ascending=False)

    lines: List[str] = []
    lines.append("# Scoring balance diagnosis (%s)" % season_tag)
    lines.append("")

    if not metrics.get("poolAdequate", True):
        thin = [p for p, info in metrics["positions"].items() if not info["poolAdequate"]]
        lines.append(
            "> Caution: the player pool for %s is not much larger than league-wide demand, "
            "so replacement level sits near the bottom of the sample and the waiver metrics "
            "understate reality. Re-run once the export covers the full pool."
            % ", ".join(thin)
        )
        lines.append("")

    lines.append("## The shape of the problem")
    lines.append("")
    lines.append(
        "Across %d players, the scoring curve runs from %.0f points at the top to %s at "
        "rank 50 and %s at rank 100. The best player is worth %s times the fiftieth. The top "
        "ten players hold %.0f%% of all points scored by the top fifty."
        % (
            metrics["players"],
            overall["rank1"],
            overall["rank50"],
            overall["rank100"],
            overall["topToFiftiethRatio"],
            100 * overall["top10ShareOfTop50"],
        )
    )
    lines.append("")
    lines.append(
        "Inequality of value over replacement, measured as a Gini coefficient where zero is "
        "perfectly flat, is **%.3f**." % overall["giniOfValue"]
    )
    lines.append("")
    lines.append("![Value curves](%s)" % chart_paths["curves"])
    lines.append("")

    churn = metrics.get("churn") or {}
    if churn:
        lines.append("## How live the waiver wire is")
        lines.append("")
        lines.append(
            "A season-long curve says who was best over the year, but what decides whether a "
            "waiver pickup can help is whether the weekly leaderboard moves. Across the season "
            "%d different players filled the %d available places in a weekly top thirty, and the "
            "twenty best players of the season held **%.0f%%** of those places. The more that "
            "share falls, the more often somebody available is worth starting."
            % (
                churn["distinctPlayersInWeeklyTop"],
                churn["weeklyTopSlots"],
                100 * churn["eliteShareOfWeeklyTop"],
            )
        )
        lines.append("")

    lines.append("## Positional balance")
    lines.append("")
    mix = metrics["topPositionMix"]
    demand = metrics["starterDemand"]
    lines.append(
        "The league must field %d defenders, %d midfielders, %d forwards and %d keepers each "
        "week across all twelve teams. The top fifty scorers break down as %s."
        % (
            demand.get("D", 0),
            demand.get("M", 0),
            demand.get("F", 0),
            demand.get("G", 0),
            ", ".join("%d %s" % (count, POSITION_NAMES[pos].lower()) for pos, count in mix.items()),
        )
    )
    lines.append("")
    lines.append("| Position | Players | Starting slots | Replacement level | Median starter | Top 12 average | Elite multiple of replacement | Usable players | Surplus over demand |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for pos in POSITION_ORDER:
        info = metrics["positions"].get(pos)
        if not info:
            continue
        lines.append(
            "| %s | %d | %d | %.0f | %.0f | %.0f | %s | %d | %+d |"
            % (
                POSITION_NAMES[pos],
                info["players"],
                info["starterDemand"],
                info["replacementPoints"],
                info["starterMedianPoints"],
                info["eliteOverReplacement"] + info["replacementPoints"],
                ("%.2f" % info["eliteMultiple"]) if info["eliteMultiple"] else "n/a",
                info["usablePlayers"],
                info["usableSurplus"],
            )
        )
    lines.append("")
    lines.append("![Usable supply](%s)" % chart_paths["waivers"])
    lines.append("")

    lines.append("## Where the points actually come from")
    lines.append("")
    lines.append(
        "Every category's contribution is its recovered weight times the volume players "
        "generate. A category that pays a large share of total points, varies little between "
        "players and tracks games played is rewarding time on the pitch rather than anything "
        "a manager can draft for."
    )
    lines.append("")
    lines.append("![Category decomposition](%s)" % chart_paths["categories"])
    lines.append("")
    for pos in POSITION_ORDER:
        if pos not in profiles:
            continue
        lines.append("### %s" % POSITION_NAMES[pos])
        lines.append("")
        lines.append(volume_table(profiles, pos))
        lines.append("")

    lines.append("## Player types")
    lines.append("")
    if metrics.get("topArchetypeMix"):
        lines.append(
            "Archetypes are derived by clustering per-game stat profiles within each position "
            "after dividing out overall involvement, so they capture playing style rather than "
            "playing time. The top fifty splits as %s."
            % ", ".join("%d %s" % (n, a) for a, n in metrics["topArchetypeMix"].items())
        )
        lines.append("")
        lines.append("![Archetype mix](%s)" % chart_paths["archetypes"])
        lines.append("")

    lines.append("## Top 25 under current scoring")
    lines.append("")
    lines.append("| Rank | Player | Position | Type | Games | Points | Over replacement |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for i, (_, row) in enumerate(ranked.head(25).iterrows(), start=1):
        lines.append(
            "| %d | %s | %s | %s | %d | %.0f | %+.0f |"
            % (
                i,
                row["name"],
                row["position"],
                row.get("archetype", ""),
                row["GP"],
                row["points"],
                row["vor"],
            )
        )
    lines.append("")

    if recommendation:
        lines.extend(recommendation_section(recommendation, weights, chart_paths, tradeoff))
        if reliability is not None:
            lines.extend(reliability_section(reliability[0], reliability[1], metrics["players"]))

    lines.append("## Current weights")
    lines.append("")
    lines.append("Recovered from the per-game data and reproducing Fantrax totals exactly.")
    lines.append("")
    matrix = pd.DataFrame(weights).reindex(sorted({c for t in weights.values() for c in t}))
    lines.append("| Category | D | M | F | G |")
    lines.append("| --- | --- | --- | --- | --- |")
    for category, row in matrix.iterrows():
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                category,
                _fmt(row.get("D")),
                _fmt(row.get("M")),
                _fmt(row.get("F")),
                _fmt(row.get("G")),
            )
        )
    lines.append("")

    return "\n".join(lines)


def tradeoff_section(frame: pd.DataFrame) -> List[str]:
    """State what the sweep shows about which goals the scoring can actually serve."""
    ordered = frame.sort_values("volumeKept")
    low, high = ordered.iloc[0], ordered.iloc[-1]
    current = ordered[ordered["volumeKept"] == 1.0]
    current = current.iloc[0] if len(current) else ordered.iloc[len(ordered) // 2]
    best_churn = ordered.loc[ordered["distinctWeekly"].idxmax()]

    lines = ["### What the scoring can and cannot fix", ""]
    lines.append(
        "Positions are re-balanced at every point below, so what is left is the effect of the "
        "volume categories themselves. Two things move in opposite directions, and one barely "
        "moves at all."
    )
    lines.append("")
    lines.append(
        "Over a whole season the best player at a position is worth about %.1f times the "
        "replacement player, and that hardly changes no matter how the volume categories are "
        "weighted: across the full sweep, from %d%% to %d%% of their current value, it only "
        "moves between %.2f and %.2f. Elite players are roughly three times a replacement "
        "player per game in nearly every category at once, so no linear reweighting can pull "
        "them back towards the pack. The gap between the top of the league and the waiver wire "
        "is a property of the sport, not of the scoring, and it needs roster rules rather than "
        "weights if it has to change."
        % (
            current["eliteMultiple"],
            int(100 * low["volumeKept"]),
            int(100 * high["volumeKept"]),
            ordered["eliteMultiple"].min(),
            ordered["eliteMultiple"].max(),
        )
    )
    lines.append("")
    lines.append(
        "What does respond is variety and week-to-week churn, and they favour cutting. Cutting "
        "the volume categories to %d%% leaves the elite tier's most common player type holding "
        "%.0f%% of it rather than %.0f%%, and the season's best twenty players holding %.0f%% of "
        "weekly top-thirty places rather than %.0f%%, with %d different players reaching a weekly "
        "top thirty against %d now. Raising those categories instead does the reverse: it buys "
        "%.2f off the elite multiple and gives up variety and churn to get it. Since waiver "
        "activity comes from players being worth starting in a given week, the cut is the side "
        "worth being on."
        % (
            int(100 * best_churn["volumeKept"]),
            100 * best_churn["biggestTypeShare"],
            100 * current["biggestTypeShare"],
            100 * best_churn["eliteWeeklyShare"],
            100 * current["eliteWeeklyShare"],
            best_churn["distinctWeekly"],
            current["distinctWeekly"],
            current["eliteMultiple"] - high["eliteMultiple"],
        )
    )
    lines.append("")
    return lines


def recommendation_section(
    result: Dict,
    weights: Dict[str, Dict[str, float]],
    chart_paths: Dict[str, str],
    tradeoff: pd.DataFrame = None,
) -> List[str]:
    """The concrete weight set, what it changes, and who it moves."""
    proposal = result["proposal"]
    lines: List[str] = ["## Recommended weights", ""]
    lines.append(proposal.get("description", ""))
    lines.append("")

    if "tradeoff" in chart_paths and tradeoff is not None:
        lines.extend(tradeoff_section(tradeoff))
        lines.append("![Trade-off](%s)" % chart_paths["tradeoff"])
        lines.append("")

    lines.append("### What changes")
    lines.append("")
    lines.append("| Category | Position | Now | Proposed |")
    lines.append("| --- | --- | --- | --- |")
    proposed = result["weights"]
    for position in POSITION_ORDER:
        for category in sorted(proposed.get(position, {})):
            before = weights.get(position, {}).get(category)
            after = proposed[position][category]
            if before is None or before == after:
                continue
            lines.append("| %s | %s | %s | %s |" % (category, position, _fmt(before), _fmt(after)))
    lines.append("")

    lines.append("### What it does")
    lines.append("")
    comparison = result["comparison"]
    lines.append("| Measure | Now | Proposed | Change |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in comparison.iterrows():
        lines.append(
            "| %s | %s | %s | %s |"
            % (row["metric"], _fmt(row["current"]), _fmt(row["proposed"]), _fmt(row["change"]))
        )
    lines.append("")

    for title, key in (("Biggest risers", "risers"), ("Biggest fallers", "fallers")):
        movers = result.get(key)
        if movers is None or len(movers) == 0:
            continue
        lines.append("### %s" % title)
        lines.append("")
        lines.append("| Player | Position | Rank now | Rank proposed | Move |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in movers.head(12).iterrows():
            lines.append(
                "| %s | %s | %d | %d | %+d |"
                % (row["name"], row["position"], row["rankBefore"], row["rankAfter"], row["move"])
            )
        lines.append("")

    return lines


def reliability_section(frame: pd.DataFrame, notes: List[str], players: int) -> List[str]:
    """State plainly which claims the data supports and which it does not."""
    lines = ["## How far to trust this", ""]
    lines.append(
        "The same weights were replayed on each season separately. A change fitted to one "
        "season can look excellent there and do nothing elsewhere, so only effects that point "
        "the same way every time are worth acting on."
    )
    lines.append("")
    lines.append("| Season | Players | Inequality before | Inequality after | %s |" % " | ".join(POSITION_ORDER))
    lines.append("| --- | --- | --- | --- | %s |" % " | ".join(["---"] * len(POSITION_ORDER)))
    for _, row in frame.iterrows():
        lines.append(
            "| %s | %d | %.3f | %.3f | %s |"
            % (
                row["season"],
                row["players"],
                row["giniBefore"],
                row["giniAfter"],
                " | ".join(str(row[pos]) for pos in POSITION_ORDER),
            )
        )
    lines.append("")
    for note in notes:
        lines.append("- %s" % note)
    lines.append("")
    lines.append(
        "The counts above are shown as before and after. Two limits matter. First, only %d "
        "players have been collected so far against a league that must start 132 every week, so "
        "replacement level sits at roughly zero and anything derived from it, including the "
        "elite-to-replacement multiples and the waiver gap, is not yet meaningful. Second, the "
        "positional mix of the top fifty differs sharply between the two samples, which is a "
        "property of which players have been collected rather than of the scoring, so the exact "
        "size of any per-position adjustment should be settled once the full pool is in hand."
        % players
    )
    lines.append("")
    return lines


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return "%g" % value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Where shared artifacts live (weights, game logs); defaults to --output-dir",
    )
    parser.add_argument("--season", help="Season label; defaults to all seasons pooled")
    parser.add_argument(
        "--recommend",
        type=Path,
        default=Path("proposals/recommended.json"),
        help="Proposal to write up as the recommendation; pass a missing path to skip",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or args.output_dir
    weights = diagnose.load_weights(data_dir)
    totals = diagnose.attach_archetypes(
        diagnose.load_totals(data_dir, args.season), data_dir
    )
    metrics, working, contributions = diagnose.compute_metrics(totals, weights, label="current")
    profiles = diagnose.volume_report(working, contributions, weights)
    metrics["churn"] = diagnose.churn_metrics(
        diagnose.load_games(data_dir, args.season), weights
    )

    season_tag = args.season or "all seasons"
    chart_dir = args.output_dir / CHART_DIR
    chart_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = {
        "curves": "%s/value_curves.png" % CHART_DIR,
        "categories": "%s/category_decomposition.png" % CHART_DIR,
        "archetypes": "%s/archetype_mix.png" % CHART_DIR,
        "waivers": "%s/waiver_depth.png" % CHART_DIR,
    }

    recommendation = None
    reliability = None
    tradeoff = None
    if args.recommend and args.recommend.exists():
        proposal = json.loads(args.recommend.read_text(encoding="utf-8"))
        recommendation = simulate.run(data_dir, proposal, args.season)
        chart_paths["tradeoff"] = "%s/tradeoff.png" % CHART_DIR
        tradeoff = sweep.trace(data_dir, args.season, with_churn=True)
        chart_tradeoff(tradeoff, args.output_dir / chart_paths["tradeoff"])
        if not args.season:
            seasons = validate.seasons_available(data_dir)
            if len(seasons) > 1:
                checked = validate.check(data_dir, proposal, seasons)
                reliability = (checked, validate.direction_summary(checked))

    chart_value_curves(working, metrics, args.output_dir / chart_paths["curves"])
    chart_category_decomposition(working, contributions, args.output_dir / chart_paths["categories"])
    chart_archetype_mix(working, args.output_dir / chart_paths["archetypes"])
    chart_waiver_depth(metrics, args.output_dir / chart_paths["waivers"])

    report = build_report(
        metrics,
        working,
        profiles,
        weights,
        season_tag,
        chart_paths,
        recommendation,
        reliability,
        tradeoff,
    )
    report_path = args.output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print("Wrote %s and %d charts" % (report_path, len(chart_paths)))


if __name__ == "__main__":
    main()
