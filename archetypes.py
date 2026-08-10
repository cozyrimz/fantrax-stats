#!/usr/bin/env python3
"""Derive player archetypes from the stat profiles themselves.

Fantrax only tags players as G, D, M or F, which hides the distinction between
a holding midfielder and an attacking one. Clustering per-game category rates
within each position recovers those real playing styles from the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import fantrax_data as fx

# A category must be recorded by at least this share of a position's players to
# be used as a clustering feature, which keeps rare events out of the distance.
MIN_OCCURRENCE_RATE = 0.25

# Keepers all play the same role, so clustering 13 of them only fits noise.
CLUSTER_POSITIONS = ["D", "M", "F"]

# Minimum games for a player-season to have a stable enough rate profile.
MIN_GAMES = 10

CANDIDATE_K = range(2, 7)
RANDOM_STATE = 7

# Category groups used to name each cluster from what it actually does well.
STYLE_GROUPS: Dict[str, List[str]] = {
    "finishing": ["G", "SOT", "BCM", "PKG", "FKG"],
    "creation": ["A", "A2", "KP", "BCC", "AC", "CF"],
    "dribbling": ["CoS", "FS", "DIS"],
    "distribution": ["AP", "LBA"],
    "defending": ["TkW", "Int", "IntB", "CLR", "BS", "BC", "CLO", "TLM"],
    "duels": ["AER", "AERL", "DW", "DL", "BR"],
}

GROUP_NAMES = {
    "finishing": "Finisher",
    "creation": "Creator",
    "dribbling": "Dribbler",
    "distribution": "Distributor",
    "defending": "Stopper",
    "duels": "Duellist",
}


def load_totals(output_dir: Path) -> pd.DataFrame:
    paths = sorted(output_dir.glob("season_totals_*.csv"))
    if not paths:
        raise SystemExit("No season_totals_*.csv in %s — run build_dataset.py first" % output_dir)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def feature_categories(totals: pd.DataFrame, position: str, categories: List[str]) -> List[str]:
    subset = totals[totals["position"] == position]
    keep = []
    for category in categories:
        if category not in subset.columns:
            continue
        players_with = (subset[category] > 0).mean()
        if players_with >= MIN_OCCURRENCE_RATE:
            keep.append(category)
    return keep


def choose_k(features: np.ndarray) -> int:
    best_k, best_score = 2, -1.0
    for k in CANDIDATE_K:
        if k >= len(features):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(features)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(features, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def name_cluster(profile: pd.Series, used: Dict[str, int]) -> str:
    """Name a cluster from the style groups it over-indexes on."""
    scores = {}
    for group, members in STYLE_GROUPS.items():
        present = [c for c in members if c in profile.index]
        if present:
            # Score on the group's strongest categories only. Averaging every
            # member dilutes a real signal, since a centre back rates highly on
            # blocks and clearances while barely tackling.
            top = profile[present].sort_values(ascending=False).head(3)
            scores[group] = float(top.mean())

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    primary = ranked[0][0] if ranked else "balanced"
    secondary = ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0.15 else None

    label = GROUP_NAMES.get(primary, "All-rounder")
    if secondary and ranked[0][1] - ranked[1][1] < 0.35:
        label = "%s-%s" % (label, GROUP_NAMES.get(secondary, "").lower())
    if all(value < 0.1 for value in scores.values()):
        label = "All-rounder"

    count = used.get(label, 0)
    used[label] = count + 1
    return label if count == 0 else "%s %d" % (label, count + 1)


ASSIGNMENT_COLUMNS = ["playerId", "season", "position", "archetype", "archetypeFull", "cluster", "usage"]


def cluster_position(totals: pd.DataFrame, position: str, categories: List[str]):
    """Cluster one position, returning (assignments, cluster profiles, cluster names)."""
    subset = totals[(totals["position"] == position) & (totals["GP"] >= MIN_GAMES)].copy()
    if position not in CLUSTER_POSITIONS:
        subset["archetype"] = "Keeper"
        subset["archetypeFull"] = position + " Keeper"
        subset["cluster"] = 0
        subset["usage"] = 1.0
        return subset[ASSIGNMENT_COLUMNS], None, {}
    if len(subset) < 8:
        subset["archetype"] = "Unclassified"
        subset["archetypeFull"] = position + " Unclassified"
        subset["cluster"] = -1
        subset["usage"] = 1.0
        return subset[ASSIGNMENT_COLUMNS], None, {}

    features = feature_categories(totals, position, categories)
    rates = subset[features].div(subset["GP"], axis=0)

    # Index each category against the positional average so categories on wildly
    # different scales are comparable, then divide by the player's own average
    # index. That strips overall involvement out of the profile, so clustering
    # separates playing style rather than playing time. Involvement is kept as a
    # separate usage measure.
    index = rates.div(rates.mean(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    usage = index.mean(axis=1)
    style = index.div(usage.replace(0, np.nan), axis=0).fillna(0.0)
    subset["usage"] = usage.round(3)

    scaled = StandardScaler().fit_transform(style.to_numpy(dtype=float))

    k = choose_k(scaled)
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    subset["cluster"] = model.fit_predict(scaled)

    scaled_frame = pd.DataFrame(scaled, columns=features, index=subset.index)
    scaled_frame["cluster"] = subset["cluster"]
    profiles = scaled_frame.groupby("cluster").mean()

    used: Dict[str, int] = {}
    ordering = profiles.mean(axis=1).sort_values(ascending=False).index
    names = {}
    for cluster in ordering:
        names[cluster] = name_cluster(profiles.loc[cluster], used)

    subset["archetype"] = subset["cluster"].map(names)
    subset["archetypeFull"] = position + " " + subset["archetype"]
    return subset[ASSIGNMENT_COLUMNS], profiles, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    totals = load_totals(args.output_dir)
    categories_meta = json.loads((args.output_dir / "categories.json").read_text(encoding="utf-8"))
    categories = sorted(categories_meta["names"].keys())

    assignments = []
    summary: Dict[str, object] = {}

    for position in fx.POSITIONS:
        frame, profiles, names = cluster_position(totals, position, categories)
        assignments.append(frame)

        if profiles is not None:
            counts = frame["archetype"].value_counts().to_dict()
            print("%s: k=%d %s" % (position, len(names), counts))
            summary[position] = {
                "clusters": {
                    names[cluster]: {
                        "players": int((frame["cluster"] == cluster).sum()),
                        "topCategories": list(
                            profiles.loc[cluster].sort_values(ascending=False).head(6).round(2).items()
                        ),
                    }
                    for cluster in profiles.index
                }
            }

    combined = pd.concat(assignments, ignore_index=True)
    combined.to_csv(args.output_dir / "archetypes.csv", index=False)
    (args.output_dir / "archetype_profiles.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Wrote %s (%d player-seasons)" % (args.output_dir / "archetypes.csv", len(combined)))


if __name__ == "__main__":
    main()
