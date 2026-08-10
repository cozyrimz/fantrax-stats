#!/usr/bin/env python3
"""Search weight space for the scoring that spreads value most evenly.

The objective combines the three imbalances that make waivers dead: a top-heavy
value curve, a top 50 that does not match the positions the league has to field,
and a top 50 dominated by one playing style. Weights move by a bounded multiple
of their current value and never change sign, so the result stays recognisable
as the same scoring system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import diagnose
import league_config as lc

# Only categories worth at least this share of a position's points are moved.
# Nudging a category nobody scores in cannot change the distribution.
LEVER_MIN_SHARE = 0.01

# How far any weight may move from its current value.
BOUND_LOW = 0.25
BOUND_HIGH = 4.0

# Goals and assists must stay at least as valuable as they are today.
PROTECTED = {"G", "A"}
PROTECTED_BOUNDS = (1.0, 3.0)

RANDOM_SEED = 11


class Evaluator:
    """Scores candidate weight sets against the evenness objective."""

    def __init__(self, totals: pd.DataFrame, weights: Dict[str, Dict[str, float]], objective: Dict[str, float]):
        self.totals = totals.reset_index(drop=True)
        self.base_weights = weights
        self.objective = objective

        self.positions = [p for p in lc.MIN_ACTIVE if (self.totals["position"] == p).any()]
        self.masks = {p: (self.totals["position"] == p).to_numpy() for p in self.positions}
        self.categories = {p: [c for c in weights.get(p, {})if c in self.totals.columns] for p in self.positions}
        self.matrices = {
            p: self.totals.loc[self.masks[p], self.categories[p]].to_numpy(dtype=float)
            for p in self.positions
        }
        self.archetypes = (
            self.totals["archetypeFull"].to_numpy()
            if "archetypeFull" in self.totals.columns
            else np.array(["all"] * len(self.totals))
        )

        demand_total = sum(lc.TEAMS * lc.BASELINE_STARTERS[p] for p in lc.BASELINE_STARTERS)
        self.demand_share = {
            p: lc.TEAMS * lc.BASELINE_STARTERS[p] / demand_total for p in lc.BASELINE_STARTERS
        }

        self.levers = self._choose_levers()

    def _choose_levers(self) -> List[Tuple[str, str]]:
        levers = []
        for position in self.positions:
            matrix = self.matrices[position]
            vector = np.array([self.base_weights[position][c] for c in self.categories[position]])
            contributions = np.abs(matrix * vector).sum(axis=0)
            total = contributions.sum()
            if total <= 0:
                continue
            for i, category in enumerate(self.categories[position]):
                if contributions[i] / total >= LEVER_MIN_SHARE:
                    levers.append((position, category))
        return levers

    def bounds(self, lever: Tuple[str, str]) -> Tuple[float, float]:
        return PROTECTED_BOUNDS if lever[1] in PROTECTED else (BOUND_LOW, BOUND_HIGH)

    def weights_from(self, multipliers: Dict[Tuple[str, str], float]) -> Dict[str, Dict[str, float]]:
        weights = {p: dict(self.base_weights[p]) for p in self.base_weights}
        for (position, category), factor in multipliers.items():
            weights[position][category] = round(self.base_weights[position][category] * factor, 4)
        return weights

    def points(self, weights: Dict[str, Dict[str, float]]) -> np.ndarray:
        result = np.zeros(len(self.totals))
        for position in self.positions:
            vector = np.array([weights[position][c] for c in self.categories[position]])
            result[self.masks[position]] = self.matrices[position] @ vector
        return result

    def score(self, multipliers: Dict[Tuple[str, str], float]) -> Tuple[float, Dict[str, float]]:
        weights = self.weights_from(multipliers)
        points = self.points(weights)
        return self.score_points(points)

    def score_points(self, points: np.ndarray) -> Tuple[float, Dict[str, float]]:
        frame = self.totals[["position"]].copy()
        series = pd.Series(points, index=frame.index)

        demand = diagnose.allocate_starters(self.totals, series)
        levels = diagnose.replacement_levels(self.totals, series, demand)
        replacement = frame["position"].map(levels).to_numpy(dtype=float)
        vor = points - replacement

        gini_value = diagnose.gini(vor)

        order = np.argsort(-points)
        top = order[: min(diagnose.TOP_N, len(order))]
        top_positions = frame["position"].to_numpy()[top]
        mix_deviation = sum(
            abs((top_positions == p).mean() - self.demand_share.get(p, 0.0)) for p in self.positions
        )

        top_archetypes = self.archetypes[top]
        counts = pd.Series(top_archetypes).value_counts().to_numpy(dtype=float)
        archetype_gini = diagnose.gini(counts) if counts.size > 1 else 0.0

        parts = {
            "giniOfValue": gini_value,
            "mixDeviation": mix_deviation,
            "archetypeGini": archetype_gini,
        }
        total = (
            self.objective["gini"] * gini_value
            + self.objective["mix"] * mix_deviation
            + self.objective["archetype"] * archetype_gini
        )
        return total, parts


def hill_climb(
    evaluator: Evaluator,
    iterations: int,
    restarts: int,
    verbose: bool = True,
) -> Tuple[Dict[Tuple[str, str], float], float, Dict[str, float]]:
    rng = np.random.default_rng(RANDOM_SEED)
    levers = evaluator.levers
    if not levers:
        raise SystemExit("No categories carry enough points to be worth adjusting")

    best_overall = None

    for restart in range(restarts):
        if restart == 0:
            multipliers = {lever: 1.0 for lever in levers}
        else:
            multipliers = {
                lever: float(np.clip(rng.normal(1.0, 0.3), *evaluator.bounds(lever)))
                for lever in levers
            }

        current, parts = evaluator.score(multipliers)
        step = 0.35

        for iteration in range(iterations):
            lever = levers[rng.integers(len(levers))]
            low, high = evaluator.bounds(lever)
            direction = 1.0 if rng.random() < 0.5 else -1.0
            candidate = dict(multipliers)
            candidate[lever] = float(np.clip(multipliers[lever] * (1 + direction * step), low, high))
            if candidate[lever] == multipliers[lever]:
                continue

            value, candidate_parts = evaluator.score(candidate)
            if value < current:
                multipliers, current, parts = candidate, value, candidate_parts

            if iteration and iteration % max(1, iterations // 8) == 0:
                step = max(0.05, step * 0.75)

        if best_overall is None or current < best_overall[1]:
            best_overall = (multipliers, current, parts)
        if verbose:
            print("  restart %d: objective %.4f" % (restart + 1, current))

    return best_overall


def to_proposal(
    evaluator: Evaluator, multipliers: Dict[Tuple[str, str], float], name: str, description: str
) -> Dict:
    multiply: Dict[str, Dict[str, float]] = {}
    for (position, category), factor in sorted(multipliers.items()):
        if abs(factor - 1.0) < 0.02:
            continue
        multiply.setdefault(position, {})[category] = round(float(factor), 3)
    return {"name": name, "description": description, "multiply": multiply, "set": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--season", help="Season label; defaults to all seasons pooled")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--gini-weight", type=float, default=1.0)
    parser.add_argument("--mix-weight", type=float, default=0.5)
    parser.add_argument("--archetype-weight", type=float, default=0.25)
    parser.add_argument("--name", default="optimized")
    args = parser.parse_args()

    weights = diagnose.load_weights(args.output_dir)
    totals = diagnose.attach_archetypes(
        diagnose.load_totals(args.output_dir, args.season), args.output_dir
    )

    objective = {
        "gini": args.gini_weight,
        "mix": args.mix_weight,
        "archetype": args.archetype_weight,
    }
    evaluator = Evaluator(totals, weights, objective)
    print("Optimising %d category weights across %d positions" % (len(evaluator.levers), len(evaluator.positions)))

    baseline, baseline_parts = evaluator.score({lever: 1.0 for lever in evaluator.levers})
    print(
        "Current objective %.4f (gini %.3f, mix deviation %.3f, archetype gini %.3f)"
        % (
            baseline,
            baseline_parts["giniOfValue"],
            baseline_parts["mixDeviation"],
            baseline_parts["archetypeGini"],
        )
    )

    multipliers, value, parts = hill_climb(evaluator, args.iterations, args.restarts)
    print(
        "Optimised objective %.4f (gini %.3f, mix deviation %.3f, archetype gini %.3f)"
        % (value, parts["giniOfValue"], parts["mixDeviation"], parts["archetypeGini"])
    )

    proposal = to_proposal(
        evaluator,
        multipliers,
        args.name,
        "Weights found by optimising the evenness objective (gini %.2f, mix %.2f, archetype %.2f)."
        % (args.gini_weight, args.mix_weight, args.archetype_weight),
    )
    path = Path("proposals") / ("%s.json" % args.name)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(proposal, indent=2, sort_keys=True), encoding="utf-8")

    print("")
    print("Largest moves:")
    ordered = sorted(multipliers.items(), key=lambda kv: -abs(kv[1] - 1.0))
    for (position, category), factor in ordered[:20]:
        before = weights[position][category]
        print("  %s %-5s %+6.0f%%  %g -> %g" % (position, category, 100 * (factor - 1), before, round(before * factor, 3)))

    print("")
    print("Wrote %s — simulate it with: python3 simulate.py --proposal %s" % (path, path))


if __name__ == "__main__":
    main()
