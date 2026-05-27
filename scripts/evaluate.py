#!/usr/bin/env python
"""Evaluate a neural checkpoint against tiny random/MCTS baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

from tinychess.ai import MCTSConfig, NeuralMCTSConfig
from tinychess.ai.evaluation import (
    MatchConfig,
    PromotionCriteria,
    evaluate_checkpoint_against_baselines,
    write_evaluation_report,
)


def main() -> None:
    args = _parse_args()
    baselines = tuple(args.baseline)
    criteria = PromotionCriteria(
        min_games_per_baseline=args.min_games_per_baseline,
        min_score_rate_vs_random=args.min_score_rate_vs_random,
        min_score_rate_vs_mcts=args.min_score_rate_vs_mcts,
        required_baselines=baselines,
    )
    report = evaluate_checkpoint_against_baselines(
        args.checkpoint,
        match_config=MatchConfig(
            games=args.games,
            max_plies=args.max_plies,
            alternate_colors=not args.no_alternate_colors,
        ),
        neural_config=NeuralMCTSConfig(
            simulations=args.neural_simulations,
            node_budget=args.neural_node_budget,
            temperature=args.neural_temperature,
            seed=args.seed,
        ),
        mcts_config=MCTSConfig(
            simulations=args.mcts_simulations,
            node_budget=args.mcts_node_budget,
            max_rollout_plies=args.mcts_rollout_plies,
            seed=args.seed,
        ),
        random_seed=args.seed,
        baselines=baselines,
        criteria=criteria,
        workers=args.workers,
    )
    if args.output is not None:
        write_evaluation_report(report, args.output)
    print(json.dumps(report, indent=2))
    promotion = report["promotion"]
    if not isinstance(promotion, dict):
        _die("internal error: malformed promotion report")
    if args.require_promotion and not promotion.get("promoted"):
        _die("checkpoint did not satisfy early promotion criteria")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint directory")
    parser.add_argument(
        "--baseline",
        action="append",
        choices=("random", "mcts"),
        default=None,
        help="Baseline to run; repeat to select multiple (default: random and mcts)",
    )
    parser.add_argument("--games", type=int, default=2, help="Games per baseline")
    parser.add_argument("--max-plies", type=int, default=40, help="Maximum plies per game")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for independent evaluation games",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic baselines/search",
    )
    parser.add_argument(
        "--neural-simulations",
        type=int,
        default=1,
        help="Neural MCTS simulations",
    )
    parser.add_argument(
        "--neural-node-budget",
        type=int,
        default=None,
        help="Optional neural node cap",
    )
    parser.add_argument(
        "--neural-temperature",
        type=float,
        default=0.0,
        help="Neural move temperature",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=1,
        help="Classical MCTS simulations",
    )
    parser.add_argument("--mcts-node-budget", type=int, default=None, help="Optional MCTS node cap")
    parser.add_argument("--mcts-rollout-plies", type=int, default=2, help="MCTS rollout cap")
    parser.add_argument(
        "--min-games-per-baseline",
        type=int,
        default=2,
        help="Minimum games required before early promotion",
    )
    parser.add_argument(
        "--min-score-rate-vs-random",
        type=float,
        default=0.5,
        help="Minimum checkpoint score rate versus random for early promotion",
    )
    parser.add_argument(
        "--min-score-rate-vs-mcts",
        type=float,
        default=0.0,
        help="Minimum checkpoint score rate versus MCTS for early promotion",
    )
    parser.add_argument("--no-alternate-colors", action="store_true", help="Keep checkpoint white")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help="Exit non-zero if early criteria are not met",
    )
    parsed = parser.parse_args()
    if parsed.workers < 1:
        parser.error(f"--workers must be at least 1, got {parsed.workers}")
    if parsed.baseline is None:
        parsed.baseline = ["random", "mcts"]
    return parsed


def _die(message: str) -> NoReturn:
    raise SystemExit(message)


if __name__ == "__main__":
    main()
