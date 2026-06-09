#!/usr/bin/env python
"""Evaluate a neural checkpoint against tiny random/MCTS baselines or another checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

from tinychess.ai import MCTSConfig, NeuralMCTSConfig
from tinychess.ai.evaluation import (
    MatchConfig,
    PromotionCriteria,
    evaluate_checkpoint_against_baselines,
    evaluate_checkpoints_head_to_head,
    write_evaluation_report,
)


def main() -> None:
    args = _parse_args()
    match_config = MatchConfig(
        games=args.games,
        max_plies=args.max_plies,
        alternate_colors=not args.no_alternate_colors,
    )
    neural_config = NeuralMCTSConfig(
        simulations=args.neural_simulations,
        node_budget=args.neural_node_budget,
        temperature=args.neural_temperature,
        seed=args.seed,
    )
    if args.opponent_checkpoint is not None:
        report = evaluate_checkpoints_head_to_head(
            args.checkpoint,
            args.opponent_checkpoint,
            match_config=match_config,
            neural_config=neural_config,
            opponent_neural_config=NeuralMCTSConfig(
                simulations=(
                    args.neural_simulations
                    if args.opponent_neural_simulations is None
                    else args.opponent_neural_simulations
                ),
                node_budget=(
                    args.neural_node_budget
                    if args.opponent_neural_node_budget is None
                    else args.opponent_neural_node_budget
                ),
                temperature=(
                    args.neural_temperature
                    if args.opponent_neural_temperature is None
                    else args.opponent_neural_temperature
                ),
                seed=args.seed if args.opponent_seed is None else args.opponent_seed,
            ),
            workers=args.workers,
        )
    else:
        baselines = tuple(args.baseline)
        criteria = PromotionCriteria(
            min_games_per_baseline=args.min_games_per_baseline,
            min_score_rate_vs_random=args.min_score_rate_vs_random,
            min_score_rate_vs_mcts=args.min_score_rate_vs_mcts,
            required_baselines=baselines,
        )
        report = evaluate_checkpoint_against_baselines(
            args.checkpoint,
            match_config=match_config,
            neural_config=neural_config,
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
    if args.opponent_checkpoint is not None:
        _print_head_to_head_summary(report)
    else:
        promotion = report["promotion"]
        if not isinstance(promotion, dict):
            _die("internal error: malformed promotion report")
        if args.require_promotion and not promotion.get("promoted"):
            _die("checkpoint did not satisfy early promotion criteria")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint directory")
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint directory for neural-vs-neural evaluation",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        choices=("random", "mcts"),
        default=None,
        help="Baseline to run; repeat to select multiple (default: random and mcts)",
    )
    parser.add_argument("--games", type=int, default=2, help="Games per baseline or match")
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
        "--opponent-seed",
        type=int,
        default=None,
        help="Opponent neural seed (default: --seed)",
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
        "--opponent-neural-simulations",
        type=int,
        default=None,
        help="Opponent neural MCTS simulations (default: --neural-simulations)",
    )
    parser.add_argument(
        "--opponent-neural-node-budget",
        type=int,
        default=None,
        help="Optional opponent neural node cap (default: --neural-node-budget)",
    )
    parser.add_argument(
        "--opponent-neural-temperature",
        type=float,
        default=None,
        help="Opponent neural move temperature (default: --neural-temperature)",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=None,
        help="Classical MCTS simulations",
    )
    parser.add_argument("--mcts-node-budget", type=int, default=None, help="Optional MCTS node cap")
    parser.add_argument(
        "--mcts-rollout-plies",
        type=int,
        default=None,
        help="Classical MCTS rollout cap; default 0 uses static leaf evaluation",
    )
    parser.add_argument(
        "--min-games-per-baseline",
        type=int,
        default=None,
        help="Minimum games required before early promotion",
    )
    parser.add_argument(
        "--min-score-rate-vs-random",
        type=float,
        default=None,
        help="Minimum checkpoint score rate versus random for early promotion",
    )
    parser.add_argument(
        "--min-score-rate-vs-mcts",
        type=float,
        default=None,
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
    if parsed.opponent_checkpoint is not None:
        if parsed.baseline is not None:
            parser.error("--baseline cannot be used with --opponent-checkpoint")
        for attr, flag in (
            ("mcts_simulations", "--mcts-simulations"),
            ("mcts_node_budget", "--mcts-node-budget"),
            ("mcts_rollout_plies", "--mcts-rollout-plies"),
            ("min_games_per_baseline", "--min-games-per-baseline"),
            ("min_score_rate_vs_random", "--min-score-rate-vs-random"),
            ("min_score_rate_vs_mcts", "--min-score-rate-vs-mcts"),
        ):
            if getattr(parsed, attr) is not None:
                parser.error(f"{flag} cannot be used with --opponent-checkpoint")
        if parsed.require_promotion:
            parser.error("--require-promotion cannot be used with --opponent-checkpoint")
    if parsed.baseline is None:
        parsed.baseline = ["random", "mcts"]
    if parsed.mcts_simulations is None:
        parsed.mcts_simulations = 1
    if parsed.mcts_rollout_plies is None:
        parsed.mcts_rollout_plies = 0
    if parsed.min_games_per_baseline is None:
        parsed.min_games_per_baseline = 2
    if parsed.min_score_rate_vs_random is None:
        parsed.min_score_rate_vs_random = 0.5
    if parsed.min_score_rate_vs_mcts is None:
        parsed.min_score_rate_vs_mcts = 0.0
    return parsed


def _print_head_to_head_summary(report: Mapping[str, object]) -> None:
    match = report.get("match")
    if not isinstance(match, Mapping):
        _die("internal error: malformed neural-vs-neural match report")

    games = match.get("games")
    player_a_score = match.get("player_a_score")
    player_b_score = match.get("player_b_score")
    player_a_score_rate = match.get("player_a_score_rate")
    player_a_wins = match.get("player_a_wins")
    player_b_wins = match.get("player_b_wins")
    draws = match.get("draws")
    if not (
        isinstance(games, int)
        and isinstance(player_a_score, int | float)
        and isinstance(player_b_score, int | float)
        and isinstance(player_a_score_rate, int | float)
        and isinstance(player_a_wins, int)
        and isinstance(player_b_wins, int)
        and isinstance(draws, int)
    ):
        _die("internal error: malformed neural-vs-neural summary fields")

    print(
        "Neural-vs-neural summary: "
        f"checkpoint {player_a_score:g}-{player_b_score:g} opponent_checkpoint "
        f"over {games} game{'s' if games != 1 else ''} "
        f"({float(player_a_score_rate):.1%} score rate); "
        f"wins {player_a_wins}-{player_b_wins}, draws {draws}",
        file=sys.stderr,
    )


def _die(message: str) -> NoReturn:
    raise SystemExit(message)


if __name__ == "__main__":
    main()
