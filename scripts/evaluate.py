#!/usr/bin/env python
"""Evaluate a neural checkpoint against tiny random/MCTS baselines or another checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

from vibechess.ai import MCTSConfig, NeuralMCTSConfig
from vibechess.ai.evaluation import (
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
    neural_config = _build_neural_config(
        simulations=args.neural_simulations,
        node_budget=args.neural_node_budget,
        temperature=args.neural_temperature,
        seed=args.seed,
        collection_batch_size=args.neural_collection_batch_size,
        virtual_loss=args.neural_virtual_loss,
        reuse_simulation_budget=args.reuse_simulation_budget,
        min_reuse_simulations=args.min_reuse_simulations,
    )
    if args.opponent_checkpoint is not None:
        report = evaluate_checkpoints_head_to_head(
            args.checkpoint,
            args.opponent_checkpoint,
            match_config=match_config,
            neural_config=neural_config,
            opponent_neural_config=_build_neural_config(
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
                collection_batch_size=(
                    args.neural_collection_batch_size
                    if args.opponent_neural_collection_batch_size is None
                    else args.opponent_neural_collection_batch_size
                ),
                virtual_loss=(
                    args.neural_virtual_loss
                    if args.opponent_neural_virtual_loss is None
                    else args.opponent_neural_virtual_loss
                ),
                reuse_simulation_budget=(
                    args.reuse_simulation_budget
                    if args.opponent_reuse_simulation_budget is None
                    else args.opponent_reuse_simulation_budget
                ),
                min_reuse_simulations=(
                    args.min_reuse_simulations
                    if args.opponent_min_reuse_simulations is None
                    else args.opponent_min_reuse_simulations
                ),
            ),
            workers=args.workers,
            batch_size=args.batch_size,
            active_games=args.active_games,
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
            batch_size=args.batch_size,
            active_games=args.active_games,
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


def _build_neural_config(
    *,
    simulations: int,
    node_budget: int | None,
    temperature: float,
    seed: int | None,
    collection_batch_size: int,
    virtual_loss: int,
    reuse_simulation_budget: bool,
    min_reuse_simulations: int,
) -> NeuralMCTSConfig:
    return NeuralMCTSConfig(
        simulations=simulations,
        node_budget=node_budget,
        temperature=temperature,
        seed=seed,
        collection_batch_size=collection_batch_size,
        virtual_loss=virtual_loss,
        reuse_simulation_budget=reuse_simulation_budget,
        min_reuse_simulations=min_reuse_simulations,
    )


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
        "--batch-size",
        type=int,
        default=1,
        help=(
            "cross-game neural inference batch size for independent evaluation games; "
            "default 1 preserves serial play"
        ),
    )
    parser.add_argument(
        "--active-games",
        type=int,
        default=None,
        help="maximum in-process active evaluation games; defaults to --batch-size",
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
        default=1.0,
        help="Neural move temperature; default 1.0 lets per-game seeds sample moves",
    )
    parser.add_argument(
        "--neural-collection-batch-size",
        type=int,
        default=1,
        help="Within-search neural leaf collection batch size; default 1 is serial",
    )
    parser.add_argument(
        "--neural-virtual-loss",
        type=int,
        default=1,
        help="Virtual loss applied while collecting batched neural leaves",
    )
    parser.add_argument(
        "--reuse-simulation-budget",
        action="store_true",
        help="Reuse adopted neural root visits instead of always running all simulations",
    )
    parser.add_argument(
        "--min-reuse-simulations",
        type=int,
        default=0,
        help="Minimum fresh simulations after neural root reuse",
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
        "--opponent-neural-collection-batch-size",
        type=int,
        default=None,
        help="Opponent neural leaf collection batch size (default: --neural-collection-batch-size)",
    )
    parser.add_argument(
        "--opponent-neural-virtual-loss",
        type=int,
        default=None,
        help="Opponent neural virtual loss (default: --neural-virtual-loss)",
    )
    parser.add_argument(
        "--opponent-reuse-simulation-budget",
        action="store_true",
        default=None,
        help="Enable visit-budget-aware root reuse for the opponent checkpoint",
    )
    parser.add_argument(
        "--opponent-min-reuse-simulations",
        type=int,
        default=None,
        help="Opponent fresh simulation floor after root reuse (default: --min-reuse-simulations)",
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
    if parsed.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if parsed.active_games is not None and parsed.active_games < 1:
        parser.error("--active-games must be at least 1")
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
    if parsed.neural_collection_batch_size < 1:
        parser.error("--neural-collection-batch-size must be at least 1")
    if parsed.neural_virtual_loss < 0:
        parser.error("--neural-virtual-loss must be non-negative")
    if parsed.min_reuse_simulations < 0:
        parser.error("--min-reuse-simulations must be non-negative")
    if (
        parsed.reuse_simulation_budget
        and parsed.min_reuse_simulations > parsed.neural_simulations
    ):
        parser.error(
            "--min-reuse-simulations must be no greater than --neural-simulations "
            "when --reuse-simulation-budget is enabled"
        )
    opponent_simulations = (
        parsed.neural_simulations
        if parsed.opponent_neural_simulations is None
        else parsed.opponent_neural_simulations
    )
    opponent_collection_batch_size = (
        parsed.neural_collection_batch_size
        if parsed.opponent_neural_collection_batch_size is None
        else parsed.opponent_neural_collection_batch_size
    )
    opponent_virtual_loss = (
        parsed.neural_virtual_loss
        if parsed.opponent_neural_virtual_loss is None
        else parsed.opponent_neural_virtual_loss
    )
    opponent_reuse_simulation_budget = (
        parsed.reuse_simulation_budget
        if parsed.opponent_reuse_simulation_budget is None
        else parsed.opponent_reuse_simulation_budget
    )
    opponent_min_reuse_simulations = (
        parsed.min_reuse_simulations
        if parsed.opponent_min_reuse_simulations is None
        else parsed.opponent_min_reuse_simulations
    )
    if opponent_collection_batch_size < 1:
        parser.error("--opponent-neural-collection-batch-size must be at least 1")
    if opponent_virtual_loss < 0:
        parser.error("--opponent-neural-virtual-loss must be non-negative")
    if opponent_min_reuse_simulations < 0:
        parser.error("--opponent-min-reuse-simulations must be non-negative")
    if (
        opponent_reuse_simulation_budget
        and opponent_min_reuse_simulations > opponent_simulations
    ):
        parser.error(
            "--opponent-min-reuse-simulations must be no greater than opponent "
            "neural simulations when opponent root reuse is enabled"
        )
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
