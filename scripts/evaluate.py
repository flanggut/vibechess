#!/usr/bin/env python
"""Evaluate a neural checkpoint against tiny random/MCTS baselines or another checkpoint."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

from _progress import AnsiProgressRenderer as _AnsiProgressRenderer
from _progress import ProgressRenderState as _ProgressRenderState
from _progress import ProgressStatus as _ProgressStatus
from _progress import WorkerProgressState as _WorkerProgressState

from vibechess.ai import MCTSConfig, NeuralMCTSConfig
from vibechess.ai.evaluation import (
    DEFAULT_EVALUATION_OPENING_COUNT,
    DEFAULT_EVALUATION_OPENING_PLIES,
    EvaluationProgress,
    MatchConfig,
    OpeningConfig,
    PromotionCriteria,
    evaluate_checkpoint_against_baselines,
    evaluate_checkpoints_head_to_head,
    write_evaluation_report,
)

_PROGRESS_REFRESH_SECONDS = 1.0


@dataclass(slots=True)
class _ProgressReporter:
    enabled: bool
    total_games: int
    initial_workers: tuple[_WorkerProgressState, ...] | None = None
    _renderer: _AnsiProgressRenderer = field(init=False)
    _workers_by_id: dict[int, _WorkerProgressState] = field(init=False)
    _status: _ProgressStatus = field(default="pending", init=False)
    _start_monotonic: float | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _refresh_stop: threading.Event = field(default_factory=threading.Event, init=False)
    _refresh_thread: threading.Thread | None = field(default=None, init=False)
    _live_score_message: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._renderer = _AnsiProgressRenderer(
            enabled=self.enabled,
            total_games=self.total_games,
            label="evaluation",
            unit_label="plies",
        )
        workers = self.initial_workers or (
            _WorkerProgressState(
                worker_id=0,
                start_game=0,
                total_games=self.total_games,
                status="pending",
            ),
        )
        self._workers_by_id = {worker.worker_id: worker for worker in workers}

    def start(self, args: argparse.Namespace) -> None:
        self._status = "running"
        self._workers_by_id = {
            worker_id: _WorkerProgressState(
                worker_id=worker.worker_id,
                start_game=worker.start_game,
                total_games=worker.total_games,
                status="running",
            )
            for worker_id, worker in self._workers_by_id.items()
        }
        self._start_monotonic = time.monotonic()
        self._start_refresh()
        mode = (
            "neural_vs_neural" if args.opponent_checkpoint is not None else "baselines"
        )
        self._write(
            " ".join(
                [
                    "starting",
                    f"mode={mode}",
                    f"games={self.total_games}",
                    f"opening_count={args.opening_count}",
                    f"max_plies={args.max_plies}",
                    f"neural_simulations={args.neural_simulations}",
                    f"workers={len(self._workers_by_id)}/{args.workers}",
                    f"batch_size={args.batch_size}",
                    f"checkpoint={args.checkpoint}",
                ]
            )
        )

    def game_completed(self, progress: EvaluationProgress) -> None:
        total_games = progress.worker_games or progress.total_games
        if progress.worker_games_completed is not None:
            games_completed = progress.worker_games_completed
        elif len(self._workers_by_id) == 1:
            games_completed = progress.games_completed
        elif progress.worker_games is not None:
            games_completed = total_games
        else:
            games_completed = min(progress.games_completed, total_games)
        plies = (
            progress.worker_plies
            if progress.worker_plies is not None
            else progress.completed_plies
        )
        self._workers_by_id[progress.worker_id] = _WorkerProgressState(
            worker_id=progress.worker_id,
            start_game=progress.worker_start_game,
            total_games=total_games,
            games_completed=games_completed,
            samples=plies,
            plies=plies,
            status="completed" if games_completed >= total_games else "running",
        )
        if progress.games_completed >= progress.total_games:
            self._status = "completed"
        fields = [
            f"completed={progress.games_completed}/{progress.total_games}",
            f"game_index={progress.game_index}",
            f"plies={progress.completed_plies}",
        ]
        if progress.baseline is not None:
            fields.insert(1, f"baseline={progress.baseline}")
        if progress.baseline == "opponent_checkpoint":
            self._live_score_message = " ".join(
                [
                    "evaluation: score",
                    f"checkpoint={progress.player_a_score:g}",
                    f"opponent_checkpoint={progress.player_b_score:g}",
                    f"completed={progress.games_completed}/{progress.total_games}",
                ]
            )
        self._write(" ".join(fields))

    def saving(self, output: Path) -> None:
        self._status = "saving"
        self._write(f"saving output={output}")

    def done(self) -> None:
        self._status = "done"
        completed_games = sum(
            worker.games_completed for worker in self._workers_by_id.values()
        )
        completed_plies = sum(worker.plies for worker in self._workers_by_id.values())
        self._write(
            f"done games={completed_games} plies={completed_plies}",
            finish=True,
        )

    def _render(self, message: str | None = None, *, finish: bool = False) -> None:
        with self._lock:
            snapshot = _ProgressRenderState(
                total_games=self.total_games,
                workers=tuple(
                    self._workers_by_id[index] for index in sorted(self._workers_by_id)
                ),
                status=self._status,
                message=message,
                detail_lines=(
                    () if self._live_score_message is None else (self._live_score_message,)
                ),
                elapsed_seconds=self._elapsed_seconds(),
            )
            if finish:
                self._renderer.finish(snapshot)
            else:
                self._renderer.render(snapshot)

    def _elapsed_seconds(self) -> float:
        if self._start_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self._start_monotonic)

    def _start_refresh(self) -> None:
        if not self.enabled or self._refresh_thread is not None:
            return
        self._refresh_stop.clear()
        thread = threading.Thread(
            target=self._refresh_loop,
            name="evaluation-progress-refresh",
            daemon=True,
        )
        self._refresh_thread = thread
        thread.start()

    def _refresh_loop(self) -> None:
        while not self._refresh_stop.wait(_PROGRESS_REFRESH_SECONDS):
            self._render()

    def _stop_refresh(self) -> None:
        self._refresh_stop.set()
        thread = self._refresh_thread
        if thread is not None:
            thread.join(timeout=_PROGRESS_REFRESH_SECONDS + 1.0)
            self._refresh_thread = None

    def _write(self, message: str, *, finish: bool = False) -> None:
        legacy_message = (
            message if message.startswith("evaluation: ") else f"evaluation: {message}"
        )
        self._render(legacy_message, finish=finish)

    def cleanup(self) -> None:
        self._stop_refresh()
        self._renderer.cleanup()


def _initial_progress_workers(
    args: argparse.Namespace,
    *,
    total_games: int,
) -> tuple[_WorkerProgressState, ...]:
    if total_games <= 0:
        return ()
    effective_workers = min(int(args.workers), total_games)
    if effective_workers <= 1:
        return (
            _WorkerProgressState(
                worker_id=0,
                start_game=0,
                total_games=total_games,
                status="pending",
            ),
        )
    if args.opponent_checkpoint is not None:
        return _progress_worker_states(
            total_games=total_games,
            chunk_size=_ceil_div(total_games, effective_workers),
        )

    games_per_baseline = int(args.opening_count) * 2
    chunk_size = _ceil_div(total_games, effective_workers)
    workers: list[_WorkerProgressState] = []
    worker_id = 0
    for baseline_index, _baseline in enumerate(tuple(args.baseline)):
        baseline_offset = baseline_index * games_per_baseline
        for start_game in range(0, games_per_baseline, chunk_size):
            games = min(chunk_size, games_per_baseline - start_game)
            workers.append(
                _WorkerProgressState(
                    worker_id=worker_id,
                    start_game=baseline_offset + start_game,
                    total_games=games,
                    status="pending",
                )
            )
            worker_id += 1
    return tuple(workers)


def _progress_worker_states(
    *,
    total_games: int,
    chunk_size: int,
) -> tuple[_WorkerProgressState, ...]:
    return tuple(
        _WorkerProgressState(
            worker_id=worker_id,
            start_game=start_game,
            total_games=min(chunk_size, total_games - start_game),
            status="pending",
        )
        for worker_id, start_game in enumerate(range(0, total_games, chunk_size))
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    return max(1, -(-numerator // denominator))


def main() -> None:
    args = _parse_args()
    match_config = MatchConfig(
        games=args.opening_count * 2,
        max_plies=args.max_plies,
    )
    opening_config = OpeningConfig(
        count=args.opening_count,
        plies=args.opening_plies,
        seed=args.opening_seed,
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
    total_games = match_config.games
    if args.opponent_checkpoint is None:
        total_games *= len(tuple(args.baseline))
    progress_reporter = _ProgressReporter(
        enabled=_progress_enabled(args.progress),
        total_games=total_games,
        initial_workers=_initial_progress_workers(args, total_games=total_games),
    )
    report: dict[str, object]
    try:
        progress_reporter.start(args)
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
                    seed=(
                        args.seed if args.opponent_seed is None else args.opponent_seed
                    ),
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
                opening_config=opening_config,
                progress=progress_reporter.game_completed,
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
                opening_config=opening_config,
                progress=progress_reporter.game_completed,
            )
        if args.output is not None:
            progress_reporter.saving(args.output)
            write_evaluation_report(report, args.output)
        progress_reporter.done()
    finally:
        progress_reporter.cleanup()
    _print_report_summary(report)
    if args.opponent_checkpoint is None:
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
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="Checkpoint directory"
    )
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
    parser.add_argument(
        "--opening-count",
        type=int,
        default=DEFAULT_EVALUATION_OPENING_COUNT,
        help=(
            "Number of unique seeded openings to evaluate; each opening is played "
            "twice with colors swapped"
        ),
    )
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=DEFAULT_EVALUATION_OPENING_PLIES,
        help="Number of random legal plies used to generate each unique opening",
    )
    parser.add_argument(
        "--opening-seed",
        type=int,
        default=None,
        help="Opening generation seed (default: --seed)",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=40,
        help="Maximum played plies after opening",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for independent evaluation games",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "cross-game neural inference batch size for independent evaluation games; "
            "default 8 matches self-play central inference batching"
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
        default=0.0,
        help="Neural move temperature; default 0.0 uses deterministic best-move play",
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
    parser.add_argument(
        "--mcts-node-budget", type=int, default=None, help="Optional MCTS node cap"
    )
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
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional JSON report path"
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="progress output mode; auto writes to stderr only when stderr is a TTY",
    )
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
    if parsed.opening_count < 1:
        parser.error("--opening-count must be at least 1")
    if parsed.opening_plies < 1:
        parser.error("--opening-plies must be at least 1")
    if parsed.opening_seed is None:
        parsed.opening_seed = parsed.seed
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
            parser.error(
                "--require-promotion cannot be used with --opponent-checkpoint"
            )
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


def _progress_enabled(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stderr.isatty()


def _print_report_summary(report: Mapping[str, object]) -> None:
    if "match" in report:
        match = _expect_mapping(report.get("match"), "match")
        match_items = [("opponent_checkpoint", match)]
        _print_game_summaries(match_items)
        _print_match_total("opponent_checkpoint", match)
        return

    matches = _expect_mapping(report.get("matches"), "matches")
    match_items = [
        (str(name), _expect_mapping(match, f"matches.{name}"))
        for name, match in matches.items()
    ]
    _print_game_summaries(match_items)
    promotion = _expect_mapping(report.get("promotion"), "promotion")
    print(
        "promotion "
        f"promoted={promotion.get('promoted')} "
        f"reasons={len(_expect_list(promotion.get('reasons'), 'promotion.reasons'))}"
    )
    for name, match in match_items:
        _print_match_total(name, match)


def _print_match_total(name: str, match: Mapping[str, object]) -> None:
    games = _expect_int(match.get("games"), f"{name}.games")
    score = _expect_number(match.get("player_a_score"), f"{name}.player_a_score")
    opponent_score = _expect_number(
        match.get("player_b_score"), f"{name}.player_b_score"
    )
    score_rate = _expect_number(
        match.get("player_a_score_rate"), f"{name}.player_a_score_rate"
    )
    wins = _expect_int(match.get("player_a_wins"), f"{name}.player_a_wins")
    losses = _expect_int(match.get("player_b_wins"), f"{name}.player_b_wins")
    draws = _expect_int(match.get("draws"), f"{name}.draws")
    print(
        f"total games={games} score={score:g}-{opponent_score:g} "
        f"score_rate={score_rate:.1%} wins={wins} losses={losses} draws={draws}"
    )


def _print_game_summaries(match_items: list[tuple[str, Mapping[str, object]]]) -> None:
    for name, match in match_items:
        records = _expect_list(match.get("records"), f"{name}.records")
        for raw_record in records:
            record = _expect_mapping(raw_record, f"{name}.records[]")
            print(_format_game_summary(name, record))


def _format_game_summary(name: str, record: Mapping[str, object]) -> str:
    winner = record.get("winner")
    winner_text = "draw" if winner is None else str(winner)
    return (
        f"game index={_expect_int(record.get('game_index'), 'game_index')} "
        f"checkpoint_color={record.get('player_a_color')} "
        f"score={_expect_number(record.get('player_a_score'), 'player_a_score'):g} "
        f"plies={_expect_int(record.get('plies'), 'plies')} "
        f"outcome={record.get('outcome_reason')} winner={winner_text} "
        f"winner_player={_winner_player_label(name, record)} "
        f"moves={len(_expect_list(record.get('moves_uci'), 'moves_uci'))} "
        f"opening={record.get('opening_index')}"
    )


def _winner_player_label(name: str, record: Mapping[str, object]) -> str:
    score = _expect_number(record.get("player_a_score"), "player_a_score")
    if score == 1.0:
        return "checkpoint"
    if score == 0.0:
        return name
    return "draw"


def _expect_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _die(f"internal error: malformed {field}")
    return value


def _expect_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        _die(f"internal error: malformed {field}")
    return value


def _expect_int(value: object, field: str) -> int:
    if not isinstance(value, int):
        _die(f"internal error: malformed {field}")
    return value


def _expect_number(value: object, field: str) -> float:
    if not isinstance(value, int | float):
        _die(f"internal error: malformed {field}")
    return float(value)


def _die(message: str) -> NoReturn:
    raise SystemExit(message)


if __name__ == "__main__":
    main()
