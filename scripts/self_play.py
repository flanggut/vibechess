#!/usr/bin/env python3
"""Generate a small vibechess MCTS self-play dataset."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, replace
from multiprocessing.managers import SyncManager
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from typing import ClassVar, Literal, Protocol, TextIO, cast

from vibechess.ai.neural_mcts import NeuralMCTSConfig
from vibechess.ai.search_config import MCTSConfig
from vibechess.nn.checkpoint import load_checkpoint, save_checkpoint
from vibechess.nn.inference import PolicyValueInference
from vibechess.nn.model import PolicyValueConfig, PolicyValueNet
from vibechess.nn.self_play import (
    DEFAULT_PROFILE_FILENAME,
    LABEL_SOURCE_CLASSICAL,
    LABEL_SOURCE_NEURAL,
    LABEL_SOURCES,
    SelfPlayConfig,
    SelfPlayProgress,
    generate_self_play_dataset,
    self_play_profile,
)
from vibechess.nn.self_play_dataset import (
    SelfPlayDataset,
    load_self_play_dataset,
    merge_self_play_datasets,
    save_self_play_dataset,
)
from vibechess.nn.self_play_profile import (
    ProfileStats,
    profile_level_from_env,
    profile_scope,
    stats_from_profile_report,
)
from vibechess.nn.self_play_profile import (
    profile_report as build_profile_report,
)

PROFILE_ENV_VAR = "VIBECHESS_SELF_PLAY_PROFILE"
_PROGRESS_POLL_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class GenerationArgs:
    checkpoint: Path | None
    checkpoint_id: str | None
    label_source: str
    max_plies: int
    simulations: int
    temperature: float
    classical_exploration: float
    classical_max_rollout_plies: int
    seed: int
    channels: int
    blocks: int
    batch_size: int
    active_games: int | None
    collection_batch_size: int
    virtual_loss: int
    reuse_simulation_budget: bool = False
    min_reuse_simulations: int = 0


_ProgressStatus = Literal["pending", "running", "completed", "saving", "done", "failed"]


@dataclass(frozen=True, slots=True)
class _WorkerProgressEvent:
    worker_id: int
    start_game: int
    total_games: int
    games_completed: int
    samples: int
    plies: int
    status: _ProgressStatus


class _ProgressEventSink(Protocol):
    def put(self, item: _WorkerProgressEvent) -> object:
        ...


class _ProgressEventSource(Protocol):
    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> _WorkerProgressEvent:
        ...


@dataclass(frozen=True, slots=True)
class ChunkTask:
    generation_args: GenerationArgs
    worker_id: int
    start_game: int
    games: int
    parent_pool_start_ns: int
    profile_level: str
    shard_output: Path
    progress_queue: _ProgressEventSink | None = None


@dataclass(frozen=True, slots=True)
class ChunkGenerationResult:
    shard_output: Path
    sample_count: int
    game_count: int
    profile: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _WorkerProgressState:
    worker_id: int
    start_game: int
    total_games: int
    games_completed: int = 0
    samples: int = 0
    plies: int = 0
    status: _ProgressStatus = "pending"

    @property
    def processed_games(self) -> int:
        return self.games_completed


@dataclass(frozen=True, slots=True)
class _ProgressRenderState:
    total_games: int
    workers: tuple[_WorkerProgressState, ...]
    status: _ProgressStatus
    message: str | None = None


@dataclass(slots=True)
class _AnsiProgressRenderer:
    enabled: bool
    total_games: int
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    _rendered_lines: int = 0
    _started: bool = False
    _finished: bool = False
    _restore_registered: bool = False

    _BAR_WIDTH: ClassVar[int] = 24
    _CLEAR_LINE: ClassVar[str] = "\x1b[2K"
    _CURSOR_HIDE: ClassVar[str] = "\x1b[?25l"
    _CURSOR_SHOW: ClassVar[str] = "\x1b[?25h"

    def render(self, state: _ProgressRenderState) -> None:
        self._draw(state, restore_cursor=False)

    def finish(self, state: _ProgressRenderState) -> None:
        self._draw(state, restore_cursor=True)
        if self.enabled:
            self._finished = True

    def _draw(self, state: _ProgressRenderState, *, restore_cursor: bool) -> None:
        if not self.enabled or self._finished:
            return
        lines = self._format_lines(state)
        width = self._terminal_width()
        if not self._started:
            self.stream.write(self._CURSOR_HIDE)
            if not self._restore_registered:
                atexit.register(self._restore_cursor)
                self._restore_registered = True
            self._started = True
        self._clear_previous()
        self.stream.write("\n".join(lines))
        if restore_cursor:
            self.stream.write(self._CURSOR_SHOW)
        self.stream.write("\n")
        self.stream.flush()
        self._rendered_lines = self._physical_rows(lines, width)

    def cleanup(self) -> None:
        self._restore_cursor()

    def _restore_cursor(self) -> None:
        if not self.enabled or not self._started or self._finished:
            return
        self.stream.write(self._CURSOR_SHOW)
        self.stream.write("\n")
        self.stream.flush()
        self._finished = True

    @staticmethod
    def _terminal_width() -> int:
        return shutil.get_terminal_size(fallback=(80, 24)).columns

    @staticmethod
    def _physical_rows(lines: list[str], width: int) -> int:
        # A status line longer than the terminal width wraps onto multiple
        # physical rows. The cursor-up (`\x1b[NF`) and clear-line counts in
        # `_clear_previous` operate on physical rows, so they must account for
        # that wrapping; otherwise the redraw clears too few rows and leaves
        # stale, partially-overwritten copies of the status block behind.
        if width <= 0:
            return len(lines)
        return sum(max(1, -(-len(line) // width)) for line in lines)

    def _clear_previous(self) -> None:
        if self._rendered_lines == 0:
            return
        self.stream.write(f"\x1b[{self._rendered_lines}F")
        for _ in range(self._rendered_lines):
            self.stream.write(f"{self._CLEAR_LINE}\n")
        self.stream.write(f"\x1b[{self._rendered_lines}F")

    def _format_lines(self, state: _ProgressRenderState) -> list[str]:
        games_completed = sum(worker.processed_games for worker in state.workers)
        games_completed = min(state.total_games, games_completed)
        samples = sum(worker.samples for worker in state.workers)
        plies = sum(worker.plies for worker in state.workers)
        header = " ".join(
            [
                "self-play",
                f"status={state.status}",
                f"games={games_completed}/{state.total_games}",
                f"samples={samples}",
                f"plies={plies}",
            ]
        )
        if state.message is not None:
            header = f"{header} {state.message}"
        lines = [
            header,
            " ".join(
                [
                    "total ",
                    f"[{self._bar(games_completed, state.total_games)}]",
                    self._percent(games_completed, state.total_games),
                ]
            ),
        ]
        lines.extend(self._format_worker(worker) for worker in state.workers)
        return lines

    def _format_worker(self, worker: _WorkerProgressState) -> str:
        game_range = self._game_range(worker.start_game, worker.total_games)
        return " ".join(
            [
                f"w{worker.worker_id:02d}",
                f"status={worker.status}",
                f"[{self._bar(worker.processed_games, worker.total_games)}]",
                f"games={worker.processed_games}/{worker.total_games}",
                f"samples={worker.samples}",
                f"plies={worker.plies}",
                f"range={game_range}",
            ]
        )

    def _bar(self, completed: int, total: int) -> str:
        if total <= 0:
            return "░" * self._BAR_WIDTH
        if completed >= total:
            filled = self._BAR_WIDTH
        else:
            filled = max(0, completed * self._BAR_WIDTH // total)
        return "█" * filled + "░" * (self._BAR_WIDTH - filled)

    @staticmethod
    def _percent(completed: int, total: int) -> str:
        if total <= 0:
            return "0.0%"
        return f"{completed / total:6.1%}"

    @staticmethod
    def _game_range(start_game: int, games: int) -> str:
        if games <= 0:
            return "none"
        return f"{start_game + 1}-{start_game + games}"


@dataclass(slots=True)
class _ProgressReporter:
    enabled: bool
    total_games: int
    _renderer: _AnsiProgressRenderer = field(init=False)
    _workers_by_start: dict[int, _WorkerProgressState] = field(
        default_factory=dict,
        init=False,
    )
    _status: _ProgressStatus = field(default="pending", init=False)

    def __post_init__(self) -> None:
        self._renderer = _AnsiProgressRenderer(
            enabled=self.enabled,
            total_games=self.total_games,
        )

    def start(self, args: argparse.Namespace) -> None:
        self._workers_by_start = self._initial_workers(args)
        self._status = "running"
        self._write(
            " ".join(
                [
                    "starting",
                    f"games={args.games}",
                    f"max_plies={args.max_plies}",
                    f"simulations={args.simulations}",
                    f"label_source={args.label_source}",
                    f"workers={args.workers}",
                    f"batch_size={args.batch_size}",
                    f"output={args.output}",
                ]
            )
        )

    def game_completed(self, progress: SelfPlayProgress) -> None:
        self._upsert_worker(
            start_game=0,
            total_games=progress.total_games,
            games_completed=progress.games_completed,
            samples=progress.samples,
            plies=progress.plies,
            status=(
                "completed"
                if progress.games_completed >= progress.total_games
                else "running"
            ),
        )
        if progress.games_completed >= progress.total_games:
            self._status = "completed"
        self._write(
            " ".join(
                [
                    f"completed={progress.games_completed}/{progress.total_games}",
                    f"game_index={progress.game_index}",
                    f"samples={progress.samples}",
                    f"plies={progress.plies}",
                ]
            )
        )

    def worker_progress(self, event: _WorkerProgressEvent) -> None:
        self._upsert_worker(
            start_game=event.start_game,
            total_games=event.total_games,
            games_completed=event.games_completed,
            samples=event.samples,
            plies=event.plies,
            status=event.status,
        )
        if event.status == "failed":
            self._status = "failed"
        elif self._status not in ("failed", "saving", "done"):
            processed_games = sum(
                worker.processed_games for worker in self._workers_by_start.values()
            )
            self._status = (
                "completed" if processed_games >= self.total_games else "running"
            )
        self._render()

    def chunk_completed(
        self,
        *,
        games_completed: int,
        total_games: int,
        start_game: int,
        games: int,
        samples: int,
    ) -> None:
        worker = self._workers_by_start.get(start_game)
        self._upsert_worker(
            start_game=start_game,
            total_games=games,
            games_completed=games,
            samples=samples,
            plies=0 if worker is None else worker.plies,
            status="completed",
        )
        if games_completed >= total_games:
            self._status = "completed"
        self._write(
            " ".join(
                [
                    f"chunk_completed={games_completed}/{total_games}",
                    f"start_game={start_game}",
                    f"games={games}",
                    f"samples={samples}",
                ]
            )
        )

    def saving(self, output: Path) -> None:
        self._status = "saving"
        self._write(f"saving output={output}")

    def done(self, dataset: SelfPlayDataset) -> None:
        self._status = "done"
        self._write(
            " ".join(
                [
                    "done",
                    f"games={dataset.metadata.game_count}",
                    f"samples={dataset.metadata.sample_count}",
                    f"schema={dataset.metadata.schema_version}",
                ]
            ),
            finish=True,
        )

    def _initial_workers(self, args: argparse.Namespace) -> dict[int, _WorkerProgressState]:
        games = int(args.games)
        requested_workers = int(args.workers)
        if games <= 0:
            return {}
        if requested_workers == 1 or games == 1:
            chunks = [(0, games)]
        else:
            chunks = _split_games(games, min(requested_workers, games))
        return {
            start_game: _WorkerProgressState(
                worker_id=worker_id,
                start_game=start_game,
                total_games=worker_games,
                status="running",
            )
            for worker_id, (start_game, worker_games) in enumerate(chunks)
        }

    def _upsert_worker(
        self,
        *,
        start_game: int,
        total_games: int,
        games_completed: int,
        samples: int,
        plies: int,
        status: _ProgressStatus,
    ) -> None:
        worker = self._workers_by_start.get(start_game)
        worker_id = len(self._workers_by_start) if worker is None else worker.worker_id
        self._workers_by_start[start_game] = _WorkerProgressState(
            worker_id=worker_id,
            start_game=start_game,
            total_games=total_games,
            games_completed=games_completed,
            samples=samples,
            plies=plies,
            status=status,
        )

    def _render(self, message: str | None = None, *, finish: bool = False) -> None:
        snapshot = _ProgressRenderState(
            total_games=self.total_games,
            workers=tuple(
                sorted(
                    self._workers_by_start.values(),
                    key=lambda worker: worker.worker_id,
                )
            ),
            status=self._status,
            message=message,
        )
        if finish:
            self._renderer.finish(snapshot)
        else:
            self._renderer.render(snapshot)

    def _write(self, message: str, *, finish: bool = False) -> None:
        legacy_message = (
            message
            if message.startswith("self-play: ")
            else f"self-play: {message}"
        )
        self._render(legacy_message, finish=finish)

    def cleanup(self) -> None:
        self._renderer.cleanup()


def _build_inference(args: GenerationArgs) -> PolicyValueInference:
    with profile_scope("model.build_inference"):
        if args.checkpoint is None:
            model_config = PolicyValueConfig(
                residual_channels=args.channels,
                residual_blocks=args.blocks,
                policy_channels=1,
                value_channels=1,
                value_hidden_dim=4,
            )
            return PolicyValueInference(PolicyValueNet(model_config))

        loaded = load_checkpoint(args.checkpoint)
        return PolicyValueInference(loaded.model)


def _self_play_config(args: GenerationArgs, *, games: int, seed: int) -> SelfPlayConfig:
    return SelfPlayConfig(
        games=games,
        max_plies=args.max_plies,
        mcts=NeuralMCTSConfig(
            simulations=args.simulations,
            temperature=args.temperature,
            seed=seed,
            collection_batch_size=args.collection_batch_size,
            virtual_loss=args.virtual_loss,
            reuse_simulation_budget=args.reuse_simulation_budget,
            min_reuse_simulations=args.min_reuse_simulations,
        ),
        classical_mcts=MCTSConfig(
            simulations=args.simulations,
            exploration=args.classical_exploration,
            max_rollout_plies=args.classical_max_rollout_plies,
            seed=seed,
        ),
        label_source=args.label_source,
        model_checkpoint_id=args.checkpoint_id,
        seed=seed,
        batch_size=args.batch_size,
        active_games=args.active_games,
    )


def _emit_worker_progress(
    task: ChunkTask,
    *,
    games_completed: int,
    samples: int,
    plies: int,
    status: _ProgressStatus,
) -> None:
    if task.progress_queue is None:
        return
    event = _WorkerProgressEvent(
        worker_id=task.worker_id,
        start_game=task.start_game,
        total_games=task.games,
        games_completed=games_completed,
        samples=samples,
        plies=plies,
        status=status,
    )
    try:
        task.progress_queue.put(event)
    except Exception:
        return


def _generate_chunk(task: ChunkTask) -> ChunkGenerationResult:
    metadata = {
        "worker_id": task.worker_id,
        "start_game": task.start_game,
        "games": task.games,
        "seed": task.generation_args.seed + task.start_game,
    }
    games_completed = 0
    samples = 0
    plies = 0

    def report_game(progress: SelfPlayProgress) -> None:
        nonlocal games_completed, samples, plies
        games_completed = progress.games_completed
        samples = progress.samples
        plies = progress.plies
        _emit_worker_progress(
            task,
            games_completed=games_completed,
            samples=samples,
            plies=plies,
            status="running",
        )

    _emit_worker_progress(
        task,
        games_completed=games_completed,
        samples=samples,
        plies=plies,
        status="running",
    )
    try:
        with self_play_profile(task.profile_level) as profiler:
            if profiler.enabled:
                profiler.stats.metadata.update(metadata)
                profiler.stats.add_counter(
                    "worker.start_lag_ns",
                    time.perf_counter_ns() - task.parent_pool_start_ns,
                )
            with profile_scope("worker.chunk_elapsed", **metadata):
                inference = None
                if task.generation_args.label_source == LABEL_SOURCE_NEURAL:
                    inference = _build_inference(task.generation_args)
                config = _self_play_config(
                    task.generation_args,
                    games=task.games,
                    seed=task.generation_args.seed + task.start_game,
                )
                dataset = generate_self_play_dataset(
                    inference,
                    config=config,
                    progress=report_game if task.progress_queue is not None else None,
                )
            samples = dataset.metadata.sample_count
            games_completed = dataset.metadata.game_count
            plies = sum(record.plies for record in dataset.games)
            _emit_worker_progress(
                task,
                games_completed=games_completed,
                samples=samples,
                plies=plies,
                status="saving",
            )
            with profile_scope("worker.shard_save", **metadata):
                save_self_play_dataset(dataset, task.shard_output)
            _emit_worker_progress(
                task,
                games_completed=games_completed,
                samples=samples,
                plies=plies,
                status="completed",
            )
            if not profiler.enabled:
                return ChunkGenerationResult(
                    shard_output=task.shard_output,
                    sample_count=samples,
                    game_count=games_completed,
                )
            profile = build_profile_report(
                profiler.stats,
                scope="self_play_worker",
                profile_level=profiler.level,
                metadata={
                    **metadata,
                    "pid": os.getpid(),
                    "samples": samples,
                    "game_count": games_completed,
                    "plies": plies,
                    "shard_output": str(task.shard_output),
                },
            )
            return ChunkGenerationResult(
                shard_output=task.shard_output,
                sample_count=samples,
                game_count=games_completed,
                profile=profile,
            )
    except Exception:
        _emit_worker_progress(
            task,
            games_completed=games_completed,
            samples=samples,
            plies=plies,
            status="failed",
        )
        raise


def _drain_worker_progress_events(
    progress_queue: _ProgressEventSource,
    progress_reporter: _ProgressReporter,
) -> None:
    while True:
        try:
            event = progress_queue.get(block=False)
        except Empty:
            return
        progress_reporter.worker_progress(event)


def _progress_enabled(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stderr.isatty()


def _profile_level() -> str:
    return profile_level_from_env(PROFILE_ENV_VAR)


def _profile_from_report(report: dict[str, object]) -> ProfileStats:
    return stats_from_profile_report(report)


def _write_profile(
    output: Path,
    profile: ProfileStats,
    *,
    profile_level: str,
    worker_profiles: list[dict[str, object]] | None = None,
    derived: dict[str, object] | None = None,
) -> None:
    with profile_scope("profile.write_sidecar"):
        report = build_profile_report(
            profile,
            scope="self_play_generation",
            profile_level=profile_level,  # type: ignore[arg-type]
            worker_profiles=worker_profiles,
            derived=derived,
        )
        (output / DEFAULT_PROFILE_FILENAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )


def _temporary_checkpoint(args: GenerationArgs) -> TemporaryDirectory[str]:
    with profile_scope("self_play.temp_checkpoint_save"):
        temp_dir = TemporaryDirectory(prefix="vibechess-self-play-")
        model_config = PolicyValueConfig(
            residual_channels=args.channels,
            residual_blocks=args.blocks,
            policy_channels=1,
            value_channels=1,
            value_hidden_dim=4,
        )
        save_checkpoint(PolicyValueNet(model_config), temp_dir.name)
        return temp_dir


def _split_games(games: int, workers: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    start = 0
    for worker_index in range(workers):
        count = games // workers + (1 if worker_index < games % workers else 0)
        if count > 0:
            chunks.append((start, count))
            start += count
    return chunks


def _worker_derived_profile(worker_reports: list[dict[str, object]]) -> dict[str, object]:
    worker_elapsed: list[float] = []
    for report in worker_reports:
        stats = stats_from_profile_report(report)
        chunk_timer = stats.zones.get("worker.chunk_elapsed")
        if chunk_timer is not None:
            worker_elapsed.append(chunk_timer.inclusive_ns / 1_000_000_000.0)
    derived: dict[str, object] = {"worker_count": len(worker_reports)}
    if worker_elapsed:
        derived.update(
            {
                "worker_time_max_seconds": max(worker_elapsed),
                "worker_time_sum_seconds": sum(worker_elapsed),
            }
        )
    return derived


def _directory_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate vibechess self-play samples.")
    parser.add_argument("--output", type=Path, default=Path("data/selfplay/smoke"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument(
        "--label-source",
        choices=LABEL_SOURCES,
        default=LABEL_SOURCE_NEURAL,
        help="search source for policy labels and self-play moves",
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument(
        "--reuse-simulation-budget",
        action="store_true",
        help=(
            "when neural tree reuse adopts an existing root, spend only the remaining "
            "visit budget instead of always running --simulations new visits"
        ),
    )
    parser.add_argument(
        "--min-reuse-simulations",
        type=int,
        default=0,
        help=(
            "minimum new neural MCTS simulations to run after visit-budget-aware "
            "root reuse; default 0 allows fully reusing an already-searched root"
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--classical-exploration", type=float, default=1.41421356237)
    parser.add_argument(
        "--classical-max-rollout-plies",
        type=int,
        default=0,
        help="classical MCTS rollout cap; default 0 uses static leaf evaluation",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "in-process central inference batch size across independent neural "
            "self-play games/searches; default 8 enables central inference batching. "
            "Set to 1 for serial behavior. This is not within-tree leaf parallelism, "
            "which has been removed."
        ),
    )
    parser.add_argument(
        "--active-games",
        type=int,
        default=None,
        help=(
            "maximum in-process active neural self-play games; defaults to --batch-size; "
            "inference calls are still capped by --batch-size"
        ),
    )
    parser.add_argument(
        "--collection-batch-size",
        type=int,
        default=1,
        help=(
            "within-search virtual-loss leaf collection width per neural search; "
            "default 1 is serial. Values >1 gather that many distinct leaves per round "
            "and evaluate them in one batched model call. Only active on the serial "
            "self-play path (--batch-size 1); ignored when central cross-game batching "
            "(--batch-size >1) is used."
        ),
    )
    parser.add_argument(
        "--virtual-loss",
        type=int,
        default=1,
        help=(
            "pessimistic visit count temporarily applied to in-flight leaf paths so "
            "collected leaves diverge; only used when --collection-batch-size >1"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes for parallel self-play generation",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="progress output mode; auto writes to stderr only when stderr is a TTY",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.active_games is not None and args.active_games < 1:
        parser.error("--active-games must be at least 1")
    if args.min_reuse_simulations < 0:
        parser.error("--min-reuse-simulations must be non-negative")
    if (
        args.reuse_simulation_budget
        and args.min_reuse_simulations > args.simulations
    ):
        parser.error(
            "--min-reuse-simulations must be no greater than --simulations "
            "when --reuse-simulation-budget is enabled"
        )
    if args.label_source == LABEL_SOURCE_CLASSICAL:
        if args.checkpoint is not None:
            parser.error("--checkpoint is only supported with --label-source neural")
        if args.checkpoint_id is not None:
            parser.error("--checkpoint-id is only supported with --label-source neural")

    checkpoint_id = args.checkpoint_id
    if args.checkpoint is not None:
        checkpoint_id = checkpoint_id or str(args.checkpoint)
    generation_args = GenerationArgs(
        checkpoint=args.checkpoint,
        checkpoint_id=checkpoint_id,
        label_source=args.label_source,
        max_plies=args.max_plies,
        simulations=args.simulations,
        temperature=args.temperature,
        classical_exploration=args.classical_exploration,
        classical_max_rollout_plies=args.classical_max_rollout_plies,
        seed=args.seed,
        channels=args.channels,
        blocks=args.blocks,
        batch_size=args.batch_size,
        active_games=args.active_games,
        collection_batch_size=args.collection_batch_size,
        virtual_loss=args.virtual_loss,
        reuse_simulation_budget=args.reuse_simulation_budget,
        min_reuse_simulations=args.min_reuse_simulations,
    )
    full_config = _self_play_config(generation_args, games=args.games, seed=args.seed)

    profile_level = _profile_level()
    progress_reporter = _ProgressReporter(
        enabled=_progress_enabled(args.progress),
        total_games=args.games,
    )
    worker_reports: list[dict[str, object]] = []
    derived_profile: dict[str, object] = {}
    try:
        with self_play_profile(profile_level) as main_profiler, profile_scope("self_play.main"):
            with profile_scope("self_play.setup"):
                progress_reporter.start(args)
            chunk_results: list[ChunkGenerationResult] = []
            if args.workers == 1 or args.games == 1:
                inference = None
                if generation_args.label_source == LABEL_SOURCE_NEURAL:
                    inference = _build_inference(generation_args)
                dataset = generate_self_play_dataset(
                    inference,
                    config=full_config,
                    progress=progress_reporter.game_completed,
                )
            else:
                workers = min(args.workers, args.games)
                chunks = _split_games(args.games, workers)
                temp_checkpoint: TemporaryDirectory[str] | None = None
                if (
                    generation_args.label_source == LABEL_SOURCE_NEURAL
                    and generation_args.checkpoint is None
                ):
                    temp_checkpoint = _temporary_checkpoint(generation_args)
                    generation_args = replace(
                        generation_args,
                        checkpoint=Path(temp_checkpoint.name),
                    )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                shard_temp = TemporaryDirectory(
                    prefix=f".{args.output.name or 'self-play'}-shards-",
                    dir=str(args.output.parent),
                )
                pool_start_ns = time.perf_counter_ns()
                completed_games = 0
                chunk_results_by_start: list[tuple[int, ChunkGenerationResult]] = []
                progress_manager: SyncManager | None = None
                progress_queue: _ProgressEventSource | None = None
                progress_sink: _ProgressEventSink | None = None
                if progress_reporter.enabled:
                    progress_manager = SyncManager()
                    progress_manager.start()
                    progress_queue = cast(
                        _ProgressEventSource,
                        progress_manager.Queue(),
                    )
                    progress_sink = cast(_ProgressEventSink, progress_queue)
                try:
                    try:
                        shard_root = Path(shard_temp.name)
                        with (
                            profile_scope("worker.pool_elapsed", workers=workers),
                            ProcessPoolExecutor(max_workers=workers) as executor,
                        ):
                            futures: dict[
                                Future[ChunkGenerationResult],
                                tuple[int, int],
                            ] = {
                                executor.submit(
                                    _generate_chunk,
                                    ChunkTask(
                                        generation_args=generation_args,
                                        worker_id=worker_id,
                                        start_game=start_game,
                                        games=games,
                                        parent_pool_start_ns=pool_start_ns,
                                        profile_level=profile_level,
                                        shard_output=(
                                            shard_root
                                            / f"worker-{worker_id:03d}-start-{start_game:06d}"
                                        ),
                                        progress_queue=progress_sink,
                                    ),
                                ): (start_game, games)
                                for worker_id, (start_game, games) in enumerate(chunks)
                            }
                            pending = set(futures)
                            worker_errors: list[Exception] = []
                            while pending:
                                done, pending = wait(
                                    pending,
                                    timeout=(
                                        _PROGRESS_POLL_SECONDS
                                        if progress_queue is not None
                                        else None
                                    ),
                                    return_when=FIRST_COMPLETED,
                                )
                                if progress_queue is not None:
                                    _drain_worker_progress_events(
                                        progress_queue,
                                        progress_reporter,
                                    )
                                for future in done:
                                    start_game, games = futures[future]
                                    try:
                                        result = future.result()
                                    except Exception as exc:
                                        worker_errors.append(exc)
                                        continue
                                    chunk_results_by_start.append((start_game, result))
                                    completed_games += games
                                    progress_reporter.chunk_completed(
                                        games_completed=completed_games,
                                        total_games=args.games,
                                        start_game=start_game,
                                        games=games,
                                        samples=result.sample_count,
                                    )
                                if worker_errors:
                                    if pending:
                                        done, pending = wait(pending)
                                        if progress_queue is not None:
                                            _drain_worker_progress_events(
                                                progress_queue,
                                                progress_reporter,
                                            )
                                        for future in done:
                                            try:
                                                future.result()
                                            except Exception as exc:
                                                worker_errors.append(exc)
                                    if len(worker_errors) == 1:
                                        raise worker_errors[0]
                                    raise ExceptionGroup(
                                        "multiple self-play workers failed",
                                        worker_errors,
                                    )
                    finally:
                        if progress_queue is not None:
                            _drain_worker_progress_events(
                                progress_queue,
                                progress_reporter,
                            )
                        if progress_manager is not None:
                            progress_manager.shutdown()
                        if temp_checkpoint is not None:
                            temp_checkpoint.cleanup()
                    chunk_results = [
                        result
                        for _start_game, result in sorted(
                            chunk_results_by_start,
                            key=lambda item: item[0],
                        )
                    ]
                    worker_reports = [
                        result.profile for result in chunk_results if result.profile is not None
                    ]
                    worker_stats = [_profile_from_report(profile) for profile in worker_reports]
                    if main_profiler.enabled:
                        for stats in worker_stats:
                            main_profiler.stats.merge(stats)
                        derived_profile.update(_worker_derived_profile(worker_reports))
                    parallel_settings: dict[str, object] = {
                        "parallel": {
                            "workers": workers,
                            "chunks": [
                                {
                                    "start_game": start_game,
                                    "games": games,
                                    "seed": args.seed + start_game,
                                }
                                for start_game, games in chunks
                            ],
                        }
                    }
                    with profile_scope("dataset.load_shards", shards=len(chunk_results)):
                        shard_datasets = [
                            load_self_play_dataset(result.shard_output)
                            for result in chunk_results
                        ]
                    dataset = merge_self_play_datasets(
                        shard_datasets,
                        config=full_config,
                        generation_settings_extra=parallel_settings,
                    )
                finally:
                    shard_temp.cleanup()
            progress_reporter.saving(args.output)
            save_self_play_dataset(dataset, args.output)
            if main_profiler.enabled:
                main_profiler.stats.add_counter(
                    "dataset.output_bytes",
                    _directory_size(args.output),
                )
                _write_profile(
                    args.output,
                    main_profiler.stats,
                    profile_level=main_profiler.level,
                    worker_profiles=worker_reports,
                    derived=derived_profile,
                )
            progress_reporter.done(dataset)
    finally:
        progress_reporter.cleanup()
    print(
        " ".join(
            [
                f"output={args.output}",
                f"games={dataset.metadata.game_count}",
                f"samples={dataset.metadata.sample_count}",
                f"schema={dataset.metadata.schema_version}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
