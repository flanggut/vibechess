#!/usr/bin/env python3
"""Generate a small vibechess MCTS self-play dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, replace
from multiprocessing.managers import SyncManager
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from typing import Protocol, cast

from _progress import AnsiProgressRenderer as _AnsiProgressRenderer
from _progress import ProgressRenderState as _ProgressRenderState
from _progress import ProgressStatus as _ProgressStatus
from _progress import WorkerProgressState as _WorkerProgressState

from vibechess import _scriptutil
from vibechess.ai.neural_mcts import NeuralMCTSConfig
from vibechess.ai.search_config import MCTSConfig
from vibechess.nn.checkpoint import load_checkpoint, save_checkpoint
from vibechess.nn.inference import PolicyValueInference
from vibechess.nn.model import PolicyValueConfig, PolicyValueNet
from vibechess.nn.self_play import (
    BATCHING_MODE_CENTRAL_INFERENCE_QUEUE,
    BATCHING_MODE_SERIAL,
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
    DEFAULT_DATASET_FILENAME,
    DEFAULT_GAMES_FILENAME,
    DEFAULT_METADATA_FILENAME,
    SelfPlayDataset,
    SelfPlayMetadata,
    append_self_play_dataset,
    existing_self_play_dataset_exists,
    load_self_play_dataset,
    load_self_play_shard_manifest,
    save_merged_self_play_shards,
    save_self_play_dataset,
    save_self_play_shard,
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
# How often the elapsed/eta counters are refreshed independently of progress
# events, so the timer keeps ticking even while a long game is in flight.
_PROGRESS_REFRESH_SECONDS = 1.0


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


@dataclass(slots=True)
class _ProgressReporter:
    enabled: bool
    total_games: int
    start_game_offset: int = 0
    _renderer: _AnsiProgressRenderer = field(init=False)
    _workers_by_start: dict[int, _WorkerProgressState] = field(
        default_factory=dict,
        init=False,
    )
    _status: _ProgressStatus = field(default="pending", init=False)
    _start_monotonic: float | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _refresh_stop: threading.Event = field(
        default_factory=threading.Event,
        init=False,
    )
    _refresh_thread: threading.Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._renderer = _AnsiProgressRenderer(
            enabled=self.enabled,
            total_games=self.total_games,
        )

    def start(self, args: argparse.Namespace, *, start_game_offset: int = 0) -> None:
        self.start_game_offset = start_game_offset
        self._workers_by_start = self._initial_workers(args)
        self._status = "running"
        self._start_monotonic = time.monotonic()
        self._start_refresh()
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
            start_game=self.start_game_offset,
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

    def done_metadata(self, metadata: SelfPlayMetadata) -> None:
        self._status = "done"
        self._write(
            " ".join(
                [
                    "done",
                    f"games={metadata.game_count}",
                    f"samples={metadata.sample_count}",
                    f"schema={metadata.schema_version}",
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
        chunks = [
            (self.start_game_offset + start_game, worker_games)
            for start_game, worker_games in chunks
        ]
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
        with self._lock:
            worker = self._workers_by_start.get(start_game)
            worker_id = (
                len(self._workers_by_start) if worker is None else worker.worker_id
            )
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
        # The lock serializes renders from the main thread (driven by progress
        # events) with the background refresh thread so their ANSI escape
        # sequences never interleave, and guards the worker snapshot against
        # concurrent mutation in `_upsert_worker`.
        with self._lock:
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
            name="self-play-progress-refresh",
            daemon=True,
        )
        self._refresh_thread = thread
        thread.start()

    def _refresh_loop(self) -> None:
        while not self._refresh_stop.wait(_PROGRESS_REFRESH_SECONDS):
            # Re-render the current state with refreshed elapsed/eta. Once the
            # display is finished the renderer ignores further draws, so this
            # becomes a cheap no-op until the thread is stopped.
            self._render()

    def _stop_refresh(self) -> None:
        self._refresh_stop.set()
        thread = self._refresh_thread
        if thread is not None:
            thread.join(timeout=_PROGRESS_REFRESH_SECONDS + 1.0)
            self._refresh_thread = None

    def _write(self, message: str, *, finish: bool = False) -> None:
        legacy_message = (
            message
            if message.startswith("self-play: ")
            else f"self-play: {message}"
        )
        self._render(legacy_message, finish=finish)

    def cleanup(self) -> None:
        self._stop_refresh()
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


def _self_play_config(
    args: GenerationArgs,
    *,
    games: int,
    seed: int,
    start_game_index: int = 0,
) -> SelfPlayConfig:
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
        start_game_index=start_game_index,
    )


def _self_play_sidecar_paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output / DEFAULT_DATASET_FILENAME,
        output / DEFAULT_METADATA_FILENAME,
        output / DEFAULT_GAMES_FILENAME,
    )


def _load_existing_dataset_for_append(
    parser: argparse.ArgumentParser,
    output: Path,
) -> SelfPlayDataset | None:
    sidecars = _self_play_sidecar_paths(output)
    existing_sidecars = [path for path in sidecars if path.exists()]
    if not existing_sidecars:
        return None
    if len(existing_sidecars) != len(sidecars):
        present = ", ".join(path.name for path in existing_sidecars)
        missing = ", ".join(path.name for path in sidecars if not path.exists())
        parser.error(
            "cannot append to incomplete self-play dataset sidecars "
            f"in {output}: present={present}; missing={missing}"
        )
    if not existing_self_play_dataset_exists(output):
        return None
    try:
        return load_self_play_dataset(output)
    except Exception as exc:
        parser.error(f"cannot append to existing self-play dataset in {output}: {exc}")
    raise AssertionError("parser.error should exit")


def _batching_metadata_for_args(args: GenerationArgs) -> tuple[str, int]:
    if args.label_source == LABEL_SOURCE_NEURAL and args.batch_size > 1:
        return BATCHING_MODE_CENTRAL_INFERENCE_QUEUE, args.batch_size
    return BATCHING_MODE_SERIAL, 1


def _requested_generation_settings(
    args: GenerationArgs,
    config: SelfPlayConfig,
) -> dict[str, object]:
    batching_mode, inference_batch_size = _batching_metadata_for_args(args)
    return config.to_dict(
        batching_mode=batching_mode,
        inference_batch_size=inference_batch_size,
    )


def _append_relevant_settings(settings: dict[str, object]) -> dict[str, object]:
    relevant_keys = (
        "max_plies",
        "label_source",
        "mcts",
        "classical_mcts",
        "model_checkpoint_id",
        "seed",
        "batch_size",
        "active_games",
        "batching_mode",
        "inference_batch_size",
    )
    return {key: settings.get(key) for key in relevant_keys}


def _validate_append_request(
    parser: argparse.ArgumentParser,
    existing: SelfPlayDataset,
    requested_settings: dict[str, object],
    checkpoint_id: str | None,
) -> None:
    if existing.metadata.model_checkpoint_id != checkpoint_id:
        parser.error(
            "cannot append to existing self-play dataset: model checkpoint differs "
            f"(existing={existing.metadata.model_checkpoint_id!r}, requested={checkpoint_id!r})"
        )
    if (
        requested_settings.get("label_source") == LABEL_SOURCE_NEURAL
        and existing.metadata.model_checkpoint_id is None
        and checkpoint_id is None
    ):
        parser.error(
            "cannot append neural self-play without an explicit checkpoint id; "
            "rerun with --checkpoint or --checkpoint-id so checkpoint continuity is verifiable"
        )
    existing_generation_settings = dict(existing.metadata.generation_settings)
    existing_generation_settings.setdefault(
        "model_checkpoint_id",
        existing.metadata.model_checkpoint_id,
    )
    existing_settings = _append_relevant_settings(existing_generation_settings)
    requested_relevant = _append_relevant_settings(requested_settings)
    for key, existing_value in existing_settings.items():
        requested_value = requested_relevant[key]
        if existing_value != requested_value:
            parser.error(
                "cannot append to existing self-play dataset: generation setting "
                f"{key!r} differs (existing={existing_value!r}, requested={requested_value!r})"
            )


def _save_self_play_dataset_atomically(dataset: SelfPlayDataset, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output.name or 'self-play'}-replace-",
        dir=str(output.parent),
    ) as temp_dir:
        temp_output = Path(temp_dir) / "dataset"
        save_self_play_dataset(dataset, temp_output)
        output.mkdir(parents=True, exist_ok=True)
        for filename in (
            DEFAULT_DATASET_FILENAME,
            DEFAULT_METADATA_FILENAME,
            DEFAULT_GAMES_FILENAME,
        ):
            os.replace(temp_output / filename, output / filename)

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
                    seed=task.generation_args.seed,
                    start_game_index=task.start_game,
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
                save_self_play_shard(dataset, task.shard_output)
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
    return _scriptutil.directory_size(directory)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate vibechess self-play samples.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/selfplay/smoke"),
        help="dataset output directory; appends when a complete dataset already exists",
    )
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
    initial_config = _self_play_config(generation_args, games=args.games, seed=args.seed)
    existing_dataset = _load_existing_dataset_for_append(parser, args.output)
    existing_game_count = 0 if existing_dataset is None else existing_dataset.metadata.game_count
    requested_settings = _requested_generation_settings(generation_args, initial_config)
    if existing_dataset is not None:
        _validate_append_request(
            parser,
            existing_dataset,
            requested_settings,
            checkpoint_id,
        )
    run_config = _self_play_config(
        generation_args,
        games=args.games,
        seed=args.seed,
        start_game_index=existing_game_count,
    )
    final_config = _self_play_config(
        generation_args,
        games=existing_game_count + args.games,
        seed=args.seed,
    )

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
                progress_reporter.start(args, start_game_offset=existing_game_count)
            chunk_results: list[ChunkGenerationResult] = []
            dataset: SelfPlayDataset | None = None
            dataset_metadata: SelfPlayMetadata | None = None
            if args.workers == 1 or args.games == 1:
                inference = None
                if generation_args.label_source == LABEL_SOURCE_NEURAL:
                    inference = _build_inference(generation_args)
                dataset = generate_self_play_dataset(
                    inference,
                    config=run_config,
                    progress=progress_reporter.game_completed,
                )
            else:
                workers = min(args.workers, args.games)
                local_chunks = _split_games(args.games, workers)
                chunks = [
                    (existing_game_count + start_game, games)
                    for start_game, games in local_chunks
                ]
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
                    chunk_results_with_start = sorted(
                        chunk_results_by_start,
                        key=lambda item: item[0],
                    )
                    chunk_results = [result for _start_game, result in chunk_results_with_start]
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
                    with profile_scope("dataset.load_shard_manifests", shards=len(chunk_results)):
                        shard_manifests = [
                            load_self_play_shard_manifest(
                                result.shard_output,
                                start_game=start_game,
                            )
                            for start_game, result in chunk_results_with_start
                        ]
                    progress_reporter.saving(args.output)
                    merge_output = (
                        args.output
                        if existing_dataset is None
                        else Path(shard_temp.name) / "additional"
                    )
                    additional_metadata = save_merged_self_play_shards(
                        shard_manifests,
                        merge_output,
                        config=run_config,
                        generation_settings_extra=parallel_settings,
                        expected_start_game=existing_game_count,
                    )
                    if existing_dataset is None:
                        dataset_metadata = additional_metadata
                    else:
                        additional_dataset = load_self_play_dataset(merge_output)
                        merged_dataset = append_self_play_dataset(
                            existing_dataset,
                            additional_dataset,
                            config=final_config,
                            generation_settings_extra=parallel_settings,
                        )
                        _save_self_play_dataset_atomically(merged_dataset, args.output)
                        dataset_metadata = merged_dataset.metadata
                finally:
                    shard_temp.cleanup()
            if dataset_metadata is None:
                if dataset is None:
                    raise RuntimeError("self-play generation produced no dataset")
                progress_reporter.saving(args.output)
                if existing_dataset is None:
                    save_self_play_dataset(dataset, args.output)
                    dataset_metadata = dataset.metadata
                else:
                    merged_dataset = append_self_play_dataset(
                        existing_dataset,
                        dataset,
                        config=final_config,
                    )
                    _save_self_play_dataset_atomically(merged_dataset, args.output)
                    dataset_metadata = merged_dataset.metadata
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
            progress_reporter.done_metadata(dataset_metadata)
    finally:
        progress_reporter.cleanup()
    print(
        " ".join(
            [
                f"output={args.output}",
                f"games={dataset_metadata.game_count}",
                f"samples={dataset_metadata.sample_count}",
                f"schema={dataset_metadata.schema_version}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
