#!/usr/bin/env python3
"""Profile PGN ingestion phases without writing dataset shards by default."""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vibechess import _scriptutil
from vibechess.engine.game import Game
from vibechess.engine.pgn import PgnParsedPly
from vibechess.engine.pgn_stream import (
    iter_pgn_records,
    parse_ingest_pgn_with_trace,
    pgn_has_fen_setup,
)
from vibechess.nn.encode import legal_move_mask_from_board_moves_np
from vibechess.nn.pgn_dataset import (
    SUPPORTED_PGN_RESULTS,
    PgnIngestConfig,
    _one_hot_policy_row,
    _TrainingReplayState,
    ingest_pgn_dataset,
)

BenchmarkMode = Literal["dry-run", "full-write"]
ReportFormat = Literal["markdown", "json"]


@dataclass(slots=True)
class PgnIngestBenchmark:
    """Aggregated timings and counters for a PGN ingestion benchmark run."""

    input_path: Path
    strict: bool
    skip_fen: bool
    max_records: int | None
    max_games: int | None
    mode: BenchmarkMode = "dry-run"
    output_dir: Path | None = None
    output_bytes: int | None = None
    output_files: int | None = None
    shard_samples: int | None = None
    timings: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_time(self, name: str, elapsed: float) -> None:
        self.timings[name] += elapsed

    def incr(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    @property
    def total_timed_seconds(self) -> float:
        return sum(self.timings.values())

    def to_dict(self) -> dict[str, object]:
        elapsed = self.total_timed_seconds
        counters = dict(sorted(self.counters.items()))
        samples = counters.get("samples", 0)
        records = counters.get("records_read", 0)
        games_accepted = counters.get("games_accepted", 0)
        games_skipped = counters.get("games_skipped", _count_prefixed(counters, "games_skipped_"))
        shards = counters.get("shards", 0)
        return {
            "mode": self.mode,
            "input_path": str(self.input_path),
            "output_dir": None if self.output_dir is None else str(self.output_dir),
            "output_bytes": self.output_bytes,
            "output_files": self.output_files,
            "strict": self.strict,
            "skip_fen": self.skip_fen,
            "max_records": self.max_records,
            "max_games": self.max_games,
            "limits": {"max_records": self.max_records, "max_games": self.max_games},
            "shard_samples": self.shard_samples,
            "elapsed_seconds": elapsed,
            "records_per_second": _rate(records, elapsed),
            "samples_per_second": _rate(samples, elapsed),
            "records_read": records,
            "games_accepted": games_accepted,
            "games_skipped": games_skipped,
            "samples": samples,
            "shards": shards,
            "counters": counters,
            "timings": dict(sorted(self.timings.items())),
            "timing_shares": {
                name: seconds / elapsed if elapsed > 0 else 0.0
                for name, seconds in sorted(self.timings.items())
            },
        }


def benchmark_pgn_ingest(
    *,
    input_path: Path,
    max_records: int | None = 100,
    max_games: int | None = None,
    strict: bool = False,
    skip_fen: bool = True,
) -> PgnIngestBenchmark:
    """Run a phase-timed PGN ingestion dry run.

    The benchmark mirrors the expensive per-game/per-ply work in
    ``ingest_pgn_dataset``: record streaming, FEN tag screening, sanitizer/parser,
    validation, replay legality checks, position encoding, legal masks, one-hot
    policy allocation, and move application. It intentionally does not retain all
    samples or write NPZ shards, so timings isolate CPU/parser/encoder costs from
    bulk disk output.
    """
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive when provided")
    if max_games is not None and max_games < 1:
        raise ValueError("max_games must be positive when provided")

    report = PgnIngestBenchmark(
        input_path=input_path.expanduser(),
        strict=strict,
        skip_fen=skip_fen,
        max_records=max_records,
        max_games=max_games,
    )
    starting_fen = Game.new().to_fen()
    records = iter_pgn_records(report.input_path)

    while max_records is None or report.counters["records_read"] < max_records:
        if max_games is not None and report.counters["games_accepted"] >= max_games:
            break

        start = time.perf_counter()
        try:
            record = next(records)
        except StopIteration:
            report.add_time("read_records", time.perf_counter() - start)
            break
        report.add_time("read_records", time.perf_counter() - start)
        report.incr("records_read")

        if skip_fen:
            start = time.perf_counter()
            has_fen = pgn_has_fen_setup(record.text)
            report.add_time("fen_tag_scan", time.perf_counter() - start)
            if has_fen:
                report.incr("games_skipped_fen")
                continue

        start = time.perf_counter()
        try:
            traced = parse_ingest_pgn_with_trace(record.text, strict=strict)
        except ValueError:
            report.add_time("parse_sanitize", time.perf_counter() - start)
            report.incr("games_skipped_parse")
            continue
        report.add_time("parse_sanitize", time.perf_counter() - start)

        pgn = traced.game
        start = time.perf_counter()
        valid = pgn.initial_game.to_fen() == starting_fen and pgn.result in SUPPORTED_PGN_RESULTS
        report.add_time("validation", time.perf_counter() - start)
        if not valid:
            report.incr("games_skipped_validation")
            continue

        try:
            samples = _replay_and_encode_game(traced.plies, report)
        except ValueError:
            report.incr("games_skipped_replay")
            continue
        if samples == 0:
            report.incr("games_skipped_empty")
            continue

        start = time.perf_counter()
        _result_values(pgn.result, samples)
        report.add_time("outcome_labels", time.perf_counter() - start)

        report.incr("games_accepted")
        report.incr("samples", samples)

    return report


def benchmark_pgn_ingest_full_write(
    *,
    input_path: Path,
    output_dir: Path | None = None,
    max_records: int | None = 100,
    max_games: int | None = None,
    strict: bool = False,
    skip_fen: bool = True,
    shard_samples: int = 50_000,
) -> PgnIngestBenchmark:
    """Run the real PGN importer and measure shard writing/compression costs.

    Unlike ``benchmark_pgn_ingest``, this mode calls ``ingest_pgn_dataset`` and
    therefore includes sparse NPZ creation, compression, manifest writing, and
    games.jsonl output. If ``max_records`` is set, the benchmark first copies at
    most that many raw PGN records into a temporary fixture file so the importer
    can run unchanged while still honoring the record limit.
    """
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive when provided")
    if max_games is not None and max_games < 1:
        raise ValueError("max_games must be positive when provided")
    if shard_samples < 1:
        raise ValueError("shard_samples must be at least 1")
    if not skip_fen:
        raise ValueError("full-write mode uses ingest_pgn_dataset(), which skips FEN records")

    report = PgnIngestBenchmark(
        input_path=input_path.expanduser(),
        strict=strict,
        skip_fen=skip_fen,
        max_records=max_records,
        max_games=max_games,
        mode="full-write",
        output_dir=None if output_dir is None else output_dir.expanduser(),
        shard_samples=shard_samples,
    )

    with ExitStack() as stack:
        ingest_input = report.input_path
        if max_records is not None:
            temp_input_root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            ingest_input = temp_input_root / "limited-input.pgn"
            start = time.perf_counter()
            limited_records = _write_limited_records(report.input_path, ingest_input, max_records)
            report.add_time("limit_records", time.perf_counter() - start)
            report.incr("records_limited", limited_records)

        if report.output_dir is None:
            temp_output_root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            actual_output_dir = temp_output_root / "pgn-benchmark-dataset"
            report.output_dir = actual_output_dir
        else:
            actual_output_dir = report.output_dir

        start = time.perf_counter()
        result = ingest_pgn_dataset(
            PgnIngestConfig(
                input_path=ingest_input,
                output_dir=actual_output_dir,
                max_games=max_games,
                shard_samples=shard_samples,
                strict=strict,
                skip_fen=skip_fen,
            )
        )
        report.add_time("ingest_pgn_dataset", time.perf_counter() - start)

        report.incr("records_read", result.games_read)
        report.incr("games_accepted", result.games_written)
        report.incr("games_skipped", result.games_skipped)
        report.incr("samples", result.samples)
        report.incr("shards", result.shards)
        report.output_bytes = _directory_size(actual_output_dir)
        report.output_files = _directory_file_count(actual_output_dir)

    return report


def _replay_and_encode_game(plies: object, report: PgnIngestBenchmark) -> int:
    if not isinstance(plies, tuple):
        raise TypeError("pgn trace plies must be a tuple")
    state = _TrainingReplayState.from_game(Game.new())
    samples = 0
    for ply in plies:
        if not isinstance(ply, PgnParsedPly):
            raise TypeError("pgn trace plies must contain PgnParsedPly values")

        start = time.perf_counter()
        valid_trace = (
            ply.board == state.board
            and ply.halfmove_clock == state.halfmove_clock
            and ply.fullmove_number == state.fullmove_number
            and ply.move in ply.legal_moves
        )
        report.add_time("validate_trace", time.perf_counter() - start)
        if not valid_trace:
            raise ValueError("PGN trace does not match replayed game")

        start = time.perf_counter()
        state.encode_position()
        report.add_time("encode_positions", time.perf_counter() - start)

        start = time.perf_counter()
        legal_move_mask_from_board_moves_np(ply.board, ply.legal_moves)
        report.add_time("legal_masks", time.perf_counter() - start)

        start = time.perf_counter()
        _one_hot_policy_row(ply.board, ply.move)
        report.add_time("one_hot_policies", time.perf_counter() - start)

        start = time.perf_counter()
        state.advance(ply.move)
        report.add_time("advance_replay_state", time.perf_counter() - start)
        samples += 1
    return samples


def _result_values(result: str, samples: int) -> list[float]:
    if result == "1/2-1/2":
        return [0.0 for _index in range(samples)]
    # This benchmark only needs to include allocation cost for labels. The exact
    # side-to-move sign pattern is validated by ingestion tests.
    return [1.0 for _index in range(samples)]


def _write_limited_records(input_path: Path, output_path: Path, max_records: int) -> int:
    records_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in iter_pgn_records(input_path):
            if records_written >= max_records:
                break
            if records_written > 0:
                handle.write("\n")
            handle.write(record.text.rstrip())
            handle.write("\n")
            records_written += 1
    return records_written


def _directory_size(path: Path) -> int:
    return _scriptutil.directory_size(path)


def _directory_file_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


def _count_prefixed(counters: Mapping[str, int], prefix: str) -> int:
    return sum(value for key, value in counters.items() if key.startswith(prefix))


def format_benchmark(report: PgnIngestBenchmark, *, output_format: ReportFormat) -> str:
    """Render a PGN ingestion benchmark report."""
    data = report.to_dict()
    if output_format == "json":
        return json.dumps(data, indent=2, sort_keys=True)
    if output_format != "markdown":
        raise ValueError(f"unsupported report format: {output_format}")

    counters = _expect_mapping(data["counters"])
    timings = _expect_mapping(data["timings"])
    elapsed = _expect_float(data["elapsed_seconds"])
    lines = [
        "# PGN Ingestion Benchmark",
        "",
        f"- mode: {data['mode']}",
        f"- input: `{data['input_path']}`",
        f"- output_dir: `{data['output_dir']}`",
        f"- output_bytes: {data['output_bytes']}",
        f"- output_files: {data['output_files']}",
        f"- strict: {data['strict']}",
        f"- skip_fen: {data['skip_fen']}",
        f"- max_records: {data['max_records']}",
        f"- max_games: {data['max_games']}",
        f"- elapsed_seconds: {_format_float(elapsed)}",
        f"- records_per_second: {_format_float(_expect_float(data['records_per_second']))}",
        f"- samples_per_second: {_format_float(_expect_float(data['samples_per_second']))}",
        "",
        "## Counters",
        "",
    ]
    for key, value in counters.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Phase timings", ""])
    for name, seconds_value in sorted(
        timings.items(), key=lambda item: _expect_float(item[1]), reverse=True
    ):
        seconds = _expect_float(seconds_value)
        share = seconds / elapsed if elapsed > 0 else 0.0
        lines.append(f"- {name}: {_format_float(seconds)}s ({share:.1%})")
    lines.append("")
    return "\n".join(lines)


def _expect_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return value


def _expect_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("expected numeric value")
    return float(value)


def _rate(count: int, elapsed: float) -> float:
    return _scriptutil.rate(count, elapsed)


def _format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("lichess_elite_2025-11.pgn"))
    parser.add_argument("--max-records", type=int, default=100, help="raw PGN records to inspect")
    parser.add_argument(
        "--max-games", type=int, default=0, help="accepted games to inspect; 0 unlimited"
    )
    parser.add_argument("--strict", action="store_true", help="disable ingestion sanitizer")
    parser.add_argument(
        "--include-fen", action="store_true", help="do not pre-skip FEN/SetUp games"
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--mode",
        choices=("dry-run", "full-write"),
        default="dry-run",
        help="dry-run keeps the historical phase benchmark; full-write runs the real importer",
    )
    parser.add_argument("--output", type=Path, default=None, help="optional report output path")
    parser.add_argument(
        "--dataset-output-dir",
        type=Path,
        default=None,
        help="full-write mode output directory; defaults to a temporary directory",
    )
    parser.add_argument(
        "--shard-samples",
        type=int,
        default=50_000,
        help="samples per shard for full-write mode",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=None,
        help="optional cProfile .prof output for drilling into hot functions",
    )
    args = parser.parse_args()

    if args.mode == "full-write" and args.include_fen:
        parser.error(
            "full-write mode uses ingest_pgn_dataset(), which does not support FEN ingestion"
        )

    max_records = None if args.max_records == 0 else args.max_records
    max_games = None if args.max_games == 0 else args.max_games
    profiler = cProfile.Profile() if args.profile_output is not None else None
    if profiler is not None:
        profiler.enable()
    if args.mode == "dry-run":
        report = benchmark_pgn_ingest(
            input_path=args.input,
            max_records=max_records,
            max_games=max_games,
            strict=args.strict,
            skip_fen=not args.include_fen,
        )
    else:
        report = benchmark_pgn_ingest_full_write(
            input_path=args.input,
            output_dir=args.dataset_output_dir,
            max_records=max_records,
            max_games=max_games,
            strict=args.strict,
            skip_fen=True,
            shard_samples=args.shard_samples,
        )
    if profiler is not None:
        profiler.disable()
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.profile_output)
    rendered = format_benchmark(report, output_format=args.format)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"wrote PGN ingestion benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
