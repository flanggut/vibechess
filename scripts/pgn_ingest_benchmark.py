#!/usr/bin/env python3
"""Profile PGN ingestion phases without writing dataset shards by default."""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from tinychess.engine.game import Game
from tinychess.engine.pgn_stream import iter_pgn_records, parse_ingest_pgn, pgn_has_fen_setup
from tinychess.nn.encode import (
    ACTION_SPACE_SIZE,
    encode_game,
    legal_move_mask_from_legal_moves,
    move_to_action_index,
)
from tinychess.nn.pgn_dataset import SUPPORTED_PGN_RESULTS

ReportFormat = Literal["markdown", "json"]


@dataclass(slots=True)
class PgnIngestBenchmark:
    """Aggregated timings and counters for a PGN ingestion dry run."""

    input_path: Path
    strict: bool
    skip_fen: bool
    max_records: int | None
    max_games: int | None
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
        samples = self.counters["samples"]
        records = self.counters["records_read"]
        return {
            "input_path": str(self.input_path),
            "strict": self.strict,
            "skip_fen": self.skip_fen,
            "max_records": self.max_records,
            "max_games": self.max_games,
            "elapsed_seconds": elapsed,
            "records_per_second": _rate(records, elapsed),
            "samples_per_second": _rate(samples, elapsed),
            "counters": dict(sorted(self.counters.items())),
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
            pgn = parse_ingest_pgn(record.text, strict=strict)
        except ValueError:
            report.add_time("parse_sanitize", time.perf_counter() - start)
            report.incr("games_skipped_parse")
            continue
        report.add_time("parse_sanitize", time.perf_counter() - start)

        start = time.perf_counter()
        valid = pgn.initial_game.to_fen() == starting_fen and pgn.result in SUPPORTED_PGN_RESULTS
        report.add_time("validation", time.perf_counter() - start)
        if not valid:
            report.incr("games_skipped_validation")
            continue

        try:
            samples = _replay_and_encode_game(pgn.moves, report)
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


def _replay_and_encode_game(moves: object, report: PgnIngestBenchmark) -> int:
    if not isinstance(moves, tuple):
        raise TypeError("pgn moves must be a tuple")
    game = Game.new()
    samples = 0
    for move in moves:
        start = time.perf_counter()
        legal = game.legal_moves
        is_legal = move in legal
        report.add_time("replay_legal_check", time.perf_counter() - start)
        if not is_legal:
            raise ValueError("PGN move is not legal in replayed game")

        start = time.perf_counter()
        np.asarray(encode_game(game), dtype=np.float32)
        report.add_time("encode_positions", time.perf_counter() - start)

        start = time.perf_counter()
        np.asarray(legal_move_mask_from_legal_moves(game, legal), dtype=np.float32)
        report.add_time("legal_masks", time.perf_counter() - start)

        start = time.perf_counter()
        policy = np.zeros((ACTION_SPACE_SIZE,), dtype=np.float32)
        policy[move_to_action_index(move, game.board)] = 1.0
        report.add_time("one_hot_policies", time.perf_counter() - start)

        start = time.perf_counter()
        game = game.play(move)
        report.add_time("play_moves", time.perf_counter() - start)
        samples += 1
    return samples


def _result_values(result: str, samples: int) -> list[float]:
    if result == "1/2-1/2":
        return [0.0 for _index in range(samples)]
    # This benchmark only needs to include allocation cost for labels. The exact
    # side-to-move sign pattern is validated by ingestion tests.
    return [1.0 for _index in range(samples)]


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
        f"- input: `{data['input_path']}`",
        f"- strict: {data['strict']}",
        f"- skip_fen: {data['skip_fen']}",
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
    if elapsed == 0:
        return math.inf
    return count / elapsed


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
    parser.add_argument("--output", type=Path, default=None, help="optional report output path")
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=None,
        help="optional cProfile .prof output for drilling into hot functions",
    )
    args = parser.parse_args()

    profiler = cProfile.Profile() if args.profile_output is not None else None
    if profiler is not None:
        profiler.enable()
    report = benchmark_pgn_ingest(
        input_path=args.input,
        max_records=None if args.max_records == 0 else args.max_records,
        max_games=None if args.max_games == 0 else args.max_games,
        strict=args.strict,
        skip_fen=not args.include_fen,
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
