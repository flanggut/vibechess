#!/usr/bin/env python3
"""Convert PGN games into tinychess training dataset shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tinychess.nn.pgn_dataset import PgnIngestConfig, PgnIngestProgress, ingest_pgn_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("~/data/chess/lichess_elite_2025-11.pgn").expanduser(),
        help="input PGN file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/selfplay/pgn-ingest"),
        help="output directory for manifest and shards",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="maximum accepted games to write; 0 means unlimited",
    )
    parser.add_argument(
        "--shard-samples",
        type=int,
        default=50_000,
        help="target maximum samples per shard before starting a new shard",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="disable PGN sanitizer and skip records rejected by the strict parser",
    )
    parser.add_argument(
        "--progress-every-games",
        type=int,
        default=100,
        help="print progress to stderr every N accepted games; 0 disables progress",
    )
    args = parser.parse_args()

    if args.max_games < 0:
        parser.error("--max-games must be non-negative")
    if args.shard_samples < 1:
        parser.error("--shard-samples must be at least 1")
    if args.progress_every_games < 0:
        parser.error("--progress-every-games must be non-negative")

    def print_progress(progress: PgnIngestProgress) -> None:
        print(
            " ".join(
                [
                    f"games_read={progress.games_read}",
                    f"games_written={progress.games_written}",
                    f"games_skipped={progress.games_skipped}",
                    f"samples={progress.samples}",
                    f"shards={progress.shards}",
                ]
            ),
            file=sys.stderr,
        )

    progress_every_games = args.progress_every_games or None
    result = ingest_pgn_dataset(
        PgnIngestConfig(
            input_path=args.input,
            output_dir=args.output,
            max_games=None if args.max_games == 0 else args.max_games,
            shard_samples=args.shard_samples,
            strict=args.strict,
            skip_fen=True,
        ),
        progress=print_progress if progress_every_games is not None else None,
        progress_every_games=progress_every_games,
    )
    print(
        " ".join(
            [
                f"output={result.output_dir}",
                f"manifest={result.manifest_path}",
                f"shards={result.shards}",
                f"games_read={result.games_read}",
                f"games_written={result.games_written}",
                f"games_skipped={result.games_skipped}",
                f"samples={result.samples}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
