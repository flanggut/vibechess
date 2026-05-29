# Component 1 Worker Handoff: Measurement and benchmark coverage

Implemented benchmark coverage only.

## Changed files

- `scripts/pgn_ingest_benchmark.py`
  - Kept dry-run as the default mode.
  - Added `--mode full-write`, `--dataset-output-dir`, and `--shard-samples`.
  - Full-write mode calls `ingest_pgn_dataset()` so NPZ compression, manifest writing, and `games.jsonl` output are included.
  - Added stable JSON fields shared by dry-run and full-write reports: mode, limits, accepted/skipped counters, records/samples/shards, rates, output bytes/files, timings, and timing shares.
- `tests/test_pgn_ingest_benchmark.py`
  - Added fixture-based unit coverage for dry-run JSON structure.
  - Added fixture-based full-write coverage asserting shard output and output-size reporting.
  - Added CLI JSON smoke coverage for both modes.
- `docs/pgn-ingestion.md`
  - Documented that dry-run excludes compression/output and that full-write mode is the authoritative end-to-end throughput measurement.
- `progress.md`
  - Maintained component progress and validation notes.

## Validation

- `uv run pytest tests/test_pgn_ingest_benchmark.py tests/nn/test_pgn_dataset.py`
  - Exit code: 0
  - Result: `12 passed in 1.37s`
- `uv run ruff check scripts/pgn_ingest_benchmark.py tests/test_pgn_ingest_benchmark.py`
  - Exit code: 0
  - Result: `All checks passed!`
- `uv run mypy scripts/pgn_ingest_benchmark.py tests/test_pgn_ingest_benchmark.py`
  - Exit code: 0
  - Result: `Success: no issues found in 2 source files`
- Tiny benchmark CLI smoke with two PGN records:
  - Dry-run command: `uv run python scripts/pgn_ingest_benchmark.py --input "$tmpdir/games.pgn" --max-records 2 --format json`
    - Exit code: 0
    - Output excerpt: `mode=dry-run`, `records_read=2`, `games_accepted=2`, `samples=4`, `shards=0`
  - Full-write command: `uv run python scripts/pgn_ingest_benchmark.py --input "$tmpdir/games.pgn" --max-records 2 --mode full-write --dataset-output-dir "$tmpdir/dataset" --shard-samples 2 --format json`
    - Exit code: 0
    - Output excerpt from this run: `mode=full-write`, `records_read=2`, `games_accepted=2`, `samples=4`, `shards=2`, `output_bytes=4653`, `output_files=7`

## Notes and risks

- Requested `context.md` and `plan.md` were not present in the checkout; implementation followed `plans/pgn-import-optimization-components.md`.
- Full-write mode honors `--max-records` by copying the limited raw PGN records to a temporary input file before invoking `ingest_pgn_dataset()` unchanged. The report includes this as a small `limit_records` timing when a record limit is used.
- No parser semantics, importer behavior, dataset schema, training code, or Swift code were changed.
