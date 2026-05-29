# PGN Import Optimization Review Fixes

Incorporated accepted reviewer feedback after the quick correctness/safety/complexity review pass.

## Findings addressed

- **Partial sample pollution on rejected traces** (`src/tinychess/nn/pgn_dataset.py`)
  - `_ShardBuilder.add_game()` now stages positions, legal masks, policies, sides, and replay state in local lists and commits them to the builder only after the entire traced game validates successfully.
  - Added `test_shard_builder_rejects_bad_trace_without_partial_samples()` to lock this behavior.

- **Replay outcome coverage gap** (`tests/nn/test_pgn_dataset.py`)
  - Added a normal-start threefold repetition PGN ingest/load test that records `repetition`.
  - Added helper-level parity coverage for a fifty-move outcome after a non-pawn/non-capture move.

- **Ambiguous replay-state API name** (`src/tinychess/nn/pgn_dataset.py`)
  - Renamed `_TrainingReplayState.to_game()` to `to_outcome_game()` and documented that it returns a minimal outcome-check snapshot, not a full-history replay game.

- **Misleading dry-run phase name** (`scripts/pgn_ingest_benchmark.py`, `docs/pgn-ingestion.md`)
  - Renamed benchmark timing from `trace_legal_reuse` to `validate_trace`.
  - Updated docs to clarify that `parse_sanitize` includes SAN/legal generation and `validate_trace` is only the cheap consistency check before reuse.

## Findings deferred

- Extracting `_TrainingReplayState` mechanics into an engine-level helper: deferred because it expands public/internal engine surface beyond the approved Components 1-4 scope. New parity tests cover the duplicated clock/repetition behavior.
- Removing all private helper imports from `scripts/pgn_ingest_benchmark.py`: deferred as optional cleanup. The benchmark remains script-internal and mirrors importer phases intentionally; broader public helper extraction can be considered later.
- Adding a public board-based legal-mask helper to deduplicate `_legal_move_mask_np`: deferred as minor cleanup outside the current accepted fixes.
- Separating full-write `limit_records` setup time from throughput: deferred; current JSON exposes `limit_records` separately and docs now emphasize benchmark mode behavior.

## Validation

- `uv run pytest tests/nn/test_pgn_dataset.py tests/test_pgn.py tests/test_pgn_stream.py tests/test_pgn_ingest_benchmark.py tests/nn/test_encode.py`
  - Exit code: 0
  - Result: `93 passed in 7.93s`
- `uv run ruff check .`
  - Exit code: 0
  - Result: `All checks passed!`
- `uv run mypy`
  - Exit code: 0
  - Result: `Success: no issues found in 62 source files`
- `uv run pytest`
  - Exit code: 0
  - Result: `280 passed in 15.56s`
- Tiny dry-run benchmark smoke with two PGN records
  - Exit code: 0
  - Result: `records_read=2`, `games_accepted=2`, `samples=4`, timings include `validate_trace`.
- Tiny full-write benchmark smoke with two PGN records
  - Exit code: 0
  - Result: `records_read=2`, `games_accepted=2`, `samples=4`, `shards=1`, `output_files=4`.

## Remaining risks

- The dense shard format is intentionally preserved; sparse PGN storage remains behind the later explicit decision gate.
- `Game.play()` is still used by core PGN parsing to preserve existing parser validation semantics; the implemented optimization removes duplicate ingestion replay/legal work, not parser legality work.
