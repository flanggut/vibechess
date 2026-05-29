# Component 4 Worker Handoff: Parser trace reuse for PGN ingestion

Implemented/repaired Component 4 from the dirty partial worktree.

## Changed files

- `src/tinychess/engine/pgn.py`
  - Added `PgnParsedPly` and `PgnGameTrace` trace carriers.
  - Refactored SAN resolution through `_parse_san_with_legal()` so parser callers can reuse the legal-move tuple computed for each SAN token.
  - Kept `parse_san(board, san) -> Move` and `parse_pgn(text) -> PgnGame` behavior stable.
  - Added `parse_pgn_with_trace(text) -> PgnGameTrace` for traced parser consumers.
- `src/tinychess/engine/pgn_stream.py`
  - Added `parse_ingest_pgn_with_trace(text, strict=False)` using the same sanitizer boundary as `parse_ingest_pgn()`.
- `src/tinychess/nn/pgn_dataset.py`
  - Switched PGN ingestion to parse records with trace data.
  - `_ShardBuilder.add_game()` now consumes traced boards/legal moves for positions, masks, policies, and replay validation, avoiding a second replay legal generation for sample construction.
  - Preserved dense shard schema and existing self-play shard compatibility.
- `scripts/pgn_ingest_benchmark.py`
  - Updated dry-run replay to use traced plies and report `trace_legal_reuse` timing.
- `tests/test_pgn.py`
  - Added trace parity coverage for moves, tags, results, per-ply boards, clocks, and legal moves.
- `tests/test_pgn_stream.py`
  - Added strict and sanitized traced-ingest parity tests.
  - Added rejection parity coverage for strict comments, result mismatch, wrong check suffix, illegal move, and tokens after result.
- `tests/nn/test_pgn_dataset.py`
  - Added dense ingestion parity comparing trace-based import outputs against a legacy-style parse/replay/legal-generation reference for positions, legal masks, policies, and outcomes.
- `docs/pgn-ingestion.md`
  - Documented parser trace/legal reuse in the dense-compatible PGN import path.

## Validation

- `uv run pytest tests/test_pgn.py tests/test_pgn_stream.py tests/nn/test_pgn_dataset.py tests/test_pgn_ingest_benchmark.py`
  - Exit code: 0
  - Result: `58 passed in 1.55s`
- `uv run pytest`
  - Exit code: 0
  - Result: `277 passed in 15.68s`
- `uv run ruff check .`
  - Exit code: 0
  - Result: `All checks passed!`
- `uv run mypy`
  - Exit code: 0
  - Result: `Success: no issues found in 62 source files`
- Tiny dry-run benchmark smoke:
  - Command: `uv run python scripts/pgn_ingest_benchmark.py --input "$tmpdir/games.pgn" --max-records 2 --format json`
  - Exit code: 0
  - Output excerpt: `mode=dry-run`, `records_read=2`, `games_accepted=2`, `samples=4`, timings include `trace_legal_reuse`.
- Tiny full-write benchmark smoke:
  - Command: `uv run python scripts/pgn_ingest_benchmark.py --input "$tmpdir/games.pgn" --max-records 2 --mode full-write --format json`
  - Exit code: 0
  - Output excerpt: `mode=full-write`, `records_read=2`, `games_accepted=2`, `samples=4`, `shards=1`.

## Notes and risks

- No trace parser acceptance/rejection divergence was observed in the added parity/rejection coverage.
- Traces are produced and consumed one PGN record at a time; the importer does not retain traces across records.
- `Game.play()` is still used inside core PGN parsing to preserve existing parser validation semantics; Component 4 only removes duplicate replay legal generation in PGN sample construction.
- Dense shard keys/schema remain unchanged: `positions`, `legal_masks`, `mcts_policies`, `outcomes`.
