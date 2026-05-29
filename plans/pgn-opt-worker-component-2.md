# Component 2 Worker Handoff: NumPy-native encoding helpers

Implemented Component 2.

## Changed files

- `src/tinychess/nn/encode.py`
  - Added `encode_board_np()` and `encode_game_np()` returning NumPy `float32` tensors with the existing `[20, 8, 8]` layout.
  - Added `legal_move_mask_from_legal_moves_np()` returning dense NumPy `float32` action masks of length `ACTION_SPACE_SIZE` from precomputed legal moves.
  - Left existing MLX encoders and masks intact.
- `src/tinychess/nn/__init__.py`
  - Exported the new NumPy helpers.
- `src/tinychess/nn/pgn_dataset.py`
  - Switched PGN shard position/legal-mask construction to use the NumPy helpers directly.
  - Preserved dense shard keys/schema: `positions`, `legal_masks`, `mcts_policies`, `outcomes`.
- `scripts/pgn_ingest_benchmark.py`
  - Switched dry-run `encode_positions` and `legal_masks` benchmark phases to use the NumPy helpers.
- `tests/nn/test_encode.py`
  - Added exact parity coverage for NumPy position encoders against `np.asarray()` of the MLX encoders across start, en-passant/clocks, castling, and promotion positions.
  - Added exact parity coverage for NumPy legal masks against the existing public MLX mask cases.
- `docs/pgn-ingestion.md`
  - Noted that PGN import writes NumPy-native tensors while preserving dense self-play schema compatibility.
- `progress.md`
  - Recorded Component 2 completion and validation.

## Validation

- `uv run pytest tests/nn/test_encode.py tests/nn/test_pgn_dataset.py tests/test_pgn_ingest_benchmark.py`
  - Exit code: 0
  - Result: `44 passed in 7.58s`
- `uv run pytest`
  - Exit code: 0
  - Result: `263 passed in 15.42s`
- `uv run ruff check .`
  - Exit code: 0
  - Result: `All checks passed!`
- `uv run mypy`
  - Exit code: 0
  - Result: `Success: no issues found in 62 source files`
- Tiny dry-run benchmark smoke:
  - Command: `uv run python scripts/pgn_ingest_benchmark.py --input "$tmpdir/games.pgn" --max-records 2 --format json`
  - Exit code: 0
  - Output excerpt: `mode=dry-run`, `records_read=2`, `games_accepted=2`, `samples=4`; `encode_positions` and `legal_masks` timings were present.

## Notes and risks

- Requested `context.md` and `plan.md` are still absent in this checkout; implementation followed `plans/pgn-import-optimization-components.md` and the Component 1 handoff.
- No parser semantics, legal move generation, action-space metadata, tensor layout, training loader, or shard schema were changed.
