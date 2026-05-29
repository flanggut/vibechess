# Component 3 Worker Handoff: Lightweight ingestion replay state

Implemented Component 3.

## Changed files

- `src/tinychess/nn/pgn_dataset.py`
  - Added private `_TrainingReplayState` for PGN ingestion replay using `Board`, halfmove/fullmove clocks, played moves, and repetition counts.
  - Replaced `_ShardBuilder.add_game()` per-ply `Game.play()` with `legal_moves(state.board)` legality checks, NumPy position/mask/policy creation from the lightweight state, and `Board.apply_move()` advancement.
  - Preserved dense shard keys/schema and `SelfPlayGameRecord` compatibility with `load_self_play_dataset()` validation.
  - Kept parser behavior unchanged and did not reuse parser legal traces.
- `scripts/pgn_ingest_benchmark.py`
  - Updated dry-run replay/encode timing to mirror the lightweight replay path and report `advance_replay_state` instead of `play_moves`.
- `tests/nn/test_pgn_dataset.py`
  - Added load/parity coverage against normal `Game.play()` replay for normal, checkmate, castling, and en-passant PGNs.
  - Added helper-level promotion replay parity coverage.
- `tests/test_pgn_ingest_benchmark.py`
  - Updated the dry-run timing assertion to expect `advance_replay_state`.
- `progress.md`
  - Recorded Component 3 completion and validation.

## Validation

- `uv run pytest tests/nn/test_pgn_dataset.py tests/test_pgn.py tests/test_pgn_ingest_benchmark.py`
  - Exit code: 0
  - Result: `47 passed in 1.50s`
- `uv run pytest`
  - Exit code: 0
  - Result: `268 passed in 15.08s`
- `uv run ruff check .`
  - Exit code: 0
  - Result: `All checks passed!`
- `uv run mypy`
  - Exit code: 0
  - Result: `Success: no issues found in 62 source files`
- Tiny dry-run benchmark smoke:
  - Command: `uv run python scripts/pgn_ingest_benchmark.py --input "$tmpdir/games.pgn" --max-records 2 --format json`
  - Exit code: 0
  - Output excerpt: `mode=dry-run`, `records_read=2`, `games_accepted=2`, `samples=4`, timings include `advance_replay_state`.

## Notes and risks

- Dense shard compatibility is preserved; generated shards still load through existing self-play validation.
- `Game.play()` is still used by PGN parsing and validation paths; this component only removes the duplicate per-ply `Game.play()` from PGN sample generation/dry-run replay.
- Repetition/capture key logic is mirrored privately in `pgn_dataset.py` to keep engine public APIs unchanged.
