# PGN Parser Single Legal Generation Worker Handoff

Implemented the approved parser fast-advance plan.

## Changed files

- `src/tinychess/engine/legal_moves.py`
  - Added `has_legal_move(board)` early-exit helper that mirrors `legal_moves()` legality filtering but returns on the first legal candidate.
- `src/tinychess/engine/pgn.py`
  - Replaced parser-internal `Game.play()` advancement in `_parse_pgn()` with private `_PgnParserState`.
  - `_PgnParserState` carries board, halfmove/fullmove counters, and repetition counts; it reuses the SAN legal tuple for pre-move terminal checks and move validation.
  - Preserved `Game.play()` outcome ordering for parser terminal rejection: no legal moves, fifty-move, repetition, insufficient material.
  - Switched SAN `+/#` validation and SAN formatting checkmate suffix generation to `has_legal_move(next_board)` instead of full `legal_moves(next_board)`.
  - Kept `parse_san`, `parse_pgn`, `parse_pgn_with_trace`, and trace data shapes stable.
- `tests/test_legal_moves.py`
  - Added parity coverage that `has_legal_move(board) == bool(legal_moves(board))` for start, castling, en-passant, promotion, pinned/check, checkmate, and stalemate-style positions.
- `tests/test_pgn.py`
  - Added monkeypatch count tests proving `parse_pgn()` performs one full `legal_moves()` generation per ply for normal and checkmate PGNs.
  - Added trace clock coverage for FEN starts with non-zero counters.
  - Added parser fast-state parity/rejection coverage for castling, en-passant, promotion, mate, repetition, fifty-move, and insufficient-material terminal cases.
- `docs/pgn-ingestion.md`
  - Documented the parser no-history advancement and clarified benchmark phase wording.
- `progress.md`
  - Updated with implementation, validation, and benchmark results.

## Validation

- `uv run pytest tests/test_pgn.py tests/test_pgn_stream.py tests/test_legal_moves.py tests/nn/test_pgn_dataset.py tests/test_pgn_ingest_benchmark.py`
  - Exit code: 0
  - Result: `94 passed in 1.93s`
- `uv run ruff check .`
  - Exit code: 0
  - Result: `All checks passed!`
- `uv run mypy`
  - Exit code: 0
  - Result: `Success: no issues found in 62 source files`
- `uv run pytest`
  - Exit code: 0
  - Result: `298 passed in 15.69s`

## Speed evidence

Recent pre-change baseline from the parent run on `lichess_elite_2025-11.pgn`, 100 accepted games / 9,693 samples:

- Dry-run: `16.98s`, `570.84 samples/sec`, `parse_sanitize` ~97.2%.
- Full-write: `17.63s`, `549.87 samples/sec`.

Post-change benchmarks on the same input and limits:

- Dry-run command:
  - `uv run python scripts/pgn_ingest_benchmark.py --input lichess_elite_2025-11.pgn --max-records 0 --max-games 100 --format json`
  - Result: `6.1887s`, `1,566.24 samples/sec`, `16.16 games/sec`, `parse_sanitize=5.7222s` / 92.46%.
  - Speedup vs baseline: ~2.74x elapsed, ~2.74x samples/sec.
- Full-write command:
  - `uv run python scripts/pgn_ingest_benchmark.py --input lichess_elite_2025-11.pgn --max-records 0 --max-games 100 --mode full-write --format json`
  - Result: `7.2827s`, `1,330.96 samples/sec`, `13.73 games/sec`, output `1.38 MB` / 1 shard.
  - Speedup vs baseline: ~2.42x elapsed, ~2.42x samples/sec.
- 1000-game dry-run command:
  - `uv run python scripts/pgn_ingest_benchmark.py --input lichess_elite_2025-11.pgn --max-records 0 --max-games 1000 --format json`
  - Result: `59.5137s`, `1,570.72 samples/sec`, `16.84 records/sec`, 1,000 accepted games / 93,479 samples, 2 parse skips.
  - Phase shares: `parse_sanitize` 92.44%, `legal_masks` 4.21%, `encode_positions` 1.51%, `advance_replay_state` 1.16%.

## Remaining risks / notes

- The required one full legal tuple per parsed ply remains because dense legal masks still need it; the duplicate parser `Game.play()` legal generations are removed.
- `_PgnParserState` duplicates a narrow subset of `Game.play()` clock/repetition updates. It is private and covered by parser/ingestion parity tests, but future engine rule changes should update this state too.
- `parse_sanitize` is still the dominant phase after the speedup, now mostly the required SAN legal tuple plus attack/check work. Next likely speedups are legal-move hot-path optimizations or bounded parser legal-move caching, not another ingestion replay change.
