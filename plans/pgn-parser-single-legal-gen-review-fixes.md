# PGN Parser Single Legal Generation Review Fixes

The final async worker was interrupted after completing edits, so this parent handoff records the accepted review incorporation and validation.

## Reviewer feedback addressed

- Added clarifying documentation/comments for the private parser fast state so future changes to `Game.play()` / `determine_outcome()` keep `_PgnParserState` in sync.
- Clarified docs wording to say the parser now performs one full legal tuple generation per ply while `+/#` suffix validation may still use `has_legal_move()` for legal-existence checks.
- Kept the optimization narrowly scoped: no schema changes, no sanitizer broadening, no multiprocessing, no Swift.

## Reviewer feedback deferred

- Replacing `_PgnParserState` duplication with a shared engine helper is deferred; the current private duplication is covered by parity/rejection tests and avoids expanding engine API in this milestone.
- Generator-style pseudo-legal iteration for `has_legal_move()` is deferred as a later legal-move hot-path optimization.

## Validation from parent session

- `uv run ruff check .`
  - Exit code: 0
  - Result: `All checks passed!`
- `uv run mypy`
  - Exit code: 0
  - Result: `Success: no issues found in 62 source files`
- `uv run pytest tests/test_pgn.py tests/test_pgn_stream.py tests/test_legal_moves.py tests/nn/test_pgn_dataset.py tests/test_pgn_ingest_benchmark.py`
  - Exit code: 0
  - Result: `95 passed in 2.02s`
- `uv run pytest`
  - Exit code: 0
  - Result: `299 passed in 15.88s`

## Remaining notes

- The implementation removes duplicate parser `Game.play()` legal generations and asserts one full `legal_moves()` tuple per parsed ply.
- `+/#` SAN suffix checks still perform `has_legal_move()` on checking moves to distinguish check from mate without materializing a full legal tuple.
