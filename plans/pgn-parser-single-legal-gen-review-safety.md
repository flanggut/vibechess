## Review

- Correct: Strict parser entry points and trace API shape remain compatible. `parse_san()` still returns only `Move`, `parse_pgn()` still returns `PgnGame`, and `parse_pgn_with_trace()` remains additive (`src/tinychess/engine/pgn.py:172-185`). The parser now advances through `_PgnParserState` after resolving SAN with the same legal tuple (`src/tinychess/engine/pgn.py:225-237`), so public return types are unchanged.
- Correct: Sanitizer boundary is preserved. `parse_ingest_pgn()` and `parse_ingest_pgn_with_trace()` still choose `text if strict else sanitize_pgn_text(text)` and delegate to the strict core parser (`src/tinychess/engine/pgn_stream.py:81-88`). Existing rejection-parity coverage remains in `tests/test_pgn_stream.py:66-83`.
- Correct: Terminal-state behavior is explicitly mirrored before parser advancement. `_PgnParserState.pre_move_outcome()` checks no legal moves, fifty-move, repetition, and insufficient material in the same order as `Game.determine_outcome()` (`src/tinychess/engine/pgn.py:526-540`), and `advance_checked()` rejects moves after such outcomes before applying the move (`src/tinychess/engine/pgn.py:542-564`). Tests cover mate/no-legal continuation, repetition, fifty-move, and insufficient-material rejections (`tests/test_pgn.py:180-219`).
- Correct: Check/checkmate suffix behavior no longer materializes a full legal tuple but still uses legal filtering via `has_legal_move()` (`src/tinychess/engine/pgn.py:166-168`, `src/tinychess/engine/pgn.py:500-505`; helper in `src/tinychess/engine/legal_moves.py:57-64`). Parity coverage asserts `has_legal_move(board) is bool(legal_moves(board))` on representative positions (`tests/test_legal_moves.py:28-45`).
- Correct: Dataset schema is not touched by this worker. The diff is limited to parser/legal-move internals, docs, progress, and parser/legal tests; no `src/tinychess/nn/pgn_dataset.py` changes appear in the current diff. Dense shard preservation from the previous import work remains intact.
- Correct: No generated datasets/checkpoints appear in `git status`; only source/docs/tests/progress/plans are modified or untracked.
- Fixed: None; review-only task.
- Blocker: None found for safety/boundary scope.
- Note: `_PgnParserState` duplicates the narrow clock/repetition/capture update logic from `Game.play()` (`src/tinychess/engine/pgn.py:551-581`). This is inside the strict parser and covered by parity/rejection tests, but it is a maintenance hotspot if outcome semantics change. Smallest safe follow-up: add a short comment on `_PgnParserState` that changes to `Game.play()`/`determine_outcome()` must keep this parser state in sync, or extract a shared internal advance/outcome helper later.
- Note: The “one full legal tuple per ply” tests count calls to `legal_moves()` through monkeypatching (`tests/test_pgn.py:65-94`), but `+/#` validation still calls `has_legal_move()` and thus may run pseudo-legal early-exit work on checking moves. This is consistent with the optimization goal as implemented, but benchmark/report wording should continue to say one full legal-move tuple rather than one total legality-related pass.

Validation run during review:

- `uv run pytest tests/test_pgn.py tests/test_pgn_stream.py tests/test_legal_moves.py tests/nn/test_pgn_dataset.py tests/test_pgn_ingest_benchmark.py`
  - Exit code: 0
  - Result: `94 passed in 1.78s`
