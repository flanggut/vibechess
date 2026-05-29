## Review

- Blockers: None found for the current targeted PGN import optimization diff.

- Concerns:
  - `src/tinychess/nn/pgn_dataset.py:274-282` appends position/mask/policy samples before all replay validation for the game has completed, while `ingest_pgn_dataset()` catches `ValueError` and skips the game at `src/tinychess/nn/pgn_dataset.py:163-167`. With internally produced traces this should not fire, but if a replay divergence or private trace misuse raises after one or more plies, the builder retains partial samples without matching outcomes. Smallest safe fix: stage samples/outcomes/games in local lists inside `_ShardBuilder.add_game()` and commit them only after the loop succeeds, or roll back list lengths in an `except ValueError` block.

- Suggestions:
  - SAN trace parity looks correct: `parse_san()` and both PGN parse paths share `_parse_san_with_legal()` (`src/tinychess/engine/pgn.py:171-201`), and `parse_pgn_with_trace()` records the same pre-move board/clocks/move/legal tuple immediately before `current.play(move)` (`src/tinychess/engine/pgn.py:224-235`). Added tests compare traced moves/legal moves to normal parse/replay in `tests/test_pgn.py` and sanitized/strict ingest parity in `tests/test_pgn_stream.py`.
  - Legal move reuse in ingestion looks coherent: `ingest_pgn_dataset()` now parses traced PGN once (`src/tinychess/nn/pgn_dataset.py:151-156`), `_ShardBuilder.add_game()` validates trace state then uses `ply.legal_moves` for masks and `ply.board` for policy indexing (`src/tinychess/nn/pgn_dataset.py:274-282`). This preserves the old tensor semantics when the trace is parser-produced.
  - Tensor/sample correctness is covered by direct parity: NumPy encoders/masks mirror MLX helpers (`src/tinychess/nn/encode.py:134-155`, `src/tinychess/nn/encode.py:276-290`) and tests assert array equality in `tests/nn/test_encode.py`. Dataset trace path is compared against a legacy parse/replay/legal-generation reference in `tests/nn/test_pgn_dataset.py:84-101` and helper code at `tests/nn/test_pgn_dataset.py:251-297`.
  - Outcome/final-record replay state appears equivalent to `Game.play()` for standard import cases: `_TrainingReplayState.advance()` mirrors board apply, capture detection, halfmove/fullmove counters, repetition counts, and move history (`src/tinychess/nn/pgn_dataset.py:360-374`; reference `src/tinychess/engine/game.py:100-135`). Tests include normal, mate, castling, en-passant, and promotion coverage in `tests/nn/test_pgn_dataset.py:103-166`.

Note: `/Users/flanggut/dev/tinychess/plan.md` was not present when inspected; `progress.md` indicates the same missing-plan state and records Component 4 validation as complete.
