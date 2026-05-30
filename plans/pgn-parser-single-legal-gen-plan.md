# Implementation Plan

## Goal
Reduce PGN parse/import full legal-move generation to one `legal_moves(board)` tuple per parsed ply by replacing parser-internal `Game.play()` replay with a strict no-history parser state and early-exit legal-existence checks where a full tuple is not needed.

## Tasks
1. **Add a legal-existence helper for suffix checks**: Implement an early-exit helper that answers whether the side to move has any legal move without constructing a full legal tuple.
   - File: `src/tinychess/engine/legal_moves.py`
   - Changes:
     - Add `has_legal_move(board: Board) -> bool` near `legal_moves()`.
     - Share as much logic as practical with `legal_moves()` while keeping behavior identical: iterate `pseudo_legal_moves(board)`, apply each candidate, and return `True` on the first candidate that does not leave `board.side_to_move` in check.
     - Do not change the public `legal_moves()` return order or semantics.
     - Exporting through `src/tinychess/engine/__init__.py` is optional; if tests can import from `tinychess.engine.legal_moves`, keep it module-level only to minimize API surface.
   - Acceptance:
     - `has_legal_move(board) == bool(legal_moves(board))` for start position, checkmate, stalemate, castling positions, en-passant positions, promotion positions, and pinned-piece positions.
     - Existing perft/legal-move tests still pass.

2. **Use early-exit helper for SAN checkmate suffix validation**: Stop `_san_suffix_matches()` from calling `legal_moves(next_board)` just to distinguish `+` from `#`.
   - File: `src/tinychess/engine/pgn.py`
   - Changes:
     - Import `has_legal_move` from `tinychess.engine.legal_moves`.
     - Change `_san_suffix_matches()` from `"#" if not legal_moves(next_board) else "+"` to `"#" if not has_legal_move(next_board) else "+"` after the existing `is_in_check(next_board, next_board.side_to_move)` check.
     - Keep `parse_san()` behavior and error messages broad-compatible (`ValueError` with current “not legal” path for bad suffixes).
   - Acceptance:
     - Existing `test_parse_san_requires_exact_checkmate_suffix` still passes.
     - New legal-generation count test for a checkmate PGN still sees exactly one full `legal_moves()` call per ply.

3. **Add a private no-history parser state**: Replace `_parse_pgn()`’s `current = current.play(move)` with a parser-local state that reuses the SAN legal tuple.
   - File: `src/tinychess/engine/pgn.py`
   - Changes:
     - Add a private dataclass, e.g. `_PgnParserState`, with:
       - `board: Board`
       - `halfmove_clock: int`
       - `fullmove_number: int`
       - `repetition_counts: dict[PositionKey, int]`
     - Add `from_game(cls, game: Game) -> _PgnParserState` to seed from `_initial_game_from_tags(tags)`.
     - Add `pre_move_outcome(legal: tuple[Move, ...]) -> Outcome | None` or equivalent private helper that matches `Game.determine_outcome()` ordering without recomputing `legal_moves()`:
       1. if `not legal`: return checkmate/stalemate using `is_in_check(self.board, self.board.side_to_move)`;
       2. if `halfmove_clock >= 100`: return fifty-move;
       3. if current repetition count is at least 3: return repetition;
       4. if `has_insufficient_material(self.board)`: return insufficient-material;
       5. else return `None`.
     - Add `advance_checked(move: Move, legal: tuple[Move, ...]) -> None`:
       - call `pre_move_outcome(legal)` after SAN resolution, before applying the move;
       - if terminal, raise `ValueError(f"cannot play move after game outcome: {outcome.reason.value}")` to match `Game.play()`’s broad behavior;
       - defensively verify `move in legal` and raise `ValueError(f"illegal move: {move}")` if it is not;
       - update board with `Board.apply_move(move)`;
       - update repetition key, halfmove clock, and fullmove number exactly like `Game.play()`.
     - Add private `_PositionKey`, `_position_key()`, and `_is_capture()` in `pgn.py` only if needed, or import private engine helpers only if the worker decides that is cleaner. Prefer local private helpers for a narrow parser change unless a small shared internal helper in `game.py` clearly reduces duplication without broadening public API.
     - Do not modify `PgnGame.final_game`; it may continue using `Game.play()` as the correctness/reference replay outside the parse hot path.
   - Acceptance:
     - `_parse_pgn()` no longer calls `Game.play()` per ply.
     - Trace entries still capture the board/clocks before the move and the same parser-computed legal tuple.
     - `parse_pgn()` and `parse_pgn_with_trace()` return types/signatures remain unchanged.

4. **Refactor `_parse_pgn()` to use the parser state safely**: Preserve current strict parse order and result handling while swapping state advancement.
   - File: `src/tinychess/engine/pgn.py`
   - Changes:
     - Initialize `initial_game = _initial_game_from_tags(tags)` and `state = _PgnParserState.from_game(initial_game)` instead of mutating `current: Game`.
     - For each movetext SAN token:
       - keep the existing `seen_result` and `RESULTS` handling unchanged;
       - call `_parse_san_with_legal(state.board, token)` first (important: existing behavior resolves SAN before `Game.play()` rejects terminal positions);
       - if tracing, append `PgnParsedPly(board=state.board, halfmove_clock=state.halfmove_clock, fullmove_number=state.fullmove_number, move=move, legal_moves=legal)`;
       - call `state.advance_checked(move, legal)`;
       - append `move` to `moves` after successful advance.
     - If the worker wants to avoid appending a trace before a terminal rejection, perform `state.advance_checked()` before `plies.append(...)` but preserve the same pre-move trace contents by storing local variables. Either ordering is okay because parse errors return no trace, but avoid appending `moves` before successful advance.
   - Acceptance:
     - For valid PGNs, `parse_pgn(text).final_game.to_fen()` matches the current `Game.play()` replay reference.
     - For invalid PGNs, trace mode and normal mode still reject the same records as `parse_ingest_pgn()` / `parse_ingest_pgn_with_trace()`.

5. **Add legal-generation count tests**: Prove the parser no longer performs duplicate full legal tuple generation.
   - File: `tests/test_pgn.py`
   - Changes:
     - Add a helper that monkeypatches both `tinychess.engine.pgn.legal_moves` and `tinychess.engine.game.legal_moves` with a counting wrapper around the original `tinychess.engine.legal_moves.legal_moves`.
     - Add `test_parse_pgn_uses_one_full_legal_generation_per_normal_ply()` using a non-check PGN such as `1. e4 e5 2. Nf3 Nc6 *`; assert count equals `len(parsed.moves)`.
     - Add `test_parse_pgn_uses_one_full_legal_generation_per_checkmate_ply()` using fool’s mate `1. f3 e5 2. g4 Qh4# 0-1`; assert count equals `len(parsed.moves)` so `_san_suffix_matches()` cannot regress to a full `legal_moves(next_board)` call.
     - If the count wrapper is too brittle because of direct imports, patch `src/tinychess/engine/pgn.py`’s imported `legal_moves` and `src/tinychess/engine/game.py`’s imported `legal_moves` aliases explicitly.
   - Acceptance:
     - Tests fail on the current committed implementation (roughly 3 calls/ply, plus suffix calls) and pass after Tasks 1-4.

6. **Add parser semantic parity/rejection tests for terminal states**: Lock down `Game.play()` pre-move outcome semantics now duplicated in parser state.
   - File: `tests/test_pgn.py`
   - Changes:
     - Add valid parity tests for normal play, castling, en-passant, promotion, checkmate, repetition-ending PGN, and fifty-move-ending helper/FEN PGN if representable.
     - Add rejection tests for a move after each practical terminal condition:
       - after checkmate: `1. f3 e5 2. g4 Qh4# 3. a3 *` should raise `ValueError`;
       - after threefold repetition: `1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. Nf3 *` should raise;
       - initial FEN with `halfmove_clock >= 100` and a legal SAN move should raise before applying that move;
       - initial FEN with insufficient material (kings only) and a legal king move should raise before applying that move.
     - For terminal rejection tests, assert broad `ValueError` and, where stable, match `"cannot play move after game outcome"` or the outcome reason string.
   - Acceptance:
     - New tests prove that parser fast state rejects moves after engine-recognized terminal outcomes without calling `Game.play()`.
     - Existing strict PGN rejection tests still pass.

7. **Update trace/ingestion parity tests if needed**: Ensure the no-history parser state did not alter downstream PGN import tensors or records.
   - Files: `tests/test_pgn_stream.py`, `tests/nn/test_pgn_dataset.py`
   - Changes:
     - Keep existing trace parity tests that compare `ply.board`, `ply.halfmove_clock`, `ply.fullmove_number`, `ply.legal_moves`, and moves against `Game.play()` replay.
     - Add or extend one trace test to include a position with non-zero halfmove/fullmove counters from FEN, because parser state now owns those counters.
     - Keep `test_ingest_pgn_dataset_trace_path_matches_legacy_replay_arrays` unchanged unless fixture expansion is useful; it should pass without schema changes.
   - Acceptance:
     - Dense PGN import arrays (`positions`, `legal_masks`, `mcts_policies`, `outcomes`) remain byte/equality identical to the legacy replay reference for existing fixtures.

8. **Update docs/benchmark text only if terminology changes**: Reflect that parse phase now uses a no-history parser state and one full legal tuple per ply.
   - File: `docs/pgn-ingestion.md`
   - Changes:
     - In the benchmark hotspot section, update wording from parser trace/legal reuse only to mention parser no-history advancement if it helps explain the new phase shares.
     - Do not document new PGN syntax or broaden sanitizer claims.
   - Acceptance:
     - Docs remain accurate: dense shard schema unchanged; strict parser boundary unchanged; dry-run `parse_sanitize` still contains SAN resolution/legal tuple generation.

9. **Run focused validation**: Verify parser, legal move, and dataset behavior before broader checks.
   - Files: no source file changes expected in this task.
   - Changes: run commands and capture results in worker handoff.
   - Acceptance:
     - `uv run pytest tests/test_pgn.py tests/test_pgn_stream.py tests/test_legal_moves.py tests/nn/test_pgn_dataset.py tests/test_pgn_ingest_benchmark.py`
     - `uv run ruff check .`
     - `uv run mypy`

10. **Benchmark before/after and report speedup**: Measure whether the parse bottleneck moved as expected.
    - Files: no source file changes expected unless benchmark output is explicitly requested as an artifact.
    - Changes:
      - Before editing (or from the known prior baseline if already edited), record current benchmark values:
        - `uv run python scripts/pgn_ingest_benchmark.py --input lichess_elite_2025-11.pgn --max-records 0 --max-games 100 --format json`
        - `uv run python scripts/pgn_ingest_benchmark.py --input lichess_elite_2025-11.pgn --max-records 0 --max-games 100 --mode full-write --format json`
      - After implementation, run the same commands.
      - If runtime allows, also run 1000 accepted games dry-run:
        - `uv run python scripts/pgn_ingest_benchmark.py --input lichess_elite_2025-11.pgn --max-records 0 --max-games 1000 --format json`
      - Optional cProfile call-count check:
        - `uv run python scripts/pgn_ingest_benchmark.py --input lichess_elite_2025-11.pgn --max-records 0 --max-games 50 --profile-output /tmp/pgn-single-legal.prof --format json`
        - inspect that `legal_moves` calls are near plies plus any non-parser validation calls, not ~3x plies.
    - Acceptance:
      - Worker reports previous baseline, new elapsed seconds, games/sec, samples/sec, `parse_sanitize` share, and speedup ratio.
      - Expected result: substantial improvement in `parse_sanitize` and full-write throughput; if not, worker reports profile evidence instead of guessing.

## Files to Modify
- `src/tinychess/engine/legal_moves.py` - add early-exit `has_legal_move()` and share legality logic where safe.
- `src/tinychess/engine/pgn.py` - add private no-history parser state, replace parser `Game.play()` advancement, and use `has_legal_move()` for SAN `#` suffix validation.
- `tests/test_pgn.py` - add legal-generation count tests and terminal/outcome parity/rejection tests.
- `tests/test_pgn_stream.py` - update/extend traced ingest parity if needed for non-default clocks/FEN behavior.
- `tests/nn/test_pgn_dataset.py` - keep/extend dense ingestion parity if parser trace changes require fixture coverage.
- `docs/pgn-ingestion.md` - update benchmark/parser-hotspot wording if terminology changes.

## New Files
- None expected. Keep this as an in-place parser/engine optimization.

## Dependencies
- Task 2 depends on Task 1 (`has_legal_move`).
- Tasks 3-4 depend on understanding `Game.play()` / `determine_outcome()` semantics and must preserve their pre-move terminal checks.
- Tasks 5-7 depend on Tasks 1-4 being implemented enough for tests to compile.
- Task 10 should run before and after code changes when possible; if pre-change was not captured, use the known recent baseline: 100 accepted games full-write `17.63s`, dry-run `16.98s`, `9,693` samples, `parse_sanitize ~97%`.

## Semantic Invariants
- `parse_san(board, san) -> Move`, `parse_pgn(text) -> PgnGame`, and `parse_pgn_with_trace(text) -> PgnGameTrace` signatures and return types do not change.
- Core PGN parser remains strict; sanitizer tolerance stays only in `parse_ingest_pgn*` through `sanitize_pgn_text()`.
- Dense PGN dataset schema remains unchanged: `positions`, `legal_masks`, `mcts_policies`, `outcomes`.
- Parser still resolves SAN against the full legal tuple for the current board; this tuple is reused for trace/legal masks.
- Parser still rejects a move after any engine-recognized terminal outcome that `Game.play()` would reject.
- At least one full `legal_moves(board)` tuple is still generated per parsed ply because dense legal masks require it; the goal is to remove duplicate full tuple generations, not remove the required tuple.
- Check/checkmate suffix validation may call `has_legal_move(next_board)` but must not call full `legal_moves(next_board)`.
- `PgnParsedPly.board`, `halfmove_clock`, `fullmove_number`, `move`, and `legal_moves` remain pre-move trace data.

## Risks
- Terminal outcome ordering is the main semantic risk. Match `determine_outcome()`: no legal moves check first, then fifty-move, repetition, insufficient material.
- Error text may differ if terminal checks move before SAN resolution. Avoid this by resolving SAN first, then checking terminal outcome with the already-computed legal tuple, matching current `parse_san()` then `Game.play()` order.
- Duplicating `_position_key()` and capture/clock logic can drift from `Game.play()`. Keep helpers tiny, private, and covered by parity tests; consider extracting a shared internal helper only if it stays narrow.
- `has_legal_move()` must handle castling, en-passant discovered check, promotions, pins, and adjacent kings exactly like `legal_moves()`.
- Legal-generation count tests can be brittle if they patch the wrong imported alias. Patch both `tinychess.engine.pgn.legal_moves` and `tinychess.engine.game.legal_moves`, and count only full tuple generation, not `has_legal_move()`.
- Benchmarks on the first 100 games can be noisy/opening-biased. Prefer 1000-game dry-run if runtime permits.

## Compact Worker Handoff
Implement the next Python-first PGN parser bottleneck optimization. Keep `parse_san`, `parse_pgn`, `parse_pgn_with_trace`, strict PGN semantics, ingestion sanitizer boundaries, and dense dataset schema stable. Add `has_legal_move(board)` as an early-exit helper and use it for SAN `#` suffix validation. Replace `_parse_pgn()`’s per-ply `Game.play()` with a private no-history parser state that reuses the legal tuple already computed by `_parse_san_with_legal()`, preserves `Game.play()` pre-move terminal rejection semantics, and records the same trace boards/clocks/legal moves. Add tests proving one full `legal_moves()` generation per ply (including a checkmate suffix PGN), plus terminal rejection/parity tests for checkmate, repetition, fifty-move, insufficient material, castling, en-passant, promotion, trace parity, and dense ingestion arrays. Run focused tests, `ruff`, `mypy`, and 100-game dry-run/full-write benchmarks before/after (1000-game dry-run if feasible). Report speedup and any remaining bottleneck phase shares; stop if parser accepts/rejects PGNs differently from the current strict parser.
