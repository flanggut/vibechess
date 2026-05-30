Inherited decisions:
- Python remains the correctness reference; Swift is optional acceleration only after Python-side waste has been reduced and benchmarks still justify it.
- Core PGN parsing stays bounded/strict. Public-dataset tolerance belongs only in the explicit ingestion sanitizer boundary.
- Dense PGN shard schema is preserved for this optimization line unless a separate schema decision approves sparse policy/mask storage.
- Commit `3c9e1fb` already removed the second ingestion replay/legal pass by adding NumPy-native encoding, lightweight ingestion replay, parser trace reuse, and full-write benchmarking.
- Current measured bottleneck is no longer shard writing or tensor construction: 100 accepted games produced 9,693 samples in 17.6s full-write / 17.0s dry-run, with `parse_sanitize`/SAN resolution at ~97% of dry-run time.

Diagnosis:
- The remaining bottleneck is mostly legal move generation inside PGN parsing, not string parsing or sanitizer regex work.
- A cProfile smoke on 50 accepted games showed the same shape:
  - `_parse_pgn` / `parse_ingest_pgn_with_trace`: ~25.5s cumulative under profiler.
  - `legal_moves`: 14,469 calls for 4,706 plies, i.e. about **3 legal-move generations per ply**.
  - `_parse_san_with_legal`: one `legal_moves(board)` per SAN token, which is still required while dense legal masks are preserved.
  - `Game.play()` in the parser accounts for most of the other two legal generations per ply: pre-move `outcome -> determine_outcome -> legal_moves` and then `self.legal_moves` for validation.
  - `is_in_check` / `is_square_attacked` dominate inside `legal_moves`; `_king_square`, `piece_at`, `make_square`, `validate_square`, and `occupied_squares` are hot allocation/lookup costs.
- Therefore the highest-consistency next move is to keep the parser strict but stop using full `Game.play()` inside the parser once SAN resolution has already produced the legal move tuple.

Drift / contradiction check:
- Do not switch to a SAN-only resolver that avoids full legal move generation while dense legal masks remain required. It may make parsing look faster but would just move the required legal generation back into mask construction, or worse silently drop legal-mask parity.
- Do not remove parser terminal/outcome checks merely because PGN result tags provide labels. Existing `parse_pgn()` semantics reject moves after engine-recognized terminal states via `Game.play()`; preserving that behavior needs explicit parity tests.
- Do not broaden SAN/PGN syntax while optimizing. `parse_ingest_pgn_with_trace(strict=False)` must remain sanitizer + strict parser, not a second permissive parser.
- Do not jump to Swift yet. The profile shows avoidable Python duplicate legal generation and hot-path Python allocation still exist.

Recommendation:

1. **Replace parser `Game.play()` with a checked no-history parser advance path.**
   - Expected impact: **~2x to 2.7x parser speedup**, likely reducing full import from ~13.7h toward ~5-7h before other changes.
   - Why: current parser does roughly 3 legal generations per ply. Dense masks require one full legal tuple per ply; the other two mostly come from `Game.play()` and can be removed without changing public APIs.
   - Approach:
     - In `src/tinychess/engine/pgn.py`, introduce a private parser state carrying `board`, `halfmove_clock`, `fullmove_number`, repetition counts, and moves.
     - After `_parse_san_with_legal()` returns `(move, legal)`, check the same pre-move terminal semantics using the already-computed `legal` tuple:
       - no legal moves => checkmate/stalemate terminal;
       - halfmove >= 100 => fifty-move terminal;
       - repetition count >= 3 => repetition terminal;
       - insufficient material => terminal.
     - If terminal, raise the same broad `ValueError` behavior as `Game.play()` would for a following move.
     - Validate `move in legal` defensively, then update board/clocks/repetition with `Board.apply_move()` and no immutable history tuple copies.
     - Keep `parse_san()` and `parse_pgn()` return types unchanged; keep traced ply contents unchanged.
   - Likely files:
     - `src/tinychess/engine/pgn.py`
     - possibly `src/tinychess/engine/game.py` if extracting a tiny internal shared helper for clock/repetition updates to avoid another duplicate of `Game.play()` logic
     - `tests/test_pgn.py`, `tests/test_pgn_stream.py`, `tests/nn/test_pgn_dataset.py`
     - `scripts/pgn_ingest_benchmark.py` only for reporting old/new timing if needed
   - Tests/benchmarks:
     - Existing PGN strict/rejection tests.
     - New parity tests comparing `parse_pgn()` before/after behavior on normal, mate, stalemate/no-legal, repetition, fifty-move, insufficient-material, castling, en-passant, and promotion cases.
     - Trace parity: boards, clocks, legal moves, moves, tags, results equal to current behavior.
     - Ingestion parity against current committed importer for a fixture.
     - Benchmark `--max-games 100` and preferably `1000` dry-run/full-write.
   - Risk: medium. The outcome check is the semantic trap; do not skip it. Prefer extracting shared clock/repetition helpers over copying logic in three places.

2. **Add a bounded legal-move cache for ingestion parsing.**
   - Expected impact: **10-30% on opening-heavy corpora**, possibly more for repeated public-dataset openings; low effect on unique middlegames.
   - Why: public PGN corpora share many early positions. `Board` is immutable/hashable enough for a bounded cache keyed by full board state (`squares`, side, castling, en-passant). Dense masks still need legal tuples, but repeated positions can reuse them.
   - Approach:
     - Do not globally change `legal_moves()` semantics first. Add an ingestion/parser-local cached wrapper, e.g. `_cached_legal_moves(board)` with `functools.lru_cache(maxsize=50_000..250_000)`.
     - Use it in `parse_ingest_pgn_with_trace()` or a dedicated traced-ingest parser path; be careful if also using it in core `parse_pgn()` because tests/perft may become cache-sensitive.
     - Expose cache stats in benchmark JSON if possible (`hits`, `misses`, `maxsize`) to prove value on the actual corpus.
   - Likely files:
     - `src/tinychess/engine/pgn.py` or `src/tinychess/engine/pgn_stream.py`
     - `scripts/pgn_ingest_benchmark.py`
     - tests around cache parity and no semantic changes
   - Tests/benchmarks:
     - Parse/ingest parity with cache enabled/disabled.
     - Memory smoke on 1000 games.
     - Benchmark first 100, 1000, and a later chunk of the corpus; opening-only wins can mislead.
   - Risk: low/medium. Bounded memory and no stale state are key. Do not use an unbounded global cache over a 27M-sample import.

3. **Optimize `legal_moves()` by avoiding repeated king-square scans during candidate legality checks.**
   - Expected impact: **15-25% legal-generation speedup** based on profile share; more after parser `Game.play()` duplication is removed because the single remaining legal generation becomes the dominant cost.
   - Why: `legal_moves()` applies each pseudo move and calls `is_in_check(next_board, moving_color)`, which rescans the board to find the moving king for every candidate. The moving king square is known: unchanged for non-king moves, `move.to_square` for king moves.
   - Approach:
     - Add internal helper `is_square_attacked(board, king_square, opponent)` is already available; add `is_in_check_at(board, color, king_square)` or inline in `legal_moves()`.
     - Compute the side-to-move king square once before looping pseudo-legal moves.
     - For each candidate, after `apply_move`, pass `move.to_square` if moving piece is king, else original king square.
     - Preserve public `is_in_check(board, color)` API.
   - Likely files:
     - `src/tinychess/engine/legal_moves.py`
     - `tests/test_legal_moves.py` / perft tests if present
   - Tests/benchmarks:
     - Full legal-move/perft test suite.
     - Compare `legal_moves(board)` exact tuples/sets for startpos, check, pins, castling, en-passant, promotion, discovered check.
     - PGN import benchmark after change.
   - Risk: low/medium. En-passant discovered check and king moves are the important edge cases.

4. **Make `is_square_attacked()` table/index based in hot paths.**
   - Expected impact: **20-40% legal-generation speedup** if done well; combine with item 3 for potentially significant gains.
   - Why: profile shows millions of calls through `make_square`, `validate_square`, `piece_at`, `_offset_square`, and `_ray_attacked`. Attack checks can use precomputed square-index tables and direct `board.squares[index]` access without changing board representation.
   - Approach:
     - Precompute knight attacker indices, king attacker indices, pawn attacker indices by color, and ray index lists for each square/direction.
     - Rewrite `is_square_attacked()` to iterate those tables and inspect `board.squares[int_index]` directly.
     - Keep function signature and semantics unchanged.
   - Likely files:
     - `src/tinychess/engine/legal_moves.py`
     - tests for attack detection and legal moves
   - Tests/benchmarks:
     - Existing full test suite plus perft depths.
     - Focused attack tests for board edges/corners, sliders blocked by own/enemy pieces, adjacent kings, pawns by color.
     - Microbenchmark `legal_moves(startpos)` / random midgames and PGN import 100/1000 games.
   - Risk: medium. Off-by-one/ray direction mistakes are easy. Keep the old implementation temporarily in tests as a reference if possible.

5. **Add early-exit legal-existence helper for checkmate suffix validation and outcome checks.**
   - Expected impact: **small now (~2-5%)**, more visible after item 1 removes duplicate parser legal generation.
   - Why: `_san_suffix_matches()` only needs to distinguish `+` from `#`, so it needs “does the opponent have any legal move?” not a full legal tuple. Parser terminal pre-checks also often only need empty/non-empty when not already carrying a legal tuple.
   - Approach:
     - Add `has_legal_move(board)` or `any_legal_move(board)` that returns on the first legal candidate.
     - Use it only where a full legal tuple is not needed, e.g. `_san_suffix_matches()` after `is_in_check(next_board, next_board.side_to_move)`.
   - Likely files:
     - `src/tinychess/engine/legal_moves.py`
     - `src/tinychess/engine/pgn.py`
   - Tests/benchmarks:
     - Check/mate SAN suffix tests, fool's mate, non-mate checks, stalemate/checkmate positions.
   - Risk: low/medium. Must handle castling/en-passant pins exactly like `legal_moves()`.

6. **Reduce hot allocation in board/legal move helpers.**
   - Expected impact: **5-15%** incremental.
   - Why: `Board.occupied_squares()` allocates a tuple repeatedly; `Board.apply_move()` allocates a list and then tuple for every pseudo move; `Move` objects are created heavily.
   - Approach:
     - Add an internal iterator for occupied squares to avoid tuple allocation in `pseudo_legal_moves()` and `_king_square()`.
     - Avoid sets created inside `is_square_attacked()` on every call (`bishop_attackers`, `rook_attackers`) by hoisting constants.
     - Consider direct tuple/list mutation patterns only after profiling; do not rewrite board representation yet.
   - Likely files:
     - `src/tinychess/engine/board.py`
     - `src/tinychess/engine/legal_moves.py`
   - Tests/benchmarks:
     - Full engine tests/perft plus import benchmark.
   - Risk: low if kept internal and parity-tested.

7. **Multiprocess PGN parsing/import by record chunks after single-process parser fixes.**
   - Expected impact: **4-8x wall-clock on Apple Silicon** until CPU, disk, or compression bottlenecks dominate.
   - Why: games are independent. Even after removing parser duplicate legal generation, full import is still CPU-bound on legal generation.
   - Approach:
     - Parent streams deterministic record chunks to worker processes.
     - Workers parse/encode/write chunk-local shard dirs or temp arrays; parent merges manifest in chunk order.
     - Avoid passing per-sample Python objects back to the parent.
   - Likely files:
     - `src/tinychess/nn/pgn_dataset.py`
     - `scripts/pgn_ingest.py`
     - `scripts/pgn_ingest_benchmark.py`
     - docs/tests for deterministic shard manifests
   - Tests/benchmarks:
     - Single-process vs 2/4/8 workers on 1000+ games.
     - Deterministic counters and loadable shards.
   - Risk: medium. This improves wall-clock but does not reduce per-ply cost; do it after item 1 so process overhead is not hiding avoidable duplication.

8. **Only after Python options: consider a narrow native accelerator.**
   - Expected impact: unknown until items 1-4 are benchmarked.
   - Scope if justified: accelerate legal move generation / attack detection / SAN resolution with Python parity tests, not a broad Swift engine rewrite.
   - Risk: high relative to Python fixes. Do not start here.

What NOT to do:
- Do not relax strict PGN parser behavior or silently accept more SAN/PGN syntax.
- Do not remove terminal/outcome checks from `parse_pgn()` without parity tests proving identical accept/reject behavior.
- Do not optimize away full legal move generation per ply while dense legal masks remain required; at least one legal tuple per sample is part of the current dense schema contract.
- Do not introduce unbounded global legal-move caches for a 27M-sample import.
- Do not change dense policy/legal-mask storage in this bottleneck pass; sparse storage is a separate decision gate.
- Do not launch Swift/native work before measuring the parser no-history path and legal-move hot-path Python optimizations.

Risks:
- Parser fast-advance can subtly diverge from `Game.play()` on repetition/fifty-move/insufficient-material/checkmate rejection unless the pre-move outcome logic is shared or heavily tested.
- Legal move hot-path rewrites can break rare rules: castling through check, en-passant discovered check, promotions, pinned pieces, adjacent kings.
- Legal-move caching can improve the first 100 opening-heavy games but disappoint over the full corpus; benchmark multiple slices.
- Multiprocessing may become I/O/compression-bound once parser speed improves.

Need from main agent:
- Decide whether the next implementation should be the parser no-history advance path alone (recommended first), or include legal-move cache/hot-path changes in the same milestone. For safety, keep item 1 as its own benchmarked commit.

Suggested execution prompt:
- Implementation handoff is warranted for item 1 only as the next safest high-impact step.

Prompt:
“Implement the next Python-first PGN parser bottleneck optimization. Scope: keep `parse_san`, `parse_pgn`, `parse_pgn_with_trace`, and ingestion sanitizer behavior stable, but replace parser-internal `Game.play()` per ply with a private checked no-history parser state that reuses the legal tuple already computed by SAN resolution. Preserve pre-move terminal/outcome rejection semantics exactly; do not broaden PGN syntax and do not change dense dataset schema. Add parity/rejection tests for normal play, mate, moves after terminal outcome, repetition, fifty-move, insufficient material, castling, en-passant, promotion, trace boards/clocks/legal moves, and PGN ingestion arrays. Benchmark 100 and preferably 1000 accepted games before/after with dry-run and full-write, and report speedup plus any remaining bottleneck phase shares.”
