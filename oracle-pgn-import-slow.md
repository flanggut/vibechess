Inherited decisions:
- PGN ingestion must keep the strict/bounded core parser boundary; ingestion may sanitize public-dataset annotations, but should not silently broaden PGN semantics.
- Dataset layout/action space/encoder metadata must remain compatible (`8*8*73 = 4672`, existing shard manifest format).
- Already implemented optimizations:
  - `src/tinychess/engine/pgn.py`: `parse_san()` reuses one legal-move tuple and `_move_to_san_from_legal()` instead of recomputing legal moves for every candidate.
  - `src/tinychess/nn/encode.py` / `src/tinychess/nn/pgn_dataset.py`: `legal_move_mask_from_legal_moves()` reuses the per-ply legal tuple for ingestion mask construction.
- Prior plan explicitly deferred higher-risk changes until profile-gated: known-legal game transition, direct SAN resolver, NumPy-native encoder, and shard-writing changes.

Diagnosis:
- The import is not hung; it is CPU-bound and still roughly scales to tens of minutes for 5000 elite games.
- Evidence from actual import of 100 games:
  - Command: `rm -rf /tmp/tc-pgn-100 && /usr/bin/time -p uv run python scripts/pgn_ingest.py --input lichess_elite_2025-11.pgn --output /tmp/tc-pgn-100 --max-games 100 --shard-samples 10000`
  - Result: `games_written=100`, `samples=9693`, `real 49.73s`, output shard only `1.2M` compressed.
  - Linear estimate for 5000 games at this rate is ~41 minutes, before allowing for variability. The default command prints only at the end, so it can appear stalled.
- Evidence from dry-run benchmark of 100 records:
  - Command: `/usr/bin/time -p uv run python scripts/pgn_ingest_benchmark.py --input lichess_elite_2025-11.pgn --max-records 100 --format json`
  - Result: `elapsed_seconds=49.30`, `records_per_second=2.03`, `samples_per_second=196.6`, `samples=9693`.
  - Phase timing: `parse_sanitize=26.04s (52.8%)`, `play_moves=11.29s (22.9%)`, `replay_legal_check=5.54s (11.2%)`, `encode_positions=4.17s (8.5%)`, `legal_masks=2.22s (4.5%)`.
- `parse_sanitize` is a misleading phase name at this point: `--strict` on 20 records produced essentially the same throughput/proportions as sanitized mode, so the sanitizer itself is not the culprit. The cost is SAN parsing/legal move work inside `parse_pgn()`.
  - Non-strict 20 records: `elapsed=9.46s`, `parse_sanitize=5.01s (52.96%)`.
  - Strict 20 records: `elapsed=9.50s`, `parse_sanitize=5.02s (52.89%)`.
- cProfile on 20 records confirms the function-level hotspot is legal-move/check work, not file IO or shard writing:
  - Command: `uv run python scripts/pgn_ingest_benchmark.py --input lichess_elite_2025-11.pgn --max-records 20 --profile-output /tmp/pgn20.prof` then `pstats` sorted by cumulative time.
  - Top cumulative functions included `legal_moves.py:legal_moves` (`13029` calls), `is_in_check` (`485392` calls), `is_square_attacked` (`487630` calls), `pgn_stream.py:parse_ingest_pgn`, `pgn.py:parse_pgn`, `game.py:play`, `pgn.py:parse_san`, and `_move_to_san_from_legal` (`55013` calls).
  - Profiling overhead inflated wall time, but the ranking and call counts are useful.

Drift / contradiction check:
- The current profile now satisfies the prior plan’s gate for considering a direct SAN resolver: `parse_sanitize` remains >35-40% and is the largest phase on a 100-record benchmark.
- Legal-mask reuse worked as intended but was never expected to solve the whole problem; `legal_masks` is now only ~4-5%.
- Optimizing shard writing would be drift: full import and dry-run timings match closely for 100 games, and the compressed shard is small. Disk/NPZ compression is not the current bottleneck.
- A broad “make Game.play faster” change can easily conflict with the existing `Game.play` semantics that reject moves after pragmatic terminal outcomes. Any known-legal path must be narrow and carefully gated.

Recommendation:
1. Prioritize a direct SAN-token resolver in `src/tinychess/engine/pgn.py`.
   - Why: parsing is still ~53% of import time, and cProfile shows `parse_san()` still formats many legal candidate moves via `_move_to_san_from_legal()`. A direct resolver can parse SAN fields once and filter the already-computed legal moves by piece, target square, capture marker, promotion, castling, and disambiguation, applying/checking only the final candidate(s) for `+`/`#`.
   - Keep `move_to_san()` and `_move_to_san_from_legal()` for formatting/writing; change only parse resolution.
   - Add tests before/with the change for ambiguous disambiguation, captures/en-passant, castling (`O-O`/`0-0` normalization), promotion/underpromotion, check/checkmate suffix, capture-marker mismatch, and unsupported annotations.
2. Then consider a narrow known-legal transition for ingestion replay only, not as a public semantics change.
   - `play_moves` is ~23% and `replay_legal_check` ~11%; after SAN improves, replay transition may become the dominant remaining cost.
   - Safest first use is `_ShardBuilder.add_game()` after `move in legal`, because `parse_pgn()` already validated the same moves through public `Game.play()`.
   - Be careful using it in `parse_pgn()`: bypassing `Game.play()` can change behavior for moves after pragmatic terminal outcomes unless equivalent outcome checks are preserved.
3. Do not target encoder or shard writing next.
   - `encode_positions` is ~8-9%; `legal_masks` is ~4-5%; one-hot labels and IO are negligible.
   - NumPy-native encoding is not yet worth duplicating encoder logic.
4. Add progress reporting to `scripts/pgn_ingest.py` as a UX improvement, but not as the performance fix.
   - For 5000 games, no output until final completion makes a ~40-minute CPU job look dead.
   - A lightweight `--progress-every-games` or periodic stderr line could prevent aborting a healthy run.

Risks:
- Direct SAN parsing is correctness-sensitive. It must preserve bounded PGN behavior and reject unsupported syntax rather than broadening parser tolerance.
- Known-legal play paths risk bypassing outcome/legality invariants; keep public `Game.play()` unchanged.
- Benchmark slices are deterministic first-record slices; final claims should use at least 500 records after changes.
- The 5000-game estimate assumes first-100-game average plies and skip rate; the actual file may vary, but the order of magnitude is clear.

Need from main agent:
- Decide whether to approve the higher-risk direct SAN resolver now. The profile evidence supports it.

Suggested execution prompt:
- Worker handoff is warranted if implementation is approved: “Implement a direct SAN-token resolver in `src/tinychess/engine/pgn.py` to replace parse-time formatting of every legal candidate. Preserve strict/bounded PGN semantics and public `move_to_san()` behavior. Add focused SAN tests for ambiguity, captures/en-passant, castling, promotions/underpromotions, check/checkmate suffixes, capture-marker mismatch, and unsupported annotations. Do not change dataset schema, action space, shard writing, or public `Game.play()` semantics. Run PGN/stream/dataset tests plus ruff/mypy on touched files, then benchmark `--max-records 100` and report phase/cProfile changes.”
