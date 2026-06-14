# Self-Play Speedup Plan

## Goal

Achieve at least a 2x speedup for:

```bash
uv run python scripts/self_play.py \
  --checkpoint data/checkpoints/strongest \
  --output data/selfplay/res8_sp_02 \
  --games 100 \
  --max-plies 300 \
  --simulations 200 \
  --temperature 1.0 \
  --workers 8 \
  --batch-size 16
```

Measured baseline on 2026-06-14:

- `elapsed_seconds`: 267.30
- `sample_count`: 10,622
- `samples_per_second`: 39.74
- `games_per_second`: 0.374
- `batching_mode`: `central_inference_queue`
- `inference_batch_size`: 16
- Benchmark output: `artifacts/selfplay-speed/requested/repeat-001`

CLI-only tuning did not reach 2x. The path to 2x is code-level hot-path work.

## Benchmark protocol

Use the same checkpoint and deterministic seeds when comparing changes.

Fast smoke benchmark:

```bash
uv run python scripts/self_play_benchmark.py \
  --checkpoint data/checkpoints/strongest \
  --output-root artifacts/selfplay-speed/smoke \
  --games 32 \
  --max-plies 20 \
  --simulations 200 \
  --temperature 1.0 \
  --workers 8 \
  --batch-size 16 \
  --no-profile \
  --format json
```

Representative benchmark:

```bash
uv run python scripts/self_play_benchmark.py \
  --checkpoint data/checkpoints/strongest \
  --output-root artifacts/selfplay-speed/representative \
  --games 64 \
  --max-plies 50 \
  --simulations 200 \
  --temperature 1.0 \
  --workers 8 \
  --batch-size 16 \
  --no-profile \
  --format json
```

Full target benchmark:

```bash
uv run python scripts/self_play_benchmark.py \
  --checkpoint data/checkpoints/strongest \
  --output-root artifacts/selfplay-speed/full-target \
  --games 100 \
  --max-plies 300 \
  --simulations 200 \
  --temperature 1.0 \
  --workers 8 \
  --batch-size 16 \
  --no-profile \
  --format json \
  --keep-output
```

Correctness gates after behavioral changes:

```bash
uv run pytest tests/ai/test_neural_mcts.py tests/nn/test_self_play.py tests/test_self_play_benchmark.py
uv run ruff check .
uv run mypy
```

## Known profiler issue

Before relying on built-in `--profile-level summary|detailed`, fix slow-item sorting in `src/vibechess/profiling.py`.

Observed failure:

- `SelfPlayProfiler._record_slow()` stores heap entries as `(seconds, item)`.
- Equal `seconds` values force Python to compare `dict` items.
- Crash location: `src/vibechess/profiling.py:490-496`.

Plan:

1. Add a monotonic integer sequence field to heap entries: `(seconds, sequence, item)`.
2. Keep top-N semantics unchanged.
3. Add a unit test with tied slow entries.
4. Re-run self-play benchmark with `--profile-level summary` and `--profile-level detailed`.

Acceptance:

- Profiled self-play no longer crashes.
- Profile output still contains `slowest_plies` and `slowest_searches` sorted by descending seconds.

## Optimization items

### 1. Remove UCI string allocation from MCTS edge tie-breaks

Current hotspot:

- `NeuralMCTSNode.best_edge()` calls `edge.move.to_uci()` in the `max()` key.
- File: `src/vibechess/ai/neural_mcts.py:217-237`.
- Profile evidence: `Move.to_uci()` consumed about 1.07s of a 7.13s cProfile run, with ~745k calls.

Change:

1. Add a cheap deterministic move-order key, e.g. `(from_square, to_square, promotion_order)`.
2. Use that key in `best_edge()` instead of `edge.move.to_uci()`.
3. Reuse the same helper anywhere else hot code uses UCI only for deterministic ordering.
4. Do not change external UCI/SAN boundaries.

Acceptance:

- MCTS move selection remains deterministic for tied scores.
- Existing neural MCTS tests pass.
- `Move.to_uci()` calls drop sharply in cProfile.
- Smoke benchmark improves without changing sample count for fixed seed/config.

Risk:

- Tie ordering may differ from previous UCI lexical order unless the numeric key reproduces it. Prefer exact UCI-order-compatible numeric key if deterministic parity matters.

### 2. Cache compact legal-index MLX arrays per node — Done 2026-06-14

Current hotspot:

- `PolicyValueInference.predict_legal_batch()` creates `mx.array(indices)` for every row, every inference batch.
- File: `src/vibechess/nn/inference.py:299-307`.
- Legal action indices are already cached as Python tuples on `NeuralMCTSNode`.

Change:

1. Add an optional cached MLX legal-index array on `NeuralMCTSNode`.
2. Populate it when `cached_legal_action_indices()` is first converted for inference.
3. Thread optional prebuilt MLX index arrays through `NeuralMCTSInferenceRequest` and `predict_legal_batch()`.
4. Keep public inference APIs backward-compatible for non-search callers.

Acceptance:

- No dense legal masks are built in the neural search inference path.
- `predict_legal_batch()` avoids per-row Python-to-MLX index conversion when cached arrays are supplied.
- Unit tests cover cached and uncached legal-index inputs.
- Smoke benchmark improves or is neutral.


Completed:

- Added `NeuralMCTSNode.cached_legal_action_index_array()` and per-node MLX index-array storage.
- Threaded optional `legal_action_index_array`/`legal_action_index_arrays` through neural MCTS requests, search helpers, central self-play batching, and `PolicyValueInference`.
- Kept tuple `legal_action_indices` in result DTOs for existing callers and validation.
- Added tests proving cached arrays are reused by nodes and `predict_legal_batch()` avoids `mx.array(indices)` when arrays are supplied.
- Measured smoke benchmark after change: 15.96s, 40.10 samples/sec on 32 games × 20 plies × 200 simulations, workers 8, batch 16.

Risk:

- MLX arrays are process-local; do not serialize them across process workers.

### 3. Vectorize compact legal-policy gather and softmax

Current hotspot:

- `predict_legal_batch()` loops rows in Python and applies one softmax per legal move vector.
- File: `src/vibechess/nn/inference.py:299-310`.

Change:

1. Pad legal-action indices per batch to `[batch, max_legal]`.
2. Gather logits in one batched operation.
3. Apply a mask for padded entries.
4. Softmax over axis 1 once.
5. Return compact per-row policies preserving existing result contracts.

Acceptance:

- Policy values for each row match old per-row softmax within float tolerance.
- Empty-legal rows still return empty policies.
- Tests cover varied legal counts in one batch.
- Inference microbenchmark improves for batch sizes 8-32.

Risk:

- Returning per-row compact MLX slices may still force Python row handling. Keep the model/gather/softmax path vectorized first; avoid premature API churn.

### 4. Let each MCTS search session contribute multiple pending leaves

Current limitation:

- Central queue batching is per process.
- `NeuralMCTSSearchSession.advance()` emits at most one inference request per active game/search at a time.
- With 8 workers, each worker owns only 12-13 games in the target run, so batches can underfill and queue rounds are frequent.
- Files: `src/vibechess/nn/self_play.py:322-386`, `src/vibechess/ai/neural_mcts.py:727-786`.

Change:

1. Add `NeuralMCTSSearchSession.advance_many(limit)`.
2. Use virtual visits/losses for all selected leaves before inference, matching existing single-request semantics.
3. Let `_run_central_neural_searches()` fill a batch from fewer active games by collecting multiple leaves per session.
4. Add a config field for per-session request cap if needed; default should preserve current behavior unless batch size requires more requests.

Acceptance:

- Completed simulation count remains exactly `simulations` per move.
- Search results remain deterministic for fixed seed/config or documented if virtual-loss ordering changes tied choices.
- Batch-size distribution moves closer to 16 in profiled runs.
- Representative benchmark improves materially.

Risk:

- Virtual loss changes search ordering. Keep tests focused on invariants: legal moves, visit counts, policy target shape/sum, reproducible fixed-seed output if expected by current tests.

### 5. Avoid repeated `SearchState.to_game()` allocation for inference requests

Current overhead:

- `NeuralMCTSSearchSession.advance()` builds compact `Game` objects for inference requests using `SearchState.to_game(include_positions=False)`.
- File: `src/vibechess/ai/search_state.py:88-107`, request creation in `src/vibechess/ai/neural_mcts.py:727-765`.

Change:

1. Extend inference request DTOs to carry `SearchState` or direct board/clocks alongside cached encoded input.
2. Make `predict_legal_batch()` require `Game` only when it needs to compute encodings or legal indices itself.
3. In search path, use cached encoded inputs and legal indices, so no `Game` reconstruction is needed.

Acceptance:

- Public `PolicyValueInference` API still accepts `Game` inputs.
- Search-only path avoids `SearchState.to_game()` during simulations.
- cProfile shows `to_game()` request allocation near zero.

Risk:

- Keep protocol boundaries clear: UCI/PGN/dataset code should continue using `Game`.

### 6. Optimize legal attack checks with precomputed tables — Done 2026-06-14

Current hotspot:

- Legal generation consumed about 1.41s of 7.13s cProfile.
- `_is_square_attacked_index()` consumed about 0.96s.
- Files: `src/vibechess/engine/legal_moves.py:73-99`, `151-212`, `425-468`.

Change:

1. Precompute knight, king, and pawn attack square lists for all 64 squares.
2. Precompute ray square lists for bishop/rook directions from every square.
3. Replace repeated file/rank/on-board arithmetic in `_is_square_attacked_index()` with table iteration.
4. Keep current board representation initially; do not jump to bitboards until table optimization is measured.

Acceptance:

- Existing legal move, FEN, game, and perft tests pass.
- Add focused tests for attack detection at board edges, pins, sliders blocked by pieces, and pawn direction.
- `legal_moves.py` cProfile time decreases.


Completed:

- Added precomputed pawn, knight, king, bishop-ray, and rook-ray attack tables in `src/vibechess/engine/legal_moves.py`.
- Replaced per-call attack file/rank/on-board arithmetic with table iteration in `_is_square_attacked_index()`.
- Added focused tests for pawn direction/edge wrapping, leaper edge wrapping, and slider blockers in `tests/test_legal_moves.py`.
- Measured cProfile slice improvement:
  - total: 7.13s -> 5.87s
  - `legal_moves.py:legal_moves`: 1.41s -> 0.59s
  - `_is_square_attacked_index`: 0.96s -> 0.19s

Risk:

- Attack direction bugs are high impact. Verify with existing perft plus targeted positions.

### 7. Cache move action indices on generated moves or nodes

Current hotspot:

- `move_to_action_index()` consumed about 0.39s of 7.13s cProfile.
- Files: `src/vibechess/nn/encode.py:178-203`, `src/vibechess/ai/neural_mcts.py:175-183`.

Change:

1. Keep node-level `legal_moves` and `legal_action_indices` aligned.
2. Avoid recomputing indices in dataset recording when the chosen root already has legal indices.
3. Consider a per-board/legal tuple cache only if node-level reuse is insufficient.

Acceptance:

- Dataset legal masks and policy targets remain identical for fixed positions.
- `move_to_action_index()` call count drops in self-play cProfile.

Risk:

- Underpromotion indices depend on board side/pawn context. Cache only in contexts tied to the exact board.

### 8. Reduce dataset write/read/merge overhead in parallel self-play

Current behavior:

- Each worker writes a compressed shard.
- Parent reads/decompresses shards, merges arrays, then writes final compressed output.
- Files: `scripts/self_play.py:572-585`, `src/vibechess/nn/self_play_dataset.py:384-435`, `495-545`.

Change:

1. Return in-memory shard summaries/arrays from workers when output size is modest.
2. Or write uncompressed temporary shards and compress once in the parent.
3. Keep final output format unchanged.
4. Preserve crash/debug usefulness of shard outputs only when requested.

Acceptance:

- Final `samples.npz`, `metadata.json`, and `games.jsonl` match current schema.
- Parallel generation avoids double compression on the common path.
- Full target benchmark shows reduced tail time, especially at larger sample counts.

Risk:

- In-memory returns can increase parent memory. Gate by expected sample count or use uncompressed temp shards first.

### 9. Evaluate a single shared inference service across workers

Current limitation:

- `--workers 8` means 8 processes each load `data/checkpoints/strongest` and each has its own in-process central inference queue.
- Files: `scripts/self_play.py:166-179`, `207-230`, `508-527`.

Change:

1. Prototype one MLX inference process and multiple CPU search workers.
2. Send encoded inputs plus legal indices to the inference process.
3. Batch globally up to larger batch sizes, e.g. 32-128.
4. Return compact legal policies and values.
5. Measure IPC overhead before committing architecture.

Acceptance:

- Global batch-size distribution improves substantially over per-worker batching.
- End-to-end full target benchmark improves after IPC overhead.
- Model checkpoint loads once for inference service.

Risk:

- Python multiprocessing serialization may erase gains. This is a later-stage item after local hot-path wins.

### 10. Revisit worker/batch defaults after code optimizations

Current measured CLI tuning:

- `workers=8 batch=16` is already near best among tested settings for representative runs.
- `batch=1` is about 2.16x slower on a small slice, proving batching already matters.
- `workers=12` was slightly slower than `workers=8` on a 64-game/50-ply representative slice.

Change:

1. After items 1-6, re-run worker/batch sweeps.
2. Test `workers` in `{4, 6, 8, 10, 12}`.
3. Test `batch-size` in `{8, 16, 24, 32}`.
4. Test `active-games` separately from `batch-size`.
5. Update README/script help only if defaults or recommended commands change.

Acceptance:

- Documented recommendation is based on full or representative benchmarks, not microbenchmarks.
- No default changes without evidence.

## Suggested implementation order

1. Fix profiler slow-item tie crash.
2. Remove `to_uci()` allocation from MCTS tie-breaks.
3. Cache MLX legal-index arrays.
4. Vectorize legal-policy gather/softmax.
5. Add multi-leaf collection per search session.
6. Optimize legal attack checks with precomputed tables.
7. Remove search-path `to_game()` allocation.
8. Reduce shard write/read/merge overhead.
9. Prototype shared inference service.
10. Re-run worker/batch sweep and update recommendations.

## Success criteria

Primary:

- Full target command equivalent reaches `samples_per_second >= 79.5` or `elapsed_seconds <= 133.7` for the same 10,622-sample deterministic workload.

Secondary:

- Smoke and representative benchmarks improve consistently.
- No chess legality, MCTS, dataset schema, or checkpoint compatibility regressions.
- No generated data/checkpoints are committed.
