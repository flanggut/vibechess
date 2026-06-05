# Implementation Plan

## Goal
First remove/disable within-tree leaf parallelism for dataset generation, then implement a deterministic central batched inference queue that batches model calls across independent self-play searches while preserving serial MCTS semantics per tree.

## Problem Statement
Leaf-parallel inference batches multiple leaves from a single MCTS tree before their values are backed up. Experiments with `data/checkpoints/res8_04_sp` showed this shifts fixed-budget root visit distributions and sampled/selected moves, especially at higher `leaf_parallelism` and with `temperature=1.0`. Dataset generation should not depend on an approximate within-tree batching mode. The preferred sequence is therefore:

1. Stop using and reject `leaf_parallelism > 1` for dataset generation immediately.
2. Simplify/remove the leaf-parallel MCTS implementation, tests, CLI, UI/protocol plumbing, and new metadata propagation.
3. Build the central cross-game inference queue on top of the simplified serial MCTS path.

Historical dataset metadata may still contain `generation_settings["mcts"]["leaf_parallelism"]`; metadata loading should remain tolerant of those old opaque settings even though new generation should not write the field.

## Goals
- Prevent new neural self-play datasets from using `leaf_parallelism > 1`.
- Preserve serial neural MCTS semantics: every simulation completes `select leaf -> infer -> expand/evaluate -> backup` before the same tree selects another leaf.
- Remove within-tree leaf-parallel search as a supported batching/throughput mode unless a later explicit product decision reverses this.
- Batch inference requests across independent self-play games/searches in the same worker process.
- Keep dataset generation deterministic for fixed seeds, fixed checkpoint/model, fixed `batch_size`, and fixed process chunking.
- Record central batching mode and queue settings in new dataset metadata/profile output.
- Keep compatibility for reading old metadata that already recorded leaf-parallel settings.

## Non-Goals
- Do not batch multiple leaves from the same tree.
- Do not introduce cross-process shared inference in the first implementation; existing `--workers` process chunks can each run their own in-process queue.
- Do not optimize classical MCTS.
- Do not broaden checkpoint schema; dataset generation settings can record central batching metadata.
- Do not preserve leaf-parallel search as an alternate high-throughput mode in the default implementation.

## Preferred Sequencing
The implementation should be staged in this order:

1. **Safety stop for dataset generation**: reject `leaf_parallelism > 1` in neural self-play/CLI entry points before other work proceeds.
2. **Leaf-parallel cleanup**: remove or hard-deprecate the leaf-parallel MCTS field/path and associated tests, CLI/UI/protocol flags, benchmark references, and new metadata propagation. Keep old metadata readers tolerant.
3. **Central queue implementation**: refactor serial MCTS into cooperative sessions and batch exactly one pending inference request per independent game/search.
4. **Validation/benchmarking**: prove central queue parity with serial search and measure throughput improvements.

## Proposed Architecture
Use a deterministic cooperative scheduler rather than OS-thread timing.

1. Each active self-play game owns its existing `NeuralMCTSPlayer` and tree reuse state.
2. For one ply across a batch of games, create one serial MCTS search session per non-terminal game.
3. Each session advances until it either:
   - completes its search and returns a `NeuralMCTSResult`, or
   - needs exactly one neural inference for a leaf expansion/evaluation.
4. The coordinator collects at most one pending request per session, in stable `game_index` order.
5. The coordinator calls `PolicyValueInference.predict_legal_batch(games, legal_moves)` for the pending requests.
6. Each session resumes with its matching prediction and backs it up before it can issue another request.
7. Completed search results are recorded with the existing self-play sample encoding and move application code.

This replaces the current root-prefetch-only batching path for neural self-play when `SelfPlayConfig.batch_size > 1` and `label_source == "neural"`. Central queue batching should be the only new supported batching mode for neural self-play.

## API Shape
1. **Leaf-parallel removal / serial-only MCTS config**
   - File: `src/tinychess/ai/neural_mcts.py`
   - Remove `NeuralMCTSConfig.leaf_parallelism` or hard-deprecate it to accept only `1` during a short transition.
   - Simplify `_search_profiled` to run only the serial simulation path.
   - Remove virtual-loss and pending-leaf helpers once no callers/tests require them.

2. **Serial MCTS session API**
   - File: `src/tinychess/ai/neural_mcts.py`
   - Add small internal/public-enough dataclasses/classes, for example:
     - `NeuralMCTSInferenceRequest`
       - fields: `session_id`, `node`, `game`, `legal_moves`, `budget_blocked`, `selection_depth`
     - `NeuralMCTSSearchSession`
       - constructed from `NeuralMCTSPlayer` and `Game`
       - `advance() -> NeuralMCTSInferenceRequest | NeuralMCTSResult | None`
       - `resume(prediction: InferenceResult | LegalPolicyResult) -> None`
       - maintains `simulations`, `nodes_created`, `root`, deadline, and selected leaf state
   - Refactor existing `_run_serial_simulations(...)` so normal `search(...)` and the session API share selection/backup code.

3. **Central self-play coordinator**
   - File: `src/tinychess/nn/self_play.py`
   - Add deterministic coordinator logic, for example:
     - `_run_batched_neural_searches(inference, states, config) -> list[tuple[state, legal, result]]`
     - `_CentralInferenceQueue` or `_BatchedInferenceCoordinator`
   - It should collect requests in stable `game_index` order and call `inference.predict_legal_batch(...)`.
   - Feed `batch.result_at(i)` back into the matching search session.

4. **Configuration/metadata**
   - File: `src/tinychess/nn/self_play.py`
   - Reuse `SelfPlayConfig.batch_size` as the in-process central inference batch / in-flight game count.
   - Add metadata fields through `generation_settings`, e.g.:
     - `"batching_mode": "serial" | "central_inference_queue"`
     - optional `"inference_batch_size": config.batch_size`
   - New self-play metadata should no longer include `generation_settings["mcts"]["leaf_parallelism"]` after cleanup.
   - Keep `SelfPlayMetadata.from_dict()` tolerant of historical metadata that contains leaf-parallel keys.

5. **CLI/UI/protocol behavior**
   - File: `scripts/self_play.py`
   - First change: reject `--leaf-parallelism > 1` for neural dataset generation.
   - Cleanup change: remove `--leaf-parallelism` or keep a deprecated compatibility flag that accepts only `1` for one release, depending on compatibility needs.
   - Files: `src/tinychess/cli.py`, `src/tinychess/ui/terminal.py`, `src/tinychess/protocols/gui.py`
   - Remove terminal/GUI leaf-parallelism options after the core config no longer exposes the field, or accept only absent/`1` if external GUI compatibility requires a transition.

## Tasks
1. **[Done] Immediately reject leaf-parallel neural self-play generation**
   - File: `scripts/self_play.py`
   - Changes: Add CLI validation so `--leaf-parallelism > 1` fails for neural self-play with a clear message explaining that within-tree leaf parallelism shifts dataset distributions and central queue batching will use `--batch-size` instead.
   - Acceptance: CLI test asserts `--leaf-parallelism 2` is rejected; `--leaf-parallelism 1` still works during transition.
   - Completed: hard-removed the flag; argparse now rejects `--leaf-parallelism` as an unknown option and tests cover the removal.

2. **[Done] Add self-play config validation against leaf-parallel generation**
   - File: `src/tinychess/nn/self_play.py`
   - Changes: Reject `SelfPlayConfig` for neural label generation when `mcts.leaf_parallelism > 1`, so programmatic callers cannot bypass CLI validation.
   - Acceptance: Unit test verifies `SelfPlayConfig(..., mcts=NeuralMCTSConfig(leaf_parallelism=2))` raises a clear `ValueError` before generating data.
   - Completed: hard removal prevents programmatic configuration via `NeuralMCTSConfig`; new generation paths no longer expose or write the field.

3. **[Done] Remove leaf-parallel MCTS internals**
   - File: `src/tinychess/ai/neural_mcts.py`
   - Changes: Remove `NeuralMCTSConfig.leaf_parallelism` and validation, `_PendingLeafEvaluation`, `_run_leaf_parallel_simulations`, `_select_pending_leaf`, `_predict_leaf_batch`, `_apply_virtual_loss`, `_revert_virtual_loss`, and `_add_virtual_path_loss` if no other code uses them. Simplify `_search_profiled` to always run serial simulations.
   - Acceptance: `rg "leaf_parallelism|leaf_parallel|virtual_loss|_PendingLeafEvaluation" src/tinychess/ai/neural_mcts.py` returns no implementation references; existing serial neural MCTS tests pass with unchanged expected results.
   - Completed: removed within-tree leaf-parallel internals and virtual-loss helpers; neural MCTS now runs the serial simulation path only.

4. **[Done] Remove leaf-parallel self-play config and metadata propagation**
   - File: `src/tinychess/nn/self_play.py`
   - Changes: Remove profile labels and new metadata output that reference `resolved.mcts.leaf_parallelism`; ensure new `SelfPlayConfig.to_dict()` / generation settings do not write `generation_settings["mcts"]["leaf_parallelism"]`.
   - Acceptance: New metadata test asserts leaf-parallelism is absent from newly generated metadata; old metadata fixture or synthetic metadata containing a historical `leaf_parallelism` key still loads because generation settings are treated as opaque historical data.
   - Completed: new metadata omits the removed field while a synthetic historical metadata object with the old opaque key still loads.

5. **[Done] Remove self-play CLI leaf-parallel flag**
   - File: `scripts/self_play.py`
   - Changes: Remove `GenerationArgs.leaf_parallelism`, `parser.add_argument("--leaf-parallelism", ...)`, validation for values below `1`, and construction of `NeuralMCTSConfig(leaf_parallelism=...)`. Update help/examples to use `--batch-size` for batching.
   - Acceptance: `uv run python scripts/self_play.py --help` no longer mentions leaf parallelism; CLI tests no longer pass/assert `--leaf-parallelism`; invoking the removed flag fails through argparse as an unknown option.
   - Compatibility option: if a transition is required, keep the flag temporarily but accept only `1`, warn/deprecate in help, and schedule hard removal before central queue release.
   - Completed: hard-removed `--leaf-parallelism`; help no longer mentions it and tests assert the removed flag is rejected.

6. **[Done] Remove terminal/UI and GUI protocol leaf-parallel options**
   - Files: `src/tinychess/cli.py`, `src/tinychess/ui/terminal.py`, `src/tinychess/protocols/gui.py`
   - Changes: Remove `--ai-leaf-parallelism`, `TerminalPlayConfig.ai_leaf_parallelism`, GUI `leafParallelism` config field serialization/parsing, and `NeuralMCTSConfig(leaf_parallelism=...)` plumbing.
   - Acceptance: `uv run tinychess play --help` no longer mentions leaf parallelism; GUI protocol tests expect no `leafParallelism` field in default/config responses.
   - Compatibility option: if external GUI clients exist, accept absent or `1` for one release and reject values greater than `1` before hard removal.
   - Completed: removed terminal and GUI leaf-parallel options/fields/plumbing, including Swift GUI DTO support.

7. **[Done] Delete or update leaf-parallel tests**
   - Files: `tests/ai/test_neural_mcts.py`, `tests/nn/test_self_play.py`, `tests/test_terminal_play.py`, `tests/test_gui_protocol.py`
   - Changes: Delete tests that specifically verify leaf-parallel behavior, virtual duplicate avoidance, leaf-parallel determinism, node-budget behavior under leaf-parallelism, CLI acceptance of `--leaf-parallelism`, terminal `--ai-leaf-parallelism`, GUI `leafParallelism`, and metadata propagation of `mcts.leaf_parallelism`.
   - Acceptance: `rg "leaf_parallelism|leaf-parallelism|leafParallelism" tests` returns no active test references except optional legacy-metadata fixture comments.
   - Completed: deleted/updated active leaf-parallel behavior tests and replaced them with serial/no-field/removal coverage.

8. **[Done] Update benchmark/docs references before central queue work**
   - Files: `scripts/self_play_benchmark.py`, `README.md`, `docs/ai.md`
   - Changes: Remove or mark as historical any user-facing references that recommend `leaf_parallelism` for throughput. Explain that within-tree leaf parallelism was removed/disabled due to dataset distribution drift and that central cross-game batching is the intended replacement.
   - Acceptance: `rg "leaf_parallelism|leaf-parallelism|leafParallelism|ai-leaf" README.md docs scripts` has no stale recommendations; any retained references are explicitly historical or compatibility-only.
   - Completed: removed stale user-facing recommendations and documented serial per-tree semantics plus future cross-game batching direction.

9. **[Done] Refactor serial neural MCTS leaf selection into reusable helpers**
   - File: `src/tinychess/ai/neural_mcts.py`
   - Changes: Extract serial simulation selection logic from `_run_serial_simulations` into a helper that returns the selected node, terminal value, budget-blocked flag, updated node count, and selection depth.
   - Acceptance: Existing `tests/ai/test_neural_mcts.py` pass with no behavior changes for `NeuralMCTSPlayer.search` after leaf-parallel cleanup.
   - Completed: extracted shared serial search preparation, leaf selection, and finish helpers used by normal search and session execution without changing serial search behavior.

10. **[Done] Add serial search session primitives**
   - File: `src/tinychess/ai/neural_mcts.py`
   - Changes: Add `NeuralMCTSInferenceRequest` and `NeuralMCTSSearchSession` (names can be adjusted) that use the refactored helpers and update the owning `NeuralMCTSPlayer.last_result`/tree root on completion.
   - Acceptance: New unit test can run one session to completion with single-request batches and produce the same `move`, `visit_counts`, `simulations`, and `nodes` as `NeuralMCTSPlayer.search` for fixed seed and fake inference.
   - Completed: added session/request primitives with single-pending-request semantics plus focused parity, resume-state, and node-budget tests.

11. **Implement deterministic central inference coordinator**
   - File: `src/tinychess/nn/self_play.py`
   - Changes: Add coordinator logic that advances sessions round-robin in stable game order, batches pending requests with `PolicyValueInference.predict_legal_batch`, and resumes sessions with `LegalPolicyResult` rows.
   - Acceptance: A test with `CountingPolicyValueInference`, `games=2`, `simulations>=3`, and `batch_size=2` observes `legal_batch_calls > 1` and at least one batch of size 2 beyond root expansion.

12. **Replace root-prefetch batched self-play path**
   - File: `src/tinychess/nn/self_play.py`
   - Changes: Update `_generate_batched_neural_self_play_dataset` to use the central coordinator instead of `_PrefetchedRootInference`/root-only `predict_batch`. Remove or retire `_PrefetchedRootInference` once tests no longer need it.
   - Acceptance: Existing dataset shape/schema tests pass; new equivalence tests show batched central queue and serial generation match for fixed seeds on small deterministic runs.

13. **Add central queue configuration metadata**
   - File: `src/tinychess/nn/self_play.py`
   - Changes: Add `batching_mode` and `inference_batch_size` to `SelfPlayConfig.to_dict()` or generation settings extra in the batched path.
   - Acceptance: Tests assert metadata includes central queue mode for `batch_size > 1` and serial mode for `batch_size == 1`.

14. **Update self-play CLI help for central batching**
   - File: `scripts/self_play.py`
   - Changes: Update `--batch-size` help text to describe in-process central inference batching across independent games/searches.
   - Acceptance: CLI help distinguishes central cross-game batching from removed within-tree leaf parallelism.

15. **Add determinism/equivalence tests for central queue**
   - File: `tests/nn/test_self_play.py`
   - Changes: Add tests comparing serial vs central queue outputs for fixed seed/checkpoint-model config:
     - `positions`, `legal_masks`, `mcts_policies`, `outcomes`
     - `games[].moves_uci`
     - `games[].final_fen`
   - Acceptance: Arrays and game records are identical for `batch_size=1` vs `batch_size=2` with no time limit and deterministic fake/compact inference.

16. **Add neural MCTS session unit tests**
   - File: `tests/ai/test_neural_mcts.py`
   - Changes: Test session state transitions, terminal handling, node-budget handling, and parity with existing `search` for fake inference.
   - Acceptance: Session API cannot issue a second request before `resume(...)`; completed result matches serial search.

17. **Update benchmark/profiling coverage for central queue**
   - File: `scripts/self_play_benchmark.py`
   - Changes: Ensure benchmark metadata labels central queue batching distinctly; include counters for `inference.predict_legal_batch.calls`, total positions, and batch-size distribution.
   - Acceptance: Smoke benchmark still passes; profile output shows central queue batch size distribution.

18. **Validate central queue quality and throughput**
   - File: no source file required; use benchmark/experiment commands.
   - Changes: Run fixed-position parity/drift checks proving central queue has no search-trajectory drift relative to serial MCTS with deterministic fake inference, plus smoke checks with real MLX inference. Run throughput benchmarks for `--batch-size 1/2/4/8`.
   - Acceptance: Results are recorded in the implementation handoff; no open blocker remains for using central queue in dataset generation.

## Files to Modify
- `src/tinychess/ai/neural_mcts.py` - remove leaf-parallel config/path first; then refactor serial simulation logic and add serial search session/request API.
- `src/tinychess/nn/self_play.py` - reject leaf-parallel dataset generation first; remove leaf-parallel metadata/profile propagation; implement central inference coordinator; record central batching metadata/profile counters.
- `scripts/self_play.py` - reject `--leaf-parallelism > 1` immediately; then remove or accept-only-`1` the flag; update `--batch-size` help for central queue batching.
- `scripts/self_play_benchmark.py` - remove leaf-parallel recommendations; label/profile central queue batching.
- `src/tinychess/cli.py` - remove `--ai-leaf-parallelism` from terminal play CLI arguments and plumbing.
- `src/tinychess/ui/terminal.py` - remove `ai_leaf_parallelism` config field and `NeuralMCTSConfig` argument.
- `src/tinychess/protocols/gui.py` - remove GUI `leafParallelism` config field/parsing/serialization, or temporarily accept only `1` if compatibility is required.
- `tests/ai/test_neural_mcts.py` - delete/update leaf-parallel tests; add session API parity tests.
- `tests/nn/test_self_play.py` - add rejection/removal tests; add central queue schema, batching, reproducibility, and serial-equivalence tests.
- `tests/test_terminal_play.py` - remove terminal CLI assertions for `--ai-leaf-parallelism`.
- `tests/test_gui_protocol.py` - remove GUI protocol assertions for `leafParallelism`.
- `README.md` - remove leaf-parallelism usage/recommendations; document central queue batching where relevant.
- `docs/ai.md` - remove leaf-parallelism semantics; document central queue batching and serial per-tree semantics.

## New Files
- None required.
- Optional: `src/tinychess/ai/neural_mcts_session.py` only if `neural_mcts.py` becomes too large; keep imports simple and avoid circular dependencies.

## Dependencies
- Tasks 1 and 2 should happen first and can be implemented before broader cleanup.
- Tasks 3 through 8 depend on deciding hard removal vs one-release accept-only-`1` compatibility for CLI/UI/GUI.
- Task 9 depends on Tasks 3 and 7 so the serial path is simplified before refactoring.
- Task 10 depends on Task 9.
- Task 11 depends on Task 10.
- Task 12 depends on Task 11.
- Task 13 depends on Task 12 and final metadata field names.
- Tasks 15 and 16 depend on Tasks 10 through 13.
- Task 17 depends on Task 12 so benchmark counters reflect the implemented path.
- Task 18 depends on Tasks 11 through 17.

## Determinism and Reproducibility Considerations
- Use a cooperative round-robin scheduler in stable `game_index` order; do not use thread arrival order for first implementation.
- Keep one RNG per `NeuralMCTSPlayer`, seeded via existing `seed + game_index` logic.
- Preserve output ordering by game index when merging per-game samples and records.
- Disallow or document `time_limit_seconds` with central queue; fixed simulation counts should be the supported deterministic path.
- Central queue batching must never select a second leaf from the same tree before backing up the prior prediction.
- Batched MLX inference may have tiny numerical differences from single-row inference. Equivalence tests should use deterministic fake/compact inference for exact parity, and separate smoke tests should cover real `PolicyValueInference` for validity.
- After cleanup, there should be no new `leaf_parallelism` setting that can reintroduce within-tree search drift in dataset generation.

## Validation / Tests / Benchmarks
- Focused tests after disabling/removal:
  - `uv run pytest tests/ai/test_neural_mcts.py tests/nn/test_self_play.py tests/test_terminal_play.py tests/test_gui_protocol.py`
- Full checks after implementation:
  - `uv run pytest`
  - `uv run ruff check .`
  - `uv run mypy`
- Smoke command after central queue implementation:
  - `uv run python scripts/self_play.py --games 2 --max-plies 2 --simulations 3 --batch-size 2 --output data/selfplay/smoke-central-queue`
- Benchmark comparison:
  - serial baseline: `--batch-size 1`
  - central queue: `--batch-size 2/4/8`
  - do not include leaf-parallel comparison in ongoing benchmark recommendations after cleanup; keep old results only as historical motivation.
- Quality/drift validation:
  - fixed checkpoint, fixed positions, compare serial vs central queue generated games/policies for exact or near-exact parity depending on inference backend.
  - confirm central queue has no visit-distribution drift with deterministic fake inference.
- Cleanup validation:
  - `rg "leaf_parallelism|leaf-parallelism|leafParallelism|ai-leaf-parallelism" src scripts tests README.md docs/ai.md` should return no active references after removal.
  - New self-play metadata should include `batching_mode`/`inference_batch_size` and omit `mcts.leaf_parallelism`.

## Risks
- Removing `leaf_parallelism` is a breaking API/CLI/protocol change for local scripts, GUI clients, benchmark recipes, or historical docs. Decide whether to use hard removal or a short accept-only-`1` compatibility shim before implementation.
- Old self-play datasets may contain `generation_settings["mcts"]["leaf_parallelism"]`; metadata loading should remain tolerant even though new generation no longer writes the field.
- If GUI protocol clients send `leafParallelism`, hard removal may fail client requests. A compatibility shim accepting only absent or `1` may be safer if external GUI clients exist.
- Refactoring MCTS search state is invasive; tree reuse/adoption must remain correct after session completion.
- Existing `batch_size` currently means root prefetch batching; replacing it changes performance characteristics while preserving intended serial semantics. Metadata/help must be explicit.
- Real MLX batched vs single inference can differ slightly numerically; tests should separate semantic scheduling parity from backend numeric parity.
- Cooperative batching may underfill batches late in a ply or near terminal games; benchmark before assuming speedup.
- Process-level `--workers` still duplicates model inference per process; cross-process central inference would need a separate design.
- Time-limited MCTS and queued inference are not reliably reproducible; keep dataset generation fixed-budget by default.
- If session API exposes too many internals, future MCTS refactors may become harder. Keep the public surface narrow and well-tested.
