# Implementation Plan

## Goal
Optimize PGN import incrementally with Python-first, strict-parser-compatible changes that each leave the repository passing, while deferring schema-breaking storage changes behind explicit versioned PGN formats and validation gates.

## Recommended Execution Order
1. **Measurement and benchmark coverage** - establish reliable parse/encode/write baselines before changing behavior.
2. **NumPy-native encoding helpers** - remove MLX-to-NumPy overhead without semantic or schema changes.
3. **Lightweight ingestion replay state** - avoid `Game.play()` history/outcome overhead during dataset sample generation while preserving dense shard compatibility.
4. **Parser trace reuse for ingestion** - reuse SAN parser boards/legal moves to remove duplicate replay legality work.
5. **Gate: decide PGN-specific storage format** - only proceed if dense self-play shard compatibility is not required for new PGN imports.
6. **Schema-versioned sparse policy targets** - store played action indices instead of dense one-hot PGN policies, with old dense shards still supported.
7. **Schema-versioned sparse legal masks** - store legal action indices/offsets instead of dense masks, with batch-time densification.
8. **Multiprocessing chunk import** - parallelize after single-process CPU and storage waste are reduced.
9. **Swift reconsideration only if still justified** - no Swift worker until Python fast path + storage + multiprocessing benchmarks show a remaining legal-move/SAN bottleneck.

---

## Component 1: Measurement and benchmark coverage

### Goal
Make the PGN ingestion benchmark measure both current dry-run CPU phases and real shard-writing costs so later workers can prove speedups and catch regressions.

### Files likely changed
- `scripts/pgn_ingest_benchmark.py`
- `tests/test_pgn_ingest_benchmark.py` (new) or `tests/test_benchmarks.py`
- `docs/pgn-ingestion.md`

### Exact scope
- Keep the existing dry-run benchmark behavior as the default.
- Add an explicit full-write benchmark mode that calls `ingest_pgn_dataset()` into a user-provided or temporary output directory and reports elapsed write/compression time, shard count, samples, and approximate output bytes.
- Add JSON fields that make old/new reports comparable: mode, records/games limits, accepted/skipped counters, sample rate, elapsed seconds, and per-phase timings when available.
- Add a small fixture-driven CLI/test path that runs on a temporary PGN file without external datasets.
- Document that dry-run excludes compression and that full-write is the authoritative import throughput measurement.

### Non-goals
- Do not optimize importer code in this component.
- Do not change parser semantics, dataset schema, or training code.
- Do not add Swift or external dependencies.

### Tests/benchmarks to add/run
- Add tests that the benchmark JSON contains stable top-level keys for dry-run and full-write modes.
- Add a CLI smoke test with a tiny PGN input and `--format json`.
- Run:
  - `uv run pytest tests/test_pgn_ingest_benchmark.py tests/nn/test_pgn_dataset.py`
  - `uv run python scripts/pgn_ingest_benchmark.py --input <small-fixture.pgn> --max-records 2 --format json`

### Risks/stop conditions
- Benchmark numbers will be noisy on small fixtures; tests should assert structure/counters, not exact timing.
- Stop and ask before adding heavyweight benchmark fixtures or generated data to the repository.

### Compact worker handoff prompt
> Implement PGN ingestion benchmark coverage only. Keep default dry-run behavior, add an explicit full-write mode that measures `ingest_pgn_dataset()` including NPZ compression/output size, and add fixture-based tests/CLI smoke coverage. Do not change parser semantics, importer behavior, dataset schema, or training. Update `docs/pgn-ingestion.md`. Run focused pytest and report benchmark commands/results.

---

## Component 2: NumPy-native encoding helpers

### Goal
Remove MLX array creation/synchronization from PGN import by adding NumPy-native position and legal-mask helpers that produce tensors identical to the current MLX encoders after `np.asarray(...)`.

### Files likely changed
- `src/tinychess/nn/encode.py`
- `src/tinychess/nn/__init__.py` if helpers are exported publicly
- `src/tinychess/nn/pgn_dataset.py`
- `scripts/pgn_ingest_benchmark.py`
- `tests/nn/test_encode.py`
- `tests/nn/test_pgn_dataset.py`
- `docs/pgn-ingestion.md`

### Exact scope
- Add `encode_board_np(...)` and `encode_game_np(...)` returning `np.ndarray[np.float32]` with shape `TENSOR_SHAPE`.
- Add `legal_move_mask_from_legal_moves_np(game_or_board_state, legal)` or an equivalent helper that fills a NumPy `float32` vector of length `ACTION_SPACE_SIZE` from already-computed legal moves.
- Keep existing MLX helpers unchanged and backward-compatible.
- Change PGN ingestion and the benchmark dry-run to use NumPy helpers for positions and legal masks.
- Preserve dense shard keys: `positions`, `legal_masks`, `mcts_policies`, `outcomes`.

### Non-goals
- Do not change the action space, encoder version, tensor layout, dataset schema, or training loader.
- Do not change legal move generation or SAN parsing.
- Do not preallocate shard arrays yet unless it is a tiny local improvement that does not obscure parity.

### Tests/benchmarks to add/run
- Add tests comparing `encode_game_np(game)` with `np.asarray(encode_game(game), dtype=np.float32)` for start position, FEN with side/en-passant/clocks, castling, and promotion positions.
- Add tests comparing NumPy legal masks with current MLX masks on the existing `test_legal_move_mask_from_legal_moves_matches_public_mask` cases.
- Run:
  - `uv run pytest tests/nn/test_encode.py tests/nn/test_pgn_dataset.py`
  - `uv run python scripts/pgn_ingest_benchmark.py --input <real-or-sample.pgn> --max-records 100 --format json`

### Risks/stop conditions
- Board orientation/channel placement is easy to get subtly wrong; stop if parity tests are not exact.
- If adding public exports causes import cycles or typing churn, keep helpers module-local/public in `encode.py` only and document that choice.

### Compact worker handoff prompt
> Add NumPy-native PGN ingestion encoders. Implement `encode_board_np`/`encode_game_np` and a NumPy legal-mask helper in `src/tinychess/nn/encode.py`; switch `pgn_dataset` and `pgn_ingest_benchmark` to use them. Existing MLX APIs and dense dataset schema must remain unchanged. Add parity tests against the current MLX encoders/masks and run focused pytest plus a benchmark smoke.

---

## Component 3: Lightweight ingestion replay state

### Goal
Avoid `Game.play()` per ply in PGN dataset generation by replaying parsed legal moves with a minimal ingestion state (`Board`, clocks, move list, repetition counts as needed) while keeping dense shards loadable by existing self-play validation.

### Files likely changed
- `src/tinychess/nn/pgn_dataset.py`
- `scripts/pgn_ingest_benchmark.py`
- `tests/nn/test_pgn_dataset.py`
- Possibly `tests/nn/test_pgn_replay_state.py` (new)

### Exact scope
- Add a private ingestion-only replay state in `pgn_dataset.py` such as `_TrainingReplayState` with:
  - current `Board`
  - `halfmove_clock`
  - `fullmove_number`
  - played `Move` list for `games.jsonl`
  - enough final-position/repetition information to record outcomes compatible with `load_self_play_dataset()` validation
- Add a private advance helper using `Board.apply_move(move)` after the existing replay legality check (`move in legal`).
- Compute clocks exactly like `Game.play()`:
  - halfmove resets on pawn moves/captures, otherwise increments
  - fullmove increments after black moves
- Replace `_ShardBuilder.add_game()` per-ply `game = game.play(move)` with the lightweight advance path.
- Generate `SelfPlayGameRecord` without per-ply `Game.play()` tuple-copying. Final record must still match validator expectations for checkmate/stalemate/fifty-move/repetition/insufficient-material cases.
- Preserve current dense shard format and manifest schema.

### Non-goals
- Do not reuse legal moves from parsing yet; this component may still call `legal_moves(board)` during replay.
- Do not change `Game.play()` or public engine APIs unless a tiny helper is clearly safer than duplicating logic.
- Do not relax the parser's current outcome/legality behavior.

### Tests/benchmarks to add/run
- Add replay-state parity tests comparing final FEN, moves UCI, outcomes, and sample counts against the current dense ingestion behavior for:
  - normal nonterminal PGN
  - checkmate PGN
  - castling PGN
  - en-passant PGN
  - promotion PGN (FEN PGNs can be tested at helper level, but ingestion still skips FEN records)
- Existing `load_self_play_dataset()` must successfully load generated shards.
- Run:
  - `uv run pytest tests/nn/test_pgn_dataset.py tests/test_pgn.py`
  - `uv run python scripts/pgn_ingest_benchmark.py --input <real-or-sample.pgn> --max-records 100 --format json`

### Risks/stop conditions
- Final `games.jsonl` records must remain replay-valid from the normal start position; stop if `load_self_play_dataset()` rejects any shard.
- Repetition outcome parity is easy to miss. If exact repetition tracking requires importing/duplicating private engine internals, prefer a small engine helper with tests over broad public API changes.
- Stop before changing draw semantics or accepting games the strict parser currently rejects.

### Compact worker handoff prompt
> Implement an ingestion-only lightweight replay state in `src/tinychess/nn/pgn_dataset.py` so PGN sample generation no longer calls `Game.play()` per ply. Preserve dense shard compatibility and existing parser behavior. Keep replay legality checks for now. Add parity tests that generated shards load with `load_self_play_dataset()` and cover normal, mate, castling, en-passant, and promotion cases. Update benchmark phase timing to show the new replay/apply path.

---

## Component 4: Parser trace reuse for ingestion

### Goal
Eliminate duplicate legal-move generation between SAN parsing and ingestion replay by exposing parser-computed per-ply boards/legal moves to the importer without broadening the strict PGN parser boundary.

### Files likely changed
- `src/tinychess/engine/pgn.py`
- `src/tinychess/engine/pgn_stream.py`
- `src/tinychess/nn/pgn_dataset.py`
- `scripts/pgn_ingest_benchmark.py`
- `tests/test_pgn.py`
- `tests/test_pgn_stream.py`
- `tests/nn/test_pgn_dataset.py`
- `docs/pgn-ingestion.md`

### Exact scope
- Refactor SAN resolution so the parser can return both the selected move and the legal-move tuple computed for that board. Keep `parse_san(board, san) -> Move` unchanged.
- Add a trace data carrier, for example `PgnParsedPly`/`PgnIngestPly`, containing:
  - board before move
  - halfmove clock before move
  - fullmove number before move
  - move
  - legal moves for that exact board
- Add an ingestion parser entry point such as `parse_ingest_pgn_with_trace(text, strict=False)` in `pgn_stream.py`, which sanitizes only when `strict=False` and otherwise follows the exact same PGN rejection rules as `parse_ingest_pgn()`.
- Change `ingest_pgn_dataset()` to use the trace path and feed parser-provided boards/legal moves to `_ShardBuilder`, avoiding a second replay legal generation for sample tensors.
- Keep `parse_pgn()` and `parse_ingest_pgn()` return types and behavior stable for existing callers.

### Non-goals
- Do not tolerate new PGN syntax or annotations.
- Do not remove parser outcome checks unless there is exact semantic parity with tests.
- Do not change dense storage yet.
- Do not store all records in memory; trace lifetime should be one game/record.

### Tests/benchmarks to add/run
- Add tests that traced parsing returns the same moves/result/tags as `parse_ingest_pgn()` for strict and sanitized records.
- Add tests that invalid PGNs still fail in trace mode with the same broad error behavior: comments in strict mode, result mismatch, wrong check suffix, illegal move, tokens after result.
- Add ingestion parity tests comparing dense tensors/masks/policies/outcomes from trace-based import against the pre-trace path for small fixtures.
- Run:
  - `uv run pytest tests/test_pgn.py tests/test_pgn_stream.py tests/nn/test_pgn_dataset.py`
  - `uv run python scripts/pgn_ingest_benchmark.py --input <real-or-sample.pgn> --max-records 100 --format json`
  - If available: same command with `--max-records 1000` for a more stable speedup estimate.

### Risks/stop conditions
- Storing board snapshots and legal tuples per game increases per-record memory; acceptable for one game at a time, but stop if implementation accumulates traces across many records.
- Parser trace must not become a looser ingestion parser. Stop if trace mode accepts a PGN that `parse_ingest_pgn()` rejects, unless the parent explicitly approves a sanitizer-bound behavior change.
- Refactoring `pgn.py` can affect SAN formatting/parsing; run full PGN tests before handoff.

### Compact worker handoff prompt
> Add parser trace reuse for PGN ingestion. Refactor SAN parsing to expose the already-computed legal moves while keeping `parse_san`/`parse_pgn` APIs and strict semantics unchanged. Add `parse_ingest_pgn_with_trace` in `pgn_stream.py` and switch `ingest_pgn_dataset` to use traced boards/legal moves for encoding/masks/policies. Do not broaden PGN syntax or change dense shard schema. Add trace parity/rejection tests and benchmark old-vs-new phase timing.

---

## Component 5: Decision gate for PGN-specific sparse storage

### Goal
Get an explicit decision before changing the on-disk PGN shard format, because dense self-play-compatible shards are currently documented and tested.

### Files likely changed
- `plans/pgn-import-optimization-components.md` only if recording the decision, or a short follow-up decision note under `plans/`
- No source files until approved

### Exact scope
- Review benchmark data from Components 1-4, including full-write compression time and output size.
- Decide whether new PGN imports may use a schema-versioned PGN-specific format by default, or only behind a CLI/config flag, while old dense shards remain readable.
- Decide whether training must support mixed dense and sparse PGN manifests in one command, or only one manifest format per run.

### Non-goals
- Do not implement sparse policy or legal-mask storage in this gate.
- Do not remove support for current dense shards.

### Tests/benchmarks to add/run
- No new code tests.
- Run/collect:
  - dry-run and full-write benchmark reports for at least `--max-records 100` and preferably `1000` on the target corpus.

### Risks/stop conditions
- Stop if the parent/user requires current `load_self_play_dataset()` compatibility for all PGN shards; then skip Components 6-7 and proceed to Component 8 using dense storage.
- Stop if disk capacity is already insufficient for dense full-write benchmarks; this strengthens the case for sparse storage but still requires approval.

### Compact worker handoff prompt
> Review benchmark results from the Python fast path and decide whether new PGN imports can use a schema-versioned PGN-specific sparse format. Do not edit source files. Return a clear decision: keep dense only, add sparse behind a flag, or make sparse the new PGN default while preserving old dense shard loading.

---

## Component 6: Schema-versioned sparse policy targets

### Goal
Reduce PGN shard size and write/compression CPU by storing one played action index per sample instead of a dense one-hot `mcts_policies` row, while keeping existing dense self-play and dense PGN shards compatible.

### Files likely changed
- `src/tinychess/nn/pgn_dataset.py`
- `src/tinychess/nn/train.py`
- `scripts/pgn_ingest.py`
- `scripts/pgn_ingest_benchmark.py`
- `tests/nn/test_pgn_dataset.py`
- Potentially `tests/nn/test_train.py` if present, otherwise extend PGN dataset training tests
- `docs/pgn-ingestion.md`

### Exact scope
- Add an approved PGN-specific shard format/version, e.g. manifest schema `tinychess-pgn-manifest-v2` or a manifest `sample_format` field with explicit versioning.
- Add `PgnIngestConfig`/CLI option for storage format, with the default determined by the Component 5 decision.
- For sparse-policy PGN shards, write `positions`, dense `legal_masks` for now, `policy_indices` (`uint16` is sufficient for 0..4671, `uint32` is safer if preferred), and `outcomes`.
- Add a PGN shard loader/batch path that does not pretend sparse-policy shards are `SelfPlayDataset` instances unless converted intentionally.
- Update training so sparse-policy batches compute policy cross-entropy via gather/index selection from masked log-probs, while dense self-play training remains unchanged.
- Keep old dense PGN manifests/shards readable and trainable.

### Non-goals
- Do not change self-play dataset schema.
- Do not sparse legal masks in this component.
- Do not change action-space version or policy head shape.
- Do not remove dense one-hot support.

### Tests/benchmarks to add/run
- Add ingest tests for sparse-policy shards verifying keys/dtypes, manifest version/format, sample counts, and `policy_indices` match the played UCI moves.
- Add training smoke tests for sparse-policy PGN manifests.
- Add loss parity test: dense one-hot loss and sparse-index loss match on the same logits/masks/targets.
- Run:
  - `uv run pytest tests/nn/test_pgn_dataset.py tests/nn/test_encode.py`
  - `uv run pytest` focused training tests that cover `train_from_directory()`
  - Full-write benchmark comparing dense vs sparse-policy on the same record limit.

### Risks/stop conditions
- Training code may become too branchy if dense and sparse datasets share one in-memory class. Stop and introduce a small batch-provider abstraction rather than overloading `SelfPlayDataset` incorrectly.
- Stop if old dense PGN manifests no longer train or `load_self_play_dataset()` behavior changes.
- Be explicit in docs that sparse-policy PGN shards are supervised PGN labels, not MCTS visit distributions.

### Compact worker handoff prompt
> Implement the approved schema-versioned sparse PGN policy format. Store `policy_indices` instead of dense one-hot `mcts_policies` for PGN shards only, update manifest/versioning and training to compute sparse policy cross-entropy, and keep dense self-play plus old dense PGN shards working. Do not sparse legal masks yet. Add loader/training/loss parity tests, update docs and CLI, and benchmark dense vs sparse-policy full writes.

---

## Component 7: Schema-versioned sparse legal masks

### Goal
Reduce PGN shard size further by storing legal action indices as a ragged sparse representation and densifying only per training batch.

### Files likely changed
- `src/tinychess/nn/pgn_dataset.py`
- `src/tinychess/nn/train.py`
- `scripts/pgn_ingest.py`
- `scripts/pgn_ingest_benchmark.py`
- `tests/nn/test_pgn_dataset.py`
- `tests/nn/test_encode.py`
- `docs/pgn-ingestion.md`

### Exact scope
- Add a new explicit PGN sample format/version building on Component 6.
- Store legal moves as:
  - `legal_indices`: flat `uint16`/`uint32` action-index array
  - `legal_offsets`: integer offsets of length `sample_count + 1`
- Ensure `legal_offsets[i]:legal_offsets[i + 1]` reconstructs the legal actions for sample `i`.
- Densify legal masks inside the PGN training batch loader only, not at shard load time for the whole shard.
- Preserve support for dense `legal_masks` shards and sparse-policy-with-dense-mask shards.

### Non-goals
- Do not change model masking semantics.
- Do not alter legal move generation or action mapping.
- Do not remove dense mask support.
- Do not introduce multiprocessing here.

### Tests/benchmarks to add/run
- Add sparse legal reconstruction tests comparing to dense masks for representative positions including castling and underpromotion.
- Add PGN ingest/load/train smoke for sparse-policy+sparse-legal format.
- Add validation tests for malformed offsets/indices if a loader is introduced.
- Run:
  - `uv run pytest tests/nn/test_encode.py tests/nn/test_pgn_dataset.py`
  - Full-write benchmark comparing dense, sparse-policy, and sparse-policy+sparse-legal formats.

### Risks/stop conditions
- Batch-time densification must not allocate a full-shard dense mask. Stop if implementation densifies the entire shard and loses memory benefits.
- Ensure action indices fit chosen dtype; `uint16` fits current 4672 action space, but loader must validate action-space metadata.
- Stop if training loss or mask semantics diverge from dense tests.

### Compact worker handoff prompt
> Add schema-versioned sparse legal masks for PGN shards. Store ragged legal action indices/offsets, densify only per training batch, and preserve all older dense/sparse-policy formats. Add reconstruction, malformed-loader, ingest, and training smoke tests. Do not change model/action semantics or self-play schema. Benchmark dense vs sparse formats including full-write size/time.

---

## Component 8: Multiprocessing chunk import

### Goal
Parallelize PGN record processing after the single-process algorithm and storage format are stable, while preserving deterministic manifests and avoiding expensive IPC of per-sample Python objects.

### Files likely changed
- `src/tinychess/nn/pgn_dataset.py`
- `scripts/pgn_ingest.py`
- `scripts/pgn_ingest_benchmark.py`
- `tests/nn/test_pgn_dataset.py`
- `docs/pgn-ingestion.md`

### Exact scope
- Add `PgnIngestConfig.workers: int = 1` and CLI `--workers` with default single-process behavior unchanged.
- Split raw PGN records into deterministic chunks by record index.
- Worker processes should either:
  - write chunk-local temporary shards and return compact metadata, or
  - return NumPy arrays only for bounded chunk sizes; prefer local temp shards to avoid IPC blowups.
- Parent merges/renames shard directories in deterministic chunk/index order and writes one manifest.
- Aggregate counters (`games_read`, `games_written`, `games_skipped`, `samples`, shards) exactly as in single-process mode.
- Support whichever dense/sparse PGN formats have been approved and implemented.

### Non-goals
- Do not parallelize writes to the same shard directory.
- Do not change parser or dataset semantics.
- Do not make multiprocessing the default until benchmark evidence supports it.
- Do not add Swift.

### Tests/benchmarks to add/run
- Add tests that `workers=1` and `workers=2` produce equivalent manifest counters and equivalent loaded training data on a small multi-game PGN.
- Add tests for deterministic shard ordering across repeated `workers=2` runs.
- Add CLI smoke for `scripts/pgn_ingest.py --workers 2`.
- Run:
  - `uv run pytest tests/nn/test_pgn_dataset.py`
  - Full-write benchmark with `--workers 1`, `2`, and a physical-core-like value on the target machine/corpus.

### Risks/stop conditions
- macOS multiprocessing uses spawn; all worker functions/configs must be picklable and module top-level.
- IPC can erase gains; stop if arrays are passed through queues at full shard size and switch to worker-local temporary shards.
- Compression/disk bandwidth may become dominant; report scaling honestly rather than increasing worker count blindly.
- Progress reporting may be approximate during parallel mode; document any difference.

### Compact worker handoff prompt
> Add optional multiprocessing PGN import after the single-process fast path. Keep `workers=1` behavior unchanged, chunk raw records deterministically, have workers produce shard-ready outputs without large per-sample IPC, and merge manifests in stable order. Add equivalence/determinism tests and CLI/docs. Benchmark full-write scaling for 1/2/N workers. Do not change parser semantics, storage semantics, or add Swift.

---

## Component 9: Swift reconsideration gate (deferred)

### Goal
Only consider Swift acceleration if Python fast path, sparse storage, and multiprocessing still leave legal move/SAN replay as the dominant bottleneck with benchmark evidence.

### Files likely changed
- None by default
- If approved later: likely `swift/`, Python bridge files, parity tests, and benchmarks

### Exact scope
- Review post-Component-8 benchmark profiles.
- Swift is justified only for a narrow, parity-tested target such as batch SAN parse/replay/legal-action-index generation.
- Any Swift plan must include Python reference parity tests and `(cd swift && swift test)` validation.

### Non-goals
- No Swift implementation in the current optimization sequence.
- No replacing the Python engine as correctness reference.

### Tests/benchmarks to add/run
- No tests unless a later Swift plan is approved.

### Risks/stop conditions
- Stop if the remaining bottleneck is compression, disk IO, Python process orchestration, or training loader memory rather than engine hot paths.
- Stop if parity tests cannot cover all PGN/SAN edge cases already supported by Python.

### Compact worker handoff prompt
> Review post-Python optimization benchmarks and decide whether Swift is justified. Do not implement Swift. Only recommend Swift if profiles show legal move/SAN replay remains dominant after Python fast path, sparse storage, and multiprocessing; define a narrow parity-tested target if so.

---

## Files to Modify
- `scripts/pgn_ingest.py` - add CLI options for workers/storage format only in later components.
- `scripts/pgn_ingest_benchmark.py` - add full-write mode, update phase timings for NumPy/replay/trace/sparse/parallel paths.
- `src/tinychess/engine/pgn.py` - refactor SAN parsing to expose parser trace while preserving existing APIs and semantics.
- `src/tinychess/engine/pgn_stream.py` - add traced ingestion parser entry point while keeping sanitizer boundary explicit.
- `src/tinychess/nn/encode.py` - add NumPy-native encoders and mask helpers.
- `src/tinychess/nn/__init__.py` - export NumPy helpers only if intentionally public.
- `src/tinychess/nn/pgn_dataset.py` - implement fast replay, trace-based import, optional sparse PGN shard formats, optional multiprocessing.
- `src/tinychess/nn/train.py` - support schema-versioned sparse PGN policy/legal batches after approval.
- `docs/pgn-ingestion.md` - document benchmark modes, fast path behavior, storage formats, and multiprocessing flags.
- `tests/nn/test_encode.py` - add NumPy encoder/mask parity tests.
- `tests/nn/test_pgn_dataset.py` - add ingestion parity, sparse format, training, and multiprocessing tests.
- `tests/test_pgn.py` - add parser trace strict-semantics coverage if trace changes parser internals.
- `tests/test_pgn_stream.py` - add traced sanitizer/strict ingestion parser coverage.
- `tests/test_benchmarks.py` or `tests/test_pgn_ingest_benchmark.py` - add benchmark structure/CLI smoke tests.

## New Files
- `tests/test_pgn_ingest_benchmark.py` - preferred location for PGN-specific benchmark tests if not folded into `tests/test_benchmarks.py`.
- `tests/nn/test_pgn_replay_state.py` - optional focused tests for private replay-state helpers if `tests/nn/test_pgn_dataset.py` becomes too large.
- `plans/<decision-note>.md` - optional sparse-storage decision record after Component 5.

## Dependencies
- Component 1 should run first; all later speedups need comparable baseline and full-write numbers.
- Component 2 is independent of parser/replay changes and should precede replay/trace work to simplify later benchmarks.
- Component 3 depends on Component 2 only for cleaner NumPy integration, but can technically be implemented independently.
- Component 4 depends on Component 3 so traced boards/legal moves can feed a builder that no longer needs `Game.play()` per ply.
- Component 5 depends on benchmark evidence from Components 1-4.
- Component 6 requires explicit approval from Component 5.
- Component 7 depends on Component 6's PGN-specific sparse loader/training path.
- Component 8 should follow the chosen final single-process storage/import path to avoid reworking parallel shard merging repeatedly.
- Component 9 is deferred until after Component 8 benchmarks.

## Risks
- Parser trace refactoring can accidentally broaden or narrow bounded PGN semantics; protect with strict parser and ingestion sanitizer parity tests.
- Fast replay state can produce `games.jsonl` records that fail `load_self_play_dataset()` if final outcome/repetition/counters diverge from `Game.play()`; shard-load tests are mandatory.
- Sparse PGN schemas can break training or confuse self-play compatibility if they reuse `SelfPlayDataset` incorrectly; keep formats explicitly versioned and old dense loading intact.
- Full import may be disk/compression-bound after CPU optimizations; benchmark full-write mode before prioritizing multiprocessing or Swift.
- Multiprocessing can be slower if workers send huge arrays through IPC; prefer worker-local temporary shards and deterministic parent merge.
- Generated benchmark outputs, PGN corpora, shards, and checkpoints must not be committed under `data/` unless explicitly requested.
