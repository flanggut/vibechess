Inherited decisions:
- Python engine is correctness reference; Swift is optional acceleration and should be gated by Python parity tests + benchmark evidence.
- PGN core parser remains strict; ingestion sanitizer may tolerate public PGN annotations, but boundary must stay explicit.
- Current dry-run benchmark excludes shard writing, so the real full import may be slower than the 29.7h estimate.
- Current dataset schema stores dense `positions`, dense `legal_masks`, dense one-hot `mcts_policies`, and outcomes.

Diagnosis:
- The main bottleneck is repeated full legal-move generation and immutable `Game.play()` overhead.
- PGN parsing resolves every SAN by generating legal moves, applying candidates, and checking `+/#`; then ingestion replays the same moves and recomputes legal moves again.
- `Game.play()` is especially expensive because it:
  - checks `outcome`, which calls `legal_moves` again;
  - recomputes `legal_moves` for move validation;
  - copies full position/move history tuples on every ply.
- The benchmark understates the full import cost because it does not write compressed NPZ shards.
- Full dataset size is likely huge: ~27M samples from current rate. Dense `legal_masks` + dense policy arrays alone are very large, and `np.savez_compressed` will add CPU cost.

Drift / contradiction check:
- Jumping straight to Swift conflicts with the project’s Python-first/reference stance unless Python duplication and schema issues are addressed first.
- Treating 29.7h as the real import time is unsafe; actual write/compression time and disk footprint are not included.
- “All options on the table” should not mean broadening PGN semantics or changing policy/action metadata silently.

Recommendation:
1. **First fix duplicate legality/play work in Python. Highest impact, low/medium risk.**
   - Add ingestion-only fast replay path that carries a `Board`/lightweight state, not full immutable `Game` history.
   - During `parse_pgn`, optionally return per-ply data already computed during SAN resolution: move, legal moves, side to move, maybe board before move.
   - Or create `parse_ingest_pgn_samples()` that parses SAN and emits training samples in one pass.
   - Avoid second replay legality check when parser already validated legality.
   - Avoid `Game.play()` in ingestion; use `board.apply_move()` plus minimal clocks/castling/en-passant state.
   - Expected impact: likely **2–4x** from removing duplicate `legal_moves` / `Game.play()` cost.

2. **Add a no-outcome/no-history apply path. Very high impact, low risk if scoped.**
   - Current profile shows `Game.play()` calls `outcome -> determine_outcome -> legal_moves` before every move.
   - Add private ingestion helper like `_advance_training_state(state, move, *, legal_already_checked=True)` that updates board, clocks, side, repetition only if needed.
   - PGN result already supplies labels; import does not need FIDE draw detection per ply.
   - Expected impact: **1.5–2.5x** alone.

3. **Replace dense one-hot policy storage with played action indices for PGN datasets. High impact, medium risk.**
   - For supervised PGN pretraining, policy target is one move index, not an MCTS distribution.
   - Store `policy_indices: uint16/int32` instead of dense `(N, 4672)` float32.
   - During training, compute sparse cross-entropy via gather instead of dense target multiplication.
   - Keep old self-play format compatible; version PGN manifest/schema explicitly.
   - Expected impact: massive disk/RAM/write reduction; import may become much faster once actual writing is included.

4. **Consider sparse/legal-index storage for legal masks. High impact, medium risk.**
   - Legal mask is sparse, around tens of moves per position, but stored as 4672 floats.
   - Store ragged legal action indices using offsets + flat `uint16` indices, or generate legal masks on batch load.
   - For training, construct dense mask per batch only.
   - Expected impact: huge storage reduction; some training loader work required.

5. **Make encoding NumPy-native for ingestion. Medium impact, low risk.**
   - `encode_game()` currently builds MLX arrays, then benchmark converts to NumPy.
   - Add `encode_game_np()` for dataset generation that fills a preallocated NumPy array directly.
   - Same for legal mask creation: fill a NumPy array from legal action indices without MLX roundtrip.
   - Expected impact: **10–20%** on current dry-run; more if writing avoids MLX sync overhead.

6. **Parallelize by PGN record chunks after single-process fast path. High impact, medium risk.**
   - Multiprocessing is natural: records/games are independent.
   - Parent process streams chunks of raw PGN records to workers; workers return shard-ready arrays or write worker-local temporary shards.
   - Avoid passing per-sample Python objects back to parent; pass chunk files or NumPy arrays.
   - Preserve deterministic manifest ordering by chunk index.
   - Expected impact: near-linear to physical cores until disk/compression bottleneck. On Apple Silicon, probably **4–8x** after Python algorithmic fixes.

7. **Separate benchmark modes. Low risk, necessary.**
   - Add benchmarks for:
     - parse-only;
     - parse+sample encode;
     - full ingest with shard writing;
     - compressed vs uncompressed/sparse storage;
     - multiprocessing scaling.
   - Current dry-run is useful but incomplete.

8. **Swift should not be first move.**
   - Swift engine hot paths may help later, but current waste is architectural: duplicated legality, dense storage, MLX→NumPy conversion, immutable history.
   - Implementing Swift before fixing these risks accelerating the wrong thing.
   - Revisit Swift only after:
     - one-pass Python importer exists;
     - dense policy/mask storage decision is made;
     - multiprocessing benchmark identifies legal move generation as remaining dominant bottleneck.
   - If still needed, Swift target should be narrow: batch SAN parse/replay/legal action index generation with Python parity tests.

Risks:
- One-pass ingestion parser may diverge from strict PGN parser behavior if not tested against existing `parse_pgn`.
- Sparse dataset schema requires train/load changes and metadata versioning.
- Multiprocessing can be defeated by IPC if passing too much data between processes.
- Full import may hit disk capacity before CPU time if dense schema remains.

Need from main agent:
- Decide whether preserving current dense self-play shard format for PGN import is required. If not, sparse PGN schema is likely the biggest practical win.

Suggested execution prompt:
- Implementation handoff is warranted if the main agent approves Python-first optimization.

Prompt:
“Implement a Python-first PGN import speedup. Do not use Swift yet. Add an ingestion-specific fast path that avoids duplicate replay/legal checks and avoids `Game.play()` history/outcome overhead. Add focused benchmarks comparing old vs new on `--max-records 100/1000`. Keep strict core PGN behavior unchanged and add tests proving imported moves/sample counts match the existing path on representative PGNs. Report speedup and any schema implications; do not change dataset schema unless explicitly approved.”