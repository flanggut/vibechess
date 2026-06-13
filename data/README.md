# Data Directory

This directory is reserved for generated datasets and checkpoints.

## Layout

```text
data/
├── selfplay/
└── checkpoints/
```

Generated self-play datasets are local artifacts and should generally not be
committed unless they are intentionally tiny fixtures.

## Self-Play Dataset Schema

WP14 writes one directory per generation run:

```text
run-dir/
├── samples.npz
├── metadata.json
└── games.jsonl
```

`metadata.json` uses schema `vibechess-selfplay-v2` for newly written datasets and includes the engine
version, git commit when available, action-space version, encoder version, model
checkpoint id, generation settings, sample count, game count, and `policy_target_format`.
Legacy `vibechess-selfplay-v1` dense-policy datasets remain loadable.

`samples.npz` is a compressed NumPy-compatible tensor batch with:

- `positions`: `[N, 20, 8, 8]` float32 encoded positions.
- `legal_masks`: `[N, 4672]` float32 legal-action masks.
- `policy_offsets`: `[N + 1]` int64 CSR row offsets for sparse policy targets.
- `policy_indices`: `[nnz]` int32 action indices for nonzero target probabilities.
- `policy_probabilities`: `[nnz]` float32 nonzero MCTS visit-count policy probabilities.
- `outcomes`: `[N]` float32 final outcomes from each sample side-to-move perspective.

New v2 shards do not store dense `[N, 4672]` `mcts_policies` arrays on disk. The
Python loader still accepts v1 shards with dense `mcts_policies` and exposes a
compatibility `dataset.mcts_policies` property that densifies sparse rows on demand.

`games.jsonl` stores one JSON object per generated game with UCI moves, final
FEN, outcome reason, winner, and ply count.

Example:

```bash
uv run python scripts/self_play.py --games 1 --max-plies 8 --simulations 1 --output data/selfplay/smoke
```

## Training and Checkpoints

WP15 consumes a self-play dataset directory and writes a local training run:

```text
train-run/
├── epoch_metrics.jsonl
├── training.json
└── checkpoint-final/
    ├── weights.safetensors
    └── metadata.json
```

Example:

```bash
uv run python scripts/train.py --dataset data/selfplay/smoke --output data/checkpoints/train-smoke --epochs 1 --batch-size 2
```

`epoch_metrics.jsonl` contains one JSON object per epoch with train/validation
total, policy, and value losses. Checkpoint sidecars include schema version,
model config, action-space version, encoder version, training step, optimizer
state availability, and notes. The checkpoint writer saves model weights and
metadata only; optimizer state is not persisted.
