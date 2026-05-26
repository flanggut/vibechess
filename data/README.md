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

`metadata.json` uses schema `tinychess-selfplay-v1` and includes the engine
version, git commit when available, action-space version, encoder version, model
checkpoint id, generation settings, sample count, and game count.

`samples.npz` is a compressed NumPy-compatible tensor batch with:

- `positions`: `[N, 20, 8, 8]` float32 encoded positions.
- `legal_masks`: `[N, 4672]` float32 legal-action masks.
- `mcts_policies`: `[N, 4672]` float32 MCTS visit-count policy targets.
- `outcomes`: `[N]` float32 final outcomes from each sample side-to-move perspective.

`games.jsonl` stores one JSON object per generated game with UCI moves, final
FEN, outcome reason, winner, and ply count.

Example:

```bash
uv run python scripts/self_play.py --games 1 --max-plies 8 --simulations 1 --output data/selfplay/smoke
```

## Checkpoint Policy

Checkpoint sidecars include schema version, model config, action-space version,
training step, optimizer state availability, and notes. Training/checkpoint
production is deferred to WP15.
