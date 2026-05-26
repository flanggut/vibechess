# AI Plan

WP09 added the shared player interface and random-player baseline. WP10 added the classical MCTS baseline. WP11 added neural input encoding and fixed policy action mapping. WP12 added the first MLX policy/value network, inference wrapper, and checkpoint format. WP13 added a functional neural PUCT MCTS player. WP14 added self-play dataset generation, and WP15 added the first MLX training loop. This document records the implemented foundations plus the planned direction from `PLAN.md` for later work packages.

## Planned Scope

The project will implement:

- A common player interface. (Implemented in `tinychess.ai.player.Player`.)
- A random player baseline. (Implemented in `tinychess.ai.player.RandomPlayer`.)
- A classical MCTS baseline. (Implemented in `tinychess.ai.mcts.MCTSPlayer`.)
- An AlphaZero-style neural MCTS player using a policy/value network and PUCT search. (Implemented in `tinychess.ai.neural_mcts.NeuralMCTSPlayer`.)
- MLX-based training and inference for Apple Silicon macOS.

## Planned Neural Design

The planned neural player uses:

- Board-to-tensor encoding. (Implemented in `tinychess.nn.encode`.)
- A fixed AlphaZero-style 8 x 8 x 73 = 4672 action space. (Implemented and versioned as `az-8x8x73-v1`.)
- Legal move masks before policy normalization and search expansion. (Implemented for `Game.legal_moves`.)
- A policy head for move priors. (Implemented in `tinychess.nn.model.PolicyValueNet`.)
- A value head for side-to-move outcome prediction. (Implemented in `tinychess.nn.model.PolicyValueNet`.)
- PUCT search using neural policy priors, legal-move expansion only, side-to-move value backup, and visit-count temperature move selection. (Implemented in `tinychess.ai.neural_mcts`.)
- Self-play datasets with versioned metadata. (Implemented in `tinychess.nn.self_play`.)
- Policy/value training losses, basic JSONL metrics, and smoke-friendly checkpoint writes. (Implemented in `tinychess.nn.train` and `scripts/train.py`.)
- MLX-native checkpoints with sidecar metadata. (Implemented in `tinychess.nn.checkpoint`.)

## Current Status

Implemented:

- `Player`: a typed protocol with `select_move(game: Game) -> Move` for human, random, MCTS, and neural-MCTS players.
- `RandomPlayer`: selects only from `Game.legal_moves`, uses a local deterministic RNG when seeded or provided, and raises `NoLegalMoveError` for terminal/no-legal positions.
- `play_game`: a simple player-vs-player simulation helper for AI-vs-AI smoke tests.
- `MCTSConfig`: simulation count, optional wall-clock limit, optional node budget, rollout cap, exploration constant, and seed.
- `MCTSPlayer`: a correctness-first classical MCTS implementation with adversarial UCB1 selection, legal-move expansion, random rollouts, and value backup from the root side's perspective. It uses only public `Game` legal-move and transition APIs.
- `tinychess.nn.encode`: deterministic MLX-native `[20][8][8]` position tensor encoding, AlphaZero-style 4672-action move mapping, and MLX length-4672 legal move masks.
- `PolicyValueNet`: a small configurable residual CNN that returns 4672 policy logits and a side-to-move value in `[-1, 1]`.
- `PolicyValueInference`: single-position inference that can normalize over all actions or mask probabilities to legal moves only.
- `NeuralMCTSConfig`: simulation, time, node, PUCT exploration, temperature, and seed settings for neural search.
- `NeuralMCTSPlayer`: AlphaZero-style PUCT search that requests masked neural policy probabilities, expands only legal moves, backs up values from each node's side-to-move perspective, and selects from visit counts with configurable temperature.
- `tinychess.nn.checkpoint`: MLX `weights.safetensors` save/load helpers with `metadata.json` sidecars containing schema, model config, encoder/action-space versions, training step, optimizer-state availability, and notes.
- `tinychess.nn.self_play`: versioned compressed NPZ datasets containing encoded positions, legal masks, MCTS policy targets, outcome targets, metadata, and game records.
- `tinychess.nn.train`: masked policy cross-entropy, value MSE, a tiny MLX optimizer loop, `metrics.jsonl`, `training.json`, and `checkpoint-final` output.

The terminal `play` command accepts `mcts` as a player kind. `scripts/mcts_benchmark.py` reports MCTS simulations/sec from the starting position, `scripts/mlx_inference_benchmark.py` reports policy/value inference latency, `scripts/self_play.py` creates small datasets, and `scripts/train.py` trains smoke-friendly checkpoints.

Planned work packages:

- WP16: evaluation harness and checkpoint-vs-baseline comparisons.

The initial goal is a functional learning/search pipeline, not competitive chess strength.
