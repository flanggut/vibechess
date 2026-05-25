# AI Plan

WP09 added the shared player interface and random-player baseline. WP10 added the classical MCTS baseline. WP11 added neural input encoding and fixed policy action mapping. This document records the implemented foundations plus the planned direction from `PLAN.md` for later work packages.

## Planned Scope

The project will implement:

- A common player interface. (Implemented in `tinychess.ai.player.Player`.)
- A random player baseline. (Implemented in `tinychess.ai.player.RandomPlayer`.)
- A classical MCTS baseline. (Implemented in `tinychess.ai.mcts.MCTSPlayer`.)
- An AlphaZero-style neural MCTS player using a policy/value network and PUCT search.
- MLX-based training and inference for Apple Silicon macOS.

## Planned Neural Design

The planned neural player uses:

- Board-to-tensor encoding. (Implemented in `tinychess.nn.encode`.)
- A fixed AlphaZero-style 8 x 8 x 73 = 4672 action space. (Implemented and versioned as `az-8x8x73-v1`.)
- Legal move masks before policy normalization and search expansion. (Implemented for `Game.legal_moves`.)
- A policy head for move priors.
- A value head for side-to-move outcome prediction.
- Self-play datasets with versioned metadata.
- MLX-native checkpoints with sidecar metadata.

## Current Status

Implemented:

- `Player`: a typed protocol with `select_move(game: Game) -> Move` for human, random, MCTS, and neural-MCTS players.
- `RandomPlayer`: selects only from `Game.legal_moves`, uses a local deterministic RNG when seeded or provided, and raises `NoLegalMoveError` for terminal/no-legal positions.
- `play_game`: a simple player-vs-player simulation helper for AI-vs-AI smoke tests.
- `MCTSConfig`: simulation count, optional wall-clock limit, optional node budget, rollout cap, exploration constant, and seed.
- `MCTSPlayer`: a correctness-first classical MCTS implementation with adversarial UCB1 selection, legal-move expansion, random rollouts, and value backup from the root side's perspective. It uses only public `Game` legal-move and transition APIs.
- `tinychess.nn.encode`: deterministic MLX-native `[20][8][8]` position tensor encoding, AlphaZero-style 4672-action move mapping, and MLX length-4672 legal move masks.

The terminal `play` command accepts `mcts` as a player kind, and `scripts/mcts_benchmark.py` reports MCTS simulations/sec from the starting position. WP11 does not add a neural player, model, checkpointing, training, or neural MCTS search.

Planned work packages:

- WP12: MLX policy/value network.
- WP13: Neural PUCT MCTS.
- WP14-WP16: self-play, training, and evaluation.

The initial goal is a functional learning/search pipeline, not competitive chess strength.
