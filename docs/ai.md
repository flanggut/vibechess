# AI Plan

WP09 has added the shared player interface and random-player baseline. This document records the implemented baseline plus the planned direction from `PLAN.md` for later work packages.

## Planned Scope

The project will implement:

- A common player interface. (Implemented in `tinychess.ai.player.Player`.)
- A random player baseline. (Implemented in `tinychess.ai.player.RandomPlayer`.)
- A classical MCTS baseline.
- An AlphaZero-style neural MCTS player using a policy/value network and PUCT search.
- MLX-based training and inference for Apple Silicon macOS.

## Planned Neural Design

The planned neural player uses:

- Board-to-tensor encoding.
- A fixed AlphaZero-style 8 x 8 x 73 = 4672 action space.
- Legal move masks before policy normalization and search expansion.
- A policy head for move priors.
- A value head for side-to-move outcome prediction.
- Self-play datasets with versioned metadata.
- MLX-native checkpoints with sidecar metadata.

## Current Status

Implemented:

- `Player`: a typed protocol with `select_move(game: Game) -> Move` for human, random, MCTS, and neural-MCTS players.
- `RandomPlayer`: selects only from `Game.legal_moves`, uses a local deterministic RNG when seeded or provided, and raises `NoLegalMoveError` for terminal/no-legal positions.
- `play_game`: a simple player-vs-player simulation helper for AI-vs-AI smoke tests.

Planned work packages:

- WP10: Classical MCTS baseline.
- WP11: MLX position encoder and policy mapping.
- WP12: MLX policy/value network.
- WP13: Neural PUCT MCTS.
- WP14-WP16: self-play, training, and evaluation.

The initial goal is a functional learning/search pipeline, not competitive chess strength.
