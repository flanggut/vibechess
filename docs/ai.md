# AI Plan

AI work has not started yet. This document records the planned direction from `PLAN.md` so future work packages can fill in implementation details.

## Planned Scope

The project will implement:

- A common player interface.
- A random player baseline.
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

Not implemented. Planned work packages:

- WP09: Player interface and random player.
- WP10: Classical MCTS baseline.
- WP11: MLX position encoder and policy mapping.
- WP12: MLX policy/value network.
- WP13: Neural PUCT MCTS.
- WP14-WP16: self-play, training, and evaluation.

The initial goal is a functional learning/search pipeline, not competitive chess strength.
