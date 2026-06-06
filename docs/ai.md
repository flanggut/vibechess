# AI Plan

WP09 added the shared player interface and random-player baseline. WP10 added the classical MCTS baseline. WP11 added neural input encoding and fixed policy action mapping. WP12 added the first MLX policy/value network, inference wrapper, and checkpoint format. WP13 added a functional neural PUCT MCTS player. WP14 added self-play dataset generation, WP15 added the first MLX training loop, and WP16 added the first evaluation harness. This document records the implemented foundations plus planned AI direction.

## Scope

The project includes:

- A common player interface. (Implemented in `tinychess.ai.player.Player`.)
- A random player baseline. (Implemented in `tinychess.ai.player.RandomPlayer`.)
- A classical MCTS baseline. (Implemented in `tinychess.ai.mcts.MCTSPlayer`.)
- An AlphaZero-style neural MCTS player using a policy/value network and PUCT search. (Implemented in `tinychess.ai.neural_mcts.NeuralMCTSPlayer`.)
- MLX-based training and inference for Apple Silicon macOS.

## Neural Design

The neural player uses:

- Board-to-tensor encoding. (Implemented in `tinychess.nn.encode`.)
- A fixed AlphaZero-style 8 x 8 x 73 = 4672 action space. (Implemented and versioned as `az-8x8x73-v1`.)
- Legal move masks before policy normalization and search expansion. (Implemented for `Game.legal_moves`.)
- A policy head for move priors. (Implemented in `tinychess.nn.model.PolicyValueNet`.)
- A value head for side-to-move outcome prediction. (Implemented in `tinychess.nn.model.PolicyValueNet`.)
- PUCT search using neural policy priors, legal-move expansion only, side-to-move value backup, and visit-count temperature move selection. (Implemented in `tinychess.ai.neural_mcts`.)
- Self-play datasets with versioned metadata. (Implemented in `tinychess.nn.self_play`.)
- Policy/value training losses, validation split tracking, epoch loss reporting, basic JSONL metrics, and smoke-friendly checkpoint writes. (Implemented in `tinychess.nn.train` and `scripts/train.py`.)
- MLX-native checkpoints with sidecar metadata. (Implemented in `tinychess.nn.checkpoint`.)
- Checkpoint-vs-baseline evaluation with early smoke promotion criteria. (Implemented in `tinychess.ai.evaluation` and `scripts/evaluate.py`.)

## Current Status

Implemented:

- `Player`: a typed protocol with `select_move(game: Game) -> Move` for human, random, MCTS, and neural-MCTS players.
- `RandomPlayer`: selects only from `Game.legal_moves`, uses a local deterministic RNG when seeded or provided, and raises `NoLegalMoveError` for terminal/no-legal positions.
- `play_game`: a simple player-vs-player simulation helper for AI-vs-AI smoke tests.
- `MCTSConfig`: simulation count, optional wall-clock limit, optional node budget, rollout cap, exploration constant, seed, and exact-subtree reuse toggle. `max_rollout_plies=0` is the supported high-simulation static leaf mode and the default; set a positive value to request random rollout plies. `reuse_tree=True` reuses prior classical-MCTS search state only when the next searched game is an exact descendant already present in the tree.
- `MCTSPlayer`: a correctness-first classical MCTS implementation with adversarial UCB1 selection, legal-move expansion, configurable random rollouts, static leaf evaluation, exact-descendant tree reuse, and value backup from the root side's perspective. Static leaf mode is much faster because it skips random rollout moves, but it can change playing strength because the material-only leaf value is more myopic. Random rollout mode remains available and configurable, and no chess outcome or draw semantics are broadened by this search setting. Reused subtrees keep previous visit counts by design, so later searches can be biased by earlier simulations; call `clear_tree()` or set `reuse_tree=False` for fresh-root searches. Per-search node budgets and `MCTSResult.nodes` count newly created nodes only, not nodes already present in an adopted subtree.
- `tinychess.nn.encode`: deterministic MLX-native `[20][8][8]` position tensor encoding, AlphaZero-style 4672-action move mapping, and MLX length-4672 legal move masks.
- `PolicyValueNet`: a small configurable residual CNN that returns 4672 policy logits and a side-to-move value in `[-1, 1]`.
- `PolicyValueInference`: single-position inference that can normalize over all actions or mask probabilities to legal moves only.
- `NeuralMCTSConfig`: simulation, time, node, PUCT exploration, temperature, and seed settings for neural search.
- `NeuralMCTSPlayer`: AlphaZero-style PUCT search that requests masked neural policy probabilities, expands only legal moves, caches each node's legal moves/outcome, stores legal edge priors/statistics lazily, backs up values from each node's side-to-move perspective, and selects from edge visit counts with configurable temperature. Expansion runs one inference at the leaf and records every cached legal move as an edge, but child `Game`/node objects are materialized only when an edge is selected for descent. `NeuralMCTSResult.nodes` counts materialized nodes only (the active root plus any selected children created during the search), and node-budget exhaustion prevents further child materialization; in that case search evaluates the current node instead of creating the selected child, preserving the cap. Root `visit_counts` reports edge visits for all expanded legal root moves, including zero-visit/unmaterialized edges, so policy targets keep an explicit legal-move distribution. It reuses exact-descendant subtrees across calls by matching `Game.moves`, but only already materialized child paths can be adopted; reused subtrees intentionally keep prior visits, so call `clear_tree()` for a fresh-root neural search. Node budgets still count the active root toward the cap, including an adopted root. Within-tree leaf-parallel search was removed because batching multiple unbacked-up leaves from one tree can shift fixed-budget visit distributions; future throughput work should batch inference across independent games while preserving serial per-tree search semantics.
- `tinychess.nn.checkpoint`: MLX `weights.safetensors` save/load helpers with `metadata.json` sidecars containing schema, model config, encoder/action-space versions, training step, optimizer-state availability, and notes.
- `tinychess.nn.self_play`: versioned compressed NPZ datasets containing encoded positions, legal masks, MCTS policy targets, outcome targets, metadata, and game records. Self-play can use neural or classical MCTS labels. Neural self-play also supports an optional throughput-oriented `batch_size` mode that batches root/decision inference across independent games while keeping a separate `NeuralMCTSPlayer` tree/RNG stream per game and preserving the dataset tensor and `games.jsonl` schema.
- `tinychess.nn.pgn_dataset`: external PGN games converted into sharded supervised policy/value datasets with one-hot played-move policy targets.
- `tinychess.nn.train`: masked policy cross-entropy, value MSE, a tiny MLX optimizer loop, default 10% validation holdout when enough samples are available, shard-wise training support with in-memory optimizer continuity, per-epoch loss reporting in `epoch_metrics.jsonl`, `training.json`, and `checkpoint-final` output.
- `tinychess.ai.evaluation`: player-vs-player match runner, checkpoint player loading, random/classical MCTS baseline comparisons, JSON reports, and explicit early promotion criteria.

The terminal `play` command accepts `mcts` as a player kind and exposes `--mcts-rollout-plies`; the default `0` selects static leaf evaluation for high-simulation play, while positive values request random rollout plies. It also accepts checkpoint-backed neural `ai` for either side, for example `uv run tinychess play --white human --black ai --ai-checkpoint data/checkpoints/train-smoke/checkpoint-final --ai-simulations 25`. The `ai` player uses `NeuralMCTSPlayer` and exposes neural search options for simulations, node budget, time limit, temperature, and PUCT exploration. `scripts/mcts_benchmark.py` reports MCTS simulations/sec from the starting position and supports `--fast-leaf` as a convenience alias for `--rollout-plies 0`. `scripts/mlx_inference_benchmark.py` reports policy/value inference latency, `scripts/benchmark.py` combines engine/search/MLX benchmark results with an optional batched-inference measurement and conservative suite-time Swift-acceleration heuristic, `scripts/self_play.py` creates neural/classical MCTS datasets and exposes `--batch-size` for batching work across independent self-play games. It also supports human progress with `--progress auto|always|never`; the default is TTY-only stderr progress, while stdout remains the final machine-readable summary line. `scripts/pgn_ingest.py` converts external PGN collections into shards, `scripts/train.py` trains single datasets or PGN shard manifests, and `scripts/evaluate.py` evaluates checkpoints against random and MCTS baselines. Checkpoint evaluation batching remains deferred; the current batching work targets model inference and self-play dataset throughput without changing default single interactive neural-MCTS semantics.

WP16 promotion criteria are intentionally early smoke/progress validation only. Passing them shows that the learning pipeline can load checkpoints, play legal games, record outcomes, and compare against simple baselines; it does not claim competitive chess strength.

Next AI/training work should focus on stronger supervised pretraining, iterative self-play, larger replay buffers, better evaluation gates, and performance improvements such as batched inference. The current goal remains a functional learning/search pipeline, not competitive chess strength.
