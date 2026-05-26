# tinychess

A small chess engine and neural-MCTS AI project for Apple Silicon macOS.

## Status

Implemented:

- WP01: Python project bootstrap.
- WP02: Core board, square, piece, color, and move primitives.
- WP03: Legal move generation, special moves, and perft benchmark.
- WP04: Game state, history, outcomes, complete-game simulation, and random-game benchmark.
- WP05: FEN parsing, serialization, round-trip tests, and fixture positions.
- WP06: Bounded PGN/SAN parsing and writing.
- WP07: Terminal board rendering and CLI play loop for human/random games.
- WP08: Bounded UCI protocol loop with random legal `bestmove` output.
- WP09: Shared `Player` protocol, deterministic `RandomPlayer`, and random-vs-random simulation helper.
- WP10: Classical MCTS baseline, configurable search budgets, MCTS-vs-random smoke path, and simulations/sec benchmark.
- WP11: Position tensor encoder, versioned 4672-action AlphaZero-style policy mapping, and legal move masks.
- WP12: Configurable MLX policy/value network, inference wrapper, checkpoint sidecar metadata, and inference latency benchmark.
- WP13: Neural PUCT MCTS with policy priors, value backup, illegal move masking, and temperature selection.
- WP14: Self-play game generation with versioned compressed NPZ tensors plus JSON/JSONL metadata.

Next planned work package: WP15, Training Loop.

## Requirements

- Apple Silicon macOS
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Development setup

```bash
uv sync --dev
```

## CI-ready checks

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run tinychess --help
uv run python scripts/perft.py 3
uv run python scripts/random_game.py --seed 7 --max-plies 40
uv run python scripts/mcts_benchmark.py --simulations 25 --seed 7
uv run python scripts/mlx_inference_benchmark.py --iterations 25 --warmup 5
uv run python scripts/self_play.py --games 1 --max-plies 8 --simulations 1 --output data/selfplay/smoke
```

## Current CLI

```bash
uv run tinychess --help
uv run tinychess --version
uv run tinychess play
uv run tinychess play --white human --black random
uv run tinychess play --white random --black random --seed 7 --max-plies 40
uv run tinychess play --white mcts --black random --seed 7 --mcts-simulations 25 --max-plies 40
uv run tinychess uci
uv run tinychess uci --seed 7
```

The `play` command renders the board in the terminal, shows side to move, castling
en-passant and move-counter status, and accepts human moves in UCI long algebraic
notation such as `e2e4` or `e7e8q`. Invalid or illegal moves are rejected with a
message and another prompt. Player kinds are `human`, `random`, and the classical
`mcts` baseline.

The `uci` command runs a bounded Universal Chess Interface loop. It supports
`uci`, `isready`, `ucinewgame`, `position startpos [moves ...]`,
`position fen ... [moves ...]`, `go`, `stop`, and `quit`. The current move source
is a local random legal selector; use `--seed` for deterministic selections.
Terminal or no-legal-move positions return `bestmove 0000`.

Deferred UCI features: pondering, rich `setoption`, MultiPV, advanced time
management, detailed `info` streaming, tablebases, and opening books.

## Engine Example

```python
from tinychess.ai import MCTSConfig, MCTSPlayer, RandomPlayer, play_game
from tinychess.engine import Board, Game, legal_moves, parse_fen, parse_pgn, perft, random_move_selector, simulate_game
from tinychess.nn import ACTION_SPACE_SIZE, PolicyValueInference, PolicyValueNet, encode_game, legal_move_mask
from tinychess.nn.self_play import SelfPlayConfig, generate_self_play_dataset, save_self_play_dataset

board = Board.starting_position()
print(len(legal_moves(board)))  # 20
print(perft(board, 3))          # 8902
print(parse_fen(board.to_fen()).board == board)  # True
print(parse_pgn('[Result "*"]\n\n1. e4 e5 *').moves[0].to_uci())  # e2e4

# Simulate a deterministic random game with a ply cap.
game = simulate_game(random_move_selector(seed=7), max_plies=40)
print(len(game.moves), game.outcome.reason.value)

# Or use the shared player API used by AI integrations.
game = play_game(RandomPlayer(seed=1), RandomPlayer(seed=2), max_plies=40)
print(len(game.moves), game.outcome.reason.value)

move = MCTSPlayer(MCTSConfig(simulations=25, seed=1)).select_move(Game.new())
print(move.to_uci())

# Neural-input foundations encode directly to MLX arrays.
encoded = encode_game(Game.new())
mask = legal_move_mask(Game.new())
print(encoded.shape, mask.shape, ACTION_SPACE_SIZE)  # (20, 8, 8) (4672,) 4672

# WP12 policy/value inference returns a masked policy and side-to-move value.
result = PolicyValueInference(PolicyValueNet()).predict(Game.new())
print(result.policy.shape, result.value)

# Generate a tiny self-play dataset for smoke/testing purposes.
inference = PolicyValueInference(PolicyValueNet())
dataset = generate_self_play_dataset(inference, SelfPlayConfig(games=1, max_plies=2))
save_self_play_dataset(dataset, "data/selfplay/smoke")
```

## Documentation

- `PLAN.md`: roadmap, work packages, and completed status.
- `docs/architecture.md`: current package and component boundaries.
- `docs/engine.md`: board representation, moves, legal move generation, and perft.
- `docs/ai.md`: planned AI/neural-MCTS direction.
- `docs/swift-backend.md`: planned Swift acceleration strategy.
- `data/README.md`: future dataset/checkpoint policy.
