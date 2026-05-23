# tinychess

A small chess engine and neural-MCTS AI project for Apple Silicon macOS.

## Status

Implemented:

- WP01: Python project bootstrap.
- WP02: Core board, square, piece, color, and move primitives.
- WP03: Legal move generation, special moves, and perft benchmark.
- WP04: Game state, history, outcomes, complete-game simulation, and random-game benchmark.

Next planned work package: WP05, FEN support.

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
```

## Current CLI

```bash
uv run tinychess --help
uv run tinychess --version
```

## Engine Example

```python
from tinychess.engine import Board, Game, legal_moves, perft, random_move_selector, simulate_game

board = Board.starting_position()
print(len(legal_moves(board)))  # 20
print(perft(board, 3))          # 8902

# Simulate a deterministic random game with a ply cap.
game = simulate_game(random_move_selector(seed=7), max_plies=40)
print(len(game.moves), game.outcome.reason.value)
```

## Documentation

- `PLAN.md`: roadmap, work packages, and completed status.
- `docs/architecture.md`: current package and component boundaries.
- `docs/engine.md`: board representation, moves, legal move generation, and perft.
- `docs/ai.md`: planned AI/neural-MCTS direction.
- `docs/swift-backend.md`: planned Swift acceleration strategy.
- `data/README.md`: future dataset/checkpoint policy.
