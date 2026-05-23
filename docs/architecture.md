# Architecture

## Current State

The project is a Python-first chess engine and AI workspace targeting Apple Silicon macOS. Swift, MLX training, PGN, FEN, UCI, and AI components are planned for later work packages.

Implemented work packages:

- WP01: Python project bootstrap.
- WP02: Core board, square, piece, color, and move primitives.
- WP03: Legal move generation, special move handling, and perft benchmark.

## Package Layout

```text
src/tinychess/
├── __init__.py
├── cli.py
└── engine/
    ├── __init__.py
    ├── board.py
    ├── legal_moves.py
    ├── move.py
    ├── piece.py
    └── square.py
```

## Engine Boundaries

The engine currently owns:

- 0..63 square indexing with `a1 == 0` and `h8 == 63`.
- Piece/color/move primitives.
- Immutable board snapshots with compact tuple-backed square storage.
- Side to move, castling rights, and en passant target state.
- Pseudo-legal move generation.
- Legal move filtering by check safety.
- Minimal `Board.apply_move()` for legal move generation and perft.

The engine does **not** yet own:

- Full game history.
- Move clocks.
- Draw/outcome semantics.
- FEN parsing/serialization beyond placement-style helpers.
- PGN or UCI protocol support.
- AI/player abstractions.

Those are covered by later work packages in `PLAN.md`.

## Tooling

Use `uv` for reproducible setup and commands:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy
uv run tinychess --help
```

A lightweight perft benchmark is available:

```bash
uv run python scripts/perft.py 3
```
