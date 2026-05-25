# Architecture

## Current State

The project is a Python-first chess engine and AI workspace targeting Apple Silicon macOS. FEN, bounded PGN, bounded UCI, terminal play, classical MCTS, and the first MLX policy/value model foundations are implemented. Swift, MLX training, and stronger AI components are planned for later work packages.

Implemented work packages:

- WP01: Python project bootstrap.
- WP02: Core board, square, piece, color, and move primitives.
- WP03: Legal move generation, special move handling, and perft benchmark.
- WP04: Game state/history, outcomes, complete-game simulation, and random-game benchmark.
- WP05: FEN parsing/serialization.
- WP06: Bounded PGN parsing/writing.
- WP07: Terminal UI and CLI play loop.
- WP08: Bounded UCI protocol with random legal best moves.
- WP09: Shared player protocol and random player.
- WP10: Classical MCTS baseline and simulations/sec benchmark.
- WP11: MLX position encoder, fixed policy action mapping, and legal move masks.
- WP12: MLX policy/value network, inference wrapper, checkpoints, and inference benchmark.

## Package Layout

```text
src/tinychess/
├── __init__.py
├── cli.py
├── engine/
│   ├── __init__.py
│   ├── board.py
│   ├── fen.py
│   ├── game.py
│   ├── legal_moves.py
│   ├── move.py
│   ├── outcome.py
│   ├── pgn.py
│   ├── piece.py
│   └── square.py
├── ai/
│   ├── __init__.py
│   ├── mcts.py
│   ├── player.py
│   └── search_config.py
├── nn/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── encode.py
│   └── model.py
├── protocols/
│   ├── __init__.py
│   └── uci.py
└── ui/
    ├── __init__.py
    ├── render.py
    └── terminal.py
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
- `Game` snapshots with immutable position/move history and copied repetition state.
- Halfmove and fullmove counters at game level.
- Checkmate, stalemate, and pragmatic draw outcomes.
- Complete-game simulation with caller-provided move selectors.

Protocol support currently includes a bounded synchronous UCI loop in
`tinychess.protocols.uci`. It accepts standard handshake/readiness commands,
`ucinewgame`, `position startpos [moves ...]`, `position fen ... [moves ...]`,
`go`, `stop`, and `quit`. `go` returns a random legal `bestmove` or `bestmove
0000` for terminal/no-legal positions.

The AI layer owns the `Player` protocol, `RandomPlayer`, `MCTSPlayer`, and search configuration. These players interact with positions through public `Game.legal_moves` and `Game.play()` APIs rather than mutating engine internals.

The engine does **not** yet own:
- Full UCI features such as pondering, rich options, MultiPV, advanced time
  management, detailed info streaming, tablebases, or opening books.
- Strict FIDE claim-vs-automatic draw semantics.

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
uv run python scripts/random_game.py --seed 7 --max-plies 40
uv run python scripts/mcts_benchmark.py --simulations 25 --seed 7
uv run python scripts/mlx_inference_benchmark.py --iterations 25 --warmup 5
```
