# Architecture

## Current State

The project is a Python-first chess engine and AI workspace targeting Apple Silicon macOS. FEN, bounded PGN, bounded UCI, terminal play, a JSON-lines GUI backend, a local-first SwiftUI macOS app, classical MCTS, neural MCTS, self-play data generation, the first MLX training loop, smoke-oriented checkpoint evaluation, benchmark reporting, and the initial Swift Package Manager bootstrap are implemented. Swift acceleration logic remains separate from the GUI and is planned for later work packages after benchmark evidence and fixture parity work.

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
- WP12: MLX policy/value network, separated inference wrapper, checkpoints, and inference benchmark.
- WP13: Neural PUCT MCTS player.
- WP14: Self-play dataset generation.
- WP15: MLX policy/value training loop, metrics logging, and checkpoint output.
- WP16: Evaluation harness for checkpoint/player matches against random and classical MCTS baselines with early progress-validation promotion criteria.
- WP17: Full benchmark suite with Swift acceleration recommendation heuristic.
- WP18: Swift package bootstrap with `TinyChessCore` and Swift tests.
- GUI MVP: `tinychess gui-server` plus a SwiftUI `TinyChessMacApp` frontend for human-vs-AI play.

## Package Layout

```text
src/tinychess/
├── __init__.py
├── cli.py
├── profiling.py
├── engine/
│   ├── __init__.py
│   ├── board.py
│   ├── fen.py
│   ├── game.py
│   ├── legal_moves.py
│   ├── move.py
│   ├── outcome.py
│   ├── pgn.py
│   ├── pgn_stream.py
│   ├── piece.py
│   ├── square.py
│   └── transition.py
├── ai/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── mcts.py
│   ├── neural_mcts.py
│   ├── player.py
│   ├── search_config.py
│   └── search_state.py
├── nn/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── encode.py
│   ├── inference.py
│   ├── model.py
│   ├── pgn_dataset.py
│   ├── self_play.py
│   ├── self_play_dataset.py
│   ├── self_play_profile.py  # compatibility re-export for tinychess.profiling
│   └── train.py
├── protocols/
│   ├── __init__.py
│   ├── gui.py
│   └── uci.py
└── ui/
    ├── __init__.py
    ├── render.py
    └── terminal.py
```

The Swift workspace currently lives separately under `swift/`:

```text
swift/
├── Package.swift
├── README.md
├── Sources/
│   ├── TinyChessCore/
│   │   └── TinyChessCore.swift
│   └── TinyChessMacApp/
│       ├── AppState.swift
│       ├── BackendClient.swift
│       ├── BackendModels.swift
│       ├── BoardView.swift
│       ├── ControlsView.swift
│       ├── MoveListView.swift
│       ├── SquareView.swift
│       └── TinyChessMacApp.swift
└── Tests/
    ├── TinyChessCoreTests/
    │   └── TinyChessCoreTests.swift
    └── TinyChessMacAppTests/
        ├── AppStateTests.swift
        ├── BackendClientTests.swift
        ├── BackendModelsTests.swift
        ├── BoardViewTests.swift
        ├── ControlsMoveListTests.swift
        └── TinyChessMacAppTests.swift
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
- Engine-owned transition primitives in `tinychess.engine.transition` for shared position keys, capture detection, known-legal state advancement, and pragmatic outcome evaluation. These helpers are an internal engine boundary for `Game`, search-state, bounded PGN parser, and PGN ingestion replay parity, not a protocol expansion, and are intentionally not re-exported from `tinychess.engine.__init__` yet.
- Checkmate, stalemate, and pragmatic draw outcomes.
- Complete-game simulation with caller-provided move selectors.

Protocol support currently includes two separate frontends:

- A bounded synchronous UCI loop in `tinychess.protocols.uci`. It accepts
  standard handshake/readiness commands, `ucinewgame`, `position startpos
  [moves ...]`, `position fen ... [moves ...]`, `go`, `stop`, and `quit`. `go`
  returns a random legal `bestmove` or `bestmove 0000` for terminal/no-legal
  positions.
- A GUI-specific JSON-lines loop in `tinychess.protocols.gui`, exposed through
  `uv run tinychess gui-server`. It reads one request object per line and writes
  one response object per line. Commands include `hello`, `newGame`, `state`,
  `makeMove`, `aiMove`, `undo`, `setAiConfig`, and `quit`. State-bearing
  responses include FEN, occupied squares, side to move, legal moves, legal
  destinations grouped by source square, move history, last move, counters, and
  outcome. The GUI protocol is intentionally not UCI: it exists so the native app
  can render and resync state without duplicating chess rules or broadening the
  bounded UCI surface.

The AI layer owns the `Player` protocol, `RandomPlayer`, `MCTSPlayer`, neural PUCT MCTS, search configuration, and the WP16 smoke evaluation harness. These players interact with positions through public `Game.legal_moves` and `Game.play()` APIs rather than mutating engine internals. The GUI backend reuses these players for `aiMove`: random and classical MCTS work without external assets, while neural MCTS remains optional and requires a local checkpoint path. The evaluation harness runs small player/checkpoint matches from fresh games, compares checkpoints against random and classical MCTS baselines, and records early promotion decisions as progress validation rather than evidence of competitive strength.

The NN layer keeps encoder/action-space definitions in `tinychess.nn.encode`, model architecture and checkpoint metadata configuration in `tinychess.nn.model`, and policy/value result DTOs plus `PolicyValueInference` in `tinychess.nn.inference`. Self-play game generation stays in `tinychess.nn.self_play`, while self-play dataset constants, record/metadata DTOs, save/load/merge helpers, and validation live in `tinychess.nn.self_play_dataset`; historical dataset IO imports from `tinychess.nn.self_play` remain compatible. Historical imports of inference symbols from `tinychess.nn.model` and top-level `tinychess.nn` also remain compatible, but internal callers should prefer `tinychess.nn.inference` for inference wrappers and `tinychess.nn.self_play_dataset` for dataset IO. These splits do not change the MLX model architecture, tensor shapes, 4672-action policy space, value range, checkpoint metadata, or self-play/PGN dataset file schemas.

Profiling instrumentation lives at `tinychess.profiling` so engine and AI hot paths can record timings, counters, and distributions without importing the neural-network package. The historical `tinychess.nn.self_play_profile` module is a compatibility re-export only; engine and AI code should not depend on `tinychess.nn` for profiling.

The native macOS app is a SwiftUI frontend, not a Swift chess engine. It maps
squares, renders Unicode pieces, displays state and controls, and sends
UCI-style move strings to the backend. Legal move generation, move application,
outcome detection, undo replay, and AI selection remain in Python. `TinyChessCore`
continues to be acceleration scaffolding only until future benchmark-driven
parity work proves a Swift implementation.

Current GUI MVP limitations:

- Local development launch assumes `uv run tinychess gui-server` is available
  from the repository checkout.
- Search is synchronous in the Python backend; the Swift app keeps the UI
  responsive by awaiting backend work outside direct button handlers, but
  protocol-level cancellation/progress streaming is deferred.
- Promotion is auto-queen for the MVP; there is no native promotion chooser or
  underpromotion UI yet.
- Move history is UCI strings, not SAN/PGN; PGN import/export and save/load are
  deferred.
- There is no drag-and-drop, clocks, opening book, tablebase, or external UCI
  engine integration.
- A distributable, codesigned/notarized `.app` with bundled Python backend is a
  later packaging slice; generated apps and checkpoints should not be committed.

The engine/protocol stack does **not** yet own:
- Full UCI features such as pondering, rich options, MultiPV, advanced time
  management, detailed info streaming, tablebases, or opening books.
- Strict FIDE claim-vs-automatic draw semantics.

Those are deferred for later roadmap work.

## Tooling

Use `uv` for reproducible setup and commands:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy
uv run tinychess --help
printf '{"id":1,"cmd":"hello"}\n{"id":2,"cmd":"quit"}\n' | uv run tinychess gui-server
```

A lightweight perft benchmark is available:

```bash
uv run python scripts/perft.py 3
uv run python scripts/random_game.py --seed 7 --max-plies 40
uv run python scripts/mcts_benchmark.py --simulations 25 --seed 7
uv run python scripts/mlx_inference_benchmark.py --iterations 25 --warmup 5
uv run python scripts/benchmark.py --smoke
uv run python scripts/self_play.py --games 1 --max-plies 8 --simulations 1 --output data/selfplay/smoke
uv run python scripts/train.py --dataset data/selfplay/smoke --output data/checkpoints/train-smoke --epochs 1 --batch-size 2
uv run python scripts/evaluate.py --checkpoint data/checkpoints/train-smoke/checkpoint-final --games 1 --max-plies 8 --neural-simulations 1
(cd swift && swift test)
(cd swift && swift build -c release)
(cd swift && swift run TinyChessMacApp)  # launches the local-first GUI app
```
