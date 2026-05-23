# tinychess Project Plan

## Goal

Build a small chess engine and AI player that can:

- Simulate complete legal chess games.
- Expose AI players that interact with the engine through stable APIs.
- Provide a neural-network AI using AlphaZero-style policy/value inference plus PUCT Monte Carlo Tree Search.
- Print board states in a simple terminal text UI.
- Target macOS/Apple Silicon with Python first, then add Swift acceleration only after correctness fixtures and benchmarks identify bottlenecks.

## Confirmed Decisions

| Area | Decision |
| --- | --- |
| Initial implementation language | Python core first, Swift acceleration later |
| Performance philosophy | Correct Python reference first; measure before optimizing; optimize proven bottlenecks rather than prematurely choosing Swift/bitboards everywhere |
| ML framework | MLX for Apple Silicon-first neural-network training/inference |
| Platform support | Apple Silicon macOS only for the whole project |
| Chess rules/interoperability | Full standard chess rules plus FEN, bounded PGN basics first, bounded UCI basics first |
| Draw-rule semantics | Pragmatic complete-game semantics first; stricter FIDE claim/automatic distinctions later |
| Tooling | `uv` + `pyproject.toml` + `pytest` + `ruff` + `mypy`; SwiftPM for Swift modules later |
| AI design | AlphaZero-style policy/value network with PUCT MCTS |
| Technical defaults | Simple Python defaults now, benchmark later |
| Roadmap scope | Ambitious full roadmap, but staged to prevent PGN/UCI/ML/Swift from blocking early engine progress |

## Scope, MVP Gates, and Non-Goals

The roadmap is intentionally ambitious, but work should be gated so each phase has independent value.

### MVP 1: Correct Python Engine Foundation

Deliver:

- Python package scaffold.
- Board, piece, square, move, game, and outcome primitives.
- Legal move generation for full standard chess movement rules.
- Move application and efficient copy/make-unmake strategy designed early enough for MCTS.
- FEN parser/serializer.
- Text board rendering.
- Random player.
- Perft-style validation with known positions.
- Lightweight benchmarks for perft speed and random complete-game speed.

Success means the engine can simulate complete legal games and has enough validation to become the reference implementation.

### MVP 2: Playable CLI and Classical Search

Deliver:

- Terminal play loop.
- Human-vs-human, human-vs-random, random-vs-random, and MCTS-vs-random modes.
- Classical MCTS baseline.
- Bounded PGN basics.
- Bounded UCI basics.
- MCTS simulations/sec benchmark.

Success means the project is usable from the terminal and can expose legal best moves through a basic UCI loop.

### MVP 3: Neural MCTS Functional Prototype

Deliver:

- MLX position encoder.
- Fixed policy action-space mapping.
- Legal move masking.
- Small policy/value model.
- Neural PUCT MCTS smoke path.
- Self-play dataset generation.
- Training/checkpoint loop.
- Evaluation harness against baselines.
- MLX inference benchmark.

Success means the neural-MCTS pipeline functions end-to-end. It does **not** mean the engine is strong or competitive.

### Later Optimization and Integration

Deliver only after Python correctness fixtures and benchmarks exist:

- Full benchmark suite.
- Swift package and fixture-based parity tests.
- Swift acceleration for measured CPU bottlenecks.
- Possible Core ML or Swift-native inference research spike.

### Non-Goals for Early Milestones

- Competitive chess strength.
- Large-scale self-play infrastructure.
- Distributed training.
- Full PGN feature support in the first PGN milestone.
- Full UCI engine option/time-management support in the first UCI milestone.
- Core ML export as an assumed path.
- Swift runtime integration before benchmark evidence.
- Portability beyond Apple Silicon macOS.

## Decision Log and Technical Defaults

### Board and Move Representation

Initial default:

- Use 0..63 square indexing.
- Use a compact array/mailbox-style Python board representation first.
- Keep piece/color data integer-based and allocation-light.
- Prefer correctness and clear tests over bitboard complexity during MVP 1.
- Design move application so MCTS can avoid expensive naive full-board cloning where practical.
- Evaluate make/unmake versus copy-on-apply with benchmarks before MCTS scaling.

Deferred:

- Bitboards are a later optimization candidate if perft/game/MCTS benchmarks show move generation or board transitions dominate runtime.

### Move Notation and APIs

Initial default:

- Use UCI long algebraic notation internally and at CLI/protocol boundaries where practical, e.g. `e2e4`, `e7e8q`.
- Keep SAN generation/parsing at the PGN boundary.
- The core engine should operate on typed `Move` objects, not strings.

### FEN Scope

Initial default:

- Implement full FEN parse/serialize for position, side to move, castling rights, en passant target, halfmove clock, and fullmove number.
- Support `startpos` as a CLI/UCI convenience alias.

### PGN Scope

Initial bounded basics:

- Mainline game parser/writer.
- Standard tag pairs for common metadata.
- SAN move generation/parsing for normal moves, captures, checks, mates, castling, promotion, and disambiguation.
- Result handling.

Deferred PGN features:

- Comments.
- Numeric annotation glyphs.
- Recursive annotation variations.
- Clock annotations.
- Engine evaluations.
- Full tolerant parsing of malformed PGNs.

### UCI Scope

Initial bounded basics:

- `uci`.
- `isready`.
- `ucinewgame`.
- `position startpos [moves ...]`.
- `position fen ... [moves ...]`.
- `go` with a simple depth, node, movetime, or default budget.
- `stop` best-effort handling.
- `quit`.
- `bestmove` output with legal moves only.

Deferred UCI features:

- Pondering.
- Rich `setoption` support.
- MultiPV.
- Advanced time management.
- Detailed `info` streaming.
- Tablebases.
- Opening books.

### Draw and Game-End Semantics

Initial pragmatic semantics:

- Always detect checkmate and stalemate.
- Detect insufficient material enough to terminate common dead games.
- Track repetition and halfmove clock to terminate self-play games pragmatically.
- Treat repetition and fifty-move-style draws as automatic engine outcomes initially, even though official rules distinguish claim-based and automatic cases.
- Document exact behavior in `docs/engine.md` once implemented.

Deferred strict semantics:

- Separate threefold repetition claim from fivefold automatic draw.
- Separate fifty-move claim from seventy-five-move automatic draw.
- More exhaustive dead-position detection.
- User/player claim mechanics.

### Neural Policy Action Space

Initial default:

- Use an AlphaZero-style fixed action space: 8 x 8 x 73 = 4672 possible move actions.
- Encode queen-like directions, knight moves, and underpromotions explicitly.
- Mask illegal moves before policy normalization, search expansion, and move selection.
- Keep action mapping versioned and covered by round-trip tests.

Deferred:

- Alternative compact legal-move-only policy heads.
- Larger or architecture-specific action spaces.

### Dataset Format

Initial default:

- Store self-play samples in a versioned local format.
- Prefer compressed NumPy-compatible files such as `.npz` for tensor batches plus JSON/JSONL metadata for game-level information.
- Include schema version, engine version/git commit when available, action-space version, model checkpoint id, and generation settings.

Deferred:

- Large-scale sharded datasets.
- Cloud/object storage.
- Database-backed replay buffers.

### Checkpoint Format

Initial default:

- Use MLX-native save/load mechanisms for model parameters.
- Store a sidecar metadata file with schema version, model config, action-space version, training step, optimizer state availability, and evaluation notes.

Deferred:

- Core ML export.
- ONNX export.
- Swift-native inference format.

### Python/Swift Boundary

Initial default:

- Python remains the correctness reference.
- Swift does not become part of the runtime path until Python fixtures and benchmarks exist.
- Swift parity is validated through fixture files generated from known-good Python engine states plus external perft references.

Deferred:

- Python extension module integration.
- Process-boundary integration.
- Swift-native CLI replacement.
- Core ML inference from Swift.

## Architecture Overview

The project should be organized as a Python package first, with a future Swift package added as an optimized backend. The Python package remains the reference implementation and test oracle. Swift modules are introduced only after correctness is established and benchmarks justify them.

```text
tinychess/
├── pyproject.toml
├── uv.lock
├── README.md
├── PLAN.md
├── LICENSE
├── src/
│   └── tinychess/
│       ├── __init__.py
│       ├── engine/
│       │   ├── board.py
│       │   ├── move.py
│       │   ├── legal_moves.py
│       │   ├── game.py
│       │   ├── fen.py
│       │   ├── pgn.py
│       │   └── outcome.py
│       ├── ai/
│       │   ├── player.py
│       │   ├── mcts.py
│       │   ├── neural_mcts.py
│       │   └── search_config.py
│       ├── nn/
│       │   ├── model.py
│       │   ├── encode.py
│       │   ├── train.py
│       │   ├── self_play.py
│       │   └── checkpoint.py
│       ├── ui/
│       │   ├── terminal.py
│       │   └── render.py
│       ├── protocols/
│       │   └── uci.py
│       └── cli.py
├── tests/
│   ├── engine/
│   ├── ai/
│   ├── nn/
│   ├── protocols/
│   └── fixtures/
├── scripts/
│   ├── self_play.py
│   ├── train.py
│   └── benchmark.py
├── data/
│   ├── README.md
│   ├── selfplay/.gitkeep
│   └── checkpoints/.gitkeep
├── swift/
│   ├── Package.swift
│   ├── Sources/
│   │   ├── TinyChessCore/
│   │   └── TinyChessCLI/
│   └── Tests/
│       └── TinyChessCoreTests/
└── docs/
    ├── architecture.md
    ├── engine.md
    ├── ai.md
    └── swift-backend.md
```

The `swift/` tree is planned, but should not be created or populated substantially until Python fixtures and benchmark baselines exist.

## Component Plan

### 1. Python Chess Engine

The Python engine is the correctness-first reference implementation.

Responsibilities:

- Board representation.
- Piece and square representation.
- Move representation.
- Legal move generation.
- Move application and undo/copy support.
- Check, checkmate, stalemate, and pragmatic draw detection.
- Castling, en passant, and promotion.
- Halfmove clock and fullmove number.
- FEN import/export.
- Bounded PGN basics.
- Bounded UCI command loop.

Recommended implementation choices:

- Start with 0..63 square indexing and compact array/mailbox-style state.
- Keep APIs simple and typed.
- Add bitboards later only if profiling shows move generation is a bottleneck.
- Design for efficient state transition early because MCTS will stress it.
- Maintain exhaustive tests before optimization.

Important public concepts:

- `Board`: current position and side to move.
- `Move`: source square, target square, promotion, and flags.
- `Game`: sequence of positions/moves and outcome state.
- `Player`: protocol/interface for human, random, MCTS, and neural-MCTS players.

### 2. Terminal UI

The terminal UI should be intentionally simple.

Responsibilities:

- Render board as text.
- Show side to move, castling rights, move counters, game status, and last move.
- Support coordinates and optional Unicode pieces.
- Provide CLI paths for human-vs-human, human-vs-AI, AI-vs-AI, and self-play smoke tests.

Initial command examples:

```bash
uv run tinychess play
uv run tinychess play --white human --black random
uv run tinychess play --white neural-mcts --black random
uv run tinychess fen "startpos"
uv run tinychess uci
```

### 3. AI Player Interfaces

All AI players should interact with the engine through the same player API.

Initial player types:

- `RandomPlayer`: chooses a legal move randomly.
- `MCTSPlayer`: uses classical MCTS without a neural model, useful for testing.
- `NeuralMCTSPlayer`: uses policy/value network plus PUCT MCTS.

The AI layer should not mutate engine internals directly. It should use legal move APIs and position transition APIs.

### 4. AlphaZero-Style Neural MCTS

The selected AI design is an AlphaZero-style policy/value network guided by PUCT MCTS.

Main pieces:

- Position encoder converts board state to MLX tensors.
- Policy head predicts move probabilities over the fixed 4672-action space.
- Value head predicts expected outcome from the side to move.
- MCTS expands legal actions only.
- PUCT balances prior probability and search value.
- Self-play generates training games.
- Training optimizes policy loss, value loss, and regularization.

Training pipeline:

1. Generate self-play games with current model.
2. Store positions, legal masks, MCTS policy targets, final game outcomes, and metadata.
3. Train model using MLX on Apple Silicon.
4. Save checkpoints with versioned metadata.
5. Evaluate new checkpoint against previous checkpoints and simple baselines.
6. Promote checkpoint if it passes evaluation criteria.

Expectation:

- The initial neural-MCTS goal is functional correctness and experimentation, not competitive playing strength.

### 5. MLX Model Layer

The MLX layer should remain isolated from the engine.

Responsibilities:

- Define model architecture.
- Encode/decode board tensors and policy vectors.
- Run inference for MCTS.
- Train from self-play datasets.
- Save/load checkpoints.

Initial model should be small enough for quick iteration:

- Residual CNN-style policy/value network.
- Configurable number of residual blocks and channels.
- Start tiny, then scale only after end-to-end correctness is verified.

macOS/MLX caveats:

- Apple Silicon macOS is the supported environment.
- Intel Mac and non-macOS support are not project goals.
- Generic CI, if added later, may run only non-ML tests unless Apple Silicon runners are available.
- Core ML export from MLX is not assumed; it is a later research spike.

### 6. Swift Acceleration Plan

Swift is introduced after the Python engine, tests, fixtures, and benchmark baselines are stable.

Candidate Swift components, selected by benchmark evidence:

1. Legal move generation and board state transitions.
2. MCTS tree search core.
3. Batched inference integration if MLX/Python overhead dominates.
4. Terminal CLI wrapper only if useful.
5. Core ML inference path only if export/integration proves viable.

Swift package goals:

- Use Swift Package Manager.
- Keep APIs close to Python reference concepts.
- Run conformance tests against Python-generated fixtures and external perft references.
- Prefer value types and compact memory layouts.
- Consider bitboards for move generation only after correctness fixtures exist.

Python/Swift interoperability options to evaluate later:

- CLI/process boundary for loose integration.
- Python extension module if tight integration is required.
- Shared fixture-based validation before runtime integration.

No Swift acceleration should be started until engine correctness and benchmark baselines exist.

## macOS Performance Strategy

- Optimize for Apple Silicon macOS.
- Use MLX for Apple Silicon-optimized training and inference.
- Use Python for algorithm exploration and correctness.
- Measure perft speed and complete-game simulation speed early.
- Measure MCTS simulations/sec before assuming Swift is the right next optimization.
- Measure model inference latency and batching behavior before assuming Core ML or Swift inference is needed.
- Use Swift for components proven to be CPU bottlenecks.
- Keep the Python engine as reference even after Swift acceleration exists.

## Testing and Validation Strategy

### Engine Tests

- Initial board legal move count.
- Perft-style move generation tests.
- Known perft positions including start position, Kiwipete, and castling/en-passant/promotion stress positions.
- Castling legality.
- En passant legality.
- Promotion handling.
- Check, checkmate, stalemate.
- Pragmatic draw handling.
- FEN round-trips.
- PGN bounded-basics import/export.
- UCI position parsing.
- Apply/copy or make/unmake invariants.
- Property/fuzz-style tests where practical.

### External Validation

- Compare selected legal move counts and FEN outcomes against trusted perft references.
- Use external fixtures during development so Swift does not merely reproduce Python bugs.
- Store curated fixture positions in `tests/fixtures/`.

### AI Tests

- MCTS only selects legal moves.
- MCTS handles terminal positions.
- MCTS respects budgets.
- Neural MCTS masks illegal moves.
- Self-play completes games without illegal states.
- Deterministic behavior with fixed seeds where practical.

### ML Tests

- Encoder shape and value tests.
- Policy action mapping round-trips.
- Legal mask tests.
- Forward pass smoke test.
- Checkpoint save/load smoke test.
- Tiny training step computes loss successfully.

### Swift Tests

- Swift fixtures generated from Python reference and cross-checked with external perft data.
- Position round-trip tests.
- Legal move parity tests.
- Perft parity tests.
- Benchmark parity reports.

## Repository Setup Plan

### Python

Use `uv` and `pyproject.toml`.

Planned dependency groups:

- Runtime: `mlx`, `numpy`, optional CLI helpers.
- Dev: `pytest`, `pytest-cov`, `ruff`, `mypy`.
- Optional benchmark/docs tools as needed.

Recommended commands:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
uv run tinychess --help
```

### Swift

Use Swift Package Manager under `swift/` once Swift work is justified.

Recommended commands once Swift package exists:

```bash
cd swift
swift test
swift build -c release
```

## Work Packages

### WP01: Python Project Bootstrap

Deliverables:

- `pyproject.toml` with package metadata and tooling config.
- `src/tinychess` package skeleton.
- `tests` skeleton.
- CLI entry point.
- Basic CI-ready commands documented in README.

Can be done independently: yes.

### WP02: Core Board and Move Types

Deliverables:

- Board, square, piece, color, and move types.
- 0..63 square indexing convention documented and tested.
- Compact array/mailbox-style board state.
- UCI long algebraic move string conversion.
- Starting position setup.
- Text board rendering helper.
- Basic unit tests.

Depends on: WP01.

### WP03: Legal Move Generation

Deliverables:

- Pseudo-legal move generation.
- Legal move filtering for check safety.
- Castling, en passant, and promotion.
- Perft-style tests with known positions.
- Lightweight perft benchmark script or command.

Depends on: WP02.

### WP04: Game State and Outcomes

Deliverables:

- Move application.
- Efficient state transition strategy for MCTS: copy-on-apply baseline and/or make-unmake design.
- Game history.
- Checkmate/stalemate detection.
- Pragmatic draw tracking: halfmove clock, repetition, insufficient material.
- Complete game simulation loop.
- Lightweight random complete-game benchmark.

Depends on: WP03.

### WP05: FEN Support

Deliverables:

- Full FEN parser.
- Full FEN serializer.
- Round-trip tests.
- Fixture positions for engine tests.

Depends on: WP04.

### WP06: PGN Bounded Basics

Deliverables:

- Mainline PGN move recording.
- Basic PGN parser/writer.
- SAN generation/parsing for bounded scope.
- Result and common tag handling.
- Tests with short sample games.
- Explicit unsupported-feature behavior for comments, NAGs, variations, and clocks.

Depends on: WP05.

### WP07: Terminal UI and CLI Play Loop

Deliverables:

- Text board renderer.
- Human input parsing using UCI long algebraic moves initially.
- Human-vs-human and random-vs-random modes.
- CLI commands.

Depends on: WP04.

### WP08: Basic UCI Protocol

Deliverables:

- UCI command loop.
- Support for `uci`, `isready`, `ucinewgame`, `position`, `go`, `stop`, and `quit` basics.
- Legal `bestmove` output.
- Random or simple MCTS move output initially.
- Deferred feature list documented.

Depends on: WP05, WP10.

### WP09: Player Interface and Random Player

Deliverables:

- `Player` interface.
- `RandomPlayer`.
- AI-vs-AI simulation tests.

Depends on: WP04.

### WP10: Classical MCTS Baseline

Deliverables:

- MCTS tree structure.
- Random rollout or simple evaluation baseline.
- Configurable simulation count/time/node budget.
- Tests for legal move selection and terminal handling.
- MCTS simulations/sec benchmark.

Depends on: WP09.

### WP11: MLX Position Encoder and Policy Mapping

Deliverables:

- Board-to-tensor encoder.
- 4672-action AlphaZero-style policy mapping.
- Legal move mask.
- Action-space version metadata.
- Tests for shapes and move round-trips.

Depends on: WP05.

### WP12: MLX Policy/Value Network

Deliverables:

- Small configurable residual policy/value model.
- Inference wrapper.
- MLX checkpoint save/load.
- Checkpoint sidecar metadata.
- Forward-pass tests.
- MLX inference latency benchmark.

Depends on: WP11.

### WP13: Neural PUCT MCTS

Deliverables:

- PUCT search implementation.
- Neural policy priors.
- Value backup.
- Illegal move masking.
- Temperature-based move selection.
- Functional smoke test against random/classical baselines.

Depends on: WP10, WP12.

### WP14: Self-Play Data Generation

Deliverables:

- Self-play game runner.
- Versioned dataset format for positions, masks, MCTS policies, outcomes, and metadata.
- Script for generating small datasets.
- Smoke tests for complete self-play games.

Depends on: WP13.

### WP15: Training Loop

Deliverables:

- MLX training script.
- Policy/value losses.
- Checkpointing.
- Basic metrics logging.
- Tiny overfit/smoke test.

Depends on: WP14.

### WP16: Evaluation Harness

Deliverables:

- Match runner between players/checkpoints.
- Baseline comparisons against random and classical MCTS.
- Promotion criteria for checkpoints.
- Clear note that early promotion criteria validate progress, not competitive strength.

Depends on: WP15.

### WP17: Full Benchmark Suite

Deliverables:

- Move generation benchmark.
- Complete game simulation benchmark.
- MCTS simulations/sec benchmark.
- MLX inference benchmark.
- Optional batched inference benchmark.
- Benchmark report script.
- Recommendation for whether Swift acceleration is justified and where.

Depends on: WP03, WP04, WP10, WP12.

### WP18: Swift Package Bootstrap

Deliverables:

- `swift/Package.swift`.
- `TinyChessCore` module skeleton.
- Swift test target.
- README/docs for Swift build commands.

Depends on: WP17 and decision that Swift work is justified.

### WP19: Swift Engine Acceleration Prototype

Deliverables:

- Swift board and move representation.
- Legal move generation prototype.
- Fixture-driven parity tests against Python and external perft references.
- Performance comparison against Python.

Depends on: WP18 and stable Python fixtures.

### WP20: Swift MCTS, Batched Inference, or Core ML Evaluation Spike

Deliverables:

- Decide based on benchmark data whether Swift MCTS, batched inference improvements, or Core ML/Swift inference integration is the next best optimization.
- Prototype selected path.
- Document performance results and integration risks.

Depends on: WP19, WP17.

## Milestones

### Milestone 1: Correct Python Chess Engine

Includes WP01-WP06.

Success criteria:

- Full legal games can be simulated.
- FEN round-trips work.
- Bounded PGN basics work.
- Perft tests pass for selected known positions.
- Lightweight perft and random-game benchmarks exist.

### Milestone 2: Playable Terminal Version

Includes WP07-WP10.

Success criteria:

- Terminal board display works.
- Human/random/MCTS players can complete games.
- Basic UCI command loop can produce legal moves.
- MCTS simulations/sec is measurable.

### Milestone 3: Neural MCTS Prototype

Includes WP11-WP13.

Success criteria:

- MLX model runs inference on Apple Silicon.
- 4672-action policy mapping works and is tested.
- Neural MCTS selects legal moves.
- Illegal move masking is tested.
- Functional neural-MCTS smoke games complete.

### Milestone 4: Self-Play and Training Loop

Includes WP14-WP16.

Success criteria:

- Self-play produces versioned datasets.
- Training loop consumes datasets and writes checkpoints.
- Evaluation harness compares checkpoints and baselines.
- Project has evidence of a working learning pipeline, not necessarily a strong chess engine.

### Milestone 5: Performance Baseline and Swift Acceleration

Includes WP17-WP20.

Success criteria:

- Python performance bottlenecks are measured.
- Swift package exists only if justified by benchmark data.
- At least one Swift or inference optimization prototype is validated against Python/external fixtures.

## Documentation Plan

- `README.md`: quickstart, commands, project status, Apple Silicon requirement.
- `docs/architecture.md`: package and data-flow overview.
- `docs/engine.md`: board representation, move generation, rules, and pragmatic draw semantics.
- `docs/ai.md`: MCTS, neural model, action space, training pipeline.
- `docs/swift-backend.md`: Swift acceleration design and parity strategy.
- `data/README.md`: dataset/checkpoint storage policy and schema versions.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Scope is much larger than “small engine” suggests | Ambitious roadmap with MVP gates and explicit deferred features |
| Chess move generation bugs | Known perft tests, FEN fixtures, exhaustive special-rule tests, external reference validation |
| PGN/UCI become time sinks | Bounded basics first, full support deferred and documented |
| Draw semantics ambiguity | Pragmatic complete-game semantics first, strict claim/automatic rules deferred |
| Python performance bottlenecks | Early lightweight benchmarks, full benchmark suite before Swift work |
| MCTS board cloning overhead | Design copy/make-unmake strategy early and benchmark it |
| Neural training complexity | Start with tiny models and smoke tests before scaling self-play |
| Neural player is weak despite working pipeline | Set expectation: functional prototype first, strength later |
| Illegal policy actions | Versioned fixed action mapping, legal masks, round-trip tests |
| MLX portability/integration risk | Apple Silicon-only target, isolated MLX layer, Core ML export treated as research |
| Python/Swift divergence | Python and external fixtures, parity tests, no Swift runtime path before validation |
| Swift may not address true bottleneck | Benchmark-driven selection among Swift engine, Swift MCTS, batching, or inference work |

## Immediate Next Steps

1. Create the Python project scaffold with `uv` and `pyproject.toml`.
2. Add the `src/tinychess` package skeleton and CLI stub.
3. Implement board, piece, square, and move primitives using the documented 0..63 convention.
4. Add the first engine tests and fixture structure.
5. Build legal move generation with perft-style validation and a lightweight perft benchmark before starting AI work.
