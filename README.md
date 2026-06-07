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
- WP15: MLX training loop with policy/value losses, metrics logging, and checkpoint output.
- WP16: Evaluation harness for player/checkpoint matches, random/classical MCTS baselines, and early smoke promotion criteria.
- WP17: Full benchmark suite with Swift acceleration recommendation heuristic.
- WP18: Swift Package Manager bootstrap with `TinyChessCore` and Swift tests.

Native macOS GUI MVP additions:

- `tinychess gui-server`: a JSON-lines backend for the SwiftUI app.
- `TinyChessMacApp`: a local-first SwiftUI macOS app for human-vs-AI play
  against random, classical MCTS, or optional checkpoint-backed neural MCTS
  players.

Recent data/training additions:

- Classical MCTS can generate self-play policy labels as an alternative to neural MCTS.
- External PGN collections can be converted into sharded policy/value datasets for supervised pretraining.
- Training can auto-detect PGN shard manifests and train shard-by-shard to reduce memory pressure.

Next planned work package: WP19, Swift Engine Acceleration Prototype.

## Requirements

- Apple Silicon macOS
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Swift 6.3+ toolchain for the Swift workspace (`swift/` targets macOS 14+)

## Architecture and Design Decisions

Tinychess is Python-first. The Python engine is the correctness reference; Swift
is an optional backend workspace reserved for benchmark-proven acceleration work.
The project targets Apple Silicon macOS and uses MLX for neural-network training
and inference.

Current package boundaries:

```text
src/tinychess/
├── engine/      # board, moves, game state, FEN, bounded PGN, PGN ingestion stream helpers
├── ai/          # Player protocol, random player, classical MCTS, neural PUCT MCTS, evaluation
├── nn/          # MLX encoding/model/checkpoints, self-play, PGN datasets, training
├── protocols/   # bounded UCI loop and GUI JSON-lines backend
└── ui/          # terminal rendering and play loop helpers
```

Key technical defaults:

- Board squares use `0..63` indexing with typed engine primitives and immutable snapshots.
- Core engine APIs operate on `Move`, `Board`, and `Game` objects; UCI strings and SAN stay at protocol/PGN boundaries.
- FEN support covers complete standard position state.
- PGN support in the core parser is intentionally bounded and strict; ingestion adds a separate sanitizer for common public-dataset annotations.
- UCI support is intentionally bounded: handshake/readiness, `ucinewgame`, `position`, `go`, `stop`, and `quit`.
- GUI support uses a separate JSON-lines protocol (`tinychess-gui-v1`) so the
  native app can request canonical board state, legal move highlights, move
  application, AI moves, undo, and reset without expanding UCI semantics.
- Draw handling is pragmatic for engine/self-play termination; strict claim-vs-automatic FIDE semantics are deferred.
- Neural policy uses a versioned AlphaZero-style `8 x 8 x 73 = 4672` action space with legal-move masking.
- Dataset shards use compressed NumPy tensors plus JSON/JSONL metadata and include schema/action-space/encoder provenance.
- Checkpoints use MLX weights plus sidecar metadata; optimizer state is not currently persisted.

Training data paths:

- Neural or classical MCTS self-play creates existing-format dataset directories under `data/selfplay/`.
- PGN ingestion creates a `manifest.json` plus compatible shard directories, allowing supervised policy/value pretraining without loading the full corpus at once.

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
printf '{"id":1,"cmd":"hello"}\n{"id":2,"cmd":"quit"}\n' | uv run tinychess gui-server
uv run python scripts/perft.py 3
uv run python scripts/random_game.py --seed 7 --max-plies 40
uv run python scripts/mcts_benchmark.py --simulations 25 --seed 7
uv run python scripts/mlx_inference_benchmark.py --iterations 25 --warmup 5
uv run python scripts/benchmark.py --smoke
uv run python scripts/self_play.py --games 1 --max-plies 8 --simulations 1 --output data/selfplay/smoke
uv run python scripts/self_play.py --games 16 --max-plies 8 --simulations 1 --workers 4 --output data/selfplay/parallel-smoke
uv run python scripts/pgn_ingest.py --input ~/data/chess/lichess_elite_2025-11.pgn --output data/selfplay/pgn-smoke --max-games 10 --shard-samples 128
uv run python scripts/train.py --dataset data/selfplay/smoke --output data/checkpoints/train-smoke --epochs 1 --batch-size 2
uv run python scripts/evaluate.py --checkpoint data/checkpoints/train-smoke/checkpoint-final --games 2 --max-plies 40 --neural-simulations 1 --mcts-simulations 1
(cd swift && swift test)
(cd swift && swift build -c release)
```

## Current CLI

```bash
uv run tinychess --help
uv run tinychess --version
uv run tinychess play
uv run tinychess play --white human --black random
uv run tinychess play --white random --black random --seed 7 --max-plies 40
uv run tinychess play --white mcts --black random --seed 7 --mcts-simulations 25 --max-plies 40
uv run tinychess play --white human --black ai --ai-checkpoint data/checkpoints/train-smoke/checkpoint-final --ai-simulations 25
uv run tinychess uci
uv run tinychess uci --seed 7
uv run tinychess gui-server
uv run tinychess gui-server --seed 7 --ai-kind mcts --ai-simulations 25
```

The `play` command renders the board in the terminal, shows side to move, castling
en-passant and move-counter status, and accepts human moves in UCI long algebraic
notation such as `e2e4` or `e7e8q`. Invalid or illegal moves are rejected with a
message and another prompt. Player kinds are `human`, `random`, the classical
`mcts` baseline, and checkpoint-backed neural `ai`. The `ai` player uses neural
MCTS and requires `--ai-checkpoint`; use `--ai-simulations`, `--ai-node-budget`,
`--ai-time-limit-seconds`, `--ai-temperature`, and `--ai-puct-exploration` to tune
its search.

The `uci` command runs a bounded Universal Chess Interface loop. It supports
`uci`, `isready`, `ucinewgame`, `position startpos [moves ...]`,
`position fen ... [moves ...]`, `go`, `stop`, and `quit`. The current move source
is a local random legal selector; use `--seed` for deterministic selections.
Terminal or no-legal-move positions return `bestmove 0000`.

Deferred UCI features: pondering, rich `setoption`, MultiPV, advanced time
management, detailed `info` streaming, tablebases, and opening books.

The `gui-server` command runs the backend used by the native macOS app. It reads
one UTF-8 JSON object per line from stdin and writes one JSON response per line
to stdout; diagnostics belong on stderr. The protocol is intentionally separate
from bounded UCI and returns full GUI state such as FEN, occupied squares,
`legalMoves`, `legalDestinationsByFrom`, move history, last move, outcome,
errors, and AI search metadata. It supports `hello`, `newGame`, `state`,
`makeMove`, `aiMove`, `undo`, `setAiConfig`, and `quit`. The backend remains the
only place that validates chess legality; Swift sends UCI-like move strings and
renders returned state.

A minimal smoke check:

```bash
printf '{"id":1,"cmd":"hello"}\n{"id":2,"cmd":"state"}\n{"id":3,"cmd":"quit"}\n' \
  | uv run tinychess gui-server
```

## Native macOS GUI

The SwiftUI app lives in the Swift package as `TinyChessMacApp`. For local
development, run it from the repository so its default backend command can find
`uv run tinychess gui-server`:

```bash
cd swift
swift build
swift run TinyChessMacApp
```

The MVP offers a Unicode-piece board, click source/destination moves, legal move
and last-move highlights, human color selection, board flipping, start/reset,
undo-last-full-move, UCI move history, status/error display, and Random/MCTS/
optional-neural AI configuration. Neural play requires a local checkpoint path;
missing or unloadable checkpoints are surfaced as backend configuration errors.

Current GUI limitations: no drag-and-drop, no native promotion chooser
(auto-queen is used for four-character promotion moves), no SAN/PGN display or
save/load, no clocks, no cancellable/progress-streaming search, and no external
UCI-engine integration. The Swift target does not implement chess rules or move
generation; `TinyChessCore` remains acceleration scaffolding only.

Distributable `.app` packaging is a later slice. The current app is local-first
and assumes a developer environment with `uv` and the tinychess checkout
available. A bundled backend, codesigning, notarization, and backend path
selection are intentionally deferred.

## Benchmarks

Individual lightweight benchmark scripts are available for recursive perft,
random games, classical MCTS, and MLX inference. The combined report script runs
legal move generation depth-1 throughput, complete-game simulation, classical
MCTS, single-position MLX inference, and optional batched MLX inference as one
suite. Recursive perft remains covered by `scripts/perft.py`. The report also
adds a conservative suite-time Swift-acceleration heuristic:

```bash
uv run python scripts/benchmark.py --smoke
uv run python scripts/benchmark.py --output benchmark-report.md
uv run python scripts/benchmark.py --format json --output benchmark-report.json
```

Use `--smoke` for fast plumbing validation. Use the default suite for local
performance snapshots; repeat full runs before using the recommendation to plan
Swift acceleration work. The recommendation compares elapsed time within the
chosen benchmark suite; it is not a full application profile.

## Engine Example

```python
from tinychess.ai import MCTSConfig, MCTSPlayer, MatchConfig, RandomPlayer, random_player_spec, run_match, play_game
from tinychess.engine import Board, Game, legal_moves, parse_fen, parse_pgn, perft, random_move_selector, simulate_game
from tinychess.nn import ACTION_SPACE_SIZE, PolicyValueInference, PolicyValueNet, encode_game, legal_move_mask
from tinychess.nn.self_play import SelfPlayConfig, generate_self_play_dataset
from tinychess.nn.self_play_dataset import save_self_play_dataset
from tinychess.nn.train import TrainingConfig, train_model

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

# Train one smoke-friendly epoch and write metrics plus an MLX checkpoint.
training = train_model(
    dataset,
    "data/checkpoints/train-smoke",
    model=PolicyValueNet(),
    config=TrainingConfig(epochs=1, batch_size=2),
)
print(training.steps, training.checkpoint_dir)

# Compare players/checkpoints with the WP16 smoke evaluation harness.
match = run_match(
    random_player_spec(seed=1, name="candidate"),
    random_player_spec(seed=2),
    MatchConfig(games=2, max_plies=4),
)
print(match.player_a_score_rate)
```

## Swift Workspace

The Swift workspace under `swift/` now contains two separate concerns:

- `TinyChessCore`: future benchmark-driven acceleration scaffolding only.
- `TinyChessMacApp`: a SwiftUI macOS frontend that talks to the Python backend
  through the JSON-lines GUI protocol.

```bash
cd swift
swift test
swift build -c release
swift run TinyChessMacApp
```

Swift does not currently implement chess rules. The Python engine remains the
correctness reference for move legality, game outcomes, and AI selection.

## Documentation

- `docs/architecture.md`: current package and component boundaries.
- `docs/engine.md`: board representation, moves, legal move generation, and perft.
- `docs/ai.md`: planned AI/neural-MCTS direction.
- `docs/pgn-ingestion.md`: converting external PGN collections into sharded training datasets.
- `docs/swift-backend.md`: planned Swift acceleration strategy.
- `data/README.md`: future dataset/checkpoint policy.
