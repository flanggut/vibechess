# vibechess

A Python-first chess engine and neural-MCTS playground for Apple Silicon macOS.
The Python implementation is the correctness reference. Swift code in `swift/` is
limited to the native macOS frontend and optional acceleration scaffolding.

This README is the project documentation. Prefer source code, tests, and
`--help` output for low-level API and command details.

## Project shape

```text
src/vibechess/
├── engine/      # chess state, rules, FEN, bounded PGN/SAN
├── ai/          # players, random/classical MCTS/neural MCTS, evaluation
├── nn/          # MLX encoding, model, checkpoints, datasets, training
├── protocols/   # bounded UCI and GUI JSON-lines backends
└── ui/          # terminal rendering and play helpers

scripts/         # benchmarks, self-play, PGN ingestion, training, evaluation
tests/           # Python correctness and integration tests
swift/           # SwiftPM macOS app plus acceleration placeholder
data/            # local generated datasets and checkpoints; not source docs
```

Core conventions:

- Board squares are `0..63` with `a1 == 0`.
- Engine internals use typed `Board`, `Game`, and `Move` objects; UCI/SAN strings
  belong at protocol and file-format boundaries.
- PGN and UCI support are intentionally bounded. Keep protocol broadening
  explicit and test-backed.
- Neural policy targets use the versioned `8 * 8 * 73 = 4672` action space.
- Generated datasets, checkpoints, corpora, benchmark reports, app bundles, and
  smoke outputs should stay out of git unless deliberately added as fixtures.

## Requirements

- Apple Silicon macOS
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Swift 6.3+ for the optional Swift workspace

## Setup and validation

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy
uv run vibechess --help
```

Useful smoke checks:

```bash
uv run python scripts/perft.py 3
uv run python scripts/benchmark.py --smoke
uv run python scripts/self_play.py --games 1 --max-plies 8 --simulations 1 --output data/selfplay/smoke
uv run python scripts/train.py --dataset data/selfplay/smoke --output data/checkpoints/train-smoke --epochs 1 --batch-size 2
uv run python scripts/evaluate.py --checkpoint data/checkpoints/train-smoke/checkpoint-final --games 2 --max-plies 40 --neural-simulations 1 --mcts-simulations 1
(cd swift && swift test)
```

## Running

Use CLI help for current options:

```bash
uv run vibechess --help
uv run vibechess play --help
uv run vibechess uci --help
uv run vibechess gui-server --help
```

Common entry points:

```bash
uv run vibechess play
uv run vibechess uci
uv run vibechess gui-server
```

The native macOS app is a SwiftPM executable that talks to the Python GUI server:

```bash
cd swift
swift run VibeChessMacApp
```

Run it from a checkout where `uv run vibechess gui-server` works. The Swift app
renders backend state and sends moves; Python remains responsible for legality,
outcomes, and AI selection.

## Data and training workflow

Local generated artifacts normally live under `data/selfplay/` and
`data/checkpoints/`.

Typical loop:

1. Generate self-play or import PGN shards with scripts under `scripts/`.
2. Train with `scripts/train.py`.
3. Evaluate with `scripts/evaluate.py`.
4. Repeat only after checks and benchmark evidence justify changes.

Use `scripts/train.py --warmup N` when resuming an expanded checkpoint that keeps
existing weights but adds fresh layers; the first `N` optimizer steps linearly
ramp from a near-zero learning rate to `--learning-rate`.

For detailed flags and formats, prefer the script `--help` output and the dataset
loader/writer tests over duplicated prose. `scripts/self_play.py` supports
interactive TUI progress on stderr via `--progress auto|always|never`: `always`
forces it, `never` disables it, and `auto` uses the TUI only when stderr is
interactive. Stdout stays a stable final summary for automation.


Neural self-play and checkpoint evaluation can opt into visit-budget-aware tree
reuse with `--reuse-simulation-budget`; add `--min-reuse-simulations N` only
when reused roots must receive a fresh visit floor. Both scripts support
cross-game neural batching with `--batch-size`/`--active-games`; evaluation also
exposes within-search leaf batching via `--neural-collection-batch-size`.
Evaluation derives deterministic per-game player seeds from `--seed`, so larger
runs do not restart every game from the same RNG stream.

## Development notes

- Keep the Python engine as the source of truth unless a Swift acceleration task
  has parity tests and benchmark evidence.
- Update this README and tests when changing behavior, command boundaries,
  dataset/checkpoint compatibility, or Swift/Python responsibilities.
- Keep documentation high-level; avoid restating signatures, schemas, or file
  layouts that are already clear from code and tests.
