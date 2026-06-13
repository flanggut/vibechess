# AGENTS.md

## Project orientation

- Python-first chess engine/AI project for Apple Silicon macOS. The Python engine is the correctness reference; Swift under `swift/` is optional acceleration scaffolding only.
- Main package layout: `src/vibechess/engine` (board, moves, game, FEN, bounded PGN/SAN, legal moves), `ai` (players, MCTS, evaluation), `nn` (MLX encoding/model/checkpoints/self-play/PGN datasets/train), `protocols` (bounded UCI), `ui` (terminal), plus `scripts/`, `tests/`, `docs/`, and optional `swift/`.
- Core chess conventions: squares are `0..63` with `a1 == 0`, `h8 == 63`; use `Move` objects internally and UCI/SAN strings only at boundaries. `Board` stores placement/side/castling/en-passant; `Game` stores history, counters, repetition, and outcome state.

## Setup and validation

- Install/sync dependencies with `uv sync --dev`.
- Run Python checks with `uv run pytest`, `uv run ruff check .`, and `uv run mypy` before committing substantive code changes.
- Useful smoke commands: `uv run vibechess --help`, `uv run python scripts/perft.py 3`, `uv run python scripts/benchmark.py --smoke`, `uv run python scripts/self_play.py --games 1 --max-plies 8 --simulations 1 --output data/selfplay/smoke`.
- Swift checks, when touching Swift or acceleration boundaries: `(cd swift && swift test)` and `(cd swift && swift build -c release)`.

## Style and boundaries

- Python 3.11+, strict mypy, Ruff line length 100. Prefer typed dataclasses and immutable snapshots following existing engine patterns.
- Keep core PGN and UCI support intentionally bounded/strict. Do not silently broaden protocols or FIDE draw semantics without tests and docs.
- PGN ingestion may sanitize common public-dataset annotations separately from the strict core parser; keep that boundary explicit.
- Keep Swift changes gated by Python parity tests and benchmark evidence.

## Neural/data facts

- Policy action space is versioned AlphaZero-style `8 * 8 * 73 = 4672`; datasets/checkpoints carry schema, action-space, and encoder metadata.
- MLX is the neural backend; keep public encoder/model/checkpoint metadata compatible when changing tensors or policies.
- Avoid committing generated data, checkpoints, downloaded PGN corpora, benchmark profiles, or local smoke outputs under `data/` unless explicitly requested.

## Documentation expectations

- Update docs/tests alongside behavioral changes, especially for protocol boundaries, PGN ingestion, dataset formats, CLI/script usage, and Swift acceleration claims.
- Prefer narrow, evidence-backed optimizations: include a benchmark or focused test when changing hot paths.
- Any temporary artifacts from planning or other agent activities should be store in an `artifacts` subfolder.
