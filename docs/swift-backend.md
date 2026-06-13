# Swift Backend and macOS App

The Swift workspace now contains two intentionally separate pieces:

- `VibeChessMacApp`: a local-first SwiftUI macOS app for human-vs-AI play.
- `VibeChessCore`: placeholder scaffolding for future Swift acceleration work.

The Python engine remains the correctness reference. The Swift app talks to
`uv run vibechess gui-server` over the GUI JSON-lines protocol and does not
implement chess rules or AI itself.

## Timing

Swift acceleration implementation beyond the GUI frontend and package scaffolding
should not begin until:

- The Python reference engine is correct enough to produce trusted fixtures.
- External perft references have validated the Python engine.
- Benchmarks show a concrete bottleneck that Swift is likely to improve.

## Native macOS GUI

The GUI is available as the `VibeChessMacApp` SwiftPM executable target. Run it
from the repository checkout so the default backend command can locate `uv` and
the Python package:

```bash
uv sync --dev
cd swift
swift run VibeChessMacApp
```

The app renders backend-provided board state, supports click source/destination
moves, legal-destination and last-move highlighting, human color selection,
board flipping, start/reset, undo-last-full-move, UCI move history, status/error
display, and Random/MCTS/optional-neural AI configuration. Neural play requires
a local checkpoint path.

Packaging note: this is not yet a distributable `.app` bundle. Bundling or
locating the Python backend, codesigning, and notarization are deferred.

## Candidate Acceleration Areas

Benchmark-driven candidates include:

1. Legal move generation and board transitions.
2. MCTS tree search.
3. Batched inference integration if Python/MLX overhead dominates.
4. Swift CLI or Core ML integration only if justified later.

## Validation Strategy

Swift implementations must be validated against:

- Python-generated fixtures.
- Known external perft counts.
- Position round-trips.
- Legal move parity tests.
- Performance benchmarks against the Python baseline.

## Current Status

The Swift workspace under `swift/` includes:

- `Package.swift` products for the `VibeChessCore` library and
  `VibeChessMacApp` executable app target.
- `Sources/VibeChessMacApp/` with SwiftUI app state, backend client/protocol
  DTOs, board rendering, controls, and move-list views.
- `Tests/VibeChessMacAppTests/` covering the app-state, backend-client,
  protocol-model, and presentation helper seams.
- `Sources/VibeChessCore/` and `Tests/VibeChessCoreTests/` as acceleration
  bootstrap scaffolding only.
- `swift/README.md` documenting local app/backend workflow and packaging status.

Run Swift checks from the package directory:

```bash
cd swift
swift test
swift build -c release
swift run VibeChessMacApp
```

Remaining planned acceleration work packages:

- WP19: Swift engine acceleration prototype.
- WP20: Swift MCTS, batched inference, or Core ML evaluation spike.
