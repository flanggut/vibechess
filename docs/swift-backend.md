# Swift Backend Plan

Swift acceleration is planned but only the Swift Package Manager bootstrap is implemented.

## Timing

Swift implementation work beyond the package skeleton should not begin until:

- The Python reference engine is correct enough to produce trusted fixtures.
- External perft references have validated the Python engine.
- Benchmarks show a concrete bottleneck that Swift is likely to improve.

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

WP18 is implemented as a minimal Swift Package Manager workspace under `swift/`:

- `Package.swift` defines the `TinyChessCore` library and `TinyChessCoreTests` test target.
- `Sources/TinyChessCore/` exposes bootstrap metadata only; chess logic remains in Python.
- `Tests/TinyChessCoreTests/` verifies the skeleton package compiles and exports expected metadata.
- `swift/README.md` documents the Swift build and test commands.

Run Swift checks from the package directory:

```bash
cd swift
swift test
swift build -c release
```

Remaining planned work packages:

- WP19: Swift engine acceleration prototype.
- WP20: Swift MCTS, batched inference, or Core ML evaluation spike.
