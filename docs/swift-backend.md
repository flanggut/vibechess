# Swift Backend Plan

Swift acceleration is planned but not implemented.

## Timing

Swift work should not begin until:

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

Not implemented. Planned work packages:

- WP18: Swift package bootstrap.
- WP19: Swift engine acceleration prototype.
- WP20: Swift MCTS, batched inference, or Core ML evaluation spike.
