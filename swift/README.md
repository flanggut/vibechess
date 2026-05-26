# TinyChess Swift Backend

This Swift package is the bootstrap for future tinychess acceleration work. The
Python engine remains the correctness reference until Swift implementations are
validated against Python-generated fixtures and external perft references.

## Targets

- `TinyChessCore`: library target for future board, move-generation, and search
  acceleration code.
- `TinyChessCoreTests`: XCTest smoke tests for the package skeleton.

## Commands

Run from this directory:

```bash
swift test
swift build -c release
```

## Current Scope

WP18 only establishes the Swift Package Manager layout and a compile-tested core
module. It does not implement chess rules, runtime integration with Python, or a
Swift CLI. Those remain gated behind benchmark review and fixture parity work in
later work packages.
