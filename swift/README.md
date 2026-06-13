# VibeChess Swift Workspace

This Swift package contains the native macOS frontend and the placeholder core
module for future acceleration work. The Python engine remains the correctness
reference for chess rules, game outcomes, and AI selection.

## Targets

- `VibeChessCore`: library target reserved for future board, move-generation,
  and search acceleration experiments. It does not currently implement chess
  rules.
- `VibeChessMacApp`: SwiftUI macOS app for local human-vs-AI play. It renders
  backend state, sends UCI-style move strings, and talks to Python through the
  JSON-lines GUI protocol.
- `VibeChessCoreTests` and `VibeChessMacAppTests`: smoke/unit tests for the Swift
  package, protocol DTOs, backend client, app state, and presentation helpers.

## Local development commands

Run from this directory:

```bash
swift test
swift build
swift build -c release
swift run VibeChessMacApp
```

When launched with `swift run VibeChessMacApp`, the app opts into normal macOS
app activation so it appears in the Dock and Cmd-Tab switcher even though it is
still a SwiftPM executable rather than a bundled `.app`.

For the app to connect with its default development command, run it from a
checkout where `uv run vibechess gui-server` works. From the repository root,
initialize Python dependencies first when needed:

```bash
uv sync --dev
printf '{"id":1,"cmd":"hello"}\n{"id":2,"cmd":"quit"}\n' | uv run vibechess gui-server
```

## GUI/backend boundary

`VibeChessMacApp` launches the Python backend with:

```bash
uv run vibechess gui-server
```

The backend reads one UTF-8 JSON request per line from stdin and writes one JSON
response per line to stdout. Stderr is reserved for diagnostics. The protocol
version is `vibechess-gui-v1`; commands include `hello`, `newGame`, `state`,
`makeMove`, `aiMove`, `undo`, `setAiConfig`, and `quit`.

The state response is the app's source of truth. It includes FEN, occupied
squares, side to move, legal moves, legal destinations grouped by source square,
move history, last move, counters, and outcome. Swift does not calculate legal
moves, validate checks, apply chess rules, or run AI searches; those remain in
Python.

## Current GUI MVP

The app currently provides:

- Unicode-piece board rendering.
- Click source/destination move input.
- Legal destination, selected square, and last-move highlighting.
- Human White/Black selection and board flipping.
- Start/reset and undo-last-full-move controls.
- Random, classical MCTS, and optional checkpoint-backed neural AI settings.
- UCI move list, status text, thinking indicator, and error banner.

Current limitations:

- No drag-and-drop move input.
- No native promotion chooser; the backend auto-promotes to queen for MVP
  four-character promotion input.
- Move history is UCI notation, not SAN/PGN.
- No clocks, PGN import/export, save/load, opening book, tablebases, or external
  UCI engine integration.
- AI search is synchronous in the Python backend; cancellable/progress-streaming
  search is deferred.
- Neural play requires a local checkpoint path and may fail gracefully when MLX or
  checkpoint files are unavailable.

## Packaging status

The Swift app is local-first for now. A distributable `.app` is a later packaging
slice and still needs a decision on how to bundle or locate the Python backend,
plus codesigning and notarization work. Do not commit generated app bundles,
embedded Python environments, checkpoints, or local smoke outputs.

## Acceleration boundary

`VibeChessCore` remains separate from the GUI. Future Swift acceleration should be
benchmark-driven and validated against Python-generated fixtures and external
perft references before replacing any Python correctness path.
