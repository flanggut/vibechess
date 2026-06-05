# Implementation Plan

## Goal
Build a small native SwiftUI macOS application that lets a human play tinychess against random, classical MCTS, or optional checkpoint-backed neural MCTS players through a Python JSON-lines backend without rewriting chess legality in Swift.

## Alignment and Current Repo Context
- Python is the correctness reference; Swift under `swift/` is currently a bootstrap package for future acceleration only.
- Existing Python entrypoint is `src/tinychess/cli.py` with `play` and bounded `uci` subcommands.
- Existing protocol code is `src/tinychess/protocols/uci.py`; GUI support should be a separate protocol module to avoid broadening bounded UCI semantics.
- Existing playable/AI integration exists in `src/tinychess/ui/terminal.py` and supports `RandomPlayer`, `MCTSPlayer`, and checkpoint-backed neural `NeuralMCTSPlayer` through `PlayConfig` patterns.
- Core state APIs to reuse: `Game.new()`, `Game.from_fen()`, `Game.to_fen()`, `Game.legal_moves`, `Game.play()`, `Game.moves`, `Game.positions`, `Game.outcome`, `Board.squares`, `Piece.symbol`, `Move.from_uci()`, and `Move.to_uci()`.
- Existing Swift package at `swift/Package.swift` only exports `TinyChessCore`; the GUI should add a macOS app target while keeping acceleration work separate.

## Target Layout

### Python backend additions
```text
src/tinychess/protocols/
├── __init__.py
├── uci.py                  # existing, unchanged except package docs if needed
└── gui.py                  # new JSON-lines GUI backend/session/protocol

tests/
└── test_gui_protocol.py    # new protocol/session/CLI tests

src/tinychess/
└── cli.py                  # add `gui-server` subcommand
```

### Swift app additions
Preferred local-first layout inside existing Swift workspace:
```text
swift/
├── Package.swift
├── README.md
├── Sources/
│   ├── TinyChessCore/              # existing acceleration bootstrap, do not add chess rules here
│   └── TinyChessMacApp/            # new executable SwiftUI app target
│       ├── TinyChessMacApp.swift
│       ├── AppState.swift
│       ├── BackendClient.swift
│       ├── BackendModels.swift
│       ├── BoardView.swift
│       ├── SquareView.swift
│       ├── ControlsView.swift
│       ├── MoveListView.swift
│       └── Assets.xcassets/        # optional later; Unicode glyphs for MVP
└── Tests/
    ├── TinyChessCoreTests/         # existing
    └── TinyChessMacAppTests/       # backend model/client parsing tests where feasible
```

If Swift Package Manager executable SwiftUI app packaging is awkward, create an Xcode app project under `swift/TinyChessMacApp/` but keep shared source names above. Do not place chess-rule logic in Swift; Swift may map squares, display pieces, and route UCI-like strings only.

## JSON-lines GUI Protocol

### Transport
- Backend command: `uv run tinychess gui-server` for development/local MVP.
- Swift launches the backend with `Process`, writes one UTF-8 JSON object per line to stdin, and reads one UTF-8 JSON object per line from stdout.
- Backend writes logs/tracebacks only to stderr; stdout is reserved for protocol JSON.
- Each request includes a monotonically increasing client-generated `id`. Each response echoes `id`.
- Search commands are synchronous per request in Python, but Swift must call them from a background `Task` and show `thinking` until the response arrives.
- Initial backend can process one command at a time. Swift serializes requests through an actor or queue; no concurrent writes.

### Envelope
Request:
```json
{ "id": 1, "cmd": "state" }
```
Success response:
```json
{ "id": 1, "ok": true, "state": { } }
```
Error response:
```json
{
  "id": 1,
  "ok": false,
  "error": {
    "code": "illegal_move",
    "message": "illegal move 'e2e5' for the current position"
  },
  "state": { }
}
```
Include `state` on recoverable errors when possible so the UI can resync. Fatal JSON/unknown command errors may omit `state` but should still return JSON.

### State model
Every state-bearing response should use one canonical schema:
```json
{
  "state": {
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "sideToMove": "white",
    "squares": [
      { "square": "a1", "index": 0, "piece": "R", "color": "white", "kind": "rook" }
    ],
    "legalMoves": ["e2e4", "g1f3"],
    "legalDestinationsByFrom": { "e2": ["e3", "e4"], "g1": ["f3", "h3"] },
    "moves": ["e2e4", "e7e5"],
    "lastMove": "e7e5",
    "halfmoveClock": 0,
    "fullmoveNumber": 2,
    "outcome": null
  }
}
```
Outcome shape:
```json
{ "reason": "checkmate", "winner": "white", "isDraw": false }
```
Draw examples use `winner: null`, `isDraw: true`, and reasons from `OutcomeReason`: `stalemate`, `fifty_move`, `repetition`, `insufficient_material`, `max_plies`.

Notes:
- `squares` should include only occupied squares to keep payloads simple.
- Square indices must follow current convention: `a1 == 0`, `h8 == 63`.
- Piece `piece` is the existing FEN symbol (`P`, `n`, etc.); `color` and `kind` are convenience fields for Swift display.
- Legal move highlighting can be driven by `legalDestinationsByFrom`; Swift does not compute legality.

### Commands
1. `hello`
   - Request: `{ "id": 1, "cmd": "hello" }`
   - Response includes backend/app metadata and capabilities:
     ```json
     {
       "id": 1,
       "ok": true,
       "version": "0.1.0",
       "protocol": "tinychess-gui-v1",
       "capabilities": {
         "players": ["random", "mcts", "neural"],
         "supportsUndo": true,
         "promotion": "auto_queen"
       },
       "state": { }
     }
     ```

2. `newGame`
   - Request:
     ```json
     {
       "id": 2,
       "cmd": "newGame",
       "humanColor": "white",
       "ai": { "kind": "mcts", "simulations": 25, "timeLimitSeconds": null, "nodeBudget": null },
       "seed": 7
     }
     ```
   - Resets `Game.new()`, stores human/AI config, clears reusable search trees, and returns `state`.
   - If `humanColor` is `black`, Swift should request `aiMove` after new game returns.

3. `state`
   - Request: `{ "id": 3, "cmd": "state" }`
   - Returns canonical state without changing game.

4. `makeMove`
   - Request: `{ "id": 4, "cmd": "makeMove", "move": "e2e4" }`
   - Backend parses `Move.from_uci()` and validates against `game.legal_moves`.
   - MVP promotion behavior: if the requested 4-character move is a legal pawn promotion only with a suffix, backend auto-selects queen promotion (`q`) when queen promotion is legal and records/returns `appliedMove: "e7e8q"`.
   - Response:
     ```json
     { "id": 4, "ok": true, "appliedMove": "e2e4", "state": { } }
     ```

5. `aiMove`
   - Request:
     ```json
     {
       "id": 5,
       "cmd": "aiMove",
       "ai": { "kind": "mcts", "simulations": 100, "timeLimitSeconds": 0.5, "nodeBudget": null }
     }
     ```
   - Backend selects a legal move with requested AI config or current stored AI config, applies it, and returns `appliedMove`, optional `search` metadata, and `state`.
   - For `random`: use `RandomPlayer(seed=...)`.
   - For `mcts`: use `MCTSPlayer(MCTSConfig(simulations, timeLimitSeconds, nodeBudget, max_rollout_plies, seed))`.
   - For `neural`: require `checkpointPath`; load via existing `load_checkpoint()` and `PolicyValueInference`; use `NeuralMCTSPlayer(NeuralMCTSConfig(...))`. Keep optional and return `configuration_error` if missing/unloadable.
   - Terminal/no-legal state returns `ok: false`, `error.code: "terminal_position"`, plus current `state`.
   - Search metadata examples:
     ```json
     { "kind": "mcts", "simulations": 25, "nodes": 25, "elapsedSeconds": 0.03 }
     ```

6. `undo`
   - Request: `{ "id": 6, "cmd": "undo", "plies": 2 }`
   - MVP should support undoing one full move for human-vs-AI (`plies: 2`) and at least one ply when only one move exists.
   - Rebuild `Game` from `Game.positions`/`moves` history rather than mutating board. Safer helper: replay the first `len(moves)-plies` UCI moves from `Game.new()`; if a custom FEN start is later added, preserve an initial-game root.
   - Clear reusable AI trees after undo.

7. `setAiConfig`
   - Request: `{ "id": 7, "cmd": "setAiConfig", "ai": { ... } }`
   - Validates and stores AI settings; does not move. Useful for UI controls.

8. `quit`
   - Request: `{ "id": 8, "cmd": "quit" }`
   - Returns `ok: true` and exits the backend loop cleanly.

### Error codes
Use stable string codes so Swift can present friendly messages:
- `invalid_json`
- `invalid_request`
- `unknown_command`
- `invalid_move`
- `illegal_move`
- `terminal_position`
- `configuration_error`
- `checkpoint_error`
- `internal_error`

## Tasks

1. **[Done] Define backend GUI protocol dataclasses and serializers**
   - File: `src/tinychess/protocols/gui.py`
   - Changes: add `GuiConfig`, `GuiSession`, request dispatch, JSON-lines loop, state serialization helpers, AI config validation, and error response helpers.
   - Acceptance: unit tests can instantiate `GuiSession`, call command handlers via strings/streams, and receive valid JSON with canonical state.
   - Completed: implemented in `src/tinychess/protocols/gui.py` with focused coverage in `tests/test_gui_protocol.py`.

2. **[Done] Add CLI entrypoint for backend server**
   - File: `src/tinychess/cli.py`
   - Changes: add `gui-server` subcommand with options for `--seed`, optional default AI budgets, and maybe `--traceback` for local debugging; call `run_gui_loop()`.
   - Acceptance: `uv run tinychess gui-server` responds to `hello`, `state`, and `quit` JSON-lines requests; `uv run tinychess --help` lists the subcommand.
   - Completed: implemented in `src/tinychess/cli.py` with focused CLI coverage in `tests/test_cli.py`.

3. **[Done] Test Python protocol state and move commands**
   - File: `tests/test_gui_protocol.py`
   - Changes: add tests for `hello`, `state`, `newGame`, legal-move shape, legal `makeMove`, illegal move error with state, auto-queen promotion from a fixture FEN, and `quit`.
   - Acceptance: targeted tests pass and validate response JSON rather than relying on text substrings.
   - Completed: expanded focused GUI protocol state/move coverage in `tests/test_gui_protocol.py`.

4. **[Done] Test Python AI commands**
   - File: `tests/test_gui_protocol.py`
   - Changes: add deterministic/smoke tests for `aiMove` with `random` and small-budget `mcts`; add neural missing-checkpoint/configuration-error test without requiring a real checkpoint.
   - Acceptance: AI commands apply legal moves and return search metadata where available; neural remains optional and fails gracefully when no checkpoint is supplied.
   - Completed: implemented and tested GUI `aiMove` support for random, classical MCTS, and graceful neural missing-checkpoint failure in `src/tinychess/protocols/gui.py` and `tests/test_gui_protocol.py`.

5. **[Done] Add Swift app target skeleton**
   - File: `swift/Package.swift`
   - Changes: add an executable target `TinyChessMacApp` for macOS SwiftUI and a test target `TinyChessMacAppTests` if SPM supports the desired app shape. Keep existing `TinyChessCore` unchanged except dependency declarations if needed.
   - File: `swift/Sources/TinyChessMacApp/TinyChessMacApp.swift`
   - Changes: define `@main` SwiftUI `App` and root window.
   - Acceptance: `(cd swift && swift build)` succeeds and launches/builds an empty native window in development tooling.
   - Completed: added the SwiftPM `TinyChessMacApp` executable target, placeholder SwiftUI root window, and focused app skeleton test.

6. **[Done] Add Swift protocol models**
   - Files: `swift/Sources/TinyChessMacApp/BackendModels.swift`, `swift/Tests/TinyChessMacAppTests/BackendModelsTests.swift`
   - Changes: define `Codable` request/response/state/piece/outcome/AI config models matching protocol v1; parse sample JSON from Python tests.
   - Acceptance: Swift tests decode representative `state`, error, and `aiMove` responses.
   - Completed: added Swift `Codable` GUI protocol DTOs and focused decode/encode tests for representative backend responses and requests.

7. **[Done] Add Swift backend subprocess client**
   - File: `swift/Sources/TinyChessMacApp/BackendClient.swift`
   - Changes: implement an actor or main-safe service that launches `uv run tinychess gui-server` in repo/dev mode, sends serialized requests, reads line-delimited responses, maps errors, and terminates on app close.
   - Acceptance: a smoke/manual run can send `hello` and `state` from the app without blocking the main thread; backend stderr is captured/logged for diagnostics.
   - Completed: added the Swift JSON-lines subprocess client with focused mock-process tests for response decoding, backend errors, stderr capture, and lifecycle cleanup.

8. **[Done] Add app state/view model**
   - File: `swift/Sources/TinyChessMacApp/AppState.swift`
   - Changes: create `@MainActor` observable state for board, selected square, legal destinations, move history, human color, AI config, orientation, thinking status, and errors. Add actions: new game, select square, make move, request AI move, undo, flip board.
   - Acceptance: unit tests or manual smoke verify state transitions from mocked backend responses; UI does not issue moves while `thinking` is true.
   - Completed: added `AppState` with a mockable backend seam and focused tests for state transitions, selection, backend errors, AI/undo request seams, and input blocking while thinking.

9. **[Done] Implement board UI with Unicode pieces**
   - Files: `swift/Sources/TinyChessMacApp/BoardView.swift`, `swift/Sources/TinyChessMacApp/SquareView.swift`
   - Changes: render 8x8 board with orientation flip, Unicode chess glyphs from FEN symbols, selected-square highlight, legal destination highlights, and last-move highlight. Use click source/destination input only; no drag/drop for MVP.
   - Acceptance: starting position displays correctly for White and Black orientation; legal destinations highlight after selecting a piece; last move remains highlighted.
   - Completed: added SwiftUI board/square rendering driven by AppState plus focused tests for orientation, square color parity, last-move parsing, and Unicode FEN glyph mapping.

10. **[Done] Implement controls, move list, and status**
   - Files: `swift/Sources/TinyChessMacApp/ControlsView.swift`, `swift/Sources/TinyChessMacApp/MoveListView.swift`, root content view if separate.
   - Changes: add start/reset controls, human color picker, AI kind/budget controls, optional neural checkpoint path field, flip board, thinking indicator, outcome/status text, move list/history, and error banner.
   - Acceptance: user can configure Random/MCTS/neural path, start/reset, see moves as UCI strings, see outcome/draw status, and manually flip orientation.
   - Completed: added SwiftUI controls, status/error presentation, UCI move history, root view composition, and focused presentation tests.

11. **Wire human-vs-AI flow**
   - Files: `swift/Sources/TinyChessMacApp/AppState.swift`, `BackendClient.swift`, UI files.
   - Changes: after a successful human `makeMove`, if game is ongoing, invoke `aiMove` in a background task; if human selected Black, invoke `aiMove` after `newGame`; block duplicate input while thinking.
   - Acceptance: manual smoke can play at least 10 plies against Random and 4 plies against MCTS without UI freeze.

12. **Add undo/reset behavior**
   - Files: Python `src/tinychess/protocols/gui.py`; Swift `AppState.swift`, `ControlsView.swift`.
   - Changes: expose `undo plies=2` in UI as "Undo last full move"; disable when no moves; reset clears selection/errors/thinking.
   - Acceptance: after human+AI moves, undo returns to the position before the human move and clears AI reusable tree state.

13. **Documentation update**
   - Files: `README.md`, `docs/architecture.md`, `swift/README.md`
   - Changes: document `tinychess gui-server`, protocol boundary, local Swift app launch, MVP limitations, and distributable-app packaging as later phase.
   - Acceptance: docs accurately describe Python-first backend and do not claim Swift chess-rule support.

14. **Local-first validation pass**
   - Files: no new product files unless fixing tests/docs.
   - Changes: run and fix issues found by targeted and full checks.
   - Acceptance: `uv run pytest`, `uv run ruff check .`, `uv run mypy`, `(cd swift && swift test)`, and `(cd swift && swift build)` pass or any environment-specific failure is documented with exact output.

15. **Bundled/distributable `.app` planning slice after MVP**
   - Files: likely `swift/README.md`, future packaging scripts under `scripts/` or `swift/`.
   - Changes: choose bundling mechanism after MVP: PyInstaller/briefcase-style backend binary, embedded Python environment, or documented `uv` dependency. Add app setting to locate backend executable for development vs bundled mode.
   - Acceptance: packaging plan identifies codesigning/notarization requirements and a reproducible command; no generated apps/checkpoints committed.

## Files to Modify
- `src/tinychess/cli.py` - add `gui-server` subcommand and backend launch options.
- `src/tinychess/protocols/__init__.py` - optionally export/document GUI protocol module.
- `tests/test_cli.py` - optionally add CLI help/smoke coverage for `gui-server`.
- `README.md` - document GUI backend and app usage after implementation.
- `docs/architecture.md` - document GUI protocol boundary and SwiftUI frontend architecture.
- `swift/Package.swift` - add SwiftUI app executable/test targets.
- `swift/README.md` - document app target and local backend launch.

## New Files
- `src/tinychess/protocols/gui.py` - JSON-lines GUI protocol/session/server loop.
- `tests/test_gui_protocol.py` - Python backend protocol tests.
- `swift/Sources/TinyChessMacApp/TinyChessMacApp.swift` - app entrypoint.
- `swift/Sources/TinyChessMacApp/BackendModels.swift` - Codable protocol models.
- `swift/Sources/TinyChessMacApp/BackendClient.swift` - Python subprocess JSON-lines client.
- `swift/Sources/TinyChessMacApp/AppState.swift` - main app view model/state machine.
- `swift/Sources/TinyChessMacApp/BoardView.swift` - chessboard rendering.
- `swift/Sources/TinyChessMacApp/SquareView.swift` - individual square rendering/click handling.
- `swift/Sources/TinyChessMacApp/ControlsView.swift` - controls and AI config UI.
- `swift/Sources/TinyChessMacApp/MoveListView.swift` - move history UI.
- `swift/Tests/TinyChessMacAppTests/BackendModelsTests.swift` - Swift decoding/model tests.
- `swift/Tests/TinyChessMacAppTests/AppStateTests.swift` - optional mocked-client view model tests.

## Dependencies
- Tasks 1-4 must land before Swift end-to-end integration because they define the protocol contract.
- Task 2 depends on Task 1.
- Tasks 6-8 depend on the protocol schema from Tasks 1-4.
- Tasks 9-10 depend on Task 8's app state model.
- Task 11 depends on Tasks 7-10.
- Task 12 depends on backend `undo` support from Task 1 and UI controls from Task 10.
- Task 13 should follow backend/app behavior stabilization.
- Task 15 should wait until the local-first MVP is playable.

## Validation Plan

### Python tests
- Add focused unit tests in `tests/test_gui_protocol.py` for request parsing, response envelope, state serialization, make/undo/AI flows, errors, and CLI loop behavior.
- Run targeted checks during implementation:
  - `uv run pytest tests/test_gui_protocol.py tests/test_cli.py`
- Run full project checks before handoff:
  - `uv run pytest`
  - `uv run ruff check .`
  - `uv run mypy`

### Swift tests/build
- Add Swift tests where feasible for `Codable` models and app-state logic with a mocked backend client.
- Run from `swift/` whenever Swift files change:
  - `swift test`
  - `swift build`
  - `swift build -c release`

### Manual smoke checks
- Backend only:
  - `printf '{"id":1,"cmd":"hello"}\n{"id":2,"cmd":"state"}\n{"id":3,"cmd":"quit"}\n' | uv run tinychess gui-server`
  - Verify every stdout line is valid JSON and stderr has no traceback.
- App local MVP:
  - Launch app in development mode and verify backend starts.
  - Start a game as White vs Random; make `e2e4`; verify AI replies and last move/move list update.
  - Start as Black vs Random; verify AI makes the first move automatically.
  - Start vs MCTS with tiny budget; verify thinking indicator appears and UI remains responsive.
  - Configure neural without checkpoint; verify friendly configuration error.
  - Configure neural with a local smoke checkpoint if available; verify one move with low budget. Do not commit checkpoint.
  - Verify illegal click attempts do not desync board.
  - Verify auto-queen promotion in a controlled backend test and, if practical, manual FEN/debug setup later.

## Risks
- Swift Package Manager may not be the best vehicle for a polished `.app`; an Xcode app target may be required for assets, signing, and app lifecycle. Keep source modular so either packaging route can use it.
- `uv run tinychess gui-server` assumes the app is launched from a developer environment with `uv` and the repo available. Bundled distribution requires a separate packaging slice.
- Neural MCTS loads MLX checkpoints and may be slow or unavailable on machines without expected MLX/Apple Silicon setup. Keep neural optional and surface checkpoint/config errors cleanly.
- Long MCTS/neural searches can block the Python backend. Swift must serialize requests, run AI calls off the main actor, and expose cancel/stop only as a deferred feature unless the backend becomes asynchronous.
- Undo by replaying moves is simple for `Game.new()` but needs extra design if future GUI supports arbitrary FEN starts.
- Auto-queen promotion is intentionally limited; underpromotion is unavailable until the deferred promotion chooser.
- Unicode glyph rendering depends on system fonts and may look inconsistent; assets are deferred.
- Protocol schema drift between Python and Swift can cause runtime failures; keep sample JSON fixtures or tests synchronized.

## Open Questions
- Should the Swift app live entirely in SPM or should an Xcode project be introduced for a more conventional macOS `.app`? Default: start with SPM target if buildable, switch to Xcode project only if needed.
- Should local dev backend path be auto-detected from app launch directory, user-configurable, or hardcoded to `uv run tinychess gui-server` for the MVP? Default: configurable command with a sensible dev default.
- Is SAN move display desired soon, or are UCI strings acceptable for MVP move history? Default: UCI for MVP to avoid expanding bounded PGN/SAN surface.
- Should MCTS config expose both simulations and time limit in UI initially, or keep time limit hidden under advanced settings? Default: expose simulations prominently and optional time limit/node budget in advanced controls.

## Deferred Items
- Drag-and-drop moves.
- Native promotion chooser and underpromotion support.
- SAN/PGN move list, PGN import/export, and save/load games.
- Rich time controls and clocks.
- Asynchronous/cancellable Python search protocol with progress streaming.
- External UCI engine integration.
- Bundled piece image/vector assets.
- Strict FIDE draw claim semantics beyond existing pragmatic outcomes.
- Swift chess-engine acceleration or rule implementation.
- Notarized, codesigned distributable `.app` with bundled Python backend.
