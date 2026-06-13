# Backlog

Future work items that are intentionally deferred from the current implementation plan. Backlog entries should preserve enough context for a future planner or worker to pick them up without rereading the original plan, including scope, likely files, acceptance criteria, constraints, and why the work is deferred.

## GUI / macOS Packaging

### Bundled/distributable `.app` planning slice after MVP

- Source plan: `plans/gui-plan.md` Task 15.
- Status: Backlog / deferred until after the local-first GUI MVP is playable and validated.
- Why deferred:
  - The current MVP is developer/local first: the SwiftUI app can launch the Python backend with `uv run vibechess gui-server`.
  - A distributable macOS `.app` needs packaging, codesigning, notarization, and dependency-bundling decisions that should be made after the app/backend flow is stable.
  - Packaging work should not block core GUI playability, protocol correctness, or Swift UI iteration.

#### Goal

Choose and document a reproducible packaging approach for a native macOS `.app` that includes or locates the Python vibechess GUI backend.

#### Likely files

- `swift/README.md` - document the chosen packaging workflow and developer-vs-bundled backend modes.
- Future packaging scripts under `scripts/` or `swift/` - automate building/copying/bundling the backend and app artifacts.
- Swift app configuration files, if introduced later - add a setting or launch configuration for locating the backend executable in development and bundled modes.

#### Scope / changes to plan

- Compare and choose a backend bundling mechanism, such as:
  - PyInstaller or similar standalone backend binary.
  - A bundled/embedded Python environment.
  - A documented external `uv` dependency for developer-only builds.
- Add an app/backend location strategy:
  - development mode can use `uv run vibechess gui-server` from a configured repo path;
  - bundled mode should locate the packaged backend relative to the `.app` bundle.
- Identify the packaging command sequence needed to build the distributable app.
- Identify codesigning and notarization requirements for macOS distribution.
- Keep generated artifacts out of source control.

#### Acceptance criteria

- A packaging plan identifies the chosen bundling mechanism and tradeoffs.
- The plan includes a reproducible command or command sequence for creating the distributable app/backend bundle.
- The plan documents codesigning and notarization requirements or explicitly records them as follow-up work with concrete next steps.
- The app has a clear strategy for locating the backend executable in both development and bundled modes.
- No generated `.app` bundles, packaged backend binaries, checkpoints, downloaded corpora, or other generated artifacts are committed.

#### Constraints and non-goals

- Do not rewrite chess legality or AI in Swift as part of packaging.
- Do not commit generated data, checkpoints, packaged apps, build outputs, or local smoke outputs.
- Keep neural checkpoint loading optional; packaging should not require committing checkpoints.
- Keep the Python engine/backend as the correctness reference unless a future Swift acceleration task has separate parity tests and benchmark evidence.
- Do not broaden GUI protocol, UCI, or PGN semantics just to support packaging.
