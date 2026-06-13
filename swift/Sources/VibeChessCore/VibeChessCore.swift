/// Bootstrap surface for the future Swift acceleration backend.
///
/// The Python package remains the correctness reference. This module intentionally
/// exposes only stable metadata until fixture-backed engine parity work begins.
public enum VibeChessCore {
    /// Package-level semantic version aligned with the Python project while the
    /// Swift backend is an experimental acceleration workspace.
    public static let version = "0.1.0"

    /// Board indexing convention shared with the Python reference engine.
    public static let squareIndexingConvention = "0..63 with a1 == 0 and h8 == 63"

    /// Current implementation stage for downstream smoke tests and docs.
    public static let implementationStage = "bootstrap"
}
