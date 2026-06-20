"""Dependency-free helpers shared by benchmark and generation scripts."""

from __future__ import annotations

import math
from pathlib import Path


def rate(count: int, elapsed_seconds: float) -> float:
    """Return ``count`` per second, or infinity when no time elapsed."""
    if elapsed_seconds == 0:
        return math.inf
    return count / elapsed_seconds


def directory_size(directory: Path) -> int:
    """Return the total size in bytes of all files under ``directory``."""
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


__all__ = ["directory_size", "rate"]
