"""Regression tests for slow-item recording with tied durations."""

from __future__ import annotations

from vibechess.profiling import SLOW_ITEM_LIMIT, SelfPlayProfiler


def test_record_slow_handles_tied_durations() -> None:
    # A constant clock makes every scope report an identical (zero) duration, so
    # heap entries tie on ``seconds`` and would otherwise force a dict comparison.
    profiler = SelfPlayProfiler(level="detailed", clock=lambda: 0)

    for ply_index in range(SLOW_ITEM_LIMIT + 5):
        with profiler.scope("self_play.ply", ply_index=ply_index):
            pass

    slowest = profiler.stats.slowest_plies
    assert len(slowest) == SLOW_ITEM_LIMIT
    assert all(item["seconds"] == 0.0 for item in slowest)
    # Distinct payloads are preserved rather than collapsed or crashing the sort.
    assert len({item["ply_index"] for item in slowest}) == SLOW_ITEM_LIMIT
