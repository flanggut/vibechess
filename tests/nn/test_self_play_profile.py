from __future__ import annotations

from contextlib import ExitStack
from typing import cast

from vibechess.profiling import (
    ProfileStats,
    activate_self_play_profile,
    active_profiler,
    record_counter,
    record_distribution,
)


def test_legacy_self_play_profile_import_path_reexports_canonical_api() -> None:
    from vibechess.nn import self_play_profile as legacy_module

    assert legacy_module.ProfileStats is ProfileStats
    assert legacy_module.activate_self_play_profile is activate_self_play_profile
    assert legacy_module.record_counter is record_counter
    assert legacy_module.record_distribution is record_distribution


class FakeClock:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __call__(self) -> int:
        return self.values.pop(0)


def test_profile_scope_records_inclusive_exclusive_and_parent_attribution() -> None:
    clock = FakeClock([0, 10, 30, 50])

    with activate_self_play_profile("detailed", clock=clock) as profiler, ExitStack() as stack:
        stack.enter_context(profiler.scope("outer"))
        stack.enter_context(profiler.scope("inner"))

    stats = profiler.stats.to_dict()
    zones = cast(dict[str, dict[str, object]], stats["zones"])
    assert zones["outer"]["inclusive_ns"] == 50
    assert zones["outer"]["exclusive_ns"] == 30
    assert zones["inner"]["inclusive_ns"] == 20
    assert zones["inner"]["exclusive_ns"] == 20
    by_parent = cast(dict[str, dict[str, dict[str, object]]], stats["by_parent"])
    assert by_parent["inner"]["outer"]["inclusive_ns"] == 20


def test_profile_stats_merge_counters_distributions_and_round_trip() -> None:
    with activate_self_play_profile("detailed") as first:
        record_counter("items", 2)
        record_distribution("sizes", 1, unit="things")
    with activate_self_play_profile("detailed") as second:
        record_counter("items", 3)
        record_distribution("sizes", 3, unit="things")

    merged = ProfileStats.merged([first.stats, second.stats])
    data = merged.to_dict()
    counters = cast(dict[str, object], data["counters"])
    distributions = cast(dict[str, dict[str, object]], data["distributions"])
    assert counters["items"] == 5
    assert distributions["sizes"]["count"] == 2
    assert distributions["sizes"]["mean"] == 2.0

    restored = ProfileStats.from_dict(data)
    assert restored.counters["items"].value == 5


def test_noop_profile_when_disabled_restores_context() -> None:
    before = active_profiler()
    with activate_self_play_profile("detailed") as outer:
        with activate_self_play_profile("none") as profiler:
            assert not profiler.enabled
            assert active_profiler() is profiler
            record_counter("ignored")
        assert active_profiler() is outer
    assert active_profiler() is before
