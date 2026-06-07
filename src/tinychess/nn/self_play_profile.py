"""Compatibility re-export for dependency-neutral profiling helpers.

The profiling implementation lives in :mod:`tinychess.profiling` so engine and AI
hot paths can use instrumentation without importing :mod:`tinychess.nn` internals.
This module keeps the historical ``tinychess.nn.self_play_profile`` import path
working.
"""

from __future__ import annotations

from tinychess.profiling import (
    PROFILE_FORMAT_VERSION,
    SLOW_ITEM_LIMIT,
    CounterAggregate,
    DistributionAggregate,
    NoOpSelfPlayProfiler,
    ProfileLevel,
    ProfileStats,
    ProfileZone,
    SelfPlayProfiler,
    SelfPlayProfileStats,
    TimerAggregate,
    activate_self_play_profile,
    active_profiler,
    normalize_profile_level,
    profile_level_from_env,
    profile_limitations,
    profile_report,
    profile_scope,
    record_counter,
    record_distribution,
    stats_from_profile_report,
)

__all__ = [
    "PROFILE_FORMAT_VERSION",
    "SLOW_ITEM_LIMIT",
    "CounterAggregate",
    "DistributionAggregate",
    "NoOpSelfPlayProfiler",
    "ProfileLevel",
    "ProfileStats",
    "ProfileZone",
    "SelfPlayProfiler",
    "SelfPlayProfileStats",
    "TimerAggregate",
    "activate_self_play_profile",
    "active_profiler",
    "normalize_profile_level",
    "profile_level_from_env",
    "profile_limitations",
    "profile_report",
    "profile_scope",
    "record_counter",
    "record_distribution",
    "stats_from_profile_report",
]
