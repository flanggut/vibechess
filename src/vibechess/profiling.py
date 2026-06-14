"""Opt-in dependency-neutral instrumentation helpers.

The profiler is intentionally small and process-local.  It records stack-based
inclusive/exclusive wall-clock timings, counters, simple distributions, and immediate
parent/child attribution.  Disabled profiling uses a no-op singleton so hot callers can
check ``profiler.enabled`` before constructing expensive tags.

This module deliberately lives outside :mod:`vibechess.nn` so engine and AI hot paths can
use profiling without initializing neural-network/MLX modules.
"""

from __future__ import annotations

import contextvars
import math
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from heapq import heappush, heappushpop
from typing import Literal, Protocol, Self

ProfileLevel = Literal["none", "summary", "detailed"]
PROFILE_FORMAT_VERSION = 2
SLOW_ITEM_LIMIT = 20


class _Clock(Protocol):
    def __call__(self) -> int: ...


@dataclass(slots=True)
class TimerAggregate:
    """Aggregate inclusive/exclusive nanoseconds for one zone name."""

    calls: int = 0
    inclusive_ns: int = 0
    exclusive_ns: int = 0

    def add(self, *, inclusive_ns: int, exclusive_ns: int) -> None:
        self.calls += 1
        self.inclusive_ns += max(0, inclusive_ns)
        self.exclusive_ns += max(0, exclusive_ns)

    def merge(self, other: TimerAggregate) -> None:
        self.calls += other.calls
        self.inclusive_ns += other.inclusive_ns
        self.exclusive_ns += other.exclusive_ns

    def to_dict(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "inclusive_ns": self.inclusive_ns,
            "exclusive_ns": self.exclusive_ns,
            "inclusive_seconds": self.inclusive_ns / 1_000_000_000.0,
            "exclusive_seconds": self.exclusive_ns / 1_000_000_000.0,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        return cls(
            calls=_expect_int(data, "calls"),
            inclusive_ns=_timer_ns(data, "inclusive"),
            exclusive_ns=_timer_ns(data, "exclusive"),
        )


@dataclass(slots=True)
class CounterAggregate:
    """Numeric counter aggregate."""

    value: float = 0.0

    def add(self, amount: int | float) -> None:
        self.value += float(amount)

    def merge(self, other: CounterAggregate) -> None:
        self.value += other.value

    def to_json_value(self) -> int | float:
        if self.value.is_integer():
            return int(self.value)
        return self.value


@dataclass(slots=True)
class DistributionAggregate:
    """Bounded online distribution summary.

    Values are retained for percentile reporting.  The plan's target runs have small enough
    aggregate cardinality for this diagnostic sidecar; callers should still prefer counters in
    extremely hot loops.
    """

    unit: str
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    min: float | None = None
    max: float | None = None
    values: list[float] = field(default_factory=list)

    def add(self, value: int | float) -> None:
        numeric = float(value)
        self.count += 1
        self.total += numeric
        self.total_square += numeric * numeric
        self.min = numeric if self.min is None else min(self.min, numeric)
        self.max = numeric if self.max is None else max(self.max, numeric)
        self.values.append(numeric)

    def merge(self, other: DistributionAggregate) -> None:
        if self.unit != other.unit:
            self.unit = self.unit or other.unit
        self.count += other.count
        self.total += other.total
        self.total_square += other.total_square
        if other.min is not None:
            self.min = other.min if self.min is None else min(self.min, other.min)
        if other.max is not None:
            self.max = other.max if self.max is None else max(self.max, other.max)
        self.values.extend(other.values)

    def to_dict(self) -> dict[str, object]:
        mean = self.total / self.count if self.count else 0.0
        variance = max(0.0, self.total_square / self.count - mean * mean) if self.count else 0.0
        return {
            "unit": self.unit,
            "count": self.count,
            "min": 0.0 if self.min is None else self.min,
            "max": 0.0 if self.max is None else self.max,
            "mean": mean,
            "stddev": math.sqrt(variance),
            "p50": self.percentile(50),
            "p90": self.percentile(90),
            "p99": self.percentile(99),
        }

    def percentile(self, percentile: int) -> float:
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        index = int(math.ceil(percentile / 100.0 * len(sorted_values))) - 1
        index = min(max(index, 0), len(sorted_values) - 1)
        return sorted_values[index]

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        distribution = cls(unit=_expect_str(data, "unit"))
        distribution.count = _expect_int(data, "count")
        mean = _expect_float(data, "mean")
        distribution.total = mean * distribution.count
        minimum = _expect_float(data, "min") if distribution.count else None
        maximum = _expect_float(data, "max") if distribution.count else None
        distribution.min = minimum
        distribution.max = maximum
        if distribution.count:
            # Raw values are intentionally not serialized; retain an approximate
            # representative so merged reports keep useful percentile fields.
            distribution.values = [mean]
        return distribution


@dataclass(slots=True)
class ProfileZone:
    """One active profiling stack frame."""

    name: str
    start_ns: int
    child_ns: int = 0
    tags: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ProfileStats:
    """Serializable process-local profiling aggregate."""

    zones: dict[str, TimerAggregate] = field(default_factory=dict)
    counters: dict[str, CounterAggregate] = field(default_factory=dict)
    distributions: dict[str, DistributionAggregate] = field(default_factory=dict)
    by_parent: dict[str, dict[str, TimerAggregate]] = field(default_factory=dict)
    slowest_plies: list[dict[str, object]] = field(default_factory=list)
    slowest_searches: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def add_zone(self, name: str, *, inclusive_ns: int, exclusive_ns: int) -> None:
        self.zones.setdefault(name, TimerAggregate()).add(
            inclusive_ns=inclusive_ns,
            exclusive_ns=exclusive_ns,
        )

    def add_parent_child(
        self,
        parent: str,
        child: str,
        *,
        inclusive_ns: int,
        exclusive_ns: int,
    ) -> None:
        children = self.by_parent.setdefault(child, {})
        children.setdefault(parent, TimerAggregate()).add(
            inclusive_ns=inclusive_ns,
            exclusive_ns=exclusive_ns,
        )

    def add_counter(self, name: str, amount: int | float = 1) -> None:
        self.counters.setdefault(name, CounterAggregate()).add(amount)

    def add_distribution(self, name: str, value: int | float, *, unit: str) -> None:
        self.distributions.setdefault(name, DistributionAggregate(unit=unit)).add(value)

    def add_slow_item(self, kind: Literal["ply", "search"], item: dict[str, object]) -> None:
        target = self.slowest_plies if kind == "ply" else self.slowest_searches
        target.append(dict(item))
        target.sort(key=lambda entry: _object_float(entry.get("seconds", 0.0)), reverse=True)
        del target[SLOW_ITEM_LIMIT:]

    def merge(self, other: ProfileStats) -> None:
        for name, timer in other.zones.items():
            self.zones.setdefault(name, TimerAggregate()).merge(timer)
        for name, counter in other.counters.items():
            self.counters.setdefault(name, CounterAggregate()).merge(counter)
        for name, distribution in other.distributions.items():
            self.distributions.setdefault(
                name,
                DistributionAggregate(unit=distribution.unit),
            ).merge(distribution)
        for child, parents in other.by_parent.items():
            target_parents = self.by_parent.setdefault(child, {})
            for parent, aggregate in parents.items():
                target_parents.setdefault(parent, TimerAggregate()).merge(aggregate)
        for item in other.slowest_plies:
            self.add_slow_item("ply", item)
        for item in other.slowest_searches:
            self.add_slow_item("search", item)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable v2 stats object plus v1 timer compatibility."""
        return {
            "format_version": PROFILE_FORMAT_VERSION,
            "zones": {name: aggregate.to_dict() for name, aggregate in sorted(self.zones.items())},
            "counters": {
                name: aggregate.to_json_value() for name, aggregate in sorted(self.counters.items())
            },
            "distributions": {
                name: aggregate.to_dict()
                for name, aggregate in sorted(self.distributions.items())
            },
            "by_parent": {
                child: {
                    parent: aggregate.to_dict()
                    for parent, aggregate in sorted(parents.items())
                }
                for child, parents in sorted(self.by_parent.items())
            },
            "timers": self.compat_timers(),
        }

    def compat_timers(self) -> dict[str, dict[str, object]]:
        """Return the older benchmark timer shape for existing report consumers."""
        return {
            "game_legal_moves": self._compat_timer(("game.legal_moves",)),
            "determine_outcome": self._compat_timer(("game.determine_outcome",)),
            "game_play_known_legal": self._compat_timer(("game.play_known_legal",)),
            "board_apply_move": self._compat_timer(("board.apply_move",)),
            "model_single": self._compat_timer(
                ("inference.predict", "inference.predict_with_legal_moves")
            ),
            "model_batch": {
                **self._compat_timer(("inference.predict_batch",)),
                "positions": int(
                    self.counters.get(
                        "inference.batch_positions",
                        CounterAggregate(),
                    ).value
                ),
                "batch_size_min": self._distribution_value("inference.batch_size", "min"),
                "batch_size_max": self._distribution_value("inference.batch_size", "max"),
                "batch_size_mean": self._distribution_value("inference.batch_size", "mean") or 0.0,
            },
            "model_legal_batch": {
                **self._compat_timer(("inference.predict_legal_batch",)),
                "positions": int(
                    self.counters.get(
                        "inference.legal_batch_positions",
                        CounterAggregate(),
                    ).value
                ),
                "batch_size_min": self._distribution_value(
                    "inference.legal_batch_size",
                    "min",
                ),
                "batch_size_max": self._distribution_value(
                    "inference.legal_batch_size",
                    "max",
                ),
                "batch_size_mean": self._distribution_value(
                    "inference.legal_batch_size",
                    "mean",
                ) or 0.0,
            },
            "search": {
                **self._compat_timer(("mcts.search",)),
                "materialized_nodes": int(
                    self.counters.get("mcts.materialized_nodes", CounterAggregate()).value
                ),
                "completed_simulations": int(
                    self.counters.get("mcts.completed_simulations", CounterAggregate()).value
                ),
            },
        }

    def _compat_timer(self, names: Sequence[str]) -> dict[str, object]:
        calls = 0
        seconds = 0.0
        for name in names:
            aggregate = self.zones.get(name)
            if aggregate is not None:
                calls += aggregate.calls
                seconds += aggregate.inclusive_ns / 1_000_000_000.0
        return {"calls": calls, "seconds": seconds}

    def _distribution_value(self, name: str, key: str) -> int | float | None:
        distribution = self.distributions.get(name)
        if distribution is None or distribution.count == 0:
            return None
        if key == "min":
            return distribution.min
        if key == "max":
            return distribution.max
        if key == "mean":
            return distribution.total / distribution.count
        raise KeyError(key)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Parse v2 stats or a v1 compatibility timer object."""
        stats = cls()
        zones_data = data.get("zones")
        if isinstance(zones_data, Mapping):
            for name, raw in zones_data.items():
                if isinstance(name, str) and isinstance(raw, Mapping):
                    stats.zones[name] = TimerAggregate.from_dict(raw)
            counters_data = data.get("counters")
            if isinstance(counters_data, Mapping):
                for name, value in counters_data.items():
                    if (
                        isinstance(name, str)
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    ):
                        stats.counters[name] = CounterAggregate(float(value))
            distributions_data = data.get("distributions")
            if isinstance(distributions_data, Mapping):
                for name, raw in distributions_data.items():
                    if isinstance(name, str) and isinstance(raw, Mapping):
                        stats.distributions[name] = DistributionAggregate.from_dict(raw)
            by_parent_data = data.get("by_parent")
            if isinstance(by_parent_data, Mapping):
                for child, parents in by_parent_data.items():
                    if isinstance(child, str) and isinstance(parents, Mapping):
                        stats.by_parent[child] = {
                            str(parent): TimerAggregate.from_dict(raw)
                            for parent, raw in parents.items()
                            if isinstance(raw, Mapping)
                        }
            return stats

        timers_data = data.get("timers")
        if not isinstance(timers_data, Mapping):
            raise TypeError("profile stats must contain 'zones' or 'timers'")
        v1_map = {
            "game_legal_moves": "game.legal_moves",
            "determine_outcome": "game.determine_outcome",
            "game_play_known_legal": "game.play_known_legal",
            "board_apply_move": "board.apply_move",
            "model_single": "inference.predict_with_legal_moves",
            "model_batch": "inference.predict_batch",
            "search": "mcts.search",
        }
        for old_name, zone_name in v1_map.items():
            raw = timers_data.get(old_name)
            if isinstance(raw, Mapping):
                seconds = _expect_float(raw, "seconds")
                calls = _expect_int(raw, "calls")
                ns = int(seconds * 1_000_000_000)
                stats.zones[zone_name] = TimerAggregate(
                    calls=calls,
                    inclusive_ns=ns,
                    exclusive_ns=ns,
                )
        batch = timers_data.get("model_batch")
        if isinstance(batch, Mapping):
            stats.counters["inference.batch_positions"] = CounterAggregate(
                float(_expect_int(batch, "positions"))
            )
        search = timers_data.get("search")
        if isinstance(search, Mapping):
            stats.counters["mcts.materialized_nodes"] = CounterAggregate(
                float(_expect_int(search, "materialized_nodes"))
            )
            stats.counters["mcts.completed_simulations"] = CounterAggregate(
                float(_expect_int(search, "completed_simulations"))
            )
        return stats

    @classmethod
    def merged(cls, profiles: Sequence[ProfileStats]) -> ProfileStats:
        merged = cls()
        for profile in profiles:
            merged.merge(profile)
        return merged


class SelfPlayProfiler:
    """Process-local hierarchical self-play profiler."""

    enabled: bool = True

    def __init__(
        self,
        *,
        level: ProfileLevel = "detailed",
        metadata: Mapping[str, object] | None = None,
        clock: _Clock = time.perf_counter_ns,
    ) -> None:
        self.level = level
        self.stats = ProfileStats(metadata=dict(metadata or {}))
        self._clock = clock
        self._stack: list[ProfileZone] = []
        self._slow_heaps: dict[
            Literal["ply", "search"], list[tuple[float, int, dict[str, object]]]
        ] = {
            "ply": [],
            "search": [],
        }
        # Monotonic tiebreaker so heap ordering never compares the payload dicts
        # when two records share an identical duration.
        self._slow_sequence = 0

    @contextmanager
    def scope(self, name: str, **tags: object) -> Iterator[None]:
        start_ns = self._clock()
        frame = ProfileZone(name=name, start_ns=start_ns, tags=dict(tags))
        self._stack.append(frame)
        try:
            yield
        finally:
            end_ns = self._clock()
            popped = self._stack.pop()
            elapsed_ns = max(0, end_ns - popped.start_ns)
            exclusive_ns = max(0, elapsed_ns - popped.child_ns)
            self.stats.add_zone(name, inclusive_ns=elapsed_ns, exclusive_ns=exclusive_ns)
            if self._stack:
                parent = self._stack[-1]
                parent.child_ns += elapsed_ns
                self.stats.add_parent_child(
                    parent.name,
                    name,
                    inclusive_ns=elapsed_ns,
                    exclusive_ns=exclusive_ns,
                )
            seconds = elapsed_ns / 1_000_000_000.0
            if name == "self_play.ply":
                self.stats.add_distribution("self_play.ply_seconds", seconds, unit="seconds")
                self._record_slow("ply", seconds, popped.tags)
            elif name == "mcts.search":
                self.stats.add_distribution("mcts.search_seconds", seconds, unit="seconds")
                self._record_slow("search", seconds, popped.tags)
            elif name == "mcts.simulation":
                self.stats.add_distribution("mcts.simulation_seconds", seconds, unit="seconds")

    def counter(self, name: str, amount: int | float = 1, **_tags: object) -> None:
        self.stats.add_counter(name, amount)

    def distribution(
        self,
        name: str,
        value: int | float,
        *,
        unit: str,
        **_tags: object,
    ) -> None:
        self.stats.add_distribution(name, value, unit=unit)

    def _record_slow(
        self,
        kind: Literal["ply", "search"],
        seconds: float,
        tags: Mapping[str, object],
    ) -> None:
        item = {"seconds": seconds, **dict(tags)}
        heap = self._slow_heaps[kind]
        self._slow_sequence += 1
        entry = (seconds, self._slow_sequence, item)
        if len(heap) < SLOW_ITEM_LIMIT:
            heappush(heap, entry)
        elif seconds > heap[0][0]:
            heappushpop(heap, entry)
        target = self.stats.slowest_plies if kind == "ply" else self.stats.slowest_searches
        target[:] = [entry_item for _seconds, _sequence, entry_item in sorted(heap, reverse=True)]

    def to_dict(self) -> dict[str, object]:
        return self.stats.to_dict()


class NoOpSelfPlayProfiler:
    """No-op profiler with a compatible API."""

    enabled = False
    level: ProfileLevel = "none"
    stats: ProfileStats = ProfileStats()

    def scope(self, _name: str, **_tags: object) -> nullcontext[None]:
        return nullcontext()

    def counter(self, _name: str, _amount: int | float = 1, **_tags: object) -> None:
        return

    def distribution(
        self,
        _name: str,
        _value: int | float,
        *,
        unit: str,
        **_tags: object,
    ) -> None:
        return

    def to_dict(self) -> dict[str, object]:
        return self.stats.to_dict()


_NOOP = NoOpSelfPlayProfiler()
_ACTIVE_PROFILER: contextvars.ContextVar[SelfPlayProfiler | NoOpSelfPlayProfiler] = (
    contextvars.ContextVar("vibechess_self_play_profiler", default=_NOOP)
)


def normalize_profile_level(value: str | None) -> ProfileLevel:
    """Normalize environment/CLI profile level values."""
    if value is None or value == "" or value == "0" or value.lower() == "none":
        return "none"
    lowered = value.lower()
    if lowered == "1":
        return "detailed"
    if lowered in {"summary", "detailed"}:
        return lowered  # type: ignore[return-value]
    raise ValueError(f"unsupported self-play profile level: {value!r}")


def profile_level_from_env(env_var: str = "VIBECHESS_SELF_PLAY_PROFILE") -> ProfileLevel:
    return normalize_profile_level(os.environ.get(env_var))


@contextmanager
def activate_self_play_profile(
    level: str | None = "detailed",
    metadata: Mapping[str, object] | None = None,
    *,
    clock: _Clock = time.perf_counter_ns,
) -> Iterator[SelfPlayProfiler | NoOpSelfPlayProfiler]:
    """Activate a profiler in the current context and restore the previous one."""
    normalized = normalize_profile_level(level)
    if normalized == "none":
        token = _ACTIVE_PROFILER.set(_NOOP)
        try:
            yield _NOOP
        finally:
            _ACTIVE_PROFILER.reset(token)
        return
    profiler = SelfPlayProfiler(level=normalized, metadata=metadata, clock=clock)
    token = _ACTIVE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _ACTIVE_PROFILER.reset(token)


def active_profiler() -> SelfPlayProfiler | NoOpSelfPlayProfiler:
    """Return the active profiler or a no-op profiler."""
    return _ACTIVE_PROFILER.get()


def profile_scope(name: str, **tags: object) -> AbstractContextManager[None]:
    """Return a named profiling scope, or a no-op context manager when disabled."""
    profiler = active_profiler()
    if not profiler.enabled:
        return nullcontext()
    return profiler.scope(name, **tags)


def record_counter(name: str, amount: int | float = 1, **tags: object) -> None:
    profiler = active_profiler()
    if profiler.enabled:
        profiler.counter(name, amount, **tags)


def record_distribution(name: str, value: int | float, *, unit: str, **tags: object) -> None:
    profiler = active_profiler()
    if profiler.enabled:
        profiler.distribution(name, value, unit=unit, **tags)


def profile_report(
    profiler_or_stats: SelfPlayProfiler | ProfileStats,
    *,
    scope: str,
    profile_level: ProfileLevel,
    worker_profiles: Sequence[Mapping[str, object]] | None = None,
    derived: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a v2 sidecar/report profile dictionary."""
    stats = (
        profiler_or_stats.stats
        if isinstance(profiler_or_stats, SelfPlayProfiler)
        else profiler_or_stats
    )
    workers = list(worker_profiles or [])
    return {
        "format_version": PROFILE_FORMAT_VERSION,
        "scope": scope,
        "profile_level": profile_level,
        "clock": "time.perf_counter_ns",
        "process": {"pid": os.getpid()},
        "metadata": dict(metadata or stats.metadata),
        "stats": stats.to_dict(),
        "worker_profiles": workers,
        "workers": workers,
        "derived": dict(derived or {}),
        "slowest_plies": stats.slowest_plies,
        "slowest_searches": stats.slowest_searches,
        "limitations": profile_limitations(),
    }


def profile_limitations() -> list[str]:
    return [
        "Exclusive times are stack-attributed within a process; parent/worker clocks "
        "are not globally synchronized.",
        "MLX compute is lazy; mlx.sync.* timers are the best proxy for realized model compute.",
        "Detailed profiling adds overhead; use --no-profile or --profile-overhead-check "
        "for throughput comparisons.",
    ]


def stats_from_profile_report(report: Mapping[str, object]) -> ProfileStats:
    stats_data = report.get("stats", report)
    if not isinstance(stats_data, Mapping):
        raise TypeError("profile report 'stats' must be an object")
    return ProfileStats.from_dict(stats_data)


# Compatibility name for older callers/tests.
SelfPlayProfileStats = ProfileStats


def _object_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return 0.0
    return float(value)


def _timer_ns(data: Mapping[str, object], prefix: str) -> int:
    raw_ns = data.get(f"{prefix}_ns")
    if isinstance(raw_ns, int) and not isinstance(raw_ns, bool):
        return raw_ns
    seconds = _expect_float(data, f"{prefix}_seconds")
    return int(seconds * 1_000_000_000)


def _expect_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"profile field {key!r} must be an integer")
    return value


def _expect_float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"profile field {key!r} must be a number")
    return float(value)


def _expect_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"profile field {key!r} must be a string")
    return value
