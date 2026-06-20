"""Shared JSON field-extraction primitives with typed validation.

These centralize the ``isinstance`` checks that several modules previously
re-implemented. Each accepts a ``label`` so callers can preserve their own
error-message prefixes (for example ``"checkpoint metadata field"``).
"""

from __future__ import annotations

from collections.abc import Mapping


def expect_str(data: Mapping[str, object], key: str, *, label: str = "field") -> str:
    """Return ``data[key]`` as a string or raise ``TypeError``."""
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{label} {key!r} must be a string")
    return value


def expect_int(data: Mapping[str, object], key: str, *, label: str = "field") -> int:
    """Return ``data[key]`` as an integer (rejecting ``bool``) or raise ``TypeError``."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} {key!r} must be an integer")
    return value


def expect_number(data: Mapping[str, object], key: str, *, label: str = "field") -> float:
    """Return ``data[key]`` as a float (rejecting ``bool``) or raise ``TypeError``."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{label} {key!r} must be a number")
    return float(value)


def expect_bool(data: Mapping[str, object], key: str, *, label: str = "field") -> bool:
    """Return ``data[key]`` as a boolean or raise ``TypeError``."""
    value = data.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{label} {key!r} must be a boolean")
    return value


__all__ = ["expect_bool", "expect_int", "expect_number", "expect_str"]
