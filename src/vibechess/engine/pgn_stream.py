"""Streaming PGN record reader and ingestion sanitizer."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from vibechess.engine.pgn import PgnGame, PgnGameTrace, parse_pgn, parse_pgn_with_trace

_TAG_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"((?:\\.|[^"\\])*)"\]$')
_NAG_RE = re.compile(r"\$\d+")
_ANNOTATION_RE = re.compile(r"(?<=\S)[!?]+(?=\s|$)")


@dataclass(frozen=True, slots=True)
class PgnRecord:
    """One raw PGN record read from a stream."""

    index: int
    text: str


def iter_pgn_records(path: str | Path) -> Iterator[PgnRecord]:
    """Yield raw PGN game records from ``path`` without loading the full file."""
    current: list[str] = []
    index = 0
    with Path(path).expanduser().open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            starts_record = line.lstrip().startswith("[Event ")
            has_current_record = current and any(item.strip() for item in current)
            if starts_record and has_current_record:
                yield PgnRecord(index=index, text="".join(current).strip() + "\n")
                index += 1
                current = [line]
            else:
                current.append(line)
    if current and any(item.strip() for item in current):
        yield PgnRecord(index=index, text="".join(current).strip() + "\n")


def sanitize_pgn_text(text: str) -> str:
    """Return PGN text with common public-dataset annotations removed.

    The core PGN parser intentionally remains strict. This sanitizer is used by
    ingestion to tolerate comments, line comments, recursive variations, NAGs,
    and simple ``!``/``?`` annotation suffixes in large external datasets.
    """
    without_line_comments = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    without_brace_comments = _strip_brace_comments(without_line_comments)
    without_variations = _strip_parenthesized_variations(without_brace_comments)
    without_nags = _NAG_RE.sub(" ", without_variations)
    without_annotations = _ANNOTATION_RE.sub("", without_nags)
    return "\n".join(line.rstrip() for line in without_annotations.splitlines()).strip() + "\n"


def pgn_has_fen_setup(text: str) -> bool:
    """Return whether a PGN record declares a non-standard setup/FEN."""
    tags = pgn_tags(text)
    return tags.get("SetUp") == "1" or "FEN" in tags


def pgn_tags(text: str) -> dict[str, str]:
    """Extract PGN tag pairs without parsing movetext."""
    tags: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if tags:
                break
            continue
        if not line.startswith("["):
            break
        match = _TAG_RE.match(line)
        if match is not None:
            tags[match.group(1)] = bytes(match.group(2), "utf-8").decode("unicode_escape")
    return tags


def parse_ingest_pgn(text: str, *, strict: bool = False) -> PgnGame:
    """Parse a PGN record for ingestion, sanitizing first unless ``strict``."""
    return parse_pgn(text if strict else sanitize_pgn_text(text))


def parse_ingest_pgn_with_trace(text: str, *, strict: bool = False) -> PgnGameTrace:
    """Parse a PGN record for ingestion with per-ply parser trace data."""
    return parse_pgn_with_trace(text if strict else sanitize_pgn_text(text))


def _strip_brace_comments(text: str) -> str:
    result: list[str] = []
    depth = 0
    for char in text:
        if char == "{":
            depth += 1
            result.append(" ")
        elif char == "}" and depth:
            depth -= 1
            result.append(" ")
        elif depth == 0:
            result.append(char)
        elif char == "\n":
            result.append("\n")
    return "".join(result)


def _strip_parenthesized_variations(text: str) -> str:
    result: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
            result.append(" ")
        elif char == ")" and depth:
            depth -= 1
            result.append(" ")
        elif depth == 0:
            result.append(char)
        elif char == "\n":
            result.append("\n")
    return "".join(result)


__all__ = [
    "PgnRecord",
    "iter_pgn_records",
    "parse_ingest_pgn",
    "parse_ingest_pgn_with_trace",
    "pgn_has_fen_setup",
    "pgn_tags",
    "sanitize_pgn_text",
]
