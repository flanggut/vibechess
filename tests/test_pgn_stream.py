from __future__ import annotations

from pathlib import Path

import pytest

from vibechess.engine.pgn_stream import (
    iter_pgn_records,
    parse_ingest_pgn,
    parse_ingest_pgn_with_trace,
    sanitize_pgn_text,
)


def test_iter_pgn_records_streams_multiple_games(tmp_path: Path) -> None:
    path = tmp_path / "games.pgn"
    path.write_text(
        '[Event "A"]\n[Result "*"]\n\n1. e4 *\n\n'
        '[Event "B"]\n[Result "*"]\n\n1. d4 *\n'
    )

    records = list(iter_pgn_records(path))

    assert [record.index for record in records] == [0, 1]
    assert '[Event "A"]' in records[0].text
    assert '[Event "B"]' in records[1].text


def test_sanitize_pgn_text_removes_common_annotations() -> None:
    text = '[Result "*"]\n\n1. e4! {comment} (1. d4) $1 e5?! ; line comment\n2. Nf3 *\n'

    sanitized = sanitize_pgn_text(text)
    pgn = parse_ingest_pgn(text)

    assert "comment" not in sanitized
    assert "$1" not in sanitized
    assert "!" not in sanitized
    assert "?" not in sanitized
    assert [move.to_uci() for move in pgn.moves] == ["e2e4", "e7e5", "g1f3"]


def test_parse_ingest_pgn_with_trace_matches_normal_ingest_for_sanitized_record() -> None:
    text = '[Event "A"]\n[Result "0-1"]\n\n1. d4! {good} d5?! (1... Nf6) 2. c4 0-1\n'

    pgn = parse_ingest_pgn(text)
    traced = parse_ingest_pgn_with_trace(text)

    assert traced.game.moves == pgn.moves
    assert traced.game.result == pgn.result
    assert traced.game.tags == pgn.tags
    assert [ply.move for ply in traced.plies] == list(pgn.moves)


def test_parse_ingest_pgn_with_trace_matches_normal_ingest_in_strict_mode() -> None:
    text = '[Event "A"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n'

    pgn = parse_ingest_pgn(text, strict=True)
    traced = parse_ingest_pgn_with_trace(text, strict=True)

    assert traced.game.moves == pgn.moves
    assert traced.game.result == pgn.result
    assert traced.game.tags == pgn.tags
    assert [ply.move for ply in traced.plies] == list(pgn.moves)


@pytest.mark.parametrize(
    ("text", "strict"),
    [
        ('[Result "*"]\n\n1. e4 {comment} *\n', True),
        ('[Result "1-0"]\n\n1. e4 *\n', False),
        ('[Result "*"]\n\n1. e4+ *\n', False),
        ('[Result "*"]\n\n1. e5 *\n', False),
        ('[Result "*"]\n\n1. e4 * 0-1\n', False),
    ],
)
def test_parse_ingest_pgn_with_trace_rejects_same_invalid_records_as_normal_ingest(
    text: str,
    strict: bool,
) -> None:
    with pytest.raises(ValueError):
        parse_ingest_pgn(text, strict=strict)
    with pytest.raises(ValueError):
        parse_ingest_pgn_with_trace(text, strict=strict)
