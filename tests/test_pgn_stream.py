from __future__ import annotations

from pathlib import Path

from tinychess.engine.pgn_stream import iter_pgn_records, parse_ingest_pgn, sanitize_pgn_text


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
