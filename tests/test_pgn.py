from __future__ import annotations

import pytest

from tinychess.engine import (
    Game,
    Move,
    PgnGame,
    format_pgn,
    legal_moves,
    move_to_san,
    parse_pgn,
    parse_san,
)
from tinychess.engine.pgn import parse_pgn_with_trace


def play_uci(*moves: str) -> Game:
    game = Game.new()
    for notation in moves:
        game = game.play(Move.from_uci(notation))
    return game


def test_simple_opening_game_round_trip_with_tags() -> None:
    text = """[Event \"Tiny Test\"]
[Site \"Local\"]
[Date \"2026.05.24\"]
[Round \"1\"]
[White \"Alice\"]
[Black \"Bob\"]
[Result \"*\"]

1. e4 e5 2. Nf3 Nc6 *
"""

    pgn = parse_pgn(text)

    assert pgn.tags["Event"] == "Tiny Test"
    assert pgn.moves == tuple(Move.from_uci(move) for move in ("e2e4", "e7e5", "g1f3", "b8c6"))
    assert pgn.final_game.to_fen() == play_uci("e2e4", "e7e5", "g1f3", "b8c6").to_fen()
    assert format_pgn(pgn) == text.strip()


def test_fools_mate_san_includes_checkmate_and_result() -> None:
    game = play_uci("f2f3", "e7e5", "g2g4", "d8h4")

    pgn_text = game.to_pgn(tags={"Event": "Mate"})

    assert "2. g4 Qh4# 0-1" in pgn_text
    parsed = parse_pgn(pgn_text)
    assert parsed.result == "0-1"
    assert parsed.final_game.outcome is not None


def test_parse_pgn_with_trace_matches_normal_parse_and_replayed_positions() -> None:
    text = """[Event \"Trace\"]
[Result \"1-0\"]

1. e4 e5 2. Nf3 Nc6 1-0
"""

    parsed = parse_pgn(text)
    traced = parse_pgn_with_trace(text)

    assert traced.game.moves == parsed.moves
    assert traced.game.result == parsed.result
    assert traced.game.tags == parsed.tags
    assert len(traced.plies) == len(parsed.moves)

    game = parsed.initial_game
    for ply, move in zip(traced.plies, parsed.moves, strict=True):
        assert ply.board == game.board
        assert ply.halfmove_clock == game.halfmove_clock
        assert ply.fullmove_number == game.fullmove_number
        assert ply.move == move
        assert ply.legal_moves == legal_moves(game.board)
        assert move in ply.legal_moves
        game = game.play(move)
    assert game.to_fen() == parsed.final_game.to_fen()


def test_castling_san_parse_and_write() -> None:
    game = play_uci("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6")

    castle = parse_san(game.board, "O-O")
    next_game = game.play(castle)

    assert castle == Move.from_uci("e1g1")
    assert move_to_san(game.board, castle) == "O-O"
    assert "5. O-O" in next_game.to_pgn(result="*")


def test_capture_san_parse_and_write() -> None:
    game = play_uci("e2e4", "d7d5")
    capture = parse_san(game.board, "exd5")

    assert capture == Move.from_uci("e4d5")
    assert move_to_san(game.board, capture) == "exd5"
    assert game.play(capture).moves[-1] == capture


def test_game_to_pgn_auto_emits_fen_setup_for_custom_start() -> None:
    game = Game.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").play(Move.from_uci("a7a8q"))

    pgn_text = game.to_pgn()
    parsed = parse_pgn(pgn_text)

    assert '[SetUp "1"]' in pgn_text
    assert '[FEN "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"]' in pgn_text
    assert parsed.moves == (Move.from_uci("a7a8q"),)
    assert parsed.final_game.to_fen() == game.to_fen()


def test_game_to_pgn_auto_emits_fen_setup_with_counters_for_unplayed_fen() -> None:
    game = Game.from_fen("4k3/8/8/8/8/8/8/4K3 b - - 99 57")

    pgn_text = game.to_pgn()
    parsed = parse_pgn(pgn_text)

    assert '[FEN "4k3/8/8/8/8/8/8/4K3 b - - 99 57"]' in pgn_text
    assert parsed.final_game.to_fen() == game.to_fen()


def test_fen_setup_and_promotion_san() -> None:
    text = """[Event \"Promotion\"]
[Site \"?\"]
[Date \"????.??.??\"]
[Round \"?\"]
[White \"?\"]
[Black \"?\"]
[Result \"*\"]
[SetUp \"1\"]
[FEN \"4k3/P7/8/8/8/8/8/4K3 w - - 0 1\"]

1. a8=Q+ *
"""

    pgn = parse_pgn(text)

    assert pgn.moves == (Move.from_uci("a7a8q"),)
    assert pgn.final_game.board.piece_at("a8") is not None
    assert format_pgn(pgn) == text.strip()


def test_disambiguation_san() -> None:
    game = Game.from_fen("4k3/8/8/8/8/2N1N3/8/4K3 w - - 0 1")

    assert move_to_san(game.board, Move.from_uci("c3d5")) == "Ncd5"
    assert move_to_san(game.board, Move.from_uci("e3d5")) == "Ned5"
    assert parse_san(game.board, "Ncd5") == Move.from_uci("c3d5")
    assert parse_san(game.board, "Ned5") == Move.from_uci("e3d5")
    with pytest.raises(ValueError, match="not legal"):
        parse_san(game.board, "Nd5")
    with pytest.raises(ValueError, match="not legal"):
        parse_san(game.board, "Nc3d5")


def test_disambiguation_requires_rank_for_same_file_conflict() -> None:
    game = Game.from_fen("4k3/8/8/3N4/8/3N4/8/4K3 w - - 0 1")

    assert move_to_san(game.board, Move.from_uci("d3f4")) == "N3f4"
    assert move_to_san(game.board, Move.from_uci("d5f4")) == "N5f4"
    assert parse_san(game.board, "N3f4") == Move.from_uci("d3f4")
    assert parse_san(game.board, "N5f4") == Move.from_uci("d5f4")
    with pytest.raises(ValueError, match="not legal"):
        parse_san(game.board, "Ndf4")
    with pytest.raises(ValueError, match="not legal"):
        parse_san(game.board, "Nf4")


def test_disambiguation_requires_full_square_for_file_and_rank_conflicts() -> None:
    game = Game.from_fen("4k3/8/8/3N4/8/3N3N/8/4K3 w - - 0 1")

    assert move_to_san(game.board, Move.from_uci("d3f4")) == "Nd3f4"
    assert parse_san(game.board, "Nd3f4") == Move.from_uci("d3f4")
    with pytest.raises(ValueError, match="not legal"):
        parse_san(game.board, "N3f4")
    with pytest.raises(ValueError, match="not legal"):
        parse_san(game.board, "Ndf4")
    with pytest.raises(ValueError, match="not legal"):
        parse_san(game.board, "Nf4")


def test_parse_pgn_allows_punctuation_in_tag_values() -> None:
    pgn = parse_pgn('[Event "Tiny (Test) {Tag}"]\n[Result "*"]\n\n*')

    assert pgn.tags["Event"] == "Tiny (Test) {Tag}"


def test_pgn_game_defaults_common_tags() -> None:
    pgn = format_pgn(PgnGame(moves=(Move.from_uci("e2e4"),), result="*"))

    assert pgn.startswith('[Event "?"]\n[Site "?"]\n[Date "????.??.??"]')
    assert "\n\n1. e4 *" in pgn


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('[Result "*"]\n\n1. e4 {comment} *', "comments"),
        ('[Result "*"]\n\n1. e4 ; comment\n*', "comments"),
        ('[Result "*"]\n\n1. e4 (1. d4) *', "variations"),
        ('[Result "*"]\n\n1. e4 $1 *', "numeric annotation glyphs"),
        ('[Result "*"]\n\n1. e4 {%clk 0:05:00} *', "clock annotations"),
        ('[Result "*"]\n\n1. e4! *', "annotation suffixes"),
        ('[Result "*"]\n\n1. e4 e5 2. exd6 e.p. *', "en-passant annotation"),
    ],
)
def test_parse_pgn_rejects_unsupported_features(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_pgn(text)


def test_parse_pgn_rejects_fen_without_setup() -> None:
    with pytest.raises(ValueError, match='SetUp "1"'):
        parse_pgn('[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]\n\n*')


def test_parse_pgn_rejects_setup_without_fen() -> None:
    with pytest.raises(ValueError, match="requires a FEN tag"):
        parse_pgn('[SetUp "1"]\n\n*')


def test_parse_pgn_rejects_invalid_setup_value() -> None:
    with pytest.raises(ValueError, match="SetUp"):
        parse_pgn('[SetUp "yes"]\n\n*')


def test_parse_pgn_rejects_result_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        parse_pgn('[Result "1-0"]\n\n1. e4 *')


def test_parse_pgn_rejects_tokens_after_result() -> None:
    with pytest.raises(ValueError, match="after PGN result"):
        parse_pgn('[Result "*"]\n\n1. e4 * 0-1')


def test_parse_san_rejects_capture_marker_mismatches() -> None:
    capture_game = play_uci("e2e4", "d7d5")

    with pytest.raises(ValueError, match="not legal"):
        parse_san(capture_game.board, "ed5")
    with pytest.raises(ValueError, match="not legal"):
        parse_san(Game.new().board, "exd3")


def test_parse_san_resolves_en_passant_without_annotation() -> None:
    game = play_uci("e2e4", "a7a6", "e4e5", "d7d5")

    assert parse_san(game.board, "exd6") == Move.from_uci("e5d6")
    with pytest.raises(ValueError, match="en-passant annotation"):
        parse_san(game.board, "exd6e.p.")


def test_castling_accepts_zero_normalization_and_exact_check_suffix() -> None:
    game = Game.from_fen("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1")

    assert parse_san(game.board, "O-O-O+") == Move.from_uci("e1c1")
    assert parse_san(game.board, "0-0-0+") == Move.from_uci("e1c1")
    with pytest.raises(ValueError, match="not legal"):
        parse_san(game.board, "O-O-O")


def test_parse_san_resolves_promotions_and_rejects_unsupported_pieces() -> None:
    game = Game.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")

    assert parse_san(game.board, "a8=Q+") == Move.from_uci("a7a8q")
    assert parse_san(game.board, "a8=N") == Move.from_uci("a7a8n")
    with pytest.raises(ValueError, match="unsupported SAN promotion"):
        parse_san(game.board, "a8=K")


def test_parse_san_requires_exact_checkmate_suffix() -> None:
    game = play_uci("f2f3", "e7e5", "g2g4")

    assert parse_san(game.board, "Qh4#") == Move.from_uci("d8h4")
    with pytest.raises(ValueError, match="not legal"):
        parse_san(game.board, "Qh4+")
    with pytest.raises(ValueError, match="not legal"):
        parse_san(game.board, "Qh4")


def test_parse_san_rejects_invalid_repeated_check_suffix() -> None:
    with pytest.raises(ValueError, match="check/mate suffix"):
        parse_san(Game.new().board, "e4++")
