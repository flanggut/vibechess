from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import cast

import pytest

from tinychess.engine import (
    Board,
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

game_module = importlib.import_module("tinychess.engine.game")
legal_moves_module = importlib.import_module("tinychess.engine.legal_moves")
pgn_module = importlib.import_module("tinychess.engine.pgn")


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


def test_parse_pgn_uses_one_full_legal_generation_per_normal_ply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = _count_full_legal_move_generations(monkeypatch)
    parsed = parse_pgn('[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 *')

    assert count() == len(parsed.moves)


def test_parse_pgn_uses_one_full_legal_generation_per_checkmate_ply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = _count_full_legal_move_generations(monkeypatch)
    suffix_checks = _count_has_legal_move_suffix_checks(monkeypatch)
    parsed = parse_pgn('[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1')

    assert count() == len(parsed.moves)
    assert suffix_checks() == 1


def test_parse_pgn_without_check_suffixes_uses_no_response_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix_checks = _count_has_legal_move_suffix_checks(monkeypatch)
    parse_pgn('[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 *')

    assert suffix_checks() == 0


def _count_full_legal_move_generations(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    original = cast(Callable[[Board], tuple[Move, ...]], legal_moves_module.legal_moves)
    calls = 0

    def counted(board: Board) -> tuple[Move, ...]:
        nonlocal calls
        calls += 1
        return original(board)

    monkeypatch.setattr(pgn_module, "legal_moves", counted)
    monkeypatch.setattr(game_module, "legal_moves", counted)
    return lambda: calls


def _count_has_legal_move_suffix_checks(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    original = cast(Callable[[Board], bool], legal_moves_module.has_legal_move)
    calls = 0

    def counted(board: Board) -> bool:
        nonlocal calls
        calls += 1
        return original(board)

    monkeypatch.setattr(pgn_module, "has_legal_move", counted)
    return lambda: calls


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


@pytest.mark.parametrize(
    "text",
    [
        """[Event "TraceCastle"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O *
""",
        """[Event "TraceEnPassant"]
[Result "*"]

1. e4 a6 2. e5 d5 3. exd6 *
""",
        """[Event "TracePromotion"]
[SetUp "1"]
[FEN "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"]
[Result "*"]

1. a8=Q+ *
""",
        """[Event "TraceBlackQuiet"]
[SetUp "1"]
[FEN "r3k3/8/8/8/8/8/8/4K3 b q - 7 12"]
[Result "*"]

12... Ra7 *
""",
    ],
)
def test_parse_pgn_trace_special_moves_preserves_pre_move_state(text: str) -> None:
    traced = parse_pgn_with_trace(text)
    reference = traced.game.initial_game

    assert len(traced.plies) == len(traced.game.moves)
    for ply, move in zip(traced.plies, traced.game.moves, strict=True):
        assert ply.board == reference.board
        assert ply.halfmove_clock == reference.halfmove_clock
        assert ply.fullmove_number == reference.fullmove_number
        assert ply.move == move
        assert move in ply.legal_moves
        reference = reference.play(move)

    assert traced.game.final_game.to_fen() == reference.to_fen()
    assert traced.game.final_game.outcome == reference.outcome


def test_parse_pgn_with_trace_preserves_fen_clocks() -> None:
    text = """[Event "TraceFen"]
[SetUp "1"]
[FEN "4k3/8/8/8/8/8/8/R3K3 w Q - 12 34"]
[Result "*"]

34. Ra2 *
"""

    traced = parse_pgn_with_trace(text)

    assert len(traced.plies) == 1
    assert traced.plies[0].halfmove_clock == 12
    assert traced.plies[0].fullmove_number == 34
    assert traced.game.final_game.to_fen() == play_uci_from_fen(
        "4k3/8/8/8/8/8/8/R3K3 w Q - 12 34", "a1a2"
    ).to_fen()


@pytest.mark.parametrize(
    "text",
    [
        """[Event "Castle"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O *
""",
        """[Event "EnPassant"]
[Result "*"]

1. e4 a6 2. e5 d5 3. exd6 *
""",
        """[Event "Promotion"]
[SetUp "1"]
[FEN "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"]
[Result "*"]

1. a8=Q+ *
""",
        """[Event "Mate"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1
""",
    ],
)
def test_parse_pgn_fast_state_matches_game_play_reference(text: str) -> None:
    parsed = parse_pgn(text)
    reference = parsed.initial_game
    for move in parsed.moves:
        reference = reference.play(move)

    assert parsed.final_game.to_fen() == reference.to_fen()
    assert parsed.final_game.outcome == reference.outcome


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (
            """[Result "*"]

1. f3 e5 2. g4 Qh4# 3. a3 *
""",
            "not legal",
        ),
        (
            """[Result "*"]

1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. Nf3 *
""",
            "repetition",
        ),
        (
            """[SetUp "1"]
[FEN "4k3/8/8/8/8/8/8/R3K3 w Q - 100 1"]
[Result "*"]

1. Ra2 *
""",
            "fifty_move",
        ),
        (
            """[SetUp "1"]
[FEN "4k3/8/8/8/8/8/8/4K3 w - - 0 1"]
[Result "*"]

1. Kd2 *
""",
            "insufficient_material",
        ),
    ],
)
def test_parse_pgn_rejects_moves_after_terminal_outcomes(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_pgn(text)


def play_uci_from_fen(fen: str, *moves: str) -> Game:
    game = Game.from_fen(fen)
    for notation in moves:
        game = game.play(Move.from_uci(notation))
    return game


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
