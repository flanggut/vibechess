import pytest

from tinychess.engine import Game, Move, parse_square
from tinychess.nn import (
    ACTION_PLANES,
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    ENCODER_CHANNELS,
    ENCODER_VERSION,
    POLICY_SHAPE,
    TENSOR_SHAPE,
    action_index_to_move,
    encode_game,
    legal_move_mask,
    move_to_action_index,
    tensor_shape,
)


def move(uci: str) -> Move:
    return Move.from_uci(uci)


def test_encoder_shape_and_starting_piece_values() -> None:
    tensor = encode_game(Game.new())

    assert tensor_shape(tensor) == TENSOR_SHAPE == (ENCODER_CHANNELS, 8, 8)
    assert tensor[0][1][4] == 1.0  # white pawn e2
    assert tensor[3][0][0] == 1.0  # white rook a1
    assert tensor[5][0][4] == 1.0  # white king e1
    assert tensor[6][6][4] == 1.0  # black pawn e7
    assert tensor[9][7][7] == 1.0  # black rook h8
    assert tensor[11][7][4] == 1.0  # black king e8
    assert all(value == 0.0 for rank in tensor[12] for value in rank)  # white to move
    for channel in (13, 14, 15, 16):
        assert all(value == 1.0 for rank in tensor[channel] for value in rank)
    assert all(value == 0.0 for rank in tensor[18] for value in rank)
    assert all(value == 0.01 for rank in tensor[19] for value in rank)


def test_encoder_side_en_passant_and_clocks() -> None:
    game = Game.from_fen("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 7 12")
    tensor = encode_game(game)

    assert all(value == 1.0 for rank in tensor[12] for value in rank)
    assert tensor[17][2][3] == 1.0  # d3
    assert all(value == 0.07 for rank in tensor[18] for value in rank)
    assert all(value == 0.12 for rank in tensor[19] for value in rank)


@pytest.mark.parametrize(
    "uci",
    [
        "e2e4",  # normal pawn push
        "d5e6",  # en-passant/capture-like diagonal mapping
        "g1f3",  # knight
        "a1a8",  # rook-like long move
        "c1g5",  # bishop-like move
        "e1g1",  # castling is king's two-square queen-like move
    ],
)
def test_move_action_round_trips_without_board_for_non_underpromotions(uci: str) -> None:
    original = move(uci)
    index = move_to_action_index(original)

    decoded = action_index_to_move(index)
    assert decoded == original


@pytest.mark.parametrize(
    ("fen", "uci"),
    [
        ("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8q"),
        ("4k3/8/8/8/8/8/4p3/4K3 b - - 0 1", "e2e1q"),
    ],
)
def test_queen_promotion_round_trips_with_board_context(fen: str, uci: str) -> None:
    game = Game.from_fen(fen)
    original = move(uci)
    index = move_to_action_index(original, game.board)

    assert action_index_to_move(index, game.board) == original


@pytest.mark.parametrize("uci", ["a7a8n", "b7a8b", "b7c8r"])
def test_white_underpromotion_round_trips_with_board(uci: str) -> None:
    game = Game.from_fen("4k3/PP6/8/8/8/8/8/4K3 w - - 0 1")
    original = move(uci)
    index = move_to_action_index(original, game.board)

    assert action_index_to_move(index, game.board) == original


@pytest.mark.parametrize("uci", ["a2a1n", "b2c1b", "b2a1r"])
def test_black_underpromotion_round_trips_with_board(uci: str) -> None:
    game = Game.from_fen("4k3/8/8/8/8/8/pp6/4K3 b - - 0 1")
    original = move(uci)
    index = move_to_action_index(original, game.board)

    assert action_index_to_move(index, game.board) == original


def test_underpromotion_requires_board_context() -> None:
    index = 8 * ACTION_PLANES + 64

    with pytest.raises(ValueError, match="underpromotion action mapping requires board state"):
        move_to_action_index(move("a7a8n"))
    with pytest.raises(ValueError, match="underpromotion action decoding requires board state"):
        action_index_to_move(index)


def test_invalid_and_unrepresentable_actions_are_rejected() -> None:
    with pytest.raises(ValueError, match="action index"):
        action_index_to_move(ACTION_SPACE_SIZE)
    with pytest.raises(ValueError, match="off-board"):
        action_index_to_move(move_to_action_index(move("h1h8")) + 7)  # h1, NE one step
    with pytest.raises(ValueError, match="not representable"):
        move_to_action_index(Move(parse_square("a1"), parse_square("b4")))


def test_malformed_promotions_are_rejected_with_board_context() -> None:
    game = Game.from_fen("4k3/8/8/8/8/8/3pP3/R3K3 w - - 0 1")

    with pytest.raises(ValueError, match="promotion target must be the final rank"):
        move_to_action_index(move("e2e3q"), game.board)
    with pytest.raises(ValueError, match="promotion move must be by the side-to-move pawn"):
        move_to_action_index(move("a1a8q"), game.board)
    with pytest.raises(ValueError, match="promotion move must be by the side-to-move pawn"):
        move_to_action_index(move("d2d1q"), game.board)


def test_legal_move_mask_marks_exact_legal_indices_from_start() -> None:
    game = Game.new()
    mask = legal_move_mask(game)
    legal_indices = {move_to_action_index(legal, game.board) for legal in game.legal_moves}

    assert len(mask) == ACTION_SPACE_SIZE
    assert sum(mask) == 20.0
    assert {index for index, value in enumerate(mask) if value == 1.0} == legal_indices
    assert mask[move_to_action_index(move("e2e4"), game.board)] == 1.0
    assert mask[move_to_action_index(move("e2e5"), game.board)] == 0.0


def test_legal_move_mask_includes_castling_and_underpromotions() -> None:
    castle_game = Game.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    castle_mask = legal_move_mask(castle_game)
    assert castle_mask[move_to_action_index(move("e1g1"), castle_game.board)] == 1.0
    assert castle_mask[move_to_action_index(move("e1c1"), castle_game.board)] == 1.0

    black_castle_game = Game.from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    black_castle_mask = legal_move_mask(black_castle_game)
    assert black_castle_mask[move_to_action_index(move("e8g8"), black_castle_game.board)] == 1.0
    assert black_castle_mask[move_to_action_index(move("e8c8"), black_castle_game.board)] == 1.0

    promo_game = Game.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    promo_mask = legal_move_mask(promo_game)
    for uci in ("a7a8q", "a7a8n", "a7a8b", "a7a8r"):
        assert promo_mask[move_to_action_index(move(uci), promo_game.board)] == 1.0


def test_action_space_metadata() -> None:
    assert ACTION_SPACE_VERSION == "az-8x8x73-v1"
    assert ENCODER_VERSION == "tinychess-board-v1"
    assert ACTION_PLANES == 73
    assert ACTION_SPACE_SIZE == 4672
    assert POLICY_SHAPE == (64, 73)
