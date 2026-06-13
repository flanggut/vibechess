"""Core chess engine primitives."""

from vibechess.engine.board import STARTING_POSITION, Board
from vibechess.engine.fen import (
    STANDARD_STARTING_FEN,
    STARTING_FEN,
    STARTPOS_FEN,
    FenPosition,
    board_from_fen,
    board_to_fen,
    format_fen,
    parse_fen,
)
from vibechess.engine.game import Game, random_move_selector, simulate_game
from vibechess.engine.legal_moves import legal_moves, perft, pseudo_legal_moves
from vibechess.engine.move import Move
from vibechess.engine.outcome import Outcome, OutcomeReason
from vibechess.engine.pgn import PgnGame, format_pgn, game_to_pgn, move_to_san, parse_pgn, parse_san
from vibechess.engine.piece import Color, Piece, PieceType
from vibechess.engine.square import (
    Square,
    file_index,
    make_square,
    parse_square,
    rank_index,
    square_name,
)

__all__ = [
    "STARTING_FEN",
    "STARTING_POSITION",
    "STARTPOS_FEN",
    "STANDARD_STARTING_FEN",
    "Board",
    "Color",
    "FenPosition",
    "Game",
    "Move",
    "Outcome",
    "OutcomeReason",
    "Piece",
    "PieceType",
    "PgnGame",
    "board_from_fen",
    "board_to_fen",
    "format_fen",
    "format_pgn",
    "game_to_pgn",
    "legal_moves",
    "move_to_san",
    "parse_fen",
    "parse_pgn",
    "parse_san",
    "perft",
    "pseudo_legal_moves",
    "random_move_selector",
    "simulate_game",
    "Square",
    "file_index",
    "make_square",
    "parse_square",
    "rank_index",
    "square_name",
]
