"""Core chess engine primitives."""

from tinychess.engine.board import STARTING_POSITION, Board
from tinychess.engine.game import Game, random_move_selector, simulate_game
from tinychess.engine.legal_moves import legal_moves, perft, pseudo_legal_moves
from tinychess.engine.move import Move
from tinychess.engine.outcome import Outcome, OutcomeReason
from tinychess.engine.piece import Color, Piece, PieceType
from tinychess.engine.square import (
    Square,
    file_index,
    make_square,
    parse_square,
    rank_index,
    square_name,
)

__all__ = [
    "STARTING_POSITION",
    "Board",
    "Color",
    "Game",
    "Move",
    "Outcome",
    "OutcomeReason",
    "Piece",
    "PieceType",
    "legal_moves",
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
