"""Game state, history, outcomes, and complete-game simulation."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from tinychess.engine.board import Board
from tinychess.engine.legal_moves import is_in_check, legal_moves
from tinychess.engine.move import Move
from tinychess.engine.outcome import Outcome, OutcomeReason
from tinychess.engine.piece import Color, Piece, PieceType
from tinychess.engine.square import Square, file_index, rank_index

MoveSelector = Callable[[Board, tuple[Move, ...]], Move]


@dataclass(frozen=True, slots=True)
class Game:
    """A chess game with immutable board snapshots and pragmatic outcome tracking.

    State transitions use the existing copy-on-apply baseline: every move creates a new
    ``Board`` snapshot and stores it in ``positions``. This keeps the API simple and safe
    for early MCTS/self-play work. A make/unmake backend can be introduced later behind
    the same game/board APIs if benchmarks justify it.
    """

    positions: tuple[Board, ...] = field(default_factory=lambda: (Board.starting_position(),))
    moves: tuple[Move, ...] = ()
    halfmove_clock: int = 0
    fullmove_number: int = 1
    repetition_counts: Mapping[PositionKey, int] = field(default_factory=dict)
    forced_outcome: Outcome | None = None

    def __post_init__(self) -> None:
        if not self.positions:
            msg = "game must contain at least one board position"
            raise ValueError(msg)
        if not self.repetition_counts:
            object.__setattr__(self, "repetition_counts", {_position_key(self.board): 1})

    @classmethod
    def new(cls, board: Board | None = None) -> Game:
        """Return a new game from ``board`` or the standard starting position."""
        start = Board.starting_position() if board is None else board
        return cls(positions=(start,), repetition_counts={_position_key(start): 1})

    @classmethod
    def from_fen(cls, fen: str) -> Game:
        """Return a game initialized from a full six-field FEN string."""
        from tinychess.engine.fen import parse_fen

        position = parse_fen(fen)
        return cls(
            positions=(position.board,),
            halfmove_clock=position.halfmove_clock,
            fullmove_number=position.fullmove_number,
            repetition_counts={_position_key(position.board): 1},
        )

    def to_fen(self) -> str:
        """Serialize the current game position to full FEN."""
        from tinychess.engine.fen import board_to_fen

        return board_to_fen(
            self.board,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
        )

    @classmethod
    def from_pgn(cls, text: str) -> Game:
        """Parse bounded PGN text and return the final game state."""
        from tinychess.engine.pgn import parse_pgn

        return parse_pgn(text).final_game

    def to_pgn(self, *, tags: Mapping[str, str] | None = None, result: str | None = None) -> str:
        """Serialize this game's mainline history to bounded PGN."""
        from tinychess.engine.pgn import game_to_pgn

        return game_to_pgn(self, tags=tags, result=result)

    @property
    def board(self) -> Board:
        """Return the current board."""
        return self.positions[-1]

    @property
    def legal_moves(self) -> tuple[Move, ...]:
        """Return legal moves in the current position."""
        return legal_moves(self.board)

    @property
    def outcome(self) -> Outcome | None:
        """Return the current outcome, or ``None`` if the game is ongoing."""
        return determine_outcome(self)

    def play(self, move: Move) -> Game:
        """Return the game after applying a legal move."""
        if self.outcome is not None:
            msg = f"cannot play move after game outcome: {self.outcome.reason.value}"
            raise ValueError(msg)
        legal = self.legal_moves
        if move not in legal:
            msg = f"illegal move: {move}"
            raise ValueError(msg)

        board = self.board
        moving_piece = board.piece_at(move.from_square)
        if moving_piece is None:  # defensive; legal membership should prevent this
            msg = f"cannot move from empty square {move.from_square}"
            raise ValueError(msg)
        is_capture = _is_capture(board, move, moving_piece)
        next_board = board.apply_move(move)
        next_key = _position_key(next_board)
        next_repetitions = dict(self.repetition_counts)
        next_repetitions[next_key] = next_repetitions.get(next_key, 0) + 1

        next_halfmove_clock = (
            0 if moving_piece.kind is PieceType.PAWN or is_capture else self.halfmove_clock + 1
        )
        next_fullmove_number = self.fullmove_number + (
            1 if board.side_to_move is Color.BLACK else 0
        )

        return Game(
            positions=(*self.positions, next_board),
            moves=(*self.moves, move),
            halfmove_clock=next_halfmove_clock,
            fullmove_number=next_fullmove_number,
            repetition_counts=dict(next_repetitions),
            forced_outcome=None,
        )


def determine_outcome(game: Game) -> Outcome | None:
    """Return the pragmatic game outcome, or ``None`` if the game is ongoing."""
    if game.forced_outcome is not None:
        return game.forced_outcome
    board = game.board
    moves = legal_moves(board)
    if not moves:
        if is_in_check(board, board.side_to_move):
            return Outcome(reason=OutcomeReason.CHECKMATE, winner=board.side_to_move.opposite)
        return Outcome(reason=OutcomeReason.STALEMATE)

    if game.halfmove_clock >= 100:
        return Outcome(reason=OutcomeReason.FIFTY_MOVE)
    if game.repetition_counts.get(_position_key(board), 0) >= 3:
        return Outcome(reason=OutcomeReason.REPETITION)
    if has_insufficient_material(board):
        return Outcome(reason=OutcomeReason.INSUFFICIENT_MATERIAL)
    return None


def simulate_game(
    selector: MoveSelector,
    *,
    game: Game | None = None,
    max_plies: int = 512,
) -> Game:
    """Play a complete game using ``selector`` until an outcome or ply cap."""
    if max_plies < 0:
        msg = f"max_plies must be non-negative, got {max_plies}"
        raise ValueError(msg)
    current = Game.new() if game is None else game
    for _ in range(max_plies):
        if current.outcome is not None:
            return current
        moves = current.legal_moves
        if not moves:
            return current
        current = current.play(selector(current.board, moves))
    if current.outcome is None:
        return _with_max_plies_draw(current)
    return current


def random_move_selector(seed: int | None = None) -> MoveSelector:
    """Return a deterministic random legal-move selector when ``seed`` is provided."""
    rng = random.Random(seed)

    def select(_board: Board, moves: tuple[Move, ...]) -> Move:
        return rng.choice(moves)

    return select


def has_insufficient_material(board: Board) -> bool:
    """Return whether material is insufficient for a pragmatic checkmate possibility."""
    pieces = tuple(piece for _square, piece in board.occupied_squares())
    non_kings = [piece for piece in pieces if piece.kind is not PieceType.KING]
    if not non_kings:
        return True
    if any(piece.kind in {PieceType.PAWN, PieceType.ROOK, PieceType.QUEEN} for piece in non_kings):
        return False
    if len(non_kings) == 1:
        return non_kings[0].kind in {PieceType.BISHOP, PieceType.KNIGHT}
    if all(piece.kind is PieceType.BISHOP for piece in non_kings):
        bishop_colors = {
            _square_color(square)
            for square, piece in board.occupied_squares()
            if piece.kind is PieceType.BISHOP
        }
        return len(bishop_colors) == 1
    return False


PositionKey = tuple[tuple[Piece | None, ...], Color, frozenset[str], Square | None]


def _position_key(board: Board) -> PositionKey:
    return (board.squares, board.side_to_move, board.castling_rights, board.en_passant_target)


def _is_capture(board: Board, move: Move, moving_piece: Piece) -> bool:
    if board.piece_at(move.to_square) is not None:
        return True
    return (
        moving_piece.kind is PieceType.PAWN
        and board.en_passant_target == move.to_square
        and abs(int(move.to_square) - int(move.from_square)) in {7, 9}
    )


def _square_color(square: Square) -> int:
    return (file_index(square) + rank_index(square)) % 2


def _with_max_plies_draw(game: Game) -> Game:
    return Game(
        positions=game.positions,
        moves=game.moves,
        halfmove_clock=game.halfmove_clock,
        fullmove_number=game.fullmove_number,
        repetition_counts=dict(game.repetition_counts),
        forced_outcome=Outcome(OutcomeReason.MAX_PLIES),
    )
