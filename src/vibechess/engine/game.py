"""Game state, history, outcomes, and complete-game simulation."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import vibechess.engine.transition as _transition
from vibechess.engine.board import Board
from vibechess.engine.legal_moves import legal_moves
from vibechess.engine.move import Move
from vibechess.engine.outcome import Outcome, OutcomeReason
from vibechess.engine.piece import Piece
from vibechess.engine.square import Square
from vibechess.profiling import profile_scope, record_counter

PositionKey = _transition.PositionKey
MoveSelector = Callable[[Board, tuple[Move, ...]], Move]
# Backward-compatible module seam for tests and callers that patch the old name.
generate_legal_moves = legal_moves


def _current_legal_moves(board: Board) -> tuple[Move, ...]:
    """Return legal moves through the patchable module seam."""
    return legal_moves(board)


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
            object.__setattr__(self, "repetition_counts", {_transition.position_key(self.board): 1})

    @classmethod
    def new(cls, board: Board | None = None) -> Game:
        """Return a new game from ``board`` or the standard starting position."""
        start = Board.starting_position() if board is None else board
        return cls(positions=(start,), repetition_counts={_transition.position_key(start): 1})

    @classmethod
    def from_fen(cls, fen: str) -> Game:
        """Return a game initialized from a full six-field FEN string."""
        from vibechess.engine.fen import parse_fen

        position = parse_fen(fen)
        return cls(
            positions=(position.board,),
            halfmove_clock=position.halfmove_clock,
            fullmove_number=position.fullmove_number,
            repetition_counts={_transition.position_key(position.board): 1},
        )

    def to_fen(self) -> str:
        """Serialize the current game position to full FEN."""
        from vibechess.engine.fen import board_to_fen

        return board_to_fen(
            self.board,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
        )

    @classmethod
    def from_pgn(cls, text: str) -> Game:
        """Parse bounded PGN text and return the final game state."""
        from vibechess.engine.pgn import parse_pgn

        return parse_pgn(text).final_game

    def to_pgn(self, *, tags: Mapping[str, str] | None = None, result: str | None = None) -> str:
        """Serialize this game's mainline history to bounded PGN."""
        from vibechess.engine.pgn import game_to_pgn

        return game_to_pgn(self, tags=tags, result=result)

    @property
    def board(self) -> Board:
        """Return the current board."""
        return self.positions[-1]

    @property
    def legal_moves(self) -> tuple[Move, ...]:
        """Return legal moves in the current position."""
        with profile_scope("game.legal_moves"):
            return legal_moves(self.board)

    @property
    def outcome(self) -> Outcome | None:
        """Return the current outcome, or ``None`` if the game is ongoing."""
        with profile_scope("game.outcome"):
            return determine_outcome(self)

    def play(self, move: Move) -> Game:
        """Return the game after applying a legal move."""
        if self.forced_outcome is not None:
            msg = f"cannot play move after game outcome: {self.forced_outcome.reason.value}"
            raise ValueError(msg)
        legal = self.legal_moves
        outcome = determine_outcome(self, legal_moves=legal)
        if outcome is not None:
            msg = f"cannot play move after game outcome: {outcome.reason.value}"
            raise ValueError(msg)
        if move not in legal:
            msg = f"illegal move: {move}"
            raise ValueError(msg)
        return self.play_known_legal(move)

    def play_known_legal(self, move: Move) -> Game:
        """Return the game after applying a move already known to be legal.

        This is a narrow performance path for search code that selected ``move`` from
        this game's legal move tuple and already established that the game is ongoing.
        Normal callers should use :meth:`play`, which preserves terminal and legal-move
        validation before delegating here.
        """
        with profile_scope("game.play_known_legal"):
            record_counter("game.play_known_legal.calls")
            result = _transition.advance_known_legal_state(
                _transition.TransitionState.from_position(self),
                move,
            )

            return Game(
                positions=(*self.positions, result.board),
                moves=(*self.moves, move),
                halfmove_clock=result.halfmove_clock,
                fullmove_number=result.fullmove_number,
                repetition_counts=dict(result.repetition_counts),
                forced_outcome=None,
            )

    def with_forced_outcome(self, outcome: Outcome) -> Game:
        """Return a copy of this game with ``forced_outcome`` set to ``outcome``.

        The position history, moves, and counters are preserved; only the recorded
        outcome changes. Used to stamp a terminal result (for example a ply-cap
        draw) onto a game whose board state is otherwise unchanged.
        """
        return Game(
            positions=self.positions,
            moves=self.moves,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
            repetition_counts=dict(self.repetition_counts),
            forced_outcome=outcome,
        )


@dataclass(frozen=True, slots=True)
class GameRecordFields:
    """Serializable per-game fields shared by self-play and evaluation records."""

    plies: int
    outcome_reason: str
    winner: str | None
    final_fen: str
    moves_uci: list[str]


def game_record_fields(game: Game, outcome: Outcome) -> GameRecordFields:
    """Extract the common serializable record fields for ``game`` and its ``outcome``.

    Callers resolve a concrete (non-``None``) outcome first — for example by stamping
    a ply-cap draw or raising — and pass it in, so this helper stays agnostic to how
    each harness treats unfinished games.
    """
    return GameRecordFields(
        plies=len(game.moves),
        outcome_reason=outcome.reason.value,
        winner=None if outcome.winner is None else outcome.winner.value,
        final_fen=game.to_fen(),
        moves_uci=[move.to_uci() for move in game.moves],
    )


def determine_outcome(
    game: Game, *, legal_moves: tuple[Move, ...] | None = None
) -> Outcome | None:
    """Return the pragmatic game outcome, or ``None`` if the game is ongoing."""
    with profile_scope("game.determine_outcome"):
        record_counter("game.determine_outcome.calls")
        if game.forced_outcome is not None:
            return game.forced_outcome
        moves = legal_moves if legal_moves is not None else _current_legal_moves(game.board)
        return _transition.outcome_for_state(
            _transition.TransitionState.from_position(game),
            moves,
        )


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
        return current.with_forced_outcome(Outcome(OutcomeReason.MAX_PLIES))
    return current


def random_move_selector(seed: int | None = None) -> MoveSelector:
    """Return a deterministic random legal-move selector when ``seed`` is provided."""
    rng = random.Random(seed)

    def select(_board: Board, moves: tuple[Move, ...]) -> Move:
        return rng.choice(moves)

    return select


def has_insufficient_material(board: Board) -> bool:
    """Return whether material is insufficient for a pragmatic checkmate possibility."""
    return _transition.has_insufficient_material(board)


def _position_key(board: Board) -> PositionKey:
    return _transition.position_key(board)


def _is_capture(board: Board, move: Move, moving_piece: Piece) -> bool:
    return _transition.is_capture(board, move, moving_piece)


def _square_color(square: Square) -> int:
    return _transition.square_color(square)
