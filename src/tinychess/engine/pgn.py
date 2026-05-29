"""Bounded PGN parsing, writing, and SAN conversion.

This module intentionally implements a small PGN subset: one mainline game with
standard tag pairs and SAN moves. Comments, NAGs, recursive variations, and clock
annotations are rejected explicitly instead of being tolerated or skipped.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from tinychess.engine.board import Board
from tinychess.engine.game import Game
from tinychess.engine.legal_moves import is_in_check, legal_moves
from tinychess.engine.move import Move
from tinychess.engine.piece import Color, PieceType
from tinychess.engine.square import FILES, RANKS, Square, file_index, parse_square, rank_index

RESULTS = frozenset({"1-0", "0-1", "1/2-1/2", "*"})
COMMON_TAGS = ("Event", "Site", "Date", "Round", "White", "Black", "Result")
_PIECE_SAN = {
    PieceType.KNIGHT: "N",
    PieceType.BISHOP: "B",
    PieceType.ROOK: "R",
    PieceType.QUEEN: "Q",
    PieceType.KING: "K",
}
_SAN_PROMOTION = {
    PieceType.QUEEN: "Q",
    PieceType.ROOK: "R",
    PieceType.BISHOP: "B",
    PieceType.KNIGHT: "N",
}
_SAN_PROMOTION_TO_PIECE = {value: key for key, value in _SAN_PROMOTION.items()}
_SAN_PIECE_TO_PIECE = {value: key for key, value in _PIECE_SAN.items()}
_TAG_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"((?:\\.|[^"\\])*)"\]$')
_MOVE_NUMBER_RE = re.compile(r"^\d+\.{1,3}$")
_MOVE_NUMBER_PREFIX_RE = re.compile(r"^\d+\.{1,3}")
_CHECK_SUFFIX_RE = re.compile(r"[+#]+$")
_CASTLING_SAN_RE = re.compile(r"^(O-O(?:-O)?)([+#])?$")
_SAN_TOKEN_RE = re.compile(
    r"^(?P<piece>[KQRBN])?"
    r"(?P<disambiguation>[a-h1-8]{0,2})"
    r"(?P<capture>x?)"
    r"(?P<target>[a-h][1-8])"
    r"(?P<promotion>=[QRBN])?"
    r"(?P<suffix>[+#])?$"
)


@dataclass(frozen=True, slots=True)
class _ParsedSan:
    piece: PieceType
    target: Square | None
    is_capture: bool
    promotion: PieceType | None
    suffix: str | None
    disambiguation: str = ""
    source_file: int | None = None
    source_rank: int | None = None
    castling_side: str | None = None


@dataclass(frozen=True, slots=True)
class PgnGame:
    """A bounded mainline PGN game.

    ``initial_game`` captures the starting position. It is the normal start
    position unless the PGN contains ``[SetUp "1"]`` and a full ``[FEN "..."]``.
    """

    tags: Mapping[str, str] = field(default_factory=dict)
    moves: tuple[Move, ...] = ()
    result: str = "*"
    initial_game: Game = field(default_factory=Game.new)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))
        if self.result not in RESULTS:
            msg = f"unsupported PGN result: {self.result!r}"
            raise ValueError(msg)

    @property
    def final_game(self) -> Game:
        """Replay and return the final game state."""
        game = self.initial_game
        for move in self.moves:
            game = game.play(move)
        return game

    def to_pgn(self) -> str:
        """Serialize this game to PGN text."""
        return format_pgn(self)


@dataclass(frozen=True, slots=True)
class PgnParsedPly:
    """Parser-computed per-ply state for ingestion reuse."""

    board: Board
    halfmove_clock: int
    fullmove_number: int
    move: Move
    legal_moves: tuple[Move, ...]


@dataclass(frozen=True, slots=True)
class PgnGameTrace:
    """A parsed PGN game plus parser-computed per-ply trace data."""

    game: PgnGame
    plies: tuple[PgnParsedPly, ...]


def move_to_san(board: Board, move: Move) -> str:
    """Return bounded SAN for a legal move from ``board``."""
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        msg = f"cannot convert move from empty square to SAN: {move}"
        raise ValueError(msg)
    legal = legal_moves(board)
    return _move_to_san_from_legal(board, move, legal, validate=True)


def _move_to_san_from_legal(
    board: Board, move: Move, legal: tuple[Move, ...], *, validate: bool
) -> str:
    """Return SAN using an already computed legal move tuple for ``board``."""
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        msg = f"cannot convert move from empty square to SAN: {move}"
        raise ValueError(msg)
    if validate and move not in legal:
        msg = f"cannot convert illegal move to SAN: {move}"
        raise ValueError(msg)

    is_castling = (
        moving_piece.kind is PieceType.KING
        and abs(int(move.to_square) - int(move.from_square)) == 2
    )
    if is_castling:
        san = "O-O" if int(move.to_square) > int(move.from_square) else "O-O-O"
    else:
        target_piece = board.piece_at(move.to_square)
        is_capture = target_piece is not None or _is_en_passant_capture(board, move)
        san = ""
        if moving_piece.kind is PieceType.PAWN:
            if is_capture:
                san += chr(ord("a") + file_index(move.from_square))
        else:
            san += _PIECE_SAN[moving_piece.kind]
            san += _disambiguation(board, move, moving_piece.kind, legal)
        if is_capture:
            san += "x"
        san += _square_name_from_move_target(move)
        if move.promotion is not None:
            if move.promotion not in _SAN_PROMOTION:
                msg = "promotion piece must be queen, rook, bishop, or knight"
                raise ValueError(msg)
            san += f"={_SAN_PROMOTION[move.promotion]}"

    next_board = board.apply_move(move)
    if is_in_check(next_board, next_board.side_to_move):
        san += "#" if not legal_moves(next_board) else "+"
    return san


def parse_san(board: Board, san: str) -> Move:
    """Resolve a bounded SAN token to a legal move from ``board``."""
    move, _legal = _parse_san_with_legal(board, san)
    return move


def parse_pgn(text: str) -> PgnGame:
    """Parse a bounded, single-mainline PGN game."""
    return _parse_pgn(text, collect_trace=False).game


def parse_pgn_with_trace(text: str) -> PgnGameTrace:
    """Parse a bounded PGN game and return parser-computed per-ply trace data."""
    return _parse_pgn(text, collect_trace=True)


def _parse_san_with_legal(board: Board, san: str) -> tuple[Move, tuple[Move, ...]]:
    _reject_unsupported_movetext(san)
    normalized = _normalize_san_token(san)
    parsed = _parse_normalized_san_token(normalized)
    legal = legal_moves(board)
    exact_candidates = [] if parsed is None else _san_exact_candidates(board, parsed, legal)
    suffix = None if parsed is None else parsed.suffix
    matches = [move for move in exact_candidates if _san_suffix_matches(board, move, suffix)]
    if not matches:
        msg = f"SAN move is not legal in the current position: {san!r}"
        raise ValueError(msg)
    if len(matches) > 1:
        msg = f"SAN move is ambiguous in the current position: {san!r}"
        raise ValueError(msg)
    return matches[0], legal


def _parse_pgn(text: str, *, collect_trace: bool) -> PgnGameTrace:
    tags, movetext = _split_tags_and_movetext(text)
    _reject_unsupported_movetext(movetext)
    result = tags.get("Result", "*")
    if result not in RESULTS:
        msg = f"unsupported PGN result tag: {result!r}"
        raise ValueError(msg)

    initial_game = _initial_game_from_tags(tags)
    current = initial_game
    moves: list[Move] = []
    plies: list[PgnParsedPly] = []
    seen_result: str | None = None
    for token in _movetext_tokens(movetext):
        if seen_result is not None:
            msg = f"unexpected token after PGN result: {token!r}"
            raise ValueError(msg)
        if token in RESULTS:
            seen_result = token
            continue
        move, legal = _parse_san_with_legal(current.board, token)
        if collect_trace:
            plies.append(
                PgnParsedPly(
                    board=current.board,
                    halfmove_clock=current.halfmove_clock,
                    fullmove_number=current.fullmove_number,
                    move=move,
                    legal_moves=legal,
                )
            )
        current = current.play(move)
        moves.append(move)

    if seen_result is not None:
        if result != "*" and result != seen_result:
            msg = f"PGN Result tag {result!r} does not match movetext result {seen_result!r}"
            raise ValueError(msg)
        result = seen_result
    tags = {**tags, "Result": result}
    game = PgnGame(tags=tags, moves=tuple(moves), result=result, initial_game=initial_game)
    return PgnGameTrace(game=game, plies=tuple(plies))


def format_pgn(game: PgnGame) -> str:
    """Serialize a bounded PGN game."""
    tags = _ordered_tags(game.tags, game.result)
    tag_lines = [f'[{name} "{_escape_tag_value(value)}"]' for name, value in tags.items()]
    movetext = _format_movetext(game.initial_game, game.moves, game.result)
    return "\n".join((*tag_lines, "", movetext))


def game_to_pgn(
    game: Game, *, tags: Mapping[str, str] | None = None, result: str | None = None
) -> str:
    """Serialize an existing ``Game`` history as PGN from its first position."""
    tag_values = dict(tags or {})
    if tag_values.get("SetUp") == "1" and "FEN" in tag_values:
        initial_game = Game.from_fen(tag_values["FEN"])
    else:
        initial_game = Game.new(game.positions[0])
        if _needs_fen_setup(game, tag_values):
            tag_values["SetUp"] = "1"
            tag_values["FEN"] = game.to_fen() if not game.moves else initial_game.to_fen()
            initial_game = Game.from_fen(tag_values["FEN"])
    pgn_result = result if result is not None else _result_from_game(game)
    return format_pgn(
        PgnGame(tags=tag_values, moves=game.moves, result=pgn_result, initial_game=initial_game)
    )


def _split_tags_and_movetext(text: str) -> tuple[dict[str, str], str]:
    tags: dict[str, str] = {}
    movetext_lines: list[str] = []
    in_movetext = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if tags:
                in_movetext = True
            continue
        if not in_movetext and line.startswith("["):
            match = _TAG_RE.match(line)
            if match is None:
                msg = f"malformed PGN tag pair: {line!r}"
                raise ValueError(msg)
            name, value = match.groups()
            tags[name] = _unescape_tag_value(value)
            continue
        in_movetext = True
        movetext_lines.append(line)
    return tags, " ".join(movetext_lines)


def _movetext_tokens(movetext: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for raw_token in movetext.split():
        token = raw_token.strip()
        while True:
            match = _MOVE_NUMBER_PREFIX_RE.match(token)
            if match is None:
                break
            prefix = match.group(0)
            if not prefix.endswith("."):
                break
            token = token[len(prefix) :]
            if not token:
                break
        if not token or _MOVE_NUMBER_RE.match(token):
            continue
        tokens.append(token)
    return tuple(tokens)


def _needs_fen_setup(game: Game, tags: Mapping[str, str]) -> bool:
    if "SetUp" in tags or "FEN" in tags:
        return False
    if game.positions[0] != Board.starting_position():
        return True
    return not game.moves and (game.halfmove_clock != 0 or game.fullmove_number != 1)


def _initial_game_from_tags(tags: Mapping[str, str]) -> Game:
    setup = tags.get("SetUp")
    fen = tags.get("FEN")
    if setup is not None and setup not in {"0", "1"}:
        msg = f"unsupported PGN SetUp tag value: {setup!r}"
        raise ValueError(msg)
    if setup == "1":
        if fen is None:
            msg = 'PGN SetUp "1" requires a FEN tag'
            raise ValueError(msg)
        return Game.from_fen(fen)
    if fen is not None:
        msg = 'PGN FEN tag requires SetUp "1"'
        raise ValueError(msg)
    return Game.new()


def _ordered_tags(tags: Mapping[str, str], result: str) -> dict[str, str]:
    ordered: dict[str, str] = {}
    defaults = {
        "Event": "?",
        "Site": "?",
        "Date": "????.??.??",
        "Round": "?",
        "White": "?",
        "Black": "?",
        "Result": result,
    }
    for name in COMMON_TAGS:
        ordered[name] = tags.get(name, defaults[name])
    for name, value in tags.items():
        if name not in ordered:
            ordered[name] = value
    ordered["Result"] = result
    return ordered


def _format_movetext(initial_game: Game, moves: tuple[Move, ...], result: str) -> str:
    current = initial_game
    tokens: list[str] = []
    for move in moves:
        if current.board.side_to_move is Color.WHITE:
            tokens.append(f"{current.fullmove_number}.")
        elif not tokens:
            tokens.append(f"{current.fullmove_number}...")
        tokens.append(move_to_san(current.board, move))
        current = current.play(move)
    tokens.append(result)
    return " ".join(tokens)


def _parse_normalized_san_token(token: str) -> _ParsedSan | None:
    castle_match = _CASTLING_SAN_RE.match(token)
    if castle_match is not None:
        castle, suffix = castle_match.groups()
        return _ParsedSan(
            piece=PieceType.KING,
            target=None,
            is_capture=False,
            promotion=None,
            suffix=suffix,
            castling_side="kingside" if castle == "O-O" else "queenside",
        )

    match = _SAN_TOKEN_RE.match(token)
    if match is None:
        return None

    piece = _SAN_PIECE_TO_PIECE.get(match.group("piece") or "", PieceType.PAWN)
    disambiguation = match.group("disambiguation")
    constraints = _san_disambiguation_constraints(disambiguation)
    if constraints is None:
        return None
    source_file, source_rank = constraints
    promotion_token = match.group("promotion")
    promotion = None
    if promotion_token is not None:
        promotion = _SAN_PROMOTION_TO_PIECE[promotion_token[1]]
    return _ParsedSan(
        piece=piece,
        target=parse_square(match.group("target")),
        is_capture=bool(match.group("capture")),
        promotion=promotion,
        suffix=match.group("suffix"),
        disambiguation=disambiguation,
        source_file=source_file,
        source_rank=source_rank,
    )


def _san_disambiguation_constraints(disambiguation: str) -> tuple[int | None, int | None] | None:
    source_file: int | None = None
    source_rank: int | None = None
    for char in disambiguation:
        if char in FILES:
            if source_file is not None:
                return None
            source_file = FILES.index(char)
            continue
        if char in RANKS:
            if source_rank is not None:
                return None
            source_rank = RANKS.index(char)
            continue
        return None
    return source_file, source_rank


def _san_exact_candidates(
    board: Board, parsed: _ParsedSan, legal: tuple[Move, ...]
) -> list[Move]:
    """Return legal moves matching SAN shape and canonical disambiguation.

    This is intentionally cheaper than formatting candidate moves as SAN: it does
    not apply moves and leaves check/mate suffix validation to the final stage.
    """
    shape_candidates = [
        move for move in legal if _san_move_shape_matches(board, move, parsed)
    ]
    if parsed.piece is PieceType.PAWN or parsed.castling_side is not None:
        return shape_candidates
    return [
        move
        for move in shape_candidates
        if parsed.disambiguation == _required_san_disambiguation(board, move, legal)
    ]


def _san_move_shape_matches(board: Board, move: Move, parsed: _ParsedSan) -> bool:
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None or moving_piece.color is not board.side_to_move:
        return False
    if moving_piece.kind is not parsed.piece:
        return False
    if parsed.castling_side is not None:
        if not _is_castling_move(move, moving_piece.kind):
            return False
        is_kingside = int(move.to_square) > int(move.from_square)
        return (parsed.castling_side == "kingside") == is_kingside
    if _is_castling_move(move, moving_piece.kind):
        return False
    if parsed.target is None or move.to_square != parsed.target:
        return False
    if move.promotion is not parsed.promotion:
        return False

    is_capture = board.piece_at(move.to_square) is not None or _is_en_passant_capture(board, move)
    if is_capture != parsed.is_capture:
        return False

    if parsed.source_file is not None and file_index(move.from_square) != parsed.source_file:
        return False
    if parsed.source_rank is not None and rank_index(move.from_square) != parsed.source_rank:
        return False

    if moving_piece.kind is PieceType.PAWN:
        if parsed.is_capture:
            return parsed.source_file is not None and parsed.source_rank is None
        return not parsed.disambiguation
    return True


def _required_san_disambiguation(board: Board, move: Move, legal: tuple[Move, ...]) -> str:
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None or moving_piece.kind is PieceType.PAWN:
        return ""
    return _disambiguation(board, move, moving_piece.kind, legal)


def _is_castling_move(move: Move, kind: PieceType) -> bool:
    return kind is PieceType.KING and abs(int(move.to_square) - int(move.from_square)) == 2


def _san_suffix_matches(board: Board, move: Move, suffix: str | None) -> bool:
    next_board = board.apply_move(move)
    if not is_in_check(next_board, next_board.side_to_move):
        return suffix is None
    expected = "#" if not legal_moves(next_board) else "+"
    return suffix == expected


def _disambiguation(
    board: Board, move: Move, kind: PieceType, legal: tuple[Move, ...]
) -> str:
    same_destination = []
    moving_piece = board.piece_at(move.from_square)
    for other in legal:
        if other == move or other.to_square != move.to_square:
            continue
        other_piece = board.piece_at(other.from_square)
        if other_piece == moving_piece and other_piece is not None and other_piece.kind is kind:
            same_destination.append(other)
    if not same_destination:
        return ""
    from_file = file_index(move.from_square)
    from_rank = rank_index(move.from_square)
    has_same_file = any(file_index(other.from_square) == from_file for other in same_destination)
    has_same_rank = any(rank_index(other.from_square) == from_rank for other in same_destination)
    if not has_same_file:
        return chr(ord("a") + from_file)
    if not has_same_rank:
        return str(from_rank + 1)
    return f"{chr(ord('a') + from_file)}{from_rank + 1}"


def _is_en_passant_capture(board: Board, move: Move) -> bool:
    moving_piece = board.piece_at(move.from_square)
    return (
        moving_piece is not None
        and moving_piece.kind is PieceType.PAWN
        and board.en_passant_target == move.to_square
        and board.piece_at(move.to_square) is None
        and abs(int(move.to_square) - int(move.from_square)) in {7, 9}
    )


def _square_name_from_move_target(move: Move) -> str:
    from tinychess.engine.square import square_name

    return square_name(move.to_square)


def _normalize_san_token(token: str) -> str:
    normalized = token.strip()
    if normalized.endswith("e.p."):
        msg = "en-passant annotation is unsupported in SAN"
        raise ValueError(msg)
    if normalized.endswith(("!", "?")):
        msg = f"SAN annotation suffixes are unsupported: {token!r}"
        raise ValueError(msg)
    normalized = normalized.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    suffix_match = _CHECK_SUFFIX_RE.search(normalized)
    if suffix_match and len(suffix_match.group(0)) > 1:
        msg = f"invalid SAN check/mate suffix: {token!r}"
        raise ValueError(msg)
    if "=" in normalized:
        promotion = normalized.rsplit("=", 1)[1][:1]
        if promotion not in _SAN_PROMOTION_TO_PIECE:
            msg = f"unsupported SAN promotion piece: {token!r}"
            raise ValueError(msg)
    return normalized


def _reject_unsupported_movetext(text: str) -> None:
    if "%clk" in text or "%emt" in text:
        msg = "PGN clock annotations are unsupported"
        raise ValueError(msg)
    if "e.p." in text:
        msg = "en-passant annotation is unsupported in SAN"
        raise ValueError(msg)
    if "{" in text or "}" in text or ";" in text:
        msg = "PGN comments are unsupported"
        raise ValueError(msg)
    if "(" in text or ")" in text:
        msg = "PGN variations are unsupported"
        raise ValueError(msg)
    if "$" in text:
        msg = "PGN numeric annotation glyphs are unsupported"
        raise ValueError(msg)


def _escape_tag_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _unescape_tag_value(value: str) -> str:
    result: list[str] = []
    escaping = False
    for char in value:
        if escaping:
            result.append(char)
            escaping = False
        elif char == "\\":
            escaping = True
        else:
            result.append(char)
    if escaping:
        result.append("\\")
    return "".join(result)


def _result_from_game(game: Game) -> str:
    outcome = game.outcome
    if outcome is None:
        return "*"
    if outcome.winner is Color.WHITE:
        return "1-0"
    if outcome.winner is Color.BLACK:
        return "0-1"
    return "1/2-1/2"
