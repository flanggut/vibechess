"""JSON-lines protocol support for the native macOS GUI frontend."""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TextIO, cast

from tinychess import __version__
from tinychess.ai.mcts import MCTSPlayer
from tinychess.ai.player import NoLegalMoveError, RandomPlayer
from tinychess.ai.search_config import MCTSConfig
from tinychess.engine.game import Game
from tinychess.engine.move import Move
from tinychess.engine.piece import Color, PieceType
from tinychess.engine.square import square_name

PROTOCOL_VERSION = "tinychess-gui-v1"

PlayerKind = Literal["random", "mcts", "neural"]
PLAYER_KINDS: frozenset[str] = frozenset(("random", "mcts", "neural"))


class SearchResult(Protocol):
    """Common search metadata exposed by classical and neural MCTS results."""

    @property
    def move(self) -> Move:
        """Return the selected move."""
        ...

    @property
    def simulations(self) -> int:
        """Return the number of completed simulations."""
        ...

    @property
    def nodes(self) -> int:
        """Return the number of materialized/search nodes."""
        ...

    @property
    def elapsed_seconds(self) -> float:
        """Return search wall-clock time in seconds."""
        ...

    @property
    def visit_counts(self) -> dict[Move, int]:
        """Return root visit counts keyed by move."""
        ...

ERROR_CODES: frozenset[str] = frozenset(
    (
        "invalid_json",
        "invalid_request",
        "unknown_command",
        "invalid_move",
        "illegal_move",
        "terminal_position",
        "configuration_error",
        "checkpoint_error",
        "internal_error",
    )
)


@dataclass(frozen=True, slots=True)
class GuiAiConfig:
    """Validated AI settings carried by the GUI session.

    These settings can be stored on the session or supplied per ``aiMove`` request.
    """

    kind: PlayerKind = "random"
    simulations: int = 25
    time_limit_seconds: float | None = None
    node_budget: int | None = None
    max_rollout_plies: int = 0
    checkpoint_path: Path | None = None
    puct_exploration: float = 1.5
    temperature: float = 0.0
    leaf_parallelism: int = 1
    seed: int | None = None

    def to_response(self) -> dict[str, object]:
        """Return a JSON-serializable camelCase representation."""
        return {
            "kind": self.kind,
            "simulations": self.simulations,
            "timeLimitSeconds": self.time_limit_seconds,
            "nodeBudget": self.node_budget,
            "maxRolloutPlies": self.max_rollout_plies,
            "checkpointPath": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "puctExploration": self.puct_exploration,
            "temperature": self.temperature,
            "leafParallelism": self.leaf_parallelism,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class GuiConfig:
    """Configuration for the GUI JSON-lines session."""

    seed: int | None = None
    default_ai: GuiAiConfig | None = None


class GuiProtocolError(ValueError):
    """Structured, recoverable GUI protocol error."""

    def __init__(self, code: str, message: str) -> None:
        if code not in ERROR_CODES:
            msg = f"unsupported GUI protocol error code: {code!r}"
            raise ValueError(msg)
        super().__init__(message)
        self.code = code
        self.message = message


class GuiSession:
    """Stateful JSON-lines command handler for the native GUI frontend."""

    def __init__(self, config: GuiConfig | None = None) -> None:
        self.config = GuiConfig() if config is None else config
        default_ai = self.config.default_ai or GuiAiConfig(seed=self.config.seed)
        self.game = Game.new()
        self._initial_game = self.game
        self.human_color = Color.WHITE
        self.ai_config = default_ai
        self._quit = False

    @property
    def should_quit(self) -> bool:
        """Return whether the session has received ``quit``."""
        return self._quit

    def handle_line(self, line: str, output: TextIO) -> None:
        """Handle one JSON-lines request and write one JSON response."""
        stripped = line.strip()
        if not stripped:
            return
        try:
            raw: object = json.loads(stripped)
        except json.JSONDecodeError as exc:
            response = self._error_response(
                None,
                "invalid_json",
                f"invalid JSON: {exc.msg}",
                include_state=False,
            )
        else:
            if not isinstance(raw, dict):
                response = self._error_response(
                    None,
                    "invalid_request",
                    "request must be a JSON object",
                    include_state=False,
                )
            else:
                response = self.handle_request(cast(Mapping[str, object], raw))
        print(json.dumps(response, separators=(",", ":")), file=output)
        output.flush()

    def handle_request(self, request: Mapping[str, object]) -> dict[str, object]:
        """Handle one decoded GUI protocol request and return a response object."""
        request_id: object = None
        try:
            request_id = _request_id(request)
            command = _required_str(request, "cmd")
            if command == "hello":
                return self._handle_hello(request_id)
            if command == "state":
                return self._success_response(request_id, state=serialize_state(self.game))
            if command == "newGame":
                return self._handle_new_game(request_id, request)
            if command == "makeMove":
                return self._handle_make_move(request_id, request)
            if command == "aiMove":
                return self._handle_ai_move(request_id, request)
            if command == "undo":
                return self._handle_undo(request_id, request)
            if command == "setAiConfig":
                return self._handle_set_ai_config(request_id, request)
            if command == "quit":
                self._quit = True
                return self._success_response(request_id, state=serialize_state(self.game))
            raise GuiProtocolError("unknown_command", f"unknown command: {command}")
        except GuiProtocolError as exc:
            return self._error_response(request_id, exc.code, exc.message)
        except Exception as exc:  # pragma: no cover - defensive protocol containment.
            return self._error_response(request_id, "internal_error", f"internal error: {exc}")

    def _handle_hello(self, request_id: object) -> dict[str, object]:
        capabilities: dict[str, object] = {
            "players": ["random", "mcts", "neural"],
            "supportsUndo": True,
            "promotion": "auto_queen",
        }
        return self._success_response(
            request_id,
            version=__version__,
            protocol=PROTOCOL_VERSION,
            capabilities=capabilities,
            state=serialize_state(self.game),
        )

    def _handle_new_game(
        self,
        request_id: object,
        request: Mapping[str, object],
    ) -> dict[str, object]:
        human_color = self.human_color
        ai_config = self.ai_config
        if "humanColor" in request:
            human_color = _parse_color(request["humanColor"], field_name="humanColor")
        if "seed" in request:
            seed = _optional_int(request, "seed", self.config.seed, minimum=None)
            ai_config = GuiAiConfig(
                kind=ai_config.kind,
                simulations=ai_config.simulations,
                time_limit_seconds=ai_config.time_limit_seconds,
                node_budget=ai_config.node_budget,
                max_rollout_plies=ai_config.max_rollout_plies,
                checkpoint_path=ai_config.checkpoint_path,
                puct_exploration=ai_config.puct_exploration,
                temperature=ai_config.temperature,
                leaf_parallelism=ai_config.leaf_parallelism,
                seed=seed,
            )
        if "ai" in request:
            ai_config = parse_ai_config(request["ai"], current=ai_config)
        self.human_color = human_color
        self.ai_config = ai_config
        self.game = Game.new()
        self._initial_game = self.game
        return self._success_response(request_id, state=serialize_state(self.game))

    def _handle_make_move(
        self,
        request_id: object,
        request: Mapping[str, object],
    ) -> dict[str, object]:
        notation = _required_str(request, "move").lower()
        legal = self.game.legal_moves
        if self.game.outcome is not None or not legal:
            raise GuiProtocolError(
                "terminal_position",
                "cannot make a move from a terminal position",
            )
        try:
            move = Move.from_uci(notation)
        except ValueError as exc:
            raise GuiProtocolError("invalid_move", f"invalid move {notation!r}: {exc}") from exc
        if move not in legal:
            auto_move = _auto_queen_promotion(notation, legal)
            if auto_move is None:
                raise GuiProtocolError(
                    "illegal_move",
                    f"illegal move {notation!r} for the current position",
                )
            move = auto_move
        self.game = self.game.play_known_legal(move)
        return self._success_response(
            request_id,
            appliedMove=move.to_uci(),
            state=serialize_state(self.game),
        )

    def _handle_ai_move(
        self,
        request_id: object,
        request: Mapping[str, object],
    ) -> dict[str, object]:
        ai_config = self.ai_config
        if "ai" in request:
            ai_config = parse_ai_config(request["ai"], current=ai_config)
        legal = self.game.legal_moves
        if self.game.outcome is not None or not legal:
            raise GuiProtocolError(
                "terminal_position",
                "cannot select an AI move from a terminal position",
            )

        try:
            move, search = _select_ai_move(self.game, ai_config)
        except NoLegalMoveError as exc:
            raise GuiProtocolError("terminal_position", str(exc)) from exc
        if move not in legal:
            raise GuiProtocolError("internal_error", f"AI selected illegal move: {move.to_uci()}")
        self.game = self.game.play_known_legal(move)
        return self._success_response(
            request_id,
            appliedMove=move.to_uci(),
            search=search,
            state=serialize_state(self.game),
        )

    def _handle_undo(
        self,
        request_id: object,
        request: Mapping[str, object],
    ) -> dict[str, object]:
        plies = _optional_request_int(request, "plies", 2, minimum=0)
        target_move_count = max(0, len(self.game.moves) - plies)
        self.game = _replay_moves(self._initial_game, self.game.moves[:target_move_count])
        return self._success_response(request_id, state=serialize_state(self.game))

    def _handle_set_ai_config(
        self,
        request_id: object,
        request: Mapping[str, object],
    ) -> dict[str, object]:
        if "ai" not in request:
            raise GuiProtocolError("invalid_request", "setAiConfig requires an 'ai' object")
        self.ai_config = parse_ai_config(request["ai"], current=self.ai_config)
        return self._success_response(
            request_id,
            ai=self.ai_config.to_response(),
            state=serialize_state(self.game),
        )

    def _success_response(self, request_id: object, **fields: object) -> dict[str, object]:
        response: dict[str, object] = {"id": request_id, "ok": True}
        response.update(fields)
        return response

    def _error_response(
        self,
        request_id: object,
        code: str,
        message: str,
        *,
        include_state: bool = True,
    ) -> dict[str, object]:
        response: dict[str, object] = {
            "id": request_id,
            "ok": False,
            "error": {"code": code, "message": message},
        }
        if include_state:
            response["state"] = serialize_state(self.game)
        return response


def run_gui_loop(
    config: GuiConfig | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> GuiSession:
    """Run the GUI JSON-lines command loop and return the final session state."""
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    session = GuiSession(config)
    for line in input_stream:
        session.handle_line(line, output_stream)
        if session.should_quit:
            break
    return session


def _select_ai_move(game: Game, config: GuiAiConfig) -> tuple[Move, dict[str, object]]:
    if config.kind == "random":
        start = time.perf_counter()
        move = RandomPlayer(seed=config.seed).select_move(game)
        return move, {"kind": "random", "elapsedSeconds": time.perf_counter() - start}
    if config.kind == "mcts":
        player = MCTSPlayer(
            MCTSConfig(
                simulations=config.simulations,
                time_limit_seconds=config.time_limit_seconds,
                node_budget=config.node_budget,
                max_rollout_plies=config.max_rollout_plies,
                seed=config.seed,
            )
        )
        result = player.search(game)
        return result.move, _serialize_mcts_search("mcts", result)
    return _select_neural_ai_move(game, config)


def _select_neural_ai_move(game: Game, config: GuiAiConfig) -> tuple[Move, dict[str, object]]:
    if config.checkpoint_path is None:
        raise GuiProtocolError("configuration_error", "neural aiMove requires checkpointPath")
    try:
        from tinychess.ai.neural_mcts import NeuralMCTSConfig, NeuralMCTSPlayer
        from tinychess.nn.checkpoint import load_checkpoint
        from tinychess.nn.model import PolicyValueInference
    except ImportError as exc:
        message = f"neural checkpoint loading unavailable: {exc}"
        raise GuiProtocolError("checkpoint_error", message) from exc
    try:
        neural_config = NeuralMCTSConfig(
            simulations=config.simulations,
            time_limit_seconds=config.time_limit_seconds,
            node_budget=config.node_budget,
            puct_exploration=config.puct_exploration,
            temperature=config.temperature,
            seed=config.seed,
            leaf_parallelism=config.leaf_parallelism,
        )
        checkpoint = load_checkpoint(config.checkpoint_path)
    except OSError as exc:
        raise GuiProtocolError(
            "checkpoint_error",
            f"failed to load neural checkpoint {config.checkpoint_path}: {exc}",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise GuiProtocolError("checkpoint_error", f"invalid neural checkpoint: {exc}") from exc
    player = NeuralMCTSPlayer(PolicyValueInference(checkpoint.model), neural_config)
    result = player.search(game)
    return result.move, _serialize_mcts_search("neural", result)


def _serialize_mcts_search(kind: str, result: SearchResult) -> dict[str, object]:
    return {
        "kind": kind,
        "simulations": result.simulations,
        "nodes": result.nodes,
        "elapsedSeconds": result.elapsed_seconds,
        "visitCounts": {move.to_uci(): visits for move, visits in result.visit_counts.items()},
    }


def _replay_moves(initial_game: Game, moves: tuple[Move, ...]) -> Game:
    game = initial_game
    for move in moves:
        if move not in game.legal_moves:
            raise GuiProtocolError(
                "internal_error",
                f"cannot replay move while undoing: {move.to_uci()}",
            )
        game = game.play_known_legal(move)
    return game


def serialize_state(game: Game) -> dict[str, object]:
    """Return the canonical JSON-serializable GUI state for ``game``."""
    legal_moves = game.legal_moves
    legal_destinations = _legal_destinations_by_from(legal_moves)
    moves = [move.to_uci() for move in game.moves]
    last_move = None if not moves else moves[-1]
    return {
        "fen": game.to_fen(),
        "sideToMove": game.board.side_to_move.value,
        "squares": _serialize_squares(game),
        "legalMoves": [move.to_uci() for move in legal_moves],
        "legalDestinationsByFrom": legal_destinations,
        "moves": moves,
        "lastMove": last_move,
        "halfmoveClock": game.halfmove_clock,
        "fullmoveNumber": game.fullmove_number,
        "outcome": _serialize_outcome(game),
    }


def parse_ai_config(value: object, *, current: GuiAiConfig | None = None) -> GuiAiConfig:
    """Parse and validate an AI config object from a GUI request."""
    base = current or GuiAiConfig()
    if value is None:
        return base
    if not isinstance(value, dict):
        raise GuiProtocolError("configuration_error", "ai must be an object")
    data = cast(Mapping[str, object], value)
    kind_text = _optional_str(data, "kind", base.kind)
    if kind_text not in PLAYER_KINDS:
        raise GuiProtocolError("configuration_error", f"unsupported ai kind: {kind_text!r}")
    kind = cast(PlayerKind, kind_text)
    simulations = cast(int, _optional_int(data, "simulations", base.simulations, minimum=1))
    time_limit_seconds = _optional_float(
        data,
        "timeLimitSeconds",
        base.time_limit_seconds,
        minimum=0.0,
    )
    node_budget = _optional_int(data, "nodeBudget", base.node_budget, minimum=1)
    max_rollout_plies = cast(
        int,
        _optional_int(
            data,
            "maxRolloutPlies",
            base.max_rollout_plies,
            minimum=0,
        ),
    )
    puct_exploration = cast(
        float,
        _optional_float(
            data,
            "puctExploration",
            base.puct_exploration,
            minimum=0.0,
        ),
    )
    temperature = cast(float, _optional_float(data, "temperature", base.temperature, minimum=0.0))
    leaf_parallelism = cast(
        int,
        _optional_int(data, "leafParallelism", base.leaf_parallelism, minimum=1),
    )
    seed = _optional_int(data, "seed", base.seed, minimum=None)
    checkpoint_path = _optional_path(data, "checkpointPath", base.checkpoint_path)

    if kind == "mcts":
        try:
            MCTSConfig(
                simulations=simulations,
                time_limit_seconds=time_limit_seconds,
                node_budget=node_budget,
                max_rollout_plies=max_rollout_plies,
                seed=seed,
            )
        except ValueError as exc:
            raise GuiProtocolError("configuration_error", str(exc)) from exc

    return GuiAiConfig(
        kind=kind,
        simulations=simulations,
        time_limit_seconds=time_limit_seconds,
        node_budget=node_budget,
        max_rollout_plies=max_rollout_plies,
        checkpoint_path=checkpoint_path,
        puct_exploration=puct_exploration,
        temperature=temperature,
        leaf_parallelism=leaf_parallelism,
        seed=seed,
    )


def _serialize_squares(game: Game) -> list[dict[str, object]]:
    squares: list[dict[str, object]] = []
    for square, piece in game.board.occupied_squares():
        squares.append(
            {
                "square": square_name(square),
                "index": int(square),
                "piece": piece.symbol,
                "color": piece.color.value,
                "kind": piece.kind.value,
            }
        )
    return squares


def _legal_destinations_by_from(moves: tuple[Move, ...]) -> dict[str, list[str]]:
    destinations: dict[str, list[str]] = {}
    for move in moves:
        from_square = square_name(move.from_square)
        to_square = square_name(move.to_square)
        existing = destinations.setdefault(from_square, [])
        if to_square not in existing:
            existing.append(to_square)
    return destinations


def _serialize_outcome(game: Game) -> dict[str, object] | None:
    outcome = game.outcome
    if outcome is None:
        return None
    return {
        "reason": outcome.reason.value,
        "winner": None if outcome.winner is None else outcome.winner.value,
        "isDraw": outcome.is_draw,
    }


def _auto_queen_promotion(notation: str, legal_moves: tuple[Move, ...]) -> Move | None:
    if len(notation) != 4:
        return None
    queen_notation = f"{notation}q"
    try:
        queen_move = Move.from_uci(queen_notation)
    except ValueError:
        return None
    if queen_move.promotion is not PieceType.QUEEN:
        return None
    return queen_move if queen_move in legal_moves else None


def _request_id(request: Mapping[str, object]) -> object:
    if "id" not in request:
        raise GuiProtocolError("invalid_request", "request requires an 'id' field")
    request_id = request["id"]
    if isinstance(request_id, dict | list):
        raise GuiProtocolError("invalid_request", "request id must be a scalar JSON value")
    return request_id


def _required_str(data: Mapping[str, object], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str):
        raise GuiProtocolError("invalid_request", f"request field {field_name!r} must be a string")
    return value


def _optional_request_int(
    data: Mapping[str, object],
    field_name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    if field_name not in data or data[field_name] is None:
        return default
    value = data[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise GuiProtocolError(
            "invalid_request",
            f"request field {field_name!r} must be an integer",
        )
    if value < minimum:
        raise GuiProtocolError(
            "invalid_request",
            f"request field {field_name!r} must be at least {minimum}, got {value}",
        )
    return value


def _optional_str(data: Mapping[str, object], field_name: str, default: str) -> str:
    if field_name not in data or data[field_name] is None:
        return default
    value = data[field_name]
    if not isinstance(value, str):
        raise GuiProtocolError(
            "configuration_error",
            f"ai field {field_name!r} must be a string",
        )
    return value


def _optional_int(
    data: Mapping[str, object],
    field_name: str,
    default: int | None,
    *,
    minimum: int | None,
) -> int | None:
    if field_name not in data or data[field_name] is None:
        return default
    value = data[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise GuiProtocolError(
            "configuration_error",
            f"ai field {field_name!r} must be an integer",
        )
    if minimum is not None and value < minimum:
        raise GuiProtocolError(
            "configuration_error",
            f"ai field {field_name!r} must be at least {minimum}, got {value}",
        )
    return value


def _optional_float(
    data: Mapping[str, object],
    field_name: str,
    default: float | None,
    *,
    minimum: float,
) -> float | None:
    if field_name not in data or data[field_name] is None:
        return default
    value = data[field_name]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise GuiProtocolError(
            "configuration_error",
            f"ai field {field_name!r} must be a number",
        )
    result = float(value)
    if not math.isfinite(result):
        raise GuiProtocolError(
            "configuration_error",
            f"ai field {field_name!r} must be finite",
        )
    if result < minimum:
        raise GuiProtocolError(
            "configuration_error",
            f"ai field {field_name!r} must be at least {minimum:g}, got {result:g}",
        )
    return result


def _optional_path(
    data: Mapping[str, object],
    field_name: str,
    default: Path | None,
) -> Path | None:
    if field_name not in data or data[field_name] is None:
        return default
    value = data[field_name]
    if not isinstance(value, str):
        raise GuiProtocolError(
            "configuration_error",
            f"ai field {field_name!r} must be a string",
        )
    if not value:
        raise GuiProtocolError("configuration_error", f"ai field {field_name!r} must not be empty")
    return Path(value)


def _parse_color(value: object, *, field_name: str) -> Color:
    if not isinstance(value, str):
        raise GuiProtocolError("invalid_request", f"request field {field_name!r} must be a string")
    try:
        return Color(value)
    except ValueError as exc:
        raise GuiProtocolError(
            "invalid_request",
            f"request field {field_name!r} must be 'white' or 'black'",
        ) from exc
