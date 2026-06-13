"""Simple terminal play loop for vibechess."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

from vibechess.ai.mcts import MCTSPlayer
from vibechess.ai.player import Player, RandomPlayer
from vibechess.ai.search_config import MCTSConfig
from vibechess.engine.game import Game
from vibechess.engine.move import Move
from vibechess.ui.render import render_game

PlayerKind = Literal["human", "random", "mcts", "ai"]


class HumanQuit(Exception):
    """Raised when a human player intentionally exits the terminal game."""


@dataclass(frozen=True, slots=True)
class PlayConfig:
    """Configuration for the terminal play loop."""

    white: PlayerKind = "human"
    black: PlayerKind = "human"
    max_plies: int = 512
    seed: int | None = None
    mcts_simulations: int = 25
    mcts_rollout_plies: int = 0
    ai_checkpoint: Path | None = None
    ai_simulations: int = 25
    ai_node_budget: int | None = None
    ai_time_limit_seconds: float | None = None
    ai_temperature: float = 0.0
    ai_puct_exploration: float = 1.5
    unicode: bool = False
    coordinates: bool = True


def _create_ai_players(config: PlayConfig, players: dict[str, PlayerKind]) -> dict[str, Player]:
    ai_sides = [side for side, player in players.items() if player == "ai"]
    if not ai_sides:
        return {}
    if config.ai_checkpoint is None:
        msg = "ai player requires --ai-checkpoint"
        raise ValueError(msg)

    from vibechess.ai.neural_mcts import NeuralMCTSConfig, NeuralMCTSPlayer
    from vibechess.nn.checkpoint import load_checkpoint
    from vibechess.nn.inference import PolicyValueInference

    try:
        checkpoint = load_checkpoint(config.ai_checkpoint)
    except OSError as exc:
        msg = f"failed to load ai checkpoint {config.ai_checkpoint}: {exc}"
        raise ValueError(msg) from exc
    inference = PolicyValueInference(checkpoint.model)

    def new_player() -> Player:
        return NeuralMCTSPlayer(
            inference,
            NeuralMCTSConfig(
                simulations=config.ai_simulations,
                time_limit_seconds=config.ai_time_limit_seconds,
                node_budget=config.ai_node_budget,
                puct_exploration=config.ai_puct_exploration,
                temperature=config.ai_temperature,
                seed=config.seed,
            ),
        )

    return {side: new_player() for side in ai_sides}


def parse_legal_uci_move(text: str, game: Game) -> Move:
    """Parse a UCI move string and require it to be legal in ``game``."""
    notation = text.strip().lower()
    if not notation:
        msg = "empty move; enter a UCI move such as e2e4"
        raise ValueError(msg)
    try:
        move = Move.from_uci(notation)
    except ValueError as exc:
        msg = f"invalid UCI move {text.strip()!r}: {exc}"
        raise ValueError(msg) from exc
    legal = game.legal_moves
    if move not in legal:
        legal_examples = ", ".join(candidate.to_uci() for candidate in legal[:8])
        suffix = f" Examples: {legal_examples}" if legal_examples else ""
        msg = f"illegal move {notation!r} for the current position.{suffix}"
        raise ValueError(msg)
    return move


def play_terminal(
    config: PlayConfig,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> Game:
    """Run a human/random terminal game and return the final game state."""
    if config.max_plies < 0:
        msg = f"max_plies must be non-negative, got {config.max_plies}"
        raise ValueError(msg)
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    random_player = RandomPlayer(seed=config.seed)
    mcts_player = MCTSPlayer(
        MCTSConfig(
            simulations=config.mcts_simulations,
            max_rollout_plies=config.mcts_rollout_plies,
            seed=config.seed,
        )
    )
    players = {"white": config.white, "black": config.black}
    ai_players = _create_ai_players(config, players)
    game = Game.new()
    last_move: Move | None = None
    plies_played = 0

    while game.outcome is None and plies_played < config.max_plies:
        print(
            render_game(
                game,
                last_move=last_move,
                unicode=config.unicode,
                coordinates=config.coordinates,
            ),
            file=output_stream,
        )
        side = game.board.side_to_move.value
        player = players[side]
        legal = game.legal_moves
        if not legal:
            break
        if player == "random":
            move = random_player.select_move(game)
            print(f"{side} random plays {move.to_uci()}", file=output_stream)
        elif player == "mcts":
            move = mcts_player.select_move(game)
            print(f"{side} mcts plays {move.to_uci()}", file=output_stream)
        elif player == "ai":
            move = ai_players[side].select_move(game)
            print(f"{side} ai plays {move.to_uci()}", file=output_stream)
        elif player == "human":
            move = _read_human_move(
                game=game,
                input_stream=input_stream,
                output_stream=output_stream,
            )
        else:  # defensive for direct API callers; argparse also constrains values.
            msg = f"unsupported player kind: {player!r}"
            raise ValueError(msg)
        game = game.play(move)
        last_move = move
        plies_played += 1
        print("", file=output_stream)

    print(
        render_game(
            game,
            last_move=last_move,
            unicode=config.unicode,
            coordinates=config.coordinates,
        ),
        file=output_stream,
    )
    if game.outcome is None and plies_played >= config.max_plies:
        print(
            f"Stopped after max plies ({config.max_plies}) without an outcome.",
            file=output_stream,
        )
    return game


def _read_human_move(*, game: Game, input_stream: TextIO, output_stream: TextIO) -> Move:
    side = game.board.side_to_move.value
    while True:
        print(f"{side} move (UCI, e.g. e2e4; 'quit' to exit): ", end="", file=output_stream)
        output_stream.flush()
        line = input_stream.readline()
        if line == "":
            raise EOFError("input ended while waiting for a human move")
        command = line.strip()
        if command.lower() in {"quit", "exit"}:
            raise HumanQuit("human player quit")
        try:
            return parse_legal_uci_move(command, game)
        except ValueError as exc:
            print(f"Invalid move: {exc}", file=output_stream)
