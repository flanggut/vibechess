"""Command-line entry point for tinychess."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from tinychess import __version__
from tinychess.protocols.gui import GuiAiConfig, GuiConfig, parse_ai_config, run_gui_loop
from tinychess.protocols.uci import UciConfig, run_uci_loop
from tinychess.ui.terminal import HumanQuit, PlayConfig, play_terminal


def build_parser() -> argparse.ArgumentParser:
    """Build the tinychess argument parser."""
    parser = argparse.ArgumentParser(
        prog="tinychess",
        description="tinychess command-line tools.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"tinychess {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    play = subparsers.add_parser(
        "play",
        help="play a terminal chess game",
        description="Play a terminal chess game using UCI moves such as e2e4 or e7e8q.",
    )
    player_choices = ("human", "random", "mcts", "ai")
    play.add_argument("--white", choices=player_choices, default="human")
    play.add_argument("--black", choices=player_choices, default="human")
    play.add_argument("--max-plies", type=int, default=512)
    play.add_argument("--seed", type=int, default=None, help="seed for random/MCTS/AI players")
    play.add_argument("--mcts-simulations", type=int, default=25, help="MCTS simulations per move")
    play.add_argument(
        "--mcts-rollout-plies",
        type=int,
        default=0,
        help="MCTS random rollout plies per simulation; default 0 uses static leaf evaluation",
    )
    play.add_argument(
        "--ai-checkpoint",
        type=Path,
        default=None,
        help="checkpoint directory for ai player kind",
    )
    play.add_argument(
        "--ai-simulations",
        type=int,
        default=25,
        help="neural MCTS simulations per move",
    )
    play.add_argument(
        "--ai-node-budget",
        type=int,
        default=None,
        help="optional neural MCTS node cap",
    )
    play.add_argument(
        "--ai-time-limit-seconds",
        type=float,
        default=None,
        help="optional neural MCTS wall-clock cap per move",
    )
    play.add_argument("--ai-temperature", type=float, default=0.0, help="neural move temperature")
    play.add_argument(
        "--ai-puct-exploration",
        type=float,
        default=1.5,
        help="neural PUCT exploration constant",
    )
    play.add_argument(
        "--ai-leaf-parallelism",
        type=int,
        default=1,
        help="optional approximate neural MCTS leaf parallelism",
    )
    play.add_argument("--unicode", action="store_true", help="render Unicode chess pieces")
    play.add_argument(
        "--no-coordinates",
        action="store_true",
        help="hide board rank/file coordinates",
    )

    uci = subparsers.add_parser(
        "uci",
        help="run the bounded UCI protocol loop",
        description="Run a basic Universal Chess Interface loop with random legal best moves.",
    )
    uci.add_argument("--seed", type=int, default=None, help="seed for deterministic best moves")

    gui_server = subparsers.add_parser(
        "gui-server",
        help="run the JSON-lines protocol backend for the native GUI",
        description=(
            "Run the tinychess GUI backend. The protocol reads one JSON request per "
            "line from stdin and writes one JSON response per line to stdout."
        ),
    )
    gui_server.add_argument(
        "--seed",
        type=int,
        default=None,
        help="default seed for GUI AI players",
    )
    gui_server.add_argument(
        "--ai-kind",
        choices=("random", "mcts", "neural"),
        default="random",
        help="default GUI AI kind",
    )
    gui_server.add_argument(
        "--ai-simulations",
        type=int,
        default=25,
        help="default MCTS/neural simulations per GUI AI move",
    )
    gui_server.add_argument(
        "--ai-node-budget",
        type=int,
        default=None,
        help="optional default GUI AI node cap",
    )
    gui_server.add_argument(
        "--ai-time-limit-seconds",
        type=float,
        default=None,
        help="optional default GUI AI wall-clock cap per move",
    )
    gui_server.add_argument(
        "--ai-checkpoint",
        type=Path,
        default=None,
        help="optional default checkpoint directory for neural GUI AI",
    )
    gui_server.add_argument(
        "--ai-max-rollout-plies",
        type=int,
        default=0,
        help="default classical MCTS random rollout plies; 0 uses static leaf evaluation",
    )
    gui_server.add_argument(
        "--ai-temperature",
        type=float,
        default=0.0,
        help="default neural GUI AI move temperature",
    )
    gui_server.add_argument(
        "--ai-puct-exploration",
        type=float,
        default=1.5,
        help="default neural GUI AI PUCT exploration constant",
    )
    gui_server.add_argument(
        "--ai-leaf-parallelism",
        type=int,
        default=1,
        help="default approximate neural GUI AI leaf parallelism",
    )
    return parser


def _gui_ai_config_from_args(args: argparse.Namespace) -> GuiAiConfig:
    """Return a validated default GUI AI config from parsed CLI arguments."""
    default_ai = GuiAiConfig(
        kind=args.ai_kind,
        simulations=args.ai_simulations,
        time_limit_seconds=args.ai_time_limit_seconds,
        node_budget=args.ai_node_budget,
        max_rollout_plies=args.ai_max_rollout_plies,
        checkpoint_path=args.ai_checkpoint,
        puct_exploration=args.ai_puct_exploration,
        temperature=args.ai_temperature,
        leaf_parallelism=args.ai_leaf_parallelism,
        seed=args.seed,
    )
    return parse_ai_config(default_ai.to_response())


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the tinychess CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr

    if args.command == "play":
        config = PlayConfig(
            white=args.white,
            black=args.black,
            max_plies=args.max_plies,
            seed=args.seed,
            mcts_simulations=args.mcts_simulations,
            mcts_rollout_plies=args.mcts_rollout_plies,
            ai_checkpoint=args.ai_checkpoint,
            ai_simulations=args.ai_simulations,
            ai_node_budget=args.ai_node_budget,
            ai_time_limit_seconds=args.ai_time_limit_seconds,
            ai_temperature=args.ai_temperature,
            ai_puct_exploration=args.ai_puct_exploration,
            ai_leaf_parallelism=args.ai_leaf_parallelism,
            unicode=args.unicode,
            coordinates=not args.no_coordinates,
        )
        try:
            play_terminal(config, stdin=stdin, stdout=output_stream)
        except HumanQuit:
            return 0
        except (EOFError, KeyboardInterrupt) as exc:
            print(f"Game ended: {exc}", file=error_stream)
            return 1
        except ValueError as exc:
            print(f"tinychess play: {exc}", file=error_stream)
            return 2
        return 0

    if args.command == "uci":
        run_uci_loop(UciConfig(seed=args.seed), stdin=stdin, stdout=output_stream)
        return 0

    if args.command == "gui-server":
        try:
            default_ai = _gui_ai_config_from_args(args)
            run_gui_loop(
                GuiConfig(seed=args.seed, default_ai=default_ai),
                stdin=stdin,
                stdout=output_stream,
            )
        except ValueError as exc:
            print(f"tinychess gui-server: {exc}", file=error_stream)
            return 2
        return 0

    parser.print_help(file=output_stream)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
