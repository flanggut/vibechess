"""Command-line entry point for tinychess."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from tinychess import __version__
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
    return parser


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

    parser.print_help(file=output_stream)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
