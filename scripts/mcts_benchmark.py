#!/usr/bin/env python3
"""Lightweight classical MCTS simulations/sec benchmark."""

from __future__ import annotations

import argparse

from tinychess.ai.mcts import MCTSPlayer
from tinychess.ai.search_config import MCTSConfig
from tinychess.engine import Game


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tinychess classical MCTS benchmark.")
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rollout-plies", type=int, default=16)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--node-budget", type=int, default=None)
    args = parser.parse_args()

    config = MCTSConfig(
        simulations=args.simulations,
        time_limit_seconds=args.time_limit,
        node_budget=args.node_budget,
        max_rollout_plies=args.rollout_plies,
        seed=args.seed,
    )
    player = MCTSPlayer(config)
    result = player.search(Game.new())
    print(
        " ".join(
            [
                f"bestmove={result.move.to_uci()}",
                f"simulations={result.simulations}",
                f"nodes={result.nodes}",
                f"elapsed={result.elapsed_seconds:.6f}s",
                f"sims_per_sec={result.simulations_per_second:.0f}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
