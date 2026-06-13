#!/usr/bin/env python3
"""Lightweight random complete-game benchmark."""

from __future__ import annotations

import argparse
import time

from vibechess.engine import OutcomeReason, random_move_selector, simulate_game


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a vibechess random-game benchmark.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=512)
    args = parser.parse_args()
    if args.max_plies < 0:
        parser.error("--max-plies must be non-negative")

    start = time.perf_counter()
    game = simulate_game(random_move_selector(args.seed), max_plies=args.max_plies)
    elapsed = time.perf_counter() - start
    outcome = game.outcome
    reason = outcome.reason.value if outcome is not None else OutcomeReason.MAX_PLIES.value
    plies_per_second = len(game.moves) / elapsed if elapsed else float("inf")
    print(
        f"plies={len(game.moves)} reason={reason} "
        f"elapsed={elapsed:.6f}s plies_per_second={plies_per_second:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
