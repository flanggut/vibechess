#!/usr/bin/env python3
"""Lightweight perft benchmark for the current Python engine."""

from __future__ import annotations

import argparse
import time

from vibechess.engine import Board, perft


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a vibechess perft benchmark from startpos.")
    parser.add_argument("depth", type=int, nargs="?", default=3)
    args = parser.parse_args()
    if args.depth < 0:
        parser.error("depth must be non-negative")

    board = Board.starting_position()
    start = time.perf_counter()
    nodes = perft(board, args.depth)
    elapsed = time.perf_counter() - start
    nodes_per_second = nodes / elapsed if elapsed else float("inf")
    print(f"depth={args.depth} nodes={nodes} elapsed={elapsed:.6f}s nps={nodes_per_second:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
