#!/usr/bin/env python3
"""Lightweight MLX policy/value inference latency benchmark."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from tinychess.engine import Game
from tinychess.nn.inference import PolicyValueInference
from tinychess.nn.model import PolicyValueConfig, PolicyValueNet


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tinychess MLX inference benchmark.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument(
        "--unmasked",
        action="store_true",
        help="skip legal-move policy masking to time raw model inference wrapper output",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    config = PolicyValueConfig(
        residual_channels=args.channels,
        residual_blocks=args.blocks,
        value_hidden_dim=args.value_hidden,
    )
    inference = PolicyValueInference(PolicyValueNet(config))
    game = Game.new()
    for _ in range(args.warmup):
        result = inference.predict(game, mask_legal_moves=not args.unmasked)
        mx.eval(result.policy, result.policy_logits)

    start = time.perf_counter()
    for _ in range(args.iterations):
        result = inference.predict(game, mask_legal_moves=not args.unmasked)
        mx.eval(result.policy, result.policy_logits)
    elapsed = time.perf_counter() - start
    average_ms = elapsed / args.iterations * 1000.0
    print(
        " ".join(
            [
                f"iterations={args.iterations}",
                f"warmup={args.warmup}",
                f"channels={args.channels}",
                f"blocks={args.blocks}",
                f"masked={not args.unmasked}",
                f"elapsed={elapsed:.6f}s",
                f"avg_latency_ms={average_ms:.3f}",
                f"inferences_per_sec={args.iterations / elapsed:.1f}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
