#!/usr/bin/env python3
"""Run the tinychess Python benchmark suite and print a report."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinychess.benchmarks import format_report, run_benchmark_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the tinychess full benchmark suite.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run very small benchmark settings suitable for validation/CI smoke checks",
    )
    parser.add_argument(
        "--no-batched",
        action="store_true",
        help="skip the optional batched MLX inference benchmark",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="report output format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path to write the report; stdout is always used when omitted",
    )
    args = parser.parse_args()

    report = run_benchmark_suite(smoke=args.smoke, include_batched=not args.no_batched)
    rendered = format_report(report, output_format=args.format)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"wrote benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
