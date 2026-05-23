"""Command-line entry point for tinychess."""

from __future__ import annotations

import argparse

from tinychess import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the tinychess argument parser."""
    parser = argparse.ArgumentParser(
        prog="tinychess",
        description=(
            "tinychess project CLI. Chess engine commands will be added in later work "
            "packages."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"tinychess {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the tinychess CLI."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
