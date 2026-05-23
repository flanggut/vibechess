# tinychess

A small chess engine and neural-MCTS AI project for Apple Silicon macOS.

## Status

Planning is complete and implementation has started with the Python project bootstrap. The chess engine itself will be implemented in later work packages.

## Requirements

- Apple Silicon macOS
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Development setup

```bash
uv sync --dev
```

## CI-ready checks

```bash
uv run pytest
uv run ruff check .
uv run mypy src
uv run tinychess --help
```

## Current CLI

```bash
uv run tinychess --help
uv run tinychess --version
```
