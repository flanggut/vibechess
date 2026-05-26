"""Benchmark suite helpers for Python engine/search/MLX baselines."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx

from tinychess.ai.mcts import MCTSPlayer
from tinychess.ai.search_config import MCTSConfig
from tinychess.engine import (
    Board,
    Game,
    OutcomeReason,
    legal_moves,
    parse_fen,
    random_move_selector,
    simulate_game,
)
from tinychess.nn import (
    PolicyValueConfig,
    PolicyValueInference,
    PolicyValueNet,
    encode_game,
)

ReportFormat = Literal["markdown", "json"]

START_POSITION_LABEL = "startpos"
KIWIPETE_LABEL = "kiwipete"
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/2P1P3/1p2P3/2N2N2/PP1B1PPP/R2QKB1R w KQkq - 0 1"


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One benchmark result with JSON-serializable metrics."""

    name: str
    metrics: dict[str, float | int | str | bool]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "metrics": self.metrics}


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """A complete benchmark report plus Swift acceleration recommendation."""

    results: tuple[BenchmarkResult, ...]
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "results": [result.to_dict() for result in self.results],
            "recommendation": self.recommendation,
        }


def benchmark_move_generation(*, iterations: int = 200) -> BenchmarkResult:
    """Measure legal move generation throughput over representative positions."""
    _require_at_least("iterations", iterations, 1)
    positions = (
        (START_POSITION_LABEL, Board.starting_position()),
        (KIWIPETE_LABEL, parse_fen(KIWIPETE_FEN).board),
    )
    total_calls = 0
    total_moves = 0
    per_position: list[str] = []
    start = time.perf_counter()
    for label, board in positions:
        position_moves = 0
        for _ in range(iterations):
            moves = legal_moves(board)
            position_moves += len(moves)
            total_calls += 1
        total_moves += position_moves
        per_position.append(f"{label}:{position_moves // iterations}")
    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        name="move_generation",
        metrics={
            "positions": len(positions),
            "iterations_per_position": iterations,
            "calls": total_calls,
            "generated_moves": total_moves,
            "elapsed_seconds": elapsed,
            "calls_per_second": _rate(total_calls, elapsed),
            "moves_per_second": _rate(total_moves, elapsed),
            "position_move_counts": ",".join(per_position),
        },
    )


def benchmark_complete_games(
    *, games: int = 3, max_plies: int = 256, seed: int = 1
) -> BenchmarkResult:
    """Measure complete random-game simulation throughput."""
    _require_at_least("games", games, 1)
    _require_at_least("max_plies", max_plies, 0)
    total_plies = 0
    outcomes: dict[str, int] = {}
    start = time.perf_counter()
    for index in range(games):
        game = simulate_game(random_move_selector(seed + index), max_plies=max_plies)
        total_plies += len(game.moves)
        reason = (
            game.outcome.reason.value
            if game.outcome is not None
            else OutcomeReason.MAX_PLIES.value
        )
        outcomes[reason] = outcomes.get(reason, 0) + 1
    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        name="complete_game_simulation",
        metrics={
            "games": games,
            "max_plies": max_plies,
            "plies": total_plies,
            "elapsed_seconds": elapsed,
            "games_per_second": _rate(games, elapsed),
            "plies_per_second": _rate(total_plies, elapsed),
            "outcomes": json.dumps(outcomes, sort_keys=True),
        },
    )


def benchmark_mcts(
    *,
    simulations: int = 50,
    rollout_plies: int = 16,
    seed: int = 1,
    time_limit_seconds: float | None = None,
    node_budget: int | None = None,
) -> BenchmarkResult:
    """Measure classical MCTS simulations/sec from the starting position."""
    _require_at_least("simulations", simulations, 1)
    _require_at_least("rollout_plies", rollout_plies, 0)
    player = MCTSPlayer(
        MCTSConfig(
            simulations=simulations,
            max_rollout_plies=rollout_plies,
            seed=seed,
            time_limit_seconds=time_limit_seconds,
            node_budget=node_budget,
        )
    )
    result = player.search(Game.new())
    return BenchmarkResult(
        name="mcts_simulations",
        metrics={
            "requested_simulations": simulations,
            "simulations": result.simulations,
            "nodes": result.nodes,
            "rollout_plies": rollout_plies,
            "bestmove": result.move.to_uci(),
            "elapsed_seconds": result.elapsed_seconds,
            "simulations_per_second": result.simulations_per_second,
            "nodes_per_second": _rate(result.nodes, result.elapsed_seconds),
        },
    )


def benchmark_mlx_inference(
    *,
    iterations: int = 50,
    warmup: int = 5,
    channels: int = 32,
    blocks: int = 2,
    value_hidden: int = 64,
    mask_legal_moves: bool = True,
) -> BenchmarkResult:
    """Measure single-position MLX policy/value inference latency."""
    _require_at_least("iterations", iterations, 1)
    _require_at_least("warmup", warmup, 0)
    config = _model_config(channels=channels, blocks=blocks, value_hidden=value_hidden)
    inference = PolicyValueInference(PolicyValueNet(config))
    game = Game.new()
    for _ in range(warmup):
        result = inference.predict(game, mask_legal_moves=mask_legal_moves)
        mx.eval(result.policy, result.policy_logits)

    start = time.perf_counter()
    for _ in range(iterations):
        result = inference.predict(game, mask_legal_moves=mask_legal_moves)
        mx.eval(result.policy, result.policy_logits)
    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        name="mlx_inference",
        metrics={
            "iterations": iterations,
            "warmup": warmup,
            "channels": channels,
            "blocks": blocks,
            "value_hidden": value_hidden,
            "masked": mask_legal_moves,
            "elapsed_seconds": elapsed,
            "avg_latency_ms": elapsed / iterations * 1000.0,
            "inferences_per_second": _rate(iterations, elapsed),
        },
    )


def benchmark_mlx_batched_inference(
    *,
    iterations: int = 25,
    warmup: int = 5,
    batch_size: int = 8,
    channels: int = 32,
    blocks: int = 2,
    value_hidden: int = 64,
) -> BenchmarkResult:
    """Measure raw batched MLX model throughput without legal-move masking."""
    _require_at_least("iterations", iterations, 1)
    _require_at_least("warmup", warmup, 0)
    _require_at_least("batch_size", batch_size, 1)
    config = _model_config(channels=channels, blocks=blocks, value_hidden=value_hidden)
    model = PolicyValueNet(config)
    tensor = encode_game(Game.new())
    batch = mx.stack([tensor for _ in range(batch_size)])
    for _ in range(warmup):
        output = model(batch)
        mx.eval(output.policy_logits, output.value)

    start = time.perf_counter()
    for _ in range(iterations):
        output = model(batch)
        mx.eval(output.policy_logits, output.value)
    elapsed = time.perf_counter() - start
    positions = iterations * batch_size
    return BenchmarkResult(
        name="mlx_batched_inference",
        metrics={
            "iterations": iterations,
            "warmup": warmup,
            "batch_size": batch_size,
            "channels": channels,
            "blocks": blocks,
            "value_hidden": value_hidden,
            "elapsed_seconds": elapsed,
            "avg_batch_latency_ms": elapsed / iterations * 1000.0,
            "positions_per_second": _rate(positions, elapsed),
        },
    )


def run_benchmark_suite(
    *,
    smoke: bool = False,
    include_batched: bool = True,
) -> BenchmarkReport:
    """Run the full Python benchmark suite and return a report object."""
    results: list[BenchmarkResult]
    if smoke:
        results = [
            benchmark_move_generation(iterations=2),
            benchmark_complete_games(games=1, max_plies=8),
            benchmark_mcts(simulations=2, rollout_plies=2),
            benchmark_mlx_inference(iterations=1, warmup=0, channels=8, blocks=1, value_hidden=8),
        ]
        if include_batched:
            results.append(
                benchmark_mlx_batched_inference(
                    iterations=1,
                    warmup=0,
                    batch_size=2,
                    channels=8,
                    blocks=1,
                    value_hidden=8,
                )
            )
    else:
        results = [
            benchmark_move_generation(),
            benchmark_complete_games(),
            benchmark_mcts(),
            benchmark_mlx_inference(),
        ]
        if include_batched:
            results.append(benchmark_mlx_batched_inference())
    report_results = tuple(results)
    return BenchmarkReport(
        results=report_results,
        recommendation=recommend_swift_acceleration(report_results, smoke=smoke),
    )


def recommend_swift_acceleration(
    results: tuple[BenchmarkResult, ...], *, smoke: bool = False
) -> str:
    """Return a conservative suite-time Swift acceleration heuristic."""
    if smoke:
        return (
            "Smoke benchmark completed. Do not justify Swift acceleration from smoke numbers. "
            "The recommendation is only a suite-time heuristic, not a full application profile; "
            "run the default/full suite repeatedly and compare bottlenecks first."
        )

    by_name = {result.name: result for result in results}
    mcts_elapsed = _metric(by_name, "mcts_simulations", "elapsed_seconds")
    game_elapsed = _metric(by_name, "complete_game_simulation", "elapsed_seconds")
    move_elapsed = _metric(by_name, "move_generation", "elapsed_seconds")
    inference_elapsed = _metric(by_name, "mlx_inference", "elapsed_seconds")
    total = mcts_elapsed + game_elapsed + move_elapsed + inference_elapsed
    if total <= 0:
        return (
            "No timing signal was captured. This suite-time heuristic does not justify Swift "
            "acceleration."
        )

    dominant_name, dominant_elapsed = max(
        (
            ("classical MCTS/search", mcts_elapsed),
            ("complete-game simulation", game_elapsed),
            ("legal move generation", move_elapsed),
            ("MLX inference wrapper", inference_elapsed),
        ),
        key=lambda item: item[1],
    )
    share = dominant_elapsed / total
    if share < 0.5:
        return (
            "No single Python component dominates the deliberately sized benchmark suite time. "
            "This heuristic does not justify Swift acceleration yet; prefer algorithmic tuning, "
            "batching, and repeated measurements first."
        )
    if dominant_name == "MLX inference wrapper":
        return (
            "Inference dominates the deliberately sized benchmark suite time. Swift engine "
            "acceleration is unlikely to address that cost; investigate batching/model sizing "
            "before considering Swift/Core ML integration."
        )
    return (
        f"{dominant_name} dominates about {share:.0%} of this deliberately sized benchmark "
        "suite time. This is a heuristic, not a full application profile; Swift acceleration "
        "may be worth a focused spike only after confirming this result across repeat runs and "
        "preserving Python/external fixture parity."
    )


def format_report(report: BenchmarkReport, *, output_format: ReportFormat = "markdown") -> str:
    """Format a benchmark report as Markdown or JSON."""
    if output_format == "json":
        return json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if output_format != "markdown":
        raise ValueError(f"unsupported report format: {output_format}")

    lines = ["# tinychess Benchmark Report", ""]
    for result in report.results:
        lines.append(f"## {result.name}")
        lines.append("")
        for key, value in result.metrics.items():
            lines.append(f"- {key}: {_format_metric(value)}")
        lines.append("")
    lines.extend(["## Swift Acceleration Recommendation", "", report.recommendation, ""])
    return "\n".join(lines)


def _model_config(*, channels: int, blocks: int, value_hidden: int) -> PolicyValueConfig:
    _require_at_least("channels", channels, 1)
    _require_at_least("blocks", blocks, 0)
    _require_at_least("value_hidden", value_hidden, 1)
    return PolicyValueConfig(
        residual_channels=channels,
        residual_blocks=blocks,
        value_hidden_dim=value_hidden,
    )


def _rate(count: int, elapsed: float) -> float:
    if elapsed == 0:
        return math.inf
    return count / elapsed


def _require_at_least(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _metric(results: dict[str, BenchmarkResult], name: str, metric: str) -> float:
    value = results.get(name, BenchmarkResult(name, {})).metrics.get(metric, 0.0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)
