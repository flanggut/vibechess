"""Smoke-friendly evaluation harness for players and neural checkpoints."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from tinychess.ai.mcts import MCTSPlayer
from tinychess.ai.neural_mcts import NeuralMCTSConfig, NeuralMCTSPlayer
from tinychess.ai.player import Player, RandomPlayer, play_game
from tinychess.ai.search_config import MCTSConfig
from tinychess.engine.game import Game
from tinychess.engine.piece import Color
from tinychess.nn.checkpoint import load_checkpoint
from tinychess.nn.inference import PolicyValueInference

PlayerFactory = Callable[[], Player]


@dataclass(frozen=True, slots=True)
class PlayerSpec:
    """Named player factory used by the evaluation harness.

    A factory is used instead of a long-lived player instance so each game can be
    reproducible and independent even when players own local RNG/search state.
    """

    name: str
    factory: PlayerFactory


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Small match settings suitable for smoke tests and local checkpoint checks."""

    games: int = 2
    max_plies: int = 80
    alternate_colors: bool = True

    def __post_init__(self) -> None:
        if self.games < 1:
            raise ValueError(f"games must be at least 1, got {self.games}")
        if self.max_plies < 0:
            raise ValueError(f"max_plies must be non-negative, got {self.max_plies}")


@dataclass(frozen=True, slots=True)
class MatchGameRecord:
    """Serializable result of one evaluated game."""

    game_index: int
    player_a_color: str
    player_b_color: str
    plies: int
    outcome_reason: str
    winner: str | None
    player_a_score: float
    final_fen: str
    moves_uci: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Aggregate result for a player-A versus player-B match."""

    player_a: str
    player_b: str
    games: int
    player_a_score: float
    player_b_score: float
    player_a_wins: int
    player_b_wins: int
    draws: int
    records: list[MatchGameRecord]

    @property
    def player_a_score_rate(self) -> float:
        """Return player A's score divided by the number of games."""
        return self.player_a_score / self.games

    @property
    def player_b_score_rate(self) -> float:
        """Return player B's score divided by the number of games."""
        return self.player_b_score / self.games

    def to_dict(self) -> dict[str, object]:
        return {
            "player_a": self.player_a,
            "player_b": self.player_b,
            "games": self.games,
            "player_a_score": self.player_a_score,
            "player_b_score": self.player_b_score,
            "player_a_score_rate": self.player_a_score_rate,
            "player_b_score_rate": self.player_b_score_rate,
            "player_a_wins": self.player_a_wins,
            "player_b_wins": self.player_b_wins,
            "draws": self.draws,
            "records": [record.to_dict() for record in self.records],
        }


def run_match(
    player_a: PlayerSpec,
    player_b: PlayerSpec,
    config: MatchConfig | None = None,
) -> MatchResult:
    """Run a legal-game match between two player specs and return aggregate scores."""
    resolved = MatchConfig() if config is None else config
    records = _run_match_records(player_a, player_b, resolved, start_game=0, games=resolved.games)
    return _match_result_from_records(player_a.name, player_b.name, records)


def _run_match_records(
    player_a: PlayerSpec,
    player_b: PlayerSpec,
    config: MatchConfig,
    *,
    start_game: int,
    games: int,
) -> list[MatchGameRecord]:
    records: list[MatchGameRecord] = []
    for game_index in range(start_game, start_game + games):
        a_is_white = (game_index % 2 == 0) or not config.alternate_colors
        white = player_a.factory() if a_is_white else player_b.factory()
        black = player_b.factory() if a_is_white else player_a.factory()
        game = play_game(white, black, game=Game.new(), max_plies=config.max_plies)
        score = _score_for_player_a(game, player_a_is_white=a_is_white)
        outcome = game.outcome
        if outcome is None:  # pragma: no cover - play_game forces max-plies at cap.
            raise RuntimeError("evaluated game ended without an outcome")
        records.append(
            MatchGameRecord(
                game_index=game_index,
                player_a_color=Color.WHITE.value if a_is_white else Color.BLACK.value,
                player_b_color=Color.BLACK.value if a_is_white else Color.WHITE.value,
                plies=len(game.moves),
                outcome_reason=outcome.reason.value,
                winner=outcome.winner.value if outcome.winner is not None else None,
                player_a_score=score,
                final_fen=game.to_fen(),
                moves_uci=[move.to_uci() for move in game.moves],
            )
        )
    return records


def _match_result_from_records(
    player_a: str,
    player_b: str,
    records: Sequence[MatchGameRecord],
) -> MatchResult:
    ordered = sorted(records, key=lambda record: record.game_index)
    a_score = sum(record.player_a_score for record in ordered)
    games = len(ordered)
    b_score = games - a_score
    a_wins = sum(1 for record in ordered if record.player_a_score == 1.0)
    b_wins = sum(1 for record in ordered if record.player_a_score == 0.0)
    draws = games - a_wins - b_wins
    return MatchResult(
        player_a=player_a,
        player_b=player_b,
        games=games,
        player_a_score=a_score,
        player_b_score=b_score,
        player_a_wins=a_wins,
        player_b_wins=b_wins,
        draws=draws,
        records=list(ordered),
    )


@dataclass(frozen=True, slots=True)
class PromotionCriteria:
    """Early checkpoint promotion thresholds.

    These criteria are intentionally smoke/progress checks for the learning
    pipeline. Passing them is not evidence of competitive chess strength.
    """

    min_games_per_baseline: int = 2
    min_score_rate_vs_random: float = 0.5
    min_score_rate_vs_mcts: float = 0.0
    required_baselines: tuple[str, ...] = ("random", "mcts")

    def __post_init__(self) -> None:
        if self.min_games_per_baseline < 1:
            raise ValueError("min_games_per_baseline must be at least 1")
        for name, value in (
            ("min_score_rate_vs_random", self.min_score_rate_vs_random),
            ("min_score_rate_vs_mcts", self.min_score_rate_vs_mcts),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["note"] = EARLY_PROMOTION_NOTE
        return data


EARLY_PROMOTION_NOTE = (
    "WP16 promotion criteria are early smoke/progress validation only; "
    "they do not claim competitive chess strength."
)


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Result of applying early promotion criteria to baseline matches."""

    promoted: bool
    reasons: list[str]
    score_rates: dict[str, float]
    note: str = EARLY_PROMOTION_NOTE

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_promotion(
    baseline_results: Mapping[str, MatchResult],
    criteria: PromotionCriteria | None = None,
) -> PromotionDecision:
    """Apply explicit early promotion criteria to checkpoint baseline results."""
    resolved = PromotionCriteria() if criteria is None else criteria
    reasons: list[str] = []
    score_rates: dict[str, float] = {}
    thresholds = {
        "random": resolved.min_score_rate_vs_random,
        "mcts": resolved.min_score_rate_vs_mcts,
    }

    for baseline in resolved.required_baselines:
        result = baseline_results.get(baseline)
        if result is None:
            reasons.append(f"missing required baseline: {baseline}")
            continue
        score_rate = result.player_a_score_rate
        score_rates[baseline] = score_rate
        if result.games < resolved.min_games_per_baseline:
            reasons.append(
                f"{baseline}: only {result.games} games; "
                f"requires at least {resolved.min_games_per_baseline}"
            )
        threshold = thresholds.get(baseline)
        if threshold is not None and score_rate < threshold:
            reasons.append(
                f"{baseline}: score rate {score_rate:.3f} below required {threshold:.3f}"
            )

    promoted = not reasons
    if promoted:
        reasons.append("all early smoke promotion criteria passed")
    return PromotionDecision(promoted=promoted, reasons=reasons, score_rates=score_rates)


def checkpoint_player_spec(
    checkpoint_dir: str | Path,
    *,
    name: str | None = None,
    config: NeuralMCTSConfig | None = None,
) -> PlayerSpec:
    """Create a neural-MCTS player spec by loading an MLX checkpoint per game."""
    path = Path(checkpoint_dir)
    player_name = name or path.name
    search_config = NeuralMCTSConfig(simulations=1) if config is None else config

    def factory() -> Player:
        loaded = load_checkpoint(path)
        return NeuralMCTSPlayer(PolicyValueInference(loaded.model), config=search_config)

    return PlayerSpec(name=player_name, factory=factory)


def _checkpoint_player_spec_reusing_loaded_checkpoint(
    checkpoint_dir: str | Path,
    *,
    name: str,
    config: NeuralMCTSConfig | None,
) -> PlayerSpec:
    path = Path(checkpoint_dir)
    search_config = NeuralMCTSConfig(simulations=1) if config is None else config
    loaded = load_checkpoint(path)
    inference = PolicyValueInference(loaded.model)

    def factory() -> Player:
        return NeuralMCTSPlayer(inference, config=search_config)

    return PlayerSpec(name=name, factory=factory)


def random_player_spec(*, seed: int | None = None, name: str = "random") -> PlayerSpec:
    """Return a random-player baseline spec."""

    def factory() -> Player:
        return RandomPlayer(seed=seed)

    return PlayerSpec(name=name, factory=factory)


def mcts_player_spec(
    *,
    config: MCTSConfig | None = None,
    name: str = "mcts",
) -> PlayerSpec:
    """Return a classical-MCTS baseline spec."""
    search_config = MCTSConfig(simulations=1, max_rollout_plies=0) if config is None else config

    def factory() -> Player:
        return MCTSPlayer(search_config)

    return PlayerSpec(name=name, factory=factory)


@dataclass(frozen=True, slots=True)
class _EvaluationChunk:
    baseline: str
    start_game: int
    games: int


@dataclass(frozen=True, slots=True)
class _MatchChunk:
    start_game: int
    games: int


def _validate_baselines(baselines: Sequence[str]) -> None:
    for baseline in baselines:
        if baseline not in {"random", "mcts"}:
            raise ValueError(f"unsupported baseline {baseline!r}; expected random or mcts")


def _run_parallel_evaluation(
    checkpoint_dir: Path,
    *,
    selected_baselines: Sequence[str],
    match_config: MatchConfig,
    neural_config: NeuralMCTSConfig | None,
    mcts_config: MCTSConfig | None,
    random_seed: int | None,
    workers: int,
) -> dict[str, MatchResult]:
    total_games = len(selected_baselines) * match_config.games
    effective_workers = min(workers, total_games)
    chunks = _evaluation_chunks(selected_baselines, match_config.games, effective_workers)
    records_by_baseline: dict[str, list[MatchGameRecord]] = {
        baseline: [] for baseline in selected_baselines
    }
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        for baseline, records in executor.map(
            _run_evaluation_chunk,
            chunks,
            [checkpoint_dir] * len(chunks),
            [match_config] * len(chunks),
            [neural_config] * len(chunks),
            [mcts_config] * len(chunks),
            [random_seed] * len(chunks),
        ):
            records_by_baseline[baseline].extend(records)

    return {
        baseline: _match_result_from_records("checkpoint", baseline, records_by_baseline[baseline])
        for baseline in selected_baselines
    }


def _evaluation_chunks(
    baselines: Sequence[str],
    games_per_baseline: int,
    effective_workers: int,
) -> list[_EvaluationChunk]:
    total_games = len(baselines) * games_per_baseline
    chunk_size = max(1, math.ceil(total_games / effective_workers))
    chunks: list[_EvaluationChunk] = []
    for baseline in baselines:
        for chunk in _match_chunks(games_per_baseline, chunk_size=chunk_size):
            chunks.append(
                _EvaluationChunk(
                    baseline=baseline,
                    start_game=chunk.start_game,
                    games=chunk.games,
                )
            )
    return chunks


def _match_chunks(games: int, *, chunk_size: int) -> list[_MatchChunk]:
    return [
        _MatchChunk(start_game=start_game, games=min(chunk_size, games - start_game))
        for start_game in range(0, games, chunk_size)
    ]


def _run_evaluation_chunk(
    chunk: _EvaluationChunk,
    checkpoint_dir: Path,
    match_config: MatchConfig,
    neural_config: NeuralMCTSConfig | None,
    mcts_config: MCTSConfig | None,
    random_seed: int | None,
) -> tuple[str, list[MatchGameRecord]]:
    checkpoint = _checkpoint_player_spec_reusing_loaded_checkpoint(
        checkpoint_dir,
        name="checkpoint",
        config=neural_config,
    )
    baseline = _baseline_spec(chunk.baseline, mcts_config=mcts_config, random_seed=random_seed)
    records = _run_match_records(
        checkpoint,
        baseline,
        match_config,
        start_game=chunk.start_game,
        games=chunk.games,
    )
    return chunk.baseline, records


def _baseline_spec(
    baseline: str,
    *,
    mcts_config: MCTSConfig | None,
    random_seed: int | None,
) -> PlayerSpec:
    if baseline == "random":
        return random_player_spec(seed=random_seed)
    if baseline == "mcts":
        return mcts_player_spec(config=mcts_config)
    raise ValueError(f"unsupported baseline {baseline!r}; expected random or mcts")


def _resolved_neural_config(config: NeuralMCTSConfig | None) -> NeuralMCTSConfig:
    return NeuralMCTSConfig(simulations=1) if config is None else config


def _run_parallel_checkpoint_match(
    checkpoint_dir: Path,
    opponent_checkpoint_dir: Path,
    *,
    match_config: MatchConfig,
    neural_config: NeuralMCTSConfig,
    opponent_neural_config: NeuralMCTSConfig,
    workers: int,
) -> MatchResult:
    effective_workers = min(workers, match_config.games)
    chunk_size = max(1, math.ceil(match_config.games / effective_workers))
    chunks = _match_chunks(match_config.games, chunk_size=chunk_size)
    records: list[MatchGameRecord] = []
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        for chunk_records in executor.map(
            _run_checkpoint_match_chunk,
            chunks,
            [checkpoint_dir] * len(chunks),
            [opponent_checkpoint_dir] * len(chunks),
            [match_config] * len(chunks),
            [neural_config] * len(chunks),
            [opponent_neural_config] * len(chunks),
        ):
            records.extend(chunk_records)
    return _match_result_from_records("checkpoint", "opponent_checkpoint", records)


def _run_checkpoint_match_chunk(
    chunk: _MatchChunk,
    checkpoint_dir: Path,
    opponent_checkpoint_dir: Path,
    match_config: MatchConfig,
    neural_config: NeuralMCTSConfig,
    opponent_neural_config: NeuralMCTSConfig,
) -> list[MatchGameRecord]:
    checkpoint = _checkpoint_player_spec_reusing_loaded_checkpoint(
        checkpoint_dir,
        name="checkpoint",
        config=neural_config,
    )
    opponent = _checkpoint_player_spec_reusing_loaded_checkpoint(
        opponent_checkpoint_dir,
        name="opponent_checkpoint",
        config=opponent_neural_config,
    )
    return _run_match_records(
        checkpoint,
        opponent,
        match_config,
        start_game=chunk.start_game,
        games=chunk.games,
    )


def evaluate_checkpoints_head_to_head(
    checkpoint_dir: str | Path,
    opponent_checkpoint_dir: str | Path,
    *,
    match_config: MatchConfig | None = None,
    neural_config: NeuralMCTSConfig | None = None,
    opponent_neural_config: NeuralMCTSConfig | None = None,
    workers: int = 1,
) -> dict[str, object]:
    """Compare two checkpoint-backed neural-MCTS players without baselines.

    This evaluator is for direct checkpoint-versus-checkpoint matches. It does
    not run random/classical-MCTS baselines and does not produce promotion
    criteria or promotion decisions.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    resolved_match_config = MatchConfig() if match_config is None else match_config
    resolved_neural_config = _resolved_neural_config(neural_config)
    resolved_opponent_config = (
        resolved_neural_config if opponent_neural_config is None else opponent_neural_config
    )
    checkpoint_path = Path(checkpoint_dir)
    opponent_path = Path(opponent_checkpoint_dir)

    if workers == 1 or resolved_match_config.games <= 1:
        checkpoint = checkpoint_player_spec(
            checkpoint_path,
            name="checkpoint",
            config=resolved_neural_config,
        )
        opponent = checkpoint_player_spec(
            opponent_path,
            name="opponent_checkpoint",
            config=resolved_opponent_config,
        )
        result = run_match(checkpoint, opponent, resolved_match_config)
    else:
        result = _run_parallel_checkpoint_match(
            checkpoint_path,
            opponent_path,
            match_config=resolved_match_config,
            neural_config=resolved_neural_config,
            opponent_neural_config=resolved_opponent_config,
            workers=workers,
        )

    return {
        "mode": "neural_vs_neural",
        "checkpoint": str(checkpoint_path),
        "opponent_checkpoint": str(opponent_path),
        "neural_configs": {
            "checkpoint": asdict(resolved_neural_config),
            "opponent": asdict(resolved_opponent_config),
        },
        "match": result.to_dict(),
    }


def evaluate_checkpoint_against_baselines(
    checkpoint_dir: str | Path,
    *,
    match_config: MatchConfig | None = None,
    neural_config: NeuralMCTSConfig | None = None,
    mcts_config: MCTSConfig | None = None,
    random_seed: int | None = 0,
    baselines: Sequence[str] = ("random", "mcts"),
    criteria: PromotionCriteria | None = None,
    workers: int = 1,
) -> dict[str, object]:
    """Compare a checkpoint player against random/classical-MCTS baselines."""
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    resolved_match_config = MatchConfig() if match_config is None else match_config
    selected_baselines = tuple(baselines)
    _validate_baselines(selected_baselines)

    if workers == 1 or len(selected_baselines) * resolved_match_config.games <= 1:
        checkpoint = checkpoint_player_spec(checkpoint_dir, name="checkpoint", config=neural_config)
        baseline_specs = {
            "random": random_player_spec(seed=random_seed),
            "mcts": mcts_player_spec(config=mcts_config),
        }
        results = {
            baseline: run_match(checkpoint, baseline_specs[baseline], resolved_match_config)
            for baseline in selected_baselines
        }
    else:
        results = _run_parallel_evaluation(
            Path(checkpoint_dir),
            selected_baselines=selected_baselines,
            match_config=resolved_match_config,
            neural_config=neural_config,
            mcts_config=mcts_config,
            random_seed=random_seed,
            workers=workers,
        )

    resolved_criteria = criteria or PromotionCriteria(required_baselines=selected_baselines)
    decision = assess_promotion(results, resolved_criteria)
    return {
        "checkpoint": str(Path(checkpoint_dir)),
        "criteria": resolved_criteria.to_dict(),
        "promotion": decision.to_dict(),
        "matches": {name: result.to_dict() for name, result in results.items()},
    }


def write_evaluation_report(report: Mapping[str, object], output_path: str | Path) -> None:
    """Write an evaluation report as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def _score_for_player_a(game: Game, *, player_a_is_white: bool) -> float:
    outcome = game.outcome
    if outcome is None or outcome.winner is None:
        return 0.5
    player_a_color = Color.WHITE if player_a_is_white else Color.BLACK
    return 1.0 if outcome.winner is player_a_color else 0.0


__all__ = [
    "EARLY_PROMOTION_NOTE",
    "MatchConfig",
    "MatchGameRecord",
    "MatchResult",
    "PlayerSpec",
    "PromotionCriteria",
    "PromotionDecision",
    "assess_promotion",
    "checkpoint_player_spec",
    "evaluate_checkpoint_against_baselines",
    "evaluate_checkpoints_head_to_head",
    "mcts_player_spec",
    "random_player_spec",
    "run_match",
    "write_evaluation_report",
]
