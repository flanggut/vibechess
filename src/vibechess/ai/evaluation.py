"""Smoke-friendly evaluation harness for players and neural checkpoints."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from hashlib import blake2b
from pathlib import Path
from typing import TypeAlias, TypeVar

from vibechess.ai.mcts import MCTSPlayer
from vibechess.ai.neural_mcts import (
    NeuralMCTSConfig,
    NeuralMCTSPlayer,
    NeuralMCTSResult,
    NeuralMCTSSearchSession,
    resolve_active_limit,
    run_batched_sessions,
)
from vibechess.ai.player import Player, RandomPlayer, play_game
from vibechess.ai.search_config import MCTSConfig
from vibechess.engine.game import Game, game_record_fields
from vibechess.engine.move import Move
from vibechess.engine.outcome import Outcome, OutcomeReason
from vibechess.engine.piece import Color
from vibechess.nn.checkpoint import load_checkpoint
from vibechess.nn.inference import PolicyValueInference

PlayerFactory = Callable[[], Player]
SeededPlayerFactory: TypeAlias = Callable[[int, str], Player]
EvaluationProgressCallback: TypeAlias = Callable[["EvaluationProgress"], None]
DEFAULT_EVALUATION_OPENING_COUNT = 64
DEFAULT_EVALUATION_OPENING_PLIES = 8


@dataclass(frozen=True, slots=True)
class PlayerSpec:
    """Named player factory used by the evaluation harness.

    A factory is used instead of a long-lived player instance so each game can be
    reproducible and independent even when players own local RNG/search state.
    Seeded factories additionally receive the global game index and player role so
    evaluation runs can derive stable per-game RNG streams from one base seed.
    """

    name: str
    factory: PlayerFactory
    seeded_factory: SeededPlayerFactory | None = None

    def create(self, *, game_index: int, role: str) -> Player:
        """Create a player for one evaluated game."""
        if self.seeded_factory is not None:
            return self.seeded_factory(game_index, role)
        return self.factory()


@dataclass(frozen=True, slots=True)
class EvaluationProgress:
    """Progress event emitted as checkpoint evaluation games complete."""

    games_completed: int
    total_games: int
    completed_plies: int
    game_index: int
    baseline: str | None = None
    worker_id: int = 0
    worker_start_game: int = 0
    worker_games: int | None = None
    worker_plies: int | None = None


@dataclass(frozen=True, slots=True)
class _EvaluationProgressContext:
    callback: EvaluationProgressCallback
    total_games: int
    completed_offset: int = 0
    plies_offset: int = 0
    baseline: str | None = None
    worker_id: int = 0
    worker_start_game: int = 0
    worker_games: int | None = None
    worker_plies: int | None = None


def _emit_evaluation_progress(
    context: _EvaluationProgressContext | None,
    *,
    local_games_completed: int,
    local_plies_completed: int,
    game_index: int,
) -> None:
    if context is None:
        return
    context.callback(
        EvaluationProgress(
            games_completed=context.completed_offset + local_games_completed,
            total_games=context.total_games,
            completed_plies=context.plies_offset + local_plies_completed,
            game_index=game_index,
            baseline=context.baseline,
            worker_id=context.worker_id,
            worker_start_game=context.worker_start_game,
            worker_games=context.worker_games,
            worker_plies=(
                local_plies_completed
                if context.worker_plies is None
                else context.worker_plies
            ),
        )
    )


def _derive_game_seed(
    base_seed: int | None,
    *,
    game_index: int,
    role: str,
    player_name: str,
) -> int | None:
    """Derive a stable per-game seed from a match-level seed."""
    if base_seed is None:
        return None
    payload = f"vibechess-evaluation-seed-v1:{base_seed}:{game_index}:{role}:{player_name}"
    return int.from_bytes(blake2b(payload.encode(), digest_size=8).digest(), "big")


def _derive_opening_seed(base_seed: int, candidate_index: int) -> int:
    payload = f"vibechess-evaluation-opening-v1:{base_seed}:{candidate_index}"
    return int.from_bytes(blake2b(payload.encode(), digest_size=8).digest(), "big")


def generate_unique_openings(config: OpeningConfig) -> tuple[MatchOpening, ...]:
    """Generate unique deterministic random opening positions.

    Uniqueness is enforced on the resulting starting FEN. If the requested number
    cannot be generated with the configured opening length, evaluation fails rather
    than silently replaying a duplicate opening.
    """
    openings: list[MatchOpening] = []
    seen_fens: set[str] = set()
    max_attempts = max(config.count * 100, 1000)
    candidate_index = 0
    while len(openings) < config.count and candidate_index < max_attempts:
        opening_seed = _derive_opening_seed(config.seed, candidate_index)
        rng = random.Random(opening_seed)
        game = Game.new()
        moves: list[Move] = []
        for _ in range(config.plies):
            legal = game.legal_moves
            if not legal or game.outcome is not None:
                break
            move = rng.choice(legal)
            game = game.play_known_legal(move)
            moves.append(move)
        candidate_index += 1
        if len(moves) != config.plies or game.outcome is not None:
            continue
        starting_fen = game.to_fen()
        if starting_fen in seen_fens:
            continue
        seen_fens.add(starting_fen)
        openings.append(
            MatchOpening(
                opening_index=len(openings),
                opening_seed=opening_seed,
                moves_uci=tuple(move.to_uci() for move in moves),
                starting_fen=starting_fen,
                game=game,
            )
        )
    if len(openings) != config.count:
        msg = (
            f"could not generate {config.count} unique openings with "
            f"{config.plies} plies after {max_attempts} attempts"
        )
        raise ValueError(msg)
    return tuple(openings)


def _paired_opening_starts(opening_config: OpeningConfig | None) -> tuple[_GameStart, ...] | None:
    if opening_config is None:
        return None
    starts: list[_GameStart] = []
    for opening in generate_unique_openings(opening_config):
        start = _GameStart(
            game=opening.game,
            start_plies=len(opening.game.moves),
            opening_index=opening.opening_index,
            opening_seed=opening.opening_seed,
            opening_moves_uci=opening.moves_uci,
            starting_fen=opening.starting_fen,
        )
        starts.extend((start, start))
    return tuple(starts)


def _validate_opening_starts(
    match_config: MatchConfig,
    opening_starts: Sequence[_GameStart] | None,
) -> None:
    if opening_starts is not None and len(opening_starts) != match_config.games:
        msg = (
            f"paired openings produce {len(opening_starts)} games, "
            f"but match config requests {match_config.games}"
        )
        raise ValueError(msg)


def _game_start_for_index(
    opening_starts: Sequence[_GameStart] | None,
    game_index: int,
) -> _GameStart:
    if opening_starts is None:
        return _GameStart(game=Game.new(), start_plies=0)
    return opening_starts[game_index]


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
class OpeningConfig:
    """Deterministic random opening generation settings for evaluation matches."""

    count: int = DEFAULT_EVALUATION_OPENING_COUNT
    plies: int = DEFAULT_EVALUATION_OPENING_PLIES
    seed: int = 0

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"opening count must be at least 1, got {self.count}")
        if self.plies < 1:
            raise ValueError(f"opening plies must be at least 1, got {self.plies}")


@dataclass(frozen=True, slots=True)
class MatchOpening:
    """One unique generated opening position used by paired evaluation games."""

    opening_index: int
    opening_seed: int
    moves_uci: tuple[str, ...]
    starting_fen: str
    game: Game


@dataclass(frozen=True, slots=True)
class _GameStart:
    game: Game
    start_plies: int
    opening_index: int | None = None
    opening_seed: int | None = None
    opening_moves_uci: tuple[str, ...] = ()
    starting_fen: str | None = None


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
    opening_index: int | None
    opening_seed: int | None
    opening_moves_uci: list[str]
    starting_fen: str | None

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
    records = _run_match_records(
        player_a,
        player_b,
        resolved,
        start_game=0,
        games=resolved.games,
        opening_starts=None,
    )
    return _match_result_from_records(player_a.name, player_b.name, records)


def _run_match_records(
    player_a: PlayerSpec,
    player_b: PlayerSpec,
    config: MatchConfig,
    *,
    start_game: int,
    games: int,
    opening_starts: Sequence[_GameStart] | None,
    progress_context: _EvaluationProgressContext | None = None,
) -> list[MatchGameRecord]:
    records: list[MatchGameRecord] = []
    completed_plies = 0
    for game_index in range(start_game, start_game + games):
        game_start = _game_start_for_index(opening_starts, game_index)
        a_is_white = (game_index % 2 == 0) or not config.alternate_colors
        white = (
            player_a.create(game_index=game_index, role="player_a")
            if a_is_white
            else player_b.create(game_index=game_index, role="player_b")
        )
        black = (
            player_b.create(game_index=game_index, role="player_b")
            if a_is_white
            else player_a.create(game_index=game_index, role="player_a")
        )
        game = play_game(white, black, game=game_start.game, max_plies=config.max_plies)
        record = _match_game_record(
            game_index,
            game,
            player_a_is_white=a_is_white,
            game_start=game_start,
        )
        records.append(record)
        completed_plies += record.plies
        _emit_evaluation_progress(
            progress_context,
            local_games_completed=len(records),
            local_plies_completed=completed_plies,
            game_index=game_index,
        )
    return records


def _run_match_chunk_records(
    player_a: PlayerSpec,
    player_b: PlayerSpec,
    config: MatchConfig,
    *,
    start_game: int,
    games: int,
    batch_size: int,
    active_games: int | None,
    opening_starts: Sequence[_GameStart] | None,
    progress_context: _EvaluationProgressContext | None = None,
) -> list[MatchGameRecord]:
    """Run one chunk of games, using batched scheduling only when it can help.

    The batched scheduler is used when ``batch_size > 1`` and more than one game is
    in play; otherwise games run serially. Both paths produce identical records, so
    this branch is centralized here rather than repeated at every call site.
    """
    if batch_size > 1 and games > 1:
        return _run_batched_match_records(
            player_a,
            player_b,
            config,
            start_game=start_game,
            games=games,
            batch_size=batch_size,
            active_games=active_games,
            opening_starts=opening_starts,
            progress_context=progress_context,
        )
    return _run_match_records(
        player_a,
        player_b,
        config,
        start_game=start_game,
        games=games,
        opening_starts=opening_starts,
        progress_context=progress_context,
    )


def _match_game_record(
    game_index: int,
    game: Game,
    *,
    player_a_is_white: bool,
    game_start: _GameStart,
) -> MatchGameRecord:
    score = _score_for_player_a(game, player_a_is_white=player_a_is_white)
    outcome = game.outcome
    if outcome is None:
        raise RuntimeError("evaluated game ended without an outcome")
    fields = game_record_fields(game, outcome)
    return MatchGameRecord(
        game_index=game_index,
        player_a_color=Color.WHITE.value if player_a_is_white else Color.BLACK.value,
        player_b_color=Color.BLACK.value if player_a_is_white else Color.WHITE.value,
        plies=fields.plies,
        outcome_reason=fields.outcome_reason,
        winner=fields.winner,
        player_a_score=score,
        final_fen=fields.final_fen,
        moves_uci=fields.moves_uci,
        opening_index=game_start.opening_index,
        opening_seed=game_start.opening_seed,
        opening_moves_uci=list(game_start.opening_moves_uci),
        starting_fen=game_start.starting_fen,
    )


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
    derive_game_seeds: bool = False,
) -> PlayerSpec:
    """Create a neural-MCTS player spec backed by one loaded MLX checkpoint.

    The returned factory still creates a fresh ``NeuralMCTSPlayer`` for each game,
    so RNG and tree state are not shared across games. When ``derive_game_seeds`` is
    enabled, each game receives a stable seed derived from ``config.seed`` and the
    global game index instead of restarting from the same seed.
    """
    path = Path(checkpoint_dir)
    player_name = name or path.name
    search_config = NeuralMCTSConfig(simulations=1) if config is None else config
    loaded = load_checkpoint(path)
    inference = PolicyValueInference(loaded.model)

    def factory() -> Player:
        return NeuralMCTSPlayer(inference, config=search_config)

    def seeded_factory(game_index: int, role: str) -> Player:
        seed = _derive_game_seed(
            search_config.seed,
            game_index=game_index,
            role=role,
            player_name=player_name,
        )
        return NeuralMCTSPlayer(inference, config=replace(search_config, seed=seed))

    return PlayerSpec(
        name=player_name,
        factory=factory,
        seeded_factory=seeded_factory if derive_game_seeds else None,
    )


def _checkpoint_player_spec_reusing_loaded_checkpoint(
    checkpoint_dir: str | Path,
    *,
    name: str,
    config: NeuralMCTSConfig | None,
) -> PlayerSpec:
    return checkpoint_player_spec(
        checkpoint_dir,
        name=name,
        config=config,
        derive_game_seeds=True,
    )


def random_player_spec(
    *,
    seed: int | None = None,
    name: str = "random",
    derive_game_seeds: bool = False,
) -> PlayerSpec:
    """Return a random-player baseline spec."""

    def factory() -> Player:
        return RandomPlayer(seed=seed)

    def seeded_factory(game_index: int, role: str) -> Player:
        return RandomPlayer(
            seed=_derive_game_seed(seed, game_index=game_index, role=role, player_name=name)
        )

    return PlayerSpec(
        name=name,
        factory=factory,
        seeded_factory=seeded_factory if derive_game_seeds else None,
    )


def mcts_player_spec(
    *,
    config: MCTSConfig | None = None,
    name: str = "mcts",
    derive_game_seeds: bool = False,
) -> PlayerSpec:
    """Return a classical-MCTS baseline spec."""
    search_config = MCTSConfig(simulations=1, max_rollout_plies=0) if config is None else config

    def factory() -> Player:
        return MCTSPlayer(search_config)

    def seeded_factory(game_index: int, role: str) -> Player:
        seed = _derive_game_seed(
            search_config.seed,
            game_index=game_index,
            role=role,
            player_name=name,
        )
        return MCTSPlayer(replace(search_config, seed=seed))

    return PlayerSpec(
        name=name,
        factory=factory,
        seeded_factory=seeded_factory if derive_game_seeds else None,
    )


@dataclass(frozen=True, slots=True)
class _EvaluationChunk:
    baseline: str
    start_game: int
    games: int



@dataclass(slots=True)
class _BatchedMatchGameState:
    game_index: int
    game: Game
    game_start: _GameStart
    start_plies: int
    player_a_is_white: bool
    white: Player
    black: Player


@dataclass(frozen=True, slots=True)
class _BatchedNeuralDecision:
    state: _BatchedMatchGameState
    player: NeuralMCTSPlayer
    legal: tuple[Move, ...]


def _run_batched_match_records(
    player_a: PlayerSpec,
    player_b: PlayerSpec,
    config: MatchConfig,
    *,
    start_game: int,
    games: int,
    batch_size: int,
    active_games: int | None,
    opening_starts: Sequence[_GameStart] | None,
    progress_context: _EvaluationProgressContext | None = None,
) -> list[MatchGameRecord]:
    active_limit = resolve_active_limit(games, batch_size, active_games)
    active: dict[int, _BatchedMatchGameState] = {}
    completed: dict[int, MatchGameRecord] = {}
    completed_plies = 0
    next_game_index = start_game
    end_game_index = start_game + games

    def launch_available_games() -> None:
        nonlocal next_game_index
        while len(active) < active_limit and next_game_index < end_game_index:
            game_start = _game_start_for_index(opening_starts, next_game_index)
            a_is_white = (next_game_index % 2 == 0) or not config.alternate_colors
            active[next_game_index] = _BatchedMatchGameState(
                game_index=next_game_index,
                game=game_start.game,
                game_start=game_start,
                start_plies=game_start.start_plies,
                player_a_is_white=a_is_white,
                white=(
                    player_a.create(game_index=next_game_index, role="player_a")
                    if a_is_white
                    else player_b.create(game_index=next_game_index, role="player_b")
                ),
                black=(
                    player_b.create(game_index=next_game_index, role="player_b")
                    if a_is_white
                    else player_a.create(game_index=next_game_index, role="player_a")
                ),
            )
            next_game_index += 1

    def complete_state(state: _BatchedMatchGameState) -> None:
        nonlocal completed_plies
        active.pop(state.game_index, None)
        game = state.game
        if game.outcome is None:
            game = game.with_forced_outcome(Outcome(OutcomeReason.MAX_PLIES))
        record = _match_game_record(
            state.game_index,
            game,
            player_a_is_white=state.player_a_is_white,
            game_start=state.game_start,
        )
        completed[state.game_index] = record
        completed_plies += record.plies
        _emit_evaluation_progress(
            progress_context,
            local_games_completed=len(completed),
            local_plies_completed=completed_plies,
            game_index=state.game_index,
        )
    launch_available_games()
    while len(completed) < games:
        neural_decisions_by_inference: dict[
            PolicyValueInference,
            list[_BatchedNeuralDecision],
        ] = {}
        progressed = False
        for state in [active[index] for index in sorted(active)]:
            if (
                state.game.outcome is not None
                or len(state.game.moves) - state.start_plies >= config.max_plies
            ):
                complete_state(state)
                progressed = True
                continue
            legal = state.game.legal_moves
            if not legal:
                complete_state(state)
                progressed = True
                continue
            player = state.white if state.game.board.side_to_move is Color.WHITE else state.black
            if isinstance(player, NeuralMCTSPlayer) and isinstance(
                player.inference,
                PolicyValueInference,
            ):
                neural_decisions_by_inference.setdefault(player.inference, []).append(
                    _BatchedNeuralDecision(
                        state=state,
                        player=player,
                        legal=legal,
                    )
                )
                continue
            move = player.select_move(state.game)
            if move not in legal:
                msg = f"player selected illegal move: {move}"
                raise ValueError(msg)
            state.game = state.game.play_known_legal(move)
            progressed = True

        for inference, decisions in neural_decisions_by_inference.items():
            for state, legal, result in _run_batched_neural_decisions(
                inference,
                decisions,
                batch_size=batch_size,
            ):
                if result.move not in legal:
                    msg = f"player selected illegal move: {result.move}"
                    raise ValueError(msg)
                state.game = state.game.play_known_legal(result.move)
                progressed = True

        if not progressed:
            raise RuntimeError("batched evaluation scheduler made no progress")
        launch_available_games()

    return [completed[index] for index in range(start_game, end_game_index)]


def _run_batched_neural_decisions(
    inference: PolicyValueInference,
    decisions: Sequence[_BatchedNeuralDecision],
    *,
    batch_size: int,
) -> list[tuple[_BatchedMatchGameState, tuple[Move, ...], NeuralMCTSResult]]:
    ordered = sorted(decisions, key=lambda item: item.state.game_index)
    sessions = [
        NeuralMCTSSearchSession(
            decision.player,
            decision.state.game,
            session_id=decision.state.game_index,
        )
        for decision in ordered
    ]
    results = run_batched_sessions(
        sessions,
        inference.predict_legal_batch,
        batch_size=batch_size,
    )
    return [
        (decision.state, decision.legal, result)
        for decision, result in zip(ordered, results, strict=True)
    ]




@dataclass(frozen=True, slots=True)
class _MatchChunk:
    start_game: int
    games: int


def _validate_baselines(baselines: Sequence[str]) -> None:
    for baseline in baselines:
        if baseline not in {"random", "mcts"}:
            raise ValueError(f"unsupported baseline {baseline!r}; expected random or mcts")



def _validate_batching(batch_size: int, active_games: int | None) -> None:
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if active_games is not None and active_games < 1:
        raise ValueError(f"active_games must be at least 1, got {active_games}")

_ChunkT = TypeVar("_ChunkT")
_ResultT = TypeVar("_ResultT")


def _map_chunks(
    worker: Callable[..., _ResultT],
    chunks: Sequence[_ChunkT],
    broadcast_args: Sequence[object],
    *,
    workers: int,
    on_result: Callable[[_ChunkT, _ResultT], None] | None = None,
) -> list[_ResultT]:
    """Run ``worker`` over ``chunks`` in a process pool, broadcasting shared args.

    Each entry in ``broadcast_args`` is repeated once per chunk so it is passed
    positionally to every worker call (matching ``executor.map`` fan-out). Results
    are returned in chunk order. Shared by the baseline and head-to-head parallel
    evaluators, which differ only in their worker, chunk type, and aggregation.
    """
    broadcasts = [[arg] * len(chunks) for arg in broadcast_args]
    results: list[_ResultT] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for chunk, result in zip(chunks, executor.map(worker, chunks, *broadcasts), strict=True):
            if on_result is not None:
                on_result(chunk, result)
            results.append(result)
    return results


def _run_parallel_evaluation(
    checkpoint_dir: Path,
    *,
    selected_baselines: Sequence[str],
    match_config: MatchConfig,
    neural_config: NeuralMCTSConfig | None,
    mcts_config: MCTSConfig | None,
    random_seed: int | None,
    workers: int,
    batch_size: int,
    active_games: int | None,
    opening_starts: Sequence[_GameStart] | None,
    progress: EvaluationProgressCallback | None = None,
) -> dict[str, MatchResult]:
    total_games = len(selected_baselines) * match_config.games
    effective_workers = min(workers, total_games)
    chunks = _evaluation_chunks(selected_baselines, match_config.games, effective_workers)
    records_by_baseline: dict[str, list[MatchGameRecord]] = {
        baseline: [] for baseline in selected_baselines
    }
    completed_games = 0
    completed_plies = 0

    def report_chunk_progress(
        chunk: _EvaluationChunk,
        result: tuple[str, list[MatchGameRecord]],
    ) -> None:
        nonlocal completed_games, completed_plies
        if progress is None:
            return
        baseline, chunk_records = result
        if not chunk_records:
            return
        completed_games += len(chunk_records)
        completed_plies += sum(record.plies for record in chunk_records)
        progress(
            EvaluationProgress(
                games_completed=completed_games,
                total_games=total_games,
                completed_plies=completed_plies,
                game_index=chunk_records[-1].game_index,
                baseline=baseline,
                worker_id=chunks.index(chunk),
                worker_start_game=(
                    selected_baselines.index(baseline) * match_config.games
                    + chunk.start_game
                ),
                worker_games=chunk.games,
                worker_plies=sum(record.plies for record in chunk_records),
            )
        )

    for baseline, records in _map_chunks(
        _run_evaluation_chunk,
        chunks,
        [
            checkpoint_dir,
            match_config,
            neural_config,
            mcts_config,
            random_seed,
            batch_size,
            active_games,
            opening_starts,
        ],
        workers=effective_workers,
        on_result=report_chunk_progress,
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
    batch_size: int,
    active_games: int | None,
    opening_starts: Sequence[_GameStart] | None,
) -> tuple[str, list[MatchGameRecord]]:
    checkpoint = _checkpoint_player_spec_reusing_loaded_checkpoint(
        checkpoint_dir,
        name="checkpoint",
        config=neural_config,
    )
    baseline = _baseline_spec(chunk.baseline, mcts_config=mcts_config, random_seed=random_seed)
    records = _run_match_chunk_records(
        checkpoint,
        baseline,
        match_config,
        start_game=chunk.start_game,
        games=chunk.games,
        batch_size=batch_size,
        active_games=active_games,
        opening_starts=opening_starts,
    )
    return chunk.baseline, records


def _baseline_spec(
    baseline: str,
    *,
    mcts_config: MCTSConfig | None,
    random_seed: int | None,
) -> PlayerSpec:
    if baseline == "random":
        return random_player_spec(seed=random_seed, derive_game_seeds=True)
    if baseline == "mcts":
        return mcts_player_spec(config=mcts_config, derive_game_seeds=True)
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
    batch_size: int,
    active_games: int | None,
    opening_starts: Sequence[_GameStart] | None,
    progress: EvaluationProgressCallback | None = None,
) -> MatchResult:
    effective_workers = min(workers, match_config.games)
    chunk_size = max(1, math.ceil(match_config.games / effective_workers))
    chunks = _match_chunks(match_config.games, chunk_size=chunk_size)
    records: list[MatchGameRecord] = []
    completed_games = 0
    completed_plies = 0

    def report_chunk_progress(chunk: _MatchChunk, chunk_records: list[MatchGameRecord]) -> None:
        nonlocal completed_games, completed_plies
        if progress is None or not chunk_records:
            return
        completed_games += len(chunk_records)
        completed_plies += sum(record.plies for record in chunk_records)
        progress(
            EvaluationProgress(
                games_completed=completed_games,
                total_games=match_config.games,
                completed_plies=completed_plies,
                game_index=chunk_records[-1].game_index,
                baseline="opponent_checkpoint",
                worker_id=chunks.index(chunk),
                worker_start_game=chunk.start_game,
                worker_games=chunk.games,
                worker_plies=sum(record.plies for record in chunk_records),
            )
        )

    for chunk_records in _map_chunks(
        _run_checkpoint_match_chunk,
        chunks,
        [
            checkpoint_dir,
            opponent_checkpoint_dir,
            match_config,
            neural_config,
            opponent_neural_config,
            batch_size,
            active_games,
            opening_starts,
        ],
        workers=effective_workers,
        on_result=report_chunk_progress,
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
    batch_size: int,
    active_games: int | None,
    opening_starts: Sequence[_GameStart] | None,
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
    return _run_match_chunk_records(
        checkpoint,
        opponent,
        match_config,
        start_game=chunk.start_game,
        games=chunk.games,
        batch_size=batch_size,
        active_games=active_games,
        opening_starts=opening_starts,
    )


def evaluate_checkpoints_head_to_head(
    checkpoint_dir: str | Path,
    opponent_checkpoint_dir: str | Path,
    *,
    match_config: MatchConfig | None = None,
    neural_config: NeuralMCTSConfig | None = None,
    opponent_neural_config: NeuralMCTSConfig | None = None,
    workers: int = 1,
    batch_size: int = 1,
    active_games: int | None = None,
    opening_config: OpeningConfig | None = None,
    progress: EvaluationProgressCallback | None = None,
) -> dict[str, object]:
    """Compare two checkpoint-backed neural-MCTS players without baselines.

    This evaluator is for direct checkpoint-versus-checkpoint matches. It does
    not run random/classical-MCTS baselines and does not produce promotion
    criteria or promotion decisions.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    _validate_batching(batch_size, active_games)
    resolved_match_config = MatchConfig() if match_config is None else match_config
    opening_starts = _paired_opening_starts(opening_config)
    _validate_opening_starts(resolved_match_config, opening_starts)
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
            derive_game_seeds=True,
        )
        opponent = checkpoint_player_spec(
            opponent_path,
            name="opponent_checkpoint",
            config=resolved_opponent_config,
            derive_game_seeds=True,
        )
        records = _run_match_chunk_records(
            checkpoint,
            opponent,
            resolved_match_config,
            start_game=0,
            games=resolved_match_config.games,
            batch_size=batch_size,
            active_games=active_games,
            opening_starts=opening_starts,
            progress_context=_EvaluationProgressContext(
                callback=progress,
                total_games=resolved_match_config.games,
                baseline="opponent_checkpoint",
                worker_games=resolved_match_config.games,
            )
            if progress is not None
            else None,
        )
        result = _match_result_from_records("checkpoint", "opponent_checkpoint", records)
    else:
        result = _run_parallel_checkpoint_match(
            checkpoint_path,
            opponent_path,
            match_config=resolved_match_config,
            neural_config=resolved_neural_config,
            opponent_neural_config=resolved_opponent_config,
            workers=workers,
            batch_size=batch_size,
            active_games=active_games,
            opening_starts=opening_starts,
            progress=progress,
        )

    return {
        "mode": "neural_vs_neural",
        "checkpoint": str(checkpoint_path),
        "opponent_checkpoint": str(opponent_path),
        "neural_configs": {
            "checkpoint": asdict(resolved_neural_config),
            "opponent": asdict(resolved_opponent_config),
        },
        "opening_config": None if opening_config is None else asdict(opening_config),
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
    batch_size: int = 1,
    active_games: int | None = None,
    opening_config: OpeningConfig | None = None,
    progress: EvaluationProgressCallback | None = None,
) -> dict[str, object]:
    """Compare a checkpoint player against random/classical-MCTS baselines."""
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    _validate_batching(batch_size, active_games)
    resolved_match_config = MatchConfig() if match_config is None else match_config
    selected_baselines = tuple(baselines)
    _validate_baselines(selected_baselines)
    opening_starts = _paired_opening_starts(opening_config)
    _validate_opening_starts(resolved_match_config, opening_starts)

    if workers == 1 or len(selected_baselines) * resolved_match_config.games <= 1:
        checkpoint = checkpoint_player_spec(
            checkpoint_dir,
            name="checkpoint",
            config=neural_config,
            derive_game_seeds=True,
        )
        baseline_specs = {
            baseline: _baseline_spec(
                baseline,
                mcts_config=mcts_config,
                random_seed=random_seed,
            )
            for baseline in selected_baselines
        }
        results: dict[str, MatchResult] = {}
        completed_games = 0
        completed_plies = 0
        total_games = len(selected_baselines) * resolved_match_config.games
        for baseline in selected_baselines:
            records = _run_match_chunk_records(
                checkpoint,
                baseline_specs[baseline],
                resolved_match_config,
                start_game=0,
                games=resolved_match_config.games,
                batch_size=batch_size,
                active_games=active_games,
                opening_starts=opening_starts,
                progress_context=_EvaluationProgressContext(
                    callback=progress,
                    total_games=total_games,
                    completed_offset=completed_games,
                    plies_offset=completed_plies,
                    baseline=baseline,
                    worker_games=resolved_match_config.games,
                )
                if progress is not None
                else None,
            )
            results[baseline] = _match_result_from_records("checkpoint", baseline, records)
            completed_games += len(records)
            completed_plies += sum(record.plies for record in records)
    else:
        results = _run_parallel_evaluation(
            Path(checkpoint_dir),
            selected_baselines=selected_baselines,
            match_config=resolved_match_config,
            neural_config=neural_config,
            mcts_config=mcts_config,
            random_seed=random_seed,
            workers=workers,
            batch_size=batch_size,
            active_games=active_games,
            opening_starts=opening_starts,
            progress=progress,
        )

    resolved_criteria = criteria or PromotionCriteria(required_baselines=selected_baselines)
    decision = assess_promotion(results, resolved_criteria)
    return {
        "checkpoint": str(Path(checkpoint_dir)),
        "criteria": resolved_criteria.to_dict(),
        "promotion": decision.to_dict(),
        "opening_config": None if opening_config is None else asdict(opening_config),
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
    "DEFAULT_EVALUATION_OPENING_COUNT",
    "DEFAULT_EVALUATION_OPENING_PLIES",
    "EvaluationProgress",
    "MatchConfig",
    "MatchGameRecord",
    "MatchResult",
    "MatchOpening",
    "OpeningConfig",
    "PlayerSpec",
    "PromotionCriteria",
    "PromotionDecision",
    "assess_promotion",
    "checkpoint_player_spec",
    "evaluate_checkpoint_against_baselines",
    "generate_unique_openings",
    "evaluate_checkpoints_head_to_head",
    "mcts_player_spec",
    "random_player_spec",
    "run_match",
    "write_evaluation_report",
]
