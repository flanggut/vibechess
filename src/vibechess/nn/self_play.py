"""Self-play game generation for neural MCTS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from vibechess.ai.mcts import MCTSPlayer
from vibechess.ai.neural_mcts import (
    NeuralInference,
    NeuralMCTSConfig,
    NeuralMCTSInferenceRequest,
    NeuralMCTSPlayer,
    NeuralMCTSResult,
    NeuralMCTSSearchSession,
)
from vibechess.ai.search_config import MCTSConfig
from vibechess.engine.game import Game, determine_outcome
from vibechess.engine.move import Move
from vibechess.engine.outcome import Outcome, OutcomeReason
from vibechess.engine.piece import Color
from vibechess.nn.encode import (
    ACTION_SPACE_SIZE,
    TENSOR_SHAPE,
    encode_game_np,
    legal_move_mask_from_legal_moves_np,
    move_to_action_index,
)
from vibechess.nn.inference import PolicyValueInference
from vibechess.nn.self_play_dataset import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_GAMES_FILENAME,
    DEFAULT_METADATA_FILENAME,
    SELF_PLAY_DATASET_SCHEMA_VERSION,
    SelfPlayDataset,
    SelfPlayGameRecord,
    SelfPlayMetadata,
    _outcome_values,
    load_self_play_dataset,
    merge_self_play_datasets,
    save_self_play_dataset,
)
from vibechess.profiling import (
    ProfileStats as SelfPlayProfileStats,
)
from vibechess.profiling import (
    activate_self_play_profile,
    profile_scope,
    record_counter,
    record_distribution,
)

DEFAULT_PROFILE_FILENAME = "profile.json"
LABEL_SOURCE_NEURAL = "neural"
LABEL_SOURCE_CLASSICAL = "classical"
LABEL_SOURCES = (LABEL_SOURCE_NEURAL, LABEL_SOURCE_CLASSICAL)
BATCHING_MODE_SERIAL = "serial"
BATCHING_MODE_CENTRAL_INFERENCE_QUEUE = "central_inference_queue"


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    """Settings for a small local self-play generation run."""

    games: int = 1
    max_plies: int = 128
    mcts: NeuralMCTSConfig = field(default_factory=NeuralMCTSConfig)
    classical_mcts: MCTSConfig = field(default_factory=MCTSConfig)
    label_source: str = LABEL_SOURCE_NEURAL
    model_checkpoint_id: str | None = None
    seed: int | None = None
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.games < 1:
            raise ValueError(f"games must be at least 1, got {self.games}")
        if self.max_plies < 0:
            raise ValueError(f"max_plies must be non-negative, got {self.max_plies}")
        if self.label_source not in LABEL_SOURCES:
            raise ValueError(
                f"label_source must be one of {LABEL_SOURCES}, got {self.label_source!r}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")

    def to_dict(
        self,
        *,
        batching_mode: str | None = None,
        inference_batch_size: int | None = None,
    ) -> dict[str, object]:
        """Return JSON-serializable generation settings."""
        resolved_batching_mode = batching_mode or BATCHING_MODE_SERIAL
        resolved_inference_batch_size = (
            inference_batch_size
            if inference_batch_size is not None
            else (
                self.batch_size
                if resolved_batching_mode == BATCHING_MODE_CENTRAL_INFERENCE_QUEUE
                else 1
            )
        )
        return {
            "games": self.games,
            "max_plies": self.max_plies,
            "label_source": self.label_source,
            "mcts": asdict(self.mcts),
            "classical_mcts": asdict(self.classical_mcts),
            "model_checkpoint_id": self.model_checkpoint_id,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "batching_mode": resolved_batching_mode,
            "inference_batch_size": resolved_inference_batch_size,
        }


@dataclass(frozen=True, slots=True)
class SelfPlayProgress:
    """Progress event emitted after one self-play game is completed."""

    games_completed: int
    total_games: int
    samples: int
    plies: int
    game_index: int


# Compatibility wrapper for older callers that imported the benchmark profile API
# from this module. The implementation lives in vibechess.profiling and
# no longer monkeypatches engine/model classes.
def self_play_profile(
    level: str | None = "detailed",
) -> Any:
    """Activate direct self-play profiling for the current context.

    Kept as a compatibility wrapper around :func:`activate_self_play_profile`.
    """
    return activate_self_play_profile(level)


def generate_self_play_dataset(
    inference: NeuralInference | None,
    config: SelfPlayConfig | None = None,
    *,
    progress: Callable[[SelfPlayProgress], None] | None = None,
) -> SelfPlayDataset:
    """Generate a small self-play dataset using neural or classical MCTS labels."""
    resolved = SelfPlayConfig() if config is None else config
    with profile_scope(
        "self_play.generate_dataset",
        games=resolved.games,
        max_plies=resolved.max_plies,
        label_source=resolved.label_source,
        batch_size=resolved.batch_size,
    ):
        if (
            resolved.batch_size > 1
            and resolved.label_source == LABEL_SOURCE_NEURAL
            and isinstance(inference, PolicyValueInference)
        ):
            return _generate_batched_neural_self_play_dataset(
                inference,
                resolved,
                progress=progress,
            )
        return _generate_serial_self_play_dataset(inference, resolved, progress=progress)


def _generate_serial_self_play_dataset(
    inference: NeuralInference | None,
    config: SelfPlayConfig,
    *,
    progress: Callable[[SelfPlayProgress], None] | None = None,
) -> SelfPlayDataset:
    positions: list[npt.NDArray[np.float32]] = []
    legal_masks: list[npt.NDArray[np.float32]] = []
    policies: list[npt.NDArray[np.float32]] = []
    outcome_values: list[float] = []
    game_records: list[SelfPlayGameRecord] = []
    completed_plies = 0

    with profile_scope("self_play.serial_loop"):
        for game_index in range(config.games):
            game = Game.new()
            player = _player_for_game(inference, config, game_index)
            game_sides: list[Color] = []
            with profile_scope("self_play.game", game_index=game_index):
                for ply_index in range(config.max_plies):
                    with profile_scope("self_play.ply", game_index=game_index, ply_index=ply_index):
                        legal = game.legal_moves
                        record_distribution(
                            "self_play.legal_moves_per_ply",
                            len(legal),
                            unit="moves",
                        )
                        with profile_scope("self_play.terminal_check"):
                            terminal = determine_outcome(game, legal_moves=legal)
                        if terminal is not None:
                            break
                        if not legal:
                            break
                        result = player.search(game)
                        if result.move not in legal:
                            msg = f"search selected illegal move: {result.move}"
                            raise ValueError(msg)
                        with profile_scope("record.position_encode_np"):
                            positions.append(encode_game_np(game))
                        with profile_scope("record.legal_mask_np"):
                            legal_masks.append(legal_move_mask_from_legal_moves_np(game, legal))
                        with profile_scope("record.policy_target"):
                            policies.append(_policy_target(game, result.visit_counts, result.move))
                        record_counter("self_play.samples")
                        record_counter("self_play.plies")
                        game_sides.append(game.board.side_to_move)
                        game = game.play_known_legal(result.move)
                if game.outcome is None:
                    game = _with_max_plies_outcome(game)
                with profile_scope("record.outcome_values"):
                    outcome_values.extend(_outcome_values(game, game_sides))
                with profile_scope("record.game_record"):
                    game_record = _game_record(game_index, game)
                    game_records.append(game_record)
                completed_plies += game_record.plies
                record_counter("self_play.games_completed")
                if progress is not None:
                    progress(
                        SelfPlayProgress(
                            games_completed=len(game_records),
                            total_games=config.games,
                            samples=len(positions),
                            plies=completed_plies,
                            game_index=game_index,
                        )
                    )

    return _self_play_dataset_from_samples(
        config,
        batching_mode=BATCHING_MODE_SERIAL,
        inference_batch_size=1,
        positions=positions,
        legal_masks=legal_masks,
        policies=policies,
        outcome_values=outcome_values,
        game_records=game_records,
    )


@dataclass(slots=True)
class _BatchedGameState:
    game_index: int
    game: Game
    player: NeuralMCTSPlayer
    game_sides: list[Color] = field(default_factory=list)
    positions: list[npt.NDArray[np.float32]] = field(default_factory=list)
    legal_masks: list[npt.NDArray[np.float32]] = field(default_factory=list)
    policies: list[npt.NDArray[np.float32]] = field(default_factory=list)


@dataclass(slots=True)
class _CentralSearch:
    state: _BatchedGameState
    legal: tuple[Move, ...]
    session: NeuralMCTSSearchSession


def _record_batched_decision(
    state: _BatchedGameState,
    legal: tuple[Move, ...],
    selected_move: Move,
    visit_counts: dict[Any, int],
) -> None:
    if selected_move not in legal:
        msg = f"search selected illegal move: {selected_move}"
        raise ValueError(msg)
    with profile_scope("record.position_encode_np"):
        state.positions.append(encode_game_np(state.game))
    with profile_scope("record.legal_mask_np"):
        state.legal_masks.append(legal_move_mask_from_legal_moves_np(state.game, legal))
    with profile_scope("record.policy_target"):
        state.policies.append(_policy_target(state.game, visit_counts, selected_move))
    record_counter("self_play.samples")
    record_counter("self_play.plies")
    state.game_sides.append(state.game.board.side_to_move)
    state.game = state.game.play_known_legal(selected_move)


def _run_central_neural_searches(
    inference: PolicyValueInference,
    decisions: list[tuple[_BatchedGameState, tuple[Move, ...]]],
    *,
    batch_size: int,
) -> list[tuple[_BatchedGameState, tuple[Move, ...], NeuralMCTSResult]]:
    """Run one neural MCTS search per decision with deterministic cross-game batching."""
    searches = [
        _CentralSearch(
            state=state,
            legal=legal,
            session=NeuralMCTSSearchSession(
                state.player,
                state.game,
                session_id=state.game_index,
            ),
        )
        for state, legal in sorted(decisions, key=lambda item: item[0].game_index)
    ]
    results: dict[int, NeuralMCTSResult] = {}
    pending: dict[int, NeuralMCTSInferenceRequest] = {}

    with profile_scope("self_play.central_inference_queue", sessions=len(searches)):
        while len(results) < len(searches):
            progressed = False
            if pending:
                batch_indexes = sorted(pending)[:batch_size]
                requests = [pending.pop(index) for index in batch_indexes]
                games = tuple(request.game for request in requests)
                legal_by_game = tuple(request.legal_moves for request in requests)
                with profile_scope(
                    "self_play.central_predict_legal_batch",
                    batch_size=len(requests),
                ):
                    batch = inference.predict_legal_batch(games, legal_by_game)
                for row_index, search_index in enumerate(batch_indexes):
                    searches[search_index].session.resume(batch.result_at(row_index))
                progressed = True
            else:
                for search_index, search in enumerate(searches):
                    if search_index in results or search.session.pending_request is not None:
                        continue
                    with profile_scope(
                        "self_play.central_session_advance",
                        game_index=search.state.game_index,
                    ):
                        advanced = search.session.advance()
                    progressed = True
                    if isinstance(advanced, NeuralMCTSResult):
                        results[search_index] = advanced
                    else:
                        pending[search_index] = advanced

            if not progressed:
                raise RuntimeError("central neural inference queue made no progress")

    return [
        (search.state, search.legal, results[search_index])
        for search_index, search in enumerate(searches)
    ]


def _generate_batched_neural_self_play_dataset(
    inference: PolicyValueInference,
    config: SelfPlayConfig,
    *,
    progress: Callable[[SelfPlayProgress], None] | None = None,
) -> SelfPlayDataset:
    positions: list[npt.NDArray[np.float32]] = []
    legal_masks: list[npt.NDArray[np.float32]] = []
    policies: list[npt.NDArray[np.float32]] = []
    outcome_values: list[float] = []
    game_records: list[SelfPlayGameRecord] = []
    completed_plies = 0

    with profile_scope("self_play.batched_loop"):
        for start_game in range(0, config.games, config.batch_size):
            game_count = min(config.batch_size, config.games - start_game)
            states = [
                _BatchedGameState(
                    game_index=start_game + offset,
                    game=Game.new(),
                    player=cast(
                        NeuralMCTSPlayer,
                        _player_for_game(inference, config, start_game + offset),
                    ),
                )
                for offset in range(game_count)
            ]
            for ply_index in range(config.max_plies):
                decisions: list[tuple[_BatchedGameState, tuple[Move, ...]]] = []
                for state in states:
                    with profile_scope(
                        "self_play.ply",
                        game_index=state.game_index,
                        ply_index=ply_index,
                    ):
                        legal = state.game.legal_moves
                        record_distribution(
                            "self_play.legal_moves_per_ply",
                            len(legal),
                            unit="moves",
                        )
                        with profile_scope("self_play.terminal_check"):
                            terminal = determine_outcome(state.game, legal_moves=legal)
                        if terminal is None and legal:
                            decisions.append((state, legal))
                if not decisions:
                    break

                central_results = _run_central_neural_searches(
                    inference,
                    decisions,
                    batch_size=config.batch_size,
                )
                for state, legal, result in central_results:
                    with profile_scope(
                        "self_play.ply_decision",
                        game_index=state.game_index,
                        ply_index=ply_index,
                    ):
                        _record_batched_decision(state, legal, result.move, result.visit_counts)

            for state in states:
                with profile_scope("self_play.game", game_index=state.game_index):
                    if state.game.outcome is None:
                        state.game = _with_max_plies_outcome(state.game)
                    positions.extend(state.positions)
                    legal_masks.extend(state.legal_masks)
                    policies.extend(state.policies)
                    with profile_scope("record.outcome_values"):
                        outcome_values.extend(_outcome_values(state.game, state.game_sides))
                    with profile_scope("record.game_record"):
                        game_record = _game_record(state.game_index, state.game)
                        game_records.append(game_record)
                    completed_plies += game_record.plies
                    record_counter("self_play.games_completed")
                    if progress is not None:
                        progress(
                            SelfPlayProgress(
                                games_completed=len(game_records),
                                total_games=config.games,
                                samples=len(positions),
                                plies=completed_plies,
                                game_index=state.game_index,
                            )
                        )

    return _self_play_dataset_from_samples(
        config,
        batching_mode=BATCHING_MODE_CENTRAL_INFERENCE_QUEUE,
        inference_batch_size=config.batch_size,
        positions=positions,
        legal_masks=legal_masks,
        policies=policies,
        outcome_values=outcome_values,
        game_records=game_records,
    )


def _self_play_dataset_from_samples(
    config: SelfPlayConfig,
    *,
    batching_mode: str,
    inference_batch_size: int,
    positions: list[npt.NDArray[np.float32]],
    legal_masks: list[npt.NDArray[np.float32]],
    policies: list[npt.NDArray[np.float32]],
    outcome_values: list[float],
    game_records: list[SelfPlayGameRecord],
) -> SelfPlayDataset:
    with profile_scope("dataset.outcomes_array"):
        outcomes = np.asarray(outcome_values, dtype=np.float32)
    metadata = SelfPlayMetadata.create(
        config,
        sample_count=len(positions),
        batching_mode=batching_mode,
        inference_batch_size=inference_batch_size,
    )
    with profile_scope("dataset.stack_positions"):
        stacked_positions = _stack_or_empty(positions, TENSOR_SHAPE)
    with profile_scope("dataset.stack_legal_masks"):
        stacked_legal_masks = _stack_or_empty(legal_masks, (ACTION_SPACE_SIZE,))
    with profile_scope("dataset.stack_policies"):
        stacked_policies = _stack_or_empty(policies, (ACTION_SPACE_SIZE,))
    return SelfPlayDataset(
        positions=stacked_positions,
        legal_masks=stacked_legal_masks,
        mcts_policies=stacked_policies,
        outcomes=outcomes,
        metadata=metadata,
        games=game_records,
    )


def _player_for_game(
    inference: NeuralInference | None,
    config: SelfPlayConfig,
    game_index: int,
) -> NeuralMCTSPlayer | MCTSPlayer:
    if config.label_source == LABEL_SOURCE_CLASSICAL:
        return MCTSPlayer(_classical_mcts_config_for_game(config, game_index))
    if inference is None:
        raise ValueError("neural self-play generation requires inference")
    return NeuralMCTSPlayer(inference, _mcts_config_for_game(config, game_index))


def _mcts_config_for_game(config: SelfPlayConfig, game_index: int) -> NeuralMCTSConfig:
    if config.seed is None:
        return config.mcts
    return replace(config.mcts, seed=config.seed + game_index)


def _classical_mcts_config_for_game(
    config: SelfPlayConfig, game_index: int
) -> MCTSConfig:
    if config.seed is None:
        return config.classical_mcts
    return replace(config.classical_mcts, seed=config.seed + game_index)


def _policy_target(
    game: Game,
    visit_counts: dict[Any, int],
    selected_move: Any,
) -> npt.NDArray[np.float32]:
    policy = np.zeros((ACTION_SPACE_SIZE,), dtype=np.float32)
    total = sum(max(0, visits) for visits in visit_counts.values())
    if total > 0:
        for move, visits in visit_counts.items():
            policy[move_to_action_index(move, game.board)] = max(0, visits) / total
    else:
        policy[move_to_action_index(selected_move, game.board)] = 1.0
    return policy




def _game_record(game_index: int, game: Game) -> SelfPlayGameRecord:
    outcome = game.outcome
    if outcome is None:
        outcome = Outcome(OutcomeReason.MAX_PLIES)
    return SelfPlayGameRecord(
        game_index=game_index,
        plies=len(game.moves),
        outcome_reason=outcome.reason.value,
        winner=None if outcome.winner is None else outcome.winner.value,
        final_fen=game.to_fen(),
        moves_uci=[move.to_uci() for move in game.moves],
    )


def _with_max_plies_outcome(game: Game) -> Game:
    return Game(
        positions=game.positions,
        moves=game.moves,
        halfmove_clock=game.halfmove_clock,
        fullmove_number=game.fullmove_number,
        repetition_counts=dict(game.repetition_counts),
        forced_outcome=Outcome(OutcomeReason.MAX_PLIES),
    )


def _stack_or_empty(
    arrays: list[npt.NDArray[np.float32]],
    trailing_shape: tuple[int, ...],
) -> npt.NDArray[np.float32]:
    if not arrays:
        return np.zeros((0, *trailing_shape), dtype=np.float32)
    return np.stack(arrays).astype(np.float32, copy=False)




__all__ = [
    "BATCHING_MODE_CENTRAL_INFERENCE_QUEUE",
    "BATCHING_MODE_SERIAL",
    "DEFAULT_DATASET_FILENAME",
    "DEFAULT_GAMES_FILENAME",
    "DEFAULT_METADATA_FILENAME",
    "DEFAULT_PROFILE_FILENAME",
    "LABEL_SOURCE_CLASSICAL",
    "LABEL_SOURCE_NEURAL",
    "LABEL_SOURCES",
    "SELF_PLAY_DATASET_SCHEMA_VERSION",
    "SelfPlayConfig",
    "SelfPlayDataset",
    "SelfPlayGameRecord",
    "SelfPlayMetadata",
    "SelfPlayProfileStats",
    "SelfPlayProgress",
    "generate_self_play_dataset",
    "load_self_play_dataset",
    "merge_self_play_datasets",
    "save_self_play_dataset",
    "self_play_profile",
]
