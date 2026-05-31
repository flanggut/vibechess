"""Self-play game generation and dataset persistence for neural MCTS."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import tinychess
from tinychess.ai.mcts import MCTSPlayer
from tinychess.ai.neural_mcts import NeuralInference, NeuralMCTSConfig, NeuralMCTSPlayer
from tinychess.ai.search_config import MCTSConfig
from tinychess.engine.game import Game
from tinychess.engine.move import Move
from tinychess.engine.outcome import Outcome, OutcomeReason
from tinychess.engine.piece import Color
from tinychess.nn.encode import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    ENCODER_VERSION,
    TENSOR_SHAPE,
    encode_game,
    legal_move_mask,
    move_to_action_index,
)

SELF_PLAY_DATASET_SCHEMA_VERSION = "tinychess-selfplay-v1"
DEFAULT_DATASET_FILENAME = "samples.npz"
DEFAULT_METADATA_FILENAME = "metadata.json"
DEFAULT_GAMES_FILENAME = "games.jsonl"
LABEL_SOURCE_NEURAL = "neural"
LABEL_SOURCE_CLASSICAL = "classical"
LABEL_SOURCES = (LABEL_SOURCE_NEURAL, LABEL_SOURCE_CLASSICAL)


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

    def __post_init__(self) -> None:
        if self.games < 1:
            raise ValueError(f"games must be at least 1, got {self.games}")
        if self.max_plies < 0:
            raise ValueError(f"max_plies must be non-negative, got {self.max_plies}")
        if self.label_source not in LABEL_SOURCES:
            raise ValueError(
                f"label_source must be one of {LABEL_SOURCES}, got {self.label_source!r}"
            )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable generation settings."""
        return {
            "games": self.games,
            "max_plies": self.max_plies,
            "label_source": self.label_source,
            "mcts": asdict(self.mcts),
            "classical_mcts": asdict(self.classical_mcts),
            "model_checkpoint_id": self.model_checkpoint_id,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class SelfPlayGameRecord:
    """Game-level metadata for one generated self-play game."""

    game_index: int
    plies: int
    outcome_reason: str
    winner: str | None
    final_fen: str
    moves_uci: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable game record."""
        return {
            "game_index": self.game_index,
            "plies": self.plies,
            "outcome_reason": self.outcome_reason,
            "winner": self.winner,
            "final_fen": self.final_fen,
            "moves_uci": self.moves_uci,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SelfPlayGameRecord:
        """Parse a game record from JSON data."""
        moves = data.get("moves_uci")
        if not isinstance(moves, list) or not all(
            isinstance(move, str) for move in moves
        ):
            raise TypeError("game record field 'moves_uci' must be a list of strings")
        winner = data.get("winner")
        if winner is not None and not isinstance(winner, str):
            raise TypeError("game record field 'winner' must be a string or null")
        return cls(
            game_index=_expect_int(data, "game_index"),
            plies=_expect_int(data, "plies"),
            outcome_reason=_expect_str(data, "outcome_reason"),
            winner=winner,
            final_fen=_expect_str(data, "final_fen"),
            moves_uci=moves,
        )


@dataclass(frozen=True, slots=True)
class SelfPlayMetadata:
    """Dataset-level metadata stored next to self-play tensor batches."""

    schema_version: str
    generated_at: str
    engine_version: str
    git_commit: str | None
    action_space_version: str
    encoder_version: str
    model_checkpoint_id: str | None
    generation_settings: dict[str, object]
    sample_count: int
    game_count: int

    @classmethod
    def create(cls, config: SelfPlayConfig, *, sample_count: int) -> SelfPlayMetadata:
        """Create metadata for a generated dataset."""
        return cls(
            schema_version=SELF_PLAY_DATASET_SCHEMA_VERSION,
            generated_at=datetime.now(UTC).isoformat(),
            engine_version=tinychess.__version__,
            git_commit=_git_commit(),
            action_space_version=ACTION_SPACE_VERSION,
            encoder_version=ENCODER_VERSION,
            model_checkpoint_id=config.model_checkpoint_id,
            generation_settings=config.to_dict(),
            sample_count=sample_count,
            game_count=config.games,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable metadata dictionary."""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "engine_version": self.engine_version,
            "git_commit": self.git_commit,
            "action_space_version": self.action_space_version,
            "encoder_version": self.encoder_version,
            "model_checkpoint_id": self.model_checkpoint_id,
            "generation_settings": self.generation_settings,
            "sample_count": self.sample_count,
            "game_count": self.game_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SelfPlayMetadata:
        """Parse and validate dataset metadata."""
        schema_version = _expect_str(data, "schema_version")
        if schema_version != SELF_PLAY_DATASET_SCHEMA_VERSION:
            raise ValueError(f"unsupported self-play dataset schema: {schema_version}")
        action_space_version = _expect_str(data, "action_space_version")
        if action_space_version != ACTION_SPACE_VERSION:
            raise ValueError(
                f"unsupported action space version: {action_space_version}"
            )
        encoder_version = _expect_str(data, "encoder_version")
        if encoder_version != ENCODER_VERSION:
            raise ValueError(f"unsupported encoder version: {encoder_version}")
        settings = data.get("generation_settings")
        if not isinstance(settings, dict):
            raise TypeError("metadata field 'generation_settings' must be an object")
        git_commit = data.get("git_commit")
        if git_commit is not None and not isinstance(git_commit, str):
            raise TypeError("metadata field 'git_commit' must be a string or null")
        model_checkpoint_id = data.get("model_checkpoint_id")
        if model_checkpoint_id is not None and not isinstance(model_checkpoint_id, str):
            raise TypeError(
                "metadata field 'model_checkpoint_id' must be a string or null"
            )
        return cls(
            schema_version=schema_version,
            generated_at=_expect_str(data, "generated_at"),
            engine_version=_expect_str(data, "engine_version"),
            git_commit=git_commit,
            action_space_version=action_space_version,
            encoder_version=encoder_version,
            model_checkpoint_id=model_checkpoint_id,
            generation_settings=dict(settings),
            sample_count=_expect_int(data, "sample_count"),
            game_count=_expect_int(data, "game_count"),
        )


@dataclass(frozen=True, slots=True)
class SelfPlayDataset:
    """In-memory self-play samples plus metadata."""

    positions: npt.NDArray[np.float32]
    legal_masks: npt.NDArray[np.float32]
    mcts_policies: npt.NDArray[np.float32]
    outcomes: npt.NDArray[np.float32]
    metadata: SelfPlayMetadata
    games: list[SelfPlayGameRecord]


def generate_self_play_dataset(
    inference: NeuralInference | None,
    config: SelfPlayConfig | None = None,
) -> SelfPlayDataset:
    """Generate a small self-play dataset using neural or classical MCTS labels."""
    resolved = SelfPlayConfig() if config is None else config
    positions: list[npt.NDArray[np.float32]] = []
    legal_masks: list[npt.NDArray[np.float32]] = []
    policies: list[npt.NDArray[np.float32]] = []
    outcome_values: list[float] = []
    game_records: list[SelfPlayGameRecord] = []

    for game_index in range(resolved.games):
        game = Game.new()
        player = _player_for_game(inference, resolved, game_index)
        game_sides: list[Color] = []
        for _ply in range(resolved.max_plies):
            if game.outcome is not None:
                break
            if not game.legal_moves:
                break
            result = player.search(game)
            positions.append(np.asarray(encode_game(game), dtype=np.float32))
            legal_masks.append(np.asarray(legal_move_mask(game), dtype=np.float32))
            policies.append(_policy_target(game, result.visit_counts, result.move))
            game_sides.append(game.board.side_to_move)
            game = game.play(result.move)
        if game.outcome is None:
            game = _with_max_plies_outcome(game)
        outcome_values.extend(_outcome_values(game, game_sides))
        game_records.append(_game_record(game_index, game))

    outcomes = np.asarray(outcome_values, dtype=np.float32)
    metadata = SelfPlayMetadata.create(resolved, sample_count=len(positions))
    return SelfPlayDataset(
        positions=_stack_or_empty(positions, TENSOR_SHAPE),
        legal_masks=_stack_or_empty(legal_masks, (ACTION_SPACE_SIZE,)),
        mcts_policies=_stack_or_empty(policies, (ACTION_SPACE_SIZE,)),
        outcomes=outcomes,
        metadata=metadata,
        games=game_records,
    )


def merge_self_play_datasets(
    datasets: list[SelfPlayDataset],
    *,
    config: SelfPlayConfig | None = None,
    generation_settings_extra: dict[str, object] | None = None,
) -> SelfPlayDataset:
    """Merge self-play dataset shards into one dataset with contiguous game indexes."""
    if not datasets:
        raise ValueError("at least one self-play dataset is required")

    first = datasets[0]
    model_checkpoint_id = first.metadata.model_checkpoint_id

    games: list[SelfPlayGameRecord] = []
    for dataset in datasets:
        _validate_dataset_counts(dataset)
        if dataset.metadata.schema_version != SELF_PLAY_DATASET_SCHEMA_VERSION:
            schema = dataset.metadata.schema_version
            raise ValueError(f"unsupported self-play dataset schema: {schema}")
        if dataset.metadata.action_space_version != ACTION_SPACE_VERSION:
            action_space = dataset.metadata.action_space_version
            raise ValueError(f"unsupported action space version: {action_space}")
        if dataset.metadata.encoder_version != ENCODER_VERSION:
            raise ValueError(
                f"unsupported encoder version: {dataset.metadata.encoder_version}"
            )
        if dataset.metadata.model_checkpoint_id != model_checkpoint_id:
            raise ValueError("cannot merge datasets from different model checkpoints")
        for record in dataset.games:
            games.append(replace(record, game_index=len(games)))

    positions = np.concatenate([dataset.positions for dataset in datasets], axis=0)
    legal_masks = np.concatenate([dataset.legal_masks for dataset in datasets], axis=0)
    policies = np.concatenate([dataset.mcts_policies for dataset in datasets], axis=0)
    outcomes = np.concatenate([dataset.outcomes for dataset in datasets], axis=0)

    if config is None:
        generation_settings: dict[str, object] = {
            "merged_from": len(datasets),
            "source_generation_settings": [
                dataset.metadata.generation_settings for dataset in datasets
            ],
            **(generation_settings_extra or {}),
        }
        metadata = SelfPlayMetadata(
            schema_version=SELF_PLAY_DATASET_SCHEMA_VERSION,
            generated_at=datetime.now(UTC).isoformat(),
            engine_version=tinychess.__version__,
            git_commit=_git_commit(),
            action_space_version=ACTION_SPACE_VERSION,
            encoder_version=ENCODER_VERSION,
            model_checkpoint_id=model_checkpoint_id,
            generation_settings=generation_settings,
            sample_count=int(outcomes.shape[0]),
            game_count=len(games),
        )
    else:
        metadata = SelfPlayMetadata.create(config, sample_count=int(outcomes.shape[0]))
        if generation_settings_extra:
            metadata = replace(
                metadata,
                generation_settings={
                    **metadata.generation_settings,
                    **generation_settings_extra,
                },
            )

    return SelfPlayDataset(
        positions=positions,
        legal_masks=legal_masks,
        mcts_policies=policies,
        outcomes=outcomes,
        metadata=metadata,
        games=games,
    )


def save_self_play_dataset(dataset: SelfPlayDataset, directory: str | Path) -> None:
    """Write a self-play dataset as compressed NPZ tensors plus JSON/JSONL metadata."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / DEFAULT_DATASET_FILENAME,
        positions=dataset.positions,
        legal_masks=dataset.legal_masks,
        mcts_policies=dataset.mcts_policies,
        outcomes=dataset.outcomes,
    )
    (output_dir / DEFAULT_METADATA_FILENAME).write_text(
        json.dumps(dataset.metadata.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / DEFAULT_GAMES_FILENAME).write_text(
        "".join(
            json.dumps(record.to_dict(), sort_keys=True) + "\n"
            for record in dataset.games
        )
    )


def load_self_play_dataset(directory: str | Path) -> SelfPlayDataset:
    """Load and validate a self-play dataset from disk."""
    input_dir = Path(directory)
    metadata_data = json.loads((input_dir / DEFAULT_METADATA_FILENAME).read_text())
    if not isinstance(metadata_data, dict):
        raise TypeError("self-play metadata must be a JSON object")
    metadata = SelfPlayMetadata.from_dict(metadata_data)
    with np.load(input_dir / DEFAULT_DATASET_FILENAME) as tensors:
        positions = np.asarray(tensors["positions"], dtype=np.float32)
        legal_masks = np.asarray(tensors["legal_masks"], dtype=np.float32)
        mcts_policies = np.asarray(tensors["mcts_policies"], dtype=np.float32)
        outcomes = np.asarray(tensors["outcomes"], dtype=np.float32)
    _validate_tensor_shapes(metadata, positions, legal_masks, mcts_policies, outcomes)
    games = [
        SelfPlayGameRecord.from_dict(record)
        for record in _read_jsonl(input_dir / DEFAULT_GAMES_FILENAME)
    ]
    if len(games) != metadata.game_count:
        raise ValueError("game metadata count does not match dataset metadata")
    return SelfPlayDataset(
        positions=positions,
        legal_masks=legal_masks,
        mcts_policies=mcts_policies,
        outcomes=outcomes,
        metadata=metadata,
        games=games,
    )


def _validate_dataset_counts(dataset: SelfPlayDataset) -> None:
    expected = dataset.metadata.sample_count
    if dataset.positions.shape[0] != expected:
        raise ValueError("positions sample count does not match dataset metadata")
    if dataset.legal_masks.shape[0] != expected:
        raise ValueError("legal_masks sample count does not match dataset metadata")
    if dataset.mcts_policies.shape[0] != expected:
        raise ValueError("mcts_policies sample count does not match dataset metadata")
    if dataset.outcomes.shape[0] != expected:
        raise ValueError("outcomes sample count does not match dataset metadata")
    if len(dataset.games) != dataset.metadata.game_count:
        raise ValueError("game count does not match dataset metadata")


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


def _outcome_values(game: Game, sides: list[Color]) -> list[float]:
    outcome = game.outcome
    if outcome is None or outcome.winner is None:
        return [0.0 for _side in sides]
    return [1.0 if outcome.winner is side else -1.0 for side in sides]


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


def _validate_tensor_shapes(
    metadata: SelfPlayMetadata,
    positions: npt.NDArray[np.float32],
    legal_masks: npt.NDArray[np.float32],
    mcts_policies: npt.NDArray[np.float32],
    outcomes: npt.NDArray[np.float32],
) -> None:
    expected = metadata.sample_count
    if positions.shape != (expected, *TENSOR_SHAPE):
        raise ValueError(f"positions shape mismatch: {positions.shape}")
    if legal_masks.shape != (expected, ACTION_SPACE_SIZE):
        raise ValueError(f"legal_masks shape mismatch: {legal_masks.shape}")
    if mcts_policies.shape != (expected, ACTION_SPACE_SIZE):
        raise ValueError(f"mcts_policies shape mismatch: {mcts_policies.shape}")
    if outcomes.shape != (expected,):
        raise ValueError(f"outcomes shape mismatch: {outcomes.shape}")


def _validate_game_records(
    metadata: SelfPlayMetadata,
    games: list[SelfPlayGameRecord],
    positions: npt.NDArray[np.float32],
    legal_masks: npt.NDArray[np.float32],
    mcts_policies: npt.NDArray[np.float32],
    outcomes: npt.NDArray[np.float32],
) -> None:
    sample_index = 0
    for expected_game_index, record in enumerate(games):
        if record.game_index != expected_game_index:
            raise ValueError("game_index values must be contiguous starting at 0")
        if record.plies != len(record.moves_uci):
            raise ValueError("game record plies must match moves_uci length")
        game = Game.new()
        sides: list[Color] = []
        for move_uci in record.moves_uci:
            if sample_index >= metadata.sample_count:
                raise ValueError("game records contain more plies than tensor samples")
            expected_position = np.asarray(encode_game(game), dtype=np.float32)
            if not np.allclose(positions[sample_index], expected_position):
                raise ValueError("position tensor does not match replayed game state")
            expected_mask = np.asarray(legal_move_mask(game), dtype=np.float32)
            if not np.array_equal(legal_masks[sample_index], expected_mask):
                raise ValueError("legal mask does not match replayed game state")
            _validate_policy_row(mcts_policies[sample_index], expected_mask)
            move = Move.from_uci(move_uci)
            if move not in game.legal_moves:
                raise ValueError(f"illegal move in game record: {move_uci}")
            sides.append(game.board.side_to_move)
            game = game.play(move)
            sample_index += 1
        if game.to_fen() != record.final_fen:
            raise ValueError("game record final_fen does not match replayed moves")
        _validate_recorded_outcome(record, game)
        expected_game = _game_with_recorded_outcome(record, game)
        expected_outcomes = np.asarray(_outcome_values(expected_game, sides))
        start = sample_index - record.plies
        if not np.allclose(outcomes[start:sample_index], expected_outcomes):
            raise ValueError("outcome targets do not match recorded game outcome")
    if sample_index != metadata.sample_count:
        raise ValueError("total game plies does not match metadata sample_count")


def _validate_policy_row(
    policy: npt.NDArray[np.float32],
    legal_mask: npt.NDArray[np.float32],
) -> None:
    if not np.all(np.isfinite(policy)):
        raise ValueError("policy target contains non-finite values")
    if np.any(policy < 0.0):
        raise ValueError("policy target contains negative values")
    if not np.isclose(float(policy.sum()), 1.0):
        raise ValueError("policy target row must sum to 1.0")
    if np.any((policy > 0.0) & (legal_mask <= 0.0)):
        raise ValueError("policy target assigns probability to illegal moves")


def _validate_recorded_outcome(record: SelfPlayGameRecord, game: Game) -> None:
    recorded = _recorded_outcome(record)
    actual = game.outcome
    if actual is None and recorded.reason is not OutcomeReason.MAX_PLIES:
        raise ValueError("non-terminal replay must be recorded as max_plies")
    if actual is not None and actual != recorded:
        raise ValueError("game record outcome does not match replayed game outcome")


def _game_with_recorded_outcome(record: SelfPlayGameRecord, game: Game) -> Game:
    recorded = _recorded_outcome(record)
    if game.outcome == recorded:
        return game
    return Game(
        positions=game.positions,
        moves=game.moves,
        halfmove_clock=game.halfmove_clock,
        fullmove_number=game.fullmove_number,
        repetition_counts=dict(game.repetition_counts),
        forced_outcome=recorded,
    )


def _recorded_outcome(record: SelfPlayGameRecord) -> Outcome:
    try:
        reason = OutcomeReason(record.outcome_reason)
    except ValueError as exc:
        raise ValueError(
            f"unsupported game outcome reason: {record.outcome_reason}"
        ) from exc
    winner = None
    if record.winner is not None:
        try:
            winner = Color(record.winner)
        except ValueError as exc:
            raise ValueError(f"unsupported game winner: {record.winner}") from exc
    return Outcome(reason=reason, winner=winner)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError("game metadata JSONL records must be objects")
        records.append(record)
    return records


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _expect_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"field {key!r} must be a string")
    return value


def _expect_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"field {key!r} must be an integer")
    return value


__all__ = [
    "DEFAULT_DATASET_FILENAME",
    "DEFAULT_GAMES_FILENAME",
    "DEFAULT_METADATA_FILENAME",
    "LABEL_SOURCE_CLASSICAL",
    "LABEL_SOURCE_NEURAL",
    "LABEL_SOURCES",
    "SELF_PLAY_DATASET_SCHEMA_VERSION",
    "SelfPlayConfig",
    "SelfPlayDataset",
    "SelfPlayGameRecord",
    "SelfPlayMetadata",
    "generate_self_play_dataset",
    "load_self_play_dataset",
    "merge_self_play_datasets",
    "save_self_play_dataset",
]
