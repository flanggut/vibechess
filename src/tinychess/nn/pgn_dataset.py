"""Convert external PGN games into tinychess policy/value dataset shards."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import tinychess
from tinychess.engine.game import Game
from tinychess.engine.outcome import OutcomeReason
from tinychess.engine.pgn import PgnGame
from tinychess.engine.pgn_stream import iter_pgn_records, parse_ingest_pgn, pgn_has_fen_setup
from tinychess.engine.piece import Color
from tinychess.nn.encode import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    ENCODER_VERSION,
    encode_game,
    legal_move_mask_from_legal_moves,
    move_to_action_index,
)
from tinychess.nn.self_play import (
    SelfPlayConfig,
    SelfPlayDataset,
    SelfPlayGameRecord,
    SelfPlayMetadata,
    save_self_play_dataset,
)

PGN_DATASET_MANIFEST_SCHEMA_VERSION = "tinychess-pgn-manifest-v1"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
PGN_LABEL_SOURCE = "pgn"
SUPPORTED_PGN_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2"})


@dataclass(frozen=True, slots=True)
class PgnIngestConfig:
    """Settings for converting PGN records into dataset shards."""

    input_path: Path
    output_dir: Path
    max_games: int | None = None
    shard_samples: int = 50_000
    strict: bool = False
    skip_fen: bool = True

    def __post_init__(self) -> None:
        if self.max_games is not None and self.max_games < 1:
            raise ValueError("max_games must be positive when provided")
        if self.shard_samples < 1:
            raise ValueError("shard_samples must be at least 1")
        if not self.skip_fen:
            raise ValueError("FEN/SetUp PGN ingestion is not supported yet")


@dataclass(frozen=True, slots=True)
class PgnIngestResult:
    """Summary of a PGN ingestion run."""

    output_dir: Path
    manifest_path: Path
    shards: int
    games_read: int
    games_written: int
    games_skipped: int
    samples: int


@dataclass(frozen=True, slots=True)
class PgnShardInfo:
    """Manifest entry for one dataset shard."""

    path: str
    games: int
    samples: int

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "games": self.games, "samples": self.samples}


def ingest_pgn_dataset(config: PgnIngestConfig) -> PgnIngestResult:
    """Convert PGN games to one or more existing-format dataset shards."""
    output_dir = config.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = _ShardBuilder(config)
    games_read = 0
    games_written = 0
    games_skipped = 0
    shards: list[PgnShardInfo] = []

    for record in iter_pgn_records(config.input_path):
        if config.max_games is not None and games_written >= config.max_games:
            break
        games_read += 1
        if config.skip_fen and pgn_has_fen_setup(record.text):
            games_skipped += 1
            continue
        try:
            pgn = parse_ingest_pgn(record.text, strict=config.strict)
        except ValueError:
            games_skipped += 1
            continue
        if pgn.initial_game.to_fen() != Game.new().to_fen():
            games_skipped += 1
            continue
        if pgn.result not in SUPPORTED_PGN_RESULTS:
            games_skipped += 1
            continue
        try:
            builder.add_game(pgn)
        except ValueError:
            games_skipped += 1
            continue
        games_written += 1
        if builder.sample_count >= config.shard_samples:
            shard = builder.flush(output_dir, len(shards))
            if shard is not None:
                shards.append(shard)

    final = builder.flush(output_dir, len(shards))
    if final is not None:
        shards.append(final)

    manifest_path = output_dir / DEFAULT_MANIFEST_FILENAME
    samples = sum(shard.samples for shard in shards)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": PGN_DATASET_MANIFEST_SCHEMA_VERSION,
                "generated_at": datetime.now(UTC).isoformat(),
                "engine_version": tinychess.__version__,
                "git_commit": _git_commit(),
                "input_path": str(config.input_path.expanduser()),
                "action_space_version": ACTION_SPACE_VERSION,
                "encoder_version": ENCODER_VERSION,
                "generation_settings": {
                    "label_source": PGN_LABEL_SOURCE,
                    "max_games": config.max_games,
                    "shard_samples": config.shard_samples,
                    "strict": config.strict,
                    "skip_fen": config.skip_fen,
                },
                "games_read": games_read,
                "games_written": games_written,
                "games_skipped": games_skipped,
                "sample_count": samples,
                "shard_count": len(shards),
                "shards": [shard.to_dict() for shard in shards],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return PgnIngestResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        shards=len(shards),
        games_read=games_read,
        games_written=games_written,
        games_skipped=games_skipped,
        samples=samples,
    )


def load_pgn_manifest(directory: str | Path) -> dict[str, object]:
    """Load and validate a PGN ingestion manifest."""
    path = Path(directory) / DEFAULT_MANIFEST_FILENAME
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("PGN manifest must be a JSON object")
    if data.get("schema_version") != PGN_DATASET_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported PGN manifest schema")
    shards = data.get("shards")
    if not isinstance(shards, list):
        raise TypeError("PGN manifest field 'shards' must be a list")
    return data


def shard_directories(directory: str | Path) -> list[Path]:
    """Return dataset shard directories from a PGN ingestion manifest."""
    root = Path(directory)
    manifest = load_pgn_manifest(root)
    paths: list[Path] = []
    shards = manifest["shards"]
    if not isinstance(shards, list):
        raise TypeError("PGN manifest field 'shards' must be a list")
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
            raise TypeError("PGN manifest shard entries must contain a string path")
        paths.append(root / shard["path"])
    return paths


class _ShardBuilder:
    def __init__(self, config: PgnIngestConfig) -> None:
        self.config = config
        self.positions: list[npt.NDArray[np.float32]] = []
        self.legal_masks: list[npt.NDArray[np.float32]] = []
        self.policies: list[npt.NDArray[np.float32]] = []
        self.outcomes: list[float] = []
        self.games: list[SelfPlayGameRecord] = []

    @property
    def sample_count(self) -> int:
        return len(self.positions)

    def add_game(self, pgn: PgnGame) -> None:
        game = pgn.initial_game
        sides: list[Color] = []
        start_sample = self.sample_count
        for move in pgn.moves:
            legal = game.legal_moves
            if move not in legal:
                raise ValueError("PGN move is not legal in replayed game")
            self.positions.append(np.asarray(encode_game(game), dtype=np.float32))
            self.legal_masks.append(
                np.asarray(legal_move_mask_from_legal_moves(game, legal), dtype=np.float32)
            )
            self.policies.append(_one_hot_policy(game, move))
            sides.append(game.board.side_to_move)
            game = game.play(move)
        self.outcomes.extend(_result_values(pgn.result, sides))
        self.games.append(_game_record(len(self.games), game, pgn.result))
        if self.sample_count == start_sample:
            raise ValueError("PGN game contains no training samples")

    def flush(self, output_dir: Path, shard_index: int) -> PgnShardInfo | None:
        if self.sample_count == 0:
            return None
        shard_name = f"shard-{shard_index:05d}"
        shard_dir = output_dir / shard_name
        dataset = SelfPlayDataset(
            positions=np.stack(self.positions).astype(np.float32, copy=False),
            legal_masks=np.stack(self.legal_masks).astype(np.float32, copy=False),
            mcts_policies=np.stack(self.policies).astype(np.float32, copy=False),
            outcomes=np.asarray(self.outcomes, dtype=np.float32),
            metadata=_metadata(
                self.config,
                sample_count=self.sample_count,
                game_count=len(self.games),
            ),
            games=self.games,
        )
        save_self_play_dataset(dataset, shard_dir)
        info = PgnShardInfo(path=shard_name, games=len(self.games), samples=self.sample_count)
        self.positions = []
        self.legal_masks = []
        self.policies = []
        self.outcomes = []
        self.games = []
        return info


def _metadata(config: PgnIngestConfig, *, sample_count: int, game_count: int) -> SelfPlayMetadata:
    base = SelfPlayMetadata.create(SelfPlayConfig(games=game_count), sample_count=sample_count)
    return replace(
        base,
        generation_settings={
            "label_source": PGN_LABEL_SOURCE,
            "input_path": str(config.input_path.expanduser()),
            "max_games": config.max_games,
            "shard_samples": config.shard_samples,
            "strict": config.strict,
            "skip_fen": config.skip_fen,
        },
        game_count=game_count,
    )


def _one_hot_policy(game: Game, move: Any) -> npt.NDArray[np.float32]:
    policy = np.zeros((ACTION_SPACE_SIZE,), dtype=np.float32)
    policy[move_to_action_index(move, game.board)] = 1.0
    return policy


def _result_values(result: str, sides: list[Color]) -> list[float]:
    winner = _winner(result)
    if winner is None:
        return [0.0 for _side in sides]
    return [1.0 if side is winner else -1.0 for side in sides]


def _winner(result: str) -> Color | None:
    if result == "1-0":
        return Color.WHITE
    if result == "0-1":
        return Color.BLACK
    return None


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


def _game_record(game_index: int, game: Game, result: str) -> SelfPlayGameRecord:
    actual = game.outcome
    winner = _winner(result)
    reason = actual.reason if actual is not None else OutcomeReason.MAX_PLIES
    if actual is not None:
        winner = actual.winner
    return SelfPlayGameRecord(
        game_index=game_index,
        plies=len(game.moves),
        outcome_reason=reason.value,
        winner=None if winner is None else winner.value,
        final_fen=game.to_fen(),
        moves_uci=[move.to_uci() for move in game.moves],
    )


__all__ = [
    "DEFAULT_MANIFEST_FILENAME",
    "PGN_DATASET_MANIFEST_SCHEMA_VERSION",
    "PGN_LABEL_SOURCE",
    "PgnIngestConfig",
    "PgnIngestResult",
    "ingest_pgn_dataset",
    "load_pgn_manifest",
    "shard_directories",
]
