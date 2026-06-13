from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import vibechess.nn.train as train_module
from vibechess.engine.game import Game
from vibechess.engine.legal_moves import legal_moves
from vibechess.engine.move import Move
from vibechess.engine.pgn import PgnGameTrace, PgnParsedPly
from vibechess.engine.pgn_stream import (
    iter_pgn_records,
    parse_ingest_pgn,
    parse_ingest_pgn_with_trace,
)
from vibechess.engine.piece import Color
from vibechess.nn.checkpoint import (
    DEFAULT_WEIGHTS_FILENAME,
    load_checkpoint,
    load_checkpoint_metadata,
)
from vibechess.nn.encode import (
    ACTION_SPACE_SIZE,
    encode_game_np,
    legal_move_mask_from_legal_moves_np,
    move_to_action_index,
)
from vibechess.nn.pgn_dataset import (
    DEFAULT_MANIFEST_FILENAME,
    SUPPORTED_PGN_RESULTS,
    PgnIngestConfig,
    PgnIngestProgress,
    _TrainingReplayState,
    ingest_pgn_dataset,
    shard_directories,
)
from vibechess.nn.self_play_dataset import load_self_play_dataset
from vibechess.nn.train import TrainingConfig, train_from_directory, train_from_sharded_directory

PGN_TEXT = """[Event "Tiny"]
[Result "1-0"]

1. e4 e5 2. Nf3 1-0

[Event "Fen"]
[SetUp "1"]
[FEN "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"]
[Result "*"]

1. a8=Q+ *

[Event "Annotated"]
[Result "0-1"]

1. d4! {good} d5?! (1... Nf6) 2. c4 0-1

[Event "Unfinished"]
[Result "*"]

1. e4 e5 *
"""


def test_ingest_pgn_dataset_writes_loadable_shards_and_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "dataset"
    input_path.write_text(PGN_TEXT)

    result = ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=output_dir, shard_samples=3)
    )

    manifest = json.loads((output_dir / DEFAULT_MANIFEST_FILENAME).read_text())
    assert result.games_read == 4
    assert result.games_written == 2
    assert result.games_skipped == 2
    assert result.samples == 6
    assert manifest["shard_count"] == 2
    assert "git_commit" in manifest

    with np.load(output_dir / "shard-00000" / "samples.npz") as tensors:
        assert "mcts_policies" not in tensors.files
        assert "policy_offsets" in tensors.files
        assert "policy_indices" in tensors.files
        assert "policy_probabilities" in tensors.files
    first = load_self_play_dataset(output_dir / "shard-00000")
    second = load_self_play_dataset(output_dir / "shard-00001")
    assert first.metadata.generation_settings["label_source"] == "pgn"
    assert first.metadata.sample_count == 3
    assert second.metadata.sample_count == 3
    assert np.allclose(first.mcts_policies.sum(axis=1), 1.0)
    assert first.outcomes.tolist() == [1.0, -1.0, 1.0]
    assert second.outcomes.tolist() == [-1.0, 1.0, -1.0]


def test_ingest_pgn_dataset_trace_path_matches_legacy_replay_arrays(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "dataset"
    input_path.write_text(PGN_TEXT)

    result = ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=output_dir, shard_samples=16)
    )
    dataset = load_self_play_dataset(shard_directories(output_dir)[0])
    expected = _legacy_replay_arrays(input_path)

    assert result.games_written == 2
    assert np.array_equal(dataset.positions, expected["positions"])
    assert np.array_equal(dataset.legal_masks, expected["legal_masks"])
    assert np.array_equal(dataset.mcts_policies, expected["mcts_policies"])
    assert np.array_equal(dataset.outcomes, expected["outcomes"])


@pytest.mark.parametrize(
    "pgn_text",
    [
        """[Event "Normal"]
[Result "1-0"]

1. e4 e5 2. Nf3 1-0
""",
        """[Event "Mate"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1
""",
        """[Event "Castle"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
""",
        """[Event "EnPassant"]
[Result "1-0"]

1. e4 a6 2. e5 d5 3. exd6 1-0
""",
    ],
)
def test_ingest_pgn_dataset_lightweight_replay_matches_game_play_reference(
    tmp_path: Path,
    pgn_text: str,
) -> None:
    input_path = tmp_path / "game.pgn"
    output_dir = tmp_path / "dataset"
    input_path.write_text(pgn_text)

    result = ingest_pgn_dataset(PgnIngestConfig(input_path=input_path, output_dir=output_dir))
    dataset = load_self_play_dataset(output_dir / "shard-00000")
    pgn = parse_ingest_pgn(pgn_text)
    reference = pgn.final_game
    record = dataset.games[0]

    assert result.games_written == 1
    assert result.samples == len(pgn.moves)
    assert dataset.metadata.sample_count == len(pgn.moves)
    assert record.moves_uci == [move.to_uci() for move in pgn.moves]
    assert record.final_fen == reference.to_fen()
    if reference.outcome is None:
        assert record.outcome_reason == "max_plies"
    else:
        assert reference.outcome.winner is not None
        assert record.outcome_reason == reference.outcome.reason.value
        assert record.winner == reference.outcome.winner.value


def test_training_replay_state_matches_game_play_for_promotion() -> None:
    game = Game.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    state = _TrainingReplayState.from_game(game)
    move = Move.from_uci("a7a8q")

    state.advance(move)
    game = game.play(move)

    assert state.to_fen() == game.to_fen()
    assert state.moves == list(game.moves)
    assert state.to_outcome_game().outcome == game.outcome
    assert np.array_equal(state.encode_position(), encode_game_np(game))


def test_training_replay_state_matches_game_play_for_en_passant() -> None:
    game = Game.new()
    for notation in ("e2e4", "a7a6", "e4e5", "d7d5"):
        game = game.play(Move.from_uci(notation))
    state = _TrainingReplayState.from_game(game)
    move = Move.from_uci("e5d6")

    state.advance(move)
    game = game.play(move)

    assert state.to_fen() == game.to_fen()
    assert state.moves == list(game.moves)
    assert state.to_outcome_game().outcome == game.outcome
    assert np.array_equal(state.encode_position(), encode_game_np(game))


def test_training_replay_state_matches_game_play_for_black_quiet_fullmove_increment() -> None:
    game = Game.from_fen("r3k3/8/8/8/8/8/8/4K3 b q - 7 12")
    state = _TrainingReplayState.from_game(game)
    move = Move.from_uci("a8a7")

    state.advance(move)
    game = game.play(move)

    assert state.to_fen() == game.to_fen()
    assert state.halfmove_clock == 8
    assert state.fullmove_number == 13
    assert state.moves == list(game.moves)
    assert state.to_outcome_game().outcome == game.outcome
    assert np.array_equal(state.encode_position(), encode_game_np(game))


def test_shard_builder_rejects_bad_trace_without_partial_samples(tmp_path: Path) -> None:
    from vibechess.nn.pgn_dataset import _ShardBuilder

    traced = parse_ingest_pgn_with_trace(
        """[Event "BadTrace"]
[Result "1-0"]

1. e4 e5 1-0
"""
    )
    bad_second_ply = PgnParsedPly(
        board=traced.plies[1].board,
        halfmove_clock=traced.plies[1].halfmove_clock,
        fullmove_number=traced.plies[1].fullmove_number,
        move=traced.plies[1].move,
        legal_moves=(),
    )
    bad_trace = PgnGameTrace(game=traced.game, plies=(traced.plies[0], bad_second_ply))
    builder = _ShardBuilder(PgnIngestConfig(input_path=tmp_path / "games.pgn", output_dir=tmp_path))

    with pytest.raises(ValueError, match="trace move is not legal"):
        builder.add_game(bad_trace)

    assert builder.sample_count == 0
    assert builder.outcomes == []
    assert builder.games == []


def test_ingest_pgn_dataset_records_repetition_outcome(tmp_path: Path) -> None:
    input_path = tmp_path / "repetition.pgn"
    output_dir = tmp_path / "dataset"
    input_path.write_text(
        """[Event "Repetition"]
[Result "1/2-1/2"]

1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 1/2-1/2
"""
    )

    result = ingest_pgn_dataset(PgnIngestConfig(input_path=input_path, output_dir=output_dir))
    dataset = load_self_play_dataset(output_dir / "shard-00000")

    assert result.games_written == 1
    assert dataset.games[0].outcome_reason == "repetition"
    assert dataset.games[0].winner is None


def test_training_replay_state_matches_game_play_for_fifty_move_outcome() -> None:
    game = Game.from_fen("4k3/8/8/8/8/8/6N1/R3K3 w - - 99 1")
    state = _TrainingReplayState.from_game(game)
    move = Move.from_uci("g2f4")

    state.advance(move)
    game = game.play(move)

    assert state.to_fen() == game.to_fen()
    assert state.moves == list(game.moves)
    assert state.to_outcome_game().outcome == game.outcome


def test_ingest_pgn_dataset_skips_empty_games_without_polluting_shards(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "dataset"
    input_path.write_text(
        """[Event "Empty"]
[Result "1-0"]

1-0

[Event "Tiny"]
[Result "1-0"]

1. e4 e5 1-0
"""
    )

    result = ingest_pgn_dataset(PgnIngestConfig(input_path=input_path, output_dir=output_dir))

    dataset = load_self_play_dataset(output_dir / "shard-00000")
    assert result.games_read == 2
    assert result.games_written == 1
    assert result.games_skipped == 1
    assert dataset.metadata.game_count == 1
    assert [record.game_index for record in dataset.games] == [0]
    assert len(dataset.games) == 1
    assert dataset.games[0].moves_uci == ["e2e4", "e7e5"]


def test_ingest_pgn_dataset_reports_progress(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "dataset"
    input_path.write_text(PGN_TEXT)
    updates: list[PgnIngestProgress] = []

    result = ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=output_dir, shard_samples=3),
        progress=updates.append,
        progress_every_games=1,
    )

    assert updates
    assert updates[-1].games_read == result.games_read
    assert updates[-1].games_written == result.games_written
    assert updates[-1].games_skipped == result.games_skipped
    assert updates[-1].samples == result.samples
    assert updates[-1].shards == result.shards


def test_ingest_pgn_dataset_progress_interval_emits_periodic_and_final(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "dataset"
    input_path.write_text(
        """[Event "One"]
[Result "1-0"]

1. e4 e5 1-0

[Event "Two"]
[Result "1-0"]

1. d4 d5 1-0

[Event "Three"]
[Result "1-0"]

1. c4 c5 1-0
"""
    )
    updates: list[PgnIngestProgress] = []

    result = ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=output_dir),
        progress=updates.append,
        progress_every_games=2,
    )

    assert result.games_written == 3
    assert [update.games_written for update in updates] == [2, 3]
    assert updates[-1].samples == result.samples


def _legacy_replay_arrays(input_path: Path) -> dict[str, npt.NDArray[np.float32]]:
    positions: list[npt.NDArray[np.float32]] = []
    legal_masks: list[npt.NDArray[np.float32]] = []
    policies: list[npt.NDArray[np.float32]] = []
    outcomes: list[float] = []
    starting_fen = Game.new().to_fen()

    for record in iter_pgn_records(input_path):
        try:
            pgn = parse_ingest_pgn(record.text)
        except ValueError:
            continue
        if pgn.initial_game.to_fen() != starting_fen or pgn.result not in SUPPORTED_PGN_RESULTS:
            continue
        if not pgn.moves:
            continue

        game = pgn.initial_game
        sides: list[Color] = []
        for move in pgn.moves:
            legal = legal_moves(game.board)
            if move not in legal:
                raise AssertionError("legacy replay found illegal parsed move")
            positions.append(encode_game_np(game))
            legal_masks.append(legal_move_mask_from_legal_moves_np(game, legal))
            policy = np.zeros((ACTION_SPACE_SIZE,), dtype=np.float32)
            policy[move_to_action_index(move, game.board)] = 1.0
            policies.append(policy)
            sides.append(game.board.side_to_move)
            game = game.play(move)
        outcomes.extend(_legacy_result_values(pgn.result, sides))

    return {
        "positions": np.stack(positions).astype(np.float32, copy=False),
        "legal_masks": np.stack(legal_masks).astype(np.float32, copy=False),
        "mcts_policies": np.stack(policies).astype(np.float32, copy=False),
        "outcomes": np.asarray(outcomes, dtype=np.float32),
    }


def _legacy_result_values(result: str, sides: list[Color]) -> list[float]:
    if result == "1/2-1/2":
        return [0.0 for _side in sides]
    winner = Color.WHITE if result == "1-0" else Color.BLACK
    return [1.0 if side is winner else -1.0 for side in sides]


def test_pgn_ingest_config_rejects_fen_ingestion() -> None:
    try:
        PgnIngestConfig(input_path=Path("games.pgn"), output_dir=Path("dataset"), skip_fen=False)
    except ValueError as exc:
        assert "FEN/SetUp" in str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("expected FEN ingestion to be rejected")


def test_pgn_ingest_script_creates_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "script-dataset"
    input_path.write_text(PGN_TEXT)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/pgn_ingest.py",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--max-games",
            "1",
            "--shard-samples",
            "2",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "games_written=1" in result.stdout
    assert "games_written=1" in result.stderr
    assert "samples=" in result.stderr
    assert (output_dir / DEFAULT_MANIFEST_FILENAME).is_file()


def test_pgn_ingest_script_progress_can_be_disabled(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    output_dir = tmp_path / "script-dataset-no-progress"
    input_path.write_text(PGN_TEXT)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/pgn_ingest.py",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--max-games",
            "1",
            "--shard-samples",
            "2",
            "--progress-every-games",
            "0",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "games_written=1" in result.stdout
    assert "games_written=" not in result.stderr
    assert (output_dir / DEFAULT_MANIFEST_FILENAME).is_file()


def test_train_from_directory_auto_detects_pgn_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "train"
    input_path.write_text(PGN_TEXT)
    output_dir.mkdir()
    (output_dir / "metrics.jsonl").write_text("stale\n")
    ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=dataset_dir, shard_samples=3)
    )

    result = train_from_directory(
        dataset_dir,
        output_dir,
        config=TrainingConfig(epochs=1, batch_size=2, learning_rate=1.0e-3),
    )

    assert result.steps == 2
    assert result.samples == 6
    assert result.training_samples == 4
    assert result.validation_samples == 2
    assert result.final_training_step == 2
    assert (output_dir / "checkpoint-final" / DEFAULT_WEIGHTS_FILENAME).is_file()
    assert not (output_dir / "metrics.jsonl").exists()
    assert (output_dir / "epoch_metrics.jsonl").read_text().count("\n") == 2
    assert (
        output_dir / "shard-train-00000" / "checkpoint-final" / DEFAULT_WEIGHTS_FILENAME
    ).is_file()
    assert (
        output_dir / "shard-train-00001" / "checkpoint-final" / DEFAULT_WEIGHTS_FILENAME
    ).is_file()
    assert load_checkpoint_metadata(output_dir / "checkpoint-final").training_step == 2


def test_sharded_training_summary_records_initial_training_step(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "train"
    input_path.write_text(PGN_TEXT)
    ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=dataset_dir, shard_samples=3)
    )

    result = train_from_sharded_directory(
        dataset_dir,
        output_dir,
        config=TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1.0e-3,
            validation_fraction=0.0,
        ),
        initial_step=7,
    )

    training_summary = json.loads((output_dir / "training.json").read_text())

    assert result.steps == 4
    assert result.final_training_step == 11
    assert training_summary["initial_training_step"] == 7
    assert training_summary["final_training_step"] == 11
    assert "final_metrics" not in training_summary
    assert load_checkpoint_metadata(output_dir / "checkpoint-final").training_step == 11


def test_sharded_training_optimizer_state_carry_is_enabled_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "games.pgn"
    dataset_dir = tmp_path / "dataset"
    default_output = tmp_path / "train-default"
    carry_output = tmp_path / "train-carry"
    input_path.write_text(PGN_TEXT)
    ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=dataset_dir, shard_samples=3)
    )

    optimizer_module = vars(train_module)["optim"]
    original_adam = optimizer_module.Adam
    constructor_calls = 0

    def counted_adam(*args: object, **kwargs: object) -> object:
        nonlocal constructor_calls
        constructor_calls += 1
        return original_adam(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "Adam", counted_adam)

    default_result = train_from_directory(
        dataset_dir,
        default_output,
        config=TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1.0e-3,
            validation_fraction=0.0,
        ),
    )
    default_metadata = load_checkpoint_metadata(default_output / "checkpoint-final")
    default_summary = json.loads((default_output / "training.json").read_text())

    assert default_result.steps == 4
    assert constructor_calls == 1
    assert default_result.final_training_step == 4
    assert default_metadata.training_step == 4
    assert default_metadata.optimizer_state_available is False
    assert default_summary["training_config"]["carry_optimizer_state_across_shards"] is True

    constructor_calls = 0
    reset_result = train_from_directory(
        dataset_dir,
        carry_output,
        config=TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1.0e-3,
            validation_fraction=0.0,
            carry_optimizer_state_across_shards=False,
        ),
    )
    reset_summary = json.loads((carry_output / "training.json").read_text())

    assert reset_result.steps == 4
    assert constructor_calls == 2
    assert reset_summary["training_config"]["carry_optimizer_state_across_shards"] is False


def test_sharded_training_can_skip_per_shard_checkpoints(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "train"
    input_path.write_text(PGN_TEXT)
    ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=dataset_dir, shard_samples=3)
    )

    result = train_from_directory(
        dataset_dir,
        output_dir,
        config=TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1.0e-3,
            write_shard_checkpoints=False,
        ),
    )

    training_summary = json.loads((output_dir / "training.json").read_text())
    loaded = load_checkpoint(output_dir / "checkpoint-final")

    assert result.steps == 2
    assert result.final_training_step == 2
    assert (output_dir / "checkpoint-final" / DEFAULT_WEIGHTS_FILENAME).is_file()
    assert loaded.metadata.training_step == 2
    assert loaded.metadata.optimizer_state_available is False
    assert not (output_dir / "shard-train-00000" / "checkpoint-final").exists()
    assert not (output_dir / "shard-train-00001" / "checkpoint-final").exists()
    assert not (output_dir / "metrics.jsonl").exists()
    assert (output_dir / "epoch_metrics.jsonl").read_text().count("\n") == 2
    assert training_summary["training_config"]["write_shard_checkpoints"] is False
    assert [shard["checkpoint_written"] for shard in training_summary["shards"]] == [
        False,
        False,
    ]


def test_train_script_consumes_pgn_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "games.pgn"
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "train-script"
    input_path.write_text(PGN_TEXT)
    ingest_pgn_dataset(
        PgnIngestConfig(input_path=input_path, output_dir=dataset_dir, shard_samples=3)
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--dataset",
            str(dataset_dir),
            "--output",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--residual-channels",
            "8",
            "--residual-blocks",
            "1",
            "--policy-channels",
            "2",
            "--value-channels",
            "1",
            "--value-hidden-dim",
            "8",
            "--skip-shard-checkpoints",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "training complete" in result.stdout
    assert "samples=6" in result.stdout
    assert "validation_loss=" in result.stdout
    training_summary = json.loads((output_dir / "training.json").read_text())

    assert (output_dir / "checkpoint-final" / DEFAULT_WEIGHTS_FILENAME).is_file()
    assert not (output_dir / "shard-train-00000" / "checkpoint-final").exists()
    assert not (output_dir / "shard-train-00001" / "checkpoint-final").exists()
    assert training_summary["training_config"]["carry_optimizer_state_across_shards"] is True
