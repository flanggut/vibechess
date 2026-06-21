from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import pytest

import vibechess.ai.evaluation as evaluation_module
from vibechess.ai import MCTSConfig, NeuralMCTSConfig, RandomPlayer
from vibechess.ai.evaluation import (
    EARLY_PROMOTION_NOTE,
    MatchConfig,
    OpeningConfig,
    PlayerSpec,
    PromotionCriteria,
    assess_promotion,
    evaluate_checkpoint_against_baselines,
    evaluate_checkpoints_head_to_head,
    generate_unique_openings,
    mcts_player_spec,
    random_player_spec,
    run_match,
    write_evaluation_report,
)
from vibechess.engine.game import Game
from vibechess.engine.move import Move
from vibechess.nn.checkpoint import CheckpointMetadata, save_checkpoint
from vibechess.nn.inference import PolicyValueInference
from vibechess.nn.model import PolicyValueConfig, PolicyValueNet


class ScriptedPlayer:
    def __init__(self, moves: tuple[str, ...]) -> None:
        self._moves = [Move.from_uci(move) for move in moves]

    def select_move(self, game: Game) -> Move:
        return self._moves[len(game.moves) // 2]


def tiny_model_config() -> PolicyValueConfig:
    return PolicyValueConfig(
        residual_channels=8,
        residual_blocks=1,
        policy_channels=2,
        value_channels=1,
        value_hidden_dim=8,
    )


def save_tiny_checkpoint(checkpoint_dir: Path) -> None:
    model = PolicyValueNet(tiny_model_config())
    mx.eval(model.parameters())
    save_checkpoint(model, checkpoint_dir, metadata=CheckpointMetadata.initial(model.config))


def test_run_match_records_legal_outcomes_and_alternates_colors() -> None:
    result = run_match(
        PlayerSpec("random-a", lambda: RandomPlayer(seed=1)),
        PlayerSpec("random-b", lambda: RandomPlayer(seed=2)),
        MatchConfig(games=2, max_plies=2),
    )

    assert result.games == 2
    assert result.player_a_score + result.player_b_score == 2.0
    assert result.draws == 2
    assert [record.player_a_color for record in result.records] == ["white", "black"]
    assert all(record.outcome_reason == "max_plies" for record in result.records)
    assert all(record.plies == 2 for record in result.records)
    assert all(record.final_fen for record in result.records)
    assert all(len(record.moves_uci) == 2 for record in result.records)


def test_baseline_specs_compare_random_and_classical_mcts() -> None:
    random_result = run_match(
        random_player_spec(seed=3, name="candidate"),
        random_player_spec(seed=4),
        MatchConfig(games=1, max_plies=1),
    )
    mcts_result = run_match(
        random_player_spec(seed=3, name="candidate"),
        mcts_player_spec(config=MCTSConfig(simulations=1, max_rollout_plies=1, seed=4)),
        MatchConfig(games=1, max_plies=1),
    )

    assert random_result.player_b == "random"
    assert mcts_result.player_b == "mcts"
    assert random_result.records[0].outcome_reason == "max_plies"
    assert mcts_result.records[0].outcome_reason == "max_plies"


def test_run_match_scores_decisive_results() -> None:
    result = run_match(
        PlayerSpec("fools-mate-white", lambda: ScriptedPlayer(("f2f3", "g2g4"))),
        PlayerSpec("fools-mate-black", lambda: ScriptedPlayer(("e7e5", "d8h4"))),
        MatchConfig(games=1, max_plies=4),
    )

    assert result.player_a_score == 0.0
    assert result.player_b_score == 1.0
    assert result.player_a_wins == 0
    assert result.player_b_wins == 1
    assert result.draws == 0
    assert result.records[0].outcome_reason == "checkmate"
    assert result.records[0].winner == "black"
    assert result.records[0].moves_uci == ["f2f3", "e7e5", "g2g4", "d8h4"]


def test_promotion_criteria_are_explicit_smoke_validation_only() -> None:
    passed = run_match(
        random_player_spec(seed=1, name="checkpoint"),
        random_player_spec(seed=2),
        MatchConfig(games=2, max_plies=1),
    )
    decision = assess_promotion(
        {"random": passed},
        PromotionCriteria(
            min_games_per_baseline=2,
            min_score_rate_vs_random=0.5,
            required_baselines=("random",),
        ),
    )
    failed = assess_promotion(
        {"random": passed},
        PromotionCriteria(
            min_games_per_baseline=3,
            min_score_rate_vs_random=0.75,
            required_baselines=("random", "mcts"),
        ),
    )

    assert decision.promoted is True
    assert decision.note == EARLY_PROMOTION_NOTE
    assert "smoke/progress" in decision.note
    assert failed.promoted is False
    assert any(reason == "missing required baseline: mcts" for reason in failed.reasons)
    assert any("requires at least 3" in reason for reason in failed.reasons)
    assert any("score rate 0.500 below required 0.750" in reason for reason in failed.reasons)


def test_generated_openings_are_unique_or_rejected() -> None:
    openings = generate_unique_openings(OpeningConfig(count=8, plies=2, seed=11))

    assert len({opening.starting_fen for opening in openings}) == 8
    assert len({opening.moves_uci for opening in openings}) == 8
    assert [opening.opening_index for opening in openings] == list(range(8))

    with pytest.raises(ValueError, match="could not generate 21 unique openings"):
        generate_unique_openings(OpeningConfig(count=21, plies=1, seed=11))


def test_checkpoint_evaluation_reuses_loaded_checkpoint_per_serial_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    save_tiny_checkpoint(checkpoint_dir)
    original = cast(Callable[[str | Path], Any], evaluation_module.__dict__["load_checkpoint"])
    loads = 0

    def counted_load_checkpoint(path: str | Path) -> Any:
        nonlocal loads
        loads += 1
        return original(path)

    monkeypatch.setattr(evaluation_module, "load_checkpoint", counted_load_checkpoint)

    evaluate_checkpoint_against_baselines(
        checkpoint_dir,
        match_config=MatchConfig(games=3, max_plies=0),
        neural_config=NeuralMCTSConfig(simulations=1, seed=7),
        baselines=("random",),
        criteria=PromotionCriteria(
            min_games_per_baseline=1,
            min_score_rate_vs_random=0.0,
        ),
    )

    assert loads == 1

def test_checkpoint_evaluation_batches_active_neural_games(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    save_tiny_checkpoint(checkpoint_dir)
    original = PolicyValueInference.predict_legal_batch
    batch_sizes: list[int] = []

    def counted_predict_legal_batch(
        self: PolicyValueInference,
        games: Any,
        legal_moves: Any,
        **kwargs: Any,
    ) -> Any:
        batch_sizes.append(len(tuple(games)))
        return original(self, games, legal_moves, **kwargs)

    monkeypatch.setattr(PolicyValueInference, "predict_legal_batch", counted_predict_legal_batch)

    report = evaluate_checkpoint_against_baselines(
        checkpoint_dir,
        match_config=MatchConfig(games=3, max_plies=1),
        neural_config=NeuralMCTSConfig(simulations=1, seed=7),
        baselines=("random",),
        criteria=PromotionCriteria(
            min_games_per_baseline=1,
            min_score_rate_vs_random=0.0,
        ),
        batch_size=2,
        active_games=3,
    )

    records = cast(
        list[dict[str, Any]],
        cast(dict[str, Any], report["matches"])["random"]["records"],
    )
    assert [record["game_index"] for record in records] == [0, 1, 2]
    assert 2 in batch_sizes


def test_checkpoint_evaluation_derives_per_game_seeds_from_base_seed(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    save_tiny_checkpoint(checkpoint_dir)

    def random_records(workers: int) -> list[dict[str, Any]]:
        report = evaluate_checkpoint_against_baselines(
            checkpoint_dir,
            match_config=MatchConfig(games=4, max_plies=6),
            neural_config=NeuralMCTSConfig(simulations=1, seed=7),
            random_seed=7,
            baselines=("random",),
            criteria=PromotionCriteria(
                min_games_per_baseline=1,
                min_score_rate_vs_random=0.0,
            ),
            workers=workers,
        )
        return cast(
            list[dict[str, Any]],
            cast(dict[str, Any], report["matches"])["random"]["records"],
        )

    serial_records = random_records(workers=1)
    parallel_records = random_records(workers=2)

    assert [record["game_index"] for record in serial_records] == [0, 1, 2, 3]
    assert [record["moves_uci"] for record in parallel_records] == [
        record["moves_uci"] for record in serial_records
    ]
    assert serial_records[0]["moves_uci"] != serial_records[2]["moves_uci"]
    assert serial_records[1]["moves_uci"] != serial_records[3]["moves_uci"]



def test_checkpoint_evaluation_loads_checkpoint_and_writes_report(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    model = PolicyValueNet(tiny_model_config())
    mx.eval(model.parameters())
    save_checkpoint(
        model,
        checkpoint_dir,
        metadata=CheckpointMetadata.initial(model.config, training_step=1, notes="eval test"),
    )

    report = evaluate_checkpoint_against_baselines(
        checkpoint_dir,
        match_config=MatchConfig(games=1, max_plies=1),
        neural_config=NeuralMCTSConfig(simulations=1, seed=7),
        mcts_config=MCTSConfig(simulations=1, max_rollout_plies=1, seed=7),
        baselines=("random", "mcts"),
        criteria=PromotionCriteria(
            min_games_per_baseline=1,
            min_score_rate_vs_random=0.0,
            min_score_rate_vs_mcts=0.0,
        ),
    )
    output = tmp_path / "nested" / "evaluation.json"
    write_evaluation_report(report, output)
    loaded = json.loads(output.read_text())

    assert set(loaded["matches"]) == {"random", "mcts"}
    assert loaded["promotion"]["promoted"] is True
    assert loaded["promotion"]["note"] == EARLY_PROMOTION_NOTE
    assert loaded["matches"]["random"]["records"][0]["outcome_reason"] == "max_plies"
    assert loaded["matches"]["mcts"]["records"][0]["outcome_reason"] == "max_plies"


def test_checkpoint_evaluation_rejects_invalid_workers(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    model = PolicyValueNet(tiny_model_config())
    mx.eval(model.parameters())
    save_checkpoint(model, checkpoint_dir, metadata=CheckpointMetadata.initial(model.config))

    try:
        evaluate_checkpoint_against_baselines(checkpoint_dir, workers=0)
    except ValueError as exc:
        assert "workers must be at least 1" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected invalid workers to raise ValueError")


def test_checkpoint_evaluation_parallel_merges_ordered_records(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    model = PolicyValueNet(tiny_model_config())
    mx.eval(model.parameters())
    save_checkpoint(model, checkpoint_dir, metadata=CheckpointMetadata.initial(model.config))

    report = evaluate_checkpoint_against_baselines(
        checkpoint_dir,
        match_config=MatchConfig(games=3, max_plies=1),
        neural_config=NeuralMCTSConfig(simulations=1, seed=7),
        mcts_config=MCTSConfig(simulations=1, max_rollout_plies=1, seed=7),
        baselines=("random", "mcts"),
        criteria=PromotionCriteria(
            min_games_per_baseline=1,
            min_score_rate_vs_random=0.0,
            min_score_rate_vs_mcts=0.0,
        ),
        workers=2,
    )

    matches = cast(dict[str, Any], report["matches"])
    assert list(matches) == ["random", "mcts"]
    for baseline in ("random", "mcts"):
        match = cast(dict[str, Any], matches[baseline])
        records = cast(list[dict[str, Any]], match["records"])
        assert [record["game_index"] for record in records] == [0, 1, 2]
        assert match["games"] == 3
        assert len(records) == 3
    promotion = cast(dict[str, Any], report["promotion"])
    assert promotion["promoted"] is True


def test_checkpoint_head_to_head_report_skips_baselines_and_orders_parallel_records(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    opponent_dir = tmp_path / "opponent"
    save_tiny_checkpoint(checkpoint_dir)
    save_tiny_checkpoint(opponent_dir)

    report = evaluate_checkpoints_head_to_head(
        checkpoint_dir,
        opponent_dir,
        match_config=MatchConfig(games=3, max_plies=1),
        neural_config=NeuralMCTSConfig(simulations=1, node_budget=5, temperature=0.0, seed=7),
        opponent_neural_config=NeuralMCTSConfig(
            simulations=1,
            node_budget=6,
            temperature=0.25,
            seed=8,
        ),
        workers=2,
    )

    assert report["mode"] == "neural_vs_neural"
    assert report["checkpoint"] == str(checkpoint_dir)
    assert report["opponent_checkpoint"] == str(opponent_dir)
    assert "promotion" not in report
    assert "criteria" not in report
    assert "matches" not in report
    neural_configs = cast(dict[str, dict[str, object]], report["neural_configs"])
    assert neural_configs["checkpoint"]["node_budget"] == 5
    assert neural_configs["opponent"]["node_budget"] == 6
    match = cast(dict[str, Any], report["match"])
    assert match["player_a"] == "checkpoint"
    assert match["player_b"] == "opponent_checkpoint"
    records = cast(list[dict[str, Any]], match["records"])
    assert [record["game_index"] for record in records] == [0, 1, 2]
    assert [record["player_a_color"] for record in records] == ["white", "black", "white"]


def test_checkpoint_head_to_head_api_defaults_opponent_config_to_main(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    opponent_dir = tmp_path / "opponent"
    save_tiny_checkpoint(checkpoint_dir)
    save_tiny_checkpoint(opponent_dir)

    report = evaluate_checkpoints_head_to_head(
        checkpoint_dir,
        opponent_dir,
        match_config=MatchConfig(games=1, max_plies=0),
        neural_config=NeuralMCTSConfig(simulations=2, node_budget=5, temperature=0.25, seed=0),
    )

    neural_configs = cast(dict[str, dict[str, object]], report["neural_configs"])
    assert neural_configs["opponent"] == neural_configs["checkpoint"]


def test_evaluate_script_smoke(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    output = tmp_path / "report.json"
    model = PolicyValueNet(tiny_model_config())
    mx.eval(model.parameters())
    save_checkpoint(model, checkpoint_dir, metadata=CheckpointMetadata.initial(model.config))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(checkpoint_dir),
            "--baseline",
            "random",
            "--opening-count",
            "1",
            "--max-plies",
            "1",
            "--neural-simulations",
            "1",
            "--min-games-per-baseline",
            "1",
            "--min-score-rate-vs-random",
            "0.0",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    report = json.loads(output.read_text())
    assert set(report["matches"]) == {"random"}
    assert report["promotion"]["promoted"] is True


def test_evaluate_script_progress_always_writes_stderr_only(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    output = tmp_path / "progress-report.json"
    save_tiny_checkpoint(checkpoint_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(checkpoint_dir),
            "--baseline",
            "random",
            "--opening-count",
            "1",
            "--max-plies",
            "1",
            "--neural-simulations",
            "1",
            "--min-games-per-baseline",
            "1",
            "--min-score-rate-vs-random",
            "0.0",
            "--progress",
            "always",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "evaluation status=" in result.stderr
    assert "evaluation: starting" in result.stderr
    assert "evaluation: completed=1/2" in result.stderr
    assert "evaluation: completed=2/2" in result.stderr
    assert "evaluation: done" in result.stderr
    assert "evaluation status=" not in result.stdout
    assert result.stdout.startswith("total opponent=random ")


def test_evaluate_script_progress_reports_effective_workers(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    output = tmp_path / "worker-progress-report.json"
    save_tiny_checkpoint(checkpoint_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(checkpoint_dir),
            "--baseline",
            "random",
            "--opening-count",
            "4",
            "--max-plies",
            "1",
            "--neural-simulations",
            "1",
            "--min-games-per-baseline",
            "1",
            "--min-score-rate-vs-random",
            "0.0",
            "--workers",
            "4",
            "--progress",
            "always",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "workers=4/4" in result.stderr
    assert "w00" in result.stderr
    assert "w01" in result.stderr
    assert "w02" in result.stderr
    assert "w03" in result.stderr
    assert "w04" not in result.stderr


def test_evaluate_script_stdout_limits_game_summary_lines(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    save_tiny_checkpoint(checkpoint_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(checkpoint_dir),
            "--baseline",
            "random",
            "--opening-count",
            "6",
            "--max-plies",
            "1",
            "--neural-simulations",
            "1",
            "--min-games-per-baseline",
            "1",
            "--min-score-rate-vs-random",
            "0.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout_lines = result.stdout.strip().splitlines()
    assert stdout_lines[0].endswith("shown_games=10/12")
    assert len([line for line in stdout_lines if line.startswith("game ")]) == 10


def test_evaluate_script_rejects_invalid_workers(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(tmp_path / "missing"),
            "--workers",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--workers must be at least 1" in result.stderr

def test_evaluate_script_rejects_reuse_floor_above_simulations(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(tmp_path / "missing"),
            "--neural-simulations",
            "1",
            "--reuse-simulation-budget",
            "--min-reuse-simulations",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--min-reuse-simulations must be no greater than --neural-simulations" in result.stderr


def test_evaluate_script_neural_vs_neural_uses_unique_paired_openings(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    opponent_dir = tmp_path / "opponent"
    output = tmp_path / "head-to-head-openings.json"
    save_tiny_checkpoint(checkpoint_dir)
    save_tiny_checkpoint(opponent_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(checkpoint_dir),
            "--opponent-checkpoint",
            str(opponent_dir),
            "--opening-count",
            "2",
            "--opening-plies",
            "4",
            "--max-plies",
            "4",
            "--neural-simulations",
            "1",
            "--seed",
            "7",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert result.stdout.startswith("total opponent=opponent_checkpoint ")
    assert "game opponent=opponent_checkpoint index=0" in result.stdout
    neural_configs = report["neural_configs"]
    assert neural_configs["checkpoint"]["temperature"] == 0.0
    assert neural_configs["opponent"]["temperature"] == 0.0
    assert report["opening_config"] == {"count": 2, "plies": 4, "seed": 7}
    records = report["match"]["records"]
    assert [record["opening_index"] for record in records] == [0, 0, 1, 1]
    assert records[0]["starting_fen"] == records[1]["starting_fen"]
    assert records[2]["starting_fen"] == records[3]["starting_fen"]
    assert records[0]["starting_fen"] != records[2]["starting_fen"]
    assert records[0]["opening_moves_uci"] == records[1]["opening_moves_uci"]
    assert records[2]["opening_moves_uci"] == records[3]["opening_moves_uci"]


def test_evaluate_script_neural_vs_neural_smoke_defaults_opponent_settings(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    opponent_dir = tmp_path / "opponent"
    output = tmp_path / "head-to-head-report.json"
    save_tiny_checkpoint(checkpoint_dir)
    save_tiny_checkpoint(opponent_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(checkpoint_dir),
            "--opponent-checkpoint",
            str(opponent_dir),
            "--opening-count",
            "1",
            "--max-plies",
            "1",
            "--neural-simulations",
            "1",
            "--neural-node-budget",
            "5",
            "--neural-temperature",
            "0.25",
            "--seed",
            "13",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    stdout_lines = result.stdout.strip().splitlines()
    assert stdout_lines[0].startswith(
        "total opponent=opponent_checkpoint games=2 score=1-1 "
    )
    assert len([line for line in stdout_lines if line.startswith("game ")]) == 2
    assert report["mode"] == "neural_vs_neural"
    assert "promotion" not in report
    assert "criteria" not in report
    assert "matches" not in report
    neural_configs = report["neural_configs"]
    assert neural_configs["checkpoint"] == neural_configs["opponent"]
    assert neural_configs["opponent"]["simulations"] == 1
    assert neural_configs["opponent"]["node_budget"] == 5
    assert neural_configs["opponent"]["temperature"] == 0.25
    assert neural_configs["opponent"]["seed"] == 13


def test_evaluate_script_neural_vs_neural_overrides_opponent_settings(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    opponent_dir = tmp_path / "opponent"
    output = tmp_path / "head-to-head-overrides.json"
    save_tiny_checkpoint(checkpoint_dir)
    save_tiny_checkpoint(opponent_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(checkpoint_dir),
            "--opponent-checkpoint",
            str(opponent_dir),
            "--opening-count",
            "1",
            "--max-plies",
            "1",
            "--neural-simulations",
            "2",
            "--neural-node-budget",
            "5",
            "--neural-temperature",
            "0.25",
            "--neural-collection-batch-size",
            "2",
            "--neural-virtual-loss",
            "3",
            "--reuse-simulation-budget",
            "--min-reuse-simulations",
            "1",
            "--seed",
            "13",
            "--opponent-neural-simulations",
            "1",
            "--opponent-neural-node-budget",
            "6",
            "--opponent-neural-temperature",
            "0.0",
            "--opponent-neural-collection-batch-size",
            "1",
            "--opponent-neural-virtual-loss",
            "0",
            "--opponent-reuse-simulation-budget",
            "--opponent-min-reuse-simulations",
            "0",
            "--opponent-seed",
            "0",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert result.stdout.startswith("total opponent=opponent_checkpoint ")
    neural_configs = report["neural_configs"]
    assert neural_configs["checkpoint"]["simulations"] == 2
    assert neural_configs["checkpoint"]["node_budget"] == 5
    assert neural_configs["checkpoint"]["temperature"] == 0.25
    assert neural_configs["checkpoint"]["seed"] == 13
    assert neural_configs["checkpoint"]["collection_batch_size"] == 2
    assert neural_configs["checkpoint"]["virtual_loss"] == 3
    assert neural_configs["checkpoint"]["reuse_simulation_budget"] is True
    assert neural_configs["checkpoint"]["min_reuse_simulations"] == 1
    assert neural_configs["opponent"]["simulations"] == 1
    assert neural_configs["opponent"]["node_budget"] == 6
    assert neural_configs["opponent"]["temperature"] == 0.0
    assert neural_configs["opponent"]["seed"] == 0
    assert neural_configs["opponent"]["collection_batch_size"] == 1
    assert neural_configs["opponent"]["virtual_loss"] == 0
    assert neural_configs["opponent"]["reuse_simulation_budget"] is True
    assert neural_configs["opponent"]["min_reuse_simulations"] == 0


def test_evaluate_script_rejects_baseline_and_promotion_with_opponent_checkpoint(
    tmp_path: Path,
) -> None:
    base_args = [
        sys.executable,
        "scripts/evaluate.py",
        "--checkpoint",
        str(tmp_path / "missing-a"),
        "--opponent-checkpoint",
        str(tmp_path / "missing-b"),
    ]

    conflicts = [
        (["--baseline", "random"], "--baseline cannot be used with --opponent-checkpoint"),
        (["--mcts-simulations", "7"], "--mcts-simulations cannot be used"),
        (["--mcts-node-budget", "7"], "--mcts-node-budget cannot be used"),
        (["--mcts-rollout-plies", "7"], "--mcts-rollout-plies cannot be used"),
        (["--min-games-per-baseline", "3"], "--min-games-per-baseline cannot be used"),
        (["--min-score-rate-vs-random", "0.75"], "--min-score-rate-vs-random cannot be used"),
        (["--min-score-rate-vs-mcts", "0.25"], "--min-score-rate-vs-mcts cannot be used"),
        (
            ["--require-promotion"],
            "--require-promotion cannot be used with --opponent-checkpoint",
        ),
    ]

    for flag_args, message in conflicts:
        result = subprocess.run(
            [*base_args, *flag_args],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert message in result.stderr


def test_evaluate_script_parallel_smoke_stdout_summary(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    output = tmp_path / "parallel-report.json"
    model = PolicyValueNet(tiny_model_config())
    mx.eval(model.parameters())
    save_checkpoint(model, checkpoint_dir, metadata=CheckpointMetadata.initial(model.config))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(checkpoint_dir),
            "--baseline",
            "random",
            "--opening-count",
            "1",
            "--max-plies",
            "1",
            "--neural-simulations",
            "1",
            "--min-games-per-baseline",
            "1",
            "--min-score-rate-vs-random",
            "0.0",
            "--batch-size",
            "2",
            "--active-games",
            "2",
            "--workers",
            "2",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    stdout_lines = result.stdout.strip().splitlines()
    assert stdout_lines[0].startswith("total opponent=random games=2 ")
    assert any(line.startswith("promotion promoted=True reasons=") for line in stdout_lines)
    assert len([line for line in stdout_lines if line.startswith("game ")]) == 2
    assert set(report["matches"]) == {"random"}
    assert [record["game_index"] for record in report["matches"]["random"]["records"]] == [
        0,
        1,
    ]
