from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import mlx.core as mx

from tinychess.ai import MCTSConfig, NeuralMCTSConfig, RandomPlayer
from tinychess.ai.evaluation import (
    EARLY_PROMOTION_NOTE,
    MatchConfig,
    PlayerSpec,
    PromotionCriteria,
    assess_promotion,
    evaluate_checkpoint_against_baselines,
    mcts_player_spec,
    random_player_spec,
    run_match,
    write_evaluation_report,
)
from tinychess.ai.mcts import MCTSPlayer
from tinychess.engine.game import Game
from tinychess.engine.move import Move
from tinychess.nn.checkpoint import CheckpointMetadata, save_checkpoint
from tinychess.nn.model import PolicyValueConfig, PolicyValueNet


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


def test_seeded_player_specs_vary_seed_by_game_index() -> None:
    random_spec = random_player_spec(seed=10)
    mcts_spec = mcts_player_spec(config=MCTSConfig(simulations=1, seed=20))

    random_player = cast(RandomPlayer, random_spec.create(3))
    mcts_player = cast(MCTSPlayer, mcts_spec.create(3))

    assert random_player.select_move(Game.new()) == RandomPlayer(seed=13).select_move(Game.new())
    assert mcts_player.config.seed == 23


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
            "--games",
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
    assert result.stderr.strip().splitlines()[-1].startswith(
        "evaluation_summary promoted=true random_games=1 random_score_rate="
    )
    report = json.loads(output.read_text())
    assert set(report["matches"]) == {"random"}
    assert report["promotion"]["promoted"] is True


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


def test_evaluate_script_parallel_smoke_stdout_json(tmp_path: Path) -> None:
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
            "--games",
            "2",
            "--max-plies",
            "1",
            "--neural-simulations",
            "1",
            "--min-games-per-baseline",
            "1",
            "--min-score-rate-vs-random",
            "0.0",
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
    assert result.stderr.strip().splitlines()[-1].startswith(
        "evaluation_summary promoted=true random_games=2 random_score_rate="
    )
    stdout_report = json.loads(result.stdout)
    file_report = json.loads(output.read_text())
    assert stdout_report == file_report
    assert set(stdout_report["matches"]) == {"random"}
    assert [record["game_index"] for record in stdout_report["matches"]["random"]["records"]] == [
        0,
        1,
    ]
