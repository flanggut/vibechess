"""AI player interfaces and baseline players."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from tinychess.ai.mcts import MCTSPlayer, MCTSResult
from tinychess.ai.player import NoLegalMoveError, Player, RandomPlayer, play_game
from tinychess.ai.search_config import MCTSConfig

if TYPE_CHECKING:
    from tinychess.ai.evaluation import (
        EARLY_PROMOTION_NOTE,
        MatchConfig,
        MatchGameRecord,
        MatchResult,
        PlayerSpec,
        PromotionCriteria,
        PromotionDecision,
        assess_promotion,
        checkpoint_player_spec,
        evaluate_checkpoint_against_baselines,
        evaluate_checkpoints_head_to_head,
        mcts_player_spec,
        random_player_spec,
        run_match,
        write_evaluation_report,
    )
    from tinychess.ai.neural_mcts import (
        NeuralInference,
        NeuralMCTSConfig,
        NeuralMCTSPlayer,
        NeuralMCTSResult,
    )

_EVALUATION_EXPORTS = {
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
}
_NEURAL_EXPORTS = {
    "NeuralInference",
    "NeuralMCTSConfig",
    "NeuralMCTSPlayer",
    "NeuralMCTSResult",
}

__all__ = [
    "EARLY_PROMOTION_NOTE",
    "MCTSConfig",
    "MatchConfig",
    "MatchGameRecord",
    "MatchResult",
    "MCTSPlayer",
    "MCTSResult",
    "NeuralInference",
    "NeuralMCTSConfig",
    "NeuralMCTSPlayer",
    "NeuralMCTSResult",
    "NoLegalMoveError",
    "Player",
    "PlayerSpec",
    "PromotionCriteria",
    "PromotionDecision",
    "RandomPlayer",
    "assess_promotion",
    "checkpoint_player_spec",
    "evaluate_checkpoint_against_baselines",
    "evaluate_checkpoints_head_to_head",
    "mcts_player_spec",
    "play_game",
    "random_player_spec",
    "run_match",
    "write_evaluation_report",
]


def __getattr__(name: str) -> Any:
    """Lazily expose neural/evaluation exports without importing MLX for baselines."""
    if name in _EVALUATION_EXPORTS:
        module = import_module("tinychess.ai.evaluation")
    elif name in _NEURAL_EXPORTS:
        module = import_module("tinychess.ai.neural_mcts")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
