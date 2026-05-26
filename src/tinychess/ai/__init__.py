"""AI player interfaces and baseline players."""

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
    mcts_player_spec,
    random_player_spec,
    run_match,
    write_evaluation_report,
)
from tinychess.ai.mcts import MCTSPlayer, MCTSResult
from tinychess.ai.neural_mcts import (
    NeuralInference,
    NeuralMCTSConfig,
    NeuralMCTSPlayer,
    NeuralMCTSResult,
)
from tinychess.ai.player import NoLegalMoveError, Player, RandomPlayer, play_game
from tinychess.ai.search_config import MCTSConfig

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
    "mcts_player_spec",
    "play_game",
    "random_player_spec",
    "run_match",
    "write_evaluation_report",
]
