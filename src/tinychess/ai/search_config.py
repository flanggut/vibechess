"""Search configuration for classical AI players."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Budgets and rollout settings for the classical MCTS baseline.

    ``max_rollout_plies`` caps the number of random rollout moves after selection and
    expansion. Set it to ``0`` for the high-simulation static leaf mode, which evaluates
    the selected leaf directly without making random rollout moves. The default remains
    ``16`` to preserve existing random-rollout behavior.
    """

    simulations: int = 25
    time_limit_seconds: float | None = None
    node_budget: int | None = None
    exploration: float = 1.41421356237
    max_rollout_plies: int = 16
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.simulations < 1:
            msg = f"simulations must be at least 1, got {self.simulations}"
            raise ValueError(msg)
        if self.time_limit_seconds is not None and self.time_limit_seconds < 0:
            msg = f"time_limit_seconds must be non-negative, got {self.time_limit_seconds}"
            raise ValueError(msg)
        if self.node_budget is not None and self.node_budget < 1:
            msg = f"node_budget must be at least 1, got {self.node_budget}"
            raise ValueError(msg)
        if self.exploration < 0:
            msg = f"exploration must be non-negative, got {self.exploration}"
            raise ValueError(msg)
        if self.max_rollout_plies < 0:
            msg = f"max_rollout_plies must be non-negative, got {self.max_rollout_plies}"
            raise ValueError(msg)
