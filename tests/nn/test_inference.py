"""Focused inference tests for compact legal-batch behavior."""

from typing import Any, cast

import mlx.core as mx
import numpy as np
import pytest

from vibechess.engine import Game
from vibechess.nn import (
    ACTION_SPACE_SIZE,
    TENSOR_SHAPE,
    PolicyValueInference,
    PolicyValueNet,
    PolicyValueOutput,
    tensor_shape,
)
from vibechess.profiling import activate_self_play_profile


def scalar(value: Any) -> float:
    return float(value.item())


class RowVaryingOutputModel:
    def __init__(self, logits: Any, values: tuple[float, ...]) -> None:
        self.logits = mx.array(logits, dtype=mx.float32)
        self.values = mx.array(values, dtype=mx.float32)

    def __call__(self, inputs: object) -> PolicyValueOutput:
        shape = tensor_shape(inputs)
        batch_size = 1 if shape == TENSOR_SHAPE else shape[0]
        if tensor_shape(self.logits) != (batch_size, ACTION_SPACE_SIZE):
            raise ValueError("row-varying logits must match the requested batch size")
        if tensor_shape(self.values) != (batch_size,):
            raise ValueError("row-varying values must match the requested batch size")
        return PolicyValueOutput(policy_logits=self.logits, value=self.values)


def test_predict_legal_batch_profiles_single_combined_sync_zone() -> None:
    games = (
        Game.new(),
        Game.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    )
    legal_moves = tuple(game.legal_moves for game in games)
    base_logits = np.linspace(-1.0, 1.0, ACTION_SPACE_SIZE, dtype=np.float32)
    logits = np.stack((base_logits, -base_logits))
    values = (0.125, -0.25)
    inference = PolicyValueInference(cast(PolicyValueNet, RowVaryingOutputModel(logits, values)))

    with activate_self_play_profile("detailed") as profiler:
        result = inference.predict_legal_batch(games, legal_moves)

    zones = profiler.stats.to_dict()["zones"]
    assert isinstance(zones, dict)
    legal_batch_zone = zones["mlx.sync.legal_batch_eval"]
    assert isinstance(legal_batch_zone, dict)
    assert legal_batch_zone["calls"] == 1
    assert "mlx.sync.value_item" not in zones
    assert "mlx.sync.policy_eval" not in zones
    assert result.values == pytest.approx(values)
    for policy, legal in zip(result.legal_policies, legal_moves, strict=True):
        assert tensor_shape(policy) == (len(legal),)
        if legal:
            assert scalar(mx.sum(policy)) == pytest.approx(1.0)
