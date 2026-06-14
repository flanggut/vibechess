"""Policy/value inference DTOs and wrappers for MLX models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import mlx.core as mx
import numpy as np

from vibechess.engine.game import Game
from vibechess.engine.move import Move
from vibechess.nn.encode import (
    ACTION_SPACE_SIZE,
    ENCODER_CHANNELS,
    TENSOR_SHAPE,
    encode_game,
    legal_move_mask,
    legal_move_mask_from_legal_moves,
    tensor_shape,
    to_mlx,
)
from vibechess.nn.encode import (
    legal_action_indices as legal_action_indices_fn,
)
from vibechess.profiling import profile_scope, record_counter, record_distribution

if TYPE_CHECKING:
    from vibechess.nn.model import PolicyValueNet

MLXArray: TypeAlias = Any


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Single-position inference result with optional legal-move masking.

    ``legal_moves``, ``legal_action_indices``, and ``legal_policy`` are populated only
    by search-oriented helpers that receive a precomputed legal move tuple. Public
    ``predict()`` callers continue to receive the full 4672-action policy contract.
    """

    policy_logits: MLXArray
    policy: MLXArray
    value: float
    legal_mask: MLXArray | None = None
    legal_moves: tuple[Move, ...] | None = None
    legal_action_indices: tuple[int, ...] = ()
    legal_policy: MLXArray | None = None


@dataclass(frozen=True, slots=True)
class BatchInferenceResult:
    """Batched policy/value inference result preserving 4672-action outputs."""

    policy_logits: MLXArray
    policy: MLXArray
    values: tuple[float, ...]
    legal_masks: MLXArray | None = None
    legal_moves: tuple[tuple[Move, ...], ...] | None = None
    legal_action_indices: tuple[tuple[int, ...], ...] = ()

    def result_at(self, index: int) -> InferenceResult:
        """Return a single-position view for one batch row."""
        legal_moves = None if self.legal_moves is None else self.legal_moves[index]
        legal_indices = () if not self.legal_action_indices else self.legal_action_indices[index]
        legal_policy = None
        if legal_moves is not None:
            if legal_indices:
                legal_policy = self.policy[index][mx.array(legal_indices)]
            else:
                legal_policy = mx.zeros((0,), dtype=mx.float32)
        return InferenceResult(
            policy_logits=self.policy_logits[index],
            policy=self.policy[index],
            value=self.values[index],
            legal_mask=None if self.legal_masks is None else self.legal_masks[index],
            legal_moves=legal_moves,
            legal_action_indices=legal_indices,
            legal_policy=legal_policy,
        )


@dataclass(frozen=True, slots=True)
class LegalPolicyResult:
    """Search-only inference result with compact priors over supplied legal moves."""

    value: float
    legal_moves: tuple[Move, ...]
    legal_action_indices: tuple[int, ...]
    legal_policy: MLXArray


@dataclass(frozen=True, slots=True)
class LegalPolicyBatchResult:
    """Batched search-only inference result without dense policy or mask tensors."""

    values: tuple[float, ...]
    legal_moves: tuple[tuple[Move, ...], ...]
    legal_action_indices: tuple[tuple[int, ...], ...]
    legal_policies: tuple[MLXArray, ...]

    def result_at(self, index: int) -> LegalPolicyResult:
        """Return a compact single-position view for one batch row."""
        return LegalPolicyResult(
            value=self.values[index],
            legal_moves=self.legal_moves[index],
            legal_action_indices=self.legal_action_indices[index],
            legal_policy=self.legal_policies[index],
        )


class PolicyValueInference:
    """Inference wrapper around :class:`PolicyValueNet` for encoded games."""

    def __init__(self, model: PolicyValueNet) -> None:
        self.model = model

    def predict(self, game: Game, *, mask_legal_moves: bool = True) -> InferenceResult:
        """Run model inference for one game position.

        Returns raw logits for all 4672 actions, a probability vector, and a
        scalar value from the side-to-move perspective in ``[-1, 1]``. When
        ``mask_legal_moves`` is true, illegal actions receive zero probability.
        Terminal/no-legal-move positions return an all-zero policy.
        """
        with profile_scope("inference.predict", mask_legal_moves=mask_legal_moves):
            record_counter("inference.predict.calls")
            with profile_scope("encode.game_mlx"):
                encoded = encode_game(game)
            with profile_scope("model.forward"):
                output = self.model(encoded)
            logits = cast(MLXArray, output.policy_logits[0])
            with profile_scope("mlx.sync.value_item"):
                value = float(output.value[0].item())
            if not mask_legal_moves:
                with profile_scope("inference.policy_softmax"):
                    policy = mx.softmax(logits)
                with profile_scope("mlx.sync.policy_eval"):
                    mx.eval(policy)
                return InferenceResult(policy_logits=logits, policy=policy, value=value)

            with profile_scope("policy.legal_mask_mlx"):
                mask = legal_move_mask(game)
            if float(mx.sum(mask).item()) == 0.0:
                policy = mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32)
            else:
                masked_logits = mx.where(mask > 0, logits, mx.full(logits.shape, -1.0e9))
                with profile_scope("inference.policy_softmax"):
                    policy = mx.softmax(masked_logits) * mask
                    policy = policy / mx.sum(policy)
            with profile_scope("mlx.sync.policy_eval"):
                mx.eval(policy)
            return InferenceResult(
                policy_logits=logits,
                policy=policy,
                value=value,
                legal_mask=mask,
            )

    def predict_with_legal_moves(
        self,
        game: Game,
        legal_moves: tuple[Move, ...],
        *,
        legal_action_indices: Sequence[int] | None = None,
        legal_action_index_array: MLXArray | None = None,
        encoded_input: MLXArray | None = None,
    ) -> InferenceResult:
        """Run inference using precomputed legal moves for compact legal priors.

        The returned ``policy`` and ``legal_mask`` keep the public 4672-action shape,
        but only the compact legal softmax vector is evaluated eagerly. This avoids
        recomputing legal moves and avoids synchronizing a full policy vector in
        search code that only needs priors for cached legal moves.
        """
        with profile_scope("inference.predict_with_legal_moves"):
            record_counter("inference.predict_with_legal_moves.calls")
            legal = tuple(legal_moves)
            record_distribution("inference.legal_moves", len(legal), unit="moves")
            if encoded_input is None:
                with profile_scope("encode.game_mlx"):
                    encoded = encode_game(game)
            else:
                with profile_scope("encode.game_mlx.cached"):
                    encoded = _prepare_encoded_batch(encoded_input)
            with profile_scope("model.forward"):
                output = self.model(encoded)
            logits = cast(MLXArray, output.policy_logits[0])
            with profile_scope("mlx.sync.value_item"):
                value = float(output.value[0].item())
            if legal_action_indices is None:
                with profile_scope("policy.legal_indices"):
                    indices = legal_action_indices_fn(game, legal)
            else:
                indices = tuple(legal_action_indices)
                if len(indices) != len(legal):
                    raise ValueError("legal_action_indices length must match legal_moves")
            if (
                legal_action_index_array is not None
                and tensor_shape(legal_action_index_array) != (len(legal),)
            ):
                raise ValueError("legal_action_index_array length must match legal_moves")
            if not indices:
                legal_policy = mx.zeros((0,), dtype=mx.float32)
                policy = mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32)
                mask = mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32)
            else:
                index_array = (
                    legal_action_index_array
                    if legal_action_index_array is not None
                    else mx.array(indices)
                )
                legal_logits = logits[index_array]
                with profile_scope("inference.policy_softmax"):
                    legal_policy = mx.softmax(legal_logits)
                policy = (
                    mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32)
                    .at[index_array]
                    .add(legal_policy)
                )
                mask = mx.zeros((ACTION_SPACE_SIZE,), dtype=mx.float32).at[index_array].add(1.0)
            with profile_scope("mlx.sync.policy_eval"):
                mx.eval(legal_policy)
            return InferenceResult(
                policy_logits=logits,
                policy=policy,
                value=value,
                legal_mask=mask,
                legal_moves=legal,
                legal_action_indices=indices,
                legal_policy=legal_policy,
            )

    def predict_legal(
        self,
        game: Game,
        legal_moves: Sequence[Move],
    ) -> LegalPolicyResult:
        """Run compact search-only inference for one game position.

        The returned policy vector is normalized over exactly ``legal_moves`` and
        has length ``len(legal_moves)``. No dense 4672-action policy or legal mask
        is constructed.
        """
        return self.predict_legal_batch((game,), (legal_moves,)).result_at(0)

    def predict_legal_batch(
        self,
        games: Sequence[Game],
        legal_moves: Sequence[Sequence[Move]],
        *,
        legal_action_indices: Sequence[Sequence[int]] | None = None,
        legal_action_index_arrays: Sequence[MLXArray] | None = None,
        encoded_inputs: Sequence[MLXArray] | MLXArray | None = None,
    ) -> LegalPolicyBatchResult:
        """Run compact batched inference over supplied legal moves only.

        This search-oriented API computes model logits once for the encoded batch,
        gathers each row's supplied legal action logits, and applies softmax only
        to those compact vectors. Empty legal rows return empty ``float32`` policy
        vectors. Search callers may provide cached legal action indices and/or
        cached encoded position tensors; public ``Game`` inputs remain required so
        the legal-move contract and existing call sites stay unchanged.
        """
        with profile_scope("inference.predict_legal_batch"):
            games_tuple = tuple(games)
            if not games_tuple:
                raise ValueError("predict_legal_batch requires at least one position")
            batch_size = len(games_tuple)
            legal_by_row = tuple(tuple(row) for row in legal_moves)
            if len(legal_by_row) != batch_size:
                raise ValueError("legal_moves length must match batch size")

            record_counter("inference.predict_legal_batch.calls")
            record_counter("inference.legal_batch_positions", batch_size)
            record_distribution("inference.legal_batch_size", batch_size, unit="positions")
            total_legal_moves = sum(len(row) for row in legal_by_row)
            record_counter("inference.legal_batch_moves", total_legal_moves)
            for legal in legal_by_row:
                record_distribution("inference.legal_moves", len(legal), unit="moves")

            if encoded_inputs is None:
                with profile_scope("encode.batch_stack"):
                    encoded = mx.stack([encode_game(game) for game in games_tuple])
            else:
                with profile_scope("encode.batch_stack.cached"):
                    encoded = _prepare_cached_encoded_batch(encoded_inputs, batch_size)
            with profile_scope("model.forward"):
                output = self.model(encoded)
            logits = output.policy_logits
            value_count = tensor_shape(output.value)[0]
            if value_count != batch_size:
                raise ValueError(f"model returned {value_count} values for batch size {batch_size}")

            if legal_action_indices is None:
                with profile_scope("policy.legal_indices"):
                    legal_indices_by_row = tuple(
                        legal_action_indices_fn(game, legal)
                        for game, legal in zip(games_tuple, legal_by_row, strict=True)
                    )
            else:
                legal_indices_by_row = tuple(tuple(row) for row in legal_action_indices)
                if len(legal_indices_by_row) != batch_size:
                    raise ValueError("legal_action_indices length must match batch size")
                for legal, indices in zip(legal_by_row, legal_indices_by_row, strict=True):
                    if len(indices) != len(legal):
                        raise ValueError("legal_action_indices rows must match legal_moves rows")
            legal_index_arrays = None
            if legal_action_index_arrays is not None:
                legal_index_arrays = tuple(legal_action_index_arrays)
                if len(legal_index_arrays) != batch_size:
                    raise ValueError("legal_action_index_arrays length must match batch size")
                for legal, index_array in zip(legal_by_row, legal_index_arrays, strict=True):
                    if tensor_shape(index_array) != (len(legal),):
                        raise ValueError(
                            "legal_action_index_arrays rows must match legal_moves rows"
                        )
            with profile_scope("inference.policy_softmax"):
                legal_policy_tuple = _compact_legal_policy_batch(
                    logits,
                    legal_indices_by_row,
                    legal_index_arrays,
                )
            with profile_scope("mlx.sync.legal_batch_eval"):
                mx.eval(output.value, *legal_policy_tuple)
            values = tuple(float(value) for value in np.asarray(output.value, dtype=np.float32))
            if len(values) != batch_size:
                raise ValueError(f"model returned {len(values)} values for batch size {batch_size}")
            return LegalPolicyBatchResult(
                values=values,
                legal_moves=legal_by_row,
                legal_action_indices=legal_indices_by_row,
                legal_policies=legal_policy_tuple,
            )

    def predict_batch(
        self,
        inputs: Sequence[Game] | MLXArray,
        *,
        legal_masks: MLXArray | None = None,
        legal_moves: Sequence[Sequence[Move]] | None = None,
        mask_legal_moves: bool = True,
    ) -> BatchInferenceResult:
        """Run batched inference for games or encoded position tensors.

        ``inputs`` may be a non-empty sequence of :class:`Game` objects or an
        encoded tensor with shape ``[N, 20, 8, 8]`` / ``[N, 8, 8, 20]``. Masked
        game batches can optionally pass precomputed legal move lists; encoded
        batches can pass a batched legal mask. In all modes logits and policies
        keep the fixed 4672-action dimension.
        """
        with profile_scope("inference.predict_batch", mask_legal_moves=mask_legal_moves):
            games = _game_sequence(inputs)
            if games is not None:
                if not games:
                    raise ValueError("predict_batch requires at least one position")
                with profile_scope("encode.batch_stack"):
                    encoded = mx.stack([encode_game(game) for game in games])
            else:
                encoded = _prepare_encoded_batch(inputs)
            batch_size = tensor_shape(encoded)[0]
            record_counter("inference.predict_batch.calls")
            record_counter("inference.batch_positions", batch_size)
            record_distribution("inference.batch_size", batch_size, unit="positions")
            if batch_size < 1:
                raise ValueError("predict_batch requires at least one position")

            if legal_moves is not None and games is None:
                raise ValueError("legal_moves require Game inputs")
            if legal_masks is not None and legal_moves is not None:
                raise ValueError("pass legal_masks or legal_moves, not both")

            with profile_scope("model.forward"):
                output = self.model(encoded)
            logits = output.policy_logits
            with profile_scope("mlx.sync.value_item"):
                values = tuple(float(value) for value in np.asarray(output.value, dtype=np.float32))
            if len(values) != batch_size:
                raise ValueError(f"model returned {len(values)} values for batch size {batch_size}")

            legal_masks_batch: MLXArray | None = None
            legal_by_row: tuple[tuple[Move, ...], ...] | None = None
            legal_indices_by_row: tuple[tuple[int, ...], ...] = ()
            if mask_legal_moves:
                if legal_moves is not None:
                    assert games is not None
                    legal_by_row = tuple(tuple(row) for row in legal_moves)
                    if len(legal_by_row) != batch_size:
                        raise ValueError("legal_moves length must match batch size")
                    with profile_scope("policy.legal_indices"):
                        legal_indices_by_row = tuple(
                            legal_action_indices_fn(game, legal)
                            for game, legal in zip(games, legal_by_row, strict=True)
                        )
                    with profile_scope("policy.legal_mask_mlx"):
                        legal_masks_batch = mx.stack(
                            [
                                legal_move_mask_from_legal_moves(game, legal)
                                for game, legal in zip(games, legal_by_row, strict=True)
                            ]
                        )
                elif legal_masks is not None:
                    with profile_scope("policy.legal_mask_mlx"):
                        legal_masks_batch = _prepare_legal_mask_batch(legal_masks, batch_size)
                elif games is not None:
                    legal_by_row = tuple(game.legal_moves for game in games)
                    with profile_scope("policy.legal_indices"):
                        legal_indices_by_row = tuple(
                            legal_action_indices_fn(game, legal)
                            for game, legal in zip(games, legal_by_row, strict=True)
                        )
                    with profile_scope("policy.legal_mask_mlx"):
                        legal_masks_batch = mx.stack([legal_move_mask(game) for game in games])
                else:
                    raise ValueError("masked encoded batch inference requires legal_masks")

            with profile_scope("inference.policy_softmax"):
                if not mask_legal_moves:
                    policy = mx.softmax(logits, axis=1)
                else:
                    assert legal_masks_batch is not None
                    masked_logits = mx.where(
                        legal_masks_batch > 0,
                        logits,
                        mx.full(logits.shape, -1.0e9),
                    )
                    policy = mx.softmax(masked_logits, axis=1) * legal_masks_batch
                    row_sums = mx.sum(policy, axis=1, keepdims=True)
                    policy = policy / mx.where(row_sums > 0, row_sums, 1.0)
            with profile_scope("mlx.sync.policy_eval"):
                if legal_masks_batch is None:
                    mx.eval(logits, policy)
                else:
                    mx.eval(logits, policy, legal_masks_batch)
            return BatchInferenceResult(
                policy_logits=logits,
                policy=policy,
                values=values,
                legal_masks=legal_masks_batch,
                legal_moves=legal_by_row,
                legal_action_indices=legal_indices_by_row,
            )


def _game_sequence(inputs: Sequence[Game] | MLXArray) -> tuple[Game, ...] | None:
    if not isinstance(inputs, Sequence):
        return None
    games = tuple(inputs)
    if all(isinstance(item, Game) for item in games):
        return cast(tuple[Game, ...], games)
    return None


def _prepare_encoded_batch(inputs: MLXArray) -> MLXArray:
    tensor = to_mlx(inputs)
    shape = tensor_shape(tensor)
    if shape == TENSOR_SHAPE:
        return tensor[None, :, :, :]
    if len(shape) == 4 and (shape[1:] == TENSOR_SHAPE or shape[1:] == (8, 8, ENCODER_CHANNELS)):
        return tensor
    raise ValueError(
        "expected encoded batch shape "
        f"{TENSOR_SHAPE}, [N, {TENSOR_SHAPE}], or "
        f"[N, 8, 8, {ENCODER_CHANNELS}], got {shape}"
    )


def _prepare_cached_encoded_batch(
    inputs: Sequence[MLXArray] | MLXArray,
    batch_size: int,
) -> MLXArray:
    if isinstance(inputs, Sequence):
        rows = tuple(inputs)
        encoded = mx.stack([to_mlx(row) for row in rows])
    else:
        encoded = _prepare_encoded_batch(inputs)
    shape = tensor_shape(encoded)
    if len(shape) != 4 or shape[0] != batch_size:
        raise ValueError(f"encoded_inputs batch size must be {batch_size}, got {shape}")
    if shape[1:] != TENSOR_SHAPE and shape[1:] != (8, 8, ENCODER_CHANNELS):
        raise ValueError(
            "encoded_inputs rows must have shape "
            f"{TENSOR_SHAPE} or [8, 8, {ENCODER_CHANNELS}], got {shape[1:]}"
        )
    return encoded



def _compact_legal_policy_batch(
    logits: MLXArray,
    legal_indices_by_row: Sequence[Sequence[int]],
    legal_index_arrays: Sequence[MLXArray] | None,
) -> tuple[MLXArray, ...]:
    """Return per-row compact legal policies using one batched gather and softmax."""
    row_lengths = tuple(len(indices) for indices in legal_indices_by_row)
    max_legal = max(row_lengths, default=0)
    if max_legal == 0:
        return tuple(mx.zeros((0,), dtype=mx.float32) for _ in row_lengths)
    padded_indices, valid_mask = _padded_legal_index_batch(
        legal_indices_by_row,
        legal_index_arrays,
        row_lengths,
        max_legal,
    )
    legal_logits = mx.take_along_axis(logits, padded_indices, axis=1)
    masked_logits = mx.where(
        valid_mask,
        legal_logits,
        mx.full(tensor_shape(legal_logits), -1.0e9, dtype=legal_logits.dtype),
    )
    legal_policy_batch = mx.softmax(masked_logits, axis=1) * valid_mask.astype(mx.float32)
    return tuple(
        legal_policy_batch[row_index, :row_length]
        for row_index, row_length in enumerate(row_lengths)
    )


def _padded_legal_index_batch(
    legal_indices_by_row: Sequence[Sequence[int]],
    legal_index_arrays: Sequence[MLXArray] | None,
    row_lengths: Sequence[int],
    max_legal: int,
) -> tuple[MLXArray, MLXArray]:
    if legal_index_arrays is not None and all(length == max_legal for length in row_lengths):
        return (
            mx.stack(list(legal_index_arrays)),
            mx.ones((len(row_lengths), max_legal), dtype=mx.bool_),
        )

    padded = np.zeros((len(legal_indices_by_row), max_legal), dtype=np.int32)
    valid = np.zeros((len(legal_indices_by_row), max_legal), dtype=np.bool_)
    for row_index, indices in enumerate(legal_indices_by_row):
        row_length = len(indices)
        if row_length:
            padded[row_index, :row_length] = indices
            valid[row_index, :row_length] = True
    return mx.array(padded), mx.array(valid)


def _prepare_legal_mask_batch(legal_masks: MLXArray, batch_size: int) -> MLXArray:
    masks = to_mlx(legal_masks)
    shape = tensor_shape(masks)
    if shape == (ACTION_SPACE_SIZE,):
        masks = masks[None, :]
        shape = tensor_shape(masks)
    if shape != (batch_size, ACTION_SPACE_SIZE):
        raise ValueError(
            f"legal_masks shape must be ({batch_size}, {ACTION_SPACE_SIZE}), got {shape}"
        )
    return masks.astype(mx.float32)
