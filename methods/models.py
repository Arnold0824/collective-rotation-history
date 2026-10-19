"""Symmetry-constrained regressions and categorical transition probabilities."""
from __future__ import annotations
from collections.abc import Sequence, Mapping, Hashable
from dataclasses import dataclass
from itertools import product
from typing import Self
import numpy as np
from scipy.interpolate import BSpline

def _parity_exponents(parity: np.ndarray, maximum_degree: int) -> np.ndarray:
    candidates = []
    dimension = len(parity)
    for degree in range(1, maximum_degree + 1):
        for exponent in product(range(degree + 1), repeat=dimension):
            if sum(exponent) != degree:
                continue
            exponent_array = np.asarray(exponent, dtype=np.int64)
            if int(np.prod(parity**exponent_array)) == -1:
                candidates.append(exponent_array)
    if not candidates:
        raise ValueError("no reflection-odd polynomial terms are available")
    return np.stack(candidates)


@dataclass(frozen=True)
class ParityRidgePredictor:
    state_center: np.ndarray
    state_scale: np.ndarray
    parity: np.ndarray
    exponents: np.ndarray
    design_scale: np.ndarray
    coefficients: np.ndarray
    ridge: float

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        target: np.ndarray,
        *,
        parity: Sequence[int],
        maximum_degree: int,
        ridge: float,
        minimum_scale: float = 1.0e-8,
    ) -> ParityRidgePredictor:
        values = np.asarray(states, dtype=np.float64)
        target_values = np.asarray(target, dtype=np.float64)
        parity_array = np.asarray(parity, dtype=np.int8)
        if values.ndim != 2 or target_values.shape != (len(values),):
            raise ValueError("ridge predictor inputs have incompatible shapes")
        if values.shape[1] != len(parity_array) or not np.all(np.isin(parity_array, (-1, 1))):
            raise ValueError("ridge state parity is invalid")
        center = np.where(parity_array == -1, 0.0, values.mean(axis=0))
        state_scale = np.sqrt(np.mean((values - center) ** 2, axis=0))
        state_scale = np.where(state_scale >= minimum_scale, state_scale, 1.0)
        standardized = (values - center) / state_scale
        exponents = _parity_exponents(parity_array, maximum_degree)
        design = np.prod(standardized[:, None, :] ** exponents[None, :, :], axis=-1)
        design_scale = np.sqrt(np.mean(design**2, axis=0))
        design_scale = np.where(design_scale >= minimum_scale, design_scale, 1.0)
        normalized = design / design_scale
        gram = normalized.T @ normalized + float(ridge) * np.eye(normalized.shape[1])
        coefficients = np.linalg.solve(gram, normalized.T @ target_values)
        return cls(
            state_center=center,
            state_scale=state_scale,
            parity=parity_array,
            exponents=exponents,
            design_scale=design_scale,
            coefficients=coefficients,
            ridge=float(ridge),
        )

    def predict(self, states: np.ndarray) -> np.ndarray:
        values = (np.asarray(states, dtype=np.float64) - self.state_center) / self.state_scale
        design = np.prod(values[:, None, :] ** self.exponents[None, :, :], axis=-1)
        return (design / self.design_scale) @ self.coefficients


@dataclass(frozen=True)
class ContinuousRows:
    current: np.ndarray
    delayed: np.ndarray
    increment: np.ndarray
    frame: np.ndarray
    run_id: np.ndarray


@dataclass(frozen=True)
class OddSplinePredictor:
    knots: np.ndarray
    design_scale: np.ndarray
    coefficients: np.ndarray
    ridge: float

    @staticmethod
    def design(values: np.ndarray, knots: np.ndarray):
        positive = BSpline.design_matrix(values, knots, 3).toarray()
        negative = BSpline.design_matrix(-values, knots, 3).toarray()
        return .5*(positive-negative)

    @classmethod
    def fit(cls, rows: list[ContinuousRows], n_knots: int, ridge: float):
        breaks = np.linspace(-1, 1, n_knots)
        knots = np.concatenate((np.repeat(-1., 3), breaks, np.repeat(1., 3)))
        x = np.concatenate([r.current for r in rows])
        target = np.concatenate([r.increment for r in rows])
        design = cls.design(x, knots)
        scale = np.sqrt(np.mean(design**2, axis=0))
        scale = np.where(scale > 1e-8, scale, 1)
        design = design/scale
        coef = np.linalg.solve(design.T@design+ridge*np.eye(design.shape[1]), design.T@target)
        return cls(knots, scale, coef, ridge)

    def predict(self, states: np.ndarray):
        return (self.design(states[:, 0], self.knots)/self.design_scale)@self.coefficients


def _positive_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_state_count(state_count: object) -> int:
    result = _positive_integer(state_count, name="state_count")
    if result % 2 == 0:
        raise ValueError("state_count must be odd")
    if result < 3:
        raise ValueError("state_count must be at least 3")
    return result


def _validate_alpha(alpha: object) -> float:
    if isinstance(alpha, (bool, np.bool_)):
        raise TypeError("alpha must be a positive finite number")
    result = float(alpha)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("alpha must be a positive finite number")
    return result


def _validated_context_targets(
    contexts: np.ndarray,
    targets: np.ndarray,
    *,
    order: int,
    state_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    supplied_contexts = np.asarray(contexts)
    supplied_targets = np.asarray(targets)
    if (
        supplied_contexts.ndim != 2
        or supplied_contexts.shape[1] < order
        or not len(supplied_contexts)
    ):
        raise ValueError("contexts must be a nonempty matrix with at least order columns")
    if supplied_targets.shape != (len(supplied_contexts),):
        raise ValueError("targets must contain one state per context")
    if not np.issubdtype(supplied_contexts.dtype, np.integer) or not np.issubdtype(
        supplied_targets.dtype, np.integer
    ):
        raise TypeError("contexts and targets must contain integer states")
    selected = supplied_contexts[:, :order].astype(np.int16, copy=False)
    selected_targets = supplied_targets.astype(np.int16, copy=False)
    if (
        np.any(selected < 0)
        or np.any(selected >= state_count)
        or np.any(selected_targets < 0)
        or np.any(selected_targets >= state_count)
    ):
        raise ValueError("a state lies outside the declared state space")
    return np.ascontiguousarray(selected), np.ascontiguousarray(selected_targets)


@dataclass(frozen=True)
class _SparseLevel:
    """Sorted integer context codes and their next-state count vectors."""

    codes: np.ndarray
    counts: np.ndarray


def _checked_state_space_size(current: int, state_count: int) -> int:
    maximum = np.iinfo(np.int64).max
    if current > maximum // state_count:
        raise ValueError("order and state_count exceed signed 64-bit context encoding")
    return current * state_count


def _build_sparse_levels(
    contexts: np.ndarray,
    targets: np.ndarray,
    *,
    state_count: int,
    reflection_pool: bool,
) -> tuple[_SparseLevel, ...]:
    """Count every prefix depth with vectorized integer encoding."""

    sample_count, order = contexts.shape
    codes = np.zeros(sample_count, dtype=np.int64)
    multiplier = 1
    levels: list[_SparseLevel] = []
    targets_i64 = targets.astype(np.int64, copy=False)
    for depth in range(order + 1):
        if reflection_pool:
            reflected_codes = multiplier - 1 - codes
            pooled_codes = np.concatenate((codes, reflected_codes))
            pooled_targets = np.concatenate((targets_i64, state_count - 1 - targets_i64))
        else:
            pooled_codes = codes
            pooled_targets = targets_i64
        unique_codes, inverse = np.unique(pooled_codes, return_inverse=True)
        flat_indices = inverse * state_count + pooled_targets
        counts = np.bincount(
            flat_indices,
            minlength=len(unique_codes) * state_count,
        ).reshape(len(unique_codes), state_count)
        levels.append(
            _SparseLevel(
                codes=np.ascontiguousarray(unique_codes, dtype=np.int64),
                counts=np.ascontiguousarray(counts, dtype=np.float64),
            )
        )
        if depth < order:
            codes += contexts[:, depth].astype(np.int64) * multiplier
            multiplier = _checked_state_space_size(multiplier, state_count)
    return tuple(levels)


class SparseBackoffMarkov:
    """A recursively smoothed finite-order categorical transition model."""

    def __init__(
        self,
        order: int,
        *,
        state_count: int = 3,
        alpha: float = 1.0,
        reflection_pool: bool = True,
    ) -> None:
        self.order = _positive_integer(order, name="order")
        self.state_count = _validate_state_count(state_count)
        self.alpha = _validate_alpha(alpha)
        if not isinstance(reflection_pool, (bool, np.bool_)):
            raise TypeError("reflection_pool must be boolean")
        self.reflection_pool = bool(reflection_pool)
        self._counts: tuple[_SparseLevel, ...] | None = None

    @property
    def fitted(self) -> bool:
        return self._counts is not None

    def fit(self, contexts: np.ndarray, targets: np.ndarray) -> Self:
        """Fit every context depth so unseen high-order contexts can back off."""

        selected, selected_targets = _validated_context_targets(
            contexts,
            targets,
            order=self.order,
            state_count=self.state_count,
        )
        self._counts = _build_sparse_levels(
            selected,
            selected_targets,
            state_count=self.state_count,
            reflection_pool=self.reflection_pool,
        )
        return self

    def _require_fitted(self) -> tuple[_SparseLevel, ...]:
        if self._counts is None:
            raise RuntimeError("the Markov model has not been fitted")
        return self._counts

    def _view(self, *, order: int, alpha: float) -> SparseBackoffMarkov:
        """Return a fitted lower-order/alternate-alpha view of shared counts."""

        selected_order = _positive_integer(order, name="order")
        counts = self._require_fitted()
        if selected_order > self.order:
            raise ValueError("a shared-count view cannot increase model order")
        result = SparseBackoffMarkov(
            selected_order,
            state_count=self.state_count,
            alpha=alpha,
            reflection_pool=self.reflection_pool,
        )
        result._counts = counts[: selected_order + 1]
        return result

    def predict_proba(self, contexts: np.ndarray) -> np.ndarray:
        """Predict the next-state distribution for each supplied context."""

        self._require_fitted()
        dummy = np.zeros(len(np.asarray(contexts)), dtype=np.int16)
        selected, _ = _validated_context_targets(
            contexts,
            dummy,
            order=self.order,
            state_count=self.state_count,
        )
        unique_contexts, inverse = np.unique(selected, axis=0, return_inverse=True)
        return self._predict_validated(unique_contexts)[inverse]

    def _predict_validated(self, contexts: np.ndarray) -> np.ndarray:
        """Vectorized prediction for already validated, duplicate-free contexts."""

        counts_by_depth = self._require_fitted()
        unique_contexts = np.asarray(contexts, dtype=np.int16)
        unique_count = len(unique_contexts)
        uniform = np.full(self.state_count, 1.0 / self.state_count, dtype=np.float64)
        base_counts = counts_by_depth[0].counts[0]
        probability = np.broadcast_to(
            (base_counts + self.alpha * uniform) / (float(base_counts.sum()) + self.alpha),
            (unique_count, self.state_count),
        ).copy()
        codes = np.zeros(unique_count, dtype=np.int64)
        multiplier = 1
        for depth in range(1, self.order + 1):
            codes += unique_contexts[:, depth - 1].astype(np.int64) * multiplier
            multiplier = _checked_state_space_size(multiplier, self.state_count)
            level = counts_by_depth[depth]
            positions = np.searchsorted(level.codes, codes)
            present = positions < len(level.codes)
            present[present] &= level.codes[positions[present]] == codes[present]
            if not np.any(present):
                continue
            selected_counts = level.counts[positions[present]]
            probability[present] = (selected_counts + self.alpha * probability[present]) / (
                selected_counts.sum(axis=1, keepdims=True) + self.alpha
            )
        return probability

    def nll(self, contexts: np.ndarray, targets: np.ndarray) -> float:
        """Return mean negative log probability on supplied rows."""

        _, selected_targets = _validated_context_targets(
            contexts,
            targets,
            order=self.order,
            state_count=self.state_count,
        )
        probabilities = self.predict_proba(contexts)
        selected = probabilities[np.arange(len(probabilities)), selected_targets]
        return float(-np.mean(np.log(selected)))


def build_order_run_ids(
    frames: np.ndarray,
    frame_valid: np.ndarray,
    *,
    minimum_run_length: int = 2,
) -> np.ndarray:
    """Assign consecutive positive run IDs and ``-1`` to unusable frames."""

    frame_values = np.asarray(frames, dtype=np.int64)
    valid = np.asarray(frame_valid, dtype=bool)
    if frame_values.ndim != 1 or valid.shape != frame_values.shape or len(frame_values) == 0:
        raise ValueError("frames and frame_valid must be nonempty one-dimensional arrays")
    if not np.all(np.diff(frame_values) > 0):
        raise ValueError("frames must be strictly increasing")
    if minimum_run_length < 1:
        raise ValueError("minimum_run_length must be positive")

    output = np.full(len(frame_values), -1, dtype=np.int64)
    next_run_id = 1
    start: int | None = None
    for index in range(len(frame_values) + 1):
        continues = (
            index < len(frame_values)
            and valid[index]
            and (index == 0 or start is None or frame_values[index] == frame_values[index - 1] + 1)
        )
        if continues:
            if start is None:
                start = index
            continue
        if start is not None:
            if index - start >= minimum_run_length:
                output[start:index] = next_run_id
                next_run_id += 1
            start = None
        if index < len(frame_values) and valid[index]:
            start = index
    return output
