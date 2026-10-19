"""Saved bounded transition models and original native-frame sampling routines.

The regression evaluation, parameter loading and random sampling follow the
original delay-transition and scalar simulation implementations. Model JSON and
NPZ files retain their source metadata and checksums.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import numpy as np



def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _readonly_float_array(values: np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result = np.ascontiguousarray(array.copy())
    result.setflags(write=False)
    return result


def _readonly_int_array(values: np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional integer array")
    result = np.ascontiguousarray(array.copy())
    result.setflags(write=False)
    return result


def evaluate_library(states: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != exponents.shape[1]:
        raise ValueError("state and exponent dimensions differ")
    return np.prod(values[:, None, :] ** exponents[None, :, :], axis=-1)


@dataclass(frozen=True)
class DelayTransitionModel:

    """Parity-constrained bounded transition for a scalar process with fixed delay.

    The conditional distribution is a scaled beta distribution.  Its mean is an
    odd linked polynomial, while its positive scale is controlled by an even
    concentration model.  Native-rate simulation retains an explicit history
    buffer and never evolves the delayed coordinate independently.
    """

    delay_steps: int

    horizon_steps: int

    maximum_degree: int

    mean_ridge: float

    scale_ridge: float

    lower_bound: float

    upper_bound: float

    boundary_epsilon: float

    concentration_floor: float

    scale_calibration_enabled: bool

    scale_log_concentration_offset: float

    scale_calibration_lower_bound: float

    scale_calibration_upper_bound: float

    scale_calibration_tolerance: float

    scale_calibration_negative_log_score_before: float

    scale_calibration_negative_log_score_after: float

    state_scale: np.ndarray

    mean_exponents: np.ndarray

    mean_design_scale: np.ndarray

    mean_coefficients: np.ndarray

    scale_exponents: np.ndarray

    scale_design_scale: np.ndarray

    scale_coefficients: np.ndarray

    def __post_init__(self) -> None:
        delay_steps = _positive_integer(self.delay_steps, name="delay_steps")
        horizon_steps = _positive_integer(self.horizon_steps, name="horizon_steps")
        if self.maximum_degree < 1:
            raise ValueError("maximum_degree must be positive")
        if not np.isfinite(self.mean_ridge) or self.mean_ridge < 0:
            raise ValueError("mean_ridge must be finite and nonnegative")
        if not np.isfinite(self.scale_ridge) or self.scale_ridge < 0:
            raise ValueError("scale_ridge must be finite and nonnegative")
        if not (
            np.isfinite(self.lower_bound)
            and np.isfinite(self.upper_bound)
            and self.lower_bound == -self.upper_bound
            and self.upper_bound > 0
        ):
            raise ValueError("bounds must be finite, increasing, and symmetric about zero")
        if not 0 < self.boundary_epsilon < 0.5:
            raise ValueError("boundary_epsilon must lie strictly between 0 and 0.5")
        if not np.isfinite(self.concentration_floor) or self.concentration_floor <= 0:
            raise ValueError("concentration_floor must be positive and finite")
        if not isinstance(self.scale_calibration_enabled, (bool, np.bool_)):
            raise TypeError("scale_calibration_enabled must be boolean")
        calibration_values = (
            self.scale_log_concentration_offset,
            self.scale_calibration_lower_bound,
            self.scale_calibration_upper_bound,
            self.scale_calibration_tolerance,
            self.scale_calibration_negative_log_score_before,
            self.scale_calibration_negative_log_score_after,
        )
        if not all(np.isfinite(value) for value in calibration_values):
            raise ValueError("scale calibration values must be finite")
        if self.scale_calibration_lower_bound >= self.scale_calibration_upper_bound:
            raise ValueError("scale calibration bounds must be increasing")
        if self.scale_calibration_tolerance <= 0:
            raise ValueError("scale calibration tolerance must be positive")
        if not (
            self.scale_calibration_lower_bound
            <= self.scale_log_concentration_offset
            <= self.scale_calibration_upper_bound
        ):
            raise ValueError("scale calibration offset lies outside its search bounds")
        if not self.scale_calibration_enabled and self.scale_log_concentration_offset != 0:
            raise ValueError("disabled scale calibration must have zero offset")

        state_scale = _readonly_float_array(self.state_scale, ndim=1, name="state_scale")
        mean_exponents = _readonly_int_array(self.mean_exponents, ndim=2, name="mean_exponents")
        mean_design_scale = _readonly_float_array(
            self.mean_design_scale, ndim=1, name="mean_design_scale"
        )
        mean_coefficients = _readonly_float_array(
            self.mean_coefficients, ndim=1, name="mean_coefficients"
        )
        scale_exponents = _readonly_int_array(self.scale_exponents, ndim=2, name="scale_exponents")
        scale_design_scale = _readonly_float_array(
            self.scale_design_scale, ndim=1, name="scale_design_scale"
        )
        scale_coefficients = _readonly_float_array(
            self.scale_coefficients, ndim=1, name="scale_coefficients"
        )
        if state_scale.shape != (2,) or np.any(state_scale <= 0):
            raise ValueError("state_scale must contain two positive values")
        for name, exponents in (
            ("mean_exponents", mean_exponents),
            ("scale_exponents", scale_exponents),
        ):
            if exponents.shape[1] != 2 or np.any(exponents < 0):
                raise ValueError(f"{name} must have shape (terms, 2) and be nonnegative")
        if not np.all(np.sum(mean_exponents, axis=1) % 2 == 1):
            raise ValueError("mean terms must be odd under simultaneous reflection")
        if not np.all(np.sum(scale_exponents, axis=1) % 2 == 0):
            raise ValueError("scale terms must be even under simultaneous reflection")
        if mean_design_scale.shape != (len(mean_exponents),) or mean_coefficients.shape != (
            len(mean_exponents),
        ):
            raise ValueError("mean arrays have inconsistent term counts")
        if scale_design_scale.shape != (len(scale_exponents),) or scale_coefficients.shape != (
            len(scale_exponents),
        ):
            raise ValueError("scale arrays have inconsistent term counts")
        if np.any(mean_design_scale <= 0) or np.any(scale_design_scale <= 0):
            raise ValueError("design scales must be positive")
        object.__setattr__(self, "state_scale", state_scale)
        object.__setattr__(self, "mean_exponents", mean_exponents)
        object.__setattr__(self, "mean_design_scale", mean_design_scale)
        object.__setattr__(self, "mean_coefficients", mean_coefficients)
        object.__setattr__(self, "scale_exponents", scale_exponents)
        object.__setattr__(self, "scale_design_scale", scale_design_scale)
        object.__setattr__(self, "scale_coefficients", scale_coefficients)
        object.__setattr__(self, "delay_steps", delay_steps)
        object.__setattr__(self, "horizon_steps", horizon_steps)

    @property
    def bound(self) -> float:
        return float(self.upper_bound)

    def _states(self, states: np.ndarray) -> tuple[np.ndarray, bool]:
        values = np.asarray(states, dtype=np.float64)
        one_state = values.ndim == 1
        if one_state:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("states must have shape (2,) or (samples, 2) and be finite")
        if np.any(values < self.lower_bound) or np.any(values > self.upper_bound):
            raise ValueError("conditioning states lie outside the physical bounds")
        return values, one_state

    def _raw_mean(self, values: np.ndarray) -> np.ndarray:
        standardized = values / self.state_scale
        design = evaluate_library(standardized, self.mean_exponents)
        return (design / self.mean_design_scale) @ self.mean_coefficients

    @staticmethod
    def _reflection_canonical_states(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map every simultaneous-reflection pair to one numerical representative."""

        first = values[:, 0]
        second = values[:, 1]
        orientation = np.where(
            first < 0.0,
            -1.0,
            np.where(first > 0.0, 1.0, np.where(second < 0.0, -1.0, 1.0)),
        )
        return values * orientation[:, None], orientation

    def conditional_mean(self, states: np.ndarray) -> np.ndarray | float:
        """Return the bounded reflection-odd conditional mean."""

        values, one_state = self._states(states)
        canonical, orientation = self._reflection_canonical_states(values)
        link_limit = np.arctanh(1.0 - self.boundary_epsilon)
        linked = np.tanh(np.clip(self._raw_mean(canonical), -link_limit, link_limit))
        result = orientation * self.bound * linked
        return float(result[0]) if one_state else result

    def conditional_concentration(self, states: np.ndarray) -> np.ndarray | float:
        """Return the positive reflection-even beta concentration."""

        values, one_state = self._states(states)
        canonical, _ = self._reflection_canonical_states(values)
        standardized = canonical / self.state_scale
        design = evaluate_library(standardized, self.scale_exponents)
        raw = (
            design / self.scale_design_scale
        ) @ self.scale_coefficients + self.scale_log_concentration_offset
        result = np.exp(np.clip(raw, -30.0, 30.0)) + self.concentration_floor
        return float(result[0]) if one_state else result

    def sample_next(
        self,
        states: np.ndarray,
        *,
        seed: int | None = None,
        generator: np.random.Generator | None = None,
    ) -> np.ndarray | float:
        """Sample a bounded future value without clipping or reflection."""

        if seed is not None and generator is not None:
            raise ValueError("provide either seed or generator, not both")
        values, one_state = self._states(states)
        random = generator if generator is not None else np.random.default_rng(seed)
        mean = np.asarray(self.conditional_mean(values), dtype=np.float64)
        concentration = np.asarray(self.conditional_concentration(values), dtype=np.float64)
        probability = (mean - self.lower_bound) / (self.upper_bound - self.lower_bound)
        alpha = probability * concentration
        beta = (1.0 - probability) * concentration
        sampled = self.lower_bound + (self.upper_bound - self.lower_bound) * random.beta(
            alpha, beta
        )
        if not np.isfinite(sampled).all():
            raise RuntimeError("bounded transition produced a non-finite sample")
        if np.any(sampled < self.lower_bound) or np.any(sampled > self.upper_bound):
            raise RuntimeError("bounded transition violated its support")
        return float(sampled[0]) if one_state else sampled

    def _history(self, history: np.ndarray) -> np.ndarray:
        values = np.asarray(history, dtype=np.float64)
        if values.shape != (self.delay_steps + 1,) or not np.isfinite(values).all():
            raise ValueError("history must contain delay_steps + 1 finite values, oldest to newest")
        if np.any(values < self.lower_bound) or np.any(values > self.upper_bound):
            raise ValueError("history values lie outside the physical bounds")
        return values

    def advance_native_history(self, history: np.ndarray, new_value: float) -> np.ndarray:
        """Append one new value and deterministically shift the delay buffer."""

        values = self._history(history)
        future = float(new_value)
        if not np.isfinite(future) or not self.lower_bound <= future <= self.upper_bound:
            raise ValueError("new_value lies outside the physical bounds")
        return np.concatenate((values[1:], np.asarray([future])))

    def simulate_native(
        self,
        initial_history: np.ndarray,
        *,
        steps: int,
        seed: int,
    ) -> np.ndarray:
        """Simulate at the native frame rate using an exact finite-delay buffer."""

        if self.horizon_steps != 1:
            raise ValueError("native-rate simulation requires horizon_steps == 1")
        if steps < 1:
            raise ValueError("steps must be positive")
        history = self._history(initial_history).copy()
        random = np.random.default_rng(seed)
        trajectory = np.empty(len(history) + steps, dtype=np.float64)
        trajectory[: len(history)] = history
        for index in range(steps):
            state = np.asarray([history[-1], history[0]], dtype=np.float64)
            new_value = float(self.sample_next(state, generator=random))
            history = self.advance_native_history(history, new_value)
            trajectory[self.delay_steps + 1 + index] = new_value
        return trajectory

    @classmethod
    def load(cls, path: str | Path) -> DelayTransitionModel:
        """Load a model saved by :meth:`save` with object loading disabled."""

        with np.load(Path(path), allow_pickle=False) as archive:
            if int(archive["schema_version"]) != 2:
                raise ValueError("unsupported delay transition schema version")
            return cls(
                delay_steps=int(archive["delay_steps"]),
                horizon_steps=int(archive["horizon_steps"]),
                maximum_degree=int(archive["maximum_degree"]),
                mean_ridge=float(archive["mean_ridge"]),
                scale_ridge=float(archive["scale_ridge"]),
                lower_bound=float(archive["lower_bound"]),
                upper_bound=float(archive["upper_bound"]),
                boundary_epsilon=float(archive["boundary_epsilon"]),
                concentration_floor=float(archive["concentration_floor"]),
                scale_calibration_enabled=bool(archive["scale_calibration_enabled"]),
                scale_log_concentration_offset=float(archive["scale_log_concentration_offset"]),
                scale_calibration_lower_bound=float(archive["scale_calibration_lower_bound"]),
                scale_calibration_upper_bound=float(archive["scale_calibration_upper_bound"]),
                scale_calibration_tolerance=float(archive["scale_calibration_tolerance"]),
                scale_calibration_negative_log_score_before=float(
                    archive["scale_calibration_negative_log_score_before"]
                ),
                scale_calibration_negative_log_score_after=float(
                    archive["scale_calibration_negative_log_score_after"]
                ),
                state_scale=archive["state_scale"],
                mean_exponents=archive["mean_exponents"],
                mean_design_scale=archive["mean_design_scale"],
                mean_coefficients=archive["mean_coefficients"],
                scale_exponents=archive["scale_exponents"],
                scale_design_scale=archive["scale_design_scale"],
                scale_coefficients=archive["scale_coefficients"],
            )


MODEL_SCHEMA_VERSION = 1


REQUIRED_DELAY_TRANSITION_METADATA = frozenset(
    {
        "schema_version",
        "model_id",
        "model_type",
        "conditioning_mode",
        "group_size",
        "training_trial",
        "delay_steps",
        "horizon_steps",
        "frame_rate_hz",
        "native_history_length",
        "maximum_degree",
        "mean_ridge",
        "scale_ridge",
        "scale_calibration_enabled",
        "scale_log_concentration_offset",
        "scale_calibration_lower_bound",
        "scale_calibration_upper_bound",
        "scale_calibration_tolerance",
        "scale_calibration_negative_log_score_before",
        "scale_calibration_negative_log_score_after",
        "scale_calibration_boundary_distance",
        "scale_calibration_boundary_margin",
        "scale_calibration_boundary_pass",
        "physical_bounds",
        "training_condition_support_lower",
        "training_condition_support_upper",
        "training_target_support",
        "eligibility_rule_id",
        "source_commit",
        "configuration_sha256",
        "input_sha256",
        "model_arrays",
        "model_arrays_sha256",
        "pickle_used",
        "ordinary_two_dimensional_sde_used",
        "simulation_interface",
        "second_output_copies_first_input_exactly",
        "all_second_input_coefficients_zero",
    }
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_file(base: Path, encoded_path: object) -> Path:
    value = Path(str(encoded_path))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("serialized model paths must be relative and cannot traverse upward")
    resolved = (base / value).resolve()
    if not resolved.is_relative_to(base.resolve()) or not resolved.is_file():
        raise ValueError("serialized model path is unavailable or outside its bundle")
    return resolved


@dataclass(frozen=True)
class FrozenDelayTransition:
    """One verified bounded delay-transition model and its metadata."""

    model: Any
    metadata: dict[str, Any]
    metadata_path: Path

    def sample_current_only(
        self,
        current_values: np.ndarray | float,
        *,
        seed: int | None = None,
        generator: np.random.Generator | None = None,
    ) -> np.ndarray | float:
        """Sample without constructing or consulting a delayed history coordinate."""

        if self.metadata.get("conditioning_mode") != "current_only":
            raise ValueError("sample_current_only requires a current-only model")
        current = np.asarray(current_values, dtype=np.float64)
        one_value = current.ndim == 0
        flattened = current.reshape(-1)
        states = np.column_stack((flattened, np.zeros(len(flattened))))
        sampled = self.model.sample_next(states, seed=seed, generator=generator)
        result = np.asarray(sampled, dtype=np.float64).reshape(current.shape)
        return float(result) if one_value else result

    def simulate_current_only(
        self,
        initial_value: float,
        *,
        steps: int,
        seed: int,
    ) -> np.ndarray:
        """Generate a scalar chain that never allocates or reads a history buffer."""

        if steps < 1:
            raise ValueError("steps must be positive")
        current = float(initial_value)
        if not np.isfinite(current):
            raise ValueError("initial_value must be finite")
        random = np.random.default_rng(seed)
        trajectory = np.empty(steps + 1, dtype=np.float64)
        trajectory[0] = current
        for index in range(steps):
            current = float(self.sample_current_only(current, generator=random))
            trajectory[index + 1] = current
        return trajectory


def load_frozen_delay_transition(
    metadata_path: str | Path,
    *,
    expected_metadata_sha256: str | None = None,
) -> FrozenDelayTransition:
    """Load one bounded delay model through its checked JSON declaration."""


    path = Path(metadata_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_metadata_sha256 is not None and sha256_file(path) != str(expected_metadata_sha256):
        raise ValueError("delay-transition metadata hash mismatch")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("delay-transition metadata must be a JSON object")
    missing = sorted(REQUIRED_DELAY_TRANSITION_METADATA - set(document))
    if missing:
        raise ValueError(f"delay-transition metadata is missing fields: {missing}")
    if int(document["schema_version"]) != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported delay-transition metadata schema")
    if document.get("model_type") != "bounded_discrete_scalar_transition":
        raise ValueError("unexpected delay-transition model type")
    if document.get("training_trial") != "D":
        raise ValueError("frozen delay-transition training trial must be D")
    if (
        document.get("pickle_used") is not False
        or document.get("ordinary_two_dimensional_sde_used") is not False
    ):
        raise ValueError("delay-transition metadata violates frozen safety rules")
    if int(document["horizon_steps"]) not in (1, 5):
        raise ValueError("delay-transition horizon must be one or five frames")
    conditioning_mode = str(document["conditioning_mode"])
    if conditioning_mode not in {"current_only", "current_and_delay"}:
        raise ValueError("unexpected bounded-transition conditioning mode")
    if conditioning_mode == "current_only":
        expected_interface = "current_only_scalar_transition"
    elif int(document["horizon_steps"]) == 1:
        expected_interface = "six_value_native_rate_delay_buffer"
    else:
        expected_interface = "strict_five_frame_delay_step_map"
    if document.get("simulation_interface") != expected_interface:
        raise ValueError("unexpected delay-transition simulation interface")
    if bool(document["second_output_copies_first_input_exactly"]) is not (
        conditioning_mode == "current_and_delay" and int(document["horizon_steps"]) == 5
    ):
        raise ValueError("delay-transition strict-copy declaration is inconsistent")
    arrays_path = _safe_relative_file(path.parent, document.get("model_arrays"))
    if sha256_file(arrays_path) != str(document.get("model_arrays_sha256")):
        raise ValueError("delay-transition array hash mismatch")
    model = DelayTransitionModel.load(arrays_path)
    if (
        int(document["delay_steps"]) != model.delay_steps
        or int(document["horizon_steps"]) != model.horizon_steps
    ):
        raise ValueError("delay-transition lag metadata differs from its arrays")
    if int(document["native_history_length"]) != model.delay_steps + 1:
        raise ValueError("delay-transition native history length is invalid")
    if not np.allclose(
        np.asarray(document["physical_bounds"], dtype=np.float64),
        np.asarray([model.lower_bound, model.upper_bound]),
    ):
        raise ValueError("delay-transition physical bounds differ from its arrays")
    condition_lower = np.asarray(document["training_condition_support_lower"], dtype=np.float64)
    condition_upper = np.asarray(document["training_condition_support_upper"], dtype=np.float64)
    target_support = np.asarray(document["training_target_support"], dtype=np.float64)
    if (
        condition_lower.shape != (2,)
        or condition_upper.shape != (2,)
        or target_support.shape != (2,)
        or not np.isfinite(condition_lower).all()
        or not np.isfinite(condition_upper).all()
        or not np.isfinite(target_support).all()
        or np.any(condition_lower > condition_upper)
        or target_support[0] > target_support[1]
        or np.any(condition_lower < model.lower_bound)
        or np.any(condition_upper > model.upper_bound)
        or target_support[0] < model.lower_bound
        or target_support[1] > model.upper_bound
    ):
        raise ValueError("delay-transition training support violates physical bounds")
    calibration_fields = (
        ("scale_calibration_enabled", model.scale_calibration_enabled),
        ("scale_log_concentration_offset", model.scale_log_concentration_offset),
        ("scale_calibration_lower_bound", model.scale_calibration_lower_bound),
        ("scale_calibration_upper_bound", model.scale_calibration_upper_bound),
        ("scale_calibration_tolerance", model.scale_calibration_tolerance),
        (
            "scale_calibration_negative_log_score_before",
            model.scale_calibration_negative_log_score_before,
        ),
        (
            "scale_calibration_negative_log_score_after",
            model.scale_calibration_negative_log_score_after,
        ),
    )
    for field, actual in calibration_fields:
        declared = document[field]
        if isinstance(actual, (bool, np.bool_)):
            if bool(declared) is not bool(actual):
                raise ValueError(f"delay-transition {field} differs from its arrays")
        elif not np.isclose(float(declared), float(actual)):
            raise ValueError(f"delay-transition {field} differs from its arrays")
    boundary_distance = min(
        model.scale_log_concentration_offset - model.scale_calibration_lower_bound,
        model.scale_calibration_upper_bound - model.scale_log_concentration_offset,
    )
    boundary_margin = float(document["scale_calibration_boundary_margin"])
    if (
        not np.isclose(float(document["scale_calibration_boundary_distance"]), boundary_distance)
        or not np.isfinite(boundary_margin)
        or boundary_margin <= 0
    ):
        raise ValueError("delay-transition calibration boundary diagnostic is invalid")
    boundary_pass = bool(boundary_distance > boundary_margin)
    if bool(document["scale_calibration_boundary_pass"]) != boundary_pass:
        raise ValueError("delay-transition calibration boundary decision is inconsistent")
    mean_delay_terms = model.mean_exponents[:, 1] > 0
    scale_delay_terms = model.scale_exponents[:, 1] > 0
    coefficients_are_zero = bool(
        np.all(model.mean_coefficients[mean_delay_terms] == 0.0)
        and np.all(model.scale_coefficients[scale_delay_terms] == 0.0)
    )
    if bool(document["all_second_input_coefficients_zero"]) != coefficients_are_zero:
        raise ValueError("bounded-transition coefficient declaration differs from arrays")
    if conditioning_mode == "current_only":
        if not coefficients_are_zero:
            raise ValueError("current-only transition contains a delayed-input coefficient")
        if condition_lower[1] != 0.0 or condition_upper[1] != 0.0:
            raise ValueError("current-only transition was not trained with zero second input")
    return FrozenDelayTransition(model=model, metadata=document, metadata_path=path)


def _simulate_bounded_h1(
    frozen: FrozenDelayTransition,
    initial_history: np.ndarray,
    *,
    total_frames: int,
    conditioning_mode: str,
    seed: int,
) -> np.ndarray:
    model = frozen.model
    if model.horizon_steps != 1 or model.delay_steps != 5:
        raise ValueError("long-time bounded simulation requires the h=1 delay-five model")
    if total_frames <= model.delay_steps + 1:
        raise ValueError("long-time bounded simulation is too short")
    history = np.asarray(initial_history, dtype=np.float64).copy()
    if conditioning_mode == "current_only":
        return frozen.simulate_current_only(float(history[-1]), steps=total_frames - 1, seed=seed)
    if conditioning_mode != "current_and_delay":
        raise ValueError("unknown bounded-transition conditioning mode")
    full = model.simulate_native(history, steps=total_frames - 1, seed=seed)
    return full[model.delay_steps :]


def load_model(json_path: str | Path) -> FrozenDelayTransition:
    """Load a bounded model and verify its declared parameter checksum."""
    return load_frozen_delay_transition(json_path)


def simulate(
    model: FrozenDelayTransition,
    initial_history: np.ndarray,
    conditioning_mode: str,
    seed: int,
    total_frames: int = 4500,
) -> np.ndarray:
    """Generate the original scalar or delay-buffer trajectory, including its origin."""
    return _simulate_bounded_h1(
        model,
        initial_history,
        total_frames=total_frames,
        conditioning_mode=conditioning_mode,
        seed=seed,
    )
