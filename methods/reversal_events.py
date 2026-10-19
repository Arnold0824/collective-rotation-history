"""Reversal events, valid exposure, and supported waiting-time comparisons.

Rotational-state thresholds are inclusive of equality.
This module never bridges missing frames or nonfinite samples. Simulation
replicates must be pooled within a recording before estimating its curve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "event_index", "segment_index", "direction", "source_state", "target_state",
    "start_frame", "end_frame", "start_seconds", "end_seconds",
    "duration_seconds", "event_observed", "start_known", "start_age_unknown",
    "end_reason", "ended_at_segment_boundary", "ended_by_interruption",
]
OBSERVATION_COLUMNS = [
    "segment_index", "first_frame", "last_frame", "first_seconds", "last_seconds",
    "n_frames", "fps", "observation_seconds", "first_direction_frame",
    "armed_observation_seconds", "initial_neutral_seconds",
    "start_reason", "end_reason", "ended_by_interruption",
]
CURVE_COLUMNS = [
    "time_seconds", "at_risk", "events", "censored", "survival_before",
    "survival", "at_risk_after",
]


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _integer_vector(values: Sequence[int], name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one dimensional")
    if raw.dtype.kind not in "iu":
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{name} must contain finite integer values")
    return raw.astype(np.int64)


def extract_reversal_tables(
    values: Sequence[float],
    frame_ids: Sequence[int],
    fps: float,
    *,
    metadata: Mapping[str, Any] | None = None,
    segment_ids: Sequence[Any] | None = None,
    threshold: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return event rows and an independent table of continuous valid exposure.

    A passage starts at the first value >= threshold or <= -threshold; only
    reaching the opposite threshold completes it. Returning to the same side
    after visiting the central region does not restart its clock. A passage
    beginning at the first sample of a valid segment has unknown start age.

    Nonfinite values, nonconsecutive frame IDs, and changes of segment_ids
    break continuity. Frame IDs must be strictly increasing. Every valid
    segment, including all-neutral and singleton segments, gets an exposure
    row. Its exposure is (last_frame - first_frame) / fps; no missing interval
    contributes. Absolute seconds are frame_id / fps, with no origin shift.

    metadata is copied to both tables (e.g. recording_id, model, replicate,
    group_size); it must not overwrite the defined columns.
    """
    x = np.asarray(values, dtype=np.float64)
    frames = _integer_vector(frame_ids, "frame_ids")
    rate = _positive(fps, "fps")
    edge = _positive(threshold, "threshold")
    if x.ndim != 1 or len(x) != len(frames):
        raise ValueError("values and frame_ids must be matching vectors")
    if np.any(np.diff(frames) <= 0):
        raise ValueError("frame_ids must be strictly increasing")
    labels = None if segment_ids is None else np.asarray(segment_ids)
    if labels is not None and (labels.ndim != 1 or len(labels) != len(x)):
        raise ValueError("segment_ids must match values")
    meta = dict(metadata or {})
    if set(meta) & set(EVENT_COLUMNS + OBSERVATION_COLUMNS):
        raise ValueError("metadata must not overwrite event or observation columns")
    finite = np.isfinite(x)
    event_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []

    def boundary_reason(left: int, right: int) -> str:
        if left < 0:
            return "recording_start"
        if right >= len(x):
            return "recording_end"
        if not finite[left] or not finite[right]:
            return "nonfinite_value"
        if frames[right] != frames[left] + 1:
            return "frame_gap"
        if labels is not None and labels[left] != labels[right]:
            return "segment_boundary"
        return "continuous"

    segments: list[tuple[int, int]] = []
    first: int | None = None
    for index in range(len(x)):
        if not finite[index]:
            if first is not None:
                segments.append((first, index - 1))
                first = None
            continue
        if first is None:
            first = index
        elif boundary_reason(index - 1, index) != "continuous":
            segments.append((first, index - 1))
            first = index
    if first is not None:
        segments.append((first, len(x) - 1))

    for segment_index, (first, last) in enumerate(segments):
        start_reason = boundary_reason(first - 1, first)
        end_reason = boundary_reason(last, last + 1)
        interrupted = end_reason in {"frame_gap", "nonfinite_value", "segment_boundary"}
        state = 0
        start = -1
        first_direction: int | None = None
        known = False

        def append_event(end: int, completed: bool) -> None:
            source = "positive" if state > 0 else "negative"
            target = "negative" if state > 0 else "positive"
            event_rows.append({
                **meta,
                "event_index": len(event_rows),
                "segment_index": segment_index,
                "direction": f"{source}_to_{target}",
                "source_state": source,
                "target_state": target,
                "start_frame": int(frames[start]),
                "end_frame": int(frames[end]),
                "start_seconds": float(frames[start] / rate),
                "end_seconds": float(frames[end] / rate),
                "duration_seconds": float((frames[end] - frames[start]) / rate),
                "event_observed": completed,
                "start_known": known,
                "start_age_unknown": not known,
                "end_reason": "opposite_threshold" if completed else end_reason,
                "ended_at_segment_boundary": end == last,
                "ended_by_interruption": bool(not completed and interrupted),
            })

        for index in range(first, last + 1):
            current = 1 if x[index] >= edge else (-1 if x[index] <= -edge else 0)
            if state == 0:
                if current:
                    state, start = current, index
                    first_direction = index
                    known = index > first
            elif current and current != state:
                append_event(index, True)
                state, start, known = current, index, True
        if state:
            append_event(last, False)
        exposure = float((frames[last] - frames[first]) / rate)
        armed = 0.0 if first_direction is None else float((frames[last] - frames[first_direction]) / rate)
        observation_rows.append({
            **meta,
            "segment_index": segment_index,
            "first_frame": int(frames[first]),
            "last_frame": int(frames[last]),
            "first_seconds": float(frames[first] / rate),
            "last_seconds": float(frames[last] / rate),
            "n_frames": last - first + 1,
            "fps": rate,
            "observation_seconds": exposure,
            "first_direction_frame": None if first_direction is None else int(frames[first_direction]),
            "armed_observation_seconds": armed,
            "initial_neutral_seconds": exposure - armed,
            "start_reason": start_reason,
            "end_reason": end_reason,
            "ended_by_interruption": interrupted,
        })
    return (
        pd.DataFrame(event_rows, columns=[*meta, *EVENT_COLUMNS]),
        pd.DataFrame(observation_rows, columns=[*meta, *OBSERVATION_COLUMNS]),
    )


def _event_arrays(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for column in ["duration_seconds", "event_observed", "start_known"]:
        if column not in events:
            raise ValueError(f"missing required event field {column}")
    durations = events["duration_seconds"].to_numpy(dtype=float)
    if not np.isfinite(durations).all() or np.any(durations < 0):
        raise ValueError("durations must be finite and nonnegative")
    for column in ["event_observed", "start_known"]:
        if not events[column].isin([True, False]).all():
            raise ValueError(f"{column} must contain Boolean values")
    return (
        durations,
        events["event_observed"].to_numpy(dtype=bool),
        events["start_known"].to_numpy(dtype=bool),
    )


def counts_and_rate(events: pd.DataFrame, observation: pd.DataFrame) -> dict[str, Any]:
    """Use all observed reversals and all continuous valid time for event rate."""
    _, observed, known = _event_arrays(events)
    exposure = observation["observation_seconds"].to_numpy(dtype=float)
    if not np.isfinite(exposure).all() or np.any(exposure < 0):
        raise ValueError("observation_seconds must be finite and nonnegative")
    seconds = float(exposure.sum())
    completed = int(observed.sum())
    return {
        "segment_count": len(observation),
        "observation_seconds": seconds,
        "completed_count": completed,
        "right_censored_count": int((~observed).sum()),
        "known_start_completed_count": int((observed & known).sum()),
        "known_start_right_censored_count": int((~observed & known).sum()),
        "unknown_start_count": int((~known).sum()),
        "unknown_start_completed_count": int((observed & ~known).sum()),
        "unknown_start_right_censored_count": int((~observed & ~known).sum()),
        "reversals_per_minute": 60.0 * completed / seconds if seconds > 0 else None,
    }


def km_from_events(events: pd.DataFrame) -> dict[str, Any]:
    """Kaplan--Meier step curve from known starts, retaining right censoring.

    At tied times, events use all observations still at risk immediately
    before that time; censored observations are then removed. Unknown-start
    rows never enter the waiting-time risk set, even if a reversal is seen.
    Zero-duration terminal censoring is retained for accounting and removed
    at time zero. No confidence interval is inferred by treating events or
    simulation seeds as independent biological replications.
    """
    all_durations, all_observed, known = _event_arrays(events)
    durations, observed = all_durations[known], all_observed[known]
    n = len(durations)
    rows: list[dict[str, Any]] = []
    survival = 1.0
    at_risk = n
    median: float | None = None
    zero_time: float | None = None
    if n and float(durations.min()) > 0:
        rows.append(dict(time_seconds=0.0, at_risk=n, events=0, censored=0,
                         survival_before=1.0, survival=1.0, at_risk_after=n))
    if n:
        order = np.argsort(durations, kind="stable")
        sorted_times, sorted_observed = durations[order], observed[order]
        times, starts, counts = np.unique(sorted_times, return_index=True, return_counts=True)
        for time, first, count in zip(times, starts, counts):
            failures = int(sorted_observed[first:first + count].sum())
            censors = int(count - failures)
            before = survival
            survival *= 1.0 - failures / at_risk
            if median is None and survival <= 0.5:
                median = float(time)
            if zero_time is None and failures == at_risk:
                zero_time = float(time)
            rows.append(dict(time_seconds=float(time), at_risk=int(at_risk), events=failures,
                             censored=censors, survival_before=before, survival=survival,
                             at_risk_after=int(at_risk - count)))
            at_risk -= count
    summary = {
        "sample_count": len(events),
        "analysis_sample_count": n,
        "completed_count": int(observed.sum()),
        "censored_count": int((~observed).sum()),
        "unknown_start_count": int((~known).sum()),
        "unknown_start_completed_count": int((all_observed & ~known).sum()),
        "unknown_start_censored_count": int((~all_observed & ~known).sum()),
        "support_end_seconds": float(durations.max()) if n else None,
        "terminal_survival": float(survival) if n else None,
        "zero_time_seconds": zero_time,
        "median_seconds": median,
        "status": "available" if n else "no_known_start",
    }
    return {"curve": pd.DataFrame(rows, columns=CURVE_COLUMNS), "summary": summary}


def evaluate_survival(km: Mapping[str, Any], times: Sequence[float]) -> pd.DataFrame:
    """Evaluate the right-continuous curve; unsupported positive tails are NaN."""
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or not np.isfinite(grid).all() or np.any(grid < 0):
        raise ValueError("times must be finite and nonnegative")
    curve, summary = km["curve"], km["summary"]
    output: list[dict[str, Any]] = []
    n = summary["analysis_sample_count"]
    if not n:
        return pd.DataFrame({"time_seconds": grid, "survival": np.nan,
                             "at_risk": 0, "supported": False, "zero_extended": False})
    knots = curve["time_seconds"].to_numpy(dtype=float)
    for time in grid:
        left = int(np.searchsorted(knots, time, side="left"))
        right = int(np.searchsorted(knots, time, side="right")) - 1
        value = 1.0 if right < 0 else float(curve.iloc[right]["survival"])
        risk = n if left == 0 else int(curve.iloc[left - 1]["at_risk_after"])
        beyond = time > summary["support_end_seconds"] + 1e-12 * max(1.0, float(time))
        zero_extended = bool(beyond and summary["zero_time_seconds"] is not None)
        supported = bool(not beyond or zero_extended)
        output.append({"time_seconds": float(time), "survival": value if supported else np.nan,
                       "at_risk": risk, "supported": supported, "zero_extended": zero_extended})
    return pd.DataFrame(output)


def restricted_mean(km: Mapping[str, Any], restriction_seconds: float) -> float | None:
    """Exact step-curve integral, or None for an unsupported nonzero tail."""
    tau = _positive(restriction_seconds, "restriction_seconds")
    summary, curve = km["summary"], km["curve"]
    if not summary["analysis_sample_count"]:
        return None
    tolerance = 1e-12 * max(1.0, tau, summary["support_end_seconds"])
    if tau > summary["support_end_seconds"] + tolerance and summary["zero_time_seconds"] is None:
        return None
    previous, survival, area = 0.0, 1.0, 0.0
    for row in curve.itertuples(index=False):
        if row.time_seconds > tau:
            break
        area += survival * (row.time_seconds - previous)
        previous, survival = float(row.time_seconds), float(row.survival)
    return float(area + survival * (tau - previous))


def _support_limit(km: Mapping[str, Any], minimum_at_risk: int, risk_fraction: float) -> dict[str, Any]:
    summary = km["summary"]
    n = int(summary["analysis_sample_count"])
    required = max(minimum_at_risk, int(np.ceil(risk_fraction * n)))
    if n < required:
        return {"limit_seconds": 0.0, "required_at_risk": required, "reason": "insufficient_initial_risk"}
    # The first interval without the prescribed risk support limits integration.
    # Becoming exactly zero while still supported permits a zero tail thereafter.
    for row in km["curve"].itertuples(index=False):
        if row.survival == 0:
            return {"limit_seconds": None, "required_at_risk": required, "reason": "zero_survival"}
        if row.at_risk_after < required:
            return {"limit_seconds": float(row.time_seconds), "required_at_risk": required,
                    "reason": "positive_survival_risk_limit"}
    raise RuntimeError("nonempty survival curve did not terminate")


def common_analysis_window(
    curves: Mapping[str, Mapping[str, Any]],
    *,
    minimum_at_risk: int = 1,
    minimum_at_risk_fraction: float = 0.0,
    maximum_seconds: float | None = None,
) -> dict[str, Any]:
    """Select one common window from support alone, before comparing errors.

    Default: one observation is sufficient for estimability, with its risk
    count always reported. A prespecified stricter risk threshold can be used
    as a precision sensitivity check. A zero tail is supported without any
    remaining observations if the curve reached zero before losing required
    risk support. A positive tail with no support is never extrapolated.

    With no upper cap, use the shortest finite support limit; if every curve
    reaches zero, use the latest zero time (later integration adds nothing).
    """
    if not curves:
        raise ValueError("at least one curve is required")
    if isinstance(minimum_at_risk, bool) or int(minimum_at_risk) != minimum_at_risk or minimum_at_risk < 1:
        raise ValueError("minimum_at_risk must be a positive integer")
    fraction = float(minimum_at_risk_fraction)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("minimum_at_risk_fraction must lie in [0, 1]")
    cap = None if maximum_seconds is None else _positive(maximum_seconds, "maximum_seconds")
    details = {name: _support_limit(curve, int(minimum_at_risk), fraction)
               for name, curve in curves.items()}
    finite_limits = [row["limit_seconds"] for row in details.values() if row["limit_seconds"] is not None]
    if finite_limits:
        tau = min(finite_limits)
    else:
        tau = max(float(km["summary"]["zero_time_seconds"]) for km in curves.values())
    if cap is not None:
        tau = min(tau, cap)
    for name, km in curves.items():
        evaluated = evaluate_survival(km, [tau]).iloc[0]
        details[name].update({
            "analysis_sample_count": int(km["summary"]["analysis_sample_count"]),
            "at_risk_at_restriction": int(evaluated["at_risk"]),
            "survival_at_restriction": None if pd.isna(evaluated["survival"]) else float(evaluated["survival"]),
            "zero_extended": bool(evaluated["zero_extended"]),
        })
    return {
        "restriction_seconds": float(tau) if tau > 0 else None,
        "estimable": bool(tau > 0),
        "minimum_at_risk": int(minimum_at_risk),
        "minimum_at_risk_fraction": fraction,
        "maximum_seconds": cap,
        "rule": "common_support" if minimum_at_risk == 1 and fraction == 0 else "common_risk_screened_support",
        "curves": details,
        "status": "available" if tau > 0 else "no_positive_common_window",
    }


def integrated_absolute_survival_error(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], restriction_seconds: float,
) -> float | None:
    """Integrate |S_candidate - S_reference| exactly on a supported shared window."""
    tau = _positive(restriction_seconds, "restriction_seconds")
    if restricted_mean(reference, tau) is None or restricted_mean(candidate, tau) is None:
        return None
    points = np.unique(np.concatenate((
        [0.0, tau],
        reference["curve"]["time_seconds"].to_numpy(dtype=float),
        candidate["curve"]["time_seconds"].to_numpy(dtype=float),
    )))
    points = points[(points >= 0) & (points <= tau)]
    values = [evaluate_survival(km, points[:-1])["survival"].to_numpy(dtype=float)
              for km in [reference, candidate]]
    return float(np.sum(np.abs(values[1] - values[0]) * np.diff(points)))


def integrated_absolute_difference(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], restriction_seconds: float,
) -> float | None:
    """Alias for the exact supported survival-curve difference integral."""
    return integrated_absolute_survival_error(reference, candidate, restriction_seconds)


def legacy_60_second_summary(events: pd.DataFrame, restriction_seconds: float = 60.0) -> dict[str, Any]:
    """Reproduce the old fixed-follow-up rule, including its zero-tail exclusion.

    The old rule required maximum known-start duration >= restriction even
    if every event had already completed. Its completed_count excludes
    unknown-start completed reversals and must not be used as a rate numerator.
    """
    tau = _positive(restriction_seconds, "restriction_seconds")
    km = km_from_events(events)
    summary = km["summary"]
    n = summary["analysis_sample_count"]
    support = summary["support_end_seconds"]
    tolerance = 1e-12 * max(1.0, tau, support or 0.0)
    estimable = bool(n and support >= tau - tolerance)
    durations, _, known = _event_arrays(events)
    at_risk = int(np.sum(durations[known] >= tau - tolerance))
    median = summary["median_seconds"]
    return {
        "sample_count": summary["sample_count"],
        "analysis_sample_count": n,
        "completed_count": summary["completed_count"],
        "censored_count": summary["censored_count"],
        "left_truncated_count": summary["unknown_start_count"],
        "restriction_seconds": tau,
        "restricted_mean_seconds": restricted_mean(km, tau) if estimable else None,
        "median_seconds": median if median is not None and median <= tau + tolerance else None,
        "at_risk_at_restriction": at_risk,
        "support_end_seconds": support,
        "estimable": estimable,
        "retained_by_old_rule": estimable,
        "status": "estimable" if estimable else ("insufficient_follow_up" if n else "no_known_start"),
    }
