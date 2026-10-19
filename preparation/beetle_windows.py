"""Causal six-observation windows and contiguous prediction support."""
from __future__ import annotations
import numpy as np
import pandas as pd
WINDOW=6

def estimate_windows(table: pd.DataFrame, fps: float) -> pd.DataFrame:
    """One input recording; preserve all input frames including invalid windows.

    Required columns: frame, M, run_id, scalar_valid. Original frame numbers
    must be unique and increasing. A changed run ID is always a boundary.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    out = table.copy().reset_index(drop=True)
    frames = out.frame.to_numpy(dtype=np.int64)
    if len(frames) == 0 or np.any(np.diff(frames) <= 0):
        raise ValueError("frames must be nonempty, unique and increasing")
    if not np.array_equal(frames, out.frame.to_numpy()):
        raise ValueError("original frame numbers must be integers")
    values = out.M.to_numpy(dtype=float)
    runs = out.run_id.to_numpy()
    valid = out.scalar_valid.fillna(False).to_numpy(dtype=bool)
    valid &= np.isfinite(values) & pd.notna(runs)
    # Invalid frames are singleton pieces. The next valid frame starts anew.
    connected = (np.diff(frames) == 1) & (runs[1:] == runs[:-1]) & valid[1:] & valid[:-1]
    pieces = np.r_[0, np.cumsum(~connected)]
    starts = np.maximum.accumulate(np.where(np.r_[True, ~connected], np.arange(len(out)), 0))
    available = np.arange(len(out)) - starts + 1
    good = valid & (available >= WINDOW)
    a = np.full(len(out), np.nan)
    b = np.full(len(out), np.nan)
    relative_seconds = np.arange(-5, 1, dtype=float) / fps
    centered = relative_seconds - relative_seconds.mean()
    slope_weights = centered / (centered @ centered)
    endpoint_weights = np.ones(WINDOW) / WINDOW - relative_seconds.mean() * slope_weights
    if len(out) >= WINDOW:
        windows = np.lib.stride_tricks.sliding_window_view(values, WINDOW)
        ix = np.flatnonzero(good)
        a[ix] = windows[ix - WINDOW + 1] @ endpoint_weights
        b[ix] = windows[ix - WINDOW + 1] @ slope_weights
    out["piece_id"] = pieces
    out["contiguous_observation_count"] = np.where(valid, available, 0)
    out["fps"] = fps
    out["time_seconds"] = (frames - 1) / fps
    out["window_start_frame"] = np.where(good, frames - WINDOW + 1, -1)
    out["window_end_frame"] = frames
    out["window_duration_seconds"] = (WINDOW - 1) / fps
    out["window_observations"] = np.where(good, WINDOW, 0)
    out["window_valid"] = good
    out["window_invalid_reason"] = np.where(~valid, "invalid_scalar", np.where(good, "", "fewer_than_six_contiguous_observations"))
    out["smoothed_current"] = a
    out["slope_per_second"] = b
    return out


def prediction_rows(windows: pd.DataFrame, horizon: int, *, history: int = 30,
                    boundaries: pd.DataFrame | None = None, purge: int = 30) -> pd.DataFrame:
    """Use complete continuous input-to-target intervals, with optional D guards."""
    if horizon not in (3, 10, 30) or history < 5 or purge < 0:
        raise ValueError("unexpected horizon/history/purge")
    n = len(windows)
    i = np.arange(n)
    j = np.minimum(i + horizon, n - 1)
    frames = windows.frame.to_numpy(dtype=np.int64)
    pieces = windows.piece_id.to_numpy()
    ok = windows.window_valid.to_numpy(dtype=bool, copy=True)
    ok &= windows.contiguous_observation_count.to_numpy() > history
    ok &= i + horizon < n
    ok &= pieces == pieces[j]
    ok &= frames[j] == frames + horizon
    block = np.zeros(n, dtype=np.int16)
    if boundaries is not None:
        if "trial" in windows and set(windows.trial.astype(str)) != {"D"}:
            raise ValueError("development boundaries may only be used with D")
        contained = np.zeros(n, dtype=bool)
        for row in boundaries.itertuples():
            mask = (frames - history >= int(row.start_frame) + purge) & (frames + horizon <= int(row.end_frame) - purge)
            if np.any(contained & mask):
                raise ValueError("overlapping development blocks")
            contained |= mask
            block[mask] = int(row.block_id)
        ok &= contained
    selected = np.flatnonzero(ok)
    out = windows.iloc[selected].copy().reset_index(drop=True)
    out["block_id"] = block[selected]
    out["horizon_frames"] = horizon
    out["horizon_seconds"] = horizon / out.fps
    out["target_frame"] = frames[selected + horizon]
    out["target_M"] = windows.M.to_numpy()[selected + horizon]
    out["delayed_M"] = windows.M.to_numpy()[selected - 5]
    out["target_change_from_smoothed"] = out.target_M - out.smoothed_current
    return out


def prepare_recording(source, suffix):
    if source.trial_id.nunique() != 1 or source.frame.duplicated().any():
        raise ValueError("one recording with unique original frame numbers required")
    s = source.sort_values("frame").reset_index(drop=True)
    v = s[f"rotation_{suffix}"].to_numpy(float)
    count = s[f"direction_count_{suffix}"].to_numpy(int)
    original = s.order_sequence_valid.to_numpy(bool)
    finite = np.isfinite(v)
    bounds = np.abs(v) <= 1
    valid = original & (count >= 2) & finite & bounds
    f = s.frame.to_numpy(int)
    rid = s.order_run_id.to_numpy()
    link = (np.diff(f) == 1) & (rid[1:] == rid[:-1]) & valid[:-1] & valid[1:]
    piece = np.r_[0, np.cumsum(~link)]
    lengths = np.bincount(piece)
    short = valid & (lengths[piece] < 30)
    valid &= ~short
    out = s[["trial_id", "group_size", "trial", "frame", "role"]].copy()
    out["M"] = v
    out["source_M"] = s.circular_polarization
    out["source_run_id"] = rid
    out["run_id"] = np.where(valid, piece, -1)
    out["scalar_valid"] = valid
    out["direction_count"] = count
    out["original_scalar_valid"] = original
    out["variant_invalid_reason"] = np.select([~original, count < 2, ~finite, ~bounds, short],
        ["original_scalar_invalid", "fewer_than_two_directions", "nonfinite_rotation", "rotation_outside_bounds", "segment_shorter_than_30"], default="")
    fps = float(s.frame_rate_hz.iloc[0])
    if s.frame_rate_hz.nunique() != 1:
        raise ValueError("multiple rates within recording")
    return estimate_windows(out, fps).assign(threshold=suffix)
