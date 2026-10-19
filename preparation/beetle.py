"""Read the published beetle files after verifying their recorded checksums."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .beetle_features import FrameQualityThresholds, build_frame_masks, features


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source: Path, reference: Path) -> tuple[Path, dict]:
    document = json.loads((reference / "core/beetle_sources.json").read_text(encoding="utf-8"))
    directory = source / "data" if (source / "data/df_50A_omega").is_file() else source
    for item in document["files"]:
        path = directory / item["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing published beetle source: {path}")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Published source checksum differs: {path.name}")
    # The original public files use pickle. Only the exact checksum-verified
    # published files are deserialized; arbitrary pickle inputs are rejected.
    return directory, document


def recording(directory: Path, size: int, trial: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tracks = pd.read_pickle(directory / f"df_{size}{trial}_omega").copy()
    tracks["frame"] = np.rint(tracks.frame).astype(np.int32)
    tracks["ind"] = np.rint(tracks.ind).astype(np.int32)
    tracks["group_size"] = np.int16(size)
    tracks["trial"] = trial
    values = np.asarray(pd.read_pickle(directory / f"cirOP{size}{trial}"), dtype=np.float64)
    frames = np.sort(tracks.frame.unique())
    if values.ndim != 1 or len(frames) != len(values):
        raise ValueError("Published order parameter and original frame count differ")
    dx = (tracks.x - tracks.x_prev).to_numpy(float)
    vx = tracks.vx.to_numpy(float)
    good = (tracks.frame.to_numpy() > 1) & np.isfinite(dx) & (dx != 0) & np.isfinite(vx)
    inferred = vx[good] / dx[good]
    fps = float(np.median(inferred))
    if fps <= 0 or not np.isfinite(fps) or np.max(abs(inferred - fps)) > 1e-10:
        raise ValueError("Source velocities do not imply one constant frame rate")
    finite_position = np.isfinite(tracks[["x", "y"]]).all(axis=1)
    finite_velocity = np.isfinite(tracks[["vx", "vy"]]).all(axis=1)
    unchanged = (np.isfinite(tracks[["x_prev", "y_prev"]]).all(axis=1)
                 & (tracks.x_prev == tracks.x) & (tracks.y_prev == tracks.y))
    nonzero = finite_velocity & ((tracks.vx != 0) | (tracks.vy != 0))
    quality = tracks.assign(finite_position=finite_position, finite_velocity=finite_velocity,
                            unchanged=unchanged, nonzero=nonzero).groupby("frame", sort=True).agg(
        detection_count=("ind", "size"), valid_position_count=("finite_position", "sum"),
        valid_velocity_count=("finite_velocity", "sum"),
        unchanged_from_previous_count=("unchanged", "sum"), nonzero_velocity_count=("nonzero", "sum")
    ).reset_index()
    quality["velocity_measurement_valid"] = ~((quality.valid_velocity_count > 0)
        & (quality.nonzero_velocity_count == 0)
        & (quality.unchanged_from_previous_count == quality.detection_count))
    quality["circular_polarization"] = values
    scalar, runs = build_frame_masks(quality, group_size=size, thresholds=FrameQualityThresholds())
    scalar.insert(0, "trial", trial)
    scalar.insert(0, "group_size", size)
    scalar["trial_id"] = f"N{size}{trial}"
    scalar["frame_rate_hz"] = fps
    scalar["time_seconds"] = (scalar.frame - 1) / fps
    scalar["role"] = "development" if trial == "D" else "evaluation"
    simultaneous, info = features(tracks, fps)
    simultaneous = scalar.merge(simultaneous, on="frame", validate="one_to_one")
    info.update(trial_id=f"N{size}{trial}", frame_rate_hz=fps, frames=len(scalar),
                retained_frames=int(scalar.order_sequence_valid.sum()),
                retained_runs=sum(bool(x["accepted"]) for x in runs["order"]))
    return scalar, simultaneous, info
