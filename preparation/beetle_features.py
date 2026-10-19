"""Reconstruct beetle frame validity and simultaneous group observables."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class FrameQualityThresholds:
    """Thresholds fixed before fitting any dynamical model."""

    minimum_detection_fraction: float = 0.5
    maximum_detection_fraction: float = 2.0
    minimum_velocity_fraction: float = 0.75
    minimum_sequence_frames: int = 30

    def __post_init__(self) -> None:
        if not 0 < self.minimum_detection_fraction < self.maximum_detection_fraction:
            raise ValueError("detection fractions must satisfy 0 < minimum < maximum")
        if not 0 <= self.minimum_velocity_fraction <= 1:
            raise ValueError("minimum_velocity_fraction must lie in [0, 1]")
        if self.minimum_sequence_frames < 1:
            raise ValueError("minimum_sequence_frames must be positive")


def label_contiguous_runs(
    frames: np.ndarray,
    valid: np.ndarray,
    *,
    minimum_length: int = 1,
) -> tuple[np.ndarray, list[dict[str, int | bool]]]:
    """Label consecutive valid frames and describe every run.

    Invalid frames receive ``-1``. Valid runs are split both when a row is
    invalid and when frame numbers are not consecutive.
    """

    frame_values = np.asarray(frames, dtype=np.int64)
    valid_values = np.asarray(valid, dtype=bool)
    if frame_values.ndim != 1 or valid_values.ndim != 1:
        raise ValueError("frames and valid must be one-dimensional")
    if len(frame_values) != len(valid_values):
        raise ValueError("frames and valid must have the same length")
    if minimum_length < 1:
        raise ValueError("minimum_length must be positive")
    if len(frame_values) > 1 and np.any(np.diff(frame_values) <= 0):
        raise ValueError("frames must be strictly increasing")

    labels = np.full(len(frame_values), -1, dtype=np.int64)
    selected_rows = np.flatnonzero(valid_values)
    if not len(selected_rows):
        return labels, []

    discontinuity = (
        (np.diff(selected_rows) != 1)
        | (np.diff(frame_values[selected_rows]) != 1)
    )
    boundaries = np.flatnonzero(discontinuity) + 1
    runs = np.split(selected_rows, boundaries)
    records: list[dict[str, int | bool]] = []
    for run_id, rows in enumerate(runs):
        labels[rows] = run_id
        length = len(rows)
        records.append(
            {
                "run_id": run_id,
                "start_frame": int(frame_values[rows[0]]),
                "end_frame": int(frame_values[rows[-1]]),
                "frame_count": length,
                "accepted": length >= minimum_length,
            }
        )
    return labels, records


def build_frame_masks(
    frame_quality: pd.DataFrame,
    *,
    group_size: int,
    thresholds: FrameQualityThresholds,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, int | bool]]]]:
    """Build outcome-independent masks for order-parameter and field analyses."""

    required = {
        "frame",
        "detection_count",
        "valid_position_count",
        "valid_velocity_count",
        "velocity_measurement_valid",
        "circular_polarization",
    }
    missing = required.difference(frame_quality.columns)
    if missing:
        raise ValueError(f"missing frame-quality columns: {sorted(missing)}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    result = frame_quality.sort_values("frame").reset_index(drop=True).copy()
    if result["frame"].duplicated().any():
        raise ValueError("frame-quality table contains duplicate frame numbers")

    count_fraction = result["detection_count"].to_numpy(dtype=float) / group_size
    detection_valid = (
        np.isfinite(count_fraction)
        & (count_fraction >= thresholds.minimum_detection_fraction)
        & (count_fraction <= thresholds.maximum_detection_fraction)
    )
    position_fraction = (
        result["valid_position_count"].to_numpy(dtype=float)
        / result["detection_count"].to_numpy(dtype=float)
    )
    velocity_fraction = (
        result["valid_velocity_count"].to_numpy(dtype=float)
        / result["detection_count"].to_numpy(dtype=float)
    )
    finite_order = np.isfinite(result["circular_polarization"].to_numpy(dtype=float))
    velocity_measurement_valid = result["velocity_measurement_valid"].to_numpy(dtype=bool)

    result["detection_fraction_of_nominal"] = count_fraction
    result["position_valid_fraction"] = position_fraction
    result["velocity_valid_fraction"] = velocity_fraction
    result["detection_count_valid"] = detection_valid
    result["finite_order_parameter"] = finite_order
    result["density_frame_valid"] = detection_valid & (position_fraction == 1.0)
    result["order_frame_valid"] = result["density_frame_valid"].to_numpy() & finite_order
    result["field_frame_valid"] = (
        result["density_frame_valid"].to_numpy()
        & (velocity_fraction >= thresholds.minimum_velocity_fraction)
        & velocity_measurement_valid
    )

    frames = result["frame"].to_numpy(dtype=np.int64)
    run_records: dict[str, list[dict[str, int | bool]]] = {}
    for purpose in ("density", "order", "field"):
        valid = result[f"{purpose}_frame_valid"].to_numpy(dtype=bool)
        labels, records = label_contiguous_runs(
            frames,
            valid,
            minimum_length=thresholds.minimum_sequence_frames,
        )
        accepted_ids = {int(item["run_id"]) for item in records if item["accepted"]}
        result[f"{purpose}_run_id"] = labels
        result[f"{purpose}_sequence_valid"] = np.fromiter(
            (int(label) in accepted_ids for label in labels),
            dtype=bool,
            count=len(labels),
        )
        run_records[purpose] = records

    return result, run_records


def features(t, fps):
    t=t.sort_values('frame',kind='stable').reset_index(drop=True)
    frames, inv=np.unique(t.frame.to_numpy(int),return_inverse=True)
    k=len(frames)
    def total(v): return np.bincount(inv,weights=np.asarray(v,float),minlength=k)
    def average(v, good):
        count=total(good)
        return np.divide(total(np.where(good,v,0)),count,out=np.full(k,np.nan),where=count>0)
    x=t.x.to_numpy(float); y=t.y.to_numpy(float)
    vx=t.vx.to_numpy(float); vy=t.vy.to_numpy(float)
    pgood=np.isfinite(x)&np.isfinite(y)
    np_=total(pgood); count=total(np.ones(len(t)))
    cx=average(x,pgood); cy=average(y,pgood)
    rx=x-cx[inv]; ry=y-cy[inv]
    xx=average(rx*rx,pgood); yy=average(ry*ry,pgood); xy=average(rx*ry,pgood)
    radius=np.sqrt(xx+yy)
    elong=np.divide(np.sqrt((xx-yy)**2+4*xy*xy),xx+yy,out=np.full(k,np.nan),where=(xx+yy)>0)
    finite_motion=pgood&np.isfinite(vx)&np.isfinite(vy)&(t.frame.to_numpy()>1)
    displacement=np.hypot(vx,vy)/fps
    dgood=finite_motion&(displacement>0)
    ux=np.divide(vx,np.hypot(vx,vy),out=np.zeros(len(t)),where=dgood)
    uy=np.divide(vy,np.hypot(vx,vy),out=np.zeros(len(t)),where=dgood)
    direction_count=total(dgood)
    alignment=np.hypot(average(ux,dgood),average(uy,dgood))
    # Physical y-up sign; all finite positions define the centroid.
    rr=np.hypot(rx,ry)
    rotate_good=dgood&(rr>0)
    contribution=np.divide(ry*vx-rx*vy,rr*np.hypot(vx,vy),out=np.full(len(t),np.nan),where=rotate_good)
    rotation=average(contribution,rotate_good)
    omega=t.omega.to_numpy(float)
    supplied_omega_mean=-average(omega,np.isfinite(omega))
    supplied_omega_direction_mean=-average(omega,rotate_good&np.isfinite(omega))
    refgood=pgood&np.isfinite(vx)&np.isfinite(vy)
    rcx=average(x,refgood); rcy=average(y,refgood)
    rrx=x-rcx[inv]; rry=y-rcy[inv]; rnorm=np.hypot(rrx,rry)
    rgood=refgood&dgood&(rnorm>0)
    refcon=np.divide(rry*vx-rrx*vy,rnorm*np.hypot(vx,vy),out=np.full(len(t),np.nan),where=rgood)
    reference_rotation=average(refcon,rgood)
    dx=x-t.x_prev.to_numpy(float); dy=y-t.y_prev.to_numpy(float)
    linked=finite_motion
    motion_error=np.hypot(vx/fps-dx,vy/fps-dy)
    f=pd.DataFrame(dict(frame=frames,detection_count_recomputed=count.astype(int),valid_position_count_new=np_.astype(int),
        valid_motion_count=total(finite_motion).astype(int),valid_direction_count=direction_count.astype(int),
        valid_direction_fraction=direction_count/count,valid_direction_fraction_of_nominal=direction_count/int(t.group_size.iloc[0]),
        valid_position_fraction_new=np_/count,zero_displacement_count=total(finite_motion&(displacement==0)).astype(int),
        nonfinite_motion_count=total(pgood&~(np.isfinite(vx)&np.isfinite(vy))).astype(int),
        mean_displacement_pixels=average(displacement,finite_motion),mean_nonzero_displacement_pixels=average(displacement,dgood),
        direction_alignment=alignment,rms_radius_pixels=radius,shape_elongation=elong,
        centroid_x_image=cx,centroid_y_image=cy,rotation_recomputed_all_position_centroid=rotation,
        negative_mean_source_omega=supplied_omega_mean,rotation_support_count=total(rotate_good).astype(int),
        negative_mean_source_omega_valid_direction=supplied_omega_direction_mean,
        rotation_reference_function=reference_rotation,
        zero_radial_count=total(pgood&(rr==0)).astype(int),
        displacement_coordinate_error_mean=average(motion_error,linked)))
    for threshold in (0.5,1.,2.):
        good=finite_motion&(displacement>threshold)
        suffix=str(threshold).replace('.','p')
        f[f'direction_count_gt_{suffix}px']=total(good).astype(int)
        f[f'direction_alignment_gt_{suffix}px']=np.hypot(average(ux,good),average(uy,good))
        f[f'rotation_gt_{suffix}px']=average(contribution,good&(rr>0))
    f['position_valid']=(np_>=2)&np.isfinite(radius)&(radius>0)&np.isfinite(elong)
    f['direction_valid']=(direction_count>=2)&np.isfinite(alignment)
    f['features_valid']=f.position_valid&f.direction_valid
    f['missing_reason']=np.select([~f.position_valid,~f.direction_valid],['insufficient_or_degenerate_positions','fewer_than_two_nonzero_directions'],default='')
    info=dict(detections=len(t),finite_motion_count=int(linked.sum()),
        maximum_displacement_coordinate_error=float(np.max(motion_error[linked])) if linked.any() else None,
        nonzero_displacement_min=float(np.min(displacement[dgood])) if dgood.any() else None,
        detected_positions_are_integer=bool(np.all(x[pgood]==np.rint(x[pgood])) and np.all(y[pgood]==np.rint(y[pgood]))))
    return f,info
