"""Reconstruct beetle analysis inputs with saved models and fixed scientific rules.

No model fitting or parameter selection occurs here. The supplied original
rotation observable, contiguous supports, actual frame rates, and simulation
seeds are retained. Large inputs are created only in the requested data folder.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .beetle import recording, verify_source
from .beetle_windows import estimate_windows, prediction_rows, prepare_recording
from .models.bounded import load_model, simulate
from .models.state_age import AgeModel, load_saved, sampled_age


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(table, root, relative):
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(destination, index=False)
    return len(table)


def _polynomial(rows, columns, parameters, base):
    x = rows[columns].to_numpy(dtype=float)
    x = (x - np.asarray(parameters["state_center"])) / np.asarray(parameters["state_scale"])
    design = np.column_stack([np.prod(x ** np.asarray(e), axis=1) for e in parameters["exponents"]])
    return rows[base].to_numpy(float) + (design / np.asarray(parameters["design_scale"])) @ np.asarray(parameters["coefficients"])


def _forecast_table(windows, features, reference):
    outputs = []
    coordinates = features[["trial_id", "frame", "direction_alignment", "rms_radius_pixels", "shape_elongation"]]
    coordinates = coordinates.rename(columns={"direction_alignment": "P", "rms_radius_pixels": "R", "shape_elongation": "A"})
    models = ("original_current", "original_history", "flexible_current", "multivariable_current", "multivariable_history")
    for trial_id, window in windows.loc[windows.role.eq("evaluation")].groupby("trial_id", sort=True):
        size = int(window.group_size.iloc[0])
        for horizon in (3, 10, 30):
            rows = prediction_rows(window.reset_index(drop=True), horizon)
            rows = rows.merge(coordinates, on=["trial_id", "frame"], validate="one_to_one")
            rows["M_delay5"] = rows.delayed_M
            if not np.isfinite(rows[["M", "M_delay5", "P", "R", "A"]].to_numpy()).all():
                raise ValueError("Nonfinite simultaneous features on common prediction support")
            base = rows[["group_size", "trial", "trial_id", "frame", "target_frame", "horizon_frames", "horizon_seconds", "M"]].copy()
            base["target"] = rows.target_M.to_numpy()
            outputs.append(base.assign(model="persistence", prediction=rows.M.to_numpy()))
            for name in models:
                folder = reference / f"models/present_state/N{size}_h{horizon:02d}_{name}"
                spec = _json(folder / "model.json")
                with np.load(folder / "parameters.npz", allow_pickle=False) as parameters:
                    prediction = _polynomial(rows, spec["specification"]["columns"], parameters, "M")
                outputs.append(base.assign(model=name, prediction=prediction))
            spec = _json(reference / f"models/trend/N{size}_h{horizon:03d}_filtered_trend6.json")
            outputs.append(base.assign(model="filtered_trend6", prediction=_polynomial(rows, spec["inputs"], spec, "smoothed_current")))
    return pd.concat(outputs, ignore_index=True)


def _state_age_table(scalar, reference):
    documents = _json(reference / "models/state_age/fitted_models.json")
    models = {key: load_saved(value) for key, value in documents.items()}
    outputs = []
    valid = scalar.loc[scalar.role.eq("evaluation") & scalar.order_sequence_valid]
    for (size, trial, run_id), run in valid.groupby(["group_size", "trial", "order_run_id"], sort=True):
        frames, values = run.frame.to_numpy(int), run.circular_polarization.to_numpy(float)
        if not np.all(np.diff(frames) == 1):
            raise ValueError("Category input crosses a frame gap")
        fps = float(run.frame_rate_hz.iloc[0])
        states = np.where(values <= -.3, 0, np.where(values >= .3, 2, 1))
        age = sampled_age(states, np.zeros(len(states), dtype=int))
        idx = np.arange(25, len(frames)-5, dtype=int)
        if not len(idx):
            continue
        ctx = np.column_stack([states[idx-i*5] for i in range(4)])
        base = pd.DataFrame(dict(group_size=size, trial=trial, trial_id=f"N{size}{trial}",
            frame=frames[idx], target_frame=frames[idx+5], horizon_frames=5,
            time_seconds=(frames[idx]-1)/fps, frame_rate_hz=fps, run_id=run_id,
            block_id=0, phase=frames[idx]%5, target_state=states[idx+5],
            current_value=values[idx], target_value=values[idx+5], age_steps=age[idx],
            age_known=age[idx]>=0, age_seconds=np.where(age[idx]>=0, age[idx]*5/fps, np.nan)))
        for j in range(4):
            base[f"state_lag{j*5}"] = ctx[:, j]
        for name in ("current", "history4", "current_age"):
            model = models[f"N{size}_{name}"]
            p = model.predict_proba(ctx, age[idx]) if isinstance(model, AgeModel) else model.predict_proba(ctx)
            out = base.assign(model=name)
            for j in range(3):
                out[f"probability_state{j}"] = p[:, j]
            out["target"] = states[idx+5]
            out["prediction"] = p.argmax(axis=1)
            out["target_probability"] = p[np.arange(len(idx)), states[idx+5]]
            outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def _displacement_table(features, suffix, reference):
    outputs = []
    columns = {"current_cubic": ["M"], "history5_cubic": ["M", "delayed_M"],
               "filtered_trend6": ["smoothed_current", "slope_per_second"]}
    for trial_id, source in features.loc[features.role.eq("evaluation")].groupby("trial_id", sort=True):
        windows = prepare_recording(source, suffix)
        size = int(source.group_size.iloc[0])
        for horizon in (3, 10, 30):
            rows = prediction_rows(windows, horizon)
            base = rows[["threshold", "group_size", "trial", "trial_id", "frame", "target_frame", "horizon_frames", "M"]].copy()
            base["target"] = rows.target_M.to_numpy()
            outputs.append(base.assign(model="persistence", prediction=rows.M.to_numpy()))
            for name, inputs in columns.items():
                spec = _json(reference / f"sensitivity/displacement/models/{suffix}_N{size}_h{horizon:03d}_{name}.json")
                outputs.append(base.assign(model=name, prediction=_polynomial(rows, inputs, spec, spec["target_offset"])))
    return pd.concat(outputs, ignore_index=True)


def _switching(scalar, reference, data):
    valid = scalar.loc[scalar.role.eq("evaluation") & scalar.order_sequence_valid]
    empirical = valid[["group_size", "trial", "trial_id", "order_run_id", "frame_rate_hz", "frame", "circular_polarization"]]
    empirical = empirical.rename(columns={"order_run_id": "segment_id", "circular_polarization": "rotation"})
    _save(empirical, data, "switching/empirical_valid_trajectories.parquet")
    expected = _json(reference / "core/initial_histories.json")
    histories = {}
    for trial_id, group in valid.groupby("trial_id", sort=True):
        run = next(iter(group.groupby("order_run_id", sort=True)))[1].iloc[:6]
        values = run.circular_polarization.to_numpy(float)
        if not np.array_equal(values, expected[trial_id]["values"]) or not np.array_equal(run.frame, expected[trial_id]["frames"]):
            raise ValueError(f"Simulation initialization differs for {trial_id}")
        histories[trial_id] = values
    manifest = pd.read_csv(reference / "switching/trajectory_manifest.csv")
    models, arrays = {}, {}
    for row in manifest.itertuples(index=False):
        mode = row.model.removeprefix("bounded_")
        key = (row.group_size, mode)
        if key not in models:
            models[key] = load_model(reference / f"core/bounded/N{row.group_size}/bounded_transition_{mode}_h001.json")
        arrays[row.trajectory_id] = simulate(models[key], histories[row.trial_id], mode,
                                            int(row.base_seed), int(row.full_frames))
        if len(arrays) % 64 == 0:
            print(f"Beetles: generated {len(arrays)}/{len(manifest)} saved-seed trajectories.", flush=True)
    np.savez_compressed(data / "switching/full_trajectories.npz", **arrays)
    return {"generated_trajectories": len(arrays), "frames_per_trajectory": 4500,
            "empirical_frames": len(empirical), "empirical_recordings": empirical.trial_id.nunique()}


def prepare_core(beetle_source: Path, reference_dir: Path, data_dir: Path) -> dict:
    """Generate all large core inputs from the verified published recordings.

    ``reference_dir`` contains the distributed small models, result tables and
    scientific configuration. The caller installs those files into ``data_dir``
    before calling this function. Source files are opened read-only.
    """
    start = time.monotonic()
    reference_dir, data_dir = Path(reference_dir), Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    directory, source_document = verify_source(Path(beetle_source), reference_dir)
    print("Beetles: all 30 published source checksums verified.", flush=True)
    scalars, feature_tables, windows, source_info = [], [], [], []
    for size in (50, 100, 200):
        for trial in "ABCDE":
            scalar, feature, info = recording(directory, size, trial)
            scalars.append(scalar); feature_tables.append(feature); source_info.append(info)
            window = scalar[["trial_id", "group_size", "trial", "frame", "circular_polarization", "order_run_id", "order_sequence_valid", "role"]]
            window = window.rename(columns={"circular_polarization": "M", "order_run_id": "run_id", "order_sequence_valid": "scalar_valid"})
            windows.append(estimate_windows(window, info["frame_rate_hz"]))
            print(f"Beetles: reconstructed {info['trial_id']}, {info['frames']} original frames.", flush=True)
    scalar, feature, window = (pd.concat(tables, ignore_index=True) for tables in (scalars, feature_tables, windows))
    _save(scalar, data_dir, "data/derived/scalar_frames.parquet")
    _save(feature, data_dir, "data/derived/group_features_all.parquet")
    _save(window, data_dir, "data/derived/window_estimates.parquet")
    forecasts = _forecast_table(window, feature, reference_dir)
    forecast_rows = _save(forecasts, data_dir, "predictions/continuous.parquet")
    support = forecasts.loc[forecasts.model.eq("persistence")].groupby("horizon_frames").size().to_dict()
    del forecasts
    state_rows = _save(_state_age_table(scalar, reference_dir), data_dir, "predictions/state_age.parquet")
    print(f"Beetles: reconstructed {forecast_rows:,} continuous and {state_rows:,} category predictions.", flush=True)
    sensitivity_rows = {}
    for suffix in ("gt_0p5px", "gt_1p0px", "gt_2p0px"):
        sensitivity_rows[suffix] = _save(_displacement_table(feature, suffix, reference_dir), data_dir,
                                          f"sensitivity/displacement/predictions_{suffix}.parquet")
    result = dict(source_doi=source_document["dataset_doi"], verified_source_files=len(source_document["files"]),
        source_bytes=sum(item["bytes"] for item in source_document["files"]), recordings=source_info,
        scalar_frames=len(scalar), retained_scalar_frames=int(scalar.order_sequence_valid.sum()),
        continuous_predictions=forecast_rows, prediction_origins_by_horizon=support,
        category_predictions=state_rows, displacement_predictions=sensitivity_rows,
        simulation=_switching(scalar, reference_dir, data_dir), model_fitting=False,
        parameter_selection=False, elapsed_seconds=time.monotonic()-start)
    (data_dir / "core_preparation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
