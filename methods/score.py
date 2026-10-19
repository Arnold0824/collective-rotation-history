"""Recompute prediction scores and switching statistics from saved scientific data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .reversal_events import (
    extract_reversal_tables, counts_and_rate, km_from_events,
    common_analysis_window, integrated_absolute_survival_error,
)

ROOT: Path
OUT: Path
checks = []


def check(name, actual, expected, tolerance=2e-10):
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != e.shape:
        raise AssertionError((name, a.shape, e.shape))
    error = float(np.max(np.abs(a - e))) if a.size else 0.0
    if not np.isfinite(a).all() or error > tolerance:
        raise AssertionError((name, error, tolerance))
    checks.append(dict(name=name, values=int(a.size), maximum_absolute_error=error,
                       tolerance=tolerance))


def balanced(values, sizes, log=False):
    d = pd.DataFrame(dict(value=np.asarray(values), size=np.asarray(sizes)))
    if log:
        d["value"] = np.log(d.value)
    answer = float(d.groupby("size").value.median().mean())
    return float(np.exp(answer)) if log else answer


def forecasts():
    data = pd.read_parquet(ROOT / "predictions/continuous.parquet")
    keys = ["trial_id", "frame", "horizon_frames"]
    assert not data.duplicated(keys + ["model"]).any()
    assert data.groupby(keys).model.nunique().eq(7).all()
    assert len(data) == 357452 * 7
    rows = []
    for (size, trial, horizon, model), g in data.groupby(
        ["group_size", "trial_id", "horizon_frames", "model"], sort=True
    ):
        y, yhat, m = (g[c].to_numpy(float) for c in ("target", "prediction", "M"))
        error = y - yhat
        mse = float(np.mean(error**2))
        persistence = float(np.mean((y-m)**2))
        rows.append(dict(group_size=size, trial_id=trial, horizon_frames=horizon, model=model,
                         n=len(g), mse=mse, rmse=float(np.sqrt(mse)), mae=float(np.mean(abs(error))),
                         r2_future=float(1-np.sum(error**2)/np.sum((y-y.mean())**2)),
                         persistence_mse=persistence, gain_vs_persistence=1-mse/persistence))
    metrics = pd.DataFrame(rows)
    join = ["group_size", "trial_id", "horizon_frames", "model"]
    expected = pd.read_csv(ROOT / "results/forecast/recording_metrics.csv")
    merged = metrics.merge(expected, on=join, suffixes=("", "_expected"), validate="one_to_one")
    assert len(merged) == 252
    for col in ("n", "mse", "rmse", "mae", "r2_future", "persistence_mse", "gain_vs_persistence"):
        check("forecast_"+col, merged[col], merged[col+"_expected"])
    metrics.to_csv(OUT / "forecast_recording_metrics.csv", index=False)
    comparisons = pd.read_csv(ROOT / "results/forecast/comparison_summary.csv")
    result = []
    for row in comparisons.itertuples(index=False):
        part = metrics.loc[metrics.horizon_frames.eq(row.horizon_frames)]
        wide = part.pivot(index=["group_size", "trial_id"], columns="model", values="mse")
        ratios = wide[row.numerator] / wide[row.denominator]
        value = balanced(ratios, wide.index.get_level_values("group_size"), log=True)
        check(f"forecast_ratio_{row.horizon_frames}_{row.numerator}_{row.denominator}", value,
              row.log_balanced_ratio)
        result.append(dict(horizon_frames=row.horizon_frames, numerator=row.numerator,
                           denominator=row.denominator, log_balanced_ratio=value,
                           wins=int((ratios < 1).sum())))
    pd.DataFrame(result).to_csv(OUT / "forecast_comparisons.csv", index=False)

    # Direct parameter replay, without the model implementation used for fitting.
    trend = data.loc[data.model.eq("filtered_trend6")].copy()
    windows = pd.read_parquet(ROOT / "data/derived/window_estimates.parquet")
    coordinates = ["trial_id", "frame", "smoothed_current", "slope_per_second"]
    trend = trend.merge(windows[coordinates], on=["trial_id", "frame"], validate="many_to_one")
    for (size, horizon), g in trend.groupby(["group_size", "horizon_frames"]):
        path = ROOT / f"models/trend/N{size}_h{horizon:03d}_filtered_trend6.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        x = g[spec["inputs"]].to_numpy(float)
        x = (x-np.asarray(spec["state_center"]))/np.asarray(spec["state_scale"])
        design = np.column_stack([np.prod(x**np.asarray(e),axis=1) for e in spec["exponents"]])
        predicted = g.smoothed_current.to_numpy() + (design/np.asarray(spec["design_scale"])) @ np.asarray(spec["coefficients"])
        check(f"trend_parameter_replay_N{size}_h{horizon}", predicted, g.prediction)

    feature_columns = ["trial_id", "frame", "direction_alignment", "rms_radius_pixels", "shape_elongation", "circular_polarization"]
    feature = pd.read_parquet(ROOT / "data/derived/group_features_all.parquet", columns=feature_columns)
    feature = feature.rename(columns=dict(direction_alignment="P", rms_radius_pixels="R", shape_elongation="A"))
    delayed = feature[["trial_id", "frame", "circular_polarization"]].rename(columns={"circular_polarization": "M_delay5"})
    delayed["frame"] = delayed.frame + 5
    feature = feature.drop(columns="circular_polarization").merge(delayed, on=["trial_id", "frame"], how="left", validate="one_to_one")
    present = data.loc[data.model.isin(["multivariable_current", "multivariable_history"])].merge(
        feature, on=["trial_id", "frame"], validate="many_to_one")
    for (size, horizon, model), g in present.groupby(["group_size", "horizon_frames", "model"]):
        folder = ROOT / f"models/present_state/N{size}_h{horizon:02d}_{model}"
        spec = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        parameters = np.load(folder / "parameters.npz", allow_pickle=False)
        x = (g[spec["specification"]["columns"]].to_numpy()-parameters["state_center"])/parameters["state_scale"]
        design = np.column_stack([np.prod(x**e, axis=1) for e in parameters["exponents"]])
        predicted = g.M.to_numpy()+(design/parameters["design_scale"]) @ parameters["coefficients"]
        check(f"present_parameter_replay_N{size}_h{horizon}_{model}", predicted, g.prediction)
    print("Forecasts: 2,502,164 predictions; 252 recording metrics; 36 comparisons; 27 model parameter replays.", flush=True)


def state_age():
    table = pd.read_parquet(ROOT / "predictions/state_age.parquet")
    assert len(table) == 358269
    rows = []
    for (size, trial, model), g in table.groupby(["group_size", "trial_id", "model"], sort=True):
        probability = g[["probability_state0", "probability_state1", "probability_state2"]].to_numpy(float)
        target = g.target.to_numpy(int)
        check("state_probability_sum_"+trial+"_"+model, probability.sum(axis=1), np.ones(len(g)))
        selected = probability[np.arange(len(g)), target]
        truth = np.eye(3)[target]
        rows.append(dict(group_size=size, trial_id=trial, model=model, row_count=len(g),
                         nll=float(-np.log(selected).mean()), brier=float(np.mean(np.sum((probability-truth)**2,axis=1))),
                         accuracy=float(np.mean(probability.argmax(axis=1)==target))))
    metrics = pd.DataFrame(rows)
    expected = pd.read_csv(ROOT / "results/state_age/trial_metrics.csv")
    merged = metrics.merge(expected,on=["group_size", "trial_id", "model"],suffixes=("", "_expected"),validate="one_to_one")
    assert len(merged)==36
    for col in ("row_count", "nll", "brier", "accuracy"):
        check("state_age_"+col, merged[col], merged[col+"_expected"])
    metrics.to_csv(OUT / "state_age_metrics.csv",index=False)
    print("State age: all 358,269 probability predictions and 36 recording scores.",flush=True)


def switching():
    path = ROOT / "switching"
    manifest = pd.read_csv(path / "trajectory_manifest.csv")
    arrays = np.load(path / "full_trajectories.npz", allow_pickle=False)
    events, exposures = [], []
    for row in manifest.itertuples(index=False):
        values = arrays[row.trajectory_id][row.burn_in_frames:]
        assert len(values) == row.comparison_frames == 3000
        metadata = dict(group_size=row.group_size, trial_id=row.trial_id, model=row.model)
        e, o = extract_reversal_tables(values, np.arange(len(values)), row.frame_rate_hz,
                                      metadata=metadata)
        # An independent direct count: delete central values, then count sign changes.
        directional = np.sign(values[np.abs(values) >= .3])
        count = int(np.sum(directional[1:] != directional[:-1]))
        check("direct_reversals_"+row.trajectory_id, int(e.event_observed.sum()), count, 0)
        events.append(e); exposures.append(o)
    empirical = pd.read_parquet(path / "empirical_valid_trajectories.parquet")
    for trial, g in empirical.groupby("trial_id", sort=True):
        g = g.sort_values("frame")
        row = g.iloc[0]
        e, o = extract_reversal_tables(g.rotation, g.frame, row.frame_rate_hz,
                    segment_ids=g.segment_id, metadata=dict(group_size=int(row.group_size),
                                                           trial_id=trial, model="empirical"))
        direct = 0
        for _, segment in g.groupby("segment_id"):
            values = segment.rotation.to_numpy()
            signs = np.sign(values[np.abs(values) >= .3])
            direct += int(np.sum(signs[1:] != signs[:-1]))
        check("direct_reversals_"+trial, int(e.event_observed.sum()), direct, 0)
        events.append(e); exposures.append(o)
    events, exposures = pd.concat(events, ignore_index=True), pd.concat(exposures, ignore_index=True)
    rates, curves = [], {}
    for key, e in events.groupby(["group_size", "trial_id", "model"], sort=True):
        size, trial, model = key
        o = exposures.loc[exposures.trial_id.eq(trial) & exposures.model.eq(model)]
        rates.append(dict(group_size=size, trial_id=trial, model=model, **counts_and_rate(e, o)))
        curves[(trial, model)] = km_from_events(e)
    rates = pd.DataFrame(rates)
    ref = pd.read_csv(path / "trial_rates_and_wait_counts.csv")
    merged = rates.merge(ref, on=["group_size", "trial_id", "model"], suffixes=("", "_expected"), validate="one_to_one")
    assert len(merged) == 36 and len(manifest) == 384
    for col in ("observation_seconds", "completed_count", "known_start_completed_count",
                "known_start_right_censored_count", "unknown_start_completed_count", "reversals_per_minute"):
        check("switching_"+col, merged[col], merged[col+"_expected"])
    rates.to_csv(OUT / "switching_rates.csv", index=False)
    windows = pd.read_csv(path / "common_wait_windows.csv")
    errors = pd.read_csv(path / "trial_survival_error_comparisons.csv")
    models = ["empirical", "bounded_current_only", "bounded_current_and_delay"]
    waits = []
    for row in windows.itertuples(index=False):
        kms = {model: curves[(row.trial_id, model)] for model in models}
        rule = dict(minimum_at_risk=1, minimum_at_risk_fraction=0.0) if row.rule == "data_support" else dict(minimum_at_risk=5, minimum_at_risk_fraction=.05)
        window = common_analysis_window(kms, **rule)
        if not row.estimable:
            assert not window["estimable"]
            continue
        check("waiting_window_"+row.trial_id+"_"+row.rule, window["restriction_seconds"], row.restriction_seconds)
        tau = window["restriction_seconds"]
        result = dict(group_size=row.group_size, trial_id=row.trial_id, rule=row.rule,
                      restriction_seconds=tau)
        expected = errors.loc[errors.trial_id.eq(row.trial_id) & errors.rule.eq(row.rule)].iloc[0]
        for model in models[1:]:
            value = integrated_absolute_survival_error(kms["empirical"], kms[model], tau) / tau
            check("waiting_error_"+row.trial_id+"_"+row.rule+"_"+model, value, expected[model])
            result[model] = value
        result["ratio"] = result[models[2]] / result[models[1]]
        waits.append(result)
    waits = pd.DataFrame(waits)
    waits.to_csv(OUT / "waiting_errors.csv", index=False)
    balanced_ref = pd.read_csv(ROOT / "results/switching/comparison_summary.csv")
    for rule in ("data_support", "risk_screened"):
        selected = waits.loc[waits.rule.eq(rule)]
        expected = balanced_ref.loc[balanced_ref.metric.eq("waiting_curve_error") & balanced_ref.rule.eq(rule) & balanced_ref.analysis_set.eq("all_available")].iloc[0]
        check("waiting_ratio_"+rule, balanced(selected.ratio, selected.group_size, log=True), expected.log_balanced_ratio)
    wide = rates.pivot(index=["group_size", "trial_id"], columns="model", values="reversals_per_minute")
    ratio = abs(wide[models[2]]-wide[models[0]]) / abs(wide[models[1]]-wide[models[0]])
    expected = balanced_ref.loc[balanced_ref.metric.eq("conversion_rate_absolute_error") & balanced_ref.analysis_set.eq("all_available")].iloc[0]
    check("switch_rate_error_ratio", balanced(ratio, wide.index.get_level_values("group_size"),log=True), expected.log_balanced_ratio)
    print("Switching: all 384 generated trajectories and 12 recordings; rates and both supported waiting rules.", flush=True)


def displacement_sensitivity():
    directory = ROOT / "sensitivity/displacement"
    reference = pd.read_csv(directory / "recording_metrics.csv")
    rows = []
    total = 0
    for path in sorted(directory.glob("predictions_*.parquet")):
        table = pd.read_parquet(path)
        total += len(table)
        assert not table.duplicated(["threshold", "trial_id", "frame", "horizon_frames", "model"]).any()
        for (threshold, size, trial, horizon, model), g in table.groupby(
            ["threshold", "group_size", "trial_id", "horizon_frames", "model"], sort=True
        ):
            target, prediction = g.target.to_numpy(float), g.prediction.to_numpy(float)
            error = target-prediction
            mse, variance = float(np.mean(error**2)), float(np.var(target))
            rows.append(dict(threshold=threshold, group_size=size, trial_id=trial,
                             horizon_frames=horizon, model=model, sample_count=len(g),
                             mse=mse, rmse=float(np.sqrt(mse)), mae=float(np.mean(abs(error))),
                             target_variance=variance, r2=1-mse/variance))
    assert total == 4289424
    metrics = pd.DataFrame(rows)
    keys = ["threshold", "group_size", "trial_id", "horizon_frames", "model"]
    joined = metrics.merge(reference,on=keys,suffixes=("", "_expected"),validate="one_to_one")
    assert len(joined)==432
    for column in ("sample_count", "mse", "rmse", "mae", "target_variance", "r2"):
        check("displacement_sensitivity_"+column, joined[column], joined[column+"_expected"])
    metrics.to_csv(OUT / "displacement_sensitivity_metrics.csv",index=False)
    comparison = pd.read_csv(directory / "comparison_summary.csv")
    for row in comparison.itertuples(index=False):
        subset = metrics.loc[metrics.threshold.eq(row.threshold)&metrics.horizon_frames.eq(row.horizon_frames)]
        wide = subset.pivot(index=["group_size", "trial_id"],columns="model",values="mse")
        ratio = wide[row.numerator]/wide[row.denominator]
        check("displacement_ratio_"+row.threshold+"_"+str(row.horizon_frames)+"_"+row.numerator+"_"+row.denominator,
              balanced(ratio,wide.index.get_level_values("group_size"),log=True),row.log_balanced_ratio)
        check("displacement_wins_"+row.threshold+"_"+str(row.horizon_frames)+"_"+row.numerator+"_"+row.denominator,
              int((ratio<1).sum()),row.wins,0)
    print("Displacement sensitivity: all 4,289,424 evaluation predictions; 432 metrics and 54 paired summaries.",flush=True)


def run(data_dir: Path, output_dir: Path):
    global ROOT, OUT, checks
    ROOT, OUT = Path(data_dir), Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    forecasts()
    state_age()
    switching()
    displacement_sensitivity()
    summary = dict(passed=True, numerical_checks=len(checks), checks=checks,
                   scope="All main, state-age and displacement-sensitivity evaluation predictions, 27 direct parameter replays, trajectory events, exposure, waiting curves and primary aggregation. No refitting or raw-video extraction.")
    (OUT / "numerical_verification.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(f"PASS: {len(checks)} numerical checks. Results in {OUT.name}/", flush=True)

    return summary
