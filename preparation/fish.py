"""Prepare paired fish inputs from the original public Dryad processed archive.

No source measurements are downloaded or modified by this module. Generated
predictions are compared with the supplied, unchanged numerical references before
they become inputs to the independent reproduction checks.
"""
from __future__ import annotations

from pathlib import Path
import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd

from . import golden_shiners_parser as parser

DATASET_DOI = "10.5061/dryad.sbcc2fr2h"
ARCHIVE_URL = "https://datadryad.org/api/v2/files/867924/download"
ARCHIVE_BYTES = 1_053_800_358
ARCHIVE_SHA256 = "e2bde9af7d112d6f0abe75808682d4ca7a86dd0515bed54f5877a8512cd00dbd"
MEMBER = "savedresults-final/data-grid.pklz"
MEMBER_BYTES = 903_184_076
MEMBER_SHA256 = "cc2d4131ad8e3825c84885aee4f2c8f5cb5ce3934b0ce604dcd68a9f1becd49a"
FISH_REL = Path("analyses/fish_directions")
ARRAY_REL = Path("data/fish_directions")


def _sha(path: Path) -> str:
    return parser.sha256_file(path)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _field_hash(value: np.ndarray) -> dict:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": digest.hexdigest()}


def _verify_arrays(directory: Path, reference: dict, milling_reference: Path) -> list[dict]:
    if _sha(milling_reference) != reference["milling_reference_sha256"]:
        raise ValueError("The supplied milling reference checksum does not match its manifest")
    with np.load(milling_reference, allow_pickle=False) as stored:
        expected_milling = {key: stored[key] for key in stored.files}
    expected = {x["file"] for x in reference["recordings"]}
    actual = {p.name for p in directory.glob("*.npz")}
    if actual != expected:
        raise ValueError(f"Expected exactly the nine primary fish arrays; missing={expected-actual}, extra={actual-expected}")
    rows = []
    for record in reference["recordings"]:
        path = directory / record["file"]
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(record["fields"]):
                raise ValueError(f"Array fields differ: {path.name}")
            exact_fields = []
            milling_error = 0.0
            for name, expected_field in record["fields"].items():
                if name == "milling":
                    actual = archive[name]
                    expected_values = expected_milling[path.stem]
                    if actual.shape != expected_values.shape or actual.dtype != expected_values.dtype:
                        raise ValueError(f"Milling shape or dtype differs: {path.name}")
                    if not np.array_equal(np.isnan(actual), np.isnan(expected_values)):
                        raise ValueError(f"Milling missing-value positions differ: {path.name}")
                    np.testing.assert_allclose(actual, expected_values, rtol=0, atol=1e-14,
                                               equal_nan=True, err_msg=f"{path.name}/milling")
                    finite = np.isfinite(actual) & np.isfinite(expected_values)
                    if finite.any():
                        milling_error = float(np.max(np.abs(actual[finite]-expected_values[finite])))
                elif _field_hash(archive[name]) != expected_field:
                    raise ValueError(f"Public-source array differs from the numerical reference: {path.name}/{name}")
                else:
                    exact_fields.append(name)
            if str(archive["source_sha256"]) != MEMBER_SHA256:
                raise ValueError(f"Unexpected source attribution: {path.name}")
            if str(archive["dataset_doi"]) != DATASET_DOI:
                raise ValueError(f"Unexpected dataset DOI: {path.name}")
            rows.append({"file": path.name, "fields_verified": len(archive.files),
                         "exact_fields": exact_fields, "milling_max_abs_error": milling_error,
                         "milling_comparison": {"rtol": 0, "atol": 1e-14,
                                                "shape_dtype_and_missing_positions_exact": True},
                         "frames": int(len(archive["frames"])), "sha256": _sha(path)})
    return rows


def _source_pickle(source: Path, temporary: Path) -> tuple[Path, dict]:
    if source.suffix.lower() == ".zip":
        if source.stat().st_size != ARCHIVE_BYTES:
            raise ValueError("The ZIP size does not match Dryad file 867924")
        parser.verify_sha256(source, ARCHIVE_SHA256)
        with zipfile.ZipFile(source) as archive:
            candidates = [x for x in archive.infolist() if x.filename == MEMBER]
            if len(candidates) != 1 or candidates[0].file_size != MEMBER_BYTES:
                raise ValueError("The archive does not contain the expected unique data-grid.pklz")
            target = temporary / "data-grid.pklz"
            with archive.open(candidates[0]) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
        parser.verify_sha256(target, MEMBER_SHA256)
        return target, {"kind": "published_zip", "archive_sha256": ARCHIVE_SHA256,
                        "archive_bytes": ARCHIVE_BYTES, "member": MEMBER,
                        "member_sha256": MEMBER_SHA256, "member_bytes": MEMBER_BYTES}
    if source.name != "data-grid.pklz":
        raise ValueError("--fish-source must be saved_data_and_results.zip or data-grid.pklz")
    if source.stat().st_size != MEMBER_BYTES:
        raise ValueError("The data-grid.pklz size does not match the published member")
    parser.verify_sha256(source, MEMBER_SHA256)
    return source, {"kind": "published_member", "member": MEMBER,
                    "member_sha256": MEMBER_SHA256, "member_bytes": MEMBER_BYTES}


def _compare_table(reference: Path, generated: Path, keys: list[str], *, separately_checked=()) -> dict:
    old = pd.read_csv(reference).set_index(keys).sort_index()
    new = pd.read_csv(generated).set_index(keys).sort_index()
    if not old.index.equals(new.index) or set(old.columns) != set(new.columns):
        raise AssertionError(f"Rows or columns differ: {reference.name}")
    errors = {}
    for column in old.columns:
        a, b = old[column], new[column]
        if column == "model_file":
            a, b = a.str.replace("\\", "/", regex=False), b.str.replace("\\", "/", regex=False)
        if column in separately_checked:
            continue
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            np.testing.assert_allclose(av, bv, rtol=0, atol=1e-10, equal_nan=True,
                                       err_msg=f"{reference.name}/{column}")
            difference = np.abs(av-bv)
            errors[column] = float(np.nanmax(difference)) if np.isfinite(difference).any() else 0.0
        elif not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
            raise AssertionError(f"Recorded identities or choices differ: {reference.name}/{column}")
    return {"table": reference.name, "rows": len(old), "numeric_max_abs_error": errors}


def _compare_json_values(expected, actual, key: str = "") -> None:
    if isinstance(expected, dict):
        expected = {k: v for k, v in expected.items() if k != "protocol_sha256"}
        actual = {k: v for k, v in actual.items() if k != "protocol_sha256"}
        if set(expected) != set(actual):
            raise AssertionError(f"Model keys differ at {key}")
        for name in expected:
            _compare_json_values(expected[name], actual[name], f"{key}/{name}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"Model list shape differs at {key}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            _compare_json_values(a, b, f"{key}/{i}")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        tolerance = 1e-7 if "/parameters/" in key else 1e-10
        if not isinstance(actual, (float, int)) or not math.isclose(expected, actual, rel_tol=0, abs_tol=tolerance):
            raise AssertionError(f"Model numerical value differs at {key}: {expected} != {actual}")
    elif expected != actual:
        raise AssertionError(f"Model value differs at {key}: {expected} != {actual}")


def _compare_refit(reference: Path, generated: Path, diagnostics: Path | None = None) -> dict:
    from .fish_comparison import compare_candidates, compare_models
    diagnostics = diagnostics or generated / "comparison_diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    # Preserve the complete small results before any comparison can fail.
    for path in generated.glob("*.csv"):
        shutil.copyfile(path, diagnostics / ("generated_" + path.name))
    shutil.copytree(generated / "models", diagnostics / "generated_models", dirs_exist_ok=True)
    cases = [
        ("recording_metrics.csv", ["variant", "definition", "trial_id", "horizon_frames", "model"]),
        ("discrete_metrics.csv", ["variant", "definition", "trial_id", "order"]),
        ("model_inventory.csv", ["variant", "definition", "trial_id", "horizon_frames", "model"]),
        ("observation_quality.csv", ["variant", "trial_id"]),
        ("sample_coverage.csv", ["variant", "trial_id", "horizon_frames"]),
    ]
    result = {"passed": False, "tables": [], "diagnostics": str(diagnostics)}
    try:
        for name, keys in cases:
            result["tables"].append(_compare_table(reference/name, generated/name, keys,
                separately_checked=("effective_df",) if name == "model_inventory.csv" else ()))
        result["candidates"] = compare_candidates(reference/"training_selection.csv", generated/"training_selection.csv", diagnostics)
        result["models"] = compare_models(reference, generated, diagnostics)
        # The inventory's diagnostic must agree with its corresponding model file.
        for folder in (reference, generated):
            inventory = pd.read_csv(folder/"model_inventory.csv")
            for row in inventory.itertuples():
                model = json.loads((folder/row.model_file).read_text())
                if not math.isclose(row.effective_df, model["parameters"]["effective_df"], rel_tol=0, abs_tol=1e-12):
                    raise AssertionError("Inventory effective_df differs from the corresponding model")
        result["passed"] = True
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        _write_json(diagnostics/"reference_comparison.json", result)
    return result


def prepare_fish(fish_source: Path, reference_dir: Path, data_dir: Path) -> dict:
    """Build the nine public-source arrays, refit all fish models, and verify them.

    Small reference metrics, model parameters and selection records remain the
    independent expectations. Only generated observations and predictions are
    copied from the temporary refit into the prepared analysis directory.
    """
    source, reference_root, destination = map(lambda p: Path(p).resolve(), (fish_source, reference_dir, data_dir))
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.is_relative_to(destination):
        raise ValueError("The generated data directory must not contain the read-only source file")
    if destination == reference_root or destination.is_relative_to(reference_root) or reference_root.is_relative_to(destination):
        raise ValueError("Reference and generated data directories must be separate")
    expected_dir = reference_root / FISH_REL
    target_dir = destination / FISH_REL
    required = ["protocol.json", "source_arrays.json", "source_milling_reference.npz", "training_selection.csv",
                "recording_metrics.csv", "discrete_metrics.csv", "model_inventory.csv",
                "observation_quality.csv", "sample_coverage.csv"]
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        expected = expected_dir / name
        if not expected.is_file():
            raise FileNotFoundError(f"Missing immutable fish reference: {expected}")
        target = target_dir / name
        if target.exists():
            if _sha(target) != _sha(expected):
                raise ValueError(f"Installed fish reference was changed: {target}")
        else:
            shutil.copyfile(expected, target)
    if not (target_dir / "models").is_dir():
        shutil.copytree(expected_dir / "models", target_dir / "models")
    before = {str(p.relative_to(expected_dir)): _sha(p) for p in expected_dir.rglob("*") if p.is_file()}
    started = time.monotonic()
    source_stat = source.stat()
    result = {"dataset_doi": DATASET_DOI, "download_url": ARCHIVE_URL, "passed": False}
    with tempfile.TemporaryDirectory(prefix="fish-source-", dir=destination.parent) as temp:
        pickle_path, source_info = _source_pickle(source, Path(temp))
        result["source"] = source_info
        arrays = destination / ARRAY_REL
        array_reference = json.loads((expected_dir / "source_arrays.json").read_text())
        if not arrays.exists():
            print("Fish: loading the verified public processed member", flush=True)
            payload = parser.restricted_numpy_load(pickle_path, expected_sha256=MEMBER_SHA256)
            trials = parser.parse_data_grid_payload(payload, source_sha256=MEMBER_SHA256,
                source_path=pickle_path, frame_rate_hz=30.0, minimum_finite_fraction=.8)
            arrays.parent.mkdir(parents=True, exist_ok=True)
            staged = Path(temp) / "arrays"
            staged.mkdir()
            for spec, trial in zip(parser.GOLDEN_SHINER_TRIAL_SPECS, trials, strict=True):
                if spec.primary_confirmation:
                    parser._write_trial_npz(staged / spec.output_filename, trial)
            result["source_array_checks"] = _verify_arrays(staged, array_reference, expected_dir / "source_milling_reference.npz")
            shutil.move(str(staged), str(arrays))
            del trials, trial, payload
            gc.collect()
        else:
            result["source_array_checks"] = _verify_arrays(arrays, array_reference, expected_dir / "source_milling_reference.npz")
    if (source.stat().st_size, source.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise RuntimeError("Source file changed during conversion")
    print("Fish: nine arrays numerically verified; original and metadata fields exact, milling within 1e-14; starting complete nested refits", flush=True)
    from fish.pipeline import refit_all
    env_names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    old_env = {k: os.environ.get(k) for k in env_names}
    try:
        for k in env_names:
            os.environ[k] = "1"
        with tempfile.TemporaryDirectory(prefix="fish-refit-", dir=destination.parent) as temp:
            generated = Path(temp) / "analysis"
            result["refit"] = refit_all(destination, generated)
            result["reference_comparison"] = _compare_refit(expected_dir, generated,
                destination / "fish_preparation_diagnostics")
            result["generated_inputs"] = []
            for name in ("observation_frames.parquet", "predictions.parquet", "discrete_predictions.parquet"):
                target = target_dir / name
                shutil.copyfile(generated / name, target)
                result["generated_inputs"].append({"file": str((FISH_REL/name).as_posix()),
                    "bytes": target.stat().st_size, "sha256": _sha(target)})
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    after = {str(p.relative_to(expected_dir)): _sha(p) for p in expected_dir.rglob("*") if p.is_file()}
    if before != after:
        raise RuntimeError("A reference file changed during preparation")
    result.update(passed=True, reference_files_unchanged=True,
                  elapsed_seconds=time.monotonic()-started,
                  scope="Public processed source to nine numerically verified arrays (original/metadata fields exact; milling within 1e-14), all observation definitions and complete nested fish refits; numerical references remain unchanged.")
    _write_json(destination / "fish_preparation.json", result)
    print("PASS: complete fish preparation and original-reference comparisons", flush=True)
    return result
