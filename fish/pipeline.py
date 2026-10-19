"""Reconstruct fish observables and verify all saved paired predictions."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys


def execute(module, data_dir, output_dir):
    environment = os.environ.copy()
    environment["ROTATION_DATA_DIR"] = str(data_dir)
    environment["ROTATION_FISH_OUTPUT_DIR"] = str(output_dir)
    subprocess.run(
        [sys.executable, "-m", module],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
    )


def run(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for module in (
        "fish.test_analysis",
        "fish.recalculate_observables",
        "fish.verify_saved_predictions",
        "fish.verify_selected_fits",
    ):
        execute(module, data_dir, output_dir)
    def read(name):
        return json.loads((output_dir / name).read_text(encoding="utf-8"))
    unit = read("test_results.json")
    source = read("observable_reconstruction_checks.json")
    predictions = read("saved_prediction_checks.json")
    fits = read("refit_and_score_checks.json")
    result = {
        "passed": unit["passed"] and predictions["passed"] and fits["checks"] == fits["passed"],
        "scientific_tests": unit["tests_run"],
        "source_recordings_reconstructed": len(source),
        "saved_prediction_checks": len(predictions["checks"]),
        "selection_and_score_checks": fits["checks"],
        "independent_selected_model_refits": fits["independent_refits"],
        "scope": "All saved paired predictions, source observations, training isolation and 20 selected-model refits; not a repeat of every model-selection fit.",
    }
    (output_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise AssertionError("Fish reproduction checks failed")
    return result


def refit_all(data_dir: Path, output_dir: Path):
    """Repeat observation construction and all nested fish fits from the protocol."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data_dir / "analyses/fish_directions/protocol.json", output_dir / "protocol.json")
    execute("fish.build_observations", data_dir, output_dir)
    execute("fish.run_analysis", data_dir, output_dir)
    result = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    result["scope"] = "Complete paired fish observation construction and model selection/refitting for all three displacement supports."
    return result
