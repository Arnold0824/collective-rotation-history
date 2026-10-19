"""Small fixtures check file isolation and the optional fish refit workflow."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from fish.pipeline import refit_all


class ReleaseWorkflow(unittest.TestCase):
    def test_rejects_outputs_inside_input_data(self):
        with tempfile.TemporaryDirectory(prefix="rotation-isolation-") as directory:
            data = Path(directory)
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "reproduce.py"),
                       "--data-dir", str(data), "--output-dir", str(data / "outputs"), "--section", "core"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((data / "outputs").exists())

    def test_optional_refit_with_small_deterministic_fixture(self):
        # This fixture tests execution and file paths, not the paper's estimates.
        with tempfile.TemporaryDirectory(prefix="rotation-refit-fixture-") as directory:
            root = Path(directory)
            self.assertEqual(root.parent.resolve(), Path(tempfile.gettempdir()).resolve())
            data = root / "data"
            sources = data / "data/fish_directions"
            sources.mkdir(parents=True)
            protocol_dir = data / "analyses/fish_directions"
            protocol_dir.mkdir(parents=True)
            protocol = {
                "input_root": "data/fish_directions",
                "variants": [{"name": "primary", "minimum_step_pixels": 1e-12},
                             {"name": "step_gt_0p5px", "minimum_step_pixels": 0.5},
                             {"name": "step_gt_1px", "minimum_step_pixels": 1.0}],
                "continuous": {"horizons": [3, 10, 30], "current_matched_degree": 11,
                               "current_flexible_degrees": [3], "spline_breakpoints": [7],
                               "ridge_grid": [0.1]},
                "discrete": {"orders": [1, 4]},
            }
            (protocol_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            frames = np.arange(90)
            for size in (10, 30, 70):
                for trial in ("0001", "0002", "0003"):
                    angle = np.arange(size)[None, :] * 2 * np.pi / size + frames[:, None] * 0.03
                    positions = 200 * np.stack((np.cos(angle), np.sin(angle)), axis=-1)
                    np.savez_compressed(sources / f"N{size}_{trial}.npz", group_size=size, trial=trial,
                                        positions=positions, heading_angles=angle + np.pi / 2,
                                        frames=frames, frame_rate_hz=30.0, milling=np.ones(len(frames)),
                                        frame_valid=np.ones(len(frames), dtype=bool),
                                        order_run_id=np.zeros(len(frames), dtype=int))
            result = refit_all(data, root / "outputs")
            self.assertEqual(result["continuous_cases"], 162)
            self.assertEqual(result["discrete_cases"], 108)
            self.assertEqual(len(list((root / "outputs/models").glob("*.json"))), 918)
            print("Optional refit fixture: 810 continuous and 108 categorical models; all outputs completed.")


if __name__ == "__main__":
    unittest.main()
