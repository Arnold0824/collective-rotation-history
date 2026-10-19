"""Generate the seven main and ten supplementary figures in paper order."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import tempfile
import numpy as np


def run(data_dir: Path, output_dir: Path):
    from . import overview, fish_transfer, style, panels, additional_panels
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = data_dir / "journal_graphics"
    overview_result = {}
    with tempfile.TemporaryDirectory(prefix="rotation-figure1-") as temporary:
        folder = Path(temporary)
        overview_result = overview.run(inputs / "overview/figure_spec.json", folder)
        for extension in ("pdf", "svg", "png"):
            shutil.copyfile(folder / f"figure_01_framework.{extension}", output_dir / f"figure1.{extension}")
        destination = output_dir / "checks" / "figure1"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("layout_checks.json", "source_manifest.json"):
            shutil.copyfile(folder / name, destination / name)

    style.HERE = output_dir
    style.MANIFEST.clear()
    panels.ROOT = inputs / "data_figures"
    panels.HERE = output_dir
    panels.OUT = output_dir
    panels.DATA = output_dir / "data"
    panels.DATA.mkdir(exist_ok=True)
    panels.INPUTS.clear(); panels.CHECKS.clear(); panels.OUTPUTS.clear()
    def export(name, frame):
        match = re.match(r"figure_(?:s\d+|\d+)", name)
        if match:
            name = style.public_name(match.group()) + name[match.end():]
        frame.to_csv(panels.DATA / f"{name}.csv", index=False, float_format="%.17g")
    panels.export = export
    panels.save = style.save
    panels.panel = style.panel
    style.style()
    for key in ("02", "03", "04", "05", "06", "s01", "s02", "s03", "s04", "s05", "s06", "s07", "s08", "s09", "s10"):
        module = additional_panels if key in {"03", "06", "s03", "s04", "s06"} else panels
        getattr(module, f"figure_{key}")()
    fish_result = fish_transfer.run(inputs / "fish_transfer", output_dir)
    names = [f"figure{i}" for i in range(1, 8)] + [f"figureS{i}" for i in range(1, 11)]
    files = [output_dir / f"{name}.{extension}" for name in names for extension in ("pdf", "svg", "png")]
    if not all(path.is_file() for path in files):
        raise AssertionError("The complete set of 17 figures was not produced")
    if not all(item["pass"] for item in panels.CHECKS.values()):
        raise AssertionError("A numerical figure check failed")
    if not all(not item["outside_text"] and item["ordinary_font_pass"] and item["standard_font_pass"] for item in style.MANIFEST.values()):
        raise AssertionError("A figure label or font check failed")
    report = {
        "passed": True,
        "figures": names,
        "file_count": len(files),
        "input_files": panels.INPUTS,
        "numeric_checks": {style.public_name(name): value for name, value in panels.CHECKS.items()},
        "figure_layouts": style.MANIFEST,
        "overview": overview_result,
        "fish_transfer": fish_result,
        "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "scope": "Figures regenerated from saved scientific results; no model fitting in this section.",
    }
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value))
    payload = json.dumps(report, indent=2, default=encode)
    (output_dir / "figure_manifest.json").write_text(payload + "\n", encoding="utf-8")
    print("Figures: all 7 main and 10 supplementary figures generated.", flush=True)
    return json.loads(payload)
