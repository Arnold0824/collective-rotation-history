"""Reproduce collective-rotation scores, fish checks and publication figures."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Analysis inputs constructed locally with prepare.py; optional for figures")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for newly computed results")
    parser.add_argument("--section", choices=("core", "fish", "figures", "all"), default="all")
    parser.add_argument("--refit-fish-all", action="store_true", help="Also repeat all paired fish model-selection fits; substantially slower")
    args = parser.parse_args()
    with ExitStack() as resources:
        if args.data_dir is None:
            if args.section != "figures":
                parser.error("Use prepare.py to construct --data-dir from the public source datasets, or select --section figures")
            from preparation.reference import reference_directory
            data_dir = resources.enter_context(reference_directory())
        else:
            data_dir = args.data_dir.resolve()
        output_dir = args.output_dir.resolve()
        if not data_dir.is_dir():
            parser.error(f"Data directory does not exist: {data_dir}")
        if output_dir == data_dir or output_dir.is_relative_to(data_dir):
            parser.error("The output directory must be outside the read-only data directory")
        if args.refit_fish_all and args.section not in ("fish", "all"):
            parser.error("--refit-fish-all requires --section fish or all")
        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        results = {}
        if args.section in ("core", "all"):
            from methods.score import run
            results["core"] = run(data_dir, output_dir / "core")
        if args.section in ("fish", "all"):
            from fish.pipeline import run, refit_all
            results["fish"] = run(data_dir, output_dir / "fish")
            if args.refit_fish_all:
                results["fish_complete_refit"] = refit_all(data_dir, output_dir / "fish_complete_refit")
        if args.section in ("figures", "all"):
            from figures.generate import run
            results["figures"] = run(data_dir, output_dir / "figures")
        result = {
            "passed": True,
            "section": args.section,
            "elapsed_seconds": time.monotonic() - start,
            "sections": results,
            "scope": "Saved-result reproduction and the explicitly reported refits; figures alone do not recompute the experiments.",
        }
        (output_dir / "reproduction_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"PASS: {args.section}; summary: {output_dir / 'reproduction_summary.json'}")
        return result


if __name__ == "__main__":
    main()
