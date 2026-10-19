"""Input and output locations supplied by the reproduction entry point."""
from pathlib import Path
import os


def data_root():
    return Path(os.environ["ROTATION_DATA_DIR"]).resolve()


def fish_input():
    return data_root() / "analyses" / "fish_directions"


def fish_output():
    path = Path(os.environ["ROTATION_FISH_OUTPUT_DIR"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
