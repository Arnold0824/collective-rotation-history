"""Access the small model-parameter and numerical-reference archive."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import shutil
import tempfile
import zipfile


@contextmanager
def reference_directory():
    root = Path(__file__).resolve().parents[1]
    archive_path = root / 'reference_data.zip'
    if archive_path.is_file():
        with tempfile.TemporaryDirectory(prefix='rotation-reference-') as temporary:
            destination = Path(temporary).resolve()
            with zipfile.ZipFile(archive_path) as archive:
                for entry in archive.infolist():
                    if not (destination / entry.filename).resolve().is_relative_to(destination):
                        raise ValueError('Reference archive contains an unsafe path')
                archive.extractall(destination)
            yield destination
    else:
        directory = root / 'reference'
        if not directory.is_dir():
            raise FileNotFoundError('reference_data.zip is missing from the code distribution')
        yield directory


def install_reference(source: Path, destination: Path):
    """Copy fixed expected results and models without overwriting existing files."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.rglob('*'):
        if not item.is_file():
            continue
        target = destination / item.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(item.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
                raise ValueError(f'Existing reference file differs: {target}. Choose a new data directory.')
        else:
            shutil.copy2(item, target)
