"""Safe ingestion helpers for the public golden-shiner trajectory data.

The processed Dryad bundle for Davidson et al. stores ``data-grid.pklz`` as a
gzip-compressed Python pickle.  Pickle is executable data, so this module never
uses :func:`pickle.load`: it admits only the small set of constructors needed
for ordinary NumPy arrays, rejects object arrays, and caps both the decoded
stream and the resulting array payload.

The public conversion target is one ``.npz`` file per biological trial.  Those
files contain only non-object NumPy arrays and can subsequently be opened with
``allow_pickle=False``.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import ntpath
import os
import pickle
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

try:
    from numpy._core import multiarray as _numpy_multiarray
    from numpy._core import numeric as _numpy_numeric
except ImportError:  # NumPy 1.x compatibility for source conversion environments
    from numpy.core import multiarray as _numpy_multiarray
    from numpy.core import numeric as _numpy_numeric

DATASET_DOI = "10.5061/dryad.sbcc2fr2h"
EXPECTED_PROCESSED_ARCHIVE_BYTES = 1_053_800_358
EXPECTED_PROCESSED_ARCHIVE_SHA256 = (
    "e2bde9af7d112d6f0abe75808682d4ca7a86dd0515bed54f5877a8512cd00dbd"
)
DEFAULT_FRAME_RATE_HZ = 30.0
RAW_HDF5_GROUP_STATE = "raw_hdf5_not_provided"

_MEBIBYTE = 1024**2
_GIBIBYTE = 1024**3
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


class ZipSafetyError(ValueError):
    """Raised before an unsafe or unexpectedly large ZIP member is extracted."""


class UnsafePickleError(pickle.UnpicklingError):
    """Raised when the processed pickle exceeds the deliberately narrow contract."""


@dataclass(frozen=True)
class GoldenShinerTrialSpec:
    """Published ordering and identity of one trial inside ``data-grid.pklz``."""

    source_index: int
    group_size: int
    trial: str
    source_directory: str
    primary_confirmation: bool = True

    @property
    def trial_id(self) -> str:
        return f"N{self.group_size}{self.trial}"

    @property
    def output_filename(self) -> str:
        safe_trial = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.trial)
        return f"N{self.group_size}_{safe_trial}.npz"


# This ordering is stated in the publication's ``4 - Data figures.ipynb``.
# The final source directory is called ``150-fish`` even though it contains 151
# tracked fish; retain both facts instead of silently renaming the source.
GOLDEN_SHINER_TRIAL_SPECS: tuple[GoldenShinerTrialSpec, ...] = (
    GoldenShinerTrialSpec(0, 10, "0066", "10-fish/0066"),
    GoldenShinerTrialSpec(1, 10, "0105", "10-fish/0105"),
    GoldenShinerTrialSpec(2, 10, "0126", "10-fish/0126"),
    GoldenShinerTrialSpec(3, 30, "0084", "30-fish/0084"),
    GoldenShinerTrialSpec(4, 30, "0115", "30-fish/0115"),
    GoldenShinerTrialSpec(5, 30, "0120", "30-fish/0120"),
    GoldenShinerTrialSpec(6, 70, "0103", "70-fish/0103"),
    GoldenShinerTrialSpec(7, 70, "0107", "70-fish/0107"),
    GoldenShinerTrialSpec(8, 70, "0124", "70-fish/0124"),
    GoldenShinerTrialSpec(
        9,
        151,
        "150-fish",
        "150-fish",
        primary_confirmation=False,
    ),
)


@dataclass(frozen=True)
class ZipMemberRecord:
    """Validated ZIP metadata, without extracting the member."""

    name: str
    normalized_name: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int
    is_directory: bool


@dataclass(frozen=True)
class MillingSeries:
    """Signed milling values and frame-level observation quality."""

    values: np.ndarray
    finite_fraction: np.ndarray
    frame_valid: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        finite_fraction = np.asarray(self.finite_fraction, dtype=np.float64)
        frame_valid = np.asarray(self.frame_valid, dtype=bool)
        if values.ndim != 1 or finite_fraction.shape != values.shape:
            raise ValueError("milling arrays must be one-dimensional and equally sized")
        if frame_valid.shape != values.shape:
            raise ValueError("frame_valid has an incompatible shape")
        if not np.isfinite(finite_fraction).all() or np.any(
            (finite_fraction < 0.0) | (finite_fraction > 1.0)
        ):
            raise ValueError("finite fractions must lie in [0, 1]")
        if not np.isfinite(values[frame_valid]).all():
            raise ValueError("valid milling frames must be finite")
        if np.isfinite(values[~frame_valid]).any():
            raise ValueError("invalid milling frames must be NaN")
        object.__setattr__(self, "values", np.ascontiguousarray(values))
        object.__setattr__(self, "finite_fraction", np.ascontiguousarray(finite_fraction))
        object.__setattr__(self, "frame_valid", np.ascontiguousarray(frame_valid))


@dataclass(frozen=True)
class GoldenShinerTrialArrays:
    """Safe, analysis-ready arrays for one converted fish trial."""

    group_size: int
    trial: str
    frame_rate_hz: float
    frames: np.ndarray
    positions: np.ndarray
    heading_angles: np.ndarray
    group_states: np.ndarray
    milling: np.ndarray
    finite_fraction: np.ndarray
    frame_valid: np.ndarray
    order_run_id: np.ndarray
    source_sha256: str
    source_path: Path

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames, dtype=np.int64)
        positions = np.asarray(self.positions, dtype=np.float64)
        headings = np.asarray(self.heading_angles, dtype=np.float64)
        states = np.asarray(self.group_states)
        milling = np.asarray(self.milling, dtype=np.float64)
        fractions = np.asarray(self.finite_fraction, dtype=np.float64)
        frame_valid = np.asarray(self.frame_valid, dtype=bool)
        run_id = np.asarray(self.order_run_id, dtype=np.int64)
        count = len(frames)
        if self.group_size <= 0 or not self.trial:
            raise ValueError("a fish trial needs a positive group size and label")
        if not np.isfinite(self.frame_rate_hz) or self.frame_rate_hz <= 0.0:
            raise ValueError("frame_rate_hz must be positive")
        if count == 0 or not np.all(np.diff(frames) > 0):
            raise ValueError("frames must be nonempty, unique, and increasing")
        if positions.shape != (count, self.group_size, 2):
            raise ValueError("positions have the wrong trial shape")
        if headings.shape != (count, self.group_size):
            raise ValueError("heading angles have the wrong trial shape")
        if states.shape != (count,) or states.dtype.kind not in "US":
            raise ValueError("group states must be a one-dimensional text array")
        for name, array in (
            ("milling", milling),
            ("finite_fraction", fractions),
            ("frame_valid", frame_valid),
            ("order_run_id", run_id),
        ):
            if array.shape != (count,):
                raise ValueError(f"{name} has the wrong trial shape")
        MillingSeries(milling, fractions, frame_valid)
        if np.any((run_id >= 0) & ~frame_valid):
            raise ValueError("an invalid frame may not belong to an order run")
        if not _SHA256_PATTERN.fullmatch(str(self.source_sha256)):
            raise ValueError("source_sha256 must contain exactly 64 hexadecimal characters")
        object.__setattr__(self, "frames", np.ascontiguousarray(frames))
        object.__setattr__(self, "positions", np.ascontiguousarray(positions))
        object.__setattr__(self, "heading_angles", np.ascontiguousarray(headings))
        object.__setattr__(self, "group_states", np.ascontiguousarray(states.astype(str)))
        object.__setattr__(self, "milling", np.ascontiguousarray(milling))
        object.__setattr__(self, "finite_fraction", np.ascontiguousarray(fractions))
        object.__setattr__(self, "frame_valid", np.ascontiguousarray(frame_valid))
        object.__setattr__(self, "order_run_id", np.ascontiguousarray(run_id))
        object.__setattr__(self, "source_sha256", str(self.source_sha256).lower())
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())

    @property
    def trial_id(self) -> str:
        return f"N{self.group_size}{self.trial}"

def sha256_file(path: str | Path, *, chunk_size: int = 4 * _MEBIBYTE) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected_sha256: str) -> str:
    """Verify a file digest and return the normalized actual digest."""

    expected = str(expected_sha256).lower()
    if not _SHA256_PATTERN.fullmatch(expected):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(f"SHA-256 mismatch for {Path(path)}: expected {expected}, found {actual}")
    return actual


def verify_processed_archive(
    path: str | Path,
    *,
    expected_sha256: str = EXPECTED_PROCESSED_ARCHIVE_SHA256,
    expected_bytes: int = EXPECTED_PROCESSED_ARCHIVE_BYTES,
) -> str:
    """Verify the published processed archive's exact byte count and digest."""

    source = Path(path)
    if source.stat().st_size != int(expected_bytes):
        raise ValueError(
            f"archive byte-count mismatch: expected {expected_bytes}, found {source.stat().st_size}"
        )
    return verify_sha256(source, expected_sha256)


def _normalized_zip_name(name: str) -> str:
    if not name or "\x00" in name:
        raise ZipSafetyError("ZIP members need a nonempty, NUL-free name")
    if "\\" in name or "//" in name:
        raise ZipSafetyError(f"ambiguous separator in ZIP member: {name!r}")
    drive, _ = ntpath.splitdrive(name)
    if drive or name.startswith(("/", "\\")):
        raise ZipSafetyError(f"absolute ZIP member path is forbidden: {name!r}")
    pure = PurePosixPath(name)
    parts = pure.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ZipSafetyError(f"unsafe ZIP member path: {name!r}")
    if any(":" in part for part in parts):
        raise ZipSafetyError(f"drive or alternate-stream syntax is forbidden: {name!r}")
    return pure.as_posix()


def inspect_zip_archive(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    max_members: int = 10_000,
    max_member_uncompressed_bytes: int = 16 * _GIBIBYTE,
    max_total_uncompressed_bytes: int = 32 * _GIBIBYTE,
    max_compression_ratio: float = 1_000.0,
) -> tuple[ZipMemberRecord, ...]:
    """Validate a ZIP inventory without writing any member to disk."""

    if expected_sha256 is not None:
        verify_sha256(path, expected_sha256)
    if max_members < 1 or max_member_uncompressed_bytes < 1 or max_total_uncompressed_bytes < 1:
        raise ValueError("ZIP size limits must be positive")
    if max_compression_ratio < 1.0:
        raise ValueError("max_compression_ratio must be at least one")

    records: list[ZipMemberRecord] = []
    normalized_seen: set[str] = set()
    total = 0
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        if not infos:
            raise ZipSafetyError("ZIP archive is empty")
        if len(infos) > max_members:
            raise ZipSafetyError(
                f"ZIP archive contains {len(infos)} members; limit is {max_members}"
            )
        for info in infos:
            normalized = _normalized_zip_name(info.filename)
            collision_key = normalized.casefold().rstrip("/")
            if collision_key in normalized_seen:
                raise ZipSafetyError(f"duplicate normalized ZIP member: {info.filename!r}")
            normalized_seen.add(collision_key)
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise ZipSafetyError(f"symbolic-link ZIP member is forbidden: {info.filename!r}")
            if info.flag_bits & 0x1:
                raise ZipSafetyError(f"encrypted ZIP member is unsupported: {info.filename!r}")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ZipSafetyError(f"unsupported ZIP compression for {info.filename!r}")
            if info.file_size < 0 or info.compress_size < 0:
                raise ZipSafetyError(f"negative ZIP member size: {info.filename!r}")
            if info.file_size > max_member_uncompressed_bytes:
                raise ZipSafetyError(f"ZIP member exceeds its size limit: {info.filename!r}")
            total += int(info.file_size)
            if total > max_total_uncompressed_bytes:
                raise ZipSafetyError("ZIP archive exceeds the total uncompressed-size limit")
            if not info.is_dir() and info.file_size:
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > max_compression_ratio:
                    raise ZipSafetyError(
                        f"ZIP member exceeds the compression-ratio limit: {info.filename!r}"
                    )
            records.append(
                ZipMemberRecord(
                    name=info.filename,
                    normalized_name=normalized,
                    compressed_bytes=int(info.compress_size),
                    uncompressed_bytes=int(info.file_size),
                    crc32=int(info.CRC),
                    is_directory=info.is_dir(),
                )
            )
    return tuple(records)


def find_unique_zip_member(records: Sequence[ZipMemberRecord], *, basename: str) -> ZipMemberRecord:
    """Find one validated member by basename, rejecting ambiguity."""

    selected = [
        record for record in records if PurePosixPath(record.normalized_name).name == basename
    ]
    if len(selected) != 1:
        raise ZipSafetyError(
            f"expected exactly one ZIP member named {basename!r}, found {len(selected)}"
        )
    if selected[0].is_directory:
        raise ZipSafetyError(f"requested ZIP member is a directory: {basename!r}")
    return selected[0]


def safe_extract_zip(
    path: str | Path,
    destination: str | Path,
    *,
    members: Sequence[str] | None = None,
    expected_sha256: str | None = None,
    max_members: int = 10_000,
    max_member_uncompressed_bytes: int = 16 * _GIBIBYTE,
    max_total_uncompressed_bytes: int = 32 * _GIBIBYTE,
    max_compression_ratio: float = 1_000.0,
    chunk_size: int = 4 * _MEBIBYTE,
) -> tuple[Path, ...]:
    """Extract validated members into a new directory through a staging directory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    records = inspect_zip_archive(
        path,
        expected_sha256=expected_sha256,
        max_members=max_members,
        max_member_uncompressed_bytes=max_member_uncompressed_bytes,
        max_total_uncompressed_bytes=max_total_uncompressed_bytes,
        max_compression_ratio=max_compression_ratio,
    )
    record_by_name = {record.name: record for record in records}
    if members is None:
        selected = [record for record in records if not record.is_directory]
    else:
        requested = list(members)
        if len(requested) != len(set(requested)):
            raise ValueError("members may not contain duplicate names")
        missing = [name for name in requested if name not in record_by_name]
        if missing:
            raise KeyError(f"ZIP members not found: {missing}")
        selected = [record_by_name[name] for name in requested]
        if any(record.is_directory for record in selected):
            raise ZipSafetyError("explicit extraction members must be files")

    target_root = Path(destination).resolve()
    if target_root.exists():
        raise FileExistsError(f"safe extraction destination already exists: {target_root}")
    target_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target_root.name}.staging-", dir=target_root.parent))
    extracted_relative: list[Path] = []
    try:
        with zipfile.ZipFile(path, "r") as archive:
            for record in selected:
                relative = Path(*PurePosixPath(record.normalized_name).parts)
                output = staging / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with archive.open(record.name, "r") as source, output.open("xb") as sink:
                    while True:
                        chunk = source.read(chunk_size)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > record.uncompressed_bytes:
                            raise ZipSafetyError(
                                f"ZIP member expanded beyond its declared size: {record.name!r}"
                            )
                        sink.write(chunk)
                if written != record.uncompressed_bytes:
                    raise ZipSafetyError(
                        f"ZIP member size differs from its declaration: {record.name!r}"
                    )
                extracted_relative.append(relative)
        os.replace(staging, target_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return tuple(target_root / relative for relative in extracted_relative)


class _LimitedReader:
    """File-like wrapper that aborts after a decoded byte budget is exceeded."""

    def __init__(self, raw: BinaryIO, limit: int) -> None:
        if limit < 1:
            raise ValueError("decoded stream limit must be positive")
        self.raw = raw
        self.limit = int(limit)
        self.consumed = 0

    def _request_size(self, size: int) -> int:
        remaining_plus_probe = self.limit - self.consumed + 1
        if size is None or size < 0:
            return max(remaining_plus_probe, 1)
        return min(int(size), max(remaining_plus_probe, 1))

    def _record(self, data: bytes) -> bytes:
        self.consumed += len(data)
        if self.consumed > self.limit:
            raise UnsafePickleError(f"decoded pickle stream exceeds {self.limit} bytes")
        return data

    def read(self, size: int = -1) -> bytes:
        return self._record(self.raw.read(self._request_size(size)))

    def readline(self, size: int = -1) -> bytes:
        return self._record(self.raw.readline(self._request_size(size)))

    def readinto(self, buffer: bytearray | memoryview) -> int:
        maximum = min(len(buffer), self._request_size(len(buffer)))
        data = self.read(maximum)
        buffer[: len(data)] = data
        return len(data)


_ALLOWED_PICKLE_GLOBALS: dict[tuple[str, str], Any] = {
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): _numpy_multiarray.scalar,
    ("numpy._core.multiarray", "scalar"): _numpy_multiarray.scalar,
    ("numpy.core.numeric", "_frombuffer"): _numpy_numeric._frombuffer,
    ("numpy._core.numeric", "_frombuffer"): _numpy_numeric._frombuffer,
    ("builtins", "complex"): complex,
    ("builtins", "slice"): slice,
}


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        key = (str(module), str(name))
        if key not in _ALLOWED_PICKLE_GLOBALS:
            raise UnsafePickleError(f"forbidden pickle global: {module}.{name}")
        return _ALLOWED_PICKLE_GLOBALS[key]

    def persistent_load(self, pid: object) -> Any:
        raise UnsafePickleError(f"persistent pickle references are forbidden: {pid!r}")


def _validate_safe_payload_tree(
    value: object,
    *,
    max_array_bytes: int,
    max_container_items: int,
) -> None:
    array_bytes = 0
    container_items = 0
    stack: list[object] = [value]
    seen_containers: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, np.ndarray):
            if current.dtype.hasobject or current.dtype.fields is not None:
                raise UnsafePickleError("object and structured NumPy arrays are forbidden")
            if current.dtype.kind not in "biufcUS":
                raise UnsafePickleError(f"unsupported NumPy dtype: {current.dtype}")
            array_bytes += int(current.nbytes)
            if array_bytes > max_array_bytes:
                raise UnsafePickleError(
                    f"NumPy payload exceeds the {max_array_bytes}-byte array budget"
                )
            continue
        if isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            container_items += len(current)
            if container_items > max_container_items:
                raise UnsafePickleError("pickle contains too many container items")
            stack.extend(current)
            continue
        if current is None or isinstance(current, (bool, int, float, complex, str, bytes)):
            if isinstance(current, (str, bytes)) and len(current) > _MEBIBYTE:
                raise UnsafePickleError("pickle contains an unexpectedly large scalar string")
            continue
        if isinstance(current, np.generic):
            if current.dtype.hasobject:
                raise UnsafePickleError("object NumPy scalars are forbidden")
            continue
        raise UnsafePickleError(f"unsupported decoded pickle value: {type(current).__name__}")


def restricted_numpy_load(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    max_decoded_bytes: int = 16 * _GIBIBYTE,
    max_array_bytes: int = 16 * _GIBIBYTE,
    max_container_items: int = 100_000,
) -> object:
    """Load a gzip/plain NumPy pickle through a constructor allowlist and budgets."""

    source = Path(path)
    if expected_sha256 is not None:
        verify_sha256(source, expected_sha256)
    if max_array_bytes < 1 or max_container_items < 1:
        raise ValueError("pickle validation budgets must be positive")

    try:
        with source.open("rb") as raw:
            magic = raw.read(2)
            raw.seek(0)
            decoded: BinaryIO
            if magic == b"\x1f\x8b":
                decoded = gzip.GzipFile(fileobj=raw, mode="rb")
            else:
                decoded = raw
            limited = _LimitedReader(decoded, int(max_decoded_bytes))
            value = _RestrictedNumpyUnpickler(limited).load()
            if decoded is not raw:
                decoded.close()
    except UnsafePickleError:
        raise
    except (EOFError, OSError, ValueError, pickle.UnpicklingError) as exc:
        raise UnsafePickleError(f"cannot decode restricted NumPy pickle: {exc}") from exc

    _validate_safe_payload_tree(
        value,
        max_array_bytes=int(max_array_bytes),
        max_container_items=int(max_container_items),
    )
    return value


def compute_signed_milling(
    positions: np.ndarray,
    orientations_or_headings: np.ndarray,
    *,
    minimum_finite_fraction: float = 0.8,
    tolerance: float = 1.0e-12,
) -> MillingSeries:
    """Compute ``mean(r_hat cross h_hat)`` independently for every frame.

    ``orientations_or_headings`` may contain heading angles with shape ``(T,N)``
    or heading vectors with shape ``(T,N,2)``.  The sign follows the supplied
    coordinate system; reflecting either spatial axis reverses it.
    """

    points = np.asarray(positions, dtype=np.float64)
    supplied = np.asarray(orientations_or_headings, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 2 or points.shape[1] < 1:
        raise ValueError("positions must have shape (frames, individuals, 2)")
    if supplied.shape == points.shape[:2]:
        heading = np.stack((np.cos(supplied), np.sin(supplied)), axis=-1)
        heading[~np.isfinite(supplied)] = np.nan
    elif supplied.shape == points.shape:
        heading = supplied
    else:
        raise ValueError("headings must be angles (T,N) or vectors (T,N,2)")
    if not 0.0 < minimum_finite_fraction <= 1.0:
        raise ValueError("minimum_finite_fraction must lie in (0, 1]")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")

    finite = np.isfinite(points).all(axis=2) & np.isfinite(heading).all(axis=2)
    fractions = finite.mean(axis=1, dtype=np.float64)
    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    frame_valid = np.zeros(points.shape[0], dtype=bool)
    for frame_index in np.flatnonzero(fractions >= minimum_finite_fraction):
        selected = finite[frame_index]
        frame_points = points[frame_index, selected]
        frame_heading = heading[frame_index, selected]
        center = frame_points.mean(axis=0)
        radial = frame_points - center
        radial_norm = np.linalg.norm(radial, axis=1)
        heading_norm = np.linalg.norm(frame_heading, axis=1)
        geometric = (radial_norm > tolerance) & (heading_norm > tolerance)
        if not geometric.any():
            continue
        numerator = (
            radial[geometric, 0] * frame_heading[geometric, 1]
            - radial[geometric, 1] * frame_heading[geometric, 0]
        )
        signed = numerator / (radial_norm[geometric] * heading_norm[geometric])
        value = float(np.mean(signed))
        if np.isfinite(value):
            values[frame_index] = float(np.clip(value, -1.0, 1.0))
            frame_valid[frame_index] = True
    return MillingSeries(values=values, finite_fraction=fractions, frame_valid=frame_valid)


def build_order_run_ids(
    frames: np.ndarray,
    frame_valid: np.ndarray,
    *,
    minimum_run_length: int = 2,
) -> np.ndarray:
    """Assign consecutive positive run IDs and ``-1`` to unusable frames."""

    frame_values = np.asarray(frames, dtype=np.int64)
    valid = np.asarray(frame_valid, dtype=bool)
    if frame_values.ndim != 1 or valid.shape != frame_values.shape or len(frame_values) == 0:
        raise ValueError("frames and frame_valid must be nonempty one-dimensional arrays")
    if not np.all(np.diff(frame_values) > 0):
        raise ValueError("frames must be strictly increasing")
    if minimum_run_length < 1:
        raise ValueError("minimum_run_length must be positive")

    output = np.full(len(frame_values), -1, dtype=np.int64)
    next_run_id = 1
    start: int | None = None
    for index in range(len(frame_values) + 1):
        continues = (
            index < len(frame_values)
            and valid[index]
            and (index == 0 or start is None or frame_values[index] == frame_values[index - 1] + 1)
        )
        if continues:
            if start is None:
                start = index
            continue
        if start is not None:
            if index - start >= minimum_run_length:
                output[start:index] = next_run_id
                next_run_id += 1
            start = None
        if index < len(frame_values) and valid[index]:
            start = index
    return output


def _text_states(values: object, *, frame_count: int) -> np.ndarray:
    states = np.asarray(values)
    if states.shape != (frame_count,) or states.dtype.kind not in "US":
        raise ValueError("each published group-state array must be one-dimensional text")
    if states.dtype.kind == "S":
        states = np.char.decode(states, "utf-8", errors="strict")
    return np.ascontiguousarray(states.astype(str))


def _published_fields(
    payload: object,
) -> tuple[Sequence[object], Sequence[object], Sequence[object]]:
    if not isinstance(payload, (list, tuple)) or len(payload) != 10:
        raise ValueError("data-grid.pklz must contain the published ten-field outer sequence")
    group_states, orientations, positions = payload[2], payload[4], payload[6]
    for name, field in (
        ("grid_groupstates", group_states),
        ("grid_orientations", orientations),
        ("grid_positions", positions),
    ):
        if not isinstance(field, (list, tuple)) or len(field) != len(GOLDEN_SHINER_TRIAL_SPECS):
            raise ValueError(f"{name} must contain exactly ten trial arrays")
    return group_states, orientations, positions


def parse_data_grid_payload(
    payload: object,
    *,
    source_sha256: str,
    source_path: str | Path,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    minimum_finite_fraction: float = 0.8,
) -> tuple[GoldenShinerTrialArrays, ...]:
    """Validate the published ten-trial layout and compute signed milling series."""

    if not _SHA256_PATTERN.fullmatch(str(source_sha256)):
        raise ValueError("source_sha256 must contain exactly 64 hexadecimal characters")
    group_states, orientations, positions = _published_fields(payload)
    trials: list[GoldenShinerTrialArrays] = []
    for spec in GOLDEN_SHINER_TRIAL_SPECS:
        trial_positions = np.asarray(positions[spec.source_index], dtype=np.float64)
        trial_angles = np.asarray(orientations[spec.source_index], dtype=np.float64)
        if trial_positions.ndim != 3 or trial_positions.shape[1:] != (spec.group_size, 2):
            raise ValueError(f"{spec.trial_id}: positions do not match the published group size")
        if trial_angles.shape != trial_positions.shape[:2]:
            raise ValueError(f"{spec.trial_id}: heading angles do not match positions")
        frame_count = len(trial_positions)
        states = _text_states(group_states[spec.source_index], frame_count=frame_count)
        milling = compute_signed_milling(
            trial_positions,
            trial_angles,
            minimum_finite_fraction=minimum_finite_fraction,
        )
        frames = np.arange(frame_count, dtype=np.int64)
        run_ids = build_order_run_ids(frames, milling.frame_valid, minimum_run_length=2)
        trials.append(
            GoldenShinerTrialArrays(
                group_size=spec.group_size,
                trial=spec.trial,
                frame_rate_hz=float(frame_rate_hz),
                frames=frames,
                positions=trial_positions,
                heading_angles=trial_angles,
                group_states=states,
                milling=milling.values,
                finite_fraction=milling.finite_fraction,
                frame_valid=milling.frame_valid,
                order_run_id=run_ids,
                source_sha256=str(source_sha256),
                source_path=Path(source_path),
            )
        )
    return tuple(trials)


def _published_trial_spec(group_size: int, trial: str) -> GoldenShinerTrialSpec:
    matches = tuple(
        spec
        for spec in GOLDEN_SHINER_TRIAL_SPECS
        if spec.group_size == int(group_size) and spec.trial == str(trial)
    )
    if len(matches) != 1:
        raise ValueError(f"unsupported published golden-shiner trial: N{group_size}{trial}")
    return matches[0]


def _raw_hdf5_filenames(spec: GoldenShinerTrialSpec) -> tuple[str, ...]:
    if spec.group_size == 151:
        return ("fov_1-18000_2_fov.h5",)
    return tuple(f"{spec.trial}_{index:03d}_fov.h5" for index in range(7))


def _validated_raw_hdf5_paths(
    source_directory: str | Path,
    spec: GoldenShinerTrialSpec,
) -> tuple[Path, ...]:
    directory = Path(source_directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"raw HDF5 trial directory does not exist: {directory}")

    expected_names = _raw_hdf5_filenames(spec)
    hdf5_entries = tuple(
        entry for entry in directory.iterdir() if entry.suffix.casefold() in {".h5", ".hdf5"}
    )
    folded_names: dict[str, Path] = {}
    for entry in hdf5_entries:
        folded = entry.name.casefold()
        if folded in folded_names:
            raise ValueError(
                "raw HDF5 trial has duplicate filenames when compared case-insensitively: "
                f"{folded_names[folded].name}, {entry.name}"
            )
        folded_names[folded] = entry

    actual_names = {entry.name for entry in hdf5_entries}
    missing = sorted(set(expected_names) - actual_names)
    unexpected = sorted(actual_names - set(expected_names))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(
            "raw HDF5 trial file set is not the published layout: " + "; ".join(details)
        )

    ordered: list[Path] = []
    for name in expected_names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"raw HDF5 source must be a regular non-symlink file: {path}")
        ordered.append(path)
    return tuple(ordered)


def _raw_hdf5_source_summary(file_digests: Sequence[tuple[str, str]]) -> str:
    """Hash a canonical filename/digest manifest for one raw trial."""

    manifest = hashlib.sha256()
    for filename, digest in file_digests:
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"invalid source digest for {filename}")
        manifest.update(filename.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(digest.lower().encode("ascii"))
        manifest.update(b"\n")
    return manifest.hexdigest()


def _load_raw_hdf5_arrays(
    paths: Sequence[Path],
    *,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("raw golden-shiner HDF5 loading requires h5py>=3.11,<4") from exc

    required = ("x", "y", "heading_x", "heading_y")
    position_parts: list[np.ndarray] = []
    angle_parts: list[np.ndarray] = []
    file_digests: list[tuple[str, str]] = []
    for path in paths:
        before = path.stat()
        digest = sha256_file(path)
        with h5py.File(path, "r") as handle:
            fields_link = handle.get("fields", getlink=True)
            if not isinstance(fields_link, h5py.HardLink):
                raise TypeError(f"{path.name}: /fields must be an internal HDF5 group")
            fields = handle["/fields"]
            if not isinstance(fields, h5py.Group):
                raise TypeError(f"{path.name}: /fields is not an HDF5 group")

            arrays: dict[str, np.ndarray] = {}
            shape: tuple[int, int] | None = None
            for name in required:
                link = fields.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise TypeError(f"{path.name}: /fields/{name} must be an internal dataset")
                dataset = fields[name]
                if not isinstance(dataset, h5py.Dataset):
                    raise TypeError(f"{path.name}: /fields/{name} is not an HDF5 dataset")
                if dataset.ndim != 2:
                    raise ValueError(f"{path.name}: /fields/{name} must be two-dimensional")
                if dataset.dtype.kind not in "fiu":
                    raise ValueError(f"{path.name}: /fields/{name} must have a real numeric dtype")
                if shape is None:
                    shape = tuple(dataset.shape)
                elif tuple(dataset.shape) != shape:
                    raise ValueError(
                        f"{path.name}: required /fields datasets have inconsistent shapes"
                    )
                arrays[name] = np.asarray(dataset[()], dtype=np.float64)

        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"raw HDF5 source changed while it was being read: {path}")
        if shape is None or shape[0] != group_size or shape[1] < 1:
            raise ValueError(
                f"{path.name}: expected ({group_size}, frames) raw field shape, got {shape}"
            )

        positions = np.stack((arrays["x"].T, arrays["y"].T), axis=-1)
        heading_x = arrays["heading_x"].T
        heading_y = arrays["heading_y"].T
        heading_norm = np.hypot(heading_x, heading_y)
        heading_valid = np.isfinite(heading_x) & np.isfinite(heading_y) & (heading_norm > 1.0e-12)
        heading_angles = np.full(heading_x.shape, np.nan, dtype=np.float64)
        heading_angles[heading_valid] = np.arctan2(
            heading_y[heading_valid], heading_x[heading_valid]
        )
        position_parts.append(positions)
        angle_parts.append(heading_angles)
        file_digests.append((path.name, digest))

    return (
        np.ascontiguousarray(np.concatenate(position_parts, axis=0)),
        np.ascontiguousarray(np.concatenate(angle_parts, axis=0)),
        _raw_hdf5_source_summary(file_digests),
    )


def load_golden_shiner_raw_hdf5_trial(
    source_directory: str | Path,
    *,
    group_size: int,
    trial: str,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    minimum_finite_fraction: float = 0.8,
) -> GoldenShinerTrialArrays:
    """Read and validate one published raw HDF5 trial without modifying it.

    The 10-, 30-, and 70-fish trials must contain exactly the seven numbered
    ``*_fov.h5`` files.  The 151-fish trial must contain only the single joined
    file.  Fish remain in their published HDF5 row order while frames are
    concatenated in the published file-number order.
    """

    spec = _published_trial_spec(group_size, trial)
    paths = _validated_raw_hdf5_paths(source_directory, spec)
    positions, heading_angles, source_summary = _load_raw_hdf5_arrays(
        paths,
        group_size=spec.group_size,
    )
    milling = compute_signed_milling(
        positions,
        heading_angles,
        minimum_finite_fraction=minimum_finite_fraction,
    )
    frames = np.arange(len(positions), dtype=np.int64)
    run_ids = build_order_run_ids(frames, milling.frame_valid, minimum_run_length=2)
    states = np.full(len(positions), RAW_HDF5_GROUP_STATE, dtype=f"<U{len(RAW_HDF5_GROUP_STATE)}")
    return GoldenShinerTrialArrays(
        group_size=spec.group_size,
        trial=spec.trial,
        frame_rate_hz=float(frame_rate_hz),
        frames=frames,
        positions=positions,
        heading_angles=heading_angles,
        group_states=states,
        milling=milling.values,
        finite_fraction=milling.finite_fraction,
        frame_valid=milling.frame_valid,
        order_run_id=run_ids,
        source_sha256=source_summary,
        source_path=Path(source_directory),
    )


def convert_golden_shiner_raw_hdf5_trial_to_npz(
    source_directory: str | Path,
    output_path: str | Path,
    *,
    group_size: int,
    trial: str,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    minimum_finite_fraction: float = 0.8,
) -> Path:
    """Validate one raw trial and atomically create its safe NPZ file."""

    converted = load_golden_shiner_raw_hdf5_trial(
        source_directory,
        group_size=group_size,
        trial=trial,
        frame_rate_hz=frame_rate_hz,
        minimum_finite_fraction=minimum_finite_fraction,
    )
    destination = Path(output_path).resolve()
    if destination.suffix.casefold() != ".npz":
        raise ValueError("raw HDF5 conversion output must use the .npz suffix")
    if destination.exists():
        raise FileExistsError(f"conversion destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    staged_output = staging / destination.name
    try:
        _write_trial_npz(staged_output, converted)
        if destination.exists():
            raise FileExistsError(f"conversion destination already exists: {destination}")
        os.replace(staged_output, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def _write_trial_npz(path: Path, trial: GoldenShinerTrialArrays) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(1, dtype=np.int16),
            dataset_doi=np.asarray(DATASET_DOI),
            source_sha256=np.asarray(trial.source_sha256),
            group_size=np.asarray(trial.group_size, dtype=np.int32),
            trial=np.asarray(trial.trial),
            frame_rate_hz=np.asarray(trial.frame_rate_hz, dtype=np.float64),
            frames=trial.frames,
            positions=trial.positions,
            heading_angles=trial.heading_angles,
            group_states=trial.group_states,
            milling=trial.milling,
            finite_fraction=trial.finite_fraction,
            frame_valid=trial.frame_valid,
            order_run_id=trial.order_run_id,
        )


def convert_data_grid_pickle_to_npz(
    source_path: str | Path,
    output_directory: str | Path,
    *,
    expected_source_sha256: str | None = None,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    minimum_finite_fraction: float = 0.8,
    max_decoded_bytes: int = 16 * _GIBIBYTE,
    max_array_bytes: int = 16 * _GIBIBYTE,
) -> tuple[Path, ...]:
    """Convert the restricted published pickle into ten atomic safe NPZ files."""

    source = Path(source_path).resolve()
    source_digest = (
        verify_sha256(source, expected_source_sha256)
        if expected_source_sha256 is not None
        else sha256_file(source)
    )
    payload = restricted_numpy_load(
        source,
        expected_sha256=source_digest,
        max_decoded_bytes=max_decoded_bytes,
        max_array_bytes=max_array_bytes,
    )
    trials = parse_data_grid_payload(
        payload,
        source_sha256=source_digest,
        source_path=source,
        frame_rate_hz=frame_rate_hz,
        minimum_finite_fraction=minimum_finite_fraction,
    )

    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"conversion destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    names: list[str] = []
    try:
        for spec, trial in zip(GOLDEN_SHINER_TRIAL_SPECS, trials, strict=True):
            _write_trial_npz(staging / spec.output_filename, trial)
            names.append(spec.output_filename)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return tuple(destination / name for name in names)


_NPZ_KEYS = {
    "schema_version",
    "dataset_doi",
    "source_sha256",
    "group_size",
    "trial",
    "frame_rate_hz",
    "frames",
    "positions",
    "heading_angles",
    "group_states",
    "milling",
    "finite_fraction",
    "frame_valid",
    "order_run_id",
}


def _validated_npz(
    path: Path, *, expected_sha256: str | None, max_bytes: int
) -> np.lib.npyio.NpzFile:
    if expected_sha256 is not None:
        verify_sha256(path, expected_sha256)
    inspect_zip_archive(
        path,
        max_member_uncompressed_bytes=max_bytes,
        max_total_uncompressed_bytes=max_bytes,
        max_compression_ratio=100_000.0,
    )
    archive = np.load(path, allow_pickle=False)
    if set(archive.files) != _NPZ_KEYS:
        archive.close()
        raise ValueError("converted golden-shiner NPZ has an unexpected field set")
    return archive


def load_golden_shiner_npz(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    max_uncompressed_bytes: int = 8 * _GIBIBYTE,
) -> GoldenShinerTrialArrays:
    """Load one converted trial with ``allow_pickle=False`` and shape checks."""

    source = Path(path).resolve()
    with _validated_npz(
        source,
        expected_sha256=expected_sha256,
        max_bytes=int(max_uncompressed_bytes),
    ) as archive:
        schema_version = int(np.asarray(archive["schema_version"]).item())
        doi = str(np.asarray(archive["dataset_doi"]).item())
        if schema_version != 1 or doi != DATASET_DOI:
            raise ValueError("unsupported converted golden-shiner schema or dataset DOI")
        return GoldenShinerTrialArrays(
            group_size=int(np.asarray(archive["group_size"]).item()),
            trial=str(np.asarray(archive["trial"]).item()),
            frame_rate_hz=float(np.asarray(archive["frame_rate_hz"]).item()),
            frames=np.asarray(archive["frames"], dtype=np.int64),
            positions=np.asarray(archive["positions"], dtype=np.float64),
            heading_angles=np.asarray(archive["heading_angles"], dtype=np.float64),
            group_states=np.asarray(archive["group_states"]),
            milling=np.asarray(archive["milling"], dtype=np.float64),
            finite_fraction=np.asarray(archive["finite_fraction"], dtype=np.float64),
            frame_valid=np.asarray(archive["frame_valid"], dtype=bool),
            order_run_id=np.asarray(archive["order_run_id"], dtype=np.int64),
            source_sha256=str(np.asarray(archive["source_sha256"]).item()),
            source_path=source,
        )


def primary_confirmation_specs() -> tuple[GoldenShinerTrialSpec, ...]:
    """Return the nine 10/30/70-fish trials used for formal confirmation."""

    return tuple(spec for spec in GOLDEN_SHINER_TRIAL_SPECS if spec.primary_confirmation)


def descriptive_specs() -> tuple[GoldenShinerTrialSpec, ...]:
    """Return the single 151-fish trial reserved for descriptive analysis."""

    return tuple(spec for spec in GOLDEN_SHINER_TRIAL_SPECS if not spec.primary_confirmation)


__all__ = [
    "DATASET_DOI",
    "DEFAULT_FRAME_RATE_HZ",
    "EXPECTED_PROCESSED_ARCHIVE_BYTES",
    "EXPECTED_PROCESSED_ARCHIVE_SHA256",
    "GOLDEN_SHINER_TRIAL_SPECS",
    "RAW_HDF5_GROUP_STATE",
    "GoldenShinerTrialArrays",
    "GoldenShinerTrialSpec",
    "MillingSeries",
    "UnsafePickleError",
    "ZipMemberRecord",
    "ZipSafetyError",
    "build_order_run_ids",
    "compute_signed_milling",
    "convert_data_grid_pickle_to_npz",
    "convert_golden_shiner_raw_hdf5_trial_to_npz",
    "descriptive_specs",
    "find_unique_zip_member",
    "inspect_zip_archive",
    "load_golden_shiner_npz",
    "load_golden_shiner_raw_hdf5_trial",
    "parse_data_grid_payload",
    "primary_confirmation_specs",
    "restricted_numpy_load",
    "safe_extract_zip",
    "sha256_file",
    "verify_processed_archive",
    "verify_sha256",
]
