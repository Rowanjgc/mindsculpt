"""Content hashing and stable tensor fingerprints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from wsp.core.json import canonical_bytes

TENSOR_FINGERPRINT_SCHEMA = "wsp.tensorfp/1"
STRUCTURE_FINGERPRINT_SCHEMA = "wsp.structurefp/1"
_CHUNK = 8 * 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = _CHUNK) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonicalize_nonfinite(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating):
        # Canonical quiet NaN avoids depending on payload bits produced by
        # different hardware or libraries.
        return np.where(np.isnan(array), np.dtype(array.dtype).type(np.nan), array)
    return array


def tensor_fingerprint(
    array: np.ndarray | Any,
    *,
    storage_dtype: str | None = None,
    shape: list[int] | None = None,
) -> str:
    """Exact fingerprint of decoded tensor values.

    The fingerprint is intentionally exact rather than tolerance-based: it is
    used to guarantee that a patch is applied to the same values it was
    validated against.  Numerical equivalence after application is measured
    separately with explicit tolerances and diff statistics.
    """

    values = np.asarray(array)
    if shape is None:
        shape = list(values.shape)
    dtype_name = storage_dtype or str(values.dtype)
    header = canonical_bytes(
        {
            "schema": TENSOR_FINGERPRINT_SCHEMA,
            "storage_dtype": dtype_name,
            "shape": list(shape),
        }
    )

    if np.issubdtype(values.dtype, np.floating):
        values = _canonicalize_nonfinite(values)
    # Normalize byte order so a big-endian copy hashes the same as a
    # little-endian copy with identical values.
    normalized = np.ascontiguousarray(values)
    if normalized.dtype.byteorder == ">":
        normalized = normalized.astype(normalized.dtype.newbyteorder("<"), copy=False)
    return sha256_bytes(header + normalized.tobytes(order="C"))


def structure_fingerprint(payload: dict[str, Any]) -> str:
    """Fingerprint structural metadata (names, dtypes, shapes, metadata)."""

    return sha256_bytes(
        canonical_bytes({"schema": STRUCTURE_FINGERPRINT_SCHEMA, **payload})
    )
