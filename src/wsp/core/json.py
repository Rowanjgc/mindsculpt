"""JSON conversion and atomic file helpers.

Machine-readable output must be strict JSON: no NaN/Infinity tokens, no numpy
scalars, no ``Path`` objects.  Non-finite floats are represented as ``null``
in previews and are always accompanied by explicit non-finite counts in
statistics so that information is not silently lost.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    """Recursively convert common Python/scientific objects to JSON types."""

    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="python"))
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    # Last resort: let pydantic/Python decide, but keep the output strict.
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(mode="python"))
    return str(value)


def dumps(value: Any, *, indent: int = 2, sort_keys: bool = True) -> str:
    return json.dumps(
        to_jsonable(value),
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def dump_bytes(value: Any, *, sort_keys: bool = True) -> bytes:
    return json.dumps(
        to_jsonable(value),
        sort_keys=sort_keys,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON bytes suitable for hashing."""

    return dump_bytes(value, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    overwrite: bool = False,
) -> Path:
    """Write ``data`` atomically.

    With ``overwrite=False`` the destination is created with an exclusive hard
    link, so a concurrent writer cannot clobber it between check and replace.
    The temporary file is always in the destination directory to keep the
    rename on one filesystem.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                raise FileExistsError(f"refusing to overwrite existing file: {destination}")
            finally:
                temporary_path.unlink(missing_ok=True)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    overwrite: bool = False,
    indent: int = 2,
) -> Path:
    payload = (dumps(value, indent=indent) + "\n").encode("utf-8")
    return atomic_write_bytes(path, payload, overwrite=overwrite)


def atomic_write_stream(
    path: str | Path,
    write: Any,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a binary file using a callback.

    ``write`` receives a binary file object.  If it raises, the temporary file
    is removed and the destination is untouched.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                raise FileExistsError(f"refusing to overwrite existing file: {destination}")
            finally:
                temporary_path.unlink(missing_ok=True)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination
