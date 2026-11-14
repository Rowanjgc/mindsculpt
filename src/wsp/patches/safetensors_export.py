"""Streaming SafeTensors rewrite for small edits.

Untouched tensors are copied byte-for-byte in bounded chunks.  Only tensors
that a patch actually targets are materialized.  The output is written to a
temporary file and atomically linked/renamed into place; the source file is
opened read-only and is never modified.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy as np

from wsp.core.dtypes import bytes_per_element, encode_safetensors_array, normalize_dtype_name
from wsp.core.errors import CompatibilityError, InputError, ResourceLimitError, SafetyError
from wsp.core.json import atomic_write_stream

_CHUNK = 8 * 1024 * 1024
_MAX_HEADER = 100 * 1024 * 1024


def _read_header(path: Path) -> tuple[dict[str, Any], int, int]:
    size = path.stat().st_size
    if size < 8:
        raise InputError(f"not a SafeTensors file: {path}")
    with open(path, "rb") as handle:
        raw_length = handle.read(8)
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > _MAX_HEADER:
            raise InputError(
                "SafeTensors header is unreasonably large",
                details={"header_bytes": header_length, "limit": _MAX_HEADER},
            )
        if 8 + header_length > size:
            raise InputError("SafeTensors header extends past end of file")
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"malformed SafeTensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise InputError("SafeTensors header must be a JSON object")
    return header, 8 + header_length, size


def _validated_entries(
    header: dict[str, Any], data_size: int
) -> list[tuple[str, dict[str, Any], int, int]]:
    entries: list[tuple[str, dict[str, Any], int, int]] = []
    intervals: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise InputError(f"malformed SafeTensors entry for {name!r}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
            raise InputError(f"incomplete SafeTensors entry for {name!r}")
        if len(offsets) != 2 or not all(isinstance(value, int) for value in offsets):
            raise InputError(f"invalid data offsets for {name!r}")
        begin, end = offsets
        if begin < 0 or end < begin or end > data_size:
            raise InputError(f"data offsets out of range for {name!r}")
        expected = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                raise InputError(f"invalid shape for {name!r}")
            expected *= dimension
        expected *= bytes_per_element(dtype)
        if expected != end - begin:
            raise InputError(
                f"declared byte length does not match shape/dtype for {name!r}",
                details={"declared": end - begin, "expected": expected},
            )
        entries.append((name, entry, begin, end))
        intervals.append((begin, end, name))
    intervals.sort()
    for index in range(1, len(intervals)):
        if intervals[index][0] < intervals[index - 1][1]:
            raise InputError(
                "SafeTensors tensor data ranges overlap",
                details={"tensor": intervals[index][2]},
            )
    return entries


def export_with_replacements(
    source_path: str | Path,
    output_path: str | Path,
    replacements: dict[str, np.ndarray[Any, Any]] | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise SafetyError(
            "refusing to write a patched checkpoint over its source",
            details={"source": str(source)},
        )
    if source.suffix.lower() != ".safetensors":
        raise InputError("source must be a .safetensors file")
    replacements = replacements or {}
    header, data_start, file_size = _read_header(source)
    data_size = file_size - data_start
    entries = _validated_entries(header, data_size)
    by_name = {name: (entry, begin, end) for name, entry, begin, end in entries}
    unknown = set(replacements) - set(by_name)
    if unknown:
        raise CompatibilityError(
            "replacement targets tensor(s) absent from the source",
            details={"tensors": sorted(unknown)},
        )

    encoded_replacements: dict[str, bytes] = {}
    for name, values in replacements.items():
        entry, begin, end = by_name[name]
        storage_dtype = normalize_dtype_name(entry["dtype"])
        shape = [int(d) for d in entry["shape"]]
        array = np.asarray(values)
        if list(array.shape) != shape:
            raise CompatibilityError(
                f"replacement shape for {name!r} does not match the checkpoint",
                details={"expected": shape, "actual": list(array.shape)},
            )
        payload = encode_safetensors_array(array, storage_dtype)
        if len(payload) != end - begin:
            raise InputError(
                f"encoded replacement size mismatch for {name!r}",
                details={"expected": end - begin, "actual": len(payload)},
            )
        encoded_replacements[name] = payload

    new_header: dict[str, Any] = {}
    if "__metadata__" in header:
        metadata = header["__metadata__"]
        if isinstance(metadata, dict):
            # SafeTensors writers do not guarantee metadata key order.  Sorting
            # makes our exports byte-stable for identical tensor content.
            metadata = {key: metadata[key] for key in sorted(metadata)}
        new_header["__metadata__"] = metadata
    offset = 0
    offsets_by_name: dict[str, tuple[int, int]] = {}
    for name, entry, _begin, end in entries:
        length = end - _begin
        new_header[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + length],
        }
        offsets_by_name[name] = (offset, offset + length)
        offset += length

    raw_header = json.dumps(
        new_header, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    padding = (-len(raw_header)) % 8
    raw_header += b" " * padding
    total_size = 8 + len(raw_header) + offset
    if total_size > source.stat().st_size * 2 + 1024 * 1024:
        raise ResourceLimitError(
            "exported checkpoint would be unexpectedly larger than the source",
            details={"source_bytes": source.stat().st_size, "output_bytes": total_size},
        )

    def writer(output: BinaryIO) -> None:
        output.write(struct.pack("<Q", len(raw_header)))
        output.write(raw_header)
        with open(source, "rb") as input_handle:
            for name, entry, begin, end in entries:
                if name in encoded_replacements:
                    output.write(encoded_replacements[name])
                    continue
                remaining = end - begin
                input_handle.seek(data_start + begin)
                while remaining > 0:
                    chunk = input_handle.read(min(_CHUNK, remaining))
                    if not chunk:
                        raise InputError(
                            "unexpected end of source while copying tensor",
                            details={"tensor": name},
                        )
                    output.write(chunk)
                    remaining -= len(chunk)
        if output.tell() != total_size:
            raise InputError(
                "internal error: exported size does not match header",
                details={"expected": total_size, "actual": output.tell()},
            )

    try:
        atomic_write_stream(destination, writer, overwrite=overwrite)
    except FileExistsError as exc:
        raise SafetyError(
            "refusing to overwrite an existing output checkpoint; choose a new path or set overwrite=true",
            details={"output_path": str(destination)},
        ) from exc
    return destination


def copy_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    return export_with_replacements(
        source_path, output_path, {}, overwrite=overwrite
    )
