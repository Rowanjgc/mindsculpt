"""Canonical dtype names and SafeTensors/PyTorch/NumPy conversions.

Canonical names are lower-case and framework-neutral (``float32``,
``bfloat16``, ``int64``, ...).  Bit-level formats such as BF16 and float8 are
represented explicitly; they are never silently cast.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wsp.core.errors import UnsupportedError

# (canonical name, numpy representation used for raw storage)
SAFETENSORS_DTYPES: dict[str, tuple[str, np.dtype[Any]]] = {
    "F64": ("float64", np.dtype("<f8")),
    "F32": ("float32", np.dtype("<f4")),
    "F16": ("float16", np.dtype("<f2")),
    "BF16": ("bfloat16", np.dtype("<u2")),
    "I64": ("int64", np.dtype("<i8")),
    "I32": ("int32", np.dtype("<i4")),
    "I16": ("int16", np.dtype("<i2")),
    "I8": ("int8", np.dtype("i1")),
    "U64": ("uint64", np.dtype("<u8")),
    "U32": ("uint32", np.dtype("<u4")),
    "U16": ("uint16", np.dtype("<u2")),
    "U8": ("uint8", np.dtype("u1")),
    "BOOL": ("bool", np.dtype("|b1")),
    "F8_E4M3": ("float8_e4m3fn", np.dtype("|u1")),
    "F8_E5M2": ("float8_e5m2", np.dtype("|u1")),
}

CANONICAL_TO_SAFETENSORS: dict[str, str] = {
    canonical: code for code, (canonical, _) in SAFETENSORS_DTYPES.items()
}

CANONICAL_TO_NUMPY: dict[str, np.dtype[Any]] = {
    canonical: dtype for canonical, dtype in SAFETENSORS_DTYPES.values()
}

NUMERIC_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
INTEGER_DTYPES = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
FLOAT8_DTYPES = {"float8_e4m3fn", "float8_e5m2"}

_ALIASES = {
    "torch.float64": "float64",
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.int64": "int64",
    "torch.int32": "int32",
    "torch.int16": "int16",
    "torch.int8": "int8",
    "torch.uint64": "uint64",
    "torch.uint32": "uint32",
    "torch.uint16": "uint16",
    "torch.uint8": "uint8",
    "torch.bool": "bool",
    "torch.float8_e4m3fn": "float8_e4m3fn",
    "torch.float8_e5m2": "float8_e5m2",
    "fp64": "float64",
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "f64": "float64",
    "f32": "float32",
    "f16": "float16",
}


def normalize_dtype_name(dtype: str) -> str:
    name = str(dtype)
    if name in SAFETENSORS_DTYPES:
        return SAFETENSORS_DTYPES[name][0]
    lowered = name.lower()
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    if lowered in CANONICAL_TO_SAFETENSORS:
        return lowered
    # numpy spellings such as "<f4" or "float32"
    try:
        numpy_dtype = np.dtype(name)
    except TypeError as exc:
        raise UnsupportedError(f"unsupported dtype: {dtype!r}") from exc
    return storage_dtype_from_numpy(numpy_dtype)


def storage_dtype_from_numpy(dtype: np.dtype[Any] | str) -> str:
    numpy_dtype = np.dtype(dtype)
    for canonical, candidate in CANONICAL_TO_NUMPY.items():
        if canonical in {"bfloat16", "float8_e4m3fn", "float8_e5m2"}:
            continue
        if numpy_dtype == candidate:
            return canonical
    if numpy_dtype.kind == "f" and numpy_dtype.itemsize == 2:
        return "float16"
    if numpy_dtype.kind == "f" and numpy_dtype.itemsize == 4:
        return "float32"
    if numpy_dtype.kind == "f" and numpy_dtype.itemsize == 8:
        return "float64"
    if numpy_dtype.kind == "b":
        return "bool"
    raise UnsupportedError(f"cannot map numpy dtype {numpy_dtype!s} to a canonical dtype")


def numpy_dtype_for(storage_dtype: str) -> np.dtype[Any]:
    canonical = normalize_dtype_name(storage_dtype)
    if canonical == "bfloat16":
        return np.dtype("<u2")
    return CANONICAL_TO_NUMPY[canonical]


def bytes_per_element(storage_dtype: str) -> int:
    canonical = normalize_dtype_name(storage_dtype)
    if canonical == "bool":
        return 1
    return CANONICAL_TO_NUMPY[canonical].itemsize


def is_float_storage(storage_dtype: str) -> bool:
    return normalize_dtype_name(storage_dtype) in NUMERIC_FLOAT_DTYPES


def is_float8_storage(storage_dtype: str) -> bool:
    return normalize_dtype_name(storage_dtype) in FLOAT8_DTYPES


def is_integer_storage(storage_dtype: str) -> bool:
    return normalize_dtype_name(storage_dtype) in INTEGER_DTYPES


def is_bool_storage(storage_dtype: str) -> bool:
    return normalize_dtype_name(storage_dtype) == "bool"


def decode_safetensors_array(raw: np.ndarray[Any, Any], storage_dtype: str) -> np.ndarray[Any, Any]:
    """Decode a raw SafeTensors numpy view into a numerically usable array.

    BF16 is widened losslessly to float32.  Float8 is recognised and rejected
    for numerical analysis; raw export still preserves it.
    """

    canonical = normalize_dtype_name(storage_dtype)
    array = np.asarray(raw)
    if canonical == "bfloat16":
        if array.dtype == np.dtype("float32"):
            return np.ascontiguousarray(array)
        if array.dtype != np.dtype("uint16") and array.dtype != np.dtype("<u2"):
            raise UnsupportedError(
                f"BF16 tensor decoded as unexpected numpy dtype {array.dtype}"
            )
        widened = array.astype(np.uint32) << 16
        return np.ascontiguousarray(widened.view(np.float32))
    if canonical in FLOAT8_DTYPES:
        raise UnsupportedError(
            f"{canonical} numerical analysis is not supported in V1; "
            "raw checkpoint copy/export preserves the tensor unchanged"
        )
    if canonical == "bool":
        return np.ascontiguousarray(array.astype(np.bool_, copy=False))
    target = CANONICAL_TO_NUMPY[canonical]
    if array.dtype != target:
        array = array.astype(target, copy=False)
    return np.ascontiguousarray(array)


def encode_safetensors_array(array: np.ndarray[Any, Any], storage_dtype: str) -> bytes:
    """Encode an array to SafeTensors raw bytes in the requested storage dtype."""

    canonical = normalize_dtype_name(storage_dtype)
    values = np.asarray(array)
    if canonical == "bfloat16":
        float32 = values.astype(np.float32, copy=False)
        bits = float32.view(np.uint32) >> np.uint32(16)
        return np.ascontiguousarray(bits.astype("<u2")).tobytes(order="C")
    if canonical in FLOAT8_DTYPES:
        raise UnsupportedError(
            f"writing {canonical} tensors from decoded values is not supported in V1"
        )
    target = CANONICAL_TO_NUMPY[canonical]
    encoded = np.ascontiguousarray(values.astype(target, copy=False))
    if encoded.dtype.byteorder == ">":
        encoded = encoded.astype(encoded.dtype.newbyteorder("<"), copy=False)
    return encoded.tobytes(order="C")


def torch_dtype_from_name(storage_dtype: str) -> Any:
    import torch

    canonical = normalize_dtype_name(storage_dtype)
    mapping = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint64": torch.uint64,
        "uint32": torch.uint32,
        "uint16": torch.uint16,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
    }
    try:
        return mapping[canonical]
    except KeyError as exc:
        raise UnsupportedError(f"no torch dtype for {storage_dtype!r}") from exc


def numpy_to_torch(array: np.ndarray[Any, Any], storage_dtype: str | None = None) -> Any:
    """Convert a numpy array to a torch tensor without silent dtype changes."""

    import torch

    canonical = storage_dtype or storage_dtype_from_numpy(np.asarray(array).dtype)
    canonical = normalize_dtype_name(canonical)
    values = np.ascontiguousarray(array)
    if canonical == "bfloat16":
        if values.dtype != np.uint16:
            float32 = values.astype(np.float32, copy=False)
            bits = float32.view(np.uint32) >> np.uint32(16)
            values = bits.astype(np.uint16)
        return torch.from_numpy(values.copy()).view(torch.bfloat16)
    if canonical in FLOAT8_DTYPES:
        target = torch_dtype_from_name(canonical)
        return torch.from_numpy(values.copy()).view(target)
    return torch.from_numpy(values.copy()).to(torch_dtype_from_name(canonical))


def torch_to_numpy(tensor: Any, *, decode_bfloat16: bool = True) -> np.ndarray[Any, Any]:
    import torch

    value = tensor.detach().to("cpu")
    if value.dtype == torch.bfloat16:
        if decode_bfloat16:
            return value.float().numpy()
        return value.view(torch.uint16).numpy()
    if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return value.float().numpy()
    return value.numpy()


def torch_dtype_name(tensor: Any) -> str:
    import torch

    dtype = tensor if isinstance(tensor, torch.dtype) else tensor.dtype
    name = str(dtype)
    if name.startswith("torch."):
        name = name[len("torch.") :]
    return normalize_dtype_name(name)
