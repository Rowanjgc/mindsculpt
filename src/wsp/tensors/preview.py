"""Bounded tensor previews."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from wsp.core.dtypes import storage_dtype_from_numpy
from wsp.core.models import Preview


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def preview_array(
    array: np.ndarray[Any, Any],
    *,
    max_elements: int,
    storage_dtype: str | None = None,
) -> Preview:
    values = np.asarray(array)
    flat = values.reshape(-1)
    numel = int(flat.size)
    if numel <= max_elements:
        selected = flat
        stride: int | None = None
        sampled = False
        truncated = False
    else:
        stride = max(1, (numel + max_elements - 1) // max_elements)
        selected = flat[::stride][:max_elements]
        sampled = True
        truncated = False
    return Preview(
        dtype=storage_dtype or storage_dtype_from_numpy(values.dtype),
        shape=[int(d) for d in values.shape],
        values=[_json_scalar(value) for value in selected.tolist()],
        numel=numel,
        returned=int(selected.size),
        truncated=truncated,
        sampled=sampled,
        stride=stride,
    )
