"""Exact summary statistics.

Population standard deviation (``ddof=0``) is used.  Statistics are computed
over finite values only; non-finite counts are always reported.  If no finite
values exist, scalar fields are ``None`` rather than NaN.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wsp.core.dtypes import is_float8_storage, storage_dtype_from_numpy
from wsp.core.errors import InputError, UnsupportedError
from wsp.core.models import StatsSummary
from wsp.core.selection import AxisSlice, ResolvedSelection, Selection
from wsp.inspection.base import CheckpointSource


def _numeric_float64(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    values = np.asarray(array)
    if values.dtype == object:
        raise UnsupportedError("object-dtype tensors are not numerically analyzable")
    if np.iscomplexobj(values):
        raise UnsupportedError("complex tensors are not supported in V1")
    return np.ascontiguousarray(values.astype(np.float64, copy=False))


def basic_stats(
    array: np.ndarray[Any, Any],
    *,
    storage_dtype: str | None = None,
    sampled: bool = False,
) -> StatsSummary:
    values = np.asarray(array)
    dtype = storage_dtype or storage_dtype_from_numpy(values.dtype)
    flat = _numeric_float64(values).ravel()
    count = int(flat.size)

    if np.issubdtype(values.dtype, np.floating):
        nan_count = int(np.isnan(flat).sum())
        posinf_count = int(np.isposinf(flat).sum())
        neginf_count = int(np.isneginf(flat).sum())
        finite_mask = np.isfinite(flat)
        finite = flat[finite_mask]
    else:
        nan_count = posinf_count = neginf_count = 0
        finite = flat

    finite_count = int(finite.size)
    zero_count = int(np.count_nonzero(finite == 0.0))
    if finite_count:
        mean = float(finite.mean())
        std = float(finite.std(ddof=0))
        minimum = float(finite.min())
        maximum = float(finite.max())
        abs_max = float(np.abs(finite).max())
    else:
        mean = std = minimum = maximum = abs_max = None

    return StatsSummary(
        dtype=dtype,
        shape=[int(d) for d in values.shape],
        count=count,
        finite_count=finite_count,
        nan_count=nan_count,
        posinf_count=posinf_count,
        neginf_count=neginf_count,
        zero_count=zero_count,
        mean=mean,
        std=std,
        min=minimum,
        max=maximum,
        abs_max=abs_max,
        finite_only=True,
        sampled=sampled,
    )


def sampled_selection(
    resolved: ResolvedSelection, max_elements: int
) -> Selection | None:
    """Build a deterministic stride sample of a resolved rectangular selection."""

    if resolved.numel <= max_elements:
        return None
    if resolved.numel == 0:
        return None
    axis_index = max(range(len(resolved.dims)), key=lambda i: resolved.dims[i].length)
    axis = resolved.dims[axis_index]
    stride = 1
    while True:
        stride = max(stride + 1, (resolved.numel + max_elements - 1) // max_elements)
        new_step = axis.step * stride
        lengths = [dim.length for dim in resolved.dims]
        lengths[axis_index] = 0 if axis.length == 0 else (axis.length + new_step - 1) // new_step
        total = 1
        for length in lengths:
            total *= length
        if total <= max_elements:
            break
        if stride > resolved.numel:
            break
    pieces = []
    for index, dim in enumerate(resolved.dims):
        step = dim.step * stride if index == axis_index else dim.step
        pieces.append(
            AxisSlice(axis=dim.axis, start=dim.start, stop=dim.stop, step=step)
        )
    return Selection(slices=pieces)


def stats_for_source(
    source: CheckpointSource,
    name: str,
    *,
    selection: Selection | None,
    max_elements: int,
    sample_elements: int,
) -> StatsSummary:
    info = source.tensor_info(name)
    if is_float8_storage(info.dtype):
        raise UnsupportedError(
            f"statistics for {info.dtype} tensors are not supported in V1",
            details={"tensor": name, "dtype": info.dtype},
        )
    resolved = (selection or Selection()).resolve(info.shape)
    if resolved.numel <= max_elements:
        values = source.read_tensor(name, selection, max_elements=max_elements)
        return basic_stats(values, storage_dtype=info.dtype)

    sampled = sampled_selection(resolved, sample_elements)
    if sampled is None:
        raise InputError(
            f"tensor {name!r} is too large to summarize",
            details={"numel": resolved.numel, "limit": max_elements},
        )
    values = source.read_tensor(name, sampled, max_elements=sample_elements)
    return basic_stats(values, storage_dtype=info.dtype, sampled=True)
