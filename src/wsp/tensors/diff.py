"""Numerical comparison between compatible tensors."""

from __future__ import annotations

from typing import Any

import numpy as np

from wsp.core.errors import CompatibilityError, UnsupportedError
from wsp.core.models import TensorDiffStats


def _float64(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    values = np.asarray(array)
    if values.dtype == object or np.iscomplexobj(values):
        raise UnsupportedError("only real numeric tensors can be compared in V1")
    return np.ascontiguousarray(values.astype(np.float64, copy=False))


def compare_arrays(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    sampled: bool = False,
) -> TensorDiffStats:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    left_shape = [int(d) for d in left_array.shape]
    right_shape = [int(d) for d in right_array.shape]
    if left_shape != right_shape:
        return TensorDiffStats(
            left_shape=left_shape,
            right_shape=right_shape,
            compatible=False,
            atol=atol,
            rtol=rtol,
            sampled=sampled,
        )
    if atol < 0 or rtol < 0:
        raise CompatibilityError("atol and rtol must be non-negative")

    a = _float64(left_array).ravel()
    b = _float64(right_array).ravel()
    count = int(a.size)
    if count == 0:
        return TensorDiffStats(
            left_shape=left_shape,
            right_shape=right_shape,
            compatible=True,
            count=0,
            allclose=True,
            atol=atol,
            rtol=rtol,
            sampled=sampled,
        )

    finite_pair = np.isfinite(a) & np.isfinite(b)
    non_finite_pairs = int(np.count_nonzero(~finite_pair))
    if np.any(finite_pair):
        delta = a[finite_pair] - b[finite_pair]
        max_abs = float(np.max(np.abs(delta)))
        mean_abs = float(np.mean(np.abs(delta)))
        rms = float(np.sqrt(np.mean(delta * delta)))
        denominator = float(np.linalg.norm(b[finite_pair]))
        relative = float(np.linalg.norm(delta) / denominator) if denominator > 0 else None
        norm_a = float(np.linalg.norm(a[finite_pair]))
        norm_b = denominator
        cosine = (
            float(np.dot(a[finite_pair], b[finite_pair]) / (norm_a * norm_b))
            if norm_a > 0 and norm_b > 0
            else None
        )
    else:
        max_abs = mean_abs = rms = relative = cosine = None

    allclose = bool(
        non_finite_pairs == 0
        and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
    )
    return TensorDiffStats(
        left_shape=left_shape,
        right_shape=right_shape,
        compatible=True,
        count=count,
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms=rms,
        relative_frobenius=relative,
        cosine=cosine,
        allclose=allclose,
        atol=atol,
        rtol=rtol,
        non_finite_pairs=non_finite_pairs,
        sampled=sampled,
    )
