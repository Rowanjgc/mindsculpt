"""Bounded numerical analysis with explicit sampling labels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wsp.core.budget import Budget, enforce_numel
from wsp.core.dtypes import storage_dtype_from_numpy
from wsp.core.errors import InputError, ResourceLimitError, UnsupportedError
from wsp.core.models import StatsSummary
from wsp.tensors.preview import _json_scalar
from wsp.tensors.stats import basic_stats

KNOWN_OPERATIONS = {"stats", "quantiles", "histogram", "norms", "svd", "rank"}
DEFAULT_QUANTILES = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]


class TensorAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stats: StatsSummary
    operations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    requested: list[str] = Field(default_factory=list)
    sampled_operations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _flatten_finite(array: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], int]:
    values = np.asarray(array)
    if values.dtype == object or np.iscomplexobj(values):
        raise UnsupportedError("only real numeric tensors can be analyzed in V1")
    flat = values.astype(np.float64, copy=False).ravel()
    if np.issubdtype(values.dtype, np.floating):
        finite = flat[np.isfinite(flat)]
        non_finite = int(flat.size - finite.size)
    else:
        finite = flat
        non_finite = 0
    return finite, non_finite


def _sample_flat(
    flat: np.ndarray[Any, Any], max_elements: int
) -> tuple[np.ndarray[Any, Any], bool, int | None]:
    if flat.size <= max_elements:
        return flat, False, None
    stride = max(1, (int(flat.size) + max_elements - 1) // max_elements)
    return flat[::stride][:max_elements], True, stride


def _quantiles(
    flat: np.ndarray[Any, Any], qs: list[float], max_elements: int
) -> dict[str, Any]:
    if not qs:
        raise InputError("at least one quantile must be requested")
    for q in qs:
        if not math.isfinite(q) or q < 0 or q > 1:
            raise InputError(f"quantile {q!r} is outside [0, 1]")
    sampled_values, sampled, stride = _sample_flat(flat, max_elements)
    if sampled_values.size == 0:
        values = [None for _ in qs]
    else:
        computed = np.quantile(sampled_values, qs, method="linear")
        values = [_json_scalar(value) for value in computed]
    return {
        "quantiles": qs,
        "values": values,
        "method": "linear",
        "finite_only": True,
        "sampled": sampled,
        "stride": stride,
        "sample_size": int(sampled_values.size),
    }


def _histogram(
    flat: np.ndarray[Any, Any], bins: int, max_elements: int, max_bins: int
) -> dict[str, Any]:
    if bins < 1 or bins > max_bins:
        raise InputError(f"bins must be between 1 and {max_bins}")
    sampled_values, sampled, stride = _sample_flat(flat, max_elements)
    if sampled_values.size == 0:
        return {
            "bins": [],
            "counts": [],
            "range": None,
            "finite_only": True,
            "sampled": sampled,
            "stride": stride,
            "sample_size": 0,
        }
    minimum = float(sampled_values.min())
    maximum = float(sampled_values.max())
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5
    counts, edges = np.histogram(sampled_values, bins=bins, range=(minimum, maximum))
    return {
        "bins": [_json_scalar(edge) for edge in edges],
        "counts": [int(count) for count in counts],
        "range": [minimum, maximum],
        "finite_only": True,
        "sampled": sampled,
        "stride": stride,
        "sample_size": int(sampled_values.size),
    }


def _norms(
    array: np.ndarray[Any, Any], budget: Budget
) -> dict[str, Any]:
    values = np.asarray(array)
    finite, non_finite = _flatten_finite(values)
    size = int(values.size)
    enforce_numel(size, budget.max_analysis_elements, what="matrix norm input")
    result: dict[str, Any] = {
        "shape": [int(d) for d in values.shape],
        "non_finite_count": non_finite,
        "finite_only": non_finite == 0,
    }
    if non_finite:
        result["vector"] = {}
        result["matrix"] = {}
        result["notes"] = ["norms are undefined for non-finite tensors"]
        return result
    flat = finite
    result["vector"] = {
        "l1": float(np.linalg.norm(flat, ord=1)),
        "l2": float(np.linalg.norm(flat, ord=2)),
        "linf": float(np.linalg.norm(flat, ord=np.inf)),
        "frobenius": float(np.linalg.norm(flat, ord=2)),
    }
    if values.ndim == 2:
        enforce_numel(size, budget.max_svd_elements, what="matrix norm input")
        singular = np.linalg.svd(values.astype(np.float64), compute_uv=False)
        result["matrix"] = {
            "frobenius": float(np.linalg.norm(values, ord="fro")),
            "spectral": float(singular[0]) if singular.size else 0.0,
            "nuclear": float(np.sum(singular)),
            "max": float(np.max(np.abs(values))) if size else 0.0,
        }
    else:
        result["matrix"] = {}
    return result


def _svd(
    array: np.ndarray[Any, Any], top_k: int, budget: Budget
) -> dict[str, Any]:
    values = np.asarray(array)
    if values.ndim != 2:
        raise InputError("svd requires a 2-D tensor")
    enforce_numel(int(values.size), budget.max_svd_elements, what="svd input")
    if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
        raise InputError("svd requires all values to be finite")
    matrix = values.astype(np.float64, copy=False)
    singular = np.linalg.svd(matrix, compute_uv=False)
    k = max(1, min(top_k, int(singular.size)))
    if singular.size:
        tolerance = float(np.max(matrix.shape)) * np.finfo(np.float64).eps * float(singular[0])
    else:
        tolerance = 0.0
    rank = int(np.count_nonzero(singular > tolerance))
    total = float(np.sum(singular))
    return {
        "shape": [int(d) for d in values.shape],
        "top_k": k,
        "singular_values": [float(value) for value in singular[:k]],
        "rank": rank,
        "rank_tolerance": tolerance,
        "nuclear_norm": total,
        "spectral_norm": float(singular[0]) if singular.size else 0.0,
        "condition_number": (
            float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else None
        ),
        "sampled": False,
    }


def _rank(array: np.ndarray[Any, Any], budget: Budget) -> dict[str, Any]:
    values = np.asarray(array)
    if values.ndim != 2:
        raise InputError("rank requires a 2-D tensor")
    enforce_numel(int(values.size), budget.max_svd_elements, what="rank input")
    if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
        raise InputError("rank requires all values to be finite")
    singular = np.linalg.svd(values.astype(np.float64), compute_uv=False)
    if singular.size:
        tolerance = float(np.max(values.shape)) * np.finfo(np.float64).eps * float(singular[0])
    else:
        tolerance = 0.0
    return {
        "shape": [int(d) for d in values.shape],
        "rank": int(np.count_nonzero(singular > tolerance)),
        "tolerance": tolerance,
        "sampled": False,
    }


def analyze_array(
    array: np.ndarray[Any, Any],
    *,
    storage_dtype: str | None = None,
    operations: list[str] | None = None,
    budget: Budget,
    quantiles: list[float] | None = None,
    bins: int = 32,
    top_k: int = 10,
) -> TensorAnalysis:
    values = np.asarray(array)
    requested = [operation.lower() for operation in (operations or ["stats"])]
    unknown = set(requested) - KNOWN_OPERATIONS
    if unknown:
        raise InputError(
            "unknown analysis operation(s): " + ", ".join(sorted(unknown)),
            details={"allowed": sorted(KNOWN_OPERATIONS)},
        )
    for operation in ["stats", "quantiles", "histogram"]:
        if operation not in requested:
            requested.append(operation)

    sampled_operations: list[str] = []
    notes: list[str] = []
    size = int(values.size)
    finite, non_finite = _flatten_finite(values)

    if size <= budget.max_analysis_elements:
        stats = basic_stats(values, storage_dtype=storage_dtype)
    else:
        sample, _, _ = _sample_flat(
            values.astype(np.float64, copy=False).ravel(), budget.max_sample_elements
        )
        stats = basic_stats(
            sample.reshape(-1), storage_dtype=storage_dtype, sampled=True
        )
        sampled_operations.append("stats")
        notes.append(
            f"statistics computed on a deterministic sample of {sample.size:,} of {size:,} elements"
        )

    result: dict[str, dict[str, Any]] = {}
    if "quantiles" in requested:
        result["quantiles"] = _quantiles(
            finite, list(quantiles or DEFAULT_QUANTILES), budget.max_sample_elements
        )
        if result["quantiles"]["sampled"]:
            sampled_operations.append("quantiles")
    if "histogram" in requested:
        result["histogram"] = _histogram(
            finite, bins, budget.max_sample_elements, budget.max_histogram_bins
        )
        if result["histogram"]["sampled"]:
            sampled_operations.append("histogram")
    if "norms" in requested:
        result["norms"] = _norms(values, budget)
    if "svd" in requested:
        result["svd"] = _svd(values, top_k, budget)
    if "rank" in requested:
        result["rank"] = _rank(values, budget)

    if non_finite:
        notes.append(f"{non_finite:,} non-finite values; distribution operations use finite values only")
    return TensorAnalysis(
        stats=stats,
        operations=result,
        requested=requested,
        sampled_operations=sampled_operations,
        notes=notes,
    )
