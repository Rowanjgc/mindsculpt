"""Patch arithmetic, validation, and precision reporting.

Patch math is performed in float64 against a bounded selection.  The result is
cast back to the tensor's storage dtype; any resulting change is reported in
the precision report.  Arithmetic on integer/bool tensors is rejected except
for exact replacement.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from wsp.core.budget import Budget, enforce_numel
from wsp.core.dtypes import (
    bytes_per_element,
    is_float_storage,
    normalize_dtype_name,
    numpy_dtype_for,
)
from wsp.core.errors import InputError, ResourceLimitError, UnsupportedError
from wsp.core.hashing import tensor_fingerprint
from wsp.core.selection import ResolvedSelection
from wsp.patches.spec import (
    AddParams,
    PatchParams,
    PayloadRef,
    ProjectionParams,
    RankOneParams,
    ReplaceParams,
    ScaleParams,
)

ArtifactLoader = Callable[[str], Any]


def _float64(values: Any, *, what: str) -> np.ndarray[Any, Any]:
    array = np.asarray(values)
    if array.dtype == object or np.iscomplexobj(array):
        raise InputError(f"{what} must be a real numeric array")
    return np.ascontiguousarray(array.astype(np.float64, copy=False))


def resolve_payload(
    payload: PayloadRef,
    *,
    expected_shape: list[int] | None,
    budget: Budget,
    loader: ArtifactLoader | None = None,
    what: str = "payload",
) -> np.ndarray[Any, Any]:
    if payload.artifact_id is not None:
        if loader is None:
            raise InputError(f"{what} references an artifact but no artifact loader is available")
        raw = loader(payload.artifact_id)
        if isinstance(raw, dict) and "values" in raw:
            raw = raw["values"]
    else:
        raw = payload.values

    array = np.asarray(raw)
    if array.dtype == object or np.iscomplexobj(array):
        raise InputError(f"{what} must contain real numeric values")
    if payload.shape is not None and list(array.shape) != list(payload.shape):
        raise InputError(
            f"{what} shape does not match its declared shape",
            details={"declared": payload.shape, "actual": list(array.shape)},
        )
    if expected_shape is not None and list(array.shape) != list(expected_shape):
        raise InputError(
            f"{what} shape does not match the selected region",
            details={"expected": expected_shape, "actual": list(array.shape)},
        )
    enforce_numel(
        int(array.size),
        budget.max_patch_payload_elements,
        what=f"{what} elements",
    )
    values = _float64(array, what=what)
    if not np.all(np.isfinite(values)):
        raise InputError(f"{what} contains non-finite values")
    return values


def payload_sha256(values: np.ndarray[Any, Any]) -> str:
    return tensor_fingerprint(values, storage_dtype="float64", shape=list(values.shape))


def payload_ref_from_values(values: np.ndarray[Any, Any]) -> PayloadRef:
    array = np.asarray(values)
    return PayloadRef(
        values=array.tolist(),
        shape=[int(d) for d in array.shape],
        dtype="float64",
        numel=int(array.size),
        sha256=payload_sha256(array),
    )


def _cast_to_storage(
    values: np.ndarray[Any, Any], storage_dtype: str
) -> tuple[np.ndarray[Any, Any], bool]:
    canonical = normalize_dtype_name(storage_dtype)
    source = _float64(values, what="patch result")

    if canonical in {"float8_e4m3fn", "float8_e5m2"}:
        raise UnsupportedError("patching float8 tensors is out of scope for V1")
    if canonical == "bfloat16":
        import torch

        casted = (
            torch.from_numpy(np.ascontiguousarray(source.astype(np.float32)))
            .to(torch.bfloat16)
            .float()
            .numpy()
        )
        return casted, True
    if canonical in {"float16", "float32", "float64"}:
        casted = source.astype(numpy_dtype_for(canonical), copy=False)
        return casted, canonical != "float64"
    if canonical == "bool":
        if not np.all((source == 0.0) | (source == 1.0)):
            raise InputError("boolean tensors can only be replaced with 0/1 values")
        return source.astype(np.bool_), True
    if canonical in {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}:
        rounded = np.rint(source)
        if not np.array_equal(source, rounded):
            raise InputError(
                f"replacement values cannot be represented exactly as {canonical}",
                details={"storage_dtype": canonical},
            )
        target = numpy_dtype_for(canonical)
        limits = np.iinfo(target)
        if source.size and (source.min() < limits.min or source.max() > limits.max):
            raise InputError(
                f"replacement values exceed the range of {canonical}",
                details={"storage_dtype": canonical},
            )
        return rounded.astype(target), True
    raise UnsupportedError(f"patching {storage_dtype} tensors is not supported")


def _apply_float64(
    base: np.ndarray[Any, Any],
    params: PatchParams,
    resolved: ResolvedSelection,
    *,
    budget: Budget,
    loader: ArtifactLoader | None,
) -> tuple[np.ndarray[Any, Any], str]:
    selected_shape = list(resolved.selected_shape)
    if isinstance(params, ReplaceParams):
        values = resolve_payload(
            params.values,
            expected_shape=selected_shape,
            budget=budget,
            loader=loader,
            what="replacement values",
        )
        return values, "replace"
    if isinstance(params, AddParams):
        delta = resolve_payload(
            params.delta,
            expected_shape=selected_shape,
            budget=budget,
            loader=loader,
            what="delta",
        )
        return base + delta, "add"
    if isinstance(params, ScaleParams):
        return base * params.factor, "scale"
    if isinstance(params, RankOneParams):
        if len(selected_shape) != 2:
            raise InputError(
                "rank-one updates require a 2-D selected region",
                details={"selected_shape": selected_shape},
            )
        rows, columns = selected_shape
        u = resolve_payload(
            params.u,
            expected_shape=[rows],
            budget=budget,
            loader=loader,
            what="rank-one u vector",
        )
        v = resolve_payload(
            params.v,
            expected_shape=[columns],
            budget=budget,
            loader=loader,
            what="rank-one v vector",
        )
        return base + params.scale * np.outer(u, v), "rank_one"
    if isinstance(params, ProjectionParams):
        if len(selected_shape) != 2:
            raise InputError(
                "projection edits require a 2-D selected region",
                details={"selected_shape": selected_shape},
            )
        rows, columns = selected_shape
        dimension = columns if params.axis == 1 else rows
        direction = resolve_payload(
            params.direction,
            expected_shape=[dimension],
            budget=budget,
            loader=loader,
            what="projection direction",
        )
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise InputError("projection direction must have non-zero norm")
        unit = direction / norm
        if params.axis == 1:
            coefficients = base @ unit
            update = np.outer(coefficients, unit)
        else:
            coefficients = unit @ base
            update = np.outer(unit, coefficients)
        if params.mode == "remove":
            return base - params.strength * update, "projection"
        return base + params.strength * update, "projection"
    raise UnsupportedError(f"unsupported patch operation: {type(params).__name__}")


def apply_operation(
    base_values: np.ndarray[Any, Any],
    params: PatchParams,
    resolved: ResolvedSelection,
    *,
    storage_dtype: str,
    budget: Budget,
    loader: ArtifactLoader | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, Any]]:
    """Apply an operation and return ``(storage_values, computed_values, report)``."""

    if not is_float_storage(storage_dtype):
        if not isinstance(params, ReplaceParams):
            raise InputError(
                f"arithmetic patches require a floating-point tensor; {storage_dtype} is not supported",
                details={"storage_dtype": storage_dtype, "operation": params.kind},
            )
    base = _float64(base_values, what="selected tensor region")
    computed, operation = _apply_float64(
        base, params, resolved, budget=budget, loader=loader
    )
    if computed.shape != base.shape:
        raise InputError(
            "patch operation changed the selected shape",
            details={"expected": list(base.shape), "actual": list(computed.shape)},
        )
    casted, cast_applied = _cast_to_storage(computed, storage_dtype)
    casted_float = _float64(casted, what="cast patch result")
    cast_changed = not np.array_equal(casted_float, computed, equal_nan=False)
    introduced = bool(np.any(~np.isfinite(casted_float) & np.isfinite(computed)))
    if introduced:
        raise InputError(
            "patch result is non-finite after storage casting",
            details={"storage_dtype": storage_dtype},
        )
    notes: list[str] = []
    if cast_applied:
        notes.append(f"computed in float64 and stored as {storage_dtype}")
    if cast_changed:
        notes.append("storage casting changed one or more values")
    report = {
        "operation": operation,
        "storage_dtype": normalize_dtype_name(storage_dtype),
        "cast_applied": cast_applied,
        "cast_changed": cast_changed,
        "notes": notes,
        "payload_elements": int(computed.size),
        "estimated_memory_bytes": int(
            computed.size * 8 * 2 + computed.size * bytes_per_element(storage_dtype) * 2
        ),
    }
    return casted_float, computed, report


def compute_effect(
    before: np.ndarray[Any, Any], after: np.ndarray[Any, Any]
) -> dict[str, Any]:
    before_values = _float64(before, what="before values")
    after_values = _float64(after, what="after values")
    if before_values.shape != after_values.shape:
        raise InputError("before and after shapes differ")
    before_flat = before_values.ravel()
    after_flat = after_values.ravel()
    both_nan = np.isnan(before_flat) & np.isnan(after_flat)
    changed = (before_flat != after_flat) & ~both_nan
    changed_count = int(np.count_nonzero(changed))
    finite_pairs = np.isfinite(before_flat) & np.isfinite(after_flat)
    if np.any(finite_pairs):
        delta = after_flat[finite_pairs] - before_flat[finite_pairs]
        max_abs = float(np.max(np.abs(delta)))
        mean_abs = float(np.mean(np.abs(delta)))
        denominator = float(np.linalg.norm(before_flat[finite_pairs]))
        relative = float(np.linalg.norm(delta) / denominator) if denominator > 0 else None
        before_norm = denominator
        after_norm = float(np.linalg.norm(after_flat[finite_pairs]))
        cosine = (
            float(np.dot(before_flat[finite_pairs], after_flat[finite_pairs]) / (before_norm * after_norm))
            if before_norm > 0 and after_norm > 0
            else None
        )
    else:
        max_abs = mean_abs = relative = cosine = None
    introduced = bool(np.any(~np.isfinite(after_flat) & np.isfinite(before_flat)))
    return {
        "changed_count": changed_count,
        "unchanged_count": int(before_flat.size - changed_count),
        "max_abs_change": max_abs,
        "mean_abs_change": mean_abs,
        "relative_frobenius_change": relative,
        "cosine_before_after": cosine,
        "introduced_non_finite": introduced,
    }
