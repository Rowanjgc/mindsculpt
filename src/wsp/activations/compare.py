"""Compare activation summaries and retained captures."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wsp.activations.capture import ActivationCapture, ActivationSummary
from wsp.core.models import StatsSummary, TensorDiffStats
from wsp.tensors.diff import compare_arrays


class ActivationComparisonEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: str
    kind: str
    left_call_index: int
    right_call_index: int
    left_shape: list[int]
    right_shape: list[int]
    left_dtype: str
    right_dtype: str
    shapes_match: bool
    dtypes_match: bool
    left_stats: StatsSummary
    right_stats: StatsSummary
    diff: TensorDiffStats | None = None
    compared_values: bool = False
    note: str | None = None


class ActivationComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.activation-comparison/1.0"
    left_capture_id: str | None = None
    right_capture_id: str | None = None
    entries: list[ActivationComparisonEntry] = Field(default_factory=list)
    missing_left: list[str] = Field(default_factory=list)
    missing_right: list[str] = Field(default_factory=list)
    identical_count: int = 0
    different_count: int = 0
    warnings: list[str] = Field(default_factory=list)


def _key(summary: ActivationSummary) -> tuple[str, str, int]:
    return (summary.module, summary.kind, summary.call_index)


def _summary_stats_match(left: StatsSummary, right: StatsSummary) -> bool:
    for field in ("count", "finite_count", "nan_count", "posinf_count", "neginf_count", "zero_count"):
        if getattr(left, field) != getattr(right, field):
            return False
    for field in ("mean", "std", "min", "max", "abs_max"):
        a = getattr(left, field)
        b = getattr(right, field)
        if a is None or b is None:
            if a is not b:
                return False
        elif abs(a - b) > 1e-9 + 1e-6 * max(abs(a), abs(b)):
            return False
    return True


def compare_activations(
    left: Iterable[ActivationSummary] | ActivationCapture,
    right: Iterable[ActivationSummary] | ActivationCapture,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> ActivationComparison:
    left_items = (
        left.summaries(include_values=True)
        if isinstance(left, ActivationCapture)
        else list(left)
    )
    right_items = (
        right.summaries(include_values=True)
        if isinstance(right, ActivationCapture)
        else list(right)
    )
    left_map = {_key(item): item for item in left_items}
    right_map = {_key(item): item for item in right_items}
    entries: list[ActivationComparisonEntry] = []
    warnings: list[str] = []
    for key in sorted(set(left_map) | set(right_map)):
        left_item = left_map.get(key)
        right_item = right_map.get(key)
        if left_item is None or right_item is None:
            continue
        shapes_match = left_item.shape == right_item.shape
        dtypes_match = left_item.original_dtype == right_item.original_dtype
        diff: TensorDiffStats | None = None
        compared_values = False
        note: str | None = None
        left_values = left_item.values
        right_values = right_item.values
        if left_values is not None and right_values is not None and shapes_match:
            sampled = bool(left_item.sampled or right_item.sampled)
            left_array = np.asarray(left_values, dtype=np.float64).reshape(
                left_item.retained_shape or [-1]
            )
            right_array = np.asarray(right_values, dtype=np.float64).reshape(
                right_item.retained_shape or [-1]
            )
            diff = compare_arrays(
                left_array,
                right_array,
                atol=atol,
                rtol=rtol,
                sampled=sampled,
            )
            compared_values = True
            if sampled:
                note = "comparison uses retained deterministic samples, not every element"
        elif left_values is None or right_values is None:
            note = "values were not retained; only summaries can be compared"
        if not shapes_match:
            note = "shapes differ; numerical comparison skipped"
        entries.append(
            ActivationComparisonEntry(
                module=left_item.module,
                kind=left_item.kind,
                left_call_index=left_item.call_index,
                right_call_index=right_item.call_index,
                left_shape=left_item.shape,
                right_shape=right_item.shape,
                left_dtype=left_item.original_dtype,
                right_dtype=right_item.original_dtype,
                shapes_match=shapes_match,
                dtypes_match=dtypes_match,
                left_stats=left_item.stats,
                right_stats=right_item.stats,
                diff=diff,
                compared_values=compared_values,
                note=note,
            )
        )
    missing_left = [
        f"{module}/{kind}/call{index}"
        for module, kind, index in sorted(set(right_map) - set(left_map))
    ]
    missing_right = [
        f"{module}/{kind}/call{index}"
        for module, kind, index in sorted(set(left_map) - set(right_map))
    ]
    if missing_left or missing_right:
        warnings.append("activation keys differ between the two captures")

    def entry_identical(entry: ActivationComparisonEntry) -> bool:
        if entry.diff is not None:
            return bool(entry.diff.allclose and entry.shapes_match)
        if not entry.shapes_match or not entry.dtypes_match:
            return False
        return _summary_stats_match(entry.left_stats, entry.right_stats)

    identical = sum(1 for entry in entries if entry_identical(entry))
    different = len(entries) - identical
    return ActivationComparison(
        left_capture_id=left_items[0].capture_id if left_items else None,
        right_capture_id=right_items[0].capture_id if right_items else None,
        entries=entries,
        missing_left=missing_left,
        missing_right=missing_right,
        identical_count=identical,
        different_count=different,
        warnings=warnings,
    )
