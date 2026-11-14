"""Comparison of two checkpoints without loading either model."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from wsp.core.errors import UnsupportedError
from wsp.core.models import CheckpointIdentity
from wsp.core.selection import Selection
from wsp.inspection.base import CheckpointSource
from wsp.tensors.diff import compare_arrays
from wsp.tensors.stats import sampled_selection


class TensorComparisonEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: Literal[
        "equivalent", "different", "sampled_different", "dtype_mismatch", "shape_mismatch", "error"
    ]
    left_dtype: str | None = None
    right_dtype: str | None = None
    left_shape: list[int] | None = None
    right_shape: list[int] | None = None
    max_abs: float | None = None
    mean_abs: float | None = None
    relative_frobenius: float | None = None
    cosine: float | None = None
    sampled: bool = False
    error: str | None = None


class CheckpointComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.comparison/1.0"
    left: CheckpointIdentity
    right: CheckpointIdentity
    compatible: bool
    identical: bool
    missing_in_left: list[str] = Field(default_factory=list)
    missing_in_right: list[str] = Field(default_factory=list)
    missing_truncated: bool = False
    entries: list[TensorComparisonEntry] = Field(default_factory=list)
    compared_count: int = 0
    equivalent_count: int = 0
    different_count: int = 0
    skipped_count: int = 0
    shared_count: int = 0
    truncated: bool = False
    atol: float = 1e-6
    rtol: float = 1e-5
    notes: list[str] = Field(default_factory=list)


def compare_checkpoints(
    left: CheckpointSource,
    right: CheckpointSource,
    *,
    max_tensors: int,
    max_elements: int,
    sample_elements: int,
    name_contains: str | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> CheckpointComparison:
    left_infos = {info.name: info for info in left.tensor_infos()}
    right_infos = {info.name: info for info in right.tensor_infos()}
    left_names = set(left_infos)
    right_names = set(right_infos)

    def keep(name: str) -> bool:
        return name_contains is None or name_contains in name

    missing_left = sorted(name for name in right_names - left_names if keep(name))
    missing_right = sorted(name for name in left_names - right_names if keep(name))
    shared = sorted(name for name in left_names & right_names if keep(name))

    entries: list[TensorComparisonEntry] = []
    truncated = False
    missing_truncated = len(missing_left) > max_tensors or len(missing_right) > max_tensors
    missing_left = missing_left[:max_tensors]
    missing_right = missing_right[:max_tensors]

    compared = 0
    equivalent = 0
    different = 0
    skipped = 0
    incompatible = False
    all_full = True

    for name in shared:
        if compared >= max_tensors:
            truncated = True
            break
        left_info = left_infos[name]
        right_info = right_infos[name]
        if left_info.dtype != right_info.dtype:
            incompatible = True
            entries.append(
                TensorComparisonEntry(
                    name=name,
                    status="dtype_mismatch",
                    left_dtype=left_info.dtype,
                    right_dtype=right_info.dtype,
                    left_shape=left_info.shape,
                    right_shape=right_info.shape,
                )
            )
            compared += 1
            different += 1
            continue
        if left_info.shape != right_info.shape:
            incompatible = True
            entries.append(
                TensorComparisonEntry(
                    name=name,
                    status="shape_mismatch",
                    left_dtype=left_info.dtype,
                    right_dtype=right_info.dtype,
                    left_shape=left_info.shape,
                    right_shape=right_info.shape,
                )
            )
            compared += 1
            different += 1
            continue

        selection: Selection | None = None
        sampled = False
        read_limit = max_elements
        if left_info.numel > max_elements:
            selection = sampled_selection(
                Selection().resolve(left_info.shape), sample_elements
            )
            sampled = True
            read_limit = sample_elements
            all_full = False
        try:
            left_values = left.read_tensor(name, selection, max_elements=read_limit)
            right_values = right.read_tensor(name, selection, max_elements=read_limit)
            diff = compare_arrays(
                left_values, right_values, atol=atol, rtol=rtol, sampled=sampled
            )
        except UnsupportedError as exc:
            entries.append(
                TensorComparisonEntry(
                    name=name,
                    status="error",
                    left_dtype=left_info.dtype,
                    right_dtype=right_info.dtype,
                    left_shape=left_info.shape,
                    right_shape=right_info.shape,
                    error=exc.message,
                )
            )
            compared += 1
            skipped += 1
            continue

        if not diff.compatible:
            status: Any = "shape_mismatch"
            incompatible = True
        elif diff.allclose:
            status = "equivalent"
            equivalent += 1
        elif sampled:
            status = "sampled_different"
            different += 1
        else:
            status = "different"
            different += 1
        entries.append(
            TensorComparisonEntry(
                name=name,
                status=status,
                left_dtype=left_info.dtype,
                right_dtype=right_info.dtype,
                left_shape=left_info.shape,
                right_shape=right_info.shape,
                max_abs=diff.max_abs,
                mean_abs=diff.mean_abs,
                relative_frobenius=diff.relative_frobenius,
                cosine=diff.cosine,
                sampled=sampled,
            )
        )
        compared += 1

    notes: list[str] = []
    if truncated:
        notes.append(f"comparison stopped after {max_tensors} shared tensors")
    if not all_full:
        notes.append("some differing tensors were compared using deterministic stride samples")
    identical = (
        not missing_left
        and not missing_right
        and not incompatible
        and different == 0
        and not truncated
        and all_full
        and len(shared) > 0
    )
    return CheckpointComparison(
        left=left.identity(),
        right=right.identity(),
        compatible=not incompatible,
        identical=identical,
        missing_in_left=missing_left,
        missing_in_right=missing_right,
        missing_truncated=missing_truncated,
        entries=entries,
        compared_count=compared,
        equivalent_count=equivalent,
        different_count=different,
        skipped_count=skipped,
        shared_count=len(shared),
        truncated=truncated,
        atol=atol,
        rtol=rtol,
        notes=notes,
    )
