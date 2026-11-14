"""Compare outputs and selected activations from two forward passes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from wsp.activations.compare import ActivationComparison, compare_activations
from wsp.core.errors import InputError
from wsp.core.models import TensorDiffStats
from wsp.execution.forward import ForwardExecution
from wsp.tensors.diff import compare_arrays


class OutputComparisonEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    left_shape: list[int]
    right_shape: list[int]
    left_dtype: str | None = None
    right_dtype: str | None = None
    diff: TensorDiffStats
    note: str | None = None


class ForwardComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.forward-comparison/1.0"
    left_run_id: str
    right_run_id: str
    outputs_identical: bool
    entries: list[OutputComparisonEntry] = Field(default_factory=list)
    missing_left: list[str] = Field(default_factory=list)
    missing_right: list[str] = Field(default_factory=list)
    activation_comparison: ActivationComparison | None = None
    warnings: list[str] = Field(default_factory=list)


def compare_forward(
    left: ForwardExecution,
    right: ForwardExecution,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    compare_activations_too: bool = True,
) -> ForwardComparison:
    left_values = left.output_values
    right_values = right.output_values
    keys = sorted(set(left_values) | set(right_values))
    entries: list[OutputComparisonEntry] = []
    warnings: list[str] = []
    for key in keys:
        if key not in left_values or key not in right_values:
            continue
        left_array = left_values[key]
        right_array = right_values[key]
        diff = compare_arrays(left_array, right_array, atol=atol, rtol=rtol)
        note = None
        left_spec = next((item for item in left.result.outputs if item.key == key), None)
        right_spec = next((item for item in right.result.outputs if item.key == key), None)
        if (left_spec and left_spec.sampled) or (right_spec and right_spec.sampled):
            note = "comparison uses retained deterministic output samples"
            warnings.append(f"output {key!r} comparison is sampled")
        entries.append(
            OutputComparisonEntry(
                key=key,
                left_shape=[int(d) for d in left_array.shape],
                right_shape=[int(d) for d in right_array.shape],
                left_dtype=left_spec.dtype if left_spec else None,
                right_dtype=right_spec.dtype if right_spec else None,
                diff=diff,
                note=note,
            )
        )
    missing_left = sorted(set(right_values) - set(left_values))
    missing_right = sorted(set(left_values) - set(right_values))
    if missing_left or missing_right:
        warnings.append("output keys differ between the two forward passes")
    outputs_identical = (
        not missing_left
        and not missing_right
        and all(entry.diff.allclose for entry in entries)
    )
    activation_comparison = None
    if compare_activations_too:
        activation_comparison = compare_activations(
            left.result.activations,
            right.result.activations,
            atol=atol,
            rtol=rtol,
        )
    return ForwardComparison(
        left_run_id=left.result.run_id,
        right_run_id=right.result.run_id,
        outputs_identical=outputs_identical,
        entries=entries,
        missing_left=missing_left,
        missing_right=missing_right,
        activation_comparison=activation_comparison,
        warnings=warnings,
    )
