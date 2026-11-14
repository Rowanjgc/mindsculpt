"""Schemas for trace-linked activation measurement.

A capture record describes exactly what was measured: which trace, which
replay, which alignment, which modules, which token positions, and which
prediction semantics those positions carry.  It never claims that a numeric
difference is meaningful.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from wsp.core.models import StatsSummary

SCHEMA_CAPTURE = "wsp.trace-capture/1.0"
SCHEMA_POINTS = "wsp.capture-points/1.0"

PointKind = Literal[
    "attention",
    "linear-attention",
    "mlp",
    "norm",
    "embedding",
    "head",
    "block",
    "recurrent",
    "aggregate",
    "other",
]
CaptureMode = Literal["summary", "selected"]
PositionSemantics = Literal["consuming", "predicting"]
TensorRole = Literal["output", "input"]

POSITION_SEMANTICS_NOTE = (
    "consuming captures the state at the token position itself; predicting captures the state "
    "at the preceding position, which is the state a causal language model actually uses to "
    "predict that token"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CapturePoint(StrictModel):
    """One addressable module in a loaded model."""

    path: str
    class_name: str
    kind: PointKind = "other"
    depth: int = 0
    parameter_count: int = 0
    numel: int = 0
    is_leaf: bool = True
    child_count: int = 0
    repeated: bool = False
    notes: list[str] = Field(default_factory=list)


class CapturePointList(StrictModel):
    schema_version: str = SCHEMA_POINTS
    model_name: str | None = None
    model_class: str | None = None
    architecture: str | None = None
    sequence_dim_hint: str = "dim 1 for 3-D tensors, dim 0 for 2-D tensors"
    total_modules: int = 0
    points: list[CapturePoint] = Field(default_factory=list)
    truncated: bool = False
    notes: list[str] = Field(default_factory=list)


class DimensionPeak(StrictModel):
    index: int
    value: float


class Measurement(StrictModel):
    """The measurement of one tensor at one token position."""

    module: str
    class_name: str
    tensor_role: TensorRole = "output"
    tensor_name: str = "output"
    call_index: int = 0
    position: int
    shape: list[int] = Field(default_factory=list)
    slice_shape: list[int] = Field(default_factory=list)
    dtype: str
    device: str
    stats: StatsSummary | None = None
    norm: float | None = None
    top_dimensions: list[DimensionPeak] = Field(default_factory=list)
    retained: bool = False
    retained_shape: list[int] | None = None
    retained_numel: int = 0
    sampled: bool = False
    stride: int | None = None
    value_artifact_id: str | None = None
    value_sha256: str | None = None
    values: list[Any] | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("shape", "slice_shape")
    @classmethod
    def _non_negative(cls, value: list[int]) -> list[int]:
        if any(int(dimension) < 0 for dimension in value):
            raise ValueError("tensor dimensions must be non-negative")
        return value


class ModuleSummary(StrictModel):
    module: str
    class_name: str
    tensor_role: TensorRole = "output"
    positions: list[int] = Field(default_factory=list)
    measurements: int = 0
    mean_of_means: float | None = None
    mean_norm: float | None = None
    max_abs: float | None = None
    notes: list[str] = Field(default_factory=list)


class CaptureResources(StrictModel):
    positions: int = 0
    modules: int = 0
    measurements: int = 0
    elements_measured: int = 0
    elements_retained: int = 0
    artifacts: int = 0
    estimated_retained_bytes: int = 0
    element_limit_per_tensor: int | None = None
    element_limit_total: int | None = None


class TraceCaptureRecord(StrictModel):
    """Evidence for one targeted activation capture."""

    schema_version: str = SCHEMA_CAPTURE
    capture_id: str
    created_at: str
    trace_id: str
    trace_artifact_sha256: str | None = None
    plan_id: str | None = None
    replay_id: str | None = None
    replay_artifact_id: str | None = None
    alignment_id: str | None = None
    alignment_artifact_id: str | None = None
    span_artifact_id: str | None = None
    model_name: str | None = None
    model_file_sha256: str | None = None
    model_structure_fingerprint: str | None = None
    model_parameter_count: int | None = None
    backend: str = "torch-forward-hooks"
    backend_version: str | None = None
    device: str = "cpu"
    dtype: str | None = None
    mode: CaptureMode = "summary"
    position_semantics: PositionSemantics = "consuming"
    position_semantics_note: str = POSITION_SEMANTICS_NOTE
    requested_tokens: list[int] = Field(default_factory=list)
    positions: list[int] = Field(default_factory=list)
    modules: list[str] = Field(default_factory=list)
    include_inputs: bool = False
    first_call_only: bool = True
    sequence_length: int = 0
    measurements: list[Measurement] = Field(default_factory=list)
    measurements_truncated: bool = False
    modules_summary: list[ModuleSummary] = Field(default_factory=list)
    resources: CaptureResources = Field(default_factory=CaptureResources)
    checksums: dict[str, str] = Field(default_factory=dict)
    replay_verification: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    unavailable: dict[str, str] = Field(default_factory=dict)
    privacy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def measurement_page(self, *, offset: int, limit: int) -> list[Measurement]:
        return self.measurements[offset : offset + limit]
