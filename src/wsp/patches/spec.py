"""Immutable patch representation (schema ``wsp.patch/1.0``)."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wsp.core.errors import InputError
from wsp.core.models import CheckpointIdentity, StatsSummary, TensorInfo
from wsp.core.selection import ResolvedSelection


class PayloadRef(BaseModel):
    """Inline tensor values or a reference to a workspace artifact.

    Exactly one of ``values`` or ``artifact_id`` is present.  Shape and dtype
    are always recorded so the patch remains self-describing.
    """

    model_config = ConfigDict(extra="forbid")

    values: Any | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    numel: int | None = None
    artifact_id: str | None = None
    sha256: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> "PayloadRef":
        if self.values is None and self.artifact_id is None:
            raise ValueError("payload must provide either inline values or an artifact_id")
        if self.values is not None and self.artifact_id is not None:
            raise ValueError("payload cannot provide both inline values and an artifact_id")
        return self


class ReplaceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["replace"] = "replace"
    values: PayloadRef


class AddParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["add"] = "add"
    delta: PayloadRef


class ScaleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["scale"] = "scale"
    factor: float

    @model_validator(mode="after")
    def _finite(self) -> "ScaleParams":
        if not math.isfinite(self.factor):
            raise ValueError("scale factor must be finite")
        return self


class RankOneParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["rank_one"] = "rank_one"
    u: PayloadRef
    v: PayloadRef
    scale: float = 1.0

    @model_validator(mode="after")
    def _finite(self) -> "RankOneParams":
        if not math.isfinite(self.scale):
            raise ValueError("rank-one scale must be finite")
        return self


class ProjectionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["projection"] = "projection"
    direction: PayloadRef
    axis: int = 1
    mode: Literal["remove", "add"] = "remove"
    strength: float = 1.0

    @model_validator(mode="after")
    def _valid(self) -> "ProjectionParams":
        if self.axis not in (0, 1):
            raise ValueError("projection axis must be 0 or 1")
        if not math.isfinite(self.strength):
            raise ValueError("projection strength must be finite")
        return self


PatchParams = Annotated[
    Union[ReplaceParams, AddParams, ScaleParams, RankOneParams, ProjectionParams],
    Field(discriminator="kind"),
]


class ExpectedTensor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fingerprint: str
    dtype: str
    shape: list[int]


class PatchEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changed_count: int
    unchanged_count: int
    max_abs_change: float | None = None
    mean_abs_change: float | None = None
    relative_frobenius_change: float | None = None
    cosine_before_after: float | None = None
    introduced_non_finite: bool = False


class PrecisionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compute_dtype: str = "float64"
    storage_dtype: str
    cast_applied: bool
    cast_changed_values: bool = False
    notes: list[str] = Field(default_factory=list)


class PatchProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_by: str = "wsp-workbench"
    tool: str = "create_patch"
    dependencies: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None
    request_id: str | None = None


class WeightPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.patch/1.0"
    patch_id: str
    created_at: str
    source: CheckpointIdentity
    target: str
    target_info: TensorInfo
    selection: ResolvedSelection
    operation: str
    params: PatchParams
    expected: ExpectedTensor
    parameter_count: int
    estimated_memory_bytes: int
    before: StatsSummary
    after: StatsSummary
    effect: PatchEffect
    precision: PrecisionReport
    provenance: PatchProvenance


class PatchPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.preview/1.0"
    patch_id: str
    source: CheckpointIdentity
    target: str
    selection: ResolvedSelection
    operation: str
    parameter_count: int
    before: StatsSummary
    after: StatsSummary
    effect: PatchEffect
    precision: PrecisionReport
    expected: ExpectedTensor
    compatible: bool = True
    checks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PatchSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.patchset/1.0"
    patchset_id: str
    created_at: str
    source: CheckpointIdentity
    patch_ids: list[str]
    patches: list[WeightPatch]
    output_path: str | None = None
    output_identity: CheckpointIdentity | None = None
    status: Literal["pending", "applied"] = "pending"


def payload_identity(params: PatchParams) -> dict[str, Any]:
    """Deterministic, hash-friendly identity for patch parameters."""

    if isinstance(params, ReplaceParams):
        return {"kind": params.kind, "values_sha256": params.values.sha256}
    if isinstance(params, AddParams):
        return {"kind": params.kind, "delta_sha256": params.delta.sha256}
    if isinstance(params, ScaleParams):
        return {"kind": params.kind, "factor": params.factor}
    if isinstance(params, RankOneParams):
        return {
            "kind": params.kind,
            "u_sha256": params.u.sha256,
            "v_sha256": params.v.sha256,
            "scale": params.scale,
        }
    if isinstance(params, ProjectionParams):
        return {
            "kind": params.kind,
            "direction_sha256": params.direction.sha256,
            "axis": params.axis,
            "mode": params.mode,
            "strength": params.strength,
        }
    raise InputError(f"unsupported patch operation: {type(params).__name__}")
