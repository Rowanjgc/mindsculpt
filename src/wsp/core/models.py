"""Shared, versioned data models.

These models are part of the public artifact format.  Additive changes are
allowed within a minor schema version; breaking changes require a new version
string.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TensorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    shape: list[int]
    numel: int
    nbytes: int
    offset: int | None = None


class CheckpointIdentity(BaseModel):
    """Stable identity for the *structure* and optionally the exact file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.identity/1.0"
    format: Literal["safetensors", "torch_module"]
    display_name: str
    structure_fingerprint: str
    file_sha256: str | None = None
    tensor_count: int
    parameter_count: int
    total_bytes: int
    metadata: dict[str, str] = Field(default_factory=dict)


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    display_name: str
    format: str
    size_bytes: int | None = None


class ManifestTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensor_count: int
    parameter_count: int
    total_bytes: int
    by_dtype: dict[str, int] = Field(default_factory=dict)
    by_dtype_bytes: dict[str, int] = Field(default_factory=dict)


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.manifest/1.0"
    created_at: str
    wsp_version: str
    identity: CheckpointIdentity
    source: SourceRef
    tensors: list[TensorInfo] = Field(default_factory=list)
    tensor_count: int
    tensors_truncated: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)
    totals: ManifestTotals


class StatsSummary(BaseModel):
    """Summary statistics over finite values, with explicit non-finite counts."""

    model_config = ConfigDict(extra="forbid")

    dtype: str
    shape: list[int]
    count: int
    finite_count: int
    nan_count: int
    posinf_count: int
    neginf_count: int
    zero_count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    abs_max: float | None = None
    finite_only: bool = True
    sampled: bool = False


class Preview(BaseModel):
    """A bounded, row-major flattened view of tensor values.

    Non-finite values are emitted as ``null``; ``stats``/the originating
    tensor summary reports their counts.  ``sampled=True`` means the values
    are a deterministic stride sample, not a prefix.  ``truncated=True``
    means only a prefix was returned.
    """

    model_config = ConfigDict(extra="forbid")

    layout: Literal["row_major_flat"] = "row_major_flat"
    dtype: str
    shape: list[int]
    values: list[Any]
    numel: int
    returned: int
    truncated: bool = False
    sampled: bool = False
    stride: int | None = None


class TensorDiffStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_shape: list[int]
    right_shape: list[int]
    compatible: bool
    count: int = 0
    max_abs: float | None = None
    mean_abs: float | None = None
    rms: float | None = None
    relative_frobenius: float | None = None
    cosine: float | None = None
    allclose: bool = False
    atol: float = 1e-6
    rtol: float = 1e-5
    non_finite_pairs: int = 0
    sampled: bool = False


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: str
    path: str
    format: str
    sha256: str
    nbytes: int
    created_at: str
    schema_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
