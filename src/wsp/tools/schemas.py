"""JSON-compatible tool operation schemas.

These schemas are provider-neutral.  The same models validate Python calls and
CLI JSON payloads, and their JSON Schema forms can be exposed to future MCP or
other tool protocols without moving validation into an adapter.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wsp.core.models import ArtifactRef
from wsp.core.selection import Selection
from wsp.patches.spec import PatchParams, WeightPatch


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget: dict[str, int] | None = None


class InspectModelInput(ToolInput):
    path: str
    include_tensors: bool = True
    compute_file_sha256: bool = False
    save_manifest: bool = False


class ListTensorsInput(ToolInput):
    path: str
    name_contains: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=10_000)


class InspectTensorInput(ToolInput):
    path: str
    tensor: str
    preview_elements: int = Field(default=32, ge=1, le=100_000)
    include_fingerprint: bool = False


class ReadTensorSliceInput(ToolInput):
    path: str
    tensor: str
    selection: Selection = Field(default_factory=Selection)
    preview_elements: int = Field(default=256, ge=1, le=100_000)


class CompareTensorsInput(ToolInput):
    left_path: str
    right_path: str
    tensor: str
    right_tensor: str | None = None
    selection: Selection = Field(default_factory=Selection)
    atol: float = Field(default=1e-6, ge=0)
    rtol: float = Field(default=1e-5, ge=0)


class AnalyzeTensorInput(ToolInput):
    path: str
    tensor: str
    selection: Selection = Field(default_factory=Selection)
    operations: list[str] = Field(default_factory=lambda: ["stats"])
    quantiles: list[float] | None = None
    bins: int = Field(default=32, ge=1)
    top_k: int = Field(default=10, ge=1)


class CompareCheckpointsInput(ToolInput):
    left_path: str
    right_path: str
    name_contains: str | None = None
    max_tensors: int | None = Field(default=None, ge=1)
    atol: float = Field(default=1e-6, ge=0)
    rtol: float = Field(default=1e-5, ge=0)


class CreatePatchInput(ToolInput):
    path: str
    target: str
    selection: Selection = Field(default_factory=Selection)
    params: PatchParams
    expected_fingerprint: str | None = None
    notes: str | None = None
    save: bool = True
    externalize_payloads: bool = True


class PreviewPatchInput(ToolInput):
    path: str
    patch_id: str | None = None
    patch: WeightPatch | None = None

    @model_validator(mode="after")
    def _one_patch(self) -> "PreviewPatchInput":
        if (self.patch_id is None) == (self.patch is None):
            raise ValueError("provide exactly one of patch_id or patch")
        return self


class ApplyPatchInput(ToolInput):
    path: str
    output_path: str
    patch_ids: list[str] = Field(default_factory=list)
    patches: list[WeightPatch] = Field(default_factory=list)
    overwrite: bool = False
    save_patchset: bool = True

    @model_validator(mode="after")
    def _one_source(self) -> "ApplyPatchInput":
        if not self.patch_ids and not self.patches:
            raise ValueError("provide patch_ids or patches")
        if self.patch_ids and self.patches:
            raise ValueError("provide either patch_ids or patches, not both")
        return self


class ListPatchesInput(ToolInput):
    source_fingerprint: str | None = None
    limit: int = Field(default=100, ge=1, le=10_000)


class RevertPatchInput(ToolInput):
    path: str
    patchset_id: str


class ExportCheckpointInput(ToolInput):
    path: str
    output_path: str
    overwrite: bool = False


class ListArtifactsInput(ToolInput):
    kind: str | None = None
    limit: int = Field(default=100, ge=1, le=10_000)


class BuiltinModelSpec(BaseModel):
    """A deterministic, locally constructed model for offline demonstrations."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["mlp", "transformer", "linear"] = "mlp"
    seed: int = 0
    input_dim: int = Field(default=8, ge=1, le=4096)
    hidden_dim: int = Field(default=16, ge=1, le=4096)
    output_dim: int = Field(default=4, ge=1, le=4096)
    vocab_size: int = Field(default=32, ge=1, le=100_000)
    model_dim: int = Field(default=16, ge=1, le=4096)
    num_heads: int = Field(default=4, ge=1, le=128)
    num_layers: int = Field(default=2, ge=1, le=64)
    max_seq_len: int = Field(default=16, ge=1, le=4096)
    checkpoint_path: str | None = None

    @model_validator(mode="after")
    def _valid_transformer(self) -> "BuiltinModelSpec":
        if self.kind == "transformer" and self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        return self


class CaptureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_names: list[str] = Field(default_factory=lambda: ["<root>"])
    record_inputs: bool = False
    token_position: int | None = None
    include_values: bool = True
    max_elements_per_tensor: int | None = Field(default=None, ge=1)
    max_elements_total: int | None = Field(default=None, ge=1)


class RunForwardInput(ToolInput):
    model: BuiltinModelSpec
    inputs: dict[str, Any] | None = None
    input_args: list[Any] = Field(default_factory=list)
    device: str | None = None
    dtype: str | None = None
    seed: int | None = None
    capture: CaptureConfig | None = None
    include_output_values: bool = False
    output_preview_elements: int = Field(default=256, ge=1, le=100_000)
    save_artifacts: bool = True


class CaptureActivationsInput(ToolInput):
    model: BuiltinModelSpec
    inputs: dict[str, Any] | None = None
    input_args: list[Any] = Field(default_factory=list)
    device: str | None = None
    dtype: str | None = None
    seed: int | None = None
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    save_artifact: bool = True


class CompareActivationsInput(ToolInput):
    left_capture_artifact: str
    right_capture_artifact: str
    atol: float = Field(default=1e-6, ge=0)
    rtol: float = Field(default=1e-5, ge=0)


class CompareForwardInput(ToolInput):
    left_run_id: str
    right_run_id: str
    atol: float = Field(default=1e-6, ge=0)
    rtol: float = Field(default=1e-5, ge=0)
    compare_activations: bool = True


class ToolError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ToolEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.tool/1.0"
    ok: bool
    operation: str
    data: Any | None = None
    error: ToolError | None = None
    run_id: str | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
