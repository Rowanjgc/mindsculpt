"""Shared, provider-neutral service layer for Python and CLI calls."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from pydantic import BaseModel, ValidationError

from wsp.activations.capture import ActivationCapture, ActivationSummary
from wsp.activations.compare import compare_activations as compare_activation_summaries
from wsp.core.budget import Budget, enforce_bytes, enforce_numel
from wsp.core.dtypes import bytes_per_element
from wsp.core.errors import (
    ArtifactError,
    CompatibilityError,
    InputError,
    NotFoundError,
    ResourceLimitError,
    UnsupportedError,
    WspError,
)
from wsp.core.json import dump_bytes, to_jsonable
from wsp.core.models import ArtifactRef, CheckpointIdentity, ModelManifest, Preview, StatsSummary
from wsp.core.selection import Selection
from wsp.execution.builtin_models import build_builtin_model, prepare_builtin_inputs
from wsp.execution.compare import compare_forward as compare_forward_executions
from wsp.execution.forward import ForwardExecution, ForwardResult, run_forward
from wsp.inspection.compare import compare_checkpoints as compare_checkpoint_sources
from wsp.inspection.factory import open_checkpoint
from wsp.inspection.manifest import build_manifest
from wsp.inspection.torch_module_source import TorchModuleSource
from wsp.patches.engine import (
    apply_patchset,
    build_patchset,
    create_patch,
    export_checkpoint as export_checkpoint_engine,
    preview_patch as preview_patch_engine,
    revert_patchset,
)
from wsp.patches.operations import payload_sha256
from wsp.patches.spec import (
    AddParams,
    PatchParams,
    PayloadRef,
    ProjectionParams,
    RankOneParams,
    ReplaceParams,
    ScaleParams,
    WeightPatch,
)
from wsp.tensors.analysis import analyze_array
from wsp.tensors.diff import compare_arrays
from wsp.tensors.preview import preview_array
from wsp.tensors.stats import basic_stats, sampled_selection
from wsp.tools.schemas import (
    AnalyzeTensorInput,
    ApplyPatchInput,
    CaptureActivationsInput,
    CaptureConfig,
    CompareActivationsInput,
    CompareCheckpointsInput,
    CompareForwardInput,
    CompareTensorsInput,
    CreatePatchInput,
    ExportCheckpointInput,
    InspectModelInput,
    InspectTensorInput,
    ListArtifactsInput,
    ListPatchesInput,
    ListTensorsInput,
    PreviewPatchInput,
    ReadTensorSliceInput,
    RevertPatchInput,
    RunForwardInput,
    ToolInput,
)
from wsp.workspace.run import RunRecorder
from wsp.workspace.store import Workspace


@dataclass
class ToolResult:
    data: dict[str, Any]
    artifacts: list[ArtifactRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


Handler = Callable[[Any, Budget, RunRecorder | None], ToolResult]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Handler
    output_description: str
    requires_workspace: bool = False


def _compact_patch(patch: WeightPatch) -> dict[str, Any]:
    return {
        "patch_id": patch.patch_id,
        "target": patch.target,
        "operation": patch.operation,
        "parameter_count": patch.parameter_count,
        "selection": patch.selection.model_dump(),
        "source_structure_fingerprint": patch.source.structure_fingerprint,
        "created_at": patch.created_at,
    }


def _compact_patchset(patchset: Any) -> dict[str, Any]:
    return {
        "patchset_id": patchset.patchset_id,
        "status": patchset.status,
        "source": patchset.source.model_dump(),
        "patch_ids": patchset.patch_ids,
        "patches": [_compact_patch(patch) for patch in patchset.patches],
        "output_path": patchset.output_path,
        "output_identity": (
            patchset.output_identity.model_dump() if patchset.output_identity else None
        ),
    }


def _resolve_workspace(workspace: Workspace | None = None) -> Workspace | None:
    if workspace is not None:
        return workspace
    import os

    configured = os.environ.get("WSP_WORKSPACE")
    if configured:
        candidate = Path(configured).expanduser()
        if (candidate / "workspace.json").exists():
            return Workspace(candidate)
        return None
    local = Path(".wsp")
    if (local / "workspace.json").exists():
        return Workspace(local)
    return None


class ToolService:
    """Validate and execute structured operations.

    The service is the single implementation shared by the Python helpers and
    the CLI.  It never downloads models, never loads pickle checkpoints, and
    never mutates a source checkpoint.
    """

    def __init__(
        self,
        workspace: Workspace | None = None,
        *,
        budget: Budget | None = None,
        raise_errors: bool = False,
        discover_workspace: bool = True,
    ) -> None:
        self.workspace = (
            _resolve_workspace(workspace) if discover_workspace else workspace
        )
        self.budget = budget or Budget()
        self.raise_errors = raise_errors
        self._definitions: dict[str, ToolDefinition] = {}
        self._models: dict[str, torch.nn.Module] = {}
        self._register_defaults()

    # ------------------------------------------------------------------ registry

    def register_tool(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise InputError(f"tool {definition.name!r} is already registered")
        self._definitions[definition.name] = definition

    def register_model(self, name: str, model: torch.nn.Module) -> None:
        if not name or "/" in name:
            raise InputError("registered model names must be simple identifiers")
        self._models[name] = model

    def operations(self) -> list[str]:
        return sorted(self._definitions)

    def describe(self, operation: str | None = None) -> dict[str, Any]:
        names = [operation] if operation else self.operations()
        entries = []
        for name in names:
            definition = self._definitions.get(name)
            if definition is None:
                raise NotFoundError(f"unknown tool operation {name!r}", details={"available": self.operations()})
            entries.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "input_schema": definition.input_model.model_json_schema(),
                    "output_schema": {
                        "type": "object",
                        "description": definition.output_description,
                    },
                    "requires_workspace": definition.requires_workspace,
                }
            )
        return {"schema_version": "wsp.tool-registry/1.0", "operations": entries}

    # --------------------------------------------------------------------- call

    def call(
        self,
        operation: str,
        payload: Mapping[str, Any] | BaseModel | None = None,
        *,
        raise_errors: bool | None = None,
    ) -> dict[str, Any]:
        should_raise = self.raise_errors if raise_errors is None else raise_errors
        definition = self._definitions.get(operation)
        if definition is None:
            error = InputError(
                f"unknown tool operation {operation!r}",
                details={"available": self.operations()},
            )
            if should_raise:
                raise error
            return self._error_envelope(operation, error)

        try:
            if isinstance(payload, BaseModel):
                request = definition.input_model.model_validate(payload.model_dump())
            else:
                request = definition.input_model.model_validate(payload or {})
        except ValidationError as exc:
            error = InputError(
                "input does not match the operation schema",
                details={
                    "errors": [
                        {"loc": list(item.get("loc", [])), "msg": item.get("msg"), "type": item.get("type")}
                        for item in exc.errors(include_url=False)
                    ]
                },
            )
            if should_raise:
                raise error from exc
            return self._error_envelope(operation, error)

        budget = self.budget.merge(getattr(request, "budget", None))
        recorder: RunRecorder | None = None
        if self.workspace is not None:
            recorder = RunRecorder(
                self.workspace,
                operation,
                request.model_dump(mode="python"),
                budget=budget,
            )
            recorder.__enter__()

        try:
            result = definition.handler(request, budget, recorder)
            if not isinstance(result, ToolResult):
                raise RuntimeError("tool handler returned an invalid result")
            envelope = {
                "schema_version": "wsp.tool/1.0",
                "ok": True,
                "operation": operation,
                "data": to_jsonable(result.data),
                "run_id": recorder.run_id if recorder else None,
                "artifacts": [artifact.model_dump() for artifact in result.artifacts],
                "warnings": result.warnings,
            }
            enforce_bytes(
                len(dump_bytes(envelope)),
                budget.max_output_bytes,
                what=f"{operation} response",
            )
            if recorder is not None:
                for artifact in result.artifacts:
                    recorder.add_artifact(artifact)
                recorder.finish(result=result.data, warnings=result.warnings)
            return envelope
        except BaseException as exc:
            if recorder is not None and not recorder._finished:
                error_payload: dict[str, Any] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                if isinstance(exc, WspError):
                    error_payload.update({"code": exc.code, "details": exc.details})
                try:
                    recorder.finish(status="failed", error=error_payload)
                except Exception:
                    pass
            if should_raise:
                if isinstance(exc, WspError):
                    raise
                if isinstance(exc, ValidationError):
                    raise InputError(str(exc)) from exc
                raise
            if isinstance(exc, WspError):
                return self._error_envelope(
                    operation, exc, run_id=recorder.run_id if recorder else None
                )
            return self._error_envelope(
                operation,
                WspError(f"{operation} failed: {exc}"),
                run_id=recorder.run_id if recorder else None,
            )

    def _error_envelope(
        self,
        operation: str,
        error: WspError,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        envelope = {
            "schema_version": "wsp.tool/1.0",
            "ok": False,
            "operation": operation,
            "data": None,
            "error": error.to_dict(),
            "run_id": run_id,
            "artifacts": [],
            "warnings": [],
        }
        enforce_bytes(
            len(dump_bytes(envelope)),
            self.budget.max_output_bytes,
            what=f"{operation} error response",
        )
        return envelope

    # --------------------------------------------------------------- registration

    def _register_defaults(self) -> None:
        self.register_tool(
            ToolDefinition(
                "inspect_model",
                "Open a SafeTensors checkpoint and return a versioned model manifest without loading tensors.",
                InspectModelInput,
                self._inspect_model,
                "ModelManifest plus optional saved manifest path.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "list_tensors",
                "Page through parameter metadata (name, shape, dtype, size) without materializing tensors.",
                ListTensorsInput,
                self._list_tensors,
                "Checkpoint identity and a bounded page of TensorInfo records.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "inspect_tensor",
                "Read a bounded tensor preview, exact or sampled statistics, and optionally a full fingerprint.",
                InspectTensorInput,
                self._inspect_tensor,
                "TensorInfo, StatsSummary, Preview, and optional fingerprint.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "read_tensor_slice",
                "Read an explicitly selected, bounded tensor slice and summarize it.",
                ReadTensorSliceInput,
                self._read_tensor_slice,
                "ResolvedSelection, StatsSummary, and Preview.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "compare_tensors",
                "Compare corresponding tensors from two checkpoints with explicit tolerances.",
                CompareTensorsInput,
                self._compare_tensors,
                "TensorDiffStats, fingerprints, and dtype/shape compatibility metadata.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "analyze_tensor",
                "Compute bounded statistics, quantiles, histograms, norms, and singular-value summaries.",
                AnalyzeTensorInput,
                self._analyze_tensor,
                "TensorAnalysis with sampled-operation labels.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "compare_checkpoints",
                "Compare two checkpoints tensor-by-tensor, detecting missing, extra, and mismatched tensors.",
                CompareCheckpointsInput,
                self._compare_checkpoints,
                "CheckpointComparison.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "create_patch",
                "Create an immutable WeightPatch for a validated tensor selection and operation.",
                CreatePatchInput,
                self._create_patch,
                "WeightPatch and optional saved path.",
                requires_workspace=False,
            )
        )
        self.register_tool(
            ToolDefinition(
                "preview_patch",
                "Revalidate a patch against its source and recompute its effects without committing it.",
                PreviewPatchInput,
                self._preview_patch,
                "PatchPreview with compatibility checks and warnings.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "apply_patch",
                "Apply one or more patches to an isolated derived checkpoint using streaming export.",
                ApplyPatchInput,
                self._apply_patch,
                "Applied PatchSet summary, output identity, and output path.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "list_patches",
                "List patches stored in the local workspace.",
                ListPatchesInput,
                self._list_patches,
                "Bounded patch summaries.",
                requires_workspace=True,
            )
        )
        self.register_tool(
            ToolDefinition(
                "revert_patch",
                "Verify a patch set and select the immutable source checkpoint as the restored state.",
                RevertPatchInput,
                self._revert_patch,
                "Source identity and restoration note; no file is changed.",
                requires_workspace=True,
            )
        )
        self.register_tool(
            ToolDefinition(
                "export_checkpoint",
                "Copy a checkpoint to a new explicit destination with an atomic write.",
                ExportCheckpointInput,
                self._export_checkpoint,
                "Destination CheckpointIdentity.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "list_artifacts",
                "List content-addressed artifacts in the local workspace.",
                ListArtifactsInput,
                self._list_artifacts,
                "Bounded ArtifactRef list.",
                requires_workspace=True,
            )
        )
        self.register_tool(
            ToolDefinition(
                "run_forward",
                "Run one deterministic inference-mode forward pass on a built-in local model.",
                RunForwardInput,
                self._run_forward,
                "ForwardResult, optional activation summaries, and output artifact references.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "capture_activations",
                "Capture bounded module inputs/outputs for one forward pass and save summaries.",
                CaptureActivationsInput,
                self._capture_activations,
                "Capture id, ActivationSummary list, and optional artifact reference.",
            )
        )
        self.register_tool(
            ToolDefinition(
                "compare_activations",
                "Compare two saved activation captures by key, shape, dtype, and retained values.",
                CompareActivationsInput,
                self._compare_activations,
                "ActivationComparison.",
                requires_workspace=True,
            )
        )
        self.register_tool(
            ToolDefinition(
                "compare_forward",
                "Compare saved outputs and activation summaries from two forward runs.",
                CompareForwardInput,
                self._compare_forward,
                "ForwardComparison.",
                requires_workspace=True,
            )
        )
        self.register_tool(
            ToolDefinition(
                "workspace_status",
                "Report workspace identity and artifact/run/patch counts.",
                ToolInput,
                self._workspace_status,
                "Workspace status.",
                requires_workspace=True,
            )
        )
        from wsp.tools.alignment_handlers import register_alignment_tools
        from wsp.tools.behavior_handlers import register_behavior_tools
        from wsp.tools.capture_handlers import register_capture_tools
        from wsp.tools.replay_handlers import register_replay_tools

        register_behavior_tools(self)
        register_replay_tools(self)
        register_alignment_tools(self)
        register_capture_tools(self)

    # ------------------------------------------------------------------- handlers

    def _inspect_model(
        self, request: InspectModelInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        with open_checkpoint(
            request.path, compute_file_sha256=request.compute_file_sha256
        ) as source:
            manifest = build_manifest(
                source,
                include_tensors=request.include_tensors,
                max_tensors=budget.max_manifest_tensors,
            )
        if run is not None:
            run.model_identity = manifest.identity
            run.source = manifest.source
        warnings: list[str] = []
        if manifest.tensors_truncated:
            warnings.append(
                f"manifest tensor list truncated at {budget.max_manifest_tensors} entries"
            )
        data: dict[str, Any] = {"manifest": manifest}
        if request.save_manifest:
            if self.workspace is None:
                raise InputError(
                    "save_manifest requires a workspace; run 'wsp workspace init' or construct Workspace(create=True)"
                )
            path = self.workspace.save_manifest(manifest)
            data["saved_manifest_path"] = str(path)
        return ToolResult(data=data, warnings=warnings)

    def _list_tensors(
        self, request: ListTensorsInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        enforce_numel(request.limit, budget.max_manifest_tensors, what="tensor listing limit")
        with open_checkpoint(request.path) as source:
            identity = source.identity()
            infos = source.tensor_infos()
        if request.name_contains:
            infos = [info for info in infos if request.name_contains in info.name]
        total = len(infos)
        page = infos[request.offset : request.offset + request.limit]
        if run is not None:
            run.model_identity = identity
        return ToolResult(
            data={
                "identity": identity,
                "total": total,
                "offset": request.offset,
                "returned": len(page),
                "truncated": request.offset + len(page) < total,
                "tensors": page,
            }
        )

    def _read_values(
        self,
        source: Any,
        tensor: str,
        selection: Selection,
        budget: Budget,
        *,
        sample_for_analysis: bool = False,
    ) -> tuple[np.ndarray[Any, Any], Any, bool]:
        info = source.tensor_info(tensor)
        resolved = selection.resolve(info.shape)
        if resolved.numel <= budget.max_analysis_elements:
            values = source.read_tensor(tensor, selection, max_elements=budget.max_analysis_elements)
            return values, resolved, False
        if not sample_for_analysis:
            raise ResourceLimitError(
                f"selection of {tensor!r} is too large to materialize",
                details={
                    "numel": resolved.numel,
                    "limit": budget.max_analysis_elements,
                    "hint": "use a smaller selection or read_tensor_slice with previews",
                },
            )
        sampled = sampled_selection(resolved, budget.max_sample_elements)
        if sampled is None:
            raise ResourceLimitError(f"cannot sample {tensor!r} within the configured budget")
        values = source.read_tensor(tensor, sampled, max_elements=budget.max_sample_elements)
        return values, resolved, True

    def _inspect_tensor(
        self, request: InspectTensorInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        warnings: list[str] = []
        with open_checkpoint(request.path, compute_file_sha256=False) as source:
            info = source.tensor_info(request.tensor)
            values, resolved, sampled = self._read_values(
                source, request.tensor, Selection(), budget, sample_for_analysis=True
            )
            fingerprint = None
            if request.include_fingerprint:
                if sampled or info.numel > budget.max_tensor_numel:
                    warnings.append(
                        "fingerprint was not computed because the full tensor exceeds the bounded-read budget"
                    )
                else:
                    fingerprint = source.tensor_fingerprint(
                        request.tensor, None, max_elements=budget.max_tensor_numel
                    )
        if run is not None:
            run.model_identity = source.identity()
        preview_limit = min(request.preview_elements, budget.max_preview_elements)
        if preview_limit < request.preview_elements:
            warnings.append(
                f"preview limited to {budget.max_preview_elements} elements by the resource budget"
            )
        stats = basic_stats(values, storage_dtype=info.dtype, sampled=sampled)
        data = {
            "identity": source.identity(),
            "tensor": info,
            "resolved": resolved,
            "stats": stats,
            "preview": preview_array(values, max_elements=preview_limit, storage_dtype=info.dtype),
            "sampled": sampled,
            "fingerprint": fingerprint,
        }
        return ToolResult(data=data, warnings=warnings)

    def _read_tensor_slice(
        self, request: ReadTensorSliceInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        warnings: list[str] = []
        with open_checkpoint(request.path) as source:
            info = source.tensor_info(request.tensor)
            resolved = request.selection.resolve(info.shape)
            enforce_numel(resolved.numel, budget.max_slice_elements, what="tensor slice")
            values = source.read_tensor(
                request.tensor, request.selection, max_elements=budget.max_slice_elements
            )
            identity = source.identity()
        if run is not None:
            run.model_identity = identity
        preview_limit = min(request.preview_elements, budget.max_preview_elements)
        if preview_limit < request.preview_elements:
            warnings.append(
                f"preview limited to {budget.max_preview_elements} elements by the resource budget"
            )
        return ToolResult(
            data={
                "identity": identity,
                "tensor": info,
                "resolved": resolved,
                "stats": basic_stats(values, storage_dtype=info.dtype),
                "preview": preview_array(
                    values, max_elements=preview_limit, storage_dtype=info.dtype
                ),
            },
            warnings=warnings,
        )

    def _compare_tensors(
        self, request: CompareTensorsInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        with open_checkpoint(request.left_path) as left, open_checkpoint(
            request.right_path
        ) as right:
            left_info = left.tensor_info(request.tensor)
            right_name = request.right_tensor or request.tensor
            right_info = right.tensor_info(right_name)
            left_values = left.read_tensor(
                request.tensor, request.selection, max_elements=budget.max_slice_elements
            )
            right_values = right.read_tensor(
                right_name, request.selection, max_elements=budget.max_slice_elements
            )
            left_fingerprint = left.tensor_fingerprint(
                request.tensor, request.selection, max_elements=budget.max_slice_elements
            )
            right_fingerprint = right.tensor_fingerprint(
                right_name, request.selection, max_elements=budget.max_slice_elements
            )
            left_identity = left.identity()
            right_identity = right.identity()
        diff = compare_arrays(left_values, right_values, atol=request.atol, rtol=request.rtol)
        warnings: list[str] = []
        if left_info.dtype != right_info.dtype:
            warnings.append("tensor dtypes differ; numerical comparison uses float64 values")
        if run is not None:
            run.model_identity = left_identity
        return ToolResult(
            data={
                "left_identity": left_identity,
                "right_identity": right_identity,
                "left_tensor": left_info,
                "right_tensor": right_info,
                "left_fingerprint": left_fingerprint,
                "right_fingerprint": right_fingerprint,
                "diff": diff,
            },
            warnings=warnings,
        )

    def _analyze_tensor(
        self, request: AnalyzeTensorInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        requested = [item.lower() for item in request.operations]
        with open_checkpoint(request.path) as source:
            info = source.tensor_info(request.tensor)
            resolved = request.selection.resolve(info.shape)
            sampled = resolved.numel > budget.max_analysis_elements
            if sampled and any(item in {"svd", "rank"} for item in requested):
                raise ResourceLimitError(
                    "svd/rank require the full selected tensor and exceed the analysis budget",
                    details={"numel": resolved.numel, "limit": budget.max_analysis_elements},
                )
            values, resolved, sampled = self._read_values(
                source, request.tensor, request.selection, budget, sample_for_analysis=True
            )
            identity = source.identity()
        warnings = []
        if sampled:
            warnings.append(
                "analysis uses a deterministic stride sample; sampled operations are labeled in the result"
            )
        analysis = analyze_array(
            values,
            storage_dtype=info.dtype,
            operations=requested,
            budget=budget,
            quantiles=request.quantiles,
            bins=request.bins,
            top_k=request.top_k,
        )
        if run is not None:
            run.model_identity = identity
        return ToolResult(
            data={
                "identity": identity,
                "tensor": info,
                "resolved": resolved,
                "sampled": sampled,
                "analysis": analysis,
            },
            warnings=warnings,
        )

    def _compare_checkpoints(
        self, request: CompareCheckpointsInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        max_tensors = request.max_tensors or budget.max_compare_tensors
        enforce_numel(max_tensors, budget.max_compare_tensors, what="checkpoint comparison tensor count")
        with open_checkpoint(request.left_path) as left, open_checkpoint(
            request.right_path
        ) as right:
            comparison = compare_checkpoint_sources(
                left,
                right,
                max_tensors=max_tensors,
                max_elements=budget.max_slice_elements,
                sample_elements=budget.max_sample_elements,
                name_contains=request.name_contains,
                atol=request.atol,
                rtol=request.rtol,
            )
        if run is not None:
            run.model_identity = comparison.left
        return ToolResult(data={"comparison": comparison})

    # ------------------------------------------------------------------- patches

    def _externalize_payload(
        self,
        payload: PayloadRef,
        *,
        budget: Budget,
        externalize: bool,
    ) -> tuple[PayloadRef, list[ArtifactRef]]:
        if payload.artifact_id is not None or not externalize:
            return payload, []
        array = np.asarray(payload.values)
        if array.dtype == object or np.iscomplexobj(array):
            raise InputError("patch payload must be a real numeric array")
        if array.size <= 4096 and array.nbytes <= 64 * 1024:
            return payload, []
        if self.workspace is None:
            raise InputError(
                "large patch payloads require a workspace so they can be stored as artifacts",
                details={
                    "elements": int(array.size),
                    "hint": "run 'wsp workspace init' or set externalize_payloads=false with a smaller payload",
                },
            )
        ref = self.workspace.artifacts.put_array(
            "patch_payload",
            array.astype(np.float64),
            metadata={"shape": [int(d) for d in array.shape], "source": "create_patch"},
        )
        return (
            PayloadRef(
                artifact_id=ref.artifact_id,
                shape=[int(d) for d in array.shape],
                dtype="float64",
                numel=int(array.size),
                sha256=payload_sha256(array.astype(np.float64)),
            ),
            [ref],
        )

    def _externalize_params(
        self, params: PatchParams, *, budget: Budget, externalize: bool
    ) -> tuple[PatchParams, list[ArtifactRef]]:
        artifacts: list[ArtifactRef] = []
        if isinstance(params, ReplaceParams):
            payload, refs = self._externalize_payload(
                params.values, budget=budget, externalize=externalize
            )
            artifacts.extend(refs)
            return params.model_copy(update={"values": payload}), artifacts
        if isinstance(params, AddParams):
            payload, refs = self._externalize_payload(
                params.delta, budget=budget, externalize=externalize
            )
            artifacts.extend(refs)
            return params.model_copy(update={"delta": payload}), artifacts
        if isinstance(params, ScaleParams):
            return params, artifacts
        if isinstance(params, RankOneParams):
            u, refs_u = self._externalize_payload(params.u, budget=budget, externalize=externalize)
            v, refs_v = self._externalize_payload(params.v, budget=budget, externalize=externalize)
            artifacts.extend(refs_u + refs_v)
            return params.model_copy(update={"u": u, "v": v}), artifacts
        if isinstance(params, ProjectionParams):
            direction, refs = self._externalize_payload(
                params.direction, budget=budget, externalize=externalize
            )
            artifacts.extend(refs)
            return params.model_copy(update={"direction": direction}), artifacts
        raise UnsupportedError(f"unsupported patch operation: {type(params).__name__}")

    def _create_patch(
        self, request: CreatePatchInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        params, artifacts = self._externalize_params(
            request.params,
            budget=budget,
            externalize=request.externalize_payloads,
        )
        with open_checkpoint(request.path) as source:
            patch = create_patch(
                source,
                target=request.target,
                selection=request.selection,
                params=params,
                budget=budget,
                loader=self.workspace.artifact_loader if self.workspace else None,
                expected_fingerprint=request.expected_fingerprint,
                notes=request.notes,
                request_id=run.run_id if run else None,
            )
            if run is not None:
                run.model_identity = patch.source
        data: dict[str, Any] = {"patch": patch}
        warnings: list[str] = []
        if request.save:
            if self.workspace is None:
                warnings.append("patch was not persisted because no workspace is active")
            else:
                path = self.workspace.save_patch(patch)
                data["saved_patch_path"] = str(path)
        return ToolResult(data=data, artifacts=artifacts, warnings=warnings)

    def _preview_patch(
        self, request: PreviewPatchInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        if request.patch_id is not None:
            if self.workspace is None:
                raise InputError("preview_patch by patch_id requires a workspace")
            patch = self.workspace.load_patch(request.patch_id)
        else:
            patch = request.patch
        assert patch is not None
        with open_checkpoint(request.path) as source:
            preview = preview_patch_engine(
                source,
                patch,
                budget=budget,
                loader=self.workspace.artifact_loader if self.workspace else None,
            )
            if run is not None:
                run.model_identity = preview.source
        return ToolResult(data={"patch_id": patch.patch_id, "preview": preview})

    def _apply_patch(
        self, request: ApplyPatchInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        if request.patch_ids:
            if self.workspace is None:
                raise InputError("apply_patch by patch_ids requires a workspace")
            patches = [self.workspace.load_patch(patch_id) for patch_id in request.patch_ids]
        else:
            patches = request.patches
        with open_checkpoint(request.path) as source:
            patchset = build_patchset(
                source, patches, budget=budget, output_path=request.output_path
            )
            applied = apply_patchset(
                source,
                patchset,
                request.output_path,
                budget=budget,
                overwrite=request.overwrite,
                loader=self.workspace.artifact_loader if self.workspace else None,
            )
        warnings: list[str] = []
        data: dict[str, Any] = {"patchset": _compact_patchset(applied)}
        if request.save_patchset:
            if self.workspace is None:
                warnings.append("patch set was not persisted because no workspace is active")
            else:
                path = self.workspace.save_patchset(applied)
                data["saved_patchset_path"] = str(path)
        if run is not None:
            run.model_identity = applied.source
        return ToolResult(data=data, warnings=warnings)

    def _list_patches(
        self, request: ListPatchesInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        if self.workspace is None:
            raise InputError("list_patches requires a workspace")
        patches = self.workspace.list_patches()
        if request.source_fingerprint:
            patches = [
                patch
                for patch in patches
                if patch.source.structure_fingerprint == request.source_fingerprint
            ]
        total = len(patches)
        page = patches[: request.limit]
        return ToolResult(
            data={
                "total": total,
                "returned": len(page),
                "truncated": len(page) < total,
                "patches": [_compact_patch(patch) for patch in page],
            }
        )

    def _revert_patch(
        self, request: RevertPatchInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        if self.workspace is None:
            raise InputError("revert_patch requires a workspace")
        patchset = self.workspace.load_patchset(request.patchset_id)
        with open_checkpoint(request.path) as source:
            identity = revert_patchset(source, patchset)
        if run is not None:
            run.model_identity = identity
        return ToolResult(
            data={
                "patchset_id": patchset.patchset_id,
                "source": identity,
                "restored_by_source": True,
                "note": (
                    "The source checkpoint was never modified. This result identifies the "
                    "immutable source to use; no files were changed."
                ),
                "discarded_output_path": patchset.output_path,
            }
        )

    def _export_checkpoint(
        self, request: ExportCheckpointInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        identity = export_checkpoint_engine(
            request.path, request.output_path, overwrite=request.overwrite
        )
        if run is not None:
            run.model_identity = identity
        return ToolResult(
            data={"output_path": str(Path(request.output_path).expanduser().resolve()), "identity": identity}
        )

    # ---------------------------------------------------------------- artifacts

    def _list_artifacts(
        self, request: ListArtifactsInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        if self.workspace is None:
            raise InputError("list_artifacts requires a workspace")
        artifacts = self.workspace.artifacts.list()
        if request.kind:
            artifacts = [artifact for artifact in artifacts if artifact.kind == request.kind]
        total = len(artifacts)
        page = artifacts[: request.limit]
        return ToolResult(
            data={"total": total, "returned": len(page), "truncated": len(page) < total, "artifacts": page}
        )

    def _workspace_status(
        self, request: ToolInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        if self.workspace is None:
            raise InputError("workspace_status requires a workspace")
        return ToolResult(data=self.workspace.status())

    # ----------------------------------------------------------------- execution

    def _execute_builtin(
        self,
        model_spec: Any,
        *,
        inputs: dict[str, Any] | None,
        input_args: list[Any],
        budget: Budget,
        device: str | None,
        dtype: str | None,
        seed: int | None,
        capture_config: CaptureConfig | None,
    ) -> tuple[Any, ForwardExecution, torch.nn.Module]:
        model = build_builtin_model(model_spec, budget=budget)
        kwargs, args = prepare_builtin_inputs(model_spec, inputs, input_args)
        capture = None
        if capture_config is not None:
            capture = ActivationCapture(
                model,
                capture_config.module_names,
                record_inputs=capture_config.record_inputs,
                max_elements_per_tensor=capture_config.max_elements_per_tensor,
                max_elements_total=capture_config.max_elements_total,
                token_position=capture_config.token_position,
                budget=budget,
                include_values=capture_config.include_values,
            )
        execution = run_forward(
            model,
            inputs=kwargs,
            input_args=args,
            device=device,
            dtype=dtype,
            seed=seed,
            capture=capture,
            budget=budget,
        )
        return model, execution, model

    def _save_output_artifacts(
        self, execution: ForwardExecution, *, run_id: str
    ) -> tuple[dict[str, str], list[ArtifactRef]]:
        if self.workspace is None:
            return {}, []
        mapping: dict[str, str] = {}
        refs: list[ArtifactRef] = []
        for key, values in execution.output_values.items():
            safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in key)
            ref = self.workspace.artifacts.put_array(
                "forward_output",
                values,
                metadata={"run_id": run_id, "key": key, "safe_key": safe},
            )
            mapping[key] = ref.artifact_id
            refs.append(ref)
        return mapping, refs

    def _save_capture_artifact(
        self, capture: ActivationCapture | None, *, run_id: str
    ) -> tuple[ArtifactRef | None, list[ArtifactRef]]:
        if capture is None or self.workspace is None:
            return None, []
        payload = {
            "schema_version": "wsp.activation-capture/1.0",
            "capture_id": capture.capture_id,
            "run_id": run_id,
            "summaries": [summary.model_dump(mode="python") for summary in capture.summaries(include_values=True)],
        }
        ref = self.workspace.artifacts.put_json(
            "activation_capture", payload, metadata={"capture_id": capture.capture_id}
        )
        return ref, [ref]

    def _run_forward(
        self, request: RunForwardInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        model, execution, _ = self._execute_builtin(
            request.model,
            inputs=request.inputs,
            input_args=request.input_args,
            budget=budget,
            device=request.device,
            dtype=request.dtype,
            seed=request.seed,
            capture_config=request.capture,
        )
        identity = TorchModuleSource(model).identity()
        source_ref = TorchModuleSource(model).source_ref()
        if run is not None:
            run.model_identity = identity
            run.source = source_ref
            run.device = execution.result.device
            run.dtype = execution.result.dtype
            run.seed = request.seed
        warnings = list(execution.result.warnings)
        output_artifacts: dict[str, str] = {}
        artifacts: list[ArtifactRef] = []
        capture_artifact: ArtifactRef | None = None
        if request.save_artifacts:
            if self.workspace is None:
                warnings.append("output artifacts were not saved because no workspace is active")
            else:
                output_artifacts, artifacts = self._save_output_artifacts(
                    execution, run_id=execution.result.run_id
                )
                capture_artifact, capture_refs = self._save_capture_artifact(
                    execution.capture, run_id=execution.result.run_id
                )
                artifacts.extend(capture_refs)
        return ToolResult(
            data={
                "forward": execution.result,
                "output_artifacts": output_artifacts,
                "capture_artifact": capture_artifact.artifact_id if capture_artifact else None,
            },
            artifacts=artifacts,
            warnings=warnings,
        )

    def _capture_activations(
        self, request: CaptureActivationsInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        model, execution, _ = self._execute_builtin(
            request.model,
            inputs=request.inputs,
            input_args=request.input_args,
            budget=budget,
            device=request.device,
            dtype=request.dtype,
            seed=request.seed,
            capture_config=request.capture,
        )
        capture = execution.capture
        if capture is None:  # pragma: no cover - capture config is required
            raise InputError("activation capture was not initialized")
        identity = TorchModuleSource(model).identity()
        if run is not None:
            run.model_identity = identity
            run.source = TorchModuleSource(model).source_ref()
            run.device = execution.result.device
            run.dtype = execution.result.dtype
            run.seed = request.seed
        summaries = capture.summaries(include_values=True)
        artifacts: list[ArtifactRef] = []
        capture_artifact: ArtifactRef | None = None
        warnings = list(execution.result.warnings)
        if request.save_artifact:
            if self.workspace is None:
                warnings.append("capture was not saved because no workspace is active")
            else:
                capture_artifact, artifacts = self._save_capture_artifact(
                    capture, run_id=execution.result.run_id
                )
        return ToolResult(
            data={
                "forward": execution.result,
                "capture_id": capture.capture_id,
                "summaries": summaries,
                "capture_artifact": capture_artifact.artifact_id if capture_artifact else None,
            },
            artifacts=artifacts,
            warnings=warnings,
        )

    def _load_capture(self, artifact_id: str) -> list[ActivationSummary]:
        if self.workspace is None:
            raise InputError("activation comparison requires a workspace")
        ref, path = self.workspace.artifacts.get(artifact_id)
        if ref.kind != "activation_capture":
            raise ArtifactError(
                "artifact is not an activation capture",
                details={"artifact_id": artifact_id, "kind": ref.kind},
            )
        payload = self.workspace.artifacts.read_json(artifact_id)
        summaries = payload.get("summaries") if isinstance(payload, dict) else None
        if not isinstance(summaries, list):
            raise ArtifactError("activation capture artifact has no summaries")
        return [ActivationSummary.model_validate(item) for item in summaries]

    def _compare_activations(
        self, request: CompareActivationsInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        left = self._load_capture(request.left_capture_artifact)
        right = self._load_capture(request.right_capture_artifact)
        comparison = compare_activation_summaries(
            left, right, atol=request.atol, rtol=request.rtol
        )
        return ToolResult(data={"comparison": comparison})

    def _compare_forward(
        self, request: CompareForwardInput, budget: Budget, run: RunRecorder | None
    ) -> ToolResult:
        if self.workspace is None:
            raise InputError("forward comparison requires a workspace")
        left_record = self.workspace.load_run(request.left_run_id)
        right_record = self.workspace.load_run(request.right_run_id)
        if left_record.status != "succeeded" or right_record.status != "succeeded":
            raise InputError("both runs must have succeeded before they can be compared")

        def reconstruct(record: Any) -> ForwardExecution:
            result_payload = (record.result or {}).get("forward")
            if result_payload is None:
                raise ArtifactError(
                    "run record does not contain a forward result",
                    details={"run_id": record.run_id},
                )
            result = ForwardResult.model_validate(result_payload)
            artifact_map = (record.result or {}).get("output_artifacts") or {}
            values = {}
            for key, artifact_id in artifact_map.items():
                values[key] = self.workspace.artifacts.read_array(artifact_id)
            return ForwardExecution(result=result, output_values=values, capture=None)

        comparison = compare_forward_executions(
            reconstruct(left_record),
            reconstruct(right_record),
            atol=request.atol,
            rtol=request.rtol,
            compare_activations_too=request.compare_activations,
        )
        if request.compare_activations:
            left_capture_id = (left_record.result or {}).get("capture_artifact")
            right_capture_id = (right_record.result or {}).get("capture_artifact")
            if left_capture_id and right_capture_id:
                activation_comparison = compare_activation_summaries(
                    self._load_capture(left_capture_id),
                    self._load_capture(right_capture_id),
                    atol=request.atol,
                    rtol=request.rtol,
                )
                comparison = comparison.model_copy(
                    update={"activation_comparison": activation_comparison}
                )
        return ToolResult(data={"comparison": comparison})
