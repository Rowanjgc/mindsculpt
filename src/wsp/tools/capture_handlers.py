"""Tool handlers for targeted activation capture."""

from __future__ import annotations

from typing import Any

from wsp.alignment.models import SCHEMA_SPAN, SpanResolution
from wsp.alignment.spans import resolve_span
from wsp.behavior.trace import TraceStore
from wsp.core.budget import Budget
from wsp.core.errors import ArtifactError, InputError
from wsp.investigation.points import CAPTURE_KIND_LABELS
from wsp.core.models import ArtifactRef
from wsp.investigation.capture import capture_checksum, position_targets, run_capture
from wsp.investigation.models import SCHEMA_CAPTURE, TraceCaptureRecord
from wsp.investigation.points import discover_points, torch_module_of
from wsp.replay.plan import prepare_plan
from wsp.replay.sources import load_replay_source
from wsp.tools.alignment_handlers import (
    ALIGNMENT_ARTIFACT_KIND,
    SPAN_ARTIFACT_KIND,
    _alignment_for,
)
from wsp.tools.capture_schemas import (
    CaptureTraceActivationsInput,
    InspectTraceCaptureInput,
    ListCapturePointsInput,
)

CAPTURE_ARTIFACT_KIND = "trace_capture"
PLAN_ARTIFACT_KIND = "replay_plan"
_MEASUREMENT_PAGE = 200


def _result(data: dict[str, Any], artifacts: list[Any] | None = None, warnings: list[str] | None = None):
    from wsp.tools.service import ToolResult

    return ToolResult(data=data, artifacts=list(artifacts or []), warnings=list(warnings or []))


def _require_workspace(service: Any) -> Any:
    if service.workspace is None:
        raise InputError(
            "this operation requires an initialized workspace; create one with 'wsp init --path <dir>', or restart 'wsp mcp' with --auto-init",
            details={"hint": "the workspace stores traces and capture evidence"},
        )
    return service.workspace


def list_capture_points(service: Any, request: ListCapturePointsInput, budget: Budget, run: Any):
    """Enumerate module paths that can be targeted by a capture request."""

    if request.instrumentation == "nnterp":
        from wsp.investigation.nnsight_backend import (
            capture_backend_capabilities,
            load_nnterp_model,
        )

        source = request.model
        if source.kind != "local_hf":
            raise InputError(
                "the nnterp backend needs a local Hugging Face checkpoint",
                details={
                    "kind": source.kind,
                    "hint": "set model.kind to 'local_hf' and model.path to a local checkpoint directory",
                },
            )
        assert source.path is not None
        model = load_nnterp_model(
            source.path,
            device=source.device,
            dtype=source.dtype,
            compute_file_sha256=source.compute_file_sha256,
        )
        listing = model.capture_points(limit=request.limit, budget=budget)
        data = listing.model_dump(mode="json")
        data["points"] = [
            point
            for point in data["points"]
            if not request.kinds or point["kind"] in set(request.kinds)
        ]
        if request.kinds and not data["points"]:
            data["notes"] = list(data.get("notes") or []) + [
                "no capture points matched kinds "
                + ", ".join(sorted(set(request.kinds)))
                + "; canonical kind labels are "
                + ", ".join(CAPTURE_KIND_LABELS)
            ]
        data["kind_labels"] = list(CAPTURE_KIND_LABELS)
        data["source"] = request.model.describe()
        data["instrumentation"] = capture_backend_capabilities()
        return _result(data)

    model = load_replay_source(request.model)
    listing = discover_points(
        torch_module_of(model),
        include_submodules=request.include_submodules,
        kinds=request.kinds,
        limit=request.limit,
        budget=budget,
    )
    data = listing.model_dump(mode="json")
    data["kind_labels"] = list(CAPTURE_KIND_LABELS)
    data["source"] = request.model.describe()
    data["instrumentation"] = {
        "backend": "torch_hooks",
        "available": True,
        "caveats": [
            "paths are the checkpoint's own module names, so they differ between architectures"
        ],
    }
    return _result(data)


def _load_span(workspace: Any, artifact_id: str, trace_id: str) -> SpanResolution:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != SPAN_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a span resolution",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    span = SpanResolution.model_validate(workspace.artifacts.read_json(artifact_id))
    if span.schema_version != SCHEMA_SPAN:
        raise ArtifactError(
            "unsupported span resolution schema",
            details={"schema_version": span.schema_version},
        )
    if span.trace_id != trace_id:
        raise InputError(
            "the stored span resolution belongs to a different trace",
            details={"span_trace": span.trace_id, "requested": trace_id},
        )
    return span


def _requested_tokens(
    request: CaptureTraceActivationsInput,
    *,
    workspace: Any,
    alignment: Any,
    trace: Any,
    budget: Budget,
    warnings: list[str],
) -> tuple[list[int], list[Any], str | None]:
    if request.span_artifact_id is not None:
        span = _load_span(workspace, request.span_artifact_id, request.trace_id)
        if span.status == "unresolved":
            raise InputError(
                "the referenced span could not be resolved onto token positions",
                details={"reason": span.ambiguity or "unresolved span"},
            )
        if span.status == "ambiguous":
            warnings.append(
                "the referenced span is ambiguous; only the resolved overlap is measured"
            )
        return list(span.full_token_indices), [span], request.span_artifact_id

    if request.role is not None:
        resolution = resolve_span(
            alignment,
            kind="token_range",
            start=request.role_start,
            end=request.role_end,
            role=request.role,
            prompt_text=trace.input.text,
            generation_text=trace.text,
            budget=budget,
        )
        if resolution.status == "unresolved":
            raise InputError(
                "the requested role range does not cover any token position",
                details={"reason": resolution.ambiguity},
            )
        warnings.extend(resolution.warnings)
        return list(resolution.full_token_indices), [resolution], resolution.resolution_id

    assert request.positions is not None
    return [int(index) for index in request.positions], [], None


def _instrument(
    request: CaptureTraceActivationsInput,
    *,
    budget: Budget,
    sequence_length: int,
    positions: list[int],
) -> tuple[Any, Any, list[str]]:
    """Resolve the requested instrumentation backend.

    The backend is never substituted silently: asking for ``nnterp`` either
    yields an nnterp capture (with its identity recorded in the evidence) or a
    clear error explaining why the optional backend is unavailable.
    """

    warnings: list[str] = []
    if request.instrumentation == "torch_hooks":
        return load_replay_source(request.model), None, warnings

    from wsp.investigation.nnsight_backend import NnterpCapture, load_nnterp_model

    source = request.model
    if source.kind != "local_hf":
        raise InputError(
            "the nnterp backend needs a local Hugging Face checkpoint",
            details={
                "kind": source.kind,
                "hint": "set model.kind to 'local_hf' and model.path to a local checkpoint directory",
            },
        )
    assert source.path is not None
    model = load_nnterp_model(
        source.path,
        device=source.device,
        dtype=source.dtype,
        compute_file_sha256=source.compute_file_sha256,
    )
    capture = NnterpCapture(
        model,
        request.modules,
        positions,
        sequence_length=sequence_length,
        mode=request.mode,
        record_inputs=request.include_inputs,
        budget=budget,
    )
    warnings.append(
        "activations were captured through the optional nnterp backend; the evidence records "
        "backend='nnterp' together with the nnterp and nnsight versions"
    )
    return model, capture, warnings


def capture_trace_activations(
    service: Any, request: CaptureTraceActivationsInput, budget: Budget, run: Any
):
    """Replay a recorded trace and measure selected internal states."""

    workspace = _require_workspace(service)
    traces = TraceStore(workspace)
    trace = traces.load(request.trace_id)
    record = traces.get_record(request.trace_id)
    alignment, alignment_artifacts, warnings = _alignment_for(workspace, request, budget)

    requested_tokens, resolutions, span_id = _requested_tokens(
        request,
        workspace=workspace,
        alignment=alignment,
        trace=trace,
        budget=budget,
        warnings=warnings,
    )
    sequence_length = alignment.full_length
    if len(requested_tokens) > int(budget.max_capture_positions):
        raise InputError(
            "too many token positions requested for one capture",
            details={"requested": len(requested_tokens), "limit": int(budget.max_capture_positions)},
        )
    positions, dropped, mapping_warnings = position_targets(
        requested_tokens, request.position_semantics, sequence_length=sequence_length
    )
    warnings.extend(mapping_warnings)

    plan = prepare_plan(
        trace,
        trace_artifact_id=record.artifact.artifact_id,
        trace_artifact_sha256=record.artifact.sha256,
        execution=request.execution,
        tolerances=request.tolerances,
        budget=budget,
    )
    artifacts: list[ArtifactRef] = list(alignment_artifacts)
    if request.save:
        artifacts.append(
            workspace.artifacts.put_json(
                PLAN_ARTIFACT_KIND,
                plan.model_dump(mode="json"),
                metadata={
                    "schema_version": plan.schema_version,
                    "plan_id": plan.plan_id,
                    "trace_id": plan.trace_id,
                    "semantics": plan.semantics,
                },
            )
        )

    model, capture_object, backend_warnings = _instrument(
        request,
        budget=budget,
        sequence_length=sequence_length,
        positions=positions,
    )
    warnings.extend(backend_warnings)
    outcome = run_capture(
        plan,
        model,
        alignment=alignment,
        alignment_artifact_id=(
            alignment_artifacts[0].artifact_id if alignment_artifacts else request.alignment_artifact_id
        ),
        span_artifact_id=span_id,
        requested_tokens=requested_tokens,
        positions=positions,
        modules=request.modules,
        mode=request.mode,
        semantics=request.position_semantics,
        include_inputs=request.include_inputs,
        first_call_only=request.first_call_only,
        store=workspace.artifacts if request.save else None,
        budget=budget,
        warnings=warnings,
        capture=capture_object,
        instrumentation=request.instrumentation,
    )
    capture_record = outcome.record
    artifacts.extend(outcome.artifacts)

    # A replay result is stored alongside the capture so the verification status
    # is independently inspectable.
    result_ref = None
    if request.save:
        result_ref = workspace.artifacts.put_json(
            "replay_result",
            outcome.result.model_dump(mode="json"),
            metadata={
                "schema_version": outcome.result.schema_version,
                "replay_id": outcome.result.replay_id,
                "trace_id": outcome.result.trace_id,
                "numerical_agreement": outcome.result.numerical_agreement,
                "evidence_level": outcome.result.evidence_level,
            },
        )
        artifacts.append(result_ref)
        capture_record.replay_artifact_id = result_ref.artifact_id
        capture_record.checksums["measurement_sha256"] = capture_checksum(capture_record)

    record_ref = None
    if request.save:
        record_ref = workspace.artifacts.put_json(
            CAPTURE_ARTIFACT_KIND,
            capture_record.model_dump(mode="json"),
            metadata={
                "schema_version": capture_record.schema_version,
                "capture_id": capture_record.capture_id,
                "trace_id": capture_record.trace_id,
                "replay_id": capture_record.replay_id,
                "mode": capture_record.mode,
                "position_semantics": capture_record.position_semantics,
                "backend": capture_record.backend,
            },
        )
        artifacts.append(record_ref)

    measurements = capture_record.measurements[:_MEASUREMENT_PAGE]
    payload_measurements = []
    for measurement in measurements:
        item = measurement.model_dump(mode="json")
        if request.include_values and measurement.value_artifact_id is not None:
            vector = workspace.artifacts.read_array(measurement.value_artifact_id)
            limit = int(budget.max_preview_elements)
            item["values"] = [float(value) for value in vector.reshape(-1)[:limit]]
            if vector.size > limit:
                item["values_truncated"] = True
        payload_measurements.append(item)

    data: dict[str, Any] = {
        "capture_id": capture_record.capture_id,
        "capture_artifact_id": record_ref.artifact_id if record_ref else None,
        "schema_version": capture_record.schema_version,
        "trace_id": capture_record.trace_id,
        "replay_id": capture_record.replay_id,
        "replay_artifact_id": capture_record.replay_artifact_id,
        "plan_id": capture_record.plan_id,
        "alignment_id": capture_record.alignment_id,
        "alignment_artifact_id": capture_record.alignment_artifact_id,
        "span_artifact_id": capture_record.span_artifact_id,
        "mode": capture_record.mode,
        "position_semantics": capture_record.position_semantics,
        "position_semantics_note": capture_record.position_semantics_note,
        "requested_tokens": capture_record.requested_tokens,
        "dropped_tokens": dropped,
        "positions": capture_record.positions,
        "modules": capture_record.modules,
        "include_inputs": capture_record.include_inputs,
        "first_call_only": capture_record.first_call_only,
        "sequence_length": capture_record.sequence_length,
        "measurement_total": len(capture_record.measurements),
        "measurements": payload_measurements,
        "measurements_truncated": len(capture_record.measurements) > len(payload_measurements),
        "modules_summary": [item.model_dump(mode="json") for item in capture_record.modules_summary],
        "resources": capture_record.resources.model_dump(mode="json"),
        "checksums": capture_record.checksums,
        "replay_verification": {
            "numerical_agreement": capture_record.replay_verification.get("numerical_agreement"),
            "evidence_level": capture_record.replay_verification.get("evidence_level"),
            "forward_executed": capture_record.replay_verification.get("forward_executed"),
            "generation_reproducibility": capture_record.replay_verification.get(
                "generation_reproducibility"
            ),
        },
        "warnings": capture_record.warnings,
        "limitations": capture_record.limitations,
        "unavailable": capture_record.unavailable,
        "privacy": capture_record.privacy,
    }
    if resolutions:
        data["span_resolutions"] = [item.model_dump(mode="json") for item in resolutions]
    return _result(data, artifacts, capture_record.warnings)


def _load_capture(workspace: Any, artifact_id: str) -> tuple[TraceCaptureRecord, ArtifactRef]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != CAPTURE_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a trace capture",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    record = TraceCaptureRecord.model_validate(workspace.artifacts.read_json(artifact_id))
    if record.schema_version != SCHEMA_CAPTURE:
        raise ArtifactError(
            "unsupported capture schema",
            details={"schema_version": record.schema_version},
        )
    return record, ref


def inspect_activation_capture(
    service: Any, request: InspectTraceCaptureInput, budget: Budget, run: Any
):
    """Read a stored capture without re-running the model."""

    workspace = _require_workspace(service)
    record, ref = _load_capture(workspace, request.artifact_id)
    page = record.measurement_page(
        offset=request.measurement_offset, limit=request.measurement_limit
    )
    measurements = []
    missing_vectors: list[str] = []
    for measurement in page:
        item = measurement.model_dump(mode="json")
        if request.include_values and measurement.value_artifact_id is not None:
            try:
                vector = workspace.artifacts.read_array(measurement.value_artifact_id)
            except ArtifactError:
                missing_vectors.append(measurement.value_artifact_id)
            else:
                limit = int(budget.max_preview_elements)
                item["values"] = [float(value) for value in vector.reshape(-1)[:limit]]
                if vector.size > limit:
                    item["values_truncated"] = True
        measurements.append(item)
    data = record.model_dump(mode="json")
    data["measurements"] = measurements
    data["measurement_offset"] = request.measurement_offset
    data["measurement_limit"] = request.measurement_limit
    data["measurement_total"] = len(record.measurements)
    data["measurements_truncated"] = (
        request.measurement_offset + len(page) < len(record.measurements)
    )
    data["artifact"] = ref.model_dump(mode="json")
    data["integrity"] = {
        "artifact_verified": True,
        "missing_vectors": missing_vectors,
    }
    return _result(data, [ref])


def register_capture_tools(service: Any) -> None:
    """Register the V1.2 activation-capture operations."""

    from wsp.tools.service import ToolDefinition

    def bind(function: Any) -> Any:
        def handler(request: Any, budget: Budget, run: Any):
            return function(service, request, budget, run)

        return handler

    definitions = [
        (
            "list_capture_points",
            "List module paths in a local model that can be targeted by an activation capture, classified by architectural role. "
            "Pass instrumentation='nnterp' for normalised layer paths from the optional nnterp backend.",
            ListCapturePointsInput,
            list_capture_points,
            "Bounded module listing with paths, class names, kinds, and parameter counts.",
            False,
        ),
        (
            "capture_trace_activations",
            "Replay a recorded trace and measure selected internal states at selected token positions, either as bounded statistics or as retained vectors. "
            "The optional nnterp instrumentation backend must be requested explicitly and is recorded in the resulting evidence.",
            CaptureTraceActivationsInput,
            capture_trace_activations,
            "A capture record with positions, measurements, resource accounting, replay verification status, and artifact references.",
            True,
        ),
        (
            "inspect_activation_capture",
            "Read a stored activation capture, optionally including retained values.",
            InspectTraceCaptureInput,
            inspect_activation_capture,
            "The capture record with a bounded measurement page.",
            True,
        ),
    ]
    for name, description, schema, function, output, needs_workspace in definitions:
        service.register_tool(
            ToolDefinition(
                name=name,
                description=description,
                input_model=schema,
                handler=bind(function),
                output_description=output,
                requires_workspace=needs_workspace,
            )
        )
