"""Tool handlers for activation comparison and investigation evidence."""

from __future__ import annotations

from typing import Any

import numpy as np

from wsp.behavior.ids import new_hypothesis_id
from wsp.behavior.models import InvestigationLink
from wsp.behavior.trace import TraceStore
from wsp.core.budget import Budget
from wsp.core.errors import ArtifactError, InputError
from wsp.core.time import utc_now_iso
from wsp.investigation.compare import (
    SCHEMA_COMPARISON,
    AlignmentSpec,
    CaptureComparison,
    PositionPair,
    compare_captures,
)
from wsp.investigation.evidence import (
    SCHEMA_EVIDENCE,
    ArtifactLink,
    CaptureRef,
    ComparisonRef,
    FindingRef,
    HypothesisNote,
    InvestigationEvidence,
    ReplayRef,
    build_evidence,
    evidence_checksum,
)
from wsp.investigation.models import SCHEMA_CAPTURE, TraceCaptureRecord
from wsp.tools.capture_handlers import CAPTURE_ARTIFACT_KIND, _load_capture
from wsp.tools.investigation_schemas import (
    AttachInvestigationEvidenceInput,
    CompareActivationCapturesInput,
    InspectInvestigationEvidenceInput,
)

COMPARISON_ARTIFACT_KIND = "capture_comparison"
EVIDENCE_ARTIFACT_KIND = "investigation_evidence"
_REPLAY_KINDS = {"replay_result": "replay", "replay_plan": "plan"}
_ENTRY_PAGE = 100


def _result(data: dict[str, Any], artifacts: list[Any] | None = None, warnings: list[str] | None = None):
    from wsp.tools.service import ToolResult

    return ToolResult(data=data, artifacts=list(artifacts or []), warnings=list(warnings or []))


def _require_workspace(service: Any) -> Any:
    if service.workspace is None:
        raise InputError(
            "this operation requires a workspace; run 'wsp init' or pass --workspace",
            details={"hint": "the workspace stores captures, comparisons, and investigation records"},
        )
    return service.workspace


def _load_comparison(workspace: Any, artifact_id: str) -> CaptureComparison:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != COMPARISON_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a capture comparison",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    comparison = CaptureComparison.model_validate(workspace.artifacts.read_json(artifact_id))
    if comparison.schema_version != SCHEMA_COMPARISON:
        raise ArtifactError(
            "unsupported comparison schema",
            details={"schema_version": comparison.schema_version},
        )
    return comparison


def _load_vectors(workspace: Any, record: TraceCaptureRecord) -> dict[tuple[str, str, str, int, int], np.ndarray]:
    """Read every retained vector a capture references, skipping missing ones."""

    vectors: dict[tuple[str, str, str, int, int], np.ndarray] = {}
    for measurement in record.measurements:
        if measurement.value_artifact_id is None:
            continue
        try:
            vector = workspace.artifacts.read_array(measurement.value_artifact_id)
        except ArtifactError:
            continue
        key = (
            measurement.module,
            measurement.tensor_role,
            measurement.tensor_name,
            measurement.call_index,
            measurement.position,
        )
        vectors[key] = np.asarray(vector)
    return vectors


def compare_activation_captures(
    service: Any, request: CompareActivationCapturesInput, budget: Budget, run: Any
):
    """Compare two stored captures under an explicit position alignment."""

    workspace = _require_workspace(service)
    left, left_ref = _load_capture(workspace, request.left_capture_artifact_id)
    right, right_ref = _load_capture(workspace, request.right_capture_artifact_id)

    alignment = AlignmentSpec(
        strategy=request.alignment.strategy,
        pairs=[
            PositionPair(
                left_position=pair.left_position,
                right_position=pair.right_position,
                left_token_index=pair.left_token_index,
                right_token_index=pair.right_token_index,
                note=pair.note,
            )
            for pair in request.alignment.pairs
        ],
        reviewed=request.alignment.reviewed,
        note=request.alignment.note,
    )
    # Vectors are resolved per capture: the two captures may have measured
    # different positions, so a single shared map would collide.
    comparison = _compare(
        left,
        right,
        left_vectors=_load_vectors(workspace, left),
        right_vectors=_load_vectors(workspace, right),
        alignment=alignment,
        request=request,
        budget=budget,
        left_artifact_id=left_ref.artifact_id,
        right_artifact_id=right_ref.artifact_id,
    )

    artifacts: list[Any] = []
    comparison_ref = None
    if request.save:
        comparison_ref = workspace.artifacts.put_json(
            COMPARISON_ARTIFACT_KIND,
            comparison.model_dump(mode="json"),
            metadata={
                "schema_version": comparison.schema_version,
                "comparison_id": comparison.comparison_id,
                "left_capture_id": comparison.left_capture_id,
                "right_capture_id": comparison.right_capture_id,
                "alignment_strategy": comparison.alignment.strategy,
                "metrics": list(comparison.metrics_requested),
                "run_id": getattr(run, "run_id", None),
            },
        )
        artifacts.append(comparison_ref)

    data = comparison.model_dump(mode="json")
    data["comparison_artifact_id"] = comparison_ref.artifact_id if comparison_ref else None
    data["entry_total"] = len(comparison.entries)
    data["entries"] = [entry.model_dump(mode="json") for entry in comparison.entries[:_ENTRY_PAGE]]
    data["entries_truncated"] = len(comparison.entries) > _ENTRY_PAGE
    data["left_capture_artifact_id"] = left_ref.artifact_id
    data["right_capture_artifact_id"] = right_ref.artifact_id
    return _result(data, artifacts, comparison.warnings)


def _compare(
    left: TraceCaptureRecord,
    right: TraceCaptureRecord,
    *,
    left_vectors: dict,
    right_vectors: dict,
    alignment: AlignmentSpec,
    request: CompareActivationCapturesInput,
    budget: Budget,
    left_artifact_id: str,
    right_artifact_id: str,
) -> CaptureComparison:
    return compare_captures(
        left,
        right,
        left_vectors=left_vectors,
        right_vectors=right_vectors,
        alignment=alignment,
        metrics=request.metrics,
        direction=request.direction,
        modules=request.modules,
        allow_different_models=request.allow_different_models,
        atol=request.atol,
        rtol=request.rtol,
        budget=budget,
        left_artifact_id=left_artifact_id,
        right_artifact_id=right_artifact_id,
    )


def _finding_ref(payload: dict[str, Any]) -> FindingRef:
    if not isinstance(payload, dict):
        raise InputError("each finding reference must be an object")
    data = dict(payload)
    if "detector_id" not in data or "kind" not in data or "trace_id" not in data:
        raise InputError(
            "a finding reference requires trace_id, detector_id, and kind",
            details={"received_keys": sorted(data)},
        )
    return FindingRef.model_validate(data)


def _artifact_link(workspace: Any, artifact_id: str, *, kinds: set[str], expected: str) -> ArtifactLink:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind not in kinds:
        raise ArtifactError(
            f"the referenced artifact is not a {expected}",
            details={"artifact_id": artifact_id, "kind": ref.kind, "expected_kinds": sorted(kinds)},
        )
    payload = workspace.artifacts.read_json(artifact_id)
    record_id = None
    trace_id = None
    if isinstance(payload, dict):
        for key in (
            "comparison_id",
            "capture_id",
            "alignment_id",
            "resolution_id",
            "replay_id",
            "plan_id",
            "evidence_id",
        ):
            if key in payload:
                record_id = payload[key]
                break
        trace_id = payload.get("trace_id")
    return ArtifactLink(
        artifact_id=artifact_id,
        record_id=record_id,
        trace_id=trace_id,
        schema_version=(payload.get("schema_version") if isinstance(payload, dict) else None),
    )


def _capture_ref(workspace: Any, artifact_id: str) -> CaptureRef:
    record, ref = _load_capture(workspace, artifact_id)
    return CaptureRef(
        capture_id=record.capture_id,
        artifact_id=ref.artifact_id,
        trace_id=record.trace_id,
        mode=record.mode,
        backend=record.backend,
        position_semantics=record.position_semantics,
        modules=list(record.modules),
        positions=list(record.positions),
        replay_id=record.replay_id,
    )


def _comparison_ref(workspace: Any, artifact_id: str) -> ComparisonRef:
    comparison = _load_comparison(workspace, artifact_id)
    return ComparisonRef(
        comparison_id=comparison.comparison_id,
        artifact_id=artifact_id,
        left_capture_id=comparison.left_capture_id,
        right_capture_id=comparison.right_capture_id,
        metrics=list(comparison.metrics_requested),
        alignment_strategy=comparison.alignment.strategy,
    )


def _replay_ref(workspace: Any, artifact_id: str) -> ReplayRef:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind not in _REPLAY_KINDS:
        raise ArtifactError(
            "the referenced artifact is not a replay plan or replay result",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    payload = workspace.artifacts.read_json(artifact_id)
    if not isinstance(payload, dict):
        raise ArtifactError("the replay artifact is not a JSON object", details={"artifact_id": artifact_id})
    if ref.kind == "replay_result":
        return ReplayRef(
            trace_id=payload.get("trace_id", ""),
            replay_id=payload.get("replay_id"),
            result_artifact_id=artifact_id,
            plan_id=(payload.get("metadata") or {}).get("plan_id"),
            numerical_agreement=payload.get("numerical_agreement"),
            evidence_level=payload.get("evidence_level"),
        )
    return ReplayRef(
        trace_id=payload.get("trace_id", ""),
        plan_id=payload.get("plan_id"),
        plan_artifact_id=artifact_id,
    )


def attach_investigation_evidence(
    service: Any, request: AttachInvestigationEvidenceInput, budget: Budget, run: Any
):
    """Assemble an investigation record and optionally attach it to a case."""

    workspace = _require_workspace(service)
    traces = TraceStore(workspace)

    case = None
    if request.case_id is not None:
        from wsp.behavior.cases import CaseStore

        case = CaseStore(workspace, traces=traces).load(request.case_id)

    findings = [_finding_ref(item) for item in request.finding_refs]
    for finding in findings:
        traces.get_record(finding.trace_id)

    replays = [_replay_ref(workspace, artifact_id) for artifact_id in request.replay_artifact_ids]
    alignments = [
        _artifact_link(workspace, artifact_id, kinds={"token_alignment"}, expected="token alignment")
        for artifact_id in request.alignment_artifact_ids
    ]
    spans = [
        _artifact_link(workspace, artifact_id, kinds={"span_resolution"}, expected="span resolution")
        for artifact_id in request.span_artifact_ids
    ]
    captures = [_capture_ref(workspace, artifact_id) for artifact_id in request.capture_artifact_ids]
    comparisons = [
        _comparison_ref(workspace, artifact_id) for artifact_id in request.comparison_artifact_ids
    ]

    if request.observation_trace_id is not None:
        traces.get_record(request.observation_trace_id)
    for trace_id in request.control_trace_ids:
        traces.get_record(trace_id)

    hypotheses = [
        HypothesisNote(
            hypothesis_id=new_hypothesis_id(),
            statement=item.statement,
            proposed_by=item.proposed_by,
            rationale=item.rationale,
            related_capture_ids=[capture for capture in item.related_capture_ids],
            related_comparison_ids=[comparison for comparison in item.related_comparison_ids],
        )
        for item in request.hypotheses
    ]

    evidence = build_evidence(
        title=request.title,
        case_id=request.case_id,
        summary=request.summary,
        observation_trace_id=request.observation_trace_id,
        control_trace_ids=list(request.control_trace_ids),
        findings=findings,
        replays=replays,
        alignments=alignments,
        spans=spans,
        captures=captures,
        comparisons=comparisons,
        hypotheses=hypotheses,
        notes=list(request.notes),
        warnings=_evidence_warnings(captures, comparisons),
        metadata=dict(request.metadata),
    )

    artifacts: list[Any] = []
    evidence_ref = None
    if request.save:
        evidence_ref = workspace.artifacts.put_json(
            EVIDENCE_ARTIFACT_KIND,
            evidence.model_dump(mode="json"),
            metadata={
                "schema_version": evidence.schema_version,
                "evidence_id": evidence.evidence_id,
                "case_id": evidence.case_id,
                "captures": len(evidence.captures),
                "comparisons": len(evidence.comparisons),
            },
        )
        artifacts.append(evidence_ref)
        evidence.checksums["artifact_sha256"] = evidence_ref.sha256

    link = None
    if case is not None and evidence_ref is not None:
        from wsp.behavior.cases import CaseStore

        store = CaseStore(workspace, traces=traces)
        now = utc_now_iso()
        link = InvestigationLink(
            evidence_id=evidence.evidence_id,
            artifact_id=evidence_ref.artifact_id,
            title=request.title,
            created_at=now,
            capture_ids=evidence.capture_ids,
            comparison_ids=evidence.comparison_ids,
            note=request.summary,
        )
        case.investigations.append(link)
        case.updated_at = now
        case.revision += 1
        store.save(case)

    data = evidence.model_dump(mode="json")
    data["evidence_artifact_id"] = evidence_ref.artifact_id if evidence_ref else None
    data["case_id"] = evidence.case_id
    data["attached_to_case"] = link is not None
    data["replay_capability"] = _replay_capability(captures)
    data["warnings"] = list(evidence.warnings)
    return _result(data, artifacts, evidence.warnings)


def _evidence_warnings(captures: list[CaptureRef], comparisons: list[ComparisonRef]) -> list[str]:
    warnings: list[str] = []
    for capture in captures:
        if capture.mode == "summary":
            warnings.append(
                f"capture '{capture.capture_id}' is a summary capture, so elementwise comparisons cannot use it"
            )
    if not comparisons and captures:
        warnings.append(
            "no comparison was attached; the record contains measurements without a reference condition"
        )
    return warnings


def _replay_capability(captures: list[CaptureRef]) -> str:
    if not captures:
        return "unavailable"
    if any(capture.mode == "selected" for capture in captures):
        return "reduced"
    return "reduced"


def inspect_investigation_evidence(
    service: Any, request: InspectInvestigationEvidenceInput, budget: Budget, run: Any
):
    """Read a stored investigation record and verify the artifacts it references."""

    workspace = _require_workspace(service)
    ref, _path = workspace.artifacts.get(request.artifact_id)
    if ref.kind != EVIDENCE_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not an investigation evidence record",
            details={"artifact_id": request.artifact_id, "kind": ref.kind},
        )
    payload = workspace.artifacts.read_json(request.artifact_id)
    evidence = InvestigationEvidence.model_validate(payload)
    if evidence.schema_version != SCHEMA_EVIDENCE:
        raise ArtifactError(
            "unsupported investigation evidence schema",
            details={"schema_version": evidence.schema_version},
        )

    recorded_checksum = evidence.checksums.get("evidence_sha256")
    integrity: dict[str, Any] = {
        "artifact_verified": True,
        "checksum_present": recorded_checksum is not None,
        # A redacted export drops checksums on purpose because the payload it
        # ships is not byte-identical to the original record.  Report that as
        # "not applicable" instead of implying tampering.
        "checksum_matches": (
            None if recorded_checksum is None else recorded_checksum == evidence_checksum(evidence)
        ),
        "missing_artifacts": [],
        "kind_mismatches": [],
    }
    referenced: list[str] = [item.artifact_id for item in evidence.alignments + evidence.spans]
    referenced += [item.artifact_id for item in evidence.captures]
    referenced += [item.artifact_id for item in evidence.comparisons]
    for replay in evidence.replays:
        for candidate in (replay.plan_artifact_id, replay.result_artifact_id):
            if candidate:
                referenced.append(candidate)
    for artifact_id in referenced:
        try:
            workspace.artifacts.get(artifact_id)
        except ArtifactError:
            integrity["missing_artifacts"].append(artifact_id)

    data = evidence.model_dump(mode="json")
    data["artifact"] = ref.model_dump(mode="json")
    data["integrity"] = integrity
    data["replay_capability"] = _replay_capability(evidence.captures)
    if request.include_records:
        data["capture_records"] = _capture_digests(workspace, evidence)
        data["comparison_records"] = _comparison_digests(workspace, evidence)
    return _result(data, [ref])


def _capture_digests(workspace: Any, evidence: InvestigationEvidence) -> list[dict[str, Any]]:
    digests: list[dict[str, Any]] = []
    for capture in evidence.captures:
        try:
            record, _ref = _load_capture(workspace, capture.artifact_id)
        except ArtifactError:
            continue
        digests.append(
            {
                "artifact_id": capture.artifact_id,
                "capture_id": record.capture_id,
                "schema_version": record.schema_version,
                "mode": record.mode,
                "backend": record.backend,
                "backend_version": record.backend_version,
                "positions": list(record.positions),
                "modules": list(record.modules),
                "measurement_total": len(record.measurements),
                "replay_verification": record.replay_verification,
                "resources": record.resources.model_dump(mode="json"),
            }
        )
    return digests


def _comparison_digests(workspace: Any, evidence: InvestigationEvidence) -> list[dict[str, Any]]:
    digests: list[dict[str, Any]] = []
    for comparison in evidence.comparisons:
        try:
            record = _load_comparison(workspace, comparison.artifact_id)
        except ArtifactError:
            continue
        digests.append(
            {
                "artifact_id": comparison.artifact_id,
                "comparison_id": record.comparison_id,
                "schema_version": record.schema_version,
                "alignment": record.alignment.model_dump(mode="json"),
                "metrics_requested": list(record.metrics_requested),
                "counts": dict(record.counts),
                "aggregates": [item.model_dump(mode="json") for item in record.aggregates],
                "compatibility": record.compatibility.model_dump(mode="json"),
                "limitations": list(record.limitations),
            }
        )
    return digests


def register_investigation_tools(service: Any) -> None:
    """Register the V1.2 comparison and evidence operations."""

    from wsp.tools.service import ToolDefinition

    def bind(function: Any) -> Any:
        def handler(request: Any, budget: Budget, run: Any):
            return function(service, request, budget, run)

        return handler

    definitions = [
        (
            "compare_activation_captures",
            "Compare two stored activation captures under an explicit position alignment and report numerical metrics only.",
            CompareActivationCapturesInput,
            compare_activation_captures,
            "A comparison record with per-position metrics, aggregates, compatibility checks, and limitations.",
            True,
        ),
        (
            "attach_investigation_evidence",
            "Assemble an investigation record from stored replays, alignments, span resolutions, captures, and comparisons, and optionally attach it to a failure case.",
            AttachInvestigationEvidenceInput,
            attach_investigation_evidence,
            "An investigation evidence record with validated references and its stored artifact.",
            True,
        ),
        (
            "inspect_investigation_evidence",
            "Read a stored investigation record and verify the artifacts it references.",
            InspectInvestigationEvidenceInput,
            inspect_investigation_evidence,
            "The evidence record with reference integrity and measurement digests.",
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
