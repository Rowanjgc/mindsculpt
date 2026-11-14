"""Tool handlers for traces, detectors, contrasts, evaluations, and cases.

The functions are thin adapters over the behavioral modules.  They share one
implementation between the Python API and the CLI, and every input is validated
by the schemas in ``wsp.tools.behavior_schemas``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wsp.adapters.inif import export_inif, import_inif_document
from wsp.adapters.inspect_eval import import_inspect_log
from wsp.behavior.backends import build_backend
from wsp.behavior.cases import (
    CaseStore,
    build_export,
    case_sha256,
    import_export,
    redact_trace,
)
from wsp.behavior.compare import compare_traces as compare_trace_records
from wsp.behavior.contrast import run_contrast as run_contrast_test
from wsp.behavior.detectors import default_registry, scan_trace as run_scan
from wsp.behavior.evaluators import default_evaluators
from wsp.behavior.ids import (
    new_annotation_id,
    new_case_id,
    new_contrast_id,
    new_hypothesis_id,
)
from wsp.behavior.models import (
    CaseAnnotation,
    CaseHypothesis,
    CaseReview,
    ContrastTest,
    FailureCase,
    GenerationParams,
    GenerationRequest,
    GenerationTrace,
    ModelRef,
    PrivacyInfo,
)
from wsp.behavior.trace import TraceStore, finalize_trace
from wsp.core.budget import Budget
from wsp.core.errors import InputError, NotFoundError, UnsupportedError
from wsp.core.hashing import sha256_bytes, sha256_file
from wsp.core.json import atomic_write_bytes, read_json
from wsp.core.time import utc_now_iso
from wsp.tools.behavior_schemas import (
    CompareTracesInput,
    CreateFailureCaseInput,
    EvaluateBehaviorInput,
    ExportFailureCaseInput,
    ExportTraceInput,
    ImportFailureCaseInput,
    ImportTraceInput,
    InspectFailureCaseInput,
    InspectTraceInput,
    ListDetectorsInput,
    ListEvaluatorsInput,
    ListFailureCasesInput,
    ListTracesInput,
    RecordTraceInput,
    RunContrastInput,
    ScanTraceInput,
    UpdateFailureCaseInput,
)

_TEXT_PREVIEW_CHARS = 4000


def _result(data: dict[str, Any], artifacts: list[Any] | None = None, warnings: list[str] | None = None):
    from wsp.tools.service import ToolResult

    return ToolResult(data=data, artifacts=list(artifacts or []), warnings=list(warnings or []))


def _require_workspace(service: Any) -> None:
    if service.workspace is None:
        raise InputError(
            "this operation requires a workspace; run 'wsp init' or pass --workspace",
            details={"hint": "the workspace stores traces, evidence, and cases"},
        )


def _traces(service: Any) -> TraceStore:
    _require_workspace(service)
    return TraceStore(service.workspace)


def _cases(service: Any, traces: TraceStore | None = None) -> CaseStore:
    _require_workspace(service)
    return CaseStore(service.workspace, traces=traces)


def _preview(text: str | None) -> tuple[str | None, bool]:
    if text is None:
        return None, False
    if len(text) <= _TEXT_PREVIEW_CHARS:
        return text, False
    return text[:_TEXT_PREVIEW_CHARS], True


def _trace_compact(trace: GenerationTrace) -> dict[str, Any]:
    payload = trace.model_dump(mode="json")
    payload.pop("tokens", None)
    payload["token_count"] = len(trace.tokens)
    return payload


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def record_trace(service: Any, request: RecordTraceInput, budget: Budget, run: Any):
    traces = _traces(service)
    backend = build_backend(request.backend, budget=budget)
    trace = backend.generate(
        GenerationRequest(
            input_text=request.input_text,
            system=request.system,
            params=request.params,
            seed=request.seed,
        ),
        budget=budget,
    )
    if request.input_artifact_id is not None:
        if service.workspace is None:
            raise InputError("input artifact references require a workspace")
        ref, _path = service.workspace.artifacts.get(request.input_artifact_id)
        trace.input.artifact_id = ref.artifact_id
        trace.input.sha256 = ref.sha256
    trace.privacy = PrivacyInfo(**request.privacy.model_dump(), redacted=False)
    trace = finalize_trace(trace, budget)
    record = traces.save(trace)
    return _result(
        {
            "trace_id": trace.trace_id,
            "summary": record.model_dump(mode="json"),
            "trace": _trace_compact(trace),
        },
        artifacts=[record.artifact],
        warnings=trace.warnings,
    )


def import_trace(service: Any, request: ImportTraceInput, budget: Budget, run: Any):
    path = Path(request.path).expanduser().resolve()
    warnings: list[str] = []
    traces: list[GenerationTrace] = []
    if request.format == "wsp":
        if not path.exists() or not path.is_file():
            raise NotFoundError(f"trace file was not found: {path}")
        try:
            payload = read_json(path)
            trace = GenerationTrace.model_validate(payload)
        except Exception as exc:
            raise InputError(f"file is not a valid WSP trace: {exc}") from exc
        traces = [trace]
    elif request.format == "inif":
        if path.suffix == ".inif":
            raise UnsupportedError(
                "indexed .inif archives are not supported; export the document as .inif.json first",
                details={"path": str(path)},
            )
        if not path.exists() or not path.is_file():
            raise NotFoundError(f"INIF document was not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InputError(f"could not read INIF document: {exc}") from exc
        traces, warnings = import_inif_document(
            payload,
            budget=budget,
            source_ref=str(path),
            source_sha256=sha256_file(path),
        )
    elif request.format == "inspect":
        traces, warnings = import_inspect_log(
            path,
            budget=budget,
            sample_ids=request.sample_ids or None,
        )
    else:  # pragma: no cover - schema constrains this
        raise InputError(f"unknown trace import format {request.format!r}")

    if request.format != "inspect":
        if request.format == "inif" and request.sample_ids:
            wanted = set(request.sample_ids)
            traces = [
                trace
                for trace in traces
                if str(trace.metadata.get("inif_sample_id")) in wanted
            ]
            if not traces:
                raise InputError("none of the requested sample ids were present in the document")
    artifacts = []
    summaries = []
    persisted: list[GenerationTrace] = []
    if request.persist:
        store = _traces(service)
        for trace in traces:
            record = store.save(trace)
            artifacts.append(record.artifact)
            summaries.append(record.model_dump(mode="json"))
            persisted.append(trace)
    for trace in traces:
        warnings.extend(trace.warnings)
    warnings = _dedupe(warnings)
    data: dict[str, Any] = {
        "format": request.format,
        "imported": len(traces),
        "persisted": request.persist,
        "traces": summaries if request.persist else [trace.model_dump(mode="json") for trace in traces],
        "warnings": warnings,
    }
    if request.persist:
        data["trace_ids"] = [trace.trace_id for trace in traces]
    return _result(data, artifacts=artifacts, warnings=warnings)


def inspect_trace(service: Any, request: InspectTraceInput, budget: Budget, run: Any):
    trace = _traces(service).load(request.trace_id)
    tokens = []
    if request.include_tokens:
        page = trace.tokens[request.token_offset : request.token_offset + request.token_limit]
        tokens = [
            {
                "index": request.token_offset + index,
                **token.model_dump(mode="json"),
            }
            for index, token in enumerate(page)
        ]
    text, text_truncated = _preview(trace.text) if request.include_text else (None, False)
    input_text, input_truncated = (
        _preview(trace.input.text) if request.include_text else (None, False)
    )
    data: dict[str, Any] = {
        "schema_version": trace.schema_version,
        "trace_id": trace.trace_id,
        "created_at": trace.created_at,
        "model": trace.model.model_dump(mode="json"),
        "tokenizer": trace.tokenizer.model_dump(mode="json") if trace.tokenizer else None,
        "backend": trace.backend.model_dump(mode="json"),
        "params": trace.params.model_dump(mode="json"),
        "seed": trace.seed,
        "input": {
            **trace.input.model_dump(mode="json"),
            "text": input_text,
            "text_truncated": input_truncated,
        },
        "text": text,
        "text_truncated": text_truncated,
        "token_count": len(trace.tokens),
        "tokens_returned": len(tokens),
        "tokens_truncated": request.token_offset + len(tokens) < len(trace.tokens),
        "tokens": tokens,
        "termination": trace.termination.model_dump(mode="json"),
        "timing": trace.timing.model_dump(mode="json"),
        "annotations": [annotation.model_dump(mode="json") for annotation in trace.annotations],
        "tool_calls": [call.model_dump(mode="json") for call in trace.tool_calls],
        "expected_answers": trace.expected_answers,
        "activation_captures": [ref.model_dump(mode="json") for ref in trace.activation_captures],
        "provenance": trace.provenance.model_dump(mode="json"),
        "privacy": trace.privacy.model_dump(mode="json"),
        "checksums": trace.checksums,
        "unavailable": trace.unavailable,
        "warnings": trace.warnings,
    }
    if request.include_metadata:
        data["metadata"] = trace.metadata
    return _result(data)


def list_traces(service: Any, request: ListTracesInput, budget: Budget, run: Any):
    summaries = _traces(service).list(limit=request.limit)
    return _result(
        {
            "returned": len(summaries),
            "traces": [summary.model_dump(mode="json") for summary in summaries],
        }
    )


def export_trace(service: Any, request: ExportTraceInput, budget: Budget, run: Any):
    traces = _traces(service)
    trace = traces.load(request.trace_id)
    warnings: list[str] = []
    if request.redact_text:
        trace = redact_trace(trace)
        warnings.append("input and output text were redacted")
    if request.format == "inif":
        payload = export_inif(trace)
        warnings.extend(
            payload.get("metadata", {}).get("extra", {}).get("wsp_export_warnings", [])
        )
    else:
        payload = trace.model_dump(mode="json")
    from wsp.core.json import dumps

    data = dumps(payload).encode("utf-8")
    output = Path(request.output_path).expanduser().resolve()
    atomic_write_bytes(output, data, overwrite=request.overwrite)
    return _result(
        {
            "trace_id": request.trace_id,
            "format": request.format,
            "output_path": str(output),
            "sha256": sha256_bytes(data),
            "nbytes": len(data),
            "redacted": request.redact_text,
        },
        warnings=_dedupe(warnings),
    )


def scan_trace(service: Any, request: ScanTraceInput, budget: Budget, run: Any):
    trace = _traces(service).load(request.trace_id)
    result = run_scan(
        trace,
        budget=budget,
        selections=request.detectors or None,
    )
    findings = [
        finding.model_dump(mode="json")
        for detector_run in result.runs
        for finding in detector_run.findings
    ]
    return _result(
        {
            "scan": result.model_dump(mode="json"),
            "findings": findings,
        },
        warnings=result.warnings,
    )


def list_detectors(service: Any, request: ListDetectorsInput, budget: Budget, run: Any):
    return _result({"detectors": default_registry().describe()})


def compare_traces(service: Any, request: CompareTracesInput, budget: Budget, run: Any):
    store = _traces(service)
    left = store.load(request.left_trace_id)
    right = store.load(request.right_trace_id)
    return _result(compare_trace_records(left, right))


def run_contrast(service: Any, request: RunContrastInput, budget: Budget, run: Any):
    workspace = service.workspace
    _require_workspace(service)
    traces = TraceStore(workspace)
    test = ContrastTest(
        test_id=new_contrast_id(),
        name=request.name,
        description=request.description,
        baseline=request.baseline,
        variants=request.variants,
        expectation=request.expectation,
        backend=request.backend,
        params=request.params,
        seed=request.seed,
        tags=request.tags,
        metadata=request.metadata,
    )
    backend = build_backend(test.backend, budget=budget)
    result = run_contrast_test(
        test,
        backend=backend,
        trace_store=traces,
        budget=budget,
    )
    ref = workspace.artifacts.put_json(
        "contrast_result",
        result.model_dump(mode="json"),
        metadata={
            "result_id": result.result_id,
            "test_id": test.test_id,
            "test_name": test.name,
            "overall_status": result.overall_status,
        },
    )
    data = result.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id
    data["artifact_sha256"] = ref.sha256
    return _result(data, artifacts=[ref], warnings=result.warnings)


def evaluate_behavior(service: Any, request: EvaluateBehaviorInput, budget: Budget, run: Any):
    trace = None
    text = request.text
    if request.trace_id is not None:
        trace = _traces(service).load(request.trace_id)
        text = trace.text
    registry = default_evaluators()
    outcome = registry.evaluate(request.evaluator, text=text, trace=trace)
    artifacts = []
    if service.workspace is not None:
        ref = service.workspace.artifacts.put_json(
            "evaluation",
            outcome.model_dump(mode="json"),
            metadata={
                "evaluation_id": outcome.evaluation_id,
                "evaluator_id": outcome.evaluator_id,
                "status": outcome.status,
            },
        )
        artifacts.append(ref)
    data = outcome.model_dump(mode="json")
    if artifacts:
        data["artifact_id"] = artifacts[0].artifact_id
    return _result(data, artifacts=artifacts)


def list_evaluators(service: Any, request: ListEvaluatorsInput, budget: Budget, run: Any):
    return _result({"evaluators": default_evaluators().describe()})


def create_failure_case(service: Any, request: CreateFailureCaseInput, budget: Budget, run: Any):
    workspace = service.workspace
    _require_workspace(service)
    traces = TraceStore(workspace)
    reference_traces: list[GenerationTrace] = []
    for trace_id in request.observations + request.controls:
        reference_traces.append(traces.load(trace_id))
    now = utc_now_iso()
    model = request.model
    inference = request.inference
    if reference_traces:
        if model is None:
            model = reference_traces[0].model.model_copy(deep=True)
        if inference is None:
            inference = reference_traces[0].params.model_copy(deep=True)
    annotations = [
        CaseAnnotation(
            annotation_id=new_annotation_id(),
            created_at=now,
            author=item.author,
            kind=item.kind,
            text=item.text,
        )
        for item in request.annotations
    ]
    hypotheses = [
        CaseHypothesis(
            hypothesis_id=new_hypothesis_id(),
            created_at=now,
            statement=item.statement,
            rationale=item.rationale,
            evidence_refs=item.evidence_refs,
        )
        for item in request.hypotheses
    ]
    captures = []
    for artifact_id in request.activation_capture_ids:
        ref, _path = workspace.artifacts.get(artifact_id)
        if ref.kind != "activation_capture":
            raise InputError(
                "activation capture reference has the wrong artifact kind",
                details={"artifact_id": artifact_id, "kind": ref.kind},
            )
        captures.append(ref)
    case_warnings: list[str] = []
    if inference is None and not reference_traces:
        case_warnings.append(
            "no inference settings were supplied and no trace was referenced; defaults are recorded"
        )
    case = FailureCase(
        case_id=new_case_id(),
        title=request.title,
        summary=request.summary,
        created_at=now,
        updated_at=now,
        review=CaseReview(status="open", updated_at=now),
        model=model if model is not None else (reference_traces[0].model if reference_traces else None) or ModelRef(),
        inference=inference if inference is not None else GenerationParams(),
        observations=request.observations,
        controls=request.controls,
        findings=request.findings,
        contrasts=request.contrasts,
        evaluations=request.evaluations,
        reproduction=request.reproduction,
        annotations=annotations,
        hypotheses=hypotheses,
        activation_captures=captures,
        patches=request.patches,
        tags=request.tags,
        metadata=request.metadata,
        warnings=case_warnings,
    )
    cases = CaseStore(workspace, traces=traces)
    path, digest = cases.save(case)
    return _result(
        {
            "case": case.model_dump(mode="json"),
            "case_sha256": digest,
            "path": str(path),
        }
    )


def inspect_failure_case(service: Any, request: InspectFailureCaseInput, budget: Budget, run: Any):
    case = _cases(service).load(request.case_id)
    return _result(
        {
            "case": case.model_dump(mode="json"),
            "case_sha256": case_sha256(case),
        }
    )


def list_failure_cases(service: Any, request: ListFailureCasesInput, budget: Budget, run: Any):
    summaries = _cases(service).list(
        status=request.status,
        tag=request.tag,
        limit=request.limit,
    )
    return _result(
        {
            "returned": len(summaries),
            "cases": [summary.model_dump(mode="json") for summary in summaries],
        }
    )


def update_failure_case(service: Any, request: UpdateFailureCaseInput, budget: Budget, run: Any):
    store = _cases(service)
    case = store.load(request.case_id)
    now = utc_now_iso()
    review = case.review.model_copy(deep=True)
    if request.review_status is not None:
        review.status = request.review_status
    if request.review_notes is not None:
        review.notes = request.review_notes
    if request.reviewer is not None and request.reviewer not in review.reviewers:
        review.reviewers.append(request.reviewer)
    review.updated_at = now
    case.review = review
    if request.add_annotation is not None:
        case.annotations.append(
            CaseAnnotation(
                annotation_id=new_annotation_id(),
                created_at=now,
                author=request.add_annotation.author,
                kind=request.add_annotation.kind,
                text=request.add_annotation.text,
            )
        )
    if request.add_hypothesis is not None:
        case.hypotheses.append(
            CaseHypothesis(
                hypothesis_id=new_hypothesis_id(),
                created_at=now,
                statement=request.add_hypothesis.statement,
                rationale=request.add_hypothesis.rationale,
                evidence_refs=request.add_hypothesis.evidence_refs,
            )
        )
    known = {hypothesis.hypothesis_id: hypothesis for hypothesis in case.hypotheses}
    for update in request.hypothesis_updates:
        hypothesis = known.get(update.hypothesis_id)
        if hypothesis is None:
            raise InputError(
                "unknown hypothesis identifier",
                details={"hypothesis_id": update.hypothesis_id},
            )
        hypothesis.status = update.status
    case.updated_at = now
    case.revision += 1
    path, digest = store.save(case)
    return _result(
        {
            "case": case.model_dump(mode="json"),
            "case_sha256": digest,
            "path": str(path),
        }
    )


def export_failure_case(service: Any, request: ExportFailureCaseInput, budget: Budget, run: Any):
    workspace = service.workspace
    _require_workspace(service)
    traces = TraceStore(workspace)
    cases = CaseStore(workspace, traces=traces)
    case = cases.load(request.case_id)
    bundle = build_export(
        case,
        traces=traces,
        artifacts=workspace.artifacts,
        include_traces=request.include_traces,
        redact_text=request.redact_text,
        budget=budget,
    )
    from wsp.core.json import dumps

    data = dumps(bundle).encode("utf-8")
    output = Path(request.output_path).expanduser().resolve()
    atomic_write_bytes(output, data, overwrite=request.overwrite)
    return _result(
        {
            "case_id": case.case_id,
            "output_path": str(output),
            "sha256": sha256_bytes(data),
            "nbytes": len(data),
            "traces_embedded": len(bundle["traces"]),
            "redacted": request.redact_text,
        },
        warnings=bundle.get("warnings", []),
    )


def import_failure_case(service: Any, request: ImportFailureCaseInput, budget: Budget, run: Any):
    workspace = service.workspace
    _require_workspace(service)
    traces = TraceStore(workspace)
    cases = CaseStore(workspace, traces=traces)
    case, warnings = import_export(
        request.path,
        traces=traces,
        cases=cases,
        overwrite=request.overwrite,
        budget=budget,
    )
    return _result(
        {
            "case": case.model_dump(mode="json"),
            "case_sha256": case_sha256(case),
            "path": str(cases.path_for(case.case_id)),
        },
        warnings=warnings,
    )


def register_behavior_tools(service: Any) -> None:
    from wsp.tools.behavior_schemas import (
        CompareTracesInput as CompareInput,
        CreateFailureCaseInput as CreateCaseInput,
        EvaluateBehaviorInput as EvaluateInput,
        ExportFailureCaseInput as ExportCaseInput,
        ExportTraceInput as ExportTraceSchema,
        ImportFailureCaseInput as ImportCaseInput,
        ImportTraceInput as ImportTraceSchema,
        InspectFailureCaseInput as InspectCaseInput,
        InspectTraceInput as InspectTraceSchema,
        ListDetectorsInput as DetectorsInput,
        ListEvaluatorsInput as EvaluatorsInput,
        ListFailureCasesInput as ListCasesInput,
        ListTracesInput as TracesInput,
        RecordTraceInput as RecordInput,
        RunContrastInput as ContrastInput,
        ScanTraceInput as ScanInput,
        UpdateFailureCaseInput as UpdateCaseInput,
    )
    from wsp.tools.service import ToolDefinition

    def bind(function):
        def handler(request, budget, run):
            return function(service, request, budget, run)

        return handler

    definitions = [
        (
            "record_trace",
            "Generate one deterministic or local-HF completion and store it as an immutable trace.",
            RecordInput,
            record_trace,
            "Trace identifier, index record, and compact trace metadata.",
            True,
        ),
        (
            "import_trace",
            "Import a WSP trace, an INIF 0.1 document, or an Inspect AI log.",
            ImportTraceSchema,
            import_trace,
            "Imported trace summaries (or full traces when persist=false).",
            False,
        ),
        (
            "inspect_trace",
            "Inspect one stored trace with a bounded token page and explicit unavailable fields.",
            InspectTraceSchema,
            inspect_trace,
            "Trace metadata, preview text, token page, provenance, and privacy policy.",
            True,
        ),
        (
            "list_traces",
            "List stored traces, newest first.",
            TracesInput,
            list_traces,
            "Bounded trace index summaries.",
            True,
        ),
        (
            "export_trace",
            "Export one trace as WSP JSON or an INIF 0.1 document, optionally redacted.",
            ExportTraceSchema,
            export_trace,
            "Written file path, checksum, size, and warnings.",
            True,
        ),
        (
            "scan_trace",
            "Run deterministic behavioral detectors over one trace.",
            ScanInput,
            scan_trace,
            "Detector runs with measurements and findings (evidence, not diagnosis).",
            True,
        ),
        (
            "list_detectors",
            "Describe registered detectors, parameters, and limitations.",
            DetectorsInput,
            list_detectors,
            "Detector registry with JSON schemas.",
            False,
        ),
        (
            "compare_traces",
            "Compare two traces structurally: inputs, tokens, outputs, termination, timing.",
            CompareInput,
            compare_traces,
            "Versioned structural comparison with no scientific judgement.",
            True,
        ),
        (
            "run_contrast",
            "Execute a paired comparison with an explicitly declared invariant.",
            ContrastInput,
            run_contrast,
            "Contrast result with exact inputs, outcomes, evaluations, and trace references.",
            True,
        ),
        (
            "evaluate_behavior",
            "Evaluate a trace or supplied text with a registered evaluator.",
            EvaluateInput,
            evaluate_behavior,
            "Evaluation outcome distinguishing pass, fail, inconclusive, abstention, and error.",
            False,
        ),
        (
            "list_evaluators",
            "Describe registered evaluators and their parameters.",
            EvaluatorsInput,
            list_evaluators,
            "Evaluator registry with JSON schemas.",
            False,
        ),
        (
            "create_failure_case",
            "Create a persistent investigation case referencing traces and evidence.",
            CreateCaseInput,
            create_failure_case,
            "Stored case with checksum and workspace path.",
            True,
        ),
        (
            "inspect_failure_case",
            "Load one failure case and its evidence references.",
            InspectCaseInput,
            inspect_failure_case,
            "Full case record and checksum.",
            True,
        ),
        (
            "list_failure_cases",
            "List investigation cases, filtered by review status or tag.",
            ListCasesInput,
            list_failure_cases,
            "Bounded case summaries.",
            True,
        ),
        (
            "update_failure_case",
            "Update review state, annotations, or hypothesis status in one case.",
            UpdateCaseInput,
            update_failure_case,
            "Updated case, checksum, and path.",
            True,
        ),
        (
            "export_failure_case",
            "Write a self-contained case bundle with embedded traces and contrast evidence.",
            ExportCaseInput,
            export_failure_case,
            "Export path, checksum, size, and warnings.",
            True,
        ),
        (
            "import_failure_case",
            "Reload a case export, restoring its traces and verifying checksums.",
            ImportCaseInput,
            import_failure_case,
            "Reloaded case, checksum, path, and warnings.",
            True,
        ),
    ]
    for name, description, schema, function, output, requires_workspace in definitions:
        service.register_tool(
            ToolDefinition(
                name,
                description,
                schema,
                bind(function),
                output,
                requires_workspace=requires_workspace,
            )
        )
