"""Tool handlers for activation interventions and counterfactual measurement."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from wsp.behavior.models import ModelRef
from wsp.behavior.trace import TraceStore
from wsp.core.budget import Budget
from wsp.core.errors import (
    ArtifactError,
    CompatibilityError,
    InputError,
    UnsupportedError,
    WspError,
)
from wsp.core.time import utc_now_iso
from wsp.investigation.evidence import (
    ContinuationRef,
    InterventionRef,
    build_evidence,
    evidence_checksum,
)
from wsp.investigation.models import SCHEMA_CAPTURE, TraceCaptureRecord
from wsp.replay.sources import load_replay_source
from wsp.tools.capture_handlers import CAPTURE_ARTIFACT_KIND, _load_capture
from wsp.tools.intervention_schemas import (
    AttachInterventionEvidenceInput,
    CompareInterventionResultsInput,
    CreateActivationInterventionInput,
    InspectCounterfactualComparisonInput,
    PreviewActivationInterventionInput,
    RunActivationInterventionInput,
    RunCounterfactualContinuationInput,
    RunPairedInterventionInput,
    ValidateActivationInterventionInput,
)

from ..interventions.continuation import (
    run_counterfactual_continuation as run_continuation_engine,
)
from ..interventions.engine import (
    build_result,
    describe_model,
    inspect_intervention,
    retarget_spec_execution,
    run_paired_intervention as run_paired_engine,
    run_pass,
)
from ..interventions.models import (
    SCHEMA_CONTINUATION,
    SCHEMA_COUNTERFACTUAL,
    SCHEMA_RESULT,
    SCHEMA_SPEC,
    ActivationIntervention,
    ContinuationComparison,
    ContinuationConfig,
    CounterfactualComparison,
    DonorActivation,
    ExecutionIdentity,
    InterventionResult,
    MeanActivation,
    MeanProvenance,
    TensorSelector,
    comparison_sha256,
    continuation_sha256,
    intervention_sha256,
    result_sha256,
)

INTERVENTION_ARTIFACT_KIND = "activation_intervention"
RESULT_ARTIFACT_KIND = "intervention_result"
COMPARISON_ARTIFACT_KIND = "counterfactual_comparison"
CONTINUATION_ARTIFACT_KIND = "continuation_comparison"

_BACKEND_MAP = {
    "torch_hooks": "torch_hooks",
    "torch-forward-hooks": "torch_hooks",
    "torch": "torch_hooks",
}


def _is_nnterp_backend(name: str | None) -> bool:
    """Only the optional nnterp backend is unsupported for interventions."""

    return bool(name) and "nnterp" in name.lower()


def _result(
    data: dict[str, Any],
    artifacts: list[Any] | None = None,
    warnings: list[str] | None = None,
):
    from wsp.tools.service import ToolResult

    return ToolResult(data=data, artifacts=list(artifacts or []), warnings=list(warnings or []))


def _require_workspace(service: Any) -> Any:
    if service.workspace is None:
        raise InputError(
            "this operation requires an initialized workspace; create one with 'wsp init --path <dir>', or restart 'wsp mcp' with --auto-init",
            details={"hint": "the workspace stores intervention specifications and results"},
        )
    return service.workspace


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _load_intervention(workspace: Any, artifact_id: str) -> tuple[ActivationIntervention, Any]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != INTERVENTION_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not an activation intervention",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    payload = workspace.artifacts.read_json(artifact_id)
    spec = ActivationIntervention.model_validate(payload)
    if spec.schema_version != SCHEMA_SPEC:
        raise ArtifactError(
            "unsupported intervention schema",
            details={"schema_version": spec.schema_version},
        )
    expected = spec.checksums.get("spec_sha256")
    if expected is not None and expected != intervention_sha256(spec):
        raise ArtifactError(
            "the intervention specification checksum does not match its content",
            details={"artifact_id": artifact_id},
        )
    return spec, ref


def _load_comparison(workspace: Any, artifact_id: str) -> tuple[CounterfactualComparison, Any]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != COMPARISON_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a counterfactual comparison",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    record = CounterfactualComparison.model_validate(workspace.artifacts.read_json(artifact_id))
    expected = record.checksums.get("comparison_sha256")
    if expected is not None and expected != comparison_sha256(record):
        raise ArtifactError("the comparison checksum does not match its content")
    return record, ref


def _load_continuation(
    workspace: Any, artifact_id: str
) -> tuple[ContinuationComparison, Any]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != CONTINUATION_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a continuation comparison",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    record = ContinuationComparison.model_validate(workspace.artifacts.read_json(artifact_id))
    expected = record.checksums.get("continuation_sha256")
    if expected is not None and expected != continuation_sha256(record):
        raise ArtifactError("the continuation checksum does not match its content")
    return record, ref


def _load_result(workspace: Any, artifact_id: str) -> tuple[InterventionResult, Any]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != RESULT_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not an intervention result",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    record = InterventionResult.model_validate(workspace.artifacts.read_json(artifact_id))
    expected = record.checksums.get("result_sha256")
    if expected is not None and expected != result_sha256(record):
        raise ArtifactError("the result checksum does not match its content")
    return record, ref


def _resolve_spec(workspace: Any, request: Any) -> ActivationIntervention:
    if request.intervention is not None:
        spec = ActivationIntervention.model_validate(request.intervention)
        if spec.schema_version != SCHEMA_SPEC:
            raise ArtifactError(
                "unsupported intervention schema",
                details={"schema_version": spec.schema_version},
            )
        return spec
    spec, _ref = _load_intervention(workspace, request.intervention_artifact_id)
    return spec


def _trace_tokens(workspace: Any, trace_id: str) -> list[int]:
    trace = TraceStore(workspace).load(trace_id)
    tokens = [int(item) for item in trace.input.token_ids]
    for record in trace.tokens:
        if record.token_id is None:
            raise InputError(
                "the trace did not record exact token identifiers; the intervention cannot "
                "reconstruct its sequence",
                details={"trace_id": trace_id},
            )
        tokens.append(int(record.token_id))
    if not tokens:
        raise InputError("the trace has no recorded token identifiers", details={"trace_id": trace_id})
    return tokens


def _tokens_for(workspace: Any, request: Any) -> list[int]:
    if request.token_ids:
        return [int(item) for item in request.token_ids]
    return _trace_tokens(workspace, request.trace_id)


def _check_model_identity(spec: ActivationIntervention, model_info: dict[str, Any]) -> None:
    expected = spec.model.file_sha256
    observed = model_info.get("file_sha256")
    if expected and observed and expected != observed:
        raise CompatibilityError(
            "the loaded model does not match the model recorded by the intervention",
            details={"expected_file_sha256": expected, "observed_file_sha256": observed},
        )
    expected_fp = spec.model.structure_fingerprint
    observed_fp = model_info.get("structure_fingerprint")
    if expected_fp and observed_fp and expected_fp != observed_fp:
        raise CompatibilityError(
            "the loaded model structure does not match the intervention specification",
            details={"expected": expected_fp, "observed": observed_fp},
        )


def _capture_measurement(
    record: TraceCaptureRecord,
    module: str,
    role: str,
    position: int,
    tensor_name: str | None,
):
    matches = [
        item
        for item in record.measurements
        if item.module == module
        and item.tensor_role == role
        and item.position == position
        and item.call_index == 0
    ]
    if tensor_name is not None:
        matches = [item for item in matches if item.tensor_name == tensor_name]
    if not matches:
        raise CompatibilityError(
            "the capture does not contain the requested position",
            details={"module": module, "tensor_role": role, "position": position},
        )
    if len(matches) > 1:
        raise CompatibilityError(
            "the capture has several tensors at this point; name the tensor explicitly",
            details={
                "module": module,
                "position": position,
                "tensors": sorted({item.tensor_name for item in matches}),
            },
        )
    return matches[0]


def _capture_vectors(
    workspace: Any,
    record: TraceCaptureRecord,
    *,
    module: str,
    role: str,
    positions: Sequence[int],
    tensor_name: str | None,
) -> list[np.ndarray]:
    if record.mode != "selected":
        raise CompatibilityError(
            "the capture did not retain activation vectors; create a 'selected' capture first",
            details={"capture_id": record.capture_id, "mode": record.mode},
        )
    vectors: list[np.ndarray] = []
    for position in positions:
        measurement = _capture_measurement(record, module, role, int(position), tensor_name)
        if measurement.value_artifact_id is None:
            raise CompatibilityError(
                "the capture measurement has no retained vector",
                details={"position": position, "module": module},
            )
        vector = np.asarray(
            workspace.artifacts.read_array(measurement.value_artifact_id), dtype=np.float64
        ).reshape(-1)
        if not np.all(np.isfinite(vector)):
            raise CompatibilityError(
                "the retained activation contains non-finite values",
                details={"position": position, "module": module},
            )
        vectors.append(vector)
    return vectors


# --------------------------------------------------------------------------- #
# creation
# --------------------------------------------------------------------------- #


def create_activation_intervention(
    service: Any, request: CreateActivationInterventionInput, budget: Budget, run: Any
):
    """Build and validate a versioned intervention specification."""

    workspace = _require_workspace(service)
    capture: TraceCaptureRecord | None = None
    capture_ref = None
    if request.capture_artifact_id is not None:
        capture, capture_ref = _load_capture(workspace, request.capture_artifact_id)

    backend = request.backend
    model_ref = request.model
    sequence_length = request.sequence_length
    dtype = request.dtype
    device = request.device
    backend_version = request.backend_version
    trace_id = request.trace_id
    replay_id = None
    replay_artifact_id = None
    alignment_artifact_id = None
    expected_shape = request.expected_shape
    expected_dtype = request.expected_dtype

    if capture is not None:
        if _is_nnterp_backend(capture.backend):
            raise UnsupportedError(
                "interventions cannot be created from this capture backend",
                details={
                    "backend": capture.backend,
                    "note": "the optional nnterp backend keeps capture support only; "
                    "intervention support there is deferred",
                },
            )
        backend = "torch_hooks"
        backend_version = capture.backend_version
        dtype = dtype or capture.dtype
        device = capture.device
        sequence_length = sequence_length or capture.sequence_length
        trace_id = trace_id or capture.trace_id
        replay_id = capture.replay_id
        replay_artifact_id = capture.replay_artifact_id
        alignment_artifact_id = capture.alignment_artifact_id
        if request.module not in capture.modules:
            raise CompatibilityError(
                "the requested module was not captured",
                details={"module": request.module, "captured": list(capture.modules)},
            )
        if capture.position_semantics != request.position_semantics:
            raise CompatibilityError(
                "the requested position semantics differ from the capture",
                details={
                    "capture": capture.position_semantics,
                    "request": request.position_semantics,
                },
            )
        if model_ref is None:
            model_ref = {
                "name": capture.model_name,
                "file_sha256": capture.model_file_sha256,
                "structure_fingerprint": capture.model_structure_fingerprint,
                "parameter_count": capture.model_parameter_count,
            }

    if model_ref is None or sequence_length is None:
        raise InputError(
            "the intervention model identity and sequence length are required",
            details={"hint": "supply capture_artifact_id or model plus sequence_length"},
        )
    model = ModelRef.model_validate(model_ref)

    selector = TensorSelector(
        axis=request.axis,
        positions=sorted({int(item) for item in request.positions}),
        feature_start=request.feature_start,
        feature_end=request.feature_end,
        feature_indices=sorted({int(item) for item in request.feature_indices}),
        tensor_name=request.tensor_name,
        tensor_index=request.tensor_index,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
    )

    mean = None
    if request.operation == "mean":
        provenance = MeanProvenance(
            method=request.mean_method
            or ("capture-mean" if request.mean_source_capture_artifact_id else "explicit"),
            source_capture_artifact_id=request.mean_source_capture_artifact_id
            or request.capture_artifact_id,
            source_trace_ids=request.mean_source_trace_ids
            or ([trace_id] if trace_id else []),
            module=request.module,
            tensor_role=request.tensor_role,
            positions=sorted({int(item) for item in request.mean_positions})
            or list(selector.positions),
            position_semantics=request.position_semantics,
            token_alignment=request.mean_token_alignment
            or "the mean is aggregated over the recorded aligned positions and applied to the target positions",
            model_file_sha256=model.file_sha256,
            model_structure_fingerprint=model.structure_fingerprint,
            notes=request.mean_notes,
        )
        if request.mean_values is not None:
            mean = MeanActivation(
                values=[float(item) for item in request.mean_values],
                dtype=request.dtype or dtype,
                provenance=provenance,
            )
        elif request.mean_artifact_id is not None:
            mean = MeanActivation(
                artifact_id=request.mean_artifact_id,
                dtype=request.dtype or dtype,
                provenance=provenance,
            )
        else:
            source_capture, _ref = _load_capture(
                workspace, request.mean_source_capture_artifact_id
            )
            if (
                model.file_sha256
                and source_capture.model_file_sha256
                and model.file_sha256 != source_capture.model_file_sha256
            ):
                raise CompatibilityError(
                    "mean provenance belongs to a different model file",
                    details={
                        "intervention": model.file_sha256,
                        "mean_source": source_capture.model_file_sha256,
                    },
                )
            vectors = _capture_vectors(
                workspace,
                source_capture,
                module=request.module,
                role=request.tensor_role,
                positions=provenance.positions,
                tensor_name=request.tensor_name,
            )
            stacked = np.stack(vectors, axis=0)
            run_warning = None
            if not np.allclose(stacked, stacked[0], atol=0.0, rtol=0.0):
                run_warning = (
                    "the mean is aggregated across positions that differ; the per-position "
                    "variation is recorded in provenance metadata"
                )
            if run_warning:
                provenance.notes = (provenance.notes or "") + (" " if provenance.notes else "") + run_warning
            values = stacked.mean(axis=0)
            mean = MeanActivation(
                values=[float(item) for item in values],
                dtype=request.dtype or dtype,
                provenance=provenance,
            )

    donor = None
    if request.operation == "patch":
        donor_capture = None
        donor_trace = request.donor_trace_id or trace_id
        if request.donor_capture_artifact_id is not None:
            donor_capture, _ref = _load_capture(workspace, request.donor_capture_artifact_id)
            if (
                model.file_sha256
                and donor_capture.model_file_sha256
                and model.file_sha256 != donor_capture.model_file_sha256
            ):
                raise CompatibilityError(
                    "donor provenance belongs to a different model file",
                    details={
                        "intervention": model.file_sha256,
                        "donor": donor_capture.model_file_sha256,
                    },
                )
            donor_positions = sorted({int(item) for item in request.donor_positions})
            vectors = _capture_vectors(
                workspace,
                donor_capture,
                module=request.module,
                role=request.tensor_role,
                positions=donor_positions,
                tensor_name=request.tensor_name,
            )
            values = (
                vectors[0]
                if len(vectors) == 1
                else np.stack(vectors, axis=0).reshape(-1)
            )
        elif request.donor_values is not None:
            values = np.asarray(request.donor_values, dtype=np.float64)
        else:
            values = np.asarray(
                workspace.artifacts.read_array(request.donor_values_artifact_id), dtype=np.float64
            ).reshape(-1)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise CompatibilityError("the donor values contain non-finite entries")
        shape = request.donor_shape or (
            [
                max(1, len(request.donor_positions))
                if request.donor_capture_artifact_id is not None
                else 1,
                int(values.size)
                // max(1, len(request.donor_positions))
                if request.donor_capture_artifact_id is not None
                else int(values.size),
            ]
        )
        if shape is not None and int(np.prod(shape)) != int(values.size):
            raise CompatibilityError(
                "the declared donor shape does not match the number of donor values",
                details={"shape": shape, "values": int(values.size)},
            )
        donor = DonorActivation(
            trace_id=donor_trace or (capture.trace_id if capture is not None else None),
            capture_artifact_id=request.donor_capture_artifact_id
            or (request.capture_artifact_id if capture is not None else None),
            execution_id=None,
            module=request.module,
            tensor_role=request.tensor_role,
            positions=sorted({int(item) for item in request.donor_positions}) or list(selector.positions),
            position_semantics=request.position_semantics,
            condition=request.donor_condition,
            values=[float(item) for item in values] if shape is not None else None,
            values_artifact_id=None if shape is not None else request.donor_values_artifact_id,
            shape=shape,
            dtype=request.donor_dtype or dtype,
            allow_position_broadcast=request.donor_allow_position_broadcast,
            note=request.donor_note,
        )

    spec = ActivationIntervention(
        intervention_id=_new_intervention_id(),
        created_at=utc_now_iso(),
        name=request.name,
        model=model,
        execution=ExecutionIdentity(
            backend=backend,
            backend_version=backend_version,
            device=device,
            dtype=dtype,
            sequence_length=int(sequence_length),
            trace_id=trace_id,
            replay_id=replay_id,
            replay_artifact_id=replay_artifact_id,
            capture_artifact_id=request.capture_artifact_id,
            alignment_artifact_id=alignment_artifact_id,
        ),
        target={
            "module": request.module,
            "capture_point": request.capture_point,
            "tensor_role": request.tensor_role,
            "position_semantics": request.position_semantics,
            "selector": selector,
        },
        operation=request.operation,
        mean=mean,
        donor=donor,
        apply_on_calls=sorted({int(item) for item in request.apply_on_calls}),
        allow_dtype_cast=request.allow_dtype_cast,
        limits={
            "max_elements": request.max_elements,
            "max_positions": request.max_positions,
        },
        notes=request.notes,
        metadata=dict(request.metadata),
    )
    spec.checksums["spec_sha256"] = intervention_sha256(spec)

    artifacts: list[Any] = []
    ref = None
    if request.save:
        ref = workspace.artifacts.put_json(
            INTERVENTION_ARTIFACT_KIND,
            spec.model_dump(mode="json"),
            metadata={
                "schema_version": spec.schema_version,
                "intervention_id": spec.intervention_id,
                "operation": spec.operation,
                "module": spec.target.module,
            },
        )
        artifacts.append(ref)

    payload = spec.model_dump(mode="json")
    payload["artifact_id"] = ref.artifact_id if ref else None
    payload["capture_artifact_id"] = request.capture_artifact_id
    if capture_ref is not None:
        payload["capture_artifact"] = capture_ref.model_dump(mode="json")
    payload["summary"] = _spec_summary(spec)
    return _result(payload, artifacts, [])


def _spec_summary(spec: ActivationIntervention) -> dict[str, Any]:
    return {
        "intervention_id": spec.intervention_id,
        "operation": spec.operation,
        "module": spec.target.module,
        "tensor_role": spec.target.tensor_role,
        "positions": list(spec.target.selector.positions),
        "position_semantics": spec.target.position_semantics,
        "has_mean": spec.mean is not None,
        "has_donor": spec.donor is not None,
        "apply_on_calls": list(spec.apply_on_calls),
        "sequence_length": spec.execution.sequence_length,
        "backend": spec.execution.backend,
    }


def _new_intervention_id() -> str:
    from ..interventions.ids import new_intervention_id

    return new_intervention_id()


# --------------------------------------------------------------------------- #
# preview / validation
# --------------------------------------------------------------------------- #


def _prepare_model(request: Any) -> Any:
    return load_replay_source(request.source)


def _preview(
    service: Any, request: Any, budget: Budget, run: Any, *, full: bool
):
    workspace = _require_workspace(service)
    spec = _resolve_spec(workspace, request)
    tokens = _tokens_for(workspace, request)
    model = _prepare_model(request)
    _check_model_identity(spec, describe_model(model))
    validation = inspect_intervention(spec, model, tokens, store=workspace.artifacts, budget=budget)
    data = {
        "valid": validation.valid,
        "checks": [item.model_dump(mode="json") for item in validation.checks],
        "errors": validation.errors,
        "warnings": validation.warnings,
        "scope": validation.scope.model_dump(mode="json") if validation.scope else None,
        "summary": _spec_summary(spec),
        "intervention_id": spec.intervention_id,
    }
    if full:
        data["human_readable"] = validation.human_readable
        data["estimated_elements"] = (
            validation.scope.estimated_elements if validation.scope else None
        )
    return _result(data, [], validation.warnings)


def validate_activation_intervention(
    service: Any, request: ValidateActivationInterventionInput, budget: Budget, run: Any
):
    """Check a specification against a model and token sequence without running it."""

    return _preview(service, request, budget, run, full=False)


def preview_activation_intervention(
    service: Any, request: PreviewActivationInterventionInput, budget: Budget, run: Any
):
    """Describe exactly what a specification would modify, and its estimated cost."""

    return _preview(service, request, budget, run, full=True)


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


def _save_result(workspace: Any, result: InterventionResult) -> Any:
    result.checksums["result_sha256"] = result_sha256(result)
    return workspace.artifacts.put_json(
        RESULT_ARTIFACT_KIND,
        result.model_dump(mode="json"),
        metadata={
            "schema_version": result.schema_version,
            "result_id": result.result_id,
            "intervention_id": result.intervention_id,
            "condition": result.condition,
        },
    )


def run_activation_intervention(
    service: Any, request: RunActivationInterventionInput, budget: Budget, run: Any
):
    """Execute one intervention pass and record its measurement."""

    workspace = _require_workspace(service)
    spec = _resolve_spec(workspace, request)
    tokens = _tokens_for(workspace, request)
    model = _prepare_model(request)
    model_info = describe_model(model)
    _check_model_identity(spec, model_info)
    outcome = run_pass(
        spec,
        model,
        tokens,
        condition="intervention",
        store=workspace.artifacts,
        measure_modules=request.measure_modules,
        measure_positions=request.measure_positions or None,
        measure_tensor_names=request.measure_tensor_names or None,
        budget=budget,
    )
    result = build_result(
        spec,
        outcome,
        condition="intervention",
        model_info=model_info,
        logits_diff=None,
        token_logprobs=[],
        downstream=[],
    )
    result.logit_positions = list(request.logit_positions)
    artifacts: list[Any] = []
    ref = None
    if request.save:
        ref = _save_result(workspace, result)
        artifacts.append(ref)
    data = result.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id if ref else None
    data["case_id"] = request.case_id
    return _result(data, artifacts, result.warnings)


def run_paired_intervention(
    service: Any, request: RunPairedInterventionInput, budget: Budget, run: Any
):
    """Run untouched, identity, and intervention passes and compare them."""

    workspace = _require_workspace(service)
    spec = _resolve_spec(workspace, request)
    tokens = _tokens_for(workspace, request)
    model = _prepare_model(request)
    model_info = describe_model(model)
    _check_model_identity(spec, model_info)
    alternative = (
        ActivationIntervention.model_validate(request.alternative_intervention)
        if request.alternative_intervention is not None
        else None
    )
    outcome = run_paired_engine(
        spec,
        model,
        tokens,
        store=workspace.artifacts,
        budget=budget,
        measure_modules=request.measure_modules,
        measure_positions=request.measure_positions or None,
        measure_tensor_names=request.measure_tensor_names or None,
        logit_positions=request.logit_positions or None,
        alternative=alternative,
        direction=request.direction,
        case_id=request.case_id,
        intervention_artifact_id=request.intervention_artifact_id,
        identity_atol=request.identity_atol,
        identity_rtol=request.identity_rtol,
        identity=request.identity,
    )
    comparison = outcome.comparison
    artifacts: list[Any] = []
    result_refs: dict[str, Any] = {}
    for name, result in outcome.results.items():
        if request.save:
            ref = _save_result(workspace, result)
            artifacts.append(ref)
            result_refs[name] = ref
    if request.save:
        conditions = [
            condition.model_copy(
                update={
                    "artifact_id": result_refs[condition.condition].artifact_id
                    if condition.condition in result_refs
                    else None
                }
            )
            for condition in comparison.conditions
        ]
        comparison = comparison.model_copy(update={"conditions": conditions})
        comparison.checksums["comparison_sha256"] = comparison_sha256(comparison)
        ref = workspace.artifacts.put_json(
            COMPARISON_ARTIFACT_KIND,
            comparison.model_dump(mode="json"),
            metadata={
                "schema_version": comparison.schema_version,
                "comparison_id": comparison.comparison_id,
                "intervention_id": comparison.intervention_id,
                "validity": comparison.validity.status,
            },
        )
        artifacts.append(ref)
    else:
        ref = None
    data = comparison.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id if ref else None
    data["result_artifacts"] = {
        name: item.artifact_id for name, item in result_refs.items()
    }
    data["summary"] = _comparison_summary(comparison)
    return _result(data, artifacts, comparison.warnings)


def _comparison_summary(record: CounterfactualComparison) -> dict[str, Any]:
    effect = record.effect.logits
    return {
        "comparison_id": record.comparison_id,
        "validity": record.validity.status,
        "identity_agreement": record.identity_control.agreement,
        "identity_passed": record.identity_control.passed,
        "logits_max_abs_diff": effect.max_abs if effect else None,
        "logits_allclose": effect.allclose if effect else None,
        "target_max_abs_diff": record.effect.target_activation.max_abs
        if record.effect.target_activation
        else None,
        "token_logprob_deltas": [
            {
                "position": item.position,
                "token_id": item.token_id,
                "intervention_delta": item.intervention_delta,
            }
            for item in record.effect.token_logprobs
        ],
        "downstream_modules": [item.module for item in record.effect.downstream],
        "conditions": [item.condition for item in record.conditions],
        "interpretation": record.interpretation,
    }


# --------------------------------------------------------------------------- #
# continuations
# --------------------------------------------------------------------------- #


def run_counterfactual_continuation(
    service: Any, request: RunCounterfactualContinuationInput, budget: Budget, run: Any
):
    """Run a bounded, uncached, greedy continuation comparison."""

    workspace = _require_workspace(service)
    spec = _resolve_spec(workspace, request)
    tokens = _tokens_for(workspace, request)
    if len(tokens) != spec.execution.sequence_length:
        raise CompatibilityError(
            "the prefix length does not match the intervention execution identity",
            details={"prefix": len(tokens), "expected": spec.execution.sequence_length},
        )
    if spec.execution.execution_mode != "uncached-continuation":
        spec = retarget_spec_execution(
            spec,
            execution_mode="uncached-continuation",
            sequence_length=len(tokens),
        )
    model = _prepare_model(request)
    model_info = describe_model(model)
    _check_model_identity(spec, model_info)
    config = ContinuationConfig(
        max_new_tokens=request.max_new_tokens,
        max_sequence_length=request.max_sequence_length,
        eos_token_id=request.eos_token_id,
        seed=request.seed,
        stop_on_eos=request.stop_on_eos,
    )
    outcome = run_continuation_engine(
        spec,
        model,
        tokens,
        config,
        store=workspace.artifacts,
        budget=budget,
        case_id=request.case_id,
        intervention_artifact_id=request.intervention_artifact_id,
        identity=request.identity,
    )
    record = outcome.record
    artifacts: list[Any] = []
    ref = None
    if request.save:
        ref = workspace.artifacts.put_json(
            CONTINUATION_ARTIFACT_KIND,
            record.model_dump(mode="json"),
            metadata={
                "schema_version": record.schema_version,
                "continuation_id": record.continuation_id,
                "intervention_id": record.intervention_id,
                "validity": record.validity.status,
            },
        )
        artifacts.append(ref)
    data = record.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id if ref else None
    data["summary"] = {
        "validity": record.validity.status,
        "identity_agreement": record.comparison.identity_agreement,
        "longest_common_prefix": record.comparison.longest_common_prefix,
        "first_divergence": record.comparison.first_divergence,
        "length_delta": record.comparison.length_delta,
        "termination_changed": record.comparison.termination_changed,
        "token_agreement_rate": record.comparison.token_agreement_rate,
        "generated": {
            condition.condition: condition.generated_token_ids
            for condition in record.conditions
        },
        "interpretation": record.interpretation,
    }
    return _result(data, artifacts, record.warnings)


# --------------------------------------------------------------------------- #
# inspection
# --------------------------------------------------------------------------- #


def compare_intervention_results(
    service: Any, request: CompareInterventionResultsInput, budget: Budget, run: Any
):
    """Read a stored paired comparison and, optionally, its result records."""

    workspace = _require_workspace(service)
    record, ref = _load_comparison(workspace, request.comparison_artifact_id)
    data = record.model_dump(mode="json")
    data["artifact"] = ref.model_dump(mode="json")
    data["summary"] = _comparison_summary(record)
    conditions = record.conditions[: request.max_entries]
    data["conditions"] = [item.model_dump(mode="json") for item in conditions]
    data["conditions_truncated"] = len(record.conditions) > len(conditions)
    missing: list[str] = []
    if request.include_result_records:
        records = []
        for condition in conditions:
            if condition.artifact_id is None:
                continue
            try:
                result, _result_ref = _load_result(workspace, condition.artifact_id)
            except ArtifactError:
                missing.append(condition.artifact_id)
                continue
            records.append(result.model_dump(mode="json"))
        data["result_records"] = records
    data["integrity"] = {"missing_result_artifacts": missing, "checksum_verified": True}
    return _result(data, [ref])


def inspect_counterfactual_comparison(
    service: Any, request: InspectCounterfactualComparisonInput, budget: Budget, run: Any
):
    """Read a stored comparison or continuation record, verifying its checksum."""

    workspace = _require_workspace(service)
    ref, _path = workspace.artifacts.get(request.artifact_id)
    if ref.kind == COMPARISON_ARTIFACT_KIND:
        record, ref = _load_comparison(workspace, request.artifact_id)
        data = record.model_dump(mode="json")
        data["record_kind"] = "counterfactual-comparison"
        data["summary"] = _comparison_summary(record)
    elif ref.kind == CONTINUATION_ARTIFACT_KIND:
        record, ref = _load_continuation(workspace, request.artifact_id)
        data = record.model_dump(mode="json")
        data["record_kind"] = "continuation-comparison"
        data["summary"] = {
            "validity": record.validity.status,
            "identity_agreement": record.comparison.identity_agreement,
            "first_divergence": record.comparison.first_divergence,
            "longest_common_prefix": record.comparison.longest_common_prefix,
            "length_delta": record.comparison.length_delta,
        }
    else:
        raise ArtifactError(
            "the referenced artifact is not a counterfactual record",
            details={"artifact_id": request.artifact_id, "kind": ref.kind},
        )
    if not request.include_records:
        data = {
            key: data.get(key)
            for key in (
                "schema_version",
                "record_kind",
                "summary",
                "validity",
                "warnings",
                "limitations",
                "interpretation",
            )
        }
    data["artifact"] = ref.model_dump(mode="json")
    data["integrity"] = {"checksum_verified": True}
    return _result(data, [ref])


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #


def _intervention_ref(workspace: Any, artifact_id: str) -> InterventionRef:
    spec, ref = _load_intervention(workspace, artifact_id)
    return InterventionRef(
        intervention_id=spec.intervention_id,
        artifact_id=ref.artifact_id,
        operation=spec.operation,
        module=spec.target.module,
        tensor_role=spec.target.tensor_role,
        position_semantics=spec.target.position_semantics,
        positions=list(spec.target.selector.positions),
    )


def _continuation_ref(workspace: Any, artifact_id: str) -> ContinuationRef:
    record, ref = _load_continuation(workspace, artifact_id)
    return ContinuationRef(
        continuation_id=record.continuation_id,
        artifact_id=ref.artifact_id,
        intervention_id=record.intervention_id,
        validity_status=record.validity.status,
        identity_agreement=record.comparison.identity_agreement,
        first_divergence=record.comparison.first_divergence,
        length_delta=record.comparison.length_delta,
        termination_changed=record.comparison.termination_changed,
    )


def attach_intervention_evidence(
    service: Any, request: AttachInterventionEvidenceInput, budget: Budget, run: Any
):
    """Attach intervention and continuation references to an investigation case."""

    workspace = _require_workspace(service)
    traces = TraceStore(workspace)
    case = None
    if request.case_id is not None:
        from wsp.behavior.cases import CaseStore

        case = CaseStore(workspace, traces=traces).load(request.case_id)

    interventions = [
        _intervention_ref(workspace, artifact_id)
        for artifact_id in request.intervention_artifact_ids
    ]
    continuations = [
        _continuation_ref(workspace, artifact_id)
        for artifact_id in request.continuation_artifact_ids
    ]
    comparisons = []
    comparison_artifact_ids: list[str] = []
    for artifact_id in request.comparison_artifact_ids:
        record, ref = _load_comparison(workspace, artifact_id)
        comparison_artifact_ids.append(ref.artifact_id)
        comparisons.append(record)
    for intervention in interventions:
        matches = [
            record
            for record in comparisons
            if record.intervention_id == intervention.intervention_id
        ]
        if matches:
            intervention.comparison_artifact_id = next(
                (
                    ref_id
                    for ref_id, record in zip(comparison_artifact_ids, comparisons)
                    if record.intervention_id == intervention.intervention_id
                ),
                None,
            )
            intervention.validity_status = matches[0].validity.status
            intervention.identity_agreement = matches[0].identity_control.agreement

    hypotheses = []
    from wsp.behavior.ids import new_hypothesis_id
    from wsp.investigation.evidence import HypothesisNote

    for item in request.hypotheses:
        statement = item.get("statement")
        if not statement:
            raise InputError("each hypothesis requires a statement")
        hypotheses.append(
            HypothesisNote(
                hypothesis_id=new_hypothesis_id(),
                statement=statement,
                proposed_by=item.get("proposed_by"),
                rationale=item.get("rationale"),
                related_capture_ids=list(item.get("related_capture_ids", [])),
                related_comparison_ids=list(item.get("related_comparison_ids", [])),
            )
        )

    captures = []
    for artifact_id in request.capture_artifact_ids:
        from wsp.tools.investigation_handlers import _capture_ref

        captures.append(_capture_ref(workspace, artifact_id))

    if request.observation_trace_id is not None:
        traces.get_record(request.observation_trace_id)
    for trace_id in request.control_trace_ids:
        traces.get_record(trace_id)

    evidence = build_evidence(
        title=request.title,
        case_id=request.case_id,
        summary=request.summary,
        observation_trace_id=request.observation_trace_id,
        control_trace_ids=list(request.control_trace_ids),
        captures=captures,
        interventions=interventions,
        continuations=continuations,
        hypotheses=hypotheses,
        notes=list(request.notes),
        warnings=_intervention_warnings(interventions, continuations),
        metadata=dict(request.metadata),
    )
    artifacts: list[Any] = []
    evidence_ref = None
    if request.save:
        evidence_ref = workspace.artifacts.put_json(
            "investigation_evidence",
            evidence.model_dump(mode="json"),
            metadata={
                "schema_version": evidence.schema_version,
                "evidence_id": evidence.evidence_id,
                "case_id": evidence.case_id,
                "interventions": len(evidence.interventions),
                "continuations": len(evidence.continuations),
            },
        )
        artifacts.append(evidence_ref)
        evidence.checksums["artifact_sha256"] = evidence_ref.sha256

    link = None
    if case is not None and evidence_ref is not None:
        from wsp.behavior.cases import CaseStore
        from wsp.behavior.models import InvestigationLink

        store = CaseStore(workspace, traces=traces)
        now = utc_now_iso()
        link = InvestigationLink(
            evidence_id=evidence.evidence_id,
            artifact_id=evidence_ref.artifact_id,
            title=request.title,
            created_at=now,
            capture_ids=evidence.capture_ids,
            comparison_ids=evidence.comparison_ids,
            intervention_ids=evidence.intervention_ids,
            continuation_ids=evidence.continuation_ids,
            note=request.summary,
        )
        case.investigations.append(link)
        case.updated_at = now
        case.revision += 1
        store.save(case)

    data = evidence.model_dump(mode="json")
    data["evidence_artifact_id"] = evidence_ref.artifact_id if evidence_ref else None
    data["attached_to_case"] = link is not None
    data["hypothesis_count"] = len(hypotheses)
    return _result(data, artifacts, evidence.warnings)


def _intervention_warnings(
    interventions: Iterable[InterventionRef], continuations: Iterable[ContinuationRef]
) -> list[str]:
    warnings: list[str] = []
    for item in interventions:
        if item.validity_status == "not-run":
            warnings.append(
                f"intervention '{item.intervention_id}' has no paired comparison attached"
            )
        elif item.validity_status == "invalid":
            warnings.append(
                f"intervention '{item.intervention_id}' is marked invalid by its comparison"
            )
    for item in continuations:
        if item.validity_status == "invalid":
            warnings.append(
                f"continuation '{item.continuation_id}' is invalid; its difference is not a "
                "controlled comparison"
            )
    if not warnings:
        warnings.append(
            "an observed effect is reported under the recorded conditions and does not "
            "identify a circuit"
        )
    return warnings


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def register_intervention_tools(service: Any) -> None:
    """Register the V1.3 intervention and counterfactual operations."""

    from wsp.tools.service import ToolDefinition

    def bind(function: Any) -> Any:
        def handler(request: Any, budget: Budget, run: Any):
            return function(service, request, budget, run)

        return handler

    definitions = [
        (
            "create_activation_intervention",
            "Create a versioned, machine-readable activation-intervention specification. "
            "Derive the model, execution, and tensor identity from a stored capture, or "
            "supply them explicitly. Mean and donor values must carry provenance.",
            CreateActivationInterventionInput,
            create_activation_intervention,
            "A validated intervention specification with its artifact reference.",
            True,
        ),
        (
            "preview_activation_intervention",
            "Resolve an intervention against a real model and report the exact tensor region, "
            "axis, positions, and estimated cost without modifying any activation.",
            PreviewActivationInterventionInput,
            preview_activation_intervention,
            "Validation checks, a resolved scope, and a human-readable preview.",
            True,
        ),
        (
            "validate_activation_intervention",
            "Check an intervention specification against a model and token sequence, failing "
            "on incompatible shapes, dtypes, modules, or resources.",
            ValidateActivationInterventionInput,
            validate_activation_intervention,
            "A machine-readable pass/fail report with explicit errors.",
            True,
        ),
        (
            "run_activation_intervention",
            "Execute one intervention pass and record the targeted activation and output "
            "statistics. This is not a controlled comparison on its own.",
            RunActivationInterventionInput,
            run_activation_intervention,
            "One intervention-result record with resource accounting.",
            True,
        ),
        (
            "run_paired_intervention",
            "Run untouched, identity-control, and intervention passes over identical inputs, "
            "and compare targeted activations, downstream activations, logits, and token "
            "log-probabilities. A failed identity control invalidates attribution.",
            RunPairedInterventionInput,
            run_paired_intervention,
            "A counterfactual comparison plus one result artifact per condition.",
            True,
        ),
        (
            "compare_intervention_results",
            "Read a stored paired comparison, including its effect sizes and, optionally, "
            "its per-condition result records.",
            CompareInterventionResultsInput,
            compare_intervention_results,
            "Bounded comparison summary with integrity information.",
            True,
        ),
        (
            "run_counterfactual_continuation",
            "Run a bounded, uncached, greedy continuation comparison under a prefix-only "
            "intervention and report generated tokens, divergence, and termination.",
            RunCounterfactualContinuationInput,
            run_counterfactual_continuation,
            "A continuation-comparison record with divergence measurements.",
            True,
        ),
        (
            "inspect_counterfactual_comparison",
            "Read a stored counterfactual or continuation comparison and verify its checksum.",
            InspectCounterfactualComparisonInput,
            inspect_counterfactual_comparison,
            "The stored record with integrity verification.",
            True,
        ),
        (
            "attach_intervention_evidence",
            "Attach intervention specifications, paired comparisons, and continuation "
            "comparisons to an investigation case and evidence record.",
            AttachInterventionEvidenceInput,
            attach_intervention_evidence,
            "An investigation-evidence record linking the experiments to a case.",
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


__all__ = [
    "COMPARISON_ARTIFACT_KIND",
    "CONTINUATION_ARTIFACT_KIND",
    "INTERVENTION_ARTIFACT_KIND",
    "RESULT_ARTIFACT_KIND",
    "attach_intervention_evidence",
    "compare_intervention_results",
    "create_activation_intervention",
    "inspect_counterfactual_comparison",
    "preview_activation_intervention",
    "register_intervention_tools",
    "run_activation_intervention",
    "run_counterfactual_continuation",
    "run_paired_intervention",
    "validate_activation_intervention",
]
