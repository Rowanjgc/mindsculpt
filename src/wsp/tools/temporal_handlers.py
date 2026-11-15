"""Shared service operations for temporal plans, compositions, and staged runs.

Every handler is a thin function over the intervention engine.  Python callers
and the CLI reach exactly the same code path, and every operation returns
bounded, provenance-bearing data with machine-readable errors.
"""

from __future__ import annotations

import time
from typing import Any

from wsp.core.budget import Budget
from wsp.core.errors import (
    ArtifactError,
    CompatibilityError,
    InputError,
    NotFoundError,
)
from wsp.core.time import utc_now_iso
from wsp.execution.staged import build_staged_decoder
from wsp.interventions.composition import run_compositional_comparison as compare_plans
from wsp.interventions.engine import describe_model, retarget_spec_execution, torch_module
from wsp.interventions.models import ActivationIntervention, SCHEMA_SPEC
from wsp.interventions.plan import (
    SCHEMA_PLAN,
    InterventionPlan,
    PlanContinuationConfig,
    PlanIntervention,
    TemporalLimits,
    identity_control_plan,
    plan_sha256,
    validate_plan,
)
from wsp.interventions.plan_engine import (
    preview_plan,
    run_temporal_intervention as execute_temporal_plan,
    summarise_event_log,
)
from wsp.interventions.plan_models import (
    CompositionalComparison,
    InterventionEventLog,
    StagedContinuation,
    StagedVerification,
    TemporalInterventionRun,
    composition_sha256,
    event_log_sha256,
    staged_continuation_sha256,
    staged_verification_sha256,
    temporal_run_sha256,
)
from wsp.interventions.schedule import InterventionSchedule, parse_schedule
from wsp.interventions.staged import (
    run_staged_intervention as execute_staged_plan,
    verify_staged_generation as verify_staged,
)
from wsp.interventions.ids import new_plan_intervention_id
from wsp.tools.temporal_schemas import (
    AttachTemporalEvidenceInput,
    CreateInterventionPlanInput,
    InspectInterventionEventsInput,
    PrepareStagedGenerationInput,
    PreviewInterventionPlanInput,
    RunCompositionalComparisonInput,
    RunStagedInterventionInput,
    RunTemporalInterventionInput,
    ValidateInterventionPlanInput,
    VerifyStagedGenerationInput,
)

PLAN_ARTIFACT_KIND = "intervention_plan"
EVENT_LOG_ARTIFACT_KIND = "intervention_event_log"
TEMPORAL_RUN_ARTIFACT_KIND = "temporal_intervention"
COMPOSITION_ARTIFACT_KIND = "compositional_comparison"
STAGED_VERIFICATION_ARTIFACT_KIND = "staged_verification"
STAGED_CONTINUATION_ARTIFACT_KIND = "staged_continuation"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _result(data: dict[str, Any], artifacts: list[Any] | None = None, warnings: list[str] | None = None):
    from wsp.tools.service import ToolResult

    return ToolResult(data=data, artifacts=list(artifacts or []), warnings=list(warnings or []))


def _require_workspace(service: Any) -> Any:
    if service.workspace is None:
        raise InputError(
            "this operation requires a workspace; run 'wsp init' or pass --workspace",
            details={"hint": "plans, event logs, and comparisons are stored as artifacts"},
        )
    return service.workspace


def _load_plan(workspace: Any, artifact_id: str) -> tuple[InterventionPlan, Any]:
    from wsp.interventions.plan import plan_sha256 as compute

    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != PLAN_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not an intervention plan",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    plan = InterventionPlan.model_validate(workspace.artifacts.read_json(artifact_id))
    if plan.schema_version != SCHEMA_PLAN:
        raise ArtifactError(
            "unsupported intervention-plan schema",
            details={"schema_version": plan.schema_version},
        )
    expected = plan.checksums.get("plan_sha256")
    if expected is not None and expected != compute(plan):
        raise ArtifactError(
            "the intervention plan checksum does not match its content",
            details={"artifact_id": artifact_id},
        )
    return plan, ref


def _resolve_plan(workspace: Any, request: Any) -> InterventionPlan:
    if getattr(request, "plan", None) is not None:
        plan = InterventionPlan.model_validate(request.plan)
        if plan.schema_version != SCHEMA_PLAN:
            raise ArtifactError(
                "unsupported intervention-plan schema",
                details={"schema_version": plan.schema_version},
            )
        return plan
    plan, _ref = _load_plan(workspace, request.plan_artifact_id)
    return plan


def _load_event_log(workspace: Any, artifact_id: str) -> tuple[InterventionEventLog, Any]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != EVENT_LOG_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not an intervention event log",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    log = InterventionEventLog.model_validate(workspace.artifacts.read_json(artifact_id))
    expected = log.checksums.get("event_log_sha256")
    if expected is not None and expected != event_log_sha256(log):
        raise ArtifactError("the event-log checksum does not match its content")
    return log, ref


def _load_run(workspace: Any, artifact_id: str) -> tuple[TemporalInterventionRun, Any]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != TEMPORAL_RUN_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a temporal intervention run",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    record = TemporalInterventionRun.model_validate(workspace.artifacts.read_json(artifact_id))
    expected = record.checksums.get("temporal_run_sha256")
    if expected is not None and expected != temporal_run_sha256(record):
        raise ArtifactError("the temporal run checksum does not match its content")
    return record, ref


def _load_composition(workspace: Any, artifact_id: str) -> tuple[CompositionalComparison, Any]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != COMPOSITION_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a compositional comparison",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    record = CompositionalComparison.model_validate(workspace.artifacts.read_json(artifact_id))
    expected = record.checksums.get("composition_sha256")
    if expected is not None and expected != composition_sha256(record):
        raise ArtifactError("the compositional comparison checksum does not match its content")
    return record, ref


def _load_staged(workspace: Any, artifact_id: str) -> tuple[StagedContinuation, Any]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != STAGED_CONTINUATION_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a staged continuation",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    record = StagedContinuation.model_validate(workspace.artifacts.read_json(artifact_id))
    expected = record.checksums.get("staged_continuation_sha256")
    if expected is not None and expected != staged_continuation_sha256(record):
        raise ArtifactError("the staged continuation checksum does not match its content")
    return record, ref


def _load_verification(workspace: Any, artifact_id: str) -> tuple[StagedVerification, Any]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != STAGED_VERIFICATION_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a staged verification",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    record = StagedVerification.model_validate(workspace.artifacts.read_json(artifact_id))
    expected = record.checksums.get("staged_verification_sha256")
    if expected is not None and expected != staged_verification_sha256(record):
        raise ArtifactError("the staged verification checksum does not match its content")
    return record, ref


def _tokens(workspace: Any, request: Any, plan: InterventionPlan | None = None) -> list[int]:
    from wsp.tools.intervention_handlers import _trace_tokens

    if getattr(request, "prompt_token_ids", None):
        return [int(item) for item in request.prompt_token_ids]
    if getattr(request, "token_ids", None):
        return [int(item) for item in request.token_ids]
    if getattr(request, "trace_id", None):
        return _trace_tokens(workspace, request.trace_id)
    if plan is not None and plan.prompt_token_ids:
        return list(plan.prompt_token_ids)
    raise InputError(
        "supply token_ids, a trace_id, or a plan that records prompt_token_ids",
    )


def _model_for(request: Any) -> Any:
    from wsp.replay.sources import load_replay_source

    return load_replay_source(request.source)


def _check_plan_model(plan: InterventionPlan, model_info: dict[str, Any]) -> None:
    expected = plan.model.file_sha256
    observed = model_info.get("file_sha256")
    if expected and observed and expected != observed:
        raise CompatibilityError(
            "the loaded model does not match the checkpoint recorded by the plan",
            details={"expected_file_sha256": expected, "observed_file_sha256": observed},
        )
    expected_fp = plan.model.structure_fingerprint
    observed_fp = model_info.get("structure_fingerprint")
    if expected_fp and observed_fp and expected_fp != observed_fp:
        raise CompatibilityError(
            "the loaded model structure does not match the plan",
            details={"expected": expected_fp, "observed": observed_fp},
        )


def _save_plan(workspace: Any, plan: InterventionPlan) -> Any:
    plan.checksums["plan_sha256"] = plan_sha256(plan)
    return workspace.artifacts.put_json(
        PLAN_ARTIFACT_KIND,
        plan.model_dump(mode="json"),
        metadata={
            "schema_version": plan.schema_version,
            "plan_id": plan.plan_id,
            "plan_kind": plan.plan_kind,
            "execution_mode": plan.execution_mode,
            "interventions": len(plan.interventions),
        },
    )


def _save_log(workspace: Any, log: InterventionEventLog) -> Any:
    log.checksums["event_log_sha256"] = event_log_sha256(log)
    return workspace.artifacts.put_json(
        EVENT_LOG_ARTIFACT_KIND,
        log.model_dump(mode="json"),
        metadata={
            "schema_version": log.schema_version,
            "log_id": log.log_id,
            "plan_id": log.plan_id,
            "condition": log.condition,
            "missing": len(log.missing_events),
        },
    )


# --------------------------------------------------------------------------- #
# plan creation and validation
# --------------------------------------------------------------------------- #


def _spec_from_request(workspace: Any, item: Any) -> ActivationIntervention:
    from wsp.tools.intervention_handlers import _load_intervention

    if item.intervention is not None:
        spec = ActivationIntervention.model_validate(item.intervention)
        if spec.schema_version != SCHEMA_SPEC:
            raise ArtifactError(
                "unsupported intervention schema",
                details={"schema_version": spec.schema_version},
            )
        return spec
    spec, _ref = _load_intervention(workspace, item.intervention_artifact_id)
    return spec


def create_intervention_plan(
    service: Any, request: CreateInterventionPlanInput, budget: Budget, run: Any
):
    """Build and validate a versioned multi-site intervention plan."""

    workspace = _require_workspace(service)
    tokens = _tokens(workspace, request)
    prompt_length = len(tokens)
    expected_spec_mode = (
        "teacher-forced" if request.execution_mode == "teacher-forced" else "uncached-continuation"
    )
    interventions: list[PlanIntervention] = []
    retargeted: list[str] = []
    for index, item in enumerate(request.interventions):
        spec = _spec_from_request(workspace, item)
        if spec.execution.sequence_length != prompt_length:
            raise CompatibilityError(
                "an intervention was recorded against a different sequence length than the "
                "plan prefix",
                details={
                    "plan_prompt_length": prompt_length,
                    "intervention_sequence_length": spec.execution.sequence_length,
                    "intervention_id": spec.intervention_id,
                },
            )
        if spec.execution.execution_mode != expected_spec_mode:
            spec = retarget_spec_execution(
                spec,
                execution_mode=expected_spec_mode,  # type: ignore[arg-type]
                sequence_length=prompt_length,
            )
            retargeted.append(spec.intervention_id)
        if item.allow_overlap:
            spec = spec.model_copy(
                update={"metadata": {**spec.metadata, "allow_overlap": True}}
            )
        schedule = (
            parse_schedule(item.schedule)
            if item.schedule is not None
            else InterventionSchedule(kind="first-prefix-forward")
        )
        interventions.append(
            PlanIntervention(
                plan_intervention_id=new_plan_intervention_id(),
                label=item.label,
                order=item.order if item.order is not None else index,
                intervention=spec,
                schedule=schedule,
                position_mode=item.position_mode,
                on_step_call_indices=list(item.on_step_call_indices),
            )
        )
    from wsp.behavior.models import ModelRef
    from wsp.core.hashing import sha256_bytes
    from wsp.interventions.ids import new_plan_id

    model_payload = request.model
    if model_payload is None:
        model = interventions[0].intervention.model
    else:
        model = ModelRef.model_validate(model_payload)
    limits_payload = dict(TemporalLimits().model_dump())
    if request.limits:
        unknown = set(request.limits) - set(limits_payload)
        if unknown:
            raise InputError(
                "unknown temporal limit field(s): " + ", ".join(sorted(unknown)),
                details={"allowed": sorted(limits_payload)},
            )
        limits_payload.update(request.limits)
    plan = InterventionPlan(
        plan_id=new_plan_id(),
        created_at=utc_now_iso(),
        name=request.name,
        model=model,
        execution_mode=request.execution_mode,
        prompt_length=prompt_length,
        prompt_sha256=sha256_bytes(
            (",".join(str(item) for item in tokens)).encode("utf-8")
        ),
        prompt_token_ids=list(tokens),
        continuation=PlanContinuationConfig(
            max_new_tokens=request.max_new_tokens,
            max_sequence_length=request.max_sequence_length,
            eos_token_id=request.eos_token_id,
            stop_on_eos=request.stop_on_eos,
        ),
        interventions=interventions,
        limits=TemporalLimits.model_validate(limits_payload),
        provenance={
            **request.provenance,
            **({"trace_id": request.trace_id} if request.trace_id else {}),
        },
        notes=request.notes,
        metadata={"retargeted_interventions": retargeted},
    )
    validation = validate_plan(plan, budget=budget)

    artifacts: list[Any] = []
    ref = None
    if request.save and validation.valid:
        ref = _save_plan(workspace, plan)
        artifacts.append(ref)
    data = plan.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id if ref else None
    data["valid"] = validation.valid
    data["validation"] = validation.model_dump(mode="json")
    data["summary"] = {
        "plan_id": plan.plan_id,
        "plan_kind": plan.plan_kind,
        "execution_mode": plan.execution_mode,
        "interventions": len(plan.interventions),
        "scheduled_events": sum(
            len(item.schedule.resolve()) for item in plan.interventions
        ),
        "estimated_forward_calls": validation.estimated_forward_calls,
        "estimated_token_positions": validation.estimated_token_positions,
    }
    warnings = list(validation.warnings)
    if not validation.valid:
        warnings.append("the plan was not saved because it failed validation")
    warnings.append(
        "an observed effect will be reported under recorded conditions and will not "
        "identify a circuit"
    )
    return _result(data, artifacts, warnings)


def validate_intervention_plan(
    service: Any, request: ValidateInterventionPlanInput, budget: Budget, run: Any
):
    """Validate a plan statically and, when a model is supplied, against it."""

    workspace = _require_workspace(service)
    plan = _resolve_plan(workspace, request)
    root = None
    if request.source is not None:
        model = _model_for(request)
        root = torch_module(model)
    validation = validate_plan(plan, root=root, budget=budget)
    data = validation.model_dump(mode="json")
    data["plan_kind"] = plan.plan_kind
    data["execution_mode"] = plan.execution_mode
    data["scheduled_events"] = sum(
        len(item.schedule.resolve()) for item in plan.interventions
    )
    return _result(data, [], validation.warnings)


def preview_intervention_plan(
    service: Any, request: PreviewInterventionPlanInput, budget: Budget, run: Any
):
    """Resolve every scheduled firing against a real model without changing it."""

    workspace = _require_workspace(service)
    plan = _resolve_plan(workspace, request)
    tokens = _tokens(workspace, request, plan)
    model = _model_for(request)
    _check_plan_model(plan, describe_model(model))
    preview = preview_plan(plan, model, tokens, store=workspace.artifacts, budget=budget)
    validation = validate_plan(plan, root=torch_module(model), budget=budget)
    data = preview.model_dump(mode="json")
    data["validation"] = validation.model_dump(mode="json")
    data["scheduled_events"] = sum(
        len(item.schedule.resolve()) for item in plan.interventions
    )
    data["not_reached_events"] = data["scheduled_events"] - len(preview.firings)
    return _result(data, [], preview.warnings + validation.warnings)


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


def run_temporal_intervention(
    service: Any, request: RunTemporalInterventionInput, budget: Budget, run: Any
):
    """Run untouched, identity-control, and scheduled-plan conditions."""

    workspace = _require_workspace(service)
    plan = _resolve_plan(workspace, request)
    tokens = _tokens(workspace, request, plan)
    model = _model_for(request)
    _check_plan_model(plan, describe_model(model))
    artifacts: list[Any] = []
    plan_ref = None
    if request.save:
        plan_ref = _save_plan(workspace, plan)
        artifacts.append(plan_ref)
    control = identity_control_plan(plan)
    control_ref = None
    if request.identity and request.save:
        control_ref = _save_plan(workspace, control)
        artifacts.append(control_ref)
    outcome = execute_temporal_plan(
        plan,
        model,
        tokens,
        store=workspace.artifacts,
        budget=budget,
        case_id=request.case_id,
        plan_artifact_id=plan_ref.artifact_id if plan_ref else None,
        identity=request.identity,
        identity_plan=control if request.identity else None,
        identity_plan_artifact_id=control_ref.artifact_id if control_ref else None,
    )
    record = outcome.record
    log_refs: dict[str, str] = {}
    summaries: dict[str, Any] = {}
    if request.save:
        for name, log in outcome.event_logs.items():
            ref = _save_log(workspace, log)
            artifacts.append(ref)
            log_refs[name] = ref.artifact_id
            summaries[name] = summarise_event_log(log).model_dump(mode="json")
        record.event_log_artifact_ids = log_refs
        record.checksums["temporal_run_sha256"] = temporal_run_sha256(record)
        run_ref = workspace.artifacts.put_json(
            TEMPORAL_RUN_ARTIFACT_KIND,
            record.model_dump(mode="json"),
            metadata={
                "schema_version": record.schema_version,
                "run_id": record.run_id,
                "plan_id": record.plan_id,
                "validity": record.validity.status,
                "events_fired": record.metadata.get("events_fired", 0),
            },
        )
        artifacts.append(run_ref)
    else:
        run_ref = None
    data = record.model_dump(mode="json")
    data["artifact_id"] = run_ref.artifact_id if run_ref else None
    data["event_log_artifact_ids"] = log_refs
    data["event_summaries"] = summaries
    data["summary"] = {
        "run_id": record.run_id,
        "validity": record.validity.status,
        "identity_agreement": record.identity_control.agreement,
        "first_divergence": record.comparison.first_divergence,
        "events_fired": sum(
            len(log.events) for log in outcome.event_logs.values() if log.condition == "plan"
        ),
        "events_missing": sum(
            len(log.missing_events) for log in outcome.event_logs.values()
        ),
    }
    return _result(data, artifacts, record.warnings)


def inspect_intervention_events(
    service: Any, request: InspectInterventionEventsInput, budget: Budget, run: Any
):
    """Read an event log, or every log attached to a temporal run."""

    workspace = _require_workspace(service)
    logs: list[InterventionEventLog] = []
    ref_ids: dict[str, str] = {}
    if request.event_log_artifact_id is not None:
        log, ref = _load_event_log(workspace, request.event_log_artifact_id)
        logs.append(log)
        ref_ids[log.condition] = ref.artifact_id
    else:
        record, _ref = _load_run(workspace, request.run_artifact_id)
        for condition, artifact_id in record.event_log_artifact_ids.items():
            log, _log_ref = _load_event_log(workspace, artifact_id)
            logs.append(log)
            ref_ids[condition] = artifact_id
        if not logs:
            raise NotFoundError(
                "the temporal run has no stored event-log artifacts",
                details={"run_id": record.run_id},
            )
    data: dict[str, Any] = {
        "logs": [],
        "artifact_ids": ref_ids,
        "total_fired": sum(len(log.events) for log in logs),
        "total_missing": sum(len(log.missing_events) for log in logs),
        "total_not_reached": sum(len(log.not_reached_events) for log in logs),
        "total_rejected": sum(len(log.rejected_events) for log in logs),
    }
    for log in logs:
        summary = summarise_event_log(log).model_dump(mode="json")
        summary["artifact_id"] = ref_ids.get(log.condition)
        if request.include_events:
            summary["events"] = [
                item.model_dump(mode="json") for item in log.events[: request.max_entries]
            ]
            summary["missing_events"] = [
                item.model_dump(mode="json")
                for item in log.missing_events[: request.max_entries]
            ]
            summary["not_reached_events"] = [
                item.model_dump(mode="json")
                for item in log.not_reached_events[: request.max_entries]
            ]
            summary["rejected_events"] = [
                item.model_dump(mode="json")
                for item in log.rejected_events[: request.max_entries]
            ]
        data["logs"].append(summary)
    return _result(data, [], [])


def run_compositional_comparison(
    service: Any, request: RunCompositionalComparisonInput, budget: Budget, run: Any
):
    """Measure baseline, identity, A, B, and A+B, and report non-additivity."""

    workspace = _require_workspace(service)
    plan_a = _resolve_plan_payload(workspace, request.plan_a, request.plan_a_artifact_id)
    plan_b = _resolve_plan_payload(workspace, request.plan_b, request.plan_b_artifact_id)
    plan_ab = _resolve_plan_payload(workspace, request.plan_ab, request.plan_ab_artifact_id)
    tokens = _tokens(workspace, request, plan_ab)
    model = _model_for(request)
    _check_plan_model(plan_ab, describe_model(model))
    negative = (
        _resolve_plan_payload(workspace, None, request.negative_control_artifact_id)
        if request.negative_control_artifact_id
        else None
    )
    artifacts: list[Any] = []
    plan_refs: dict[str, str] = {}
    if request.save:
        for name, plan in (("A", plan_a), ("B", plan_b), ("AB", plan_ab)):
            ref = _save_plan(workspace, plan)
            plan_refs[name] = ref.artifact_id
            artifacts.append(ref)
    outcome = compare_plans(
        plan_a,
        plan_b,
        plan_ab,
        model,
        tokens,
        metric=request.metric,
        store=workspace.artifacts,
        budget=budget,
        case_id=request.case_id,
        plan_artifact_ids=plan_refs,
        identity=request.identity,
        negative_control=negative,
        isolation=request.isolation,
    )
    record = outcome.record
    log_refs: dict[str, str] = {}
    if request.save:
        for name, log in outcome.event_logs.items():
            ref = _save_log(workspace, log)
            artifacts.append(ref)
            log_refs[name] = ref.artifact_id
        record.event_log_artifact_ids = log_refs
        record.checksums["composition_sha256"] = composition_sha256(record)
        ref = workspace.artifacts.put_json(
            COMPOSITION_ARTIFACT_KIND,
            record.model_dump(mode="json"),
            metadata={
                "schema_version": record.schema_version,
                "comparison_id": record.comparison_id,
                "metric": record.metric,
                "validity": record.validity.status,
            },
        )
        artifacts.append(ref)
    else:
        ref = None
    primary = record.interactions[0] if record.interactions else None
    data = record.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id if ref else None
    data["event_log_artifact_ids"] = log_refs
    data["summary"] = {
        "comparison_id": record.comparison_id,
        "validity": record.validity.status,
        "identity_agreement": record.identity_control.agreement,
        "metric": record.metric,
        "interaction": primary.interaction if primary else None,
        "delta_a": primary.delta_a if primary else None,
        "delta_b": primary.delta_b if primary else None,
        "delta_ab": primary.delta_ab if primary else None,
        "per_condition_tokens": {
            item.condition: len(item.generated_token_ids) for item in record.conditions
        },
        "first_divergence": {
            item.condition: item.first_divergence for item in record.pairwise
        },
    }
    return _result(data, artifacts, record.warnings)


def _resolve_plan_payload(workspace: Any, payload: Any, artifact_id: str | None) -> InterventionPlan:
    if payload is not None:
        plan = InterventionPlan.model_validate(payload)
        if plan.schema_version != SCHEMA_PLAN:
            raise ArtifactError(
                "unsupported intervention-plan schema",
                details={"schema_version": plan.schema_version},
            )
        return plan
    plan, _ref = _load_plan(workspace, artifact_id)
    return plan


# --------------------------------------------------------------------------- #
# staged generation
# --------------------------------------------------------------------------- #


def prepare_staged_generation(
    service: Any, request: PrepareStagedGenerationInput, budget: Budget, run: Any
):
    """Build a generation state from a prefix and report what is supported."""

    _require_workspace(service)
    model = _model_for(request)
    decoder = build_staged_decoder(model)
    capabilities = decoder.capabilities()
    started = time.perf_counter()
    logits, state = decoder.prefill([int(item) for item in request.token_ids])
    prefill_ms = (time.perf_counter() - started) * 1000.0
    summary = {
        "supported": capabilities.supported,
        "backend": capabilities.backend,
        "architecture_class": capabilities.architecture_class,
        "state_kind": capabilities.state_kind,
        "reason": capabilities.reason,
        "position_ids_mode": capabilities.position_ids_mode,
        "attention_mask_mode": capabilities.attention_mask_mode,
        "state_hash": decoder.state_hash(state),
        "state_bytes": decoder.state_bytes(state),
        "state_length": decoder.state_length(state),
        "prefill_ms": round(prefill_ms, 3),
        "logits_shape": [int(item) for item in logits.shape],
    }
    warnings = [
        "preparing a state does not verify it; call verify_staged_generation before "
        "interpreting any staged intervention result"
    ]
    return _result(summary, [], warnings)


def verify_staged_generation(
    service: Any, request: VerifyStagedGenerationInput, budget: Budget, run: Any
):
    """Compare an unmodified staged run with the uncached reference."""

    workspace = _require_workspace(service)
    plan = _resolve_plan(workspace, request)
    tokens = _tokens(workspace, request, plan)
    model = _model_for(request)
    _check_plan_model(plan, describe_model(model))
    verification, timings = verify_staged(
        plan,
        model,
        tokens,
        budget=budget,
        identity=request.identity,
        atol=request.atol,
        rtol=request.rtol,
        isolation=request.isolation,
    )
    artifacts: list[Any] = []
    ref = None
    if request.save:
        ref = workspace.artifacts.put_json(
            STAGED_VERIFICATION_ARTIFACT_KIND,
            verification.model_dump(mode="json"),
            metadata={
                "schema_version": verification.schema_version,
                "verification_id": verification.verification_id,
                "agreement": verification.agreement,
                "architecture_class": verification.architecture_class,
            },
        )
        artifacts.append(ref)
    data = verification.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id if ref else None
    data["timings"] = {key: round(value, 3) for key, value in timings.items()}
    data["verified"] = verification.agreement in ("exact", "within_tolerance")
    return _result(data, artifacts, verification.warnings)


def run_staged_intervention(
    service: Any, request: RunStagedInterventionInput, budget: Budget, run: Any
):
    """Run a verified staged continuation under a scheduled plan."""

    workspace = _require_workspace(service)
    plan = _resolve_plan(workspace, request)
    tokens = _tokens(workspace, request, plan)
    model = _model_for(request)
    _check_plan_model(plan, describe_model(model))
    artifacts: list[Any] = []
    plan_ref = None
    if request.save:
        plan_ref = _save_plan(workspace, plan)
        artifacts.append(plan_ref)
    outcome = execute_staged_plan(
        plan,
        model,
        tokens,
        store=workspace.artifacts,
        budget=budget,
        case_id=request.case_id,
        plan_artifact_id=plan_ref.artifact_id if plan_ref else None,
        identity=request.identity,
        isolation=request.isolation,
        atol=request.atol,
        rtol=request.rtol,
    )
    record = outcome.record
    verification = outcome.verification
    verification_ref = None
    if request.save and verification is not None:
        verification_ref = workspace.artifacts.put_json(
            STAGED_VERIFICATION_ARTIFACT_KIND,
            verification.model_dump(mode="json"),
            metadata={
                "schema_version": verification.schema_version,
                "verification_id": verification.verification_id,
                "agreement": verification.agreement,
            },
        )
        artifacts.append(verification_ref)
    log_refs: dict[str, str] = {}
    if request.save:
        for name, log in outcome.event_logs.items():
            ref = _save_log(workspace, log)
            artifacts.append(ref)
            log_refs[name] = ref.artifact_id
        record.verification_artifact_id = (
            verification_ref.artifact_id if verification_ref else None
        )
        record.event_log_artifact_ids = log_refs
        record.checksums["staged_continuation_sha256"] = staged_continuation_sha256(record)
        ref = workspace.artifacts.put_json(
            STAGED_CONTINUATION_ARTIFACT_KIND,
            record.model_dump(mode="json"),
            metadata={
                "schema_version": record.schema_version,
                "continuation_id": record.continuation_id,
                "validity": record.validity.status,
                "verification": record.verification.agreement,
            },
        )
        artifacts.append(ref)
    else:
        ref = None
    data = record.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id if ref else None
    data["verification_artifact_id"] = (
        verification_ref.artifact_id if verification_ref else None
    )
    data["event_log_artifact_ids"] = log_refs
    data["summary"] = {
        "continuation_id": record.continuation_id,
        "validity": record.validity.status,
        "verification": record.verification.agreement,
        "identity_agreement": record.identity_control.agreement,
        "first_divergence": record.comparison.first_divergence,
        "cache_isolation": record.metadata.get("cache_isolation"),
        "performance": record.performance.model_dump(mode="json"),
    }
    return _result(data, artifacts, record.warnings)


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #


def attach_temporal_evidence(
    service: Any, request: AttachTemporalEvidenceInput, budget: Budget, run: Any
):
    """Attach temporal plans, event logs, and staged runs to a failure case."""

    workspace = _require_workspace(service)
    from wsp.behavior.cases import CaseStore
    from wsp.behavior.ids import new_hypothesis_id
    from wsp.behavior.trace import TraceStore
    from wsp.investigation.evidence import (
        CompositionRef,
        EventLogRef,
        HypothesisNote,
        PlanRef,
        StagedRef,
        TemporalRunRef,
        build_evidence,
    )
    traces = TraceStore(workspace)
    case = None
    if request.case_id is not None:
        case = CaseStore(workspace, traces=traces).load(request.case_id)

    plans = []
    for artifact_id in request.plan_artifact_ids:
        plan, ref = _load_plan(workspace, artifact_id)
        plans.append(
            PlanRef(
                plan_id=plan.plan_id,
                artifact_id=ref.artifact_id,
                plan_kind=plan.plan_kind,
                execution_mode=plan.execution_mode,
                interventions=len(plan.interventions),
                scheduled_events=sum(len(item.schedule.resolve()) for item in plan.interventions),
                schedule_kinds=sorted({item.schedule.kind for item in plan.interventions}),
            )
        )

    event_logs = []
    for artifact_id in request.event_log_artifact_ids:
        log, ref = _load_event_log(workspace, artifact_id)
        event_logs.append(
            EventLogRef(
                log_id=log.log_id,
                artifact_id=ref.artifact_id,
                plan_id=log.plan_id,
                condition=log.condition,
                execution_mode=log.execution_mode,
                steps_executed=log.steps_executed,
                fired=len(log.events),
                missing=len(log.missing_events),
                not_reached=len(log.not_reached_events),
                rejected=len(log.rejected_events),
                truncated=log.truncated,
            )
        )

    temporal_runs = []
    for artifact_id in request.run_artifact_ids:
        record, ref = _load_run(workspace, artifact_id)
        temporal_runs.append(
            TemporalRunRef(
                run_id=record.run_id,
                artifact_id=ref.artifact_id,
                plan_id=record.plan_id,
                validity_status=record.validity.status,
                identity_agreement=record.identity_control.agreement,
                first_divergence=record.comparison.first_divergence,
                events_fired=int(record.metadata.get("events_fired", 0) or 0),
                events_missing=sum(
                    1 for item in record.conditions if item.events_missing
                ),
            )
        )

    compositions = []
    for artifact_id in request.composition_artifact_ids:
        record, ref = _load_composition(workspace, artifact_id)
        primary = record.interactions[0] if record.interactions else None
        compositions.append(
            CompositionRef(
                comparison_id=record.comparison_id,
                artifact_id=ref.artifact_id,
                plan_ids=[record.plan_a_id, record.plan_b_id, record.plan_ab_id],
                metric=record.metric,
                interaction=primary.interaction if primary else None,
                validity_status=record.validity.status,
            )
        )

    staged = []
    for artifact_id in request.staged_artifact_ids:
        record, ref = _load_staged(workspace, artifact_id)
        staged.append(
            StagedRef(
                continuation_id=record.continuation_id,
                artifact_id=ref.artifact_id,
                plan_id=record.plan_id,
                verification_id=record.verification.verification_id,
                verification_artifact_id=record.verification_artifact_id,
                architecture_class=record.verification.architecture_class,
                agreement=record.verification.agreement,
                validity_status=record.validity.status,
            )
        )

    for artifact_id in request.verification_artifact_ids:
        verification, ref = _load_verification(workspace, artifact_id)
        staged.append(
            StagedRef(
                continuation_id=verification.verification_id,
                artifact_id=ref.artifact_id,
                plan_id="",
                verification_id=verification.verification_id,
                verification_artifact_id=ref.artifact_id,
                architecture_class=verification.architecture_class,
                agreement=verification.agreement,
                validity_status=(
                    "valid"
                    if verification.agreement in ("exact", "within_tolerance")
                    else "invalid"
                ),
            )
        )

    captures = []
    for artifact_id in request.capture_artifact_ids:
        from wsp.tools.investigation_handlers import _capture_ref

        captures.append(_capture_ref(workspace, artifact_id))

    hypotheses = []
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

    warnings: list[str] = []
    for item in temporal_runs:
        if item.validity_status == "invalid":
            warnings.append(
                f"temporal run '{item.run_id}' is invalid; its difference is not controlled"
            )
        if item.events_missing:
            warnings.append(
                f"temporal run '{item.run_id}' reports {item.events_missing} condition(s) "
                "with missing scheduled events"
            )
    for item in staged:
        if item.validity_status == "invalid":
            warnings.append(
                f"staged record '{item.continuation_id}' is invalid and cannot support a "
                "causal interpretation"
            )
        elif item.agreement not in ("exact", "within_tolerance"):
            warnings.append(
                f"staged record '{item.continuation_id}' has verification "
                f"'{item.agreement}', which does not license causal interpretation"
            )
    if not warnings:
        warnings.append(
            "a fired intervention is instrumentation evidence; no causal label is attached"
        )

    evidence = build_evidence(
        title=request.title,
        case_id=request.case_id,
        summary=request.summary,
        observation_trace_id=request.observation_trace_id,
        captures=captures,
        plans=plans,
        event_logs=event_logs,
        temporal_runs=temporal_runs,
        compositions=compositions,
        staged_continuations=staged,
        hypotheses=hypotheses,
        notes=list(request.notes),
        warnings=warnings,
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
                "plans": len(evidence.plans),
                "temporal_runs": len(evidence.temporal_runs),
                "compositions": len(evidence.compositions),
                "staged": len(evidence.staged_continuations),
            },
        )
        artifacts.append(evidence_ref)
        evidence.checksums["artifact_sha256"] = evidence_ref.sha256

    link = None
    if case is not None and evidence_ref is not None:
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
            plan_ids=evidence.plan_ids,
            temporal_run_ids=evidence.temporal_run_ids,
            composition_ids=evidence.composition_ids,
            staged_ids=evidence.staged_ids,
            note=request.summary,
        )
        case.investigations.append(link)
        case.updated_at = now
        case.revision += 1
        store.save(case)

    data = evidence.model_dump(mode="json")
    data["evidence_artifact_id"] = evidence_ref.artifact_id if evidence_ref else None
    data["attached_to_case"] = link is not None
    data["summary"] = {
        "plans": len(plans),
        "event_logs": len(event_logs),
        "temporal_runs": len(temporal_runs),
        "compositions": len(compositions),
        "staged": len(staged),
        "hypotheses": len(hypotheses),
    }
    return _result(data, artifacts, warnings)


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def register_temporal_tools(service: Any) -> None:
    """Register the V1.4 temporal, compositional, and staged operations."""

    from wsp.tools.service import ToolDefinition

    def bind(function: Any) -> Any:
        def handler(request: Any, budget: Budget, run: Any):
            return function(service, request, budget, run)

        return handler

    definitions = [
        (
            "create_intervention_plan",
            "Create a versioned, multi-site intervention plan with generation-step schedules, "
            "explicit ordering, resource limits, and provenance. Schedules are declarative; "
            "no executable predicate is accepted.",
            CreateInterventionPlanInput,
            create_intervention_plan,
            "A validated plan, its resolved schedule summary, and its artifact reference.",
            True,
        ),
        (
            "validate_intervention_plan",
            "Check a plan's schedules, position semantics, targeting, ordering, conflicts, and "
            "budgets, optionally resolving modules against a loaded model.",
            ValidateInterventionPlanInput,
            validate_intervention_plan,
            "A machine-readable validation report with explicit errors and conflicts.",
            True,
        ),
        (
            "preview_intervention_plan",
            "Resolve every scheduled firing against a real model and report the exact tensor "
            "shape, axis, rows, token indices, and element count without changing anything.",
            PreviewInterventionPlanInput,
            preview_intervention_plan,
            "Resolved firings per generation step plus validation and cost estimates.",
            True,
        ),
        (
            "run_temporal_intervention",
            "Execute untouched, identity-control, and scheduled-plan conditions over the same "
            "prefix. Missing or rejected scheduled events invalidate the run.",
            RunTemporalInterventionInput,
            run_temporal_intervention,
            "A temporal-run record, per-condition event logs, and divergence measurements.",
            True,
        ),
        (
            "inspect_intervention_events",
            "Read an intervention event log, or every log attached to a temporal run, with "
            "bounded per-event detail and applied/missing/not-reached/rejected counts.",
            InspectInterventionEventsInput,
            inspect_intervention_events,
            "Event summaries, bounded event records, and integrity-checked artifact ids.",
            True,
        ),
        (
            "run_compositional_comparison",
            "Run baseline, identity, A, B, and A+B conditions over one prefix and report a "
            "descriptive interaction contrast on one explicitly named metric.",
            RunCompositionalComparisonInput,
            run_compositional_comparison,
            "A compositional-comparison record with pairwise effects and non-additivity.",
            True,
        ),
        (
            "prepare_staged_generation",
            "Build a cached generation state from a prefix and report whether staged execution "
            "is supported for this architecture. This does not verify equivalence.",
            PrepareStagedGenerationInput,
            prepare_staged_generation,
            "Capabilities, state identity, cache size, and prefill timing.",
            True,
        ),
        (
            "verify_staged_generation",
            "Compare an unmodified staged run against the uncached reference on tokens, "
            "stepwise logits, termination, and identity control. Hybrid architectures are "
            "rejected explicitly.",
            VerifyStagedGenerationInput,
            verify_staged_generation,
            "A staged-verification record with declared tolerances and per-step agreement.",
            True,
        ),
        (
            "run_staged_intervention",
            "Run a scheduled plan through a verified staged decoder. A failed verification "
            "aborts execution instead of producing an interpretable result.",
            RunStagedInterventionInput,
            run_staged_intervention,
            "A staged-continuation record with verification, event logs, and performance.",
            True,
        ),
        (
            "attach_temporal_evidence",
            "Attach plans, event logs, temporal runs, compositional comparisons, and staged "
            "records to an investigation case and preserve reference integrity.",
            AttachTemporalEvidenceInput,
            attach_temporal_evidence,
            "An investigation-evidence record linking the temporal experiment to a case.",
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
    "COMPOSITION_ARTIFACT_KIND",
    "EVENT_LOG_ARTIFACT_KIND",
    "PLAN_ARTIFACT_KIND",
    "STAGED_CONTINUATION_ARTIFACT_KIND",
    "STAGED_VERIFICATION_ARTIFACT_KIND",
    "TEMPORAL_RUN_ARTIFACT_KIND",
    "attach_temporal_evidence",
    "create_intervention_plan",
    "inspect_intervention_events",
    "prepare_staged_generation",
    "preview_intervention_plan",
    "register_temporal_tools",
    "run_compositional_comparison",
    "run_staged_intervention",
    "run_temporal_intervention",
    "validate_intervention_plan",
    "verify_staged_generation",
]
