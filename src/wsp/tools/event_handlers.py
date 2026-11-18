"""Tool handlers for behavioral events, triggers, fixed-context replay, and dossiers."""

from __future__ import annotations

import time
from typing import Any

from wsp.behavior.cases import CaseStore
from wsp.behavior.events import (
    SCHEMA_EVENT,
    ArtifactPointer,
    BehavioralEvent,
    EventTriggerDefinition,
    detect_events,
    detect_text_pattern,
    event_sha256,
    manual_event,
    resolve_finding_event,
)
from wsp.behavior.ids import new_hypothesis_id
from wsp.behavior.models import InvestigationLink
from wsp.behavior.trace import TraceStore
from wsp.core.budget import Budget
from wsp.core.errors import ArtifactError, InputError, NotFoundError
from wsp.core.models import ArtifactRef
from wsp.core.time import utc_now_iso
from wsp.investigation.dossier import (
    SCHEMA_DOSSIER,
    InvestigationDossier,
    build_dossier,
    dossier_sha256,
)
from wsp.investigation.evidence import (
    DossierRef,
    EventRef,
    FixedContextRef,
    HypothesisNote,
    RecipeRunRef,
    build_evidence,
    evidence_checksum,
)
from wsp.interventions.event_trigger import (
    SCHEMA_TRIGGER,
    EventInterventionRun,
    EventTrigger,
    event_run_sha256,
    retrospective_trigger,
    run_online_event_trigger,
    trigger_sha256,
    validate_trigger_for_run,
)
from wsp.interventions.fixed_context import (
    SCHEMA_FIXED_COMPARISON,
    SCHEMA_FIXED_REPLAY,
    FixedContextComparison,
    FixedContextReplay,
    fixed_comparison_sha256,
    fixed_replay_sha256,
    prepare_fixed_context,
    run_fixed_context_comparison as compare_fixed_context,
)
from wsp.interventions.models import ActivationIntervention, SCHEMA_SPEC, intervention_sha256
from wsp.replay.models import ReplayExecutionConfig
from wsp.replay.plan import prepare_plan
from wsp.replay.sources import load_replay_source
from wsp.tools.event_schemas import (
    AttachEventEvidenceInput,
    CreateEventTriggerInput,
    CreateInvestigationDossierInput,
    DetectBehavioralEventsInput,
    DiscoverCapabilitiesInput,
    InspectInvestigationDossierInput,
    ListEventTriggersInput,
    PrepareFixedContextReplayInput,
    ResolveEventPositionInput,
    RunEventTriggeredInterventionInput,
    RunFixedContextComparisonInput,
)

EVENT_ARTIFACT_KIND = "behavioral_event"
TRIGGER_ARTIFACT_KIND = "event_trigger"
EVENT_RUN_ARTIFACT_KIND = "event_intervention"
FIXED_REPLAY_ARTIFACT_KIND = "fixed_context_replay"
FIXED_COMPARISON_ARTIFACT_KIND = "fixed_context_comparison"
DOSSIER_ARTIFACT_KIND = "investigation_dossier"
EVIDENCE_ARTIFACT_KIND = "investigation_evidence"


def _result(data: dict[str, Any], artifacts: list[Any] | None = None, warnings: list[str] | None = None):
    from wsp.tools.service import ToolResult

    return ToolResult(data=data, artifacts=list(artifacts or []), warnings=list(warnings or []))


def _require_workspace(service: Any) -> Any:
    if service.workspace is None:
        raise InputError(
            "this operation requires an initialized workspace; create one with 'wsp init --path <dir>', or restart 'wsp mcp' with --auto-init",
            details={"hint": "events, replays, comparisons, and dossiers are stored as artifacts"},
        )
    return service.workspace


def _save_event(workspace: Any, event: BehavioralEvent) -> ArtifactRef:
    event.checksums["event_sha256"] = event_sha256(event)
    return workspace.artifacts.put_json(
        EVENT_ARTIFACT_KIND,
        event.model_dump(mode="json"),
        metadata={
            "schema_version": event.schema_version,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "alignment": event.alignment.status,
            "trace_id": event.trace_id,
        },
    )


def _load_event(workspace: Any, artifact_id: str) -> tuple[BehavioralEvent, ArtifactRef]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != EVENT_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a behavioral event",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    event = BehavioralEvent.model_validate(workspace.artifacts.read_json(artifact_id))
    if event.schema_version != SCHEMA_EVENT:
        raise ArtifactError(
            "unsupported behavioral-event schema",
            details={"schema_version": event.schema_version},
        )
    expected = event.checksums.get("event_sha256")
    if expected is not None and expected != event_sha256(event):
        raise ArtifactError(
            "the behavioral event checksum does not match its content",
            details={"artifact_id": artifact_id},
        )
    return event, ref


def _load_trigger(workspace: Any, artifact_id: str) -> tuple[EventTrigger, ArtifactRef]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != TRIGGER_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not an event trigger",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    trigger = EventTrigger.model_validate(workspace.artifacts.read_json(artifact_id))
    if trigger.schema_version != SCHEMA_TRIGGER:
        raise ArtifactError(
            "unsupported event-trigger schema",
            details={"schema_version": trigger.schema_version},
        )
    return trigger, ref


def _load_replay(workspace: Any, artifact_id: str) -> tuple[FixedContextReplay, ArtifactRef]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != FIXED_REPLAY_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a fixed-context replay",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    replay = FixedContextReplay.model_validate(workspace.artifacts.read_json(artifact_id))
    if replay.schema_version != SCHEMA_FIXED_REPLAY:
        raise ArtifactError(
            "unsupported fixed-context replay schema",
            details={"schema_version": replay.schema_version},
        )
    expected = replay.checksums.get("replay_sha256")
    if expected is not None and expected != fixed_replay_sha256(replay):
        raise ArtifactError(
            "the fixed-context replay checksum does not match its content",
            details={"artifact_id": artifact_id},
        )
    return replay, ref


def _load_comparison(workspace: Any, artifact_id: str) -> tuple[FixedContextComparison, ArtifactRef]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != FIXED_COMPARISON_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a fixed-context comparison",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    comparison = FixedContextComparison.model_validate(
        workspace.artifacts.read_json(artifact_id)
    )
    if comparison.schema_version != SCHEMA_FIXED_COMPARISON:
        raise ArtifactError(
            "unsupported fixed-context comparison schema",
            details={"schema_version": comparison.schema_version},
        )
    expected = comparison.checksums.get("comparison_sha256")
    if expected is not None and expected != fixed_comparison_sha256(comparison):
        raise ArtifactError(
            "the fixed-context comparison checksum does not match its content",
            details={"artifact_id": artifact_id},
        )
    return comparison, ref


def _load_dossier(workspace: Any, artifact_id: str) -> tuple[InvestigationDossier, ArtifactRef]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != DOSSIER_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not an investigation dossier",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    dossier = InvestigationDossier.model_validate(
        workspace.artifacts.read_json(artifact_id)
    )
    if dossier.schema_version != SCHEMA_DOSSIER:
        raise ArtifactError(
            "unsupported investigation-dossier schema",
            details={"schema_version": dossier.schema_version},
        )
    expected = dossier.checksums.get("dossier_sha256")
    if expected is not None and expected != dossier_sha256(dossier):
        raise ArtifactError(
            "the dossier checksum does not match its content",
            details={"artifact_id": artifact_id},
        )
    return dossier, ref


def _load_intervention(workspace: Any, request: Any) -> ActivationIntervention:
    if request.intervention is not None:
        spec = ActivationIntervention.model_validate(request.intervention)
        if spec.schema_version != SCHEMA_SPEC:
            raise ArtifactError(
                "unsupported activation-intervention schema",
                details={"schema_version": spec.schema_version},
            )
        return spec
    from wsp.tools.intervention_handlers import _load_intervention as load

    spec, _ref = load(workspace, request.intervention_artifact_id)
    return spec


def _model(request: Any) -> Any:
    return load_replay_source(request.source)


def _trace_tokens(workspace: Any, trace_id: str | None) -> list[int]:
    if trace_id is None:
        raise InputError("a trace id is required")
    trace = TraceStore(workspace).load(trace_id)
    return [int(value) for value in (trace.input.token_ids or [])]


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #


def detect_behavioral_events(
    service: Any, request: DetectBehavioralEventsInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    traces = TraceStore(workspace)
    trace = traces.load(request.trace_id)
    record = traces.get_record(request.trace_id)
    prompt_length = len(trace.input.token_ids or [])
    full_tokens = [int(value) for value in (trace.input.token_ids or [])] + [
        int(token.token_id) for token in trace.tokens
    ]
    if not trace.input.token_ids:
        raise InputError(
            "the trace did not record exact input token identifiers; event positions cannot be "
            "aligned to an absolute sequence",
            details={"trace_id": trace.trace_id},
        )
    if len(full_tokens) > budget.max_event_sequence_tokens:
        from wsp.core.errors import ResourceLimitError

        raise ResourceLimitError(
            "the trace exceeds the event-detection token budget",
            details={"tokens": len(full_tokens), "limit": budget.max_event_sequence_tokens},
        )

    events: list[BehavioralEvent]
    if request.finding is not None or request.finding_index is not None:
        if request.finding is not None:
            from wsp.behavior.models import Finding

            finding = Finding.model_validate(request.finding)
        else:
            from wsp.behavior.detectors import default_registry, scan_trace

            scan = scan_trace(trace, budget=budget)
            findings = [item for detector_run in scan.runs for item in detector_run.findings]
            index = int(request.finding_index or 0)
            if index >= len(findings):
                raise NotFoundError(
                    "the trace has no finding at that index",
                    details={"finding_index": index, "findings": len(findings)},
                )
            finding = findings[index]
        event = resolve_finding_event(
            trace,
            finding,
            finding_index=int(request.finding_index or 0),
            trace_artifact_id=record.artifact.artifact_id,
        )
        events = [event]
    elif request.text_pattern is not None:
        events = detect_text_pattern(
            trace,
            request.text_pattern,
            prompt_length=prompt_length,
            trace_artifact_id=record.artifact.artifact_id,
            max_events=min(request.max_events, budget.max_events_per_trace),
        )
    elif request.manual_position is not None or request.manual_start is not None:
        events = [
            manual_event(
                full_tokens,
                prompt_length=prompt_length,
                position=request.manual_position,
                start=request.manual_start,
                end=request.manual_end,
                trace_id=trace.trace_id,
                trace_artifact_id=record.artifact.artifact_id,
                note=request.manual_note,
            )
        ]
    else:
        assert request.event_definition is not None
        definition = EventTriggerDefinition.model_validate(request.event_definition)
        if definition.kind in ("detector_finding", "text_pattern"):
            raise InputError(
                "detector findings and text patterns need their own source fields",
                details={"kind": definition.kind},
            )
        events = detect_events(
            full_tokens,
            definition=definition,
            prompt_length=prompt_length,
            observation_mode=request.observation_mode,
            trace_id=trace.trace_id,
            trace_artifact_id=record.artifact.artifact_id,
            budget=budget,
            max_events=min(request.max_events, budget.max_events_per_trace),
        )
    if len(events) > budget.max_events_per_trace:
        from wsp.core.errors import ResourceLimitError

        raise ResourceLimitError(
            "the request produced more events than the budget allows",
            details={"events": len(events), "limit": budget.max_events_per_trace},
        )
    artifacts: list[ArtifactRef] = []
    saved: list[dict[str, Any]] = []
    for event in events:
        ref = _save_event(workspace, event) if request.save else None
        if ref is not None:
            artifacts.append(ref)
        payload = event.model_dump(mode="json")
        payload["artifact_id"] = ref.artifact_id if ref else None
        saved.append(payload)
    first = saved[0] if saved else None
    return _result(
        {
            "events": saved,
            "event_id": first.get("event_id") if first else None,
            "event_artifact_id": first.get("artifact_id") if first else None,
            "alignment": first.get("alignment") if first else None,
            "count": len(saved),
            "selection": {
                "trace_id": trace.trace_id,
                "prompt_length": prompt_length,
                "full_length": len(full_tokens),
            },
            "interpretation_note": (
                "Events are deterministic observations. They are not fault labels, and a "
                "repetition event is not evidence of a reasoning loop."
            ),
        },
        artifacts,
    )


def create_event_trigger(
    service: Any, request: CreateEventTriggerInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    trigger = EventTrigger.model_validate(request.trigger)
    validate_trigger_for_run(trigger, budget=budget)
    artifacts: list[ArtifactRef] = []
    ref = None
    if request.save:
        ref = workspace.artifacts.put_json(
            TRIGGER_ARTIFACT_KIND,
            trigger.model_dump(mode="json"),
            metadata={
                "schema_version": trigger.schema_version,
                "trigger_id": trigger.trigger_id,
                "mode": trigger.mode,
                "event_type": trigger.definition.kind,
            },
        )
        artifacts.append(ref)
    data = trigger.model_dump(mode="json")
    data["trigger_sha256"] = trigger_sha256(trigger)
    data["artifact_id"] = ref.artifact_id if ref else None
    data["online_semantics"] = (
        "an event observed after generated token k is eligible no earlier than step k+1"
    )
    return _result(data, artifacts)


def list_event_triggers(
    service: Any, request: ListEventTriggersInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    items = [
        {
            "artifact_id": ref.artifact_id,
            "trigger_id": (ref.metadata or {}).get("trigger_id"),
            "mode": (ref.metadata or {}).get("mode"),
            "event_type": (ref.metadata or {}).get("event_type"),
            "created_at": ref.created_at if hasattr(ref, "created_at") else None,
        }
        for ref in workspace.artifacts.list()
        if ref.kind == TRIGGER_ARTIFACT_KIND
    ][: request.limit]
    return _result({"triggers": items, "count": len(items)})


def resolve_event_position(
    service: Any, request: ResolveEventPositionInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    event, ref = _load_event(workspace, request.event_artifact_id)
    data = event.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id
    data["timing"] = {
        "token_position": event.alignment.token_position,
        "consumed_position": event.alignment.consumed_position,
        "generation_step": event.alignment.generation_step,
        "earliest_online_step": event.alignment.earliest_online_step,
        "note": (
            "The event token cannot affect the forward that already produced it; the earliest "
            "online intervention is normally the following forward."
        ),
    }
    return _result(data)


# --------------------------------------------------------------------------- #
# fixed context
# --------------------------------------------------------------------------- #


def prepare_fixed_context_replay(
    service: Any, request: PrepareFixedContextReplayInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    traces = TraceStore(workspace)
    trace = traces.load(request.trace_id)
    record = traces.get_record(request.trace_id)
    events = [
        _load_event(workspace, artifact_id)[0]
        for artifact_id in request.event_artifact_ids
    ]
    execution = (
        ReplayExecutionConfig.model_validate(request.execution)
        if request.execution
        else None
    )
    replay = prepare_fixed_context(
        trace,
        trace_artifact_id=record.artifact.artifact_id,
        trace_artifact_sha256=record.artifact.sha256,
        events=events,
        replay_step=request.replay_step,
        target_position=request.target_position,
        execution=execution,
        budget=budget,
    )
    artifacts: list[ArtifactRef] = []
    ref = None
    if request.save:
        ref = workspace.artifacts.put_json(
            FIXED_REPLAY_ARTIFACT_KIND,
            replay.model_dump(mode="json"),
            metadata={
                "schema_version": replay.schema_version,
                "replay_id": replay.replay_id,
                "trace_id": replay.trace_id,
                "tokens": replay.sequence.full_length,
                "events": len(replay.event_ids),
            },
        )
        artifacts.append(ref)
    data = replay.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id if ref else None
    data["prefix"] = {
        "prompt_length": replay.sequence.prompt_length,
        "generated_length": replay.sequence.generated_length,
        "full_length": replay.sequence.full_length,
        "sequence_sha256": replay.sequence.sequence_sha256,
    }
    return _result(data, artifacts, replay.warnings)


def run_fixed_context_comparison(
    service: Any, request: RunFixedContextComparisonInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    replay, replay_ref = _load_replay(workspace, request.replay_artifact_id)
    spec = _load_intervention(workspace, request)
    model = _model(request)
    comparison = compare_fixed_context(
        replay,
        spec,
        model,
        budget=budget,
        logit_positions=request.logit_positions,
        measure_modules=request.measure_modules,
        measure_positions=request.measure_positions,
        use_donor_store=workspace.artifacts,
        case_id=request.case_id,
        intervention_artifact_id=request.intervention_artifact_id,
        continuation=request.continuation,
        identity=request.identity,
    )
    artifacts: list[ArtifactRef] = [replay_ref]
    ref = None
    if request.save:
        ref = workspace.artifacts.put_json(
            FIXED_COMPARISON_ARTIFACT_KIND,
            comparison.model_dump(mode="json"),
            metadata={
                "schema_version": comparison.schema_version,
                "comparison_id": comparison.comparison_id,
                "trace_id": comparison.trace_id,
                "validity": comparison.validity.status,
                "identity": comparison.identity_control.agreement,
                "events": len(comparison.event_ids),
            },
        )
        artifacts.append(ref)
    data = comparison.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id if ref else None
    data["replay_artifact_id"] = replay_ref.artifact_id
    data["summary"] = {
        "validity": comparison.validity.status,
        "identity_control": comparison.identity_control.agreement,
        "identity_passed": comparison.identity_control.passed,
        "measurements": len(comparison.measurements),
        "downstream_modules": len(comparison.downstream),
        "continuation_divergence": None
        if comparison.continuation is None
        else comparison.continuation.diff.first_divergence,
    }
    return _result(data, artifacts, comparison.warnings)


# --------------------------------------------------------------------------- #
# event-triggered execution
# --------------------------------------------------------------------------- #


def run_event_triggered_intervention(
    service: Any, request: RunEventTriggeredInterventionInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    if request.trigger is not None:
        trigger = EventTrigger.model_validate(request.trigger)
    else:
        trigger, _ref = _load_trigger(workspace, request.trigger_artifact_id)
    spec = _load_intervention(workspace, request)
    model = _model(request)

    if trigger.mode == "online":
        if request.event_artifact_id is not None:
            raise InputError(
                "an online trigger must not select an already recorded event; use mode='retrospective'"
            )
        prompt = (
            [int(value) for value in request.token_ids]
            if request.token_ids
            else _trace_tokens(workspace, request.trace_id)
        )
        record = run_online_event_trigger(
            trigger,
            spec,
            model,
            prompt,
            max_new_tokens=request.max_new_tokens,
            eos_token_id=request.eos_token_id,
            budget=budget,
            identity=request.identity,
            case_id=request.case_id,
        )
        artifacts: list[ArtifactRef] = []
        ref = None
        if request.save:
            ref = workspace.artifacts.put_json(
                EVENT_RUN_ARTIFACT_KIND,
                record.model_dump(mode="json"),
                metadata={
                    "schema_version": record.schema_version,
                    "run_id": record.run_id,
                    "mode": "online",
                    "validity": record.validity.status,
                    "fired": record.fired_events,
                    "not_reached": record.not_reached_events,
                },
            )
            artifacts.append(ref)
        data = record.model_dump(mode="json")
        data["artifact_id"] = ref.artifact_id if ref else None
        data["mode_note"] = (
            "Online: every fired step is strictly after the event's observation step."
        )
        return _result(data, artifacts, record.warnings)

    # Retrospective: reconstruct the recorded prefix and run a fixed-context
    # comparison at the explicitly selected replay step.  This is deliberately
    # offline and is labelled as such.
    if request.event_artifact_id is None:
        raise InputError("a retrospective trigger requires event_artifact_id")
    event, event_ref = _load_event(workspace, request.event_artifact_id)
    validate_trigger_for_run(trigger, budget=budget, available_event=event)
    if request.trace_id is None:
        raise InputError("a retrospective run reconstructs a recorded trace; trace_id is required")
    trace = TraceStore(workspace).load(request.trace_id)
    trace_record = TraceStore(workspace).get_record(request.trace_id)
    replay = prepare_fixed_context(
        trace,
        trace_artifact_id=trace_record.artifact.artifact_id,
        trace_artifact_sha256=trace_record.artifact.sha256,
        events=[event],
        replay_step=trigger.replay_step,
        budget=budget,
    )
    comparison = compare_fixed_context(
        replay,
        spec,
        model,
        budget=budget,
        case_id=request.case_id,
        intervention_artifact_id=request.intervention_artifact_id,
        identity=request.identity,
    )
    artifacts: list[ArtifactRef] = [event_ref]
    replay_ref = None
    comparison_ref = None
    if request.save:
        replay_ref = workspace.artifacts.put_json(
            FIXED_REPLAY_ARTIFACT_KIND,
            replay.model_dump(mode="json"),
            metadata={
                "schema_version": replay.schema_version,
                "replay_id": replay.replay_id,
                "trace_id": replay.trace_id,
            },
        )
        comparison_ref = workspace.artifacts.put_json(
            FIXED_COMPARISON_ARTIFACT_KIND,
            comparison.model_dump(mode="json"),
            metadata={
                "schema_version": comparison.schema_version,
                "comparison_id": comparison.comparison_id,
                "trace_id": comparison.trace_id,
                "validity": comparison.validity.status,
                "mode": "retrospective",
            },
        )
        artifacts.extend([replay_ref, comparison_ref])
    data = {
        "mode": "retrospective",
        "trigger": trigger.model_dump(mode="json"),
        "trigger_sha256": trigger_sha256(trigger),
        "event": event.model_dump(mode="json"),
        "event_artifact_id": event_ref.artifact_id,
        "replay": replay.model_dump(mode="json"),
        "replay_artifact_id": replay_ref.artifact_id if replay_ref else None,
        "comparison": comparison.model_dump(mode="json"),
        "comparison_artifact_id": comparison_ref.artifact_id if comparison_ref else None,
        "mode_note": (
            "Retrospective: the replay step was selected explicitly. This is not an online "
            "prediction and cannot establish that an online trigger would have fired there."
        ),
        "summary": {
            "validity": comparison.validity.status,
            "identity_control": comparison.identity_control.agreement,
            "consumed_position": replay.consumed_position,
        },
    }
    return _result(data, artifacts, comparison.warnings)


# --------------------------------------------------------------------------- #
# dossiers and evidence
# --------------------------------------------------------------------------- #


def create_investigation_dossier(
    service: Any, request: CreateInvestigationDossierInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    events = [_load_event(workspace, artifact_id)[0] for artifact_id in request.event_artifact_ids]
    comparison = None
    comparison_ref = None
    if request.fixed_context_comparison_artifact_id is not None:
        comparison, comparison_ref = _load_comparison(
            workspace, request.fixed_context_comparison_artifact_id
        )
    elif request.counterfactual_comparison_artifact_id is not None:
        from wsp.tools.intervention_handlers import _load_comparison as load_cf

        comparison, comparison_ref = load_cf(
            workspace, request.counterfactual_comparison_artifact_id
        )
    dossier = build_dossier(
        title=request.title,
        case_id=request.case_id,
        trace_id=request.trace_id,
        events=events,
        comparison=comparison,
        comparison_artifact_id=comparison_ref.artifact_id if comparison_ref else None,
        hypotheses=request.hypotheses,
        extra_limitations=request.extra_limitations,
        summary=request.summary,
    )
    if len(dossier.artifact_refs) > budget.max_dossier_artifacts:
        raise InputError(
            "the dossier references more artifacts than the budget allows",
            details={"references": len(dossier.artifact_refs), "limit": budget.max_dossier_artifacts},
        )
    artifacts: list[ArtifactRef] = []
    ref = None
    if request.save:
        ref = workspace.artifacts.put_json(
            DOSSIER_ARTIFACT_KIND,
            dossier.model_dump(mode="json"),
            metadata={
                "schema_version": dossier.schema_version,
                "dossier_id": dossier.dossier_id,
                "case_id": dossier.case_id,
                "validity": dossier.metadata.get("validity_status"),
                "measurements": len(dossier.measurements),
            },
        )
        artifacts.append(ref)
        dossier.checksums["artifact_sha256"] = ref.sha256
    if request.case_id is not None and ref is not None:
        traces = TraceStore(workspace)
        store = CaseStore(workspace, traces=traces)
        case = store.load(request.case_id)
        now = utc_now_iso()
        from wsp.investigation.evidence import ArtifactLink, EventRef, FixedContextRef

        evidence = build_evidence(
            title=request.title,
            case_id=request.case_id,
            summary=request.summary,
            observation_trace_id=request.trace_id,
            events=[
                EventRef(
                    event_id=event.event_id,
                    artifact_id=(
                        request.event_artifact_ids[index]
                        if index < len(request.event_artifact_ids)
                        else None
                    ),
                    event_type=event.event_type,
                    trace_id=event.trace_id,
                    token_position=event.token_position,
                    generation_step=event.generation_step,
                    alignment_status=event.alignment.status,
                    observation_mode=event.observation_mode,
                )
                for index, event in enumerate(events)
            ],
            fixed_context_comparisons=[]
            if comparison is None or comparison_ref is None
            else [
                FixedContextRef(
                    comparison_id=comparison.comparison_id,
                    artifact_id=comparison_ref.artifact_id,
                    replay_id=getattr(comparison, "replay_id", None),
                    trace_id=getattr(comparison, "trace_id", None),
                    validity_status=getattr(
                        getattr(comparison, "validity", None), "status", None
                    ),
                    identity_agreement=getattr(
                        getattr(comparison, "identity_control", None), "agreement", None
                    ),
                    consumed_position=getattr(comparison, "consumed_position", None),
                )
            ],
            dossiers=[
                DossierRef(
                    dossier_id=dossier.dossier_id,
                    artifact_id=ref.artifact_id,
                    title=dossier.title,
                    case_id=dossier.case_id,
                    validity_status=dossier.metadata.get("validity_status"),
                )
            ],
            notes=[],
        )
        evidence_ref = workspace.artifacts.put_json(
            EVIDENCE_ARTIFACT_KIND,
            evidence.model_dump(mode="json"),
            metadata={
                "schema_version": evidence.schema_version,
                "evidence_id": evidence.evidence_id,
                "case_id": evidence.case_id,
                "events": len(evidence.events),
            },
        )
        artifacts.append(evidence_ref)
        case.investigations.append(
            InvestigationLink(
                evidence_id=evidence.evidence_id,
                artifact_id=evidence_ref.artifact_id,
                title=request.title,
                created_at=now,
                event_ids=[event.event_id for event in events],
                fixed_context_ids=[]
                if comparison is None
                else [
                    value
                    for value in [getattr(comparison, "comparison_id", None)]
                    if value
                ],
                dossier_ids=[dossier.dossier_id],
                note=request.summary,
            )
        )
        case.updated_at = now
        case.revision += 1
        store.save(case)
    data = dossier.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id if ref else None
    data["attached_to_case"] = bool(request.case_id and ref)
    return _result(data, artifacts)


def inspect_investigation_dossier(
    service: Any, request: InspectInvestigationDossierInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    dossier, ref = _load_dossier(workspace, request.dossier_artifact_id)
    data = dossier.model_dump(mode="json")
    if not request.include_measurements:
        data["measurements"] = []
    else:
        data["measurements"] = data["measurements"][: request.max_entries]
    data["artifact_id"] = ref.artifact_id
    data["counts"] = {
        "observations": len(dossier.observations),
        "measurements": len(dossier.measurements),
        "controls": len(dossier.controls),
        "hypotheses": len(dossier.hypotheses),
        "next_operations": len(dossier.next_operations),
    }
    return _result(data)


def attach_event_evidence(
    service: Any, request: AttachEventEvidenceInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    traces = TraceStore(workspace)
    if request.trace_id is not None:
        traces.get_record(request.trace_id)
    events = []
    for artifact_id in request.event_artifact_ids:
        event, ref = _load_event(workspace, artifact_id)
        events.append(
            EventRef(
                event_id=event.event_id,
                artifact_id=ref.artifact_id,
                event_type=event.event_type,
                trace_id=event.trace_id,
                token_position=event.token_position,
                generation_step=event.generation_step,
                alignment_status=event.alignment.status,
                observation_mode=event.observation_mode,
                note=None,
            )
        )
    replays = []
    for artifact_id in request.fixed_context_replay_artifact_ids:
        replay, ref = _load_replay(workspace, artifact_id)
        replays.append(
            __import__("wsp.investigation.evidence", fromlist=["ArtifactLink"]).ArtifactLink(
                artifact_id=ref.artifact_id,
                record_id=replay.replay_id,
                trace_id=replay.trace_id,
                schema_version=replay.schema_version,
            )
        )
    comparisons = []
    for artifact_id in request.fixed_context_comparison_artifact_ids:
        comparison, ref = _load_comparison(workspace, artifact_id)
        delta = (
            comparison.measurements[0].intervention_delta
            if comparison.measurements
            else None
        )
        comparisons.append(
            FixedContextRef(
                comparison_id=comparison.comparison_id,
                artifact_id=ref.artifact_id,
                replay_id=comparison.replay_id,
                trace_id=comparison.trace_id,
                validity_status=comparison.validity.status,
                identity_agreement=comparison.identity_control.agreement,
                consumed_position=comparison.consumed_position,
                intervention_delta=delta,
            )
        )
    recipe_runs = []
    for artifact_id in request.recipe_run_artifact_ids:
        ref, _path = workspace.artifacts.get(artifact_id)
        if ref.kind != "recipe_run":
            raise ArtifactError(
                "the referenced artifact is not a recipe run",
                details={"artifact_id": artifact_id, "kind": ref.kind},
            )
        from wsp.recipes.models import RecipeRun

        recipe_run = RecipeRun.model_validate(workspace.artifacts.read_json(artifact_id))
        recipe_runs.append(
            RecipeRunRef(
                run_id=recipe_run.run_id,
                artifact_id=ref.artifact_id,
                recipe_id=recipe_run.recipe_id,
                status=recipe_run.status,
                steps_completed=sum(1 for item in recipe_run.steps if item.status == "completed"),
                steps_failed=sum(1 for item in recipe_run.steps if item.status == "failed"),
                dossier_id=recipe_run.dossier_id,
            )
        )
    dossiers = []
    for artifact_id in request.dossier_artifact_ids:
        dossier, ref = _load_dossier(workspace, artifact_id)
        dossiers.append(
            DossierRef(
                dossier_id=dossier.dossier_id,
                artifact_id=ref.artifact_id,
                title=dossier.title,
                case_id=dossier.case_id,
                validity_status=dossier.metadata.get("validity_status"),
            )
        )
    hypotheses = [
        HypothesisNote(
            hypothesis_id=new_hypothesis_id(),
            statement=item.get("statement", ""),
            proposed_by=item.get("proposed_by"),
            rationale=item.get("rationale"),
        )
        for item in request.hypotheses
    ]
    evidence = build_evidence(
        title=request.title,
        case_id=request.case_id,
        summary=request.summary,
        observation_trace_id=request.trace_id,
        events=events,
        replays=replays,
        fixed_context_comparisons=comparisons,
        recipe_runs=recipe_runs,
        dossiers=dossiers,
        hypotheses=hypotheses,
        notes=list(request.notes),
        warnings=[],
    )
    artifacts: list[ArtifactRef] = []
    ref = None
    if request.save:
        ref = workspace.artifacts.put_json(
            EVIDENCE_ARTIFACT_KIND,
            evidence.model_dump(mode="json"),
            metadata={
                "schema_version": evidence.schema_version,
                "evidence_id": evidence.evidence_id,
                "case_id": evidence.case_id,
                "events": len(evidence.events),
                "fixed_context_comparisons": len(evidence.fixed_context_comparisons),
            },
        )
        artifacts.append(ref)
        evidence.checksums["artifact_sha256"] = ref.sha256
    link = None
    if request.case_id is not None and ref is not None:
        store = CaseStore(workspace, traces=traces)
        case = store.load(request.case_id)
        now = utc_now_iso()
        link = InvestigationLink(
            evidence_id=evidence.evidence_id,
            artifact_id=ref.artifact_id,
            title=request.title,
            created_at=now,
            event_ids=[item.event_id for item in evidence.events],
            fixed_context_ids=[
                item.comparison_id for item in evidence.fixed_context_comparisons
            ],
            recipe_run_ids=[item.run_id for item in evidence.recipe_runs],
            dossier_ids=[item.dossier_id for item in evidence.dossiers],
            note=request.summary,
        )
        case.investigations.append(link)
        case.updated_at = now
        case.revision += 1
        store.save(case)
    data = evidence.model_dump(mode="json")
    data["evidence_artifact_id"] = ref.artifact_id if ref else None
    data["attached_to_case"] = link is not None
    return _result(data, artifacts)


def discover_capabilities(
    service: Any, request: DiscoverCapabilitiesInput, budget: Budget, run: Any
):
    from wsp.tools.discovery import capability_overview

    data = capability_overview(
        service,
        query=request.query,
        group=request.group,
        detail=request.detail,
        include_recipes=request.include_recipes,
        max_operations=request.max_operations,
    )
    return _result(data)


def _definitions() -> list[tuple]:
    return [
        (
            "discover_capabilities",
            "Deterministically find relevant operations and recipes for a question, with progressive disclosure.",
            DiscoverCapabilitiesInput,
            discover_capabilities,
            "A bounded capability overview with groups, matching operations, recipes, and next operations.",
            False,
        ),
        (
            "detect_behavioral_events",
            "Resolve a token pattern, detector finding, text match, or manual mark to an exact behavioral event.",
            DetectBehavioralEventsInput,
            detect_behavioral_events,
            "Versioned behavioral events with absolute token positions, generation steps, and alignment status.",
            True,
        ),
        (
            "resolve_event_position",
            "Read a stored behavioral event and report its exact token position, generation step, and earliest online intervention step.",
            ResolveEventPositionInput,
            resolve_event_position,
            "The event record plus explicit timing semantics.",
            True,
        ),
        (
            "create_event_trigger",
            "Create a validated, declarative online or retrospective event trigger with delay and budget limits.",
            CreateEventTriggerInput,
            create_event_trigger,
            "A versioned event trigger and its artifact reference.",
            True,
        ),
        (
            "list_event_triggers",
            "List stored event triggers with their mode and event type.",
            ListEventTriggersInput,
            list_event_triggers,
            "A bounded trigger listing.",
            True,
        ),
        (
            "prepare_fixed_context_replay",
            "Reconstruct the exact recorded token prefix for a fixed-context experiment.",
            PrepareFixedContextReplayInput,
            prepare_fixed_context_replay,
            "A fixed-context replay with token digests, alignment checks, and limitations.",
            True,
        ),
        (
            "run_fixed_context_comparison",
            "Run baseline, mandatory identity control, and one activation intervention on identical token identifiers.",
            RunFixedContextComparisonInput,
            run_fixed_context_comparison,
            "A fixed-context comparison with measurements, validity, and optional continuation divergence.",
            True,
        ),
        (
            "run_event_triggered_intervention",
            "Run an online event-triggered intervention, or a retrospective fixed-context investigation of a recorded event.",
            RunEventTriggeredInterventionInput,
            run_event_triggered_intervention,
            "An event-intervention record (online) or replay/comparison pair (retrospective), with validity.",
            True,
        ),
        (
            "create_investigation_dossier",
            "Build a concise agent-readable dossier separating observations, measurements, controls, hypotheses, and limitations.",
            CreateInvestigationDossierInput,
            create_investigation_dossier,
            "An investigation dossier, optionally attached to a failure case.",
            True,
        ),
        (
            "inspect_investigation_dossier",
            "Read a stored investigation dossier with a bounded measurement page.",
            InspectInvestigationDossierInput,
            inspect_investigation_dossier,
            "The dossier, its counts, and its next-operation suggestions.",
            True,
        ),
        (
            "attach_event_evidence",
            "Attach events, fixed-context replays/comparisons, recipe runs, and dossiers to a failure case.",
            AttachEventEvidenceInput,
            attach_event_evidence,
            "An investigation-evidence record and the updated case link.",
            True,
        ),
    ]


def register_event_tools(service: Any) -> None:
    from wsp.tools.service import ToolDefinition

    def bind(function: Any) -> Any:
        def handler(request: Any, budget: Budget, run: Any):
            return function(service, request, budget, run)

        return handler

    for name, description, schema, function, output, needs_workspace in _definitions():
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
    "EVENT_ARTIFACT_KIND",
    "TRIGGER_ARTIFACT_KIND",
    "EVENT_RUN_ARTIFACT_KIND",
    "FIXED_REPLAY_ARTIFACT_KIND",
    "FIXED_COMPARISON_ARTIFACT_KIND",
    "DOSSIER_ARTIFACT_KIND",
    "register_event_tools",
]
