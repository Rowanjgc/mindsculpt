"""Event-triggered intervention scheduling.

Two execution modes are deliberately separate:

``online``
    The model generates normally.  A trigger condition is evaluated after each
    emitted token.  An event observed after token ``k`` becomes eligible at the
    earliest at generation step ``k + 1 + delay_steps``.  The scheduler cannot
    use information from tokens that have not been generated yet, so an
    intervention is never reported as fired before its observation.

``retrospective``
    A previously recorded event is selected explicitly.  The investigator names
    a replay step from the existing trace.  That step may precede the event,
    because the experiment is a deliberate reconstruction.  The record is
    labelled retrospective and is never presented as an online prediction.

The online executor applies the same :class:`ActivationIntervention` operations
as the V1.4 plan engine.  It does not introduce a second patching mechanism.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal, Sequence

import torch
from pydantic import Field, model_validator

from wsp.behavior.events import (
    SCHEMA_EVENT,
    BehavioralEvent,
    EventTriggerDefinition,
    new_event_run_id,
    new_trigger_id,
)
from wsp.core.budget import Budget
from wsp.core.errors import (
    CompatibilityError,
    ExecutionError,
    InputError,
    ResourceLimitError,
    WspError,
)
from wsp.core.hashing import sha256_bytes
from wsp.core.time import utc_now_iso
from wsp.interventions.engine import (
    _check_logits,
    _spec_variant,
    describe_model,
    retarget_spec_execution,
    run_pass,
    torch_module,
)
from wsp.interventions.models import (
    ActivationIntervention,
    IdentityControl,
    InterventionApplication,
    ValidationCheck,
    ValidityReport,
)
from wsp.interventions.plan_engine import (
    mutable_state_fingerprint,
    installed_hook_count,
)
from wsp.investigation.models import StrictModel
from wsp.replay.engine import parameter_fingerprint

SCHEMA_TRIGGER = "wsp.event-trigger/1.0"
SCHEMA_EVENT_INTERVENTION = "wsp.event-intervention/1.0"

TriggerMode = Literal["online", "retrospective"]
TriggerState = Literal[
    "observed", "scheduled", "fired", "not_reached", "skipped", "rejected"
]

EVENT_INTERVENTION_LIMITATIONS = [
    "An online trigger can act only after the event token has been generated; it cannot predict the event.",
    "An observed event that has no subsequent forward is reported as not reached, not as a fired intervention.",
    "A triggered numerical difference is not a diagnosis, a repair, or evidence of a mechanism.",
    "The trigger measures tokens exactly as configured; it does not judge whether repetition is meaningful.",
]


def _new(prefix: str) -> str:
    return f"{prefix}_{__import__('secrets').token_hex(8)}"


class EventTrigger(StrictModel):
    """A declarative event-to-intervention schedule."""

    schema_version: Literal[SCHEMA_TRIGGER] = SCHEMA_TRIGGER
    trigger_id: str = Field(pattern=r"^et_[0-9a-f]{16}$")
    created_at: str = Field(default_factory=utc_now_iso)
    name: str | None = None
    mode: TriggerMode
    definition: EventTriggerDefinition
    #: Retrospective triggers name the observed event they investigate.
    event_id: str | None = None
    #: Retrospective triggers may select a replay step explicitly.
    replay_step: int | None = Field(default=None, ge=0)
    delay_steps: int = Field(default=0, ge=0)
    steps_after_event: int = Field(default=1, ge=1, le=64)
    max_triggers: int = Field(default=1, ge=1, le=64)
    fire_on: Literal["once", "every_occurrence"] = "once"
    max_observed_events: int = Field(default=8, ge=1, le=256)
    position_mode: Literal["absolute", "last"] = "last"
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> "EventTrigger":
        if self.mode == "online":
            if self.event_id is not None or self.replay_step is not None:
                raise ValueError(
                    "an online trigger must not reference an already recorded event or replay step"
                )
        else:
            if self.event_id is None and self.replay_step is None:
                raise ValueError(
                    "a retrospective trigger requires an event_id or an explicit replay_step"
                )
            if self.delay_steps != 0:
                raise ValueError(
                    "a retrospective trigger selects an explicit replay step; a delay would be ambiguous"
                )
        if self.fire_on == "once" and self.max_triggers != 1:
            raise ValueError("fire_on='once' requires max_triggers=1")
        if self.steps_after_event > self.max_triggers and self.fire_on == "once":
            raise ValueError(
                "steps_after_event cannot exceed max_triggers for a once-only trigger"
            )
        return self


class EventApplicationRecord(StrictModel):
    generation_step: int = Field(ge=0)
    forward_length: int = Field(ge=0)
    module: str
    tensor_role: str
    operation: str
    rows: list[int] = Field(default_factory=list)
    tensor_shape: list[int] = Field(default_factory=list)
    dtype: str | None = None
    elements_modified: int = Field(default=0, ge=0)
    replacement_sha256: str | None = None
    dtype_cast: str | None = None
    execution_result: Literal["applied", "identity", "rejected", "skipped"] = "applied"
    rejection_reason: str | None = None


class EventInterventionRecord(StrictModel):
    """The full life of one observed event in the trigger scheduler."""

    event: BehavioralEvent
    state: TriggerState
    observation_step: int | None = Field(default=None, ge=0)
    earliest_eligible_step: int | None = Field(default=None, ge=0)
    scheduled_steps: list[int] = Field(default_factory=list)
    fired_steps: list[int] = Field(default_factory=list)
    applications: list[EventApplicationRecord] = Field(default_factory=list)
    no_subsequent_forward: bool = False
    rejection_reason: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _state(self) -> "EventInterventionRecord":
        if self.state == "fired" and not self.fired_steps:
            raise ValueError("a fired event must record at least one fired step")
        if self.state == "not_reached" and self.fired_steps:
            raise ValueError("a not-reached event cannot have fired steps")
        if self.state == "rejected" and not self.rejection_reason:
            raise ValueError("a rejected event requires a reason")
        return self


class EventCondition(StrictModel):
    condition: Literal["baseline", "identity", "intervention"]
    generated_token_ids: list[int] = Field(default_factory=list)
    logprobs: list[float] = Field(default_factory=list)
    top1_token_ids: list[int] = Field(default_factory=list)
    steps: int = Field(default=0, ge=0)
    termination_reason: Literal[
        "eos", "max_new_tokens", "max_sequence_length", "error"
    ] = "max_new_tokens"
    events_applied: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)


class EventInterventionRun(StrictModel):
    """One online event-triggered experiment."""

    schema_version: Literal[SCHEMA_EVENT_INTERVENTION] = SCHEMA_EVENT_INTERVENTION
    run_id: str = Field(pattern=r"^ei_[0-9a-f]{16}$")
    created_at: str = Field(default_factory=utc_now_iso)
    case_id: str | None = None
    trigger: EventTrigger
    trigger_sha256: str
    intervention_id: str
    intervention_sha256: str
    model: dict[str, Any] = Field(default_factory=dict)
    execution_mode: Literal["uncached-generation"] = "uncached-generation"
    backend: str = "torch_hooks"
    prompt_length: int = Field(ge=0)
    prompt_token_ids_sha256: str | None = None
    max_new_tokens: int = Field(ge=0)
    conditions: list[EventCondition] = Field(default_factory=list)
    events: list[EventInterventionRecord] = Field(default_factory=list)
    observed_events: int = Field(default=0, ge=0)
    scheduled_events: int = Field(default=0, ge=0)
    fired_events: int = Field(default=0, ge=0)
    not_reached_events: int = Field(default=0, ge=0)
    rejected_events: int = Field(default=0, ge=0)
    identity_control: IdentityControl = Field(default_factory=IdentityControl)
    stepwise: list[dict[str, Any]] = Field(default_factory=list)
    first_divergence: int | None = Field(default=None, ge=0)
    validity: ValidityReport = Field(default_factory=ValidityReport)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=lambda: list(EVENT_INTERVENTION_LIMITATIONS)
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)


def _canonical(payload: dict[str, Any], skip: str | None = None) -> str:
    data = json.loads(json.dumps(payload, default=str))
    if skip:
        data.get("checksums", {}).pop(skip, None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def trigger_sha256(trigger: EventTrigger) -> str:
    return "sha256:" + sha256_bytes(
        _canonical(trigger.model_dump(mode="json")).encode("utf-8")
    ).split(":", 1)[1]


def event_run_sha256(record: EventInterventionRun) -> str:
    return "sha256:" + sha256_bytes(
        _canonical(record.model_dump(mode="json"), "run_sha256").encode("utf-8")
    ).split(":", 1)[1]


def intervention_sha256(spec: ActivationIntervention) -> str:
    from wsp.interventions.models import intervention_sha256 as compute

    return compute(spec)


def _retarget(spec: ActivationIntervention, length: int, *, mode: str) -> ActivationIntervention:
    return retarget_spec_execution(
        spec, execution_mode=mode, sequence_length=length  # type: ignore[arg-type]
    )


def _last_position(spec: ActivationIntervention, length: int) -> ActivationIntervention:
    if spec.target.selector.positions == [length - 1]:
        return spec
    payload = spec.model_dump(mode="python")
    payload["target"]["selector"]["positions"] = [max(0, length - 1)]
    return ActivationIntervention.model_validate(payload)


def _logprob_row(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits[0, -1].to(torch.float64), dim=-1)


def _detect_new_events(
    trigger: EventTrigger,
    tokens: Sequence[int],
    *,
    prompt_length: int,
    last_position: int | None,
    budget: Budget,
    trace_id: str | None,
    trace_artifact_id: str | None,
) -> list[BehavioralEvent]:
    from wsp.behavior.events import detect_events

    remaining = max(0, trigger.max_observed_events)
    if remaining == 0:
        return []
    return detect_events(
        tokens,
        definition=trigger.definition,
        prompt_length=prompt_length,
        observation_mode="online",
        trace_id=trace_id,
        trace_artifact_id=trace_artifact_id,
        budget=budget,
        max_events=remaining,
        start_after=last_position,
    )


def _run_condition(
    *,
    condition: str,
    trigger: EventTrigger,
    spec: ActivationIntervention | None,
    operation: str | None,
    model: Any,
    prompt: Sequence[int],
    max_new_tokens: int,
    eos_token_id: int | None,
    budget: Budget,
    detect: bool,
    fixed_steps: Sequence[int] | None = None,    trace_id: str | None = None,
    trace_artifact_id: str | None = None,
) -> tuple[EventCondition, list[EventInterventionRecord], list[torch.Tensor], list[int]]:
    """Run one generation condition and return its events and step logits."""

    sequence = [int(value) for value in prompt]
    prompt_length = len(sequence)
    generated: list[int] = []
    logprobs: list[float] = []
    top1: list[int] = []
    step_logits: list[torch.Tensor] = []
    step_tokens: list[int] = []
    records: list[EventInterventionRecord] = []
    last_position: int | None = None
    scheduled: dict[int, list[EventInterventionRecord]] = {}
    trigger_count = 0
    event_count = 0
    termination = "max_new_tokens"
    errors: list[str] = []

    root = torch_module(model)
    was_training = root.training
    hook_before = installed_hook_count(root)
    fingerprint_before = parameter_fingerprint(root)
    state_before = mutable_state_fingerprint(root)
    root.eval()
    try:
        with torch.inference_mode():
            for step in range(max_new_tokens):
                armed = scheduled.get(step, [])
                forced = fixed_steps is not None and step in fixed_steps
                applied_here = (bool(armed) or forced) and spec is not None
                if applied_here:
                    assert spec is not None
                    effective = _retarget(spec, len(sequence), mode="uncached-continuation")
                    effective = effective.model_copy(update={"apply_on_calls": [0]})
                    if trigger.position_mode == "last":
                        effective = _last_position(effective, len(sequence))
                    if operation == "identity":
                        effective = _spec_variant(effective, "identity")
                    try:
                        outcome = run_pass(
                            effective,
                            model,
                            sequence,
                            condition=condition,
                            operation=operation,
                            apply=True,
                            budget=budget,
                        )
                        logits = outcome.logits
                        app = outcome.applications[0] if outcome.applications else None
                        for record in armed:
                            record.fired_steps.append(step)
                            record.state = "fired"
                            record.applications.append(
                                EventApplicationRecord(
                                    generation_step=step,
                                    forward_length=len(sequence),
                                    module=app.module if app else effective.target.module,
                                    tensor_role=app.tensor_role if app else effective.target.tensor_role,
                                    operation=app.operation if app else effective.operation,
                                    rows=list(app.rows) if app else [],
                                    tensor_shape=list(app.tensor_shape) if app else [],
                                    dtype=app.dtype if app else None,
                                    elements_modified=app.elements_modified if app else 0,
                                    replacement_sha256=app.replacement_sha256 if app else None,
                                    dtype_cast=app.dtype_cast if app else None,
                                    execution_result="identity"
                                    if operation == "identity"
                                    else "applied",
                                )
                            )
                    except WspError as exc:
                        for record in armed:
                            record.state = "rejected"
                            record.rejection_reason = f"{exc.code}: {exc.message}"
                            record.notes.append("the scheduled forward was rejected and the run aborted")
                        raise
                else:
                    logits = model.forward_token_ids(sequence)
                    assert spec is not None
                    logits = _check_logits(logits, spec=spec, budget=budget)

                row = _logprob_row(logits)
                token_id = int(torch.argmax(row).item())
                logprobs.append(float(row[token_id].item()))
                top1.append(token_id)
                step_logits.append(row.clone())
                step_tokens.append(int(sequence[-1]))
                generated.append(token_id)
                sequence.append(token_id)

                if detect:
                    new_events = _detect_new_events(
                        trigger,
                        sequence,
                        prompt_length=prompt_length,
                        last_position=last_position,
                        budget=budget,
                        trace_id=trace_id,
                        trace_artifact_id=trace_artifact_id,
                    )
                    for event in new_events:
                        if event.token_position is not None:
                            last_position = max(last_position or 0, event.token_position)
                        observation = (
                            None
                            if event.generation_step is None
                            else int(event.generation_step)
                        )
                        earliest = (
                            None if observation is None else observation + 1 + trigger.delay_steps
                        )
                        record = EventInterventionRecord(
                            event=event,
                            state="observed",
                            observation_step=observation,
                            earliest_eligible_step=earliest,
                        )
                        event_count += 1
                        if earliest is None:
                            record.state = "skipped"
                            record.rejection_reason = "the event has no exact token alignment"
                        elif trigger_count >= trigger.max_triggers or event_count > trigger.max_observed_events:
                            record.state = "skipped"
                            record.rejection_reason = (
                                "the trigger budget was exhausted before this event could fire"
                            )
                        elif fixed_steps is not None:
                            allowed = [value for value in fixed_steps if value > observation]
                            if not allowed:
                                record.state = "not_reached"
                                record.no_subsequent_forward = True
                                record.notes.append(
                                    "the identity condition uses the intervention schedule; "
                                    "no eligible forward remained"
                                )
                            else:
                                record.scheduled_steps = [allowed[0]]
                                record.state = "scheduled"
                                scheduled.setdefault(allowed[0], []).append(record)
                                trigger_count += 1
                        elif earliest >= max_new_tokens:
                            record.state = "not_reached"
                            record.no_subsequent_forward = True
                            record.notes.append(
                                "generation ends before the earliest eligible forward; "
                                "no intervention was attempted"
                            )
                        else:
                            end = min(
                                max_new_tokens,
                                earliest + trigger.steps_after_event,
                            )
                            record.scheduled_steps = list(range(earliest, end))
                            record.state = "scheduled"
                            for value in record.scheduled_steps:
                                scheduled.setdefault(value, []).append(record)
                            trigger_count += 1
                        records.append(record)

                if eos_token_id is not None and token_id == eos_token_id:
                    termination = "eos"
                    break
                if len(sequence) >= budget.max_replay_sequence_tokens:
                    termination = "max_sequence_length"
                    break
                if len(sequence) >= 2_048 and step + 1 < max_new_tokens:
                    termination = "max_sequence_length"
                    break
    except WspError as exc:
        termination = "error"
        errors.append(str(exc))
    except Exception as exc:  # pragma: no cover - model-specific failure
        termination = "error"
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        root.train(was_training)

    fingerprint_after = parameter_fingerprint(root)
    state_after = mutable_state_fingerprint(root)
    if fingerprint_before != fingerprint_after:
        raise ExecutionError(
            "model parameters changed during event-triggered execution; no result was recorded"
        )
    if state_before != state_after:
        raise ExecutionError(
            "model buffers or training flags changed during event-triggered execution"
        )
    if installed_hook_count(root) != hook_before:
        raise ExecutionError("intervention hooks were not fully removed after execution")

    final_records: list[EventInterventionRecord] = []
    for record in records:
        if record.state == "scheduled":
            if record.fired_steps:
                record.state = "fired"
            else:
                record.state = "not_reached"
                record.no_subsequent_forward = True
                if not record.notes:
                    record.notes.append(
                        "the scheduled forward was never reached before generation ended"
                    )
        final_records.append(record)

    condition_record = EventCondition(
        condition=condition,  # type: ignore[arg-type]
        generated_token_ids=generated,
        logprobs=logprobs,
        top1_token_ids=top1,
        steps=len(generated),
        termination_reason=termination,  # type: ignore[arg-type]
        events_applied=sum(len(item.fired_steps) for item in final_records),
        errors=errors,
    )
    return condition_record, final_records, step_logits, step_tokens


def _stepwise_diff(
    baseline: Sequence[torch.Tensor],
    other: Sequence[torch.Tensor],
    *,
    baseline_tokens: Sequence[int],
    other_tokens: Sequence[int],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> tuple[list[dict[str, Any]], int | None]:
    rows: list[dict[str, Any]] = []
    first: int | None = None
    length = min(len(baseline), len(other))
    for index in range(length):
        left, right = baseline[index], other[index]
        if left.shape != right.shape:
            continue
        diff = (left - right).abs()
        matches = baseline_tokens[index] == other_tokens[index] if index < len(baseline_tokens) and index < len(other_tokens) else None
        allclose = bool(torch.allclose(left, right, atol=atol, rtol=rtol))
        rows.append(
            {
                "step": index,
                "left_token_id": baseline_tokens[index] if index < len(baseline_tokens) else None,
                "right_token_id": other_tokens[index] if index < len(other_tokens) else None,
                "token_matches": matches,
                "max_abs": float(diff.max().item()) if diff.numel() else None,
                "mean_abs": float(diff.mean().item()) if diff.numel() else None,
                "allclose": allclose,
            }
        )
        if first is None and not allclose:
            first = index
    return rows, first


def _token_divergence(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None


def run_online_event_trigger(
    trigger: EventTrigger,
    spec: ActivationIntervention,
    model: Any,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    budget: Budget | None = None,
    identity: bool = True,
    case_id: str | None = None,
    trace_id: str | None = None,
    trace_artifact_id: str | None = None,
) -> EventInterventionRun:
    """Generate with an online event trigger and report exactly what happened."""

    budget = budget or Budget()
    if trigger.mode != "online":
        raise InputError("run_online_event_trigger requires mode='online'")
    prompt = [int(value) for value in prompt_token_ids]
    if not prompt:
        raise InputError("an online trigger requires a non-empty prompt")
    if max_new_tokens < 1:
        raise InputError("max_new_tokens must be positive")
    if max_new_tokens > budget.max_generation_steps:
        raise ResourceLimitError(
            "the requested generation exceeds the generation-step budget",
            details={"requested": max_new_tokens, "limit": budget.max_generation_steps},
        )
    if trigger.delay_steps > budget.max_trigger_delay:
        raise ResourceLimitError(
            "the trigger delay exceeds the configured delay budget",
            details={"delay": trigger.delay_steps, "limit": budget.max_trigger_delay},
        )
    if trigger.max_triggers > budget.max_trigger_occurrences:
        raise ResourceLimitError(
            "the trigger count exceeds the configured occurrence budget",
            details={"max_triggers": trigger.max_triggers, "limit": budget.max_trigger_occurrences},
        )
    if trigger.max_triggers * trigger.steps_after_event > budget.max_triggered_interventions:
        raise ResourceLimitError(
            "the trigger could schedule more applications than the intervention budget allows",
            details={
                "max_applications": trigger.max_triggers * trigger.steps_after_event,
                "limit": budget.max_triggered_interventions,
            },
        )
    if spec.execution.backend != "torch_hooks":
        raise CompatibilityError(
            "online event triggering supports only the torch-hooks intervention backend",
            details={"backend": spec.execution.backend},
        )
    if spec.target.module in ("", "<root>"):
        raise CompatibilityError("the intervention must name a concrete module")
    if trigger.steps_after_event > budget.max_scheduled_events:
        raise ResourceLimitError(
            "the scheduled intervention window exceeds the schedule budget",
            details={"steps": trigger.steps_after_event, "limit": budget.max_scheduled_events},
        )
    if len(prompt) > budget.max_replay_sequence_tokens:
        raise ResourceLimitError(
            "the prompt exceeds the replay sequence budget",
            details={"tokens": len(prompt), "limit": budget.max_replay_sequence_tokens},
        )

    started = time.perf_counter()
    model_info = describe_model(model)
    baseline, baseline_events, baseline_logits, baseline_tokens = _run_condition(
        condition="baseline",
        trigger=trigger,
        spec=spec,
        operation=None,
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        budget=budget,
        detect=False,
        trace_id=trace_id,
        trace_artifact_id=trace_artifact_id,
    )
    intervention, records, intervention_logits, intervention_tokens = _run_condition(
        condition="intervention",
        trigger=trigger,
        spec=spec,
        operation=None,
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        budget=budget,
        detect=True,
        trace_id=trace_id,
        trace_artifact_id=trace_artifact_id,
    )

    fired_steps = sorted(
        {step for record in records for step in record.fired_steps}
    )
    identity = bool(identity and fired_steps)
    identity_control = IdentityControl(performed=False)
    identity_condition: EventCondition | None = None
    identity_logits: list[torch.Tensor] = []
    identity_tokens: list[int] = []
    if identity:
        identity_condition, _, identity_logits, identity_tokens = _run_condition(
            condition="identity",
            trigger=trigger,
            spec=spec,
            operation="identity",
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            budget=budget,
            detect=False,
            fixed_steps=fired_steps,
            trace_id=trace_id,
            trace_artifact_id=trace_artifact_id,
        )
        exact_stream = identity_condition.generated_token_ids == baseline.generated_token_ids
        compared = min(len(identity_logits), len(baseline_logits))
        if compared == 0:
            agreement = "failed"
            passed = False
            detail = "no baseline forward could be compared with the identity condition"
            max_abs = mean_abs = None
        else:
            diffs = [
                (identity_logits[index] - baseline_logits[index]).abs()
                for index in range(compared)
            ]
            max_abs = max(float(item.max().item()) for item in diffs)
            mean_abs = sum(float(item.mean().item()) for item in diffs) / compared
            allclose = all(
                torch.allclose(identity_logits[index], baseline_logits[index], atol=1e-6, rtol=1e-5)
                for index in range(compared)
            )
            if exact_stream and allclose:
                agreement = "exact" if max_abs == 0.0 else "within_tolerance"
                passed = True
                detail = (
                    "identity condition reproduced the baseline token stream and logits"
                    if max_abs == 0.0
                    else "identity condition reproduced the baseline stream within tolerance"
                )
            else:
                agreement = "failed"
                passed = False
                detail = (
                    "the identity condition did not reproduce the baseline stream/logits; "
                    "no observed difference can be attributed solely to the requested intervention"
                )
        identity_control = IdentityControl(
            performed=True,
            passed=passed,
            agreement=agreement,  # type: ignore[arg-type]
            max_abs_logits_diff=max_abs,
            mean_abs_logits_diff=mean_abs,
            detail=detail,
        )

    stepwise, first_logit_diff = _stepwise_diff(
        baseline_logits,
        intervention_logits,
        baseline_tokens=baseline_tokens,
        other_tokens=intervention_tokens,
    )
    first_divergence = _token_divergence(
        baseline.generated_token_ids, intervention.generated_token_ids
    )

    checks = [
        ValidationCheck(
            name="causal_ordering",
            status="pass",
            detail=(
                "every fired step is strictly after the generation step at which its event "
                "was observed"
            ),
        )
    ]
    for record in records:
        if record.fired_steps and record.observation_step is not None:
            if min(record.fired_steps) <= record.observation_step:
                checks[0] = ValidationCheck(
                    name="causal_ordering",
                    status="fail",
                    detail="an intervention fired at or before its event observation",
                )
    checks.append(
        ValidationCheck(
            name="identity_control",
            status=("pass" if identity_control.passed else "fail")
            if identity_control.performed
            else "skipped",
            detail=identity_control.detail,
        )
    )
    fired_events = sum(1 for record in records if record.state == "fired")
    not_reached = sum(1 for record in records if record.state == "not_reached")
    rejected = sum(1 for record in records if record.state == "rejected")
    reasons: list[str] = []
    status = "valid"
    if identity_control.performed and not identity_control.passed:
        status = "invalid"
        reasons.append(
            "the identity control failed; the numerical difference cannot be attributed "
            "solely to the requested intervention"
        )
    if fired_events == 0:
        status = "limited" if status == "valid" else status
        reasons.append(
            "no event led to a fired intervention; the run records observations only"
        )
    if not identity_control.performed:
        status = "limited" if status == "valid" else status
        reasons.append("no identity control was performed")
    if rejected:
        status = "invalid"
        reasons.append("at least one scheduled intervention was rejected")
    if intervention.termination_reason == "error":
        status = "invalid"
        reasons.append("generation failed; the run is not a complete experiment")

    warnings: list[str] = []
    if not_reached:
        warnings.append(
            f"{not_reached} observed event(s) had no eligible subsequent forward; "
            "no intervention was attempted for them"
        )
    if first_divergence is not None:
        warnings.append(
            "the baseline and intervention token streams diverge at generated step "
            f"{first_divergence}; later steps are conditioned on different histories"
        )

    record = EventInterventionRun(
        run_id=new_event_run_id(),
        case_id=case_id,
        trigger=trigger,
        trigger_sha256=trigger_sha256(trigger),
        intervention_id=spec.intervention_id,
        intervention_sha256=intervention_sha256(spec),
        model=model_info,
        prompt_length=len(prompt),
        prompt_token_ids_sha256="sha256:" + sha256_bytes(
            json.dumps(prompt, separators=(",", ":")).encode("utf-8")
        ).split(":", 1)[1],
        max_new_tokens=max_new_tokens,
        conditions=[
            baseline,
            *([identity_condition] if identity_condition is not None else []),
            intervention,
        ],
        events=records,
        observed_events=len(records),
        scheduled_events=sum(1 for item in records if item.scheduled_steps),
        fired_events=fired_events,
        not_reached_events=not_reached,
        rejected_events=rejected,
        identity_control=identity_control,
        stepwise=stepwise,
        first_divergence=first_divergence,
        validity=ValidityReport(status=status, checks=checks, reasons=reasons),  # type: ignore[arg-type]
        warnings=warnings,
        metadata={
            "mode": "online",
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "fired_steps": fired_steps,
            "first_logit_difference_step": first_logit_diff,
            "trace_id": trace_id,
        },
    )
    record.checksums["run_sha256"] = event_run_sha256(record)
    return record


def retrospective_trigger(
    *,
    event: BehavioralEvent,
    definition: EventTriggerDefinition | None = None,
    replay_step: int | None = None,
    position_mode: Literal["absolute", "last"] = "absolute",
    name: str | None = None,
    notes: str | None = None,
) -> EventTrigger:
    """Build a retrospective trigger that names an existing event."""

    if event.observation_mode == "online":
        # An event recorded online can still be investigated retrospectively;
        # the trigger, not the event, carries the distinction.
        pass
    step = event.generation_step if replay_step is None else replay_step
    if step is None:
        raise InputError(
            "the event has no exact generation step; supply an explicit replay_step "
            "or resolve its alignment first"
        )
    if event.alignment.status != "exact" and replay_step is None:
        raise InputError(
            "an ambiguous event cannot supply its own replay step; select one explicitly",
            details={"alignment": event.alignment.model_dump(mode="json")},
        )
    return EventTrigger(
        trigger_id=new_trigger_id(),
        name=name,
        mode="retrospective",
        definition=definition or event.trigger,
        event_id=event.event_id,
        replay_step=step,
        delay_steps=0,
        steps_after_event=1,
        max_triggers=1,
        fire_on="once",
        position_mode=position_mode,
        notes=notes,
    )


def validate_trigger_for_run(
    trigger: EventTrigger,
    *,
    budget: Budget | None = None,
    available_event: BehavioralEvent | None = None,
) -> None:
    """Validate trigger/event/mode coherence before spending any compute."""

    budget = budget or Budget()
    if trigger.mode == "retrospective":
        if available_event is None:
            raise InputError(
                "a retrospective trigger requires the recorded event it references"
            )
        if trigger.event_id is not None and trigger.event_id != available_event.event_id:
            raise InputError(
                "the trigger references a different event than the one supplied",
                details={
                    "trigger_event_id": trigger.event_id,
                    "event_id": available_event.event_id,
                },
            )
        if trigger.replay_step is None:
            raise InputError("a retrospective trigger requires a replay step")
        if trigger.replay_step > (available_event.full_length or 0):
            raise InputError(
                "the replay step lies outside the recorded sequence",
                details={
                    "replay_step": trigger.replay_step,
                    "full_length": available_event.full_length,
                },
            )
    if trigger.delay_steps > budget.max_trigger_delay:
        raise ResourceLimitError(
            "the trigger delay exceeds the configured delay budget",
            details={"delay": trigger.delay_steps, "limit": budget.max_trigger_delay},
        )
    if trigger.max_triggers > budget.max_trigger_occurrences:
        raise ResourceLimitError(
            "the trigger count exceeds the configured occurrence budget",
            details={"trigger": trigger.max_triggers, "limit": budget.max_trigger_occurrences},
        )
    if trigger.max_triggers * trigger.steps_after_event > budget.max_triggered_interventions:
        raise ResourceLimitError(
            "the trigger could schedule more applications than the intervention budget allows",
            details={
                "max_applications": trigger.max_triggers * trigger.steps_after_event,
                "limit": budget.max_triggered_interventions,
            },
        )


__all__ = [
    "SCHEMA_TRIGGER",
    "SCHEMA_EVENT_INTERVENTION",
    "EVENT_INTERVENTION_LIMITATIONS",
    "EventApplicationRecord",
    "EventCondition",
    "EventInterventionRecord",
    "EventInterventionRun",
    "EventTrigger",
    "TriggerMode",
    "TriggerState",
    "event_run_sha256",
    "intervention_sha256",
    "retrospective_trigger",
    "run_online_event_trigger",
    "trigger_sha256",
    "validate_trigger_for_run",
]
