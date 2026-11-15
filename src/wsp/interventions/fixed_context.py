"""Fixed-context counterfactual replay.

The experiment holds the token context fixed and changes only a selected
activation.  Baseline, identity, and requested conditions all consume exactly
the same recorded token identifiers, in the same order, with the same
attention mask and position convention.  Measurement is a single teacher-forced
forward over that sequence, so no condition can benefit from a different
generated history.

A successful comparison establishes a numerical effect under the tested
conditions.  It does not establish semantic meaning, a complete mechanism, a
repair, or the existence of an equivalent permanent weight edit.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Sequence

import torch
from pydantic import Field, model_validator

from wsp.behavior.events import BehavioralEvent
from wsp.behavior.models import GenerationTrace, ModelRef, TokenizerRef
from wsp.core.budget import Budget
from wsp.core.errors import (
    CompatibilityError,
    InputError,
    ResourceLimitError,
)
from wsp.core.hashing import sha256_bytes
from wsp.core.time import utc_now_iso
from wsp.interventions.engine import (
    _spec_variant,
    describe_model,
    retarget_spec_execution,
    run_paired_intervention,
    run_pass,
)
from wsp.interventions.models import (
    ActivationIntervention,
    ContinuationDiff,
    DownstreamActivation,
    IdentityControl,
    TokenLogprobDelta,
    ValidationCheck,
    ValidityReport,
)
from wsp.investigation.models import StrictModel
from wsp.replay.models import (
    ReplayCheck,
    ReplayExecutionConfig,
    ReplayTokenSequence,
)
from wsp.replay.plan import prepare_plan, sequence_checksum

SCHEMA_FIXED_REPLAY = "wsp.fixed-context-replay/1.0"
SCHEMA_FIXED_COMPARISON = "wsp.fixed-context-comparison/1.0"

FIXED_CONTEXT_LIMITATIONS = [
    "Only the selected activation differs between conditions; the token context is matched.",
    "A measured numerical effect does not establish the semantic meaning of the activation.",
    "A successful intervention is not a repair, and it does not imply that an equivalent permanent weight edit exists.",
    "Optional continuations diverge once tokens differ; later continuation states are conditioned on different generated histories.",
    "The comparison is limited to the model, dtype, backend, and sequence recorded in this artifact.",
]


def _new(prefix: str) -> str:
    return f"{prefix}_{__import__('secrets').token_hex(8)}"


class FixedContextCheck(StrictModel):
    name: str
    status: Literal["pass", "fail", "warning", "skipped"]
    detail: str | None = None


class FixedContextReplay(StrictModel):
    """A validated reconstruction of one recorded token prefix."""

    schema_version: Literal[SCHEMA_FIXED_REPLAY] = SCHEMA_FIXED_REPLAY
    replay_id: str = Field(pattern=r"^fcr_[0-9a-f]{16}$")
    created_at: str = Field(default_factory=utc_now_iso)
    trace_id: str
    trace_artifact_id: str | None = None
    event_ids: list[str] = Field(default_factory=list)
    replay_step: int | None = Field(default=None, ge=0)
    consumed_position: int | None = Field(default=None, ge=0)
    prompt_length: int = Field(ge=0)
    sequence: ReplayTokenSequence
    model: ModelRef = Field(default_factory=ModelRef)
    tokenizer: TokenizerRef | None = None
    execution: ReplayExecutionConfig = Field(default_factory=ReplayExecutionConfig)
    replay_plan_id: str | None = None
    replay_checks: list[ReplayCheck] = Field(default_factory=list)
    checks: list[FixedContextCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: list(FIXED_CONTEXT_LIMITATIONS))
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> "FixedContextReplay":
        if self.prompt_length != self.sequence.prompt_length:
            raise ValueError("prompt_length must match the replay sequence")
        if self.consumed_position is not None and self.consumed_position >= self.sequence.full_length:
            raise ValueError("the consumed position lies outside the replay sequence")
        return self


class FixedContextMeasurement(StrictModel):
    position: int = Field(ge=0)
    token_id: int | None = Field(default=None, ge=0)
    baseline_logprob: float | None = None
    identity_logprob: float | None = None
    intervention_logprob: float | None = None
    identity_delta: float | None = None
    intervention_delta: float | None = None
    baseline_top1_token_id: int | None = Field(default=None, ge=0)
    intervention_top1_token_id: int | None = Field(default=None, ge=0)


class FixedContextContinuation(StrictModel):
    config: dict[str, Any] = Field(default_factory=dict)
    baseline_token_ids: list[int] = Field(default_factory=list)
    intervention_token_ids: list[int] = Field(default_factory=list)
    diff: ContinuationDiff
    intervention_applied_steps: list[int] = Field(default_factory=list)
    note: str = (
        "The intervention was applied only on the first continuation forward; later "
        "tokens are generated without it and are conditioned on diverging histories."
    )


class FixedContextComparison(StrictModel):
    schema_version: Literal[SCHEMA_FIXED_COMPARISON] = SCHEMA_FIXED_COMPARISON
    comparison_id: str = Field(pattern=r"^fcp_[0-9a-f]{16}$")
    created_at: str = Field(default_factory=utc_now_iso)
    case_id: str | None = None
    trace_id: str
    trace_artifact_id: str | None = None
    replay_id: str
    replay_step: int | None = Field(default=None, ge=0)
    consumed_position: int | None = Field(default=None, ge=0)
    event_ids: list[str] = Field(default_factory=list)
    prompt_length: int = Field(ge=0)
    full_length: int = Field(ge=0)
    token_ids_sha256: str
    model: dict[str, Any] = Field(default_factory=dict)
    execution: ReplayExecutionConfig = Field(default_factory=ReplayExecutionConfig)
    intervention_id: str
    intervention_sha256: str
    intervention_artifact_id: str | None = None
    counterfactual_comparison_id: str | None = None
    conditions: list[dict[str, Any]] = Field(default_factory=list)
    measurements: list[FixedContextMeasurement] = Field(default_factory=list)
    downstream: list[DownstreamActivation] = Field(default_factory=list)
    identity_control: IdentityControl = Field(default_factory=IdentityControl)
    validity: ValidityReport = Field(default_factory=ValidityReport)
    continuation: FixedContextContinuation | None = None
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: list(FIXED_CONTEXT_LIMITATIONS))
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fixed_replay_sha256(replay: FixedContextReplay) -> str:
    payload = replay.model_dump(mode="json")
    payload.get("checksums", {}).pop("replay_sha256", None)
    return "sha256:" + sha256_bytes(_canonical(payload).encode("utf-8")).split(":", 1)[1]


def fixed_comparison_sha256(record: FixedContextComparison) -> str:
    payload = record.model_dump(mode="json")
    payload.get("checksums", {}).pop("comparison_sha256", None)
    return "sha256:" + sha256_bytes(_canonical(payload).encode("utf-8")).split(":", 1)[1]


def prepare_fixed_context(
    trace: GenerationTrace,
    *,
    trace_artifact_id: str | None = None,
    trace_artifact_sha256: str | None = None,
    event: BehavioralEvent | None = None,
    events: Sequence[BehavioralEvent] = (),
    replay_step: int | None = None,
    target_position: int | None = None,
    execution: ReplayExecutionConfig | None = None,
    budget: Budget | None = None,
    notes: list[str] | None = None,
) -> FixedContextReplay:
    """Reconstruct the exact token prefix for a fixed-context experiment."""

    budget = budget or Budget()
    all_events = list(events) or ([event] if event is not None else [])
    for item in all_events:
        if item.trace_id is not None and item.trace_id != trace.trace_id:
            raise InputError(
                "an event references a different trace than the one being replayed",
                details={"event_id": item.event_id, "event_trace": item.trace_id, "trace": trace.trace_id},
            )
        if item.alignment.status != "exact":
            raise InputError(
                "a fixed-context replay requires an exactly aligned event; ambiguous "
                "positions cannot be used for activation targeting",
                details={"event_id": item.event_id, "alignment": item.alignment.status},
            )
    plan = prepare_plan(
        trace,
        trace_artifact_id=trace_artifact_id,
        trace_artifact_sha256=trace_artifact_sha256,
        execution=execution,
        budget=budget,
    )
    if plan.failed_checks():
        raise CompatibilityError(
            "the trace could not be reconstructed as an exact token prefix",
            details={
                "failed_checks": [
                    {"name": check.name, "detail": check.detail}
                    for check in plan.failed_checks()[:8]
                ]
            },
        )
    if plan.sequence.full_length > budget.max_replay_sequence_tokens:
        raise ResourceLimitError(
            "the replay sequence exceeds the configured limit",
            details={
                "tokens": plan.sequence.full_length,
                "limit": budget.max_replay_sequence_tokens,
            },
        )
    if replay_step is not None:
        step = int(replay_step)
        if step >= plan.sequence.full_length:
            raise InputError(
                "the replay step lies outside the reconstructed sequence",
                details={"replay_step": step, "full_length": plan.sequence.full_length},
            )
        consumed = plan.sequence.prompt_length + step - 1
        if consumed < 0 or consumed >= plan.sequence.full_length:
            raise InputError(
                "the replay step does not correspond to an available consumed token",
                details={"replay_step": step, "consumed_position": consumed},
            )
    elif all_events:
        consumed = all_events[0].alignment.consumed_position
        if consumed is None:
            raise InputError("the selected event has no exact consumed position")
        step = consumed - plan.sequence.prompt_length + 1
        if step < 0:
            raise InputError(
                "the selected event lies in the prompt; it is not an event produced by generation"
            )
    elif target_position is not None:
        consumed = int(target_position)
        if consumed >= plan.sequence.full_length:
            raise InputError(
                "the target position lies outside the reconstructed sequence",
                details={"target_position": consumed, "full_length": plan.sequence.full_length},
            )
        step = consumed - plan.sequence.prompt_length + 1
        if step < 0:
            raise InputError("the target position lies in the prompt")
    else:
        consumed = None
        step = None

    warnings = list(plan.warnings)
    checks = [
        FixedContextCheck(
            name="exact_token_prefix",
            status="pass"
            if plan.sequence.source == "recorded-token-ids"
            else "warning",
            detail=(
                "the trace recorded exact input token identifiers"
                if plan.sequence.source == "recorded-token-ids"
                else "the input prefix was re-derived from stored text and is a reconstruction"
            ),
        ),
        FixedContextCheck(
            name="generated_token_identifiers",
            status="pass",
            detail="the trace recorded exact generated token identifiers",
        ),
        FixedContextCheck(
            name="sequence_length",
            status="pass",
            detail=f"{plan.sequence.full_length} tokens, prompt length {plan.sequence.prompt_length}",
        ),
    ]
    if all_events:
        checks.append(
            FixedContextCheck(
                name="event_alignment",
                status="pass",
                detail="every referenced event has an exact token alignment",
            )
        )
    record = FixedContextReplay(
        replay_id=_new("fcr"),
        trace_id=trace.trace_id,
        trace_artifact_id=trace_artifact_id,
        event_ids=[item.event_id for item in all_events],
        replay_step=step,
        consumed_position=consumed,
        prompt_length=plan.sequence.prompt_length,
        sequence=plan.sequence,
        model=plan.model,
        tokenizer=plan.tokenizer,
        execution=plan.execution,
        replay_plan_id=plan.plan_id,
        replay_checks=list(plan.checks),
        checks=checks,
        warnings=warnings,
        provenance={
            "created_by": "wsp.fixed-context",
            "trace_artifact_sha256": trace_artifact_sha256,
            "notes": list(notes or []),
        },
        metadata={
            "trace_plan_limitations": list(plan.limitations),
            "event_ids": [item.event_id for item in all_events],
        },
    )
    record.checksums["replay_sha256"] = fixed_replay_sha256(record)
    return record


def _check_model(trace_model: ModelRef, model: Any) -> tuple[dict[str, Any], list[str], list[FixedContextCheck]]:
    info = describe_model(model)
    warnings: list[str] = []
    checks: list[FixedContextCheck] = []
    if trace_model.structure_fingerprint and info.get("structure_fingerprint"):
        if trace_model.structure_fingerprint != info["structure_fingerprint"]:
            raise CompatibilityError(
                "the loaded model structure does not match the trace's recorded model",
                details={
                    "trace_structure_fingerprint": trace_model.structure_fingerprint,
                    "loaded_structure_fingerprint": info["structure_fingerprint"],
                },
            )
        checks.append(
            FixedContextCheck(
                name="model_structure",
                status="pass",
                detail="the loaded model structure matches the trace",
            )
        )
    else:
        warnings.append(
            "the trace did not record a structure fingerprint; model compatibility is "
            "established only by the caller-supplied model"
        )
        checks.append(
            FixedContextCheck(
                name="model_structure",
                status="warning",
                detail="no recorded structure fingerprint to compare",
            )
        )
    if trace_model.parameter_count is not None and info.get("parameter_count") != trace_model.parameter_count:
        raise CompatibilityError(
            "the loaded model parameter count differs from the trace",
            details={
                "trace_parameter_count": trace_model.parameter_count,
                "loaded_parameter_count": info.get("parameter_count"),
            },
        )
    return info, warnings, checks


def _teacher_forced_spec(
    spec: ActivationIntervention,
    *,
    replay: FixedContextReplay,
    consumed_position: int | None,
) -> ActivationIntervention:
    if spec.execution.trace_id is not None and spec.execution.trace_id != replay.trace_id:
        raise CompatibilityError(
            "the intervention was recorded against a different trace",
            details={
                "intervention_trace_id": spec.execution.trace_id,
                "replay_trace_id": replay.trace_id,
            },
        )
    if replay.consumed_position is not None:
        positions = list(spec.target.selector.positions)
        if replay.consumed_position not in positions:
            raise CompatibilityError(
                "the intervention does not target the selected replay position",
                details={
                    "selected_position": replay.consumed_position,
                    "intervention_positions": positions,
                    "hint": (
                        "a fixed-context experiment changes the activation at the selected "
                        "forward; targeting a different position would not test that selection"
                    ),
                },
            )
    payload = spec.model_dump(mode="python")
    payload["execution"]["execution_mode"] = "teacher-forced"
    payload["execution"]["sequence_length"] = replay.sequence.full_length
    payload["execution"]["trace_id"] = replay.trace_id
    payload["execution"]["replay_id"] = replay.replay_id
    payload["execution"]["replay_artifact_id"] = None
    effective = ActivationIntervention.model_validate(payload)
    return retarget_spec_execution(
        effective,
        execution_mode="teacher-forced",
        sequence_length=replay.sequence.full_length,
    )


def _continuation(
    *,
    model: Any,
    prefix: Sequence[int],
    spec: ActivationIntervention,
    other_spec: ActivationIntervention | None,
    config: dict[str, Any],
    budget: Budget,
) -> FixedContextContinuation:
    max_new = int(config.get("max_new_tokens", 8))
    eos = config.get("eos_token_id")
    if max_new < 1 or max_new > budget.max_continuation_tokens:
        raise ResourceLimitError(
            "the continuation length exceeds the configured budget",
            details={"max_new_tokens": max_new, "limit": budget.max_continuation_tokens},
        )
    if len(prefix) + max_new > budget.max_continuation_sequence_tokens:
        raise ResourceLimitError(
            "the continuation would exceed the sequence budget",
            details={
                "sequence": len(prefix) + max_new,
                "limit": budget.max_continuation_sequence_tokens,
            },
        )

    def generate(effective: ActivationIntervention | None, apply_first: bool) -> tuple[list[int], list[int], str]:
        sequence = [int(value) for value in prefix]
        generated: list[int] = []
        applied: list[int] = []
        termination = "max_new_tokens"
        for step in range(max_new):
            if effective is not None and apply_first and step == 0:
                current = retarget_spec_execution(
                    effective,
                    execution_mode="uncached-continuation",
                    sequence_length=len(sequence),
                )
                current = current.model_copy(update={"apply_on_calls": [0]})
                logits = run_pass(
                    current,
                    model,
                    sequence,
                    condition="continuation-intervention",
                    apply=True,
                    budget=budget,
                ).logits
                applied.append(step)
            else:
                logits = model.forward_token_ids(sequence)
            token = int(torch.argmax(logits[0, -1]).item())
            generated.append(token)
            sequence.append(token)
            if eos is not None and token == int(eos):
                termination = "eos"
                break
        return generated, applied, termination

    baseline_tokens, _, baseline_termination = generate(None, False)
    intervention_tokens, applied, intervention_termination = generate(spec, True)
    common = 0
    for left, right in zip(baseline_tokens, intervention_tokens):
        if left != right:
            break
        common += 1
    first = None if common == len(baseline_tokens) == len(intervention_tokens) else common
    agreement = (
        1.0
        if not baseline_tokens and not intervention_tokens
        else common / max(len(baseline_tokens), len(intervention_tokens))
    )
    diff = ContinuationDiff(
        longest_common_prefix=common,
        first_divergence=first,
        length_delta=len(intervention_tokens) - len(baseline_tokens),
        termination_changed=baseline_termination != intervention_termination,
        token_agreement_rate=agreement,
    )
    return FixedContextContinuation(
        config={
            "strategy": "greedy",
            "max_new_tokens": max_new,
            "eos_token_id": eos,
            "baseline_termination": baseline_termination,
            "intervention_termination": intervention_termination,
        },
        baseline_token_ids=baseline_tokens,
        intervention_token_ids=intervention_tokens,
        diff=diff,
        intervention_applied_steps=applied,
    )


def run_fixed_context_comparison(
    replay: FixedContextReplay,
    spec: ActivationIntervention,
    model: Any,
    *,
    budget: Budget | None = None,
    logit_positions: Sequence[int] | None = None,
    measure_modules: Sequence[str] = (),
    measure_positions: Sequence[int] | None = None,
    use_donor_store: Any | None = None,
    case_id: str | None = None,
    intervention_artifact_id: str | None = None,
    continuation: dict[str, Any] | None = None,
    identity: bool = True,
) -> FixedContextComparison:
    """Run baseline, identity, and requested intervention on one fixed prefix."""

    budget = budget or Budget()
    tokens = [int(value) for value in replay.sequence.full_token_ids]
    if not tokens:
        raise InputError("the fixed-context replay has no token identifiers")
    if len(tokens) != replay.sequence.full_length:
        raise InputError("the replay sequence length is inconsistent")
    if replay.sequence.sequence_sha256:
        expected = sequence_checksum(
            list(replay.sequence.input_token_ids),
            list(replay.sequence.generated_token_ids),
        )
        if expected != replay.sequence.sequence_sha256:
            raise CompatibilityError(
                "the replay token sequence does not match its recorded digest"
            )
    info, warnings, checks = _check_model(replay.model, model)
    if replay.consumed_position is None:
        raise InputError(
            "the replay has no selected forward; select an event, replay_step, or "
            "target_position before running a fixed-context comparison"
        )
    effective = _teacher_forced_spec(
        spec, replay=replay, consumed_position=replay.consumed_position
    )
    if logit_positions is not None:
        positions = [int(value) for value in logit_positions]
    elif replay.consumed_position is not None:
        positions = [replay.consumed_position]
        if replay.consumed_position + 1 < len(tokens):
            positions.append(replay.consumed_position + 1)
    else:
        positions = [len(tokens) - 1]
    if len(positions) > budget.max_fixed_context_positions:
        raise ResourceLimitError(
            "too many logit positions were requested",
            details={"requested": len(positions), "limit": budget.max_fixed_context_positions},
        )
    if any(value < 0 or value >= len(tokens) for value in positions):
        raise InputError("a logit position lies outside the fixed token sequence")
    if any(value < (replay.consumed_position or 0) for value in positions):
        warnings.append(
            "a measurement position precedes the intervention position; a causal "
            "language model cannot propagate the change backwards"
        )
    measurement_positions = list(measure_positions or positions)
    outcome = run_paired_intervention(
        effective,
        model,
        tokens,
        store=use_donor_store,
        budget=budget,
        measure_modules=list(measure_modules),
        measure_positions=measurement_positions,
        logit_positions=positions,
        case_id=case_id,
        intervention_artifact_id=intervention_artifact_id,
        identity=identity,
    )
    comparison = outcome.comparison
    result = outcome.results["intervention"]
    measurements = [
        FixedContextMeasurement(
            position=item.position,
            token_id=item.token_id,
            baseline_logprob=item.baseline,
            identity_logprob=item.identity,
            intervention_logprob=item.intervention,
            identity_delta=item.identity_delta,
            intervention_delta=item.intervention_delta,
            baseline_top1_token_id=_top1(comparison, "baseline", item.position),
            intervention_top1_token_id=_top1(comparison, "intervention", item.position),
        )
        for item in result.token_logprobs
    ]
    checks = list(checks)
    checks.append(
        FixedContextCheck(
            name="fixed_token_context",
            status="pass",
            detail=(
                "baseline, identity, and intervention consumed identical token identifiers "
                f"({len(tokens)} tokens, sha256 {replay.sequence.sequence_sha256})"
            ),
        )
    )
    checks.append(
        FixedContextCheck(
            name="target_position",
            status="pass",
            detail=f"the intervention targets position {replay.consumed_position}",
        )
    )
    validity = comparison.validity.model_copy(deep=True)
    for check in checks:
        if check.status == "fail":
            validity.status = "invalid"
            validity.reasons.append(check.detail or check.name)
    if comparison.identity_control.performed and not comparison.identity_control.passed:
        validity.status = "invalid"
        validity.reasons.append(
            "the identity control failed; the comparison cannot attribute the difference "
            "to the requested intervention"
        )
    if not comparison.identity_control.performed:
        if validity.status == "valid":
            validity.status = "limited"
        validity.reasons.append("no identity control was performed")

    continuation_record = None
    if continuation:
        other = None
        if continuation.get("control") == "identity":
            other = _spec_variant(effective, "identity")
        continuation_record = _continuation(
            model=model,
            prefix=tokens,
            spec=effective,
            other_spec=other,
            config=continuation,
            budget=budget,
        )

    record = FixedContextComparison(
        comparison_id=_new("fcp"),
        case_id=case_id,
        trace_id=replay.trace_id,
        trace_artifact_id=replay.trace_artifact_id,
        replay_id=replay.replay_id,
        replay_step=replay.replay_step,
        consumed_position=replay.consumed_position,
        event_ids=list(replay.event_ids),
        prompt_length=replay.prompt_length,
        full_length=replay.sequence.full_length,
        token_ids_sha256=replay.sequence.sequence_sha256
        or sequence_checksum(
            list(replay.sequence.input_token_ids),
            list(replay.sequence.generated_token_ids),
        ),
        model=info,
        execution=replay.execution,
        intervention_id=spec.intervention_id,
        intervention_sha256=comparison.intervention_sha256,
        intervention_artifact_id=intervention_artifact_id,
        counterfactual_comparison_id=comparison.comparison_id,
        conditions=[item.model_dump(mode="json") for item in comparison.conditions],
        measurements=measurements,
        downstream=result.downstream,
        identity_control=comparison.identity_control,
        validity=validity,
        continuation=continuation_record,
        warnings=warnings + list(comparison.warnings),
        metadata={
            "replay_plan_id": replay.replay_plan_id,
            "logit_positions": positions,
            "measure_modules": list(measure_modules),
            "measure_positions": measurement_positions,
            "condition_results": list(outcome.results),
            "single_forward_per_condition": True,
            "checks": [item.model_dump(mode="json") for item in checks],
        },
    )
    record.checksums["comparison_sha256"] = fixed_comparison_sha256(record)
    return record


def _top1(comparison: Any, condition: str, position: int) -> int | None:
    for item in comparison.conditions:
        if item.condition == condition:
            value = item.top1_token_ids.get(str(position))
            return int(value) if value is not None else None
    return None


__all__ = [
    "SCHEMA_FIXED_REPLAY",
    "SCHEMA_FIXED_COMPARISON",
    "FIXED_CONTEXT_LIMITATIONS",
    "FixedContextCheck",
    "FixedContextComparison",
    "FixedContextContinuation",
    "FixedContextMeasurement",
    "FixedContextReplay",
    "fixed_comparison_sha256",
    "fixed_replay_sha256",
    "prepare_fixed_context",
    "run_fixed_context_comparison",
]
