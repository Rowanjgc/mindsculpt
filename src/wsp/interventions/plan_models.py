"""Versioned records for temporal, compositional, and staged experiments.

These records are the instrument readings of V1.4.  They state which
interventions fired, at which generation step, on which token index, with which
before/after summary, and what changed downstream.  They deliberately do not
state a mechanism, a circuit, or an improvement.

Two invariants are enforced by the schemas themselves:

* an event log can distinguish *missing* events (a step that ran and did not
  fire) from *not-reached* events (a step that never ran);
* a staged record carries the verification that licenses its numbers, and a
  failed verification cannot be stored as ``valid``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from wsp.core.hashing import sha256_bytes
from wsp.core.json import canonical_bytes
from wsp.core.models import TensorDiffStats
from wsp.investigation.models import StrictModel
from wsp.interventions.models import (
    ContinuationDiff,
    IdentityControl,
    PositionSemantics,
    SliceStats,
    TensorRole,
    ValidationCheck,
    ValidityReport,
)

SCHEMA_EVENT_LOG = "wsp.intervention-event-log/1.0"
SCHEMA_TEMPORAL_RUN = "wsp.temporal-intervention/1.0"
SCHEMA_COMPOSITION = "wsp.compositional-comparison/1.0"
SCHEMA_STAGED_VERIFICATION = "wsp.staged-verification/1.0"
SCHEMA_STAGED_CONTINUATION = "wsp.staged-continuation/1.0"

EventResult = Literal["applied", "identity", "rejected", "skipped"]
STAGED_VERIFICATION_ATOL = 1e-5
STAGED_VERIFICATION_RTOL = 1e-5

TEMPORAL_LIMITATIONS = [
    "an intervention event is a record that instrumentation fired as requested, not a "
    "claim that the targeted component causes the observed behaviour",
    "uncached full-prefix decoding recomputes earlier positions, so a one-time "
    "intervention is not persistent model state",
    "a difference in generated tokens is a controlled numerical observation and is not "
    "automatically an improvement or a failure",
    "results are specific to the recorded model, dtype, device, backend, and token sequence",
]

COMPOSITION_LIMITATIONS = [
    "the interaction contrast measures non-additivity on one chosen scale; it does not "
    "establish that two components form a functional circuit",
    "a near-zero interaction is not evidence that two components are independent",
    "discrete outcomes such as generated text are reported as observations, not as a "
    "continuous quantity",
    "conditions executed in different execution modes must not be combined silently",
]

STAGED_LIMITATIONS = [
    "staged execution is only meaningful for architectures whose generation state was "
    "verified against the uncached reference",
    "matching generated text is not numerical equivalence; this record includes stepwise "
    "logit agreement for that reason",
    "cached decoding changes floating-point reduction order, so agreement is tolerance "
    "bounded rather than bitwise in general",
    "a branch must be rebuilt or cloned through a verified isolation path; a shallow copy "
    "of a mutable cache is not evidence of isolation",
]


class ExpectedEvent(StrictModel):
    """One firing the plan asked for and the execution was supposed to reach."""

    plan_intervention_id: str
    generation_step: int = Field(ge=0)
    on_step_call_index: int = Field(ge=0)
    module: str


class InterventionEventRecord(StrictModel):
    """One firing, in the order the model actually executed it."""

    event_index: int = Field(ge=0)
    plan_id: str
    plan_intervention_id: str
    intervention_id: str
    label: str | None = None
    generation_step: int = Field(ge=0)
    forward_invocation: int = Field(ge=0)
    module: str
    tensor_role: TensorRole
    position_mode: Literal["absolute", "last"]
    position_semantics: PositionSemantics
    rows: list[int] = Field(default_factory=list)
    token_indices: list[int] = Field(default_factory=list)
    feature_start: int | None = None
    feature_end: int | None = None
    feature_indices: list[int] = Field(default_factory=list)
    operation: str
    tensor_shape: list[int] = Field(default_factory=list)
    dtype: str | None = None
    elements_modified: int = Field(default=0, ge=0)
    dtype_cast: str | None = None
    replacement_sha256: str | None = None
    before: SliceStats | None = None
    after: SliceStats | None = None
    max_abs: float | None = None
    mean_abs: float | None = None
    execution_result: EventResult = "applied"
    rejection_reason: str | None = None
    provenance: dict[str, str] = Field(default_factory=dict)


class InterventionEventLog(StrictModel):
    """Every firing, every expected firing, and the difference between them."""

    schema_version: Literal[SCHEMA_EVENT_LOG] = SCHEMA_EVENT_LOG
    log_id: str = Field(pattern=r"^el_[0-9a-f]{16}$")
    created_at: str
    plan_id: str
    plan_sha256: str
    plan_kind: str
    execution_mode: str
    backend: str
    condition: str = "plan"
    prompt_length: int = Field(ge=0)
    steps_executed: int = Field(ge=0)
    events: list[InterventionEventRecord] = Field(default_factory=list)
    expected_events: list[ExpectedEvent] = Field(default_factory=list)
    missing_events: list[ExpectedEvent] = Field(default_factory=list)
    not_reached_events: list[ExpectedEvent] = Field(default_factory=list)
    rejected_events: list[InterventionEventRecord] = Field(default_factory=list)
    event_limit: int = Field(ge=0)
    truncated: bool = False
    total_events_observed: int = Field(default=0, ge=0)
    limitations: list[str] = Field(default_factory=lambda: list(TEMPORAL_LIMITATIONS))
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _accounting(self) -> "InterventionEventLog":
        if self.missing_events and not self.expected_events:
            raise ValueError("missing events require a non-empty expected-event list")
        if len(self.events) > self.event_limit:
            raise ValueError("the event log exceeds its recorded event limit")
        return self

    @property
    def all_fired(self) -> bool:
        return not self.missing_events and not self.rejected_events


class StepwiseLogitDiff(StrictModel):
    step: int = Field(ge=0)
    token_id: int | None = None
    token_matches: bool | None = None
    max_abs: float | None = None
    mean_abs: float | None = None
    allclose: bool | None = None


class ConditionOutcome(StrictModel):
    """One executed condition (baseline, identity, plan, A, B, AB, control)."""

    condition: str
    plan_id: str | None = None
    plan_kind: str | None = None
    applied: bool = False
    generated_token_ids: list[int] = Field(default_factory=list)
    generated_text: str | None = None
    logprobs: list[float] = Field(default_factory=list)
    top1_token_ids: list[int] = Field(default_factory=list)
    termination_reason: Literal["eos", "max_new_tokens", "max_sequence_length", "error"] = (
        "max_new_tokens"
    )
    steps: int = Field(default=0, ge=0)
    events_applied: int = Field(default=0, ge=0)
    events_identity: int = Field(default=0, ge=0)
    events_missing: int = Field(default=0, ge=0)
    events_not_reached: int = Field(default=0, ge=0)
    event_log_id: str | None = None
    elapsed_ms: float | None = None
    errors: list[str] = Field(default_factory=list)


class TemporalInterventionRun(StrictModel):
    """A scheduled intervention compared against untouched and identity runs."""

    schema_version: Literal[SCHEMA_TEMPORAL_RUN] = SCHEMA_TEMPORAL_RUN
    run_id: str = Field(pattern=r"^tr2_[0-9a-f]{16}$")
    created_at: str
    case_id: str | None = None
    plan_id: str
    plan_sha256: str
    plan_artifact_id: str | None = None
    identity_plan_id: str | None = None
    identity_plan_artifact_id: str | None = None
    trace_id: str | None = None
    model: dict[str, Any] = Field(default_factory=dict)
    execution_mode: str
    backend: str
    prompt_length: int = Field(ge=1)
    prompt_sha256: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    conditions: list[ConditionOutcome] = Field(default_factory=list)
    identity_control: IdentityControl
    comparison: ContinuationDiff = Field(default_factory=ContinuationDiff)
    stepwise_logits: list[StepwiseLogitDiff] = Field(default_factory=list)
    event_log_artifact_ids: dict[str, str] = Field(default_factory=dict)
    validity: ValidityReport = Field(default_factory=ValidityReport)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: list(TEMPORAL_LIMITATIONS))
    interpretation: str = (
        "This record reports where scheduled interventions fired and what changed under "
        "the recorded conditions. It carries no automatic causal label."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# compositional comparison
# --------------------------------------------------------------------------- #

InteractionMetric = Literal[
    "mean_generated_logprob",
    "final_token_logprob",
    "logit_l2_vs_baseline",
    "generated_length",
]


class PairwiseEffect(StrictModel):
    condition: str
    reference: str
    first_divergence: int | None = Field(default=None, ge=0)
    longest_common_prefix: int = Field(default=0, ge=0)
    token_agreement_rate: float | None = None
    length_delta: int = 0
    termination_changed: bool = False
    logits: TensorDiffStats | None = None
    logit_l2_vs_reference: float | None = None
    stepwise: list[StepwiseLogitDiff] = Field(default_factory=list)


class InteractionContrast(StrictModel):
    """Descriptive non-additivity on one explicitly named scalar scale."""

    metric: InteractionMetric
    y_baseline: float | None = None
    y_a: float | None = None
    y_b: float | None = None
    y_ab: float | None = None
    delta_a: float | None = None
    delta_b: float | None = None
    delta_ab: float | None = None
    interaction: float | None = None
    continuous: bool = True
    scale_note: str = (
        "I_AB = (Y_AB - Y_0) - (Y_A - Y_0) - (Y_B - Y_0) on the recorded scale; a "
        "non-zero value is non-additivity, not proof of a functional circuit"
    )


class CompositionalComparison(StrictModel):
    """Individual versus combined intervention measurements over one prefix."""

    schema_version: Literal[SCHEMA_COMPOSITION] = SCHEMA_COMPOSITION
    comparison_id: str = Field(pattern=r"^cx_[0-9a-f]{16}$")
    created_at: str
    case_id: str | None = None
    trace_id: str | None = None
    plan_a_id: str
    plan_b_id: str
    plan_ab_id: str
    plan_sha256: dict[str, str] = Field(default_factory=dict)
    plan_artifact_ids: dict[str, str] = Field(default_factory=dict)
    identity_plan_id: str | None = None
    identity_plan_artifact_id: str | None = None
    negative_control_plan_id: str | None = None
    model: dict[str, Any] = Field(default_factory=dict)
    execution_mode: str
    backend: str
    prompt_length: int = Field(ge=1)
    prompt_sha256: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    metric: InteractionMetric
    conditions: list[ConditionOutcome] = Field(default_factory=list)
    identity_control: IdentityControl
    interactions: list[InteractionContrast] = Field(default_factory=list)
    pairwise: list[PairwiseEffect] = Field(default_factory=list)
    event_log_artifact_ids: dict[str, str] = Field(default_factory=dict)
    validity: ValidityReport = Field(default_factory=ValidityReport)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: list(COMPOSITION_LIMITATIONS))
    interpretation: str = (
        "This record compares individual and combined intervention effects numerically. "
        "Non-additivity on one scale is a measurement, not a discovered mechanism."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _conditions_present(self) -> "CompositionalComparison":
        names = {item.condition for item in self.conditions}
        missing = {"baseline", "A", "B", "AB"} - names
        if missing:
            raise ValueError(
                "a compositional comparison requires baseline, A, B, and AB conditions; "
                f"missing {sorted(missing)}"
            )
        return self


# --------------------------------------------------------------------------- #
# staged generation
# --------------------------------------------------------------------------- #

StagedAgreement = Literal["exact", "within_tolerance", "failed", "not-performed"]


class StagedStep(StrictModel):
    step: int = Field(ge=0)
    token_uncached: int | None = None
    token_staged: int | None = None
    token_matches: bool | None = None
    max_abs_logits: float | None = None
    mean_abs_logits: float | None = None
    allclose: bool | None = None


class StagedVerification(StrictModel):
    """Evidence that a staged/cached decoder reproduces the uncached reference."""

    schema_version: Literal[SCHEMA_STAGED_VERIFICATION] = SCHEMA_STAGED_VERIFICATION
    verification_id: str = Field(pattern=r"^sv_[0-9a-f]{16}$")
    created_at: str
    model: dict[str, Any] = Field(default_factory=dict)
    execution_backend: str
    execution_mode: Literal["staged-prefill", "cached-decode"]
    architecture_class: Literal["full-attention-transformer", "hybrid-recurrent", "unknown"]
    prompt_length: int = Field(ge=1)
    prompt_sha256: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    steps: int = Field(ge=0)
    agreement: StagedAgreement
    exact_logits: bool | None = None
    atol: float = STAGED_VERIFICATION_ATOL
    rtol: float = STAGED_VERIFICATION_RTOL
    max_abs_logits: float | None = None
    mean_abs_logits: float | None = None
    token_sequence_match: bool | None = None
    termination_match: bool | None = None
    identity_control_agreement: StagedAgreement | None = None
    position_ids_mode: str | None = None
    attention_mask_mode: str | None = None
    state_identity_sha256: str | None = None
    state_bytes: int | None = None
    isolation: Literal["recompute", "clone", "shared"] = "recompute"
    isolation_verified: bool | None = None
    steps_detail: list[StagedStep] = Field(default_factory=list)
    checks: list[ValidationCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: list(STAGED_LIMITATIONS))
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_unverified_claim(self) -> "StagedVerification":
        if self.agreement in ("exact", "within_tolerance") and self.token_sequence_match is False:
            raise ValueError(
                "a staged run whose greedy tokens differ from the uncached reference "
                "cannot be recorded as verified"
            )
        if self.isolation == "shared" and self.agreement != "failed":
            raise ValueError(
                "a shared mutable cache is not an acceptable isolation strategy for a "
                "verified record"
            )
        return self


class StagedPerformance(StrictModel):
    prefill_ms: float | None = None
    decode_ms_total: float | None = None
    decode_ms_per_step: float | None = None
    total_ms: float | None = None
    uncached_reference_ms: float | None = None
    speedup_ratio: float | None = None
    peak_process_rss_bytes: int | None = None
    cache_state_bytes: int | None = None
    note: str = (
        "timing and memory are engineering measurements on this machine; they are not "
        "model capability benchmarks"
    )


class StagedContinuation(StrictModel):
    """A verified staged continuation under a scheduled intervention plan."""

    schema_version: Literal[SCHEMA_STAGED_CONTINUATION] = SCHEMA_STAGED_CONTINUATION
    continuation_id: str = Field(pattern=r"^sc_[0-9a-f]{16}$")
    created_at: str
    case_id: str | None = None
    plan_id: str
    plan_sha256: str
    plan_artifact_id: str | None = None
    identity_plan_id: str | None = None
    trace_id: str | None = None
    model: dict[str, Any] = Field(default_factory=dict)
    backend: str
    #: Staged execution is never reported as plain uncached generation.
    execution_mode: Literal["staged-generation"] = "staged-generation"
    prompt_length: int = Field(ge=1)
    prompt_sha256: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    verification: StagedVerification
    verification_artifact_id: str | None = None
    conditions: list[ConditionOutcome] = Field(default_factory=list)
    identity_control: IdentityControl
    comparison: ContinuationDiff = Field(default_factory=ContinuationDiff)
    stepwise_logits: list[StagedStep] = Field(default_factory=list)
    performance: StagedPerformance = Field(default_factory=StagedPerformance)
    event_log_artifact_ids: dict[str, str] = Field(default_factory=dict)
    validity: ValidityReport = Field(default_factory=ValidityReport)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: list(STAGED_LIMITATIONS))
    interpretation: str = (
        "This record describes a staged continuation only for the architecture and "
        "tolerances recorded in its verification block."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _verified_or_invalid(self) -> "StagedContinuation":
        if self.verification.agreement not in ("exact", "within_tolerance"):
            if self.validity.status == "valid":
                raise ValueError(
                    "staged execution without a passing verification cannot be valid"
                )
        return self


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #


def _hash(record: Any, field: str) -> str:
    payload = record.model_dump(mode="json")
    payload.pop("created_at", None)
    payload.pop("checksums", None)
    return sha256_bytes(canonical_bytes(payload))


def event_log_sha256(record: InterventionEventLog) -> str:
    return _hash(record, "event_log_sha256")


def temporal_run_sha256(record: TemporalInterventionRun) -> str:
    return _hash(record, "temporal_run_sha256")


def composition_sha256(record: CompositionalComparison) -> str:
    return _hash(record, "composition_sha256")


def staged_verification_sha256(record: StagedVerification) -> str:
    return _hash(record, "staged_verification_sha256")


def staged_continuation_sha256(record: StagedContinuation) -> str:
    return _hash(record, "staged_continuation_sha256")


__all__ = [
    "COMPOSITION_LIMITATIONS",
    "CompositionalComparison",
    "Condition",
    "ConditionOutcome",
    "ExpectedEvent",
    "InteractionContrast",
    "InteractionMetric",
    "InterventionEventLog",
    "InterventionEventRecord",
    "PairwiseEffect",
    "SCHEMA_COMPOSITION",
    "SCHEMA_EVENT_LOG",
    "SCHEMA_STAGED_CONTINUATION",
    "SCHEMA_STAGED_VERIFICATION",
    "SCHEMA_TEMPORAL_RUN",
    "STAGED_LIMITATIONS",
    "STAGED_VERIFICATION_ATOL",
    "STAGED_VERIFICATION_RTOL",
    "StagedContinuation",
    "StagedPerformance",
    "StagedStep",
    "StagedVerification",
    "TEMPORAL_LIMITATIONS",
    "TemporalInterventionRun",
    "composition_sha256",
    "event_log_sha256",
    "staged_continuation_sha256",
    "staged_verification_sha256",
    "temporal_run_sha256",
]
