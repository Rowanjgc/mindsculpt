"""Versioned records for controlled activation interventions.

The records in this module are the *instrument readings* of V1.3.  They state
what was replaced, where, under which execution identity, and what changed
downstream.  They deliberately do not contain a causal conclusion: a measured
paired difference is evidence of an effect under the recorded conditions, and
the interpretation lives in human-authored hypotheses elsewhere.

All models forbid unknown fields.  A record that does not validate is not a
record this workbench will execute or export.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, model_validator

from wsp.behavior.models import ModelRef
from wsp.core.hashing import sha256_bytes
from wsp.core.json import canonical_bytes
from wsp.core.models import TensorDiffStats
from wsp.investigation.models import StrictModel

SCHEMA_SPEC = "wsp.activation-intervention/1.0"
SCHEMA_RESULT = "wsp.intervention-result/1.0"
SCHEMA_COUNTERFACTUAL = "wsp.counterfactual-comparison/1.0"
SCHEMA_CONTINUATION = "wsp.continuation-comparison/1.0"

InterventionOperation = Literal["zero", "mean", "patch", "identity"]
TensorRole = Literal["input", "output"]
PositionSemantics = Literal["consuming", "predicting"]
ExecutionMode = Literal["teacher-forced", "uncached-continuation"]
InstrumentationBackend = Literal["torch_hooks", "nnterp"]
Condition = Literal["baseline", "identity", "intervention", "alternative-control"]

MAX_INLINE_VALUES = 100_000

INTERVENTION_LIMITATIONS = [
    "an observed downstream difference is an effect under the recorded conditions, "
    "not proof that the targeted component explains a behaviour",
    "one intervention at one module and position does not identify a circuit",
    "an identity-control disagreement invalidates attribution of the difference to "
    "the requested intervention",
    "results are specific to the recorded model, dtype, device, backend, and token sequence",
]

COUNTERFACTUAL_LIMITATIONS = [
    "paired measurements compare numerical outputs, not model capability or correctness",
    "no automatic causal label is produced; hypotheses must be attached by an investigator",
    "only the measured modules, positions, and tokens are covered by this comparison",
]

CONTINUATION_LIMITATIONS = [
    "greedy uncached continuation is a controlled experiment, not a production decoder",
    "a changed or shorter continuation is not automatically an improvement",
    "cached or stateful generation is not used here, so no claim is made about "
    "generation-time KV-cache or recurrent-state compatibility",
    "behaviour beyond max_new_tokens is not observed",
]


def _finite_vector(values: list[float], what: str) -> list[float]:
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise ValueError(f"{what} contains a non-finite value at index {index}")
    return values


class TensorSelector(StrictModel):
    """Exactly which part of an activation tensor is replaced.

    ``positions`` are *tensor rows* after position semantics have been resolved
    by the caller.  ``axis`` names the row axis explicitly; when omitted the
    engine infers it from the tensor rank (1 for ``[batch, seq, hidden]`` and 0
    for ``[seq, hidden]``) and records the inference in the result.  Feature
    selection is the last axis: either explicit indices or a half-open range.
    """

    axis: int | None = Field(default=None, ge=0)
    positions: list[int] = Field(min_length=1, max_length=4096)
    feature_start: int | None = Field(default=None, ge=0)
    feature_end: int | None = Field(default=None, ge=0)
    feature_indices: list[int] = Field(default_factory=list, max_length=MAX_INLINE_VALUES)
    tensor_index: int | None = Field(
        default=None,
        ge=0,
        description="Index into a tuple/list module output (or positional input argument).",
    )
    tensor_name: str | None = Field(
        default=None,
        description="Mapping key of a dict-shaped module output.",
    )
    expected_shape: list[int] | None = None
    expected_dtype: str | None = None

    @model_validator(mode="after")
    def _well_defined(self) -> "TensorSelector":
        if any(isinstance(item, bool) or item < 0 for item in self.positions):
            raise ValueError("positions must be non-negative integers")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("positions must be unique")
        if self.positions != sorted(self.positions):
            raise ValueError("positions must be sorted ascending")
        if self.tensor_index is not None and self.tensor_name is not None:
            raise ValueError("select a tuple/list element or a mapping key, not both")
        if self.feature_indices:
            if any(item < 0 for item in self.feature_indices):
                raise ValueError("feature_indices must be non-negative")
            if len(set(self.feature_indices)) != len(self.feature_indices):
                raise ValueError("feature_indices must be unique")
            if self.feature_indices != sorted(self.feature_indices):
                raise ValueError("feature_indices must be sorted ascending")
            if self.feature_start is not None or self.feature_end is not None:
                raise ValueError("use feature_indices or a feature range, not both")
        if (self.feature_start is None) != (self.feature_end is None):
            raise ValueError("feature_start and feature_end must be supplied together")
        if (
            self.feature_start is not None
            and self.feature_end is not None
            and self.feature_end <= self.feature_start
        ):
            raise ValueError("feature_end must be greater than feature_start")
        return self

    @property
    def feature_span_selects_all(self) -> bool:
        return not self.feature_indices and self.feature_start is None

    def feature_width_hint(self) -> int | None:
        if self.feature_indices:
            return len(self.feature_indices)
        if self.feature_start is not None and self.feature_end is not None:
            return self.feature_end - self.feature_start
        if self.expected_shape:
            return int(self.expected_shape[-1])
        return None


class InterventionTarget(StrictModel):
    """The module event and tensor region an intervention addresses."""

    module: str = Field(min_length=1)
    capture_point: str | None = None
    tensor_role: TensorRole = "output"
    position_semantics: PositionSemantics
    selector: TensorSelector

    @model_validator(mode="after")
    def _capture_point(self) -> "InterventionTarget":
        if self.capture_point is not None and self.capture_point != self.module:
            raise ValueError(
                "capture_point must name the same module path as the intervention target; "
                "the workbench does not silently substitute one capture point for another"
            )
        return self


class ExecutionIdentity(StrictModel):
    """The execution the intervention is defined against."""

    backend: InstrumentationBackend = "torch_hooks"
    backend_version: str | None = None
    device: str = "cpu"
    dtype: str | None = None
    execution_mode: ExecutionMode = "teacher-forced"
    sequence_length: int = Field(gt=0)
    inference_mode: Literal[True] = True
    attention_mask: Literal["all-ones"] = "all-ones"
    position_ids: Literal["contiguous-implicit"] = "contiguous-implicit"
    trace_id: str | None = None
    replay_id: str | None = None
    replay_artifact_id: str | None = None
    capture_artifact_id: str | None = None
    alignment_artifact_id: str | None = None


class MeanProvenance(StrictModel):
    """Where a mean-ablation vector came from.

    A mean without this record is rejected.  The workbench never computes an
    arbitrary mean from unrelated examples and calls it a control.
    """

    method: Literal["capture-mean", "explicit"]
    source_capture_artifact_id: str | None = None
    source_trace_ids: list[str] = Field(default_factory=list)
    module: str = Field(min_length=1)
    tensor_role: TensorRole
    positions: list[int] = Field(default_factory=list)
    position_semantics: PositionSemantics
    token_alignment: str = Field(
        min_length=1,
        description="How donor positions were aligned to the target positions.",
    )
    model_file_sha256: str | None = None
    model_structure_fingerprint: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _provenance(self) -> "MeanProvenance":
        if self.method == "capture-mean" and self.source_capture_artifact_id is None:
            raise ValueError("a capture-derived mean needs source_capture_artifact_id")
        if self.method == "capture-mean" and not self.source_trace_ids:
            raise ValueError("a capture-derived mean needs at least one source trace id")
        if not self.token_alignment.strip():
            raise ValueError("token_alignment must state how donor positions were chosen")
        return self


class MeanActivation(StrictModel):
    """An explicit replacement vector plus its provenance."""

    values: list[float] | None = Field(default=None, max_length=MAX_INLINE_VALUES)
    artifact_id: str | None = None
    dtype: str | None = None
    provenance: MeanProvenance

    @model_validator(mode="after")
    def _source(self) -> "MeanActivation":
        if (self.values is None) == (self.artifact_id is None):
            raise ValueError("provide exactly one of mean values or a mean artifact id")
        if self.values is not None:
            _finite_vector(self.values, "mean activation")
        return self


class DonorActivation(StrictModel):
    """A compatible activation captured from a controlled donor execution."""

    trace_id: str | None = None
    capture_artifact_id: str | None = None
    execution_id: str | None = None
    module: str = Field(min_length=1)
    tensor_role: TensorRole = "output"
    positions: list[int] = Field(default_factory=list)
    position_semantics: PositionSemantics = "consuming"
    condition: Literal["healthy-control", "alternative-input", "other"] = "other"
    values: list[float] | None = Field(default=None, max_length=MAX_INLINE_VALUES)
    values_artifact_id: str | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    allow_position_broadcast: bool = False
    note: str | None = None

    @model_validator(mode="after")
    def _source(self) -> "DonorActivation":
        if (self.values is None) == (self.values_artifact_id is None):
            raise ValueError("provide exactly one of donor values or a donor artifact id")
        if self.values is not None:
            _finite_vector(self.values, "donor activation")
        if self.values is not None and self.shape is None:
            raise ValueError("inline donor values require an explicit shape")
        if self.module.strip() == "":
            raise ValueError("a donor must name the module it was captured from")
        return self


class InterventionLimits(StrictModel):
    """Budgets that are enforced before and during execution."""

    max_elements: int = Field(default=500_000, ge=1)
    max_positions: int = Field(default=128, ge=1)
    max_applications: int = Field(default=1, ge=1)
    max_logit_elements: int = Field(default=2_000_000, ge=1)
    max_measured_modules: int = Field(default=16, ge=1)


class ActivationIntervention(StrictModel):
    """A machine-readable, non-executable intervention specification."""

    schema_version: Literal[SCHEMA_SPEC] = SCHEMA_SPEC
    intervention_id: str = Field(pattern=r"^iv_[0-9a-f]{16}$")
    created_at: str
    name: str | None = None
    model: ModelRef
    execution: ExecutionIdentity
    target: InterventionTarget
    operation: InterventionOperation
    mean: MeanActivation | None = None
    donor: DonorActivation | None = None
    apply_on_calls: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=16)
    allow_dtype_cast: bool = False
    limits: InterventionLimits = Field(default_factory=InterventionLimits)
    provenance: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    limitations: list[str] = Field(default_factory=lambda: list(INTERVENTION_LIMITATIONS))
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> "ActivationIntervention":
        if self.operation == "mean" and self.mean is None:
            raise ValueError("a mean ablation requires an explicit mean activation")
        if self.operation == "patch" and self.donor is None:
            raise ValueError("an activation patch requires an explicit donor activation")
        if self.operation in ("zero", "identity") and (self.mean is not None or self.donor is not None):
            raise ValueError(f"operation '{self.operation}' does not take a replacement source")
        if any(isinstance(item, bool) or item < 0 for item in self.apply_on_calls):
            raise ValueError("apply_on_calls must be non-negative integers")
        if len(set(self.apply_on_calls)) != len(self.apply_on_calls):
            raise ValueError("apply_on_calls must be unique")
        if len(self.apply_on_calls) > self.limits.max_applications:
            raise ValueError(
                "apply_on_calls exceeds limits.max_applications; the budget would be exceeded"
            )
        positions = self.target.selector.positions
        if len(positions) > self.limits.max_positions:
            raise ValueError("positions exceed limits.max_positions")
        if positions[-1] >= self.execution.sequence_length:
            raise ValueError(
                "a selected position lies outside the recorded sequence length",
                # pydantic wraps this; the handler report repeats it structurally
            )
        if self.mean is not None:
            provenance = self.mean.provenance
            if provenance.module != self.target.module:
                raise ValueError(
                    "mean provenance names a different module than the intervention target"
                )
            if provenance.tensor_role != self.target.tensor_role:
                raise ValueError(
                    "mean provenance names a different tensor role than the intervention target"
                )
            if (
                provenance.model_file_sha256 is not None
                and self.model.file_sha256 is not None
                and provenance.model_file_sha256 != self.model.file_sha256
            ):
                raise ValueError("mean provenance belongs to a different model file")
        if self.donor is not None:
            if self.donor.module != self.target.module:
                raise ValueError(
                    "donor values must come from the same module path as the target; "
                    "patching across capture points is not supported"
                )
            if self.donor.tensor_role != self.target.tensor_role:
                raise ValueError("donor tensor role does not match the target tensor role")
            width = self.target.selector.feature_width_hint()
            if width is not None and self.donor.values is not None:
                expected_positions = len(self.target.selector.positions)
                total = len(self.donor.values)
                if total not in (width, width * expected_positions):
                    raise ValueError(
                        "donor values do not match the selected region: expected "
                        f"{width} values for one position or {width * expected_positions} "
                        f"for {expected_positions} positions, got {total}"
                    )
        if self.operation == "mean" and self.mean is not None and self.mean.values is not None:
            width = self.target.selector.feature_width_hint()
            if width is not None and len(self.mean.values) != width:
                raise ValueError(
                    f"mean vector has {len(self.mean.values)} values but the selected feature "
                    f"region has width {width}"
                )
        return self

    @property
    def resolved_rows(self) -> list[int]:
        return list(self.target.selector.positions)


class ValidationCheck(StrictModel):
    name: str
    status: Literal["pass", "fail", "warning", "skipped"]
    detail: str | None = None


class InterventionScope(StrictModel):
    """A resolved, human-readable description of one intervention."""

    module: str
    tensor_role: TensorRole
    axis: int
    rows: list[int]
    feature_start: int | None = None
    feature_end: int | None = None
    feature_indices: list[int] = Field(default_factory=list)
    operation: InterventionOperation
    replacement_source: str
    position_semantics: PositionSemantics
    apply_on_calls: list[int]
    estimated_elements: int = Field(ge=0)
    expected_shape: list[int] | None = None
    expected_dtype: str | None = None
    human_readable: list[str] = Field(default_factory=list)


class InterventionValidation(StrictModel):
    valid: bool
    checks: list[ValidationCheck] = Field(default_factory=list)
    scope: InterventionScope | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    human_readable: list[str] = Field(default_factory=list)


class SliceStats(StrictModel):
    count: int = Field(ge=0)
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    l2_norm: float | None = None
    finite: bool = True


class SliceComparison(StrictModel):
    positions: list[int] = Field(default_factory=list)
    before: SliceStats
    after: SliceStats
    max_abs: float | None = None
    mean_abs: float | None = None
    relative_l2: float | None = None
    cosine: float | None = None


class InterventionApplication(StrictModel):
    module: str
    call_index: int = Field(ge=0)
    tensor_role: TensorRole
    tensor_shape: list[int]
    dtype: str
    axis: int
    rows: list[int]
    feature_start: int | None = None
    feature_end: int | None = None
    operation: InterventionOperation
    elements_modified: int = Field(ge=0)
    replacement_sha256: str | None = None
    dtype_cast: str | None = None


class TokenLogprob(StrictModel):
    position: int = Field(ge=0)
    token_id: int = Field(ge=0)
    logprob: float
    rank: int | None = Field(default=None, ge=1)


class TokenLogprobDelta(StrictModel):
    position: int = Field(ge=0)
    token_id: int = Field(ge=0)
    baseline: float | None = None
    identity: float | None = None
    intervention: float | None = None
    identity_delta: float | None = None
    intervention_delta: float | None = None


class DownstreamActivation(StrictModel):
    module: str
    positions: list[int]
    shape: list[int]
    diff: TensorDiffStats


class InterventionResult(StrictModel):
    """One forward pass under one condition, measured against a baseline."""

    schema_version: Literal[SCHEMA_RESULT] = SCHEMA_RESULT
    result_id: str = Field(pattern=r"^ir_[0-9a-f]{16}$")
    created_at: str
    intervention_id: str
    intervention_sha256: str
    condition: Condition
    applied: bool = False
    execution_mode: ExecutionMode
    backend: InstrumentationBackend
    backend_version: str | None = None
    model: dict[str, Any] = Field(default_factory=dict)
    sequence_length: int = Field(ge=0)
    token_ids_sha256: str | None = None
    forward_calls: int = Field(default=1, ge=0)
    logits_shape: list[int] = Field(default_factory=list)
    vocabulary_size: int | None = None
    elapsed_ms: float | None = None
    application: InterventionApplication | None = None
    target_activation: SliceComparison | None = None
    logits: TensorDiffStats | None = None
    logit_positions: list[int] = Field(default_factory=list)
    token_logprobs: list[TokenLogprobDelta] = Field(default_factory=list)
    downstream: list[DownstreamActivation] = Field(default_factory=list)
    resources: dict[str, int] = Field(default_factory=dict)
    checks: list[ValidationCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)


class ConditionSummary(StrictModel):
    condition: Condition
    result_id: str
    artifact_id: str | None = None
    applied: bool = False
    elapsed_ms: float | None = None
    logits_shape: list[int] = Field(default_factory=list)
    top1_token_ids: dict[str, int] = Field(default_factory=dict)
    target_l2_after: float | None = None


class IdentityControl(StrictModel):
    performed: bool
    passed: bool | None = None
    agreement: Literal["exact", "within_tolerance", "failed", "not-performed"] = "not-performed"
    max_abs_logits_diff: float | None = None
    mean_abs_logits_diff: float | None = None
    atol: float = 1e-6
    rtol: float = 1e-5
    detail: str | None = None


class EffectSummary(StrictModel):
    logits: TensorDiffStats | None = None
    target_activation: SliceComparison | None = None
    downstream: list[DownstreamActivation] = Field(default_factory=list)
    token_logprobs: list[TokenLogprobDelta] = Field(default_factory=list)
    direction_projection: dict[str, float | None] = Field(default_factory=dict)


class ValidityReport(StrictModel):
    status: Literal["valid", "limited", "invalid"] = "valid"
    checks: list[ValidationCheck] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class CounterfactualComparison(StrictModel):
    """Paired baseline / identity / intervention measurements."""

    schema_version: Literal[SCHEMA_COUNTERFACTUAL] = SCHEMA_COUNTERFACTUAL
    comparison_id: str = Field(pattern=r"^cf_[0-9a-f]{16}$")
    created_at: str
    case_id: str | None = None
    intervention_id: str
    intervention_sha256: str
    intervention_artifact_id: str | None = None
    trace_id: str | None = None
    replay_id: str | None = None
    replay_artifact_id: str | None = None
    capture_artifact_id: str | None = None
    model: dict[str, Any] = Field(default_factory=dict)
    execution_mode: ExecutionMode
    backend: InstrumentationBackend
    conditions: list[ConditionSummary] = Field(default_factory=list)
    identity_control: IdentityControl
    effect: EffectSummary = Field(default_factory=EffectSummary)
    validity: ValidityReport = Field(default_factory=ValidityReport)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: list(COUNTERFACTUAL_LIMITATIONS))
    interpretation: str = (
        "This record reports a paired numerical difference under the recorded conditions. "
        "It does not establish that the targeted component explains a behaviour, and it "
        "carries no automatic causal label."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)


class ContinuationConfig(StrictModel):
    """Deterministic decoding settings for a controlled continuation."""

    strategy: Literal["greedy"] = "greedy"
    max_new_tokens: int = Field(default=8, ge=1, le=4096)
    max_sequence_length: int = Field(default=2048, ge=2)
    eos_token_id: int | None = Field(default=None, ge=0)
    seed: int | None = None
    do_sample: Literal[False] = False
    temperature: Literal[0.0] = 0.0
    use_cache: Literal[False] = False
    execution_mode: Literal["uncached-full-prefix"] = "uncached-full-prefix"
    intervention_event: Literal["first-prefix-forward"] = "first-prefix-forward"
    stop_on_eos: bool = True


class InterventionEvent(StrictModel):
    step: int = Field(ge=0)
    call_index: int = Field(ge=0)
    position: int = Field(ge=0)
    operation: InterventionOperation
    elements_modified: int = Field(ge=0)
    note: str | None = None


class ContinuationCondition(StrictModel):
    condition: Condition
    generated_token_ids: list[int] = Field(default_factory=list)
    generated_text: str | None = None
    logprobs: list[float] = Field(default_factory=list)
    termination_reason: Literal["eos", "max_new_tokens", "max_sequence_length", "error"] = (
        "max_new_tokens"
    )
    steps: int = Field(default=0, ge=0)
    intervention_events: list[InterventionEvent] = Field(default_factory=list)
    elapsed_ms: float | None = None
    errors: list[str] = Field(default_factory=list)


class ContinuationDiff(StrictModel):
    baseline_identity_match: bool | None = None
    identity_agreement: Literal["exact", "failed", "not-performed"] = "not-performed"
    longest_common_prefix: int = Field(default=0, ge=0)
    first_divergence: int | None = Field(default=None, ge=0)
    length_delta: int = 0
    termination_changed: bool = False
    token_agreement_rate: float | None = None


class ContinuationComparison(StrictModel):
    """A bounded, deterministic continuation experiment under an intervention."""

    schema_version: Literal[SCHEMA_CONTINUATION] = SCHEMA_CONTINUATION
    continuation_id: str = Field(pattern=r"^cc_[0-9a-f]{16}$")
    created_at: str
    case_id: str | None = None
    trace_id: str | None = None
    replay_id: str | None = None
    replay_artifact_id: str | None = None
    intervention_id: str
    intervention_sha256: str
    intervention_artifact_id: str | None = None
    model: dict[str, Any] = Field(default_factory=dict)
    backend: InstrumentationBackend
    prefix_length: int = Field(ge=1)
    prefix_token_ids: list[int] = Field(default_factory=list)
    prefix_sha256: str | None = None
    config: ContinuationConfig
    conditions: list[ContinuationCondition] = Field(default_factory=list)
    comparison: ContinuationDiff = Field(default_factory=ContinuationDiff)
    validity: ValidityReport = Field(default_factory=ValidityReport)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: list(CONTINUATION_LIMITATIONS))
    interpretation: str = (
        "This record compares generated token sequences under the recorded conditions. "
        "A difference is not automatically an improvement, and a shorter continuation is "
        "not automatically a failure."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)


def intervention_sha256(spec: ActivationIntervention) -> str:
    """Checksum for a specification, ignoring its creation timestamp."""

    payload = spec.model_dump(mode="json")
    payload.pop("created_at", None)
    payload.pop("checksums", None)
    return sha256_bytes(canonical_bytes(payload))


def result_sha256(result: InterventionResult) -> str:
    payload = result.model_dump(mode="json")
    payload.pop("created_at", None)
    payload.pop("checksums", None)
    return sha256_bytes(canonical_bytes(payload))


def comparison_sha256(record: CounterfactualComparison) -> str:
    payload = record.model_dump(mode="json")
    payload.pop("created_at", None)
    payload.pop("checksums", None)
    return sha256_bytes(canonical_bytes(payload))


def continuation_sha256(record: ContinuationComparison) -> str:
    payload = record.model_dump(mode="json")
    payload.pop("created_at", None)
    payload.pop("checksums", None)
    return sha256_bytes(canonical_bytes(payload))


__all__ = [
    "ActivationIntervention",
    "Condition",
    "ConditionSummary",
    "ContinuationComparison",
    "ContinuationCondition",
    "ContinuationConfig",
    "ContinuationDiff",
    "CounterfactualComparison",
    "DonorActivation",
    "EffectSummary",
    "ExecutionIdentity",
    "IdentityControl",
    "InterventionApplication",
    "InterventionEvent",
    "InterventionLimits",
    "InterventionOperation",
    "InterventionResult",
    "InterventionScope",
    "InterventionTarget",
    "InterventionValidation",
    "MeanActivation",
    "MeanProvenance",
    "PositionSemantics",
    "SCHEMA_CONTINUATION",
    "SCHEMA_COUNTERFACTUAL",
    "SCHEMA_RESULT",
    "SCHEMA_SPEC",
    "SliceComparison",
    "SliceStats",
    "TensorSelector",
    "TokenLogprobDelta",
    "ValidationCheck",
    "ValidityReport",
    "comparison_sha256",
    "continuation_sha256",
    "intervention_sha256",
    "result_sha256",
]
