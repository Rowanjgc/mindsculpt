"""Versioned schemas for verified execution and teacher-forced replay.

Replay is deliberately *not* regeneration.  A replay takes the token
identifiers that a trace recorded, presents exactly that sequence to a model,
and measures internal state.  Four properties are tracked separately and are
never collapsed into a single "verified" flag:

* ``token_sequence_reconstructed`` -- the exact identifier sequence is known.
* ``forward_executed`` -- the model consumed that sequence without error.
* ``numerical_agreement`` -- replayed measurements agree with the recorded
  ones within stated tolerances (``verified``/``partial``/``mismatch``/
  ``unavailable``).
* ``generation_reproducibility`` -- whether the model would *sample* the same
  continuation again.  Teacher forcing never claims this, so it stays
  ``not-claimed``.

A replay that executed but did not reproduce measurements is still a valid,
honest record.  It is labelled ``exploratory`` rather than ``verified``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from wsp.behavior.models import ModelRef, StrictModel, TokenEncoding, TokenizerRef

SCHEMA_PLAN = "wsp.replay-plan/1.0"
SCHEMA_RESULT = "wsp.replay/1.0"

CheckStatus = Literal["pass", "fail", "warning", "unavailable", "skipped"]
AgreementLevel = Literal["verified", "partial", "mismatch", "unavailable"]
EvidenceLevel = Literal["verified", "exploratory"]
ReplaySemantics = Literal["teacher-forced"]
GenerationReproducibility = Literal["not-claimed"]


class ReplayTolerances(StrictModel):
    """Explicit, adjustable tolerances for numerical comparison.

    Defaults are loose enough for a bfloat16 CPU replay of a cached
    autoregressive generation, where kernel selection and accumulation order
    legitimately differ.  They are never used to hide a systematic mismatch:
    every individual comparison is reported alongside the aggregate.
    """

    logprob_atol: float = Field(default=0.05, ge=0.0)
    logprob_rtol: float = Field(default=0.05, ge=0.0)
    top1_min_agreement: float = Field(default=0.95, ge=0.0, le=1.0)
    verified_min_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    partial_min_fraction: float = Field(default=0.5, ge=0.0, le=1.0)


class ReplayExecutionConfig(StrictModel):
    """Device placement and bounded execution settings."""

    device: str = "cpu"
    dtype: str | None = None
    max_sequence_length: int = Field(default=2_048, ge=1)
    seed: int | None = None
    use_cache: Literal[False] = False
    inference_mode: Literal[True] = True
    notes: list[str] = Field(default_factory=list)


class ReplayTokenSequence(StrictModel):
    """The exact sequence presented to the model, in absolute positions.

    ``prompt_length`` splits input from generated tokens.  Positions are
    zero-based indices into ``full_token_ids``.
    """

    input_token_ids: list[int] = Field(default_factory=list)
    generated_token_ids: list[int] = Field(default_factory=list)
    full_token_ids: list[int] = Field(default_factory=list)
    prompt_length: int = Field(default=0, ge=0)
    generated_length: int = Field(default=0, ge=0)
    full_length: int = Field(default=0, ge=0)
    sequence_sha256: str | None = None
    token_encoding: TokenEncoding = "unknown"
    add_special_tokens: bool | None = None
    source: Literal["recorded-token-ids", "re-derived-from-text"] = "recorded-token-ids"
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coherent(self) -> "ReplayTokenSequence":
        if any(token < 0 for token in self.full_token_ids):
            raise ValueError("token identifiers must be non-negative")
        if self.prompt_length != len(self.input_token_ids):
            raise ValueError("prompt_length must match input_token_ids")
        if self.generated_length != len(self.generated_token_ids):
            raise ValueError("generated_length must match generated_token_ids")
        if self.full_length != len(self.full_token_ids):
            raise ValueError("full_length must match full_token_ids")
        if self.full_token_ids != self.input_token_ids + self.generated_token_ids:
            raise ValueError("full_token_ids must be the input followed by generated tokens")
        return self


class ReplayExpectation(StrictModel):
    """What the trace recorded, kept verbatim for comparison.

    These are observations carried over from the original execution.  Teacher
    forcing only compares measurements; it never re-runs sampling.
    """

    recorded_generated_token_ids: list[int] = Field(default_factory=list)
    recorded_logprobs: list[float | None] = Field(default_factory=list)
    recorded_output_sha256: str | None = None
    recorded_text_chars: int | None = Field(default=None, ge=0)
    recorded_termination_reason: str | None = None
    recorded_params: dict[str, Any] = Field(default_factory=dict)
    recorded_seed: int | None = None

    @model_validator(mode="after")
    def _aligned(self) -> "ReplayExpectation":
        if self.recorded_logprobs and len(self.recorded_logprobs) != len(
            self.recorded_generated_token_ids
        ):
            raise ValueError("recorded_logprobs must align with recorded_generated_token_ids")
        return self


class ReplayCheck(StrictModel):
    """One verification check with an explicit status."""

    name: str = Field(min_length=1)
    status: CheckStatus
    detail: str | None = None
    expected: Any | None = None
    observed: Any | None = None


class ReplayPlan(StrictModel):
    """A reproducible plan to re-present one recorded token sequence."""

    schema_version: str = SCHEMA_PLAN
    plan_id: str = Field(pattern=r"^rp_[0-9a-f]{16}$")
    created_at: str
    trace_id: str = Field(pattern=r"^tr_[0-9a-f]{16}$")
    trace_artifact_id: str | None = None
    trace_artifact_sha256: str | None = None
    semantics: ReplaySemantics = "teacher-forced"
    model: ModelRef = Field(default_factory=ModelRef)
    tokenizer: TokenizerRef | None = None
    sequence: ReplayTokenSequence
    execution: ReplayExecutionConfig = Field(default_factory=ReplayExecutionConfig)
    tolerances: ReplayTolerances = Field(default_factory=ReplayTolerances)
    expectations: ReplayExpectation = Field(default_factory=ReplayExpectation)
    checks: list[ReplayCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def failed_checks(self) -> list[ReplayCheck]:
        return [check for check in self.checks if check.status == "fail"]


class ReplayModelInfo(StrictModel):
    """What a loaded model and tokenizer actually are.

    Filled from the live objects, not from the trace, so that identity checks
    compare a claim against a measurement.
    """

    name: str | None = None
    model_class: str
    parameter_count: int = Field(ge=0)
    file_sha256: str | None = None
    structure_fingerprint: str | None = None
    config_architectures: list[str] = Field(default_factory=list)
    tokenizer_name: str | None = None
    tokenizer_kind: str | None = None
    tokenizer_fingerprint: str | None = None
    tokenizer_vocab_size: int | None = Field(default=None, ge=0)
    tokenizer_max_length: int | None = Field(default=None, ge=0)
    chat_template_sha256: str | None = None
    device: str = "cpu"
    dtype: str | None = None
    backend: str
    backend_version: str | None = None
    runtime: dict[str, str] = Field(default_factory=dict)
    capabilities: dict[str, bool] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)


class ReplayIdentity(StrictModel):
    """Provenance for the execution that produced a replay record."""

    model_name: str | None = None
    model_class: str
    parameter_count: int = Field(ge=0)
    file_sha256: str | None = None
    structure_fingerprint: str | None = None
    tokenizer_name: str | None = None
    tokenizer_kind: str | None = None
    tokenizer_fingerprint: str | None = None
    tokenizer_vocab_size: int | None = Field(default=None, ge=0)
    chat_template_sha256: str | None = None
    device: str = "cpu"
    dtype: str | None = None
    backend: str
    backend_version: str | None = None
    runtime: dict[str, str] = Field(default_factory=dict)


class ReplayForwardInfo(StrictModel):
    """Measured facts about the forward execution itself."""

    sequence_length: int = Field(ge=0)
    batch_size: int = Field(default=1, ge=1)
    forward_calls: int = Field(default=1, ge=0)
    logits_shape: list[int] = Field(default_factory=list)
    logits_dtype: str | None = None
    vocabulary_size: int | None = Field(default=None, ge=0)
    device: str = "cpu"
    inference_mode: bool = True
    use_cache: bool = False
    attention_mask: Literal["all-ones"] = "all-ones"
    position_ids: Literal["contiguous-implicit"] = "contiguous-implicit"
    logits_positions: Literal["generated-positions"] = "generated-positions"
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    estimated_logits_elements: int = Field(default=0, ge=0)


class ReplayTokenAgreement(StrictModel):
    """One prediction-position comparison.

    ``generated_index`` counts generated tokens.  ``predicting_position`` is
    the absolute position whose state predicts that token, which is
    ``prompt_length - 1 + generated_index``.  Conflating the two is the most
    common off-by-one error in interpretability tooling, so both are stored.
    """

    generated_index: int = Field(ge=0)
    token_id: int = Field(ge=0)
    predicting_position: int = Field(ge=0)
    recorded_logprob: float | None = None
    replay_logprob: float | None = None
    diff: float | None = None
    tolerance: float | None = None
    within_tolerance: bool | None = None
    replay_top1_token_id: int | None = None
    top1_matches: bool | None = None


class ReplayAgreement(StrictModel):
    """Aggregate numerical comparison of replayed and recorded log-probabilities."""

    tokens_compared: int = Field(default=0, ge=0)
    tokens_within_tolerance: int = Field(default=0, ge=0)
    top1_matches: int = Field(default=0, ge=0)
    top1_agreement_rate: float | None = None
    within_tolerance_fraction: float | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    rms_diff: float | None = None
    correlation: float | None = None
    worst_generated_index: int | None = None
    level: AgreementLevel = "unavailable"
    tolerance: ReplayTolerances = Field(default_factory=ReplayTolerances)
    per_token: list[ReplayTokenAgreement] = Field(default_factory=list)
    per_token_truncated: bool = False
    unavailable_reason: str | None = None


class ReplayResult(StrictModel):
    """The outcome of executing one replay plan."""

    schema_version: str = SCHEMA_RESULT
    replay_id: str = Field(pattern=r"^ry_[0-9a-f]{16}$")
    created_at: str
    plan_id: str = Field(pattern=r"^rp_[0-9a-f]{16}$")
    plan_sha256: str
    trace_id: str = Field(pattern=r"^tr_[0-9a-f]{16}$")
    trace_artifact_sha256: str | None = None
    semantics: ReplaySemantics = "teacher-forced"
    token_sequence_reconstructed: bool = False
    forward_executed: bool = False
    numerical_agreement: AgreementLevel = "unavailable"
    generation_reproducibility: GenerationReproducibility = "not-claimed"
    evidence_level: EvidenceLevel = "exploratory"
    exploratory_reason: str | None = None
    identity: ReplayIdentity
    sequence: ReplayTokenSequence
    forward: ReplayForwardInfo | None = None
    agreement: ReplayAgreement | None = None
    checks: list[ReplayCheck] = Field(default_factory=list)
    special_token_ids: list[int] = Field(default_factory=list)
    special_positions: list[int] = Field(default_factory=list)
    captures: list[Any] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> "ReplayResult":
        if self.forward_executed and self.forward is None:
            raise ValueError("forward_executed requires forward measurements")
        if self.evidence_level == "verified" and self.numerical_agreement != "verified":
            raise ValueError("verified evidence requires verified numerical agreement")
        if self.evidence_level == "exploratory" and not self.exploratory_reason:
            raise ValueError("exploratory replays must state why they are exploratory")
        return self


TEACHER_FORCING_LIMITATIONS = [
    "Teacher-forced replay presents recorded tokens as a known sequence; it does "
    "not regenerate them and makes no claim about sampling reproducibility.",
    "A single full-sequence forward pass may differ numerically from a cached "
    "autoregressive generation because of kernel selection and accumulation order.",
    "Position identifiers are contiguous and implicit; no padding is used.",
]

HYBRID_ARCHITECTURE_NOTE = (
    "Hybrid recurrent/attention architectures are executed through the library's "
    "full-sequence forward path; recurrent state is not carried across separate "
    "calls, and chunked replay is not supported."
)
