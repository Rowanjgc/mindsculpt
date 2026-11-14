"""Versioned data models for behavioral observation and investigation.

The Behavioral Observatory deliberately keeps four kinds of records separate:

* a **trace** is an observation of one generation,
* a **finding** is a measurement produced by a detector,
* an **evaluation** is a judgement produced by an evaluator,
* a **case** links evidence for later investigation.

None of these models can express a causal claim.  Hypotheses in a case are
explicitly labelled as hypotheses, and a finding never asserts that a model is
defective.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wsp.core.models import ArtifactRef

REASONING_NOTE = (
    "Measurements and evaluations in this record describe observed behavior. "
    "They do not establish a cause or a weight-space defect."
)

TerminationReason = Literal[
    "eos",
    "stop_sequence",
    "tool_handoff",
    "length_limit",
    "error",
    "unknown",
]

Severity = Literal["info", "low", "medium", "high"]
EvaluationStatus = Literal["pass", "fail", "inconclusive", "abstain", "error"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ModelRef(StrictModel):
    """Identity of the model whose behavior was observed."""

    name: str | None = None
    revision: str | None = None
    checkpoint_path: str | None = None
    structure_fingerprint: str | None = None
    file_sha256: str | None = None
    parameter_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenizerRef(StrictModel):
    name: str | None = None
    revision: str | None = None
    kind: str | None = None
    vocab_size: int | None = Field(default=None, ge=0)
    fingerprint: str | None = None


class InputRef(StrictModel):
    """The exact input, or a reference to it when text is not persisted."""

    text: str | None = None
    system: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    artifact_id: str | None = None
    sha256: str | None = None
    media_type: str = "text/plain"


class PromptInfo(StrictModel):
    template_name: str | None = None
    template_sha256: str | None = None
    rendered: str | None = None


class BackendRef(StrictModel):
    name: str
    version: str | None = None
    device: str | None = None
    dtype: str | None = None
    deterministic: bool | None = None
    runtime: dict[str, Any] = Field(default_factory=dict)


class GenerationParams(StrictModel):
    max_new_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    stop_sequences: list[str] = Field(default_factory=list)
    do_sample: bool | None = None
    return_logprobs: bool = True
    top_logprobs: int = Field(default=0, ge=0, le=20)
    extra: dict[str, Any] = Field(default_factory=dict)

    def sampling_enabled(self) -> bool:
        if self.do_sample is not None:
            return self.do_sample
        return self.temperature > 0.0

    @model_validator(mode="after")
    def _sampling_consistency(self) -> "GenerationParams":
        if self.do_sample is False and self.temperature > 0.0:
            raise ValueError("temperature > 0 requires do_sample to be true or unset")
        if any(not sequence for sequence in self.stop_sequences):
            raise ValueError("stop sequences must be non-empty strings")
        return self


class TokenAlternative(StrictModel):
    token_id: int
    text: str | None = None
    logprob: float


class TokenRecord(StrictModel):
    token_id: int | None = None
    text: str | None = None
    logprob: float | None = None
    top_logprobs: list[TokenAlternative] = Field(default_factory=list)


class TerminationInfo(StrictModel):
    reason: TerminationReason = "unknown"
    detail: str | None = None
    stop_sequence: str | None = None
    generated_tokens: int | None = Field(default=None, ge=0)


class TimingInfo(StrictModel):
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0)


class SpanAnnotation(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ordered(self) -> "SpanAnnotation":
        if self.end <= self.start:
            raise ValueError("annotation end must be greater than start")
        return self


class ToolCallRecord(StrictModel):
    index: int = Field(ge=0)
    name: str = Field(min_length=1)
    arguments: Any | None = None
    arguments_sha256: str | None = None
    status: Literal["ok", "error", "unknown"] = "unknown"
    result_summary: str | None = None


class Provenance(StrictModel):
    origin: Literal["generated", "imported"] = "generated"
    source_system: str | None = None
    source_version: str | None = None
    source_ref: str | None = None
    source_sha256: str | None = None
    importer: str | None = None
    imported_at: str | None = None
    created_by: str | None = None
    notes: str | None = None


class TracePrivacy(StrictModel):
    """Explicit persistence policy for potentially sensitive trace text."""

    store_input_text: bool = True
    store_output_text: bool = True
    classification: Literal["unspecified", "public", "internal", "sensitive"] = "unspecified"
    note: str | None = None


class PrivacyInfo(TracePrivacy):
    redacted: bool = False


class GenerationTrace(StrictModel):
    """One observed generation.

    Token positions are implicit list indices.  Fields that could not be
    measured are either ``None`` or listed in ``unavailable`` with a reason;
    neither form is ever filled with a guessed value.
    """

    schema_version: str = "wsp.trace/1.0"
    trace_id: str = Field(pattern=r"^tr_[0-9a-f]{16}$")
    created_at: str
    model: ModelRef = Field(default_factory=ModelRef)
    tokenizer: TokenizerRef | None = None
    input: InputRef = Field(default_factory=InputRef)
    prompt: PromptInfo | None = None
    backend: BackendRef
    params: GenerationParams = Field(default_factory=GenerationParams)
    seed: int | None = None
    tokens: list[TokenRecord] = Field(default_factory=list)
    text: str | None = None
    termination: TerminationInfo = Field(default_factory=TerminationInfo)
    timing: TimingInfo = Field(default_factory=TimingInfo)
    annotations: list[SpanAnnotation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    activation_captures: list[ArtifactRef] = Field(default_factory=list)
    expected_answers: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    privacy: PrivacyInfo = Field(default_factory=PrivacyInfo)
    checksums: dict[str, str] = Field(default_factory=dict)
    unavailable: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "GenerationTrace":
        if self.tokens:
            for annotation in self.annotations:
                if annotation.end > len(self.tokens):
                    raise ValueError(
                        f"annotation {annotation.name!r} ends at {annotation.end}, "
                        f"beyond {len(self.tokens)} tokens"
                    )
        if self.termination.generated_tokens is not None and self.tokens:
            if self.termination.generated_tokens != len(self.tokens):
                raise ValueError("termination.generated_tokens must match the token count")
        if not self.privacy.store_input_text and self.input.text is not None:
            raise ValueError("input text is present but privacy.store_input_text is false")
        if not self.privacy.store_output_text and self.text is not None:
            raise ValueError("output text is present but privacy.store_output_text is false")
        return self


class TraceRecord(StrictModel):
    """Index entry pointing at an immutable trace artifact."""

    schema_version: str = "wsp.trace-index/1.0"
    trace_id: str = Field(pattern=r"^tr_[0-9a-f]{16}$")
    artifact: ArtifactRef
    created_at: str
    model_name: str | None = None
    model_revision: str | None = None
    backend: str
    token_count: int = Field(ge=0)
    text_chars: int = Field(ge=0)
    termination_reason: TerminationReason
    input_sha256: str | None = None
    provenance_origin: Literal["generated", "imported"] = "generated"


class TraceSummary(StrictModel):
    trace_id: str
    created_at: str
    model_name: str | None = None
    model_revision: str | None = None
    backend: str
    token_count: int
    text_chars: int
    termination_reason: TerminationReason
    provenance_origin: str
    artifact_id: str
    artifact_sha256: str


class EvidenceRef(StrictModel):
    kind: Literal["token_range", "word_range", "text_span", "tool_call", "field", "artifact"]
    trace_id: str | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    name: str | None = None
    artifact_id: str | None = None
    detail: Any | None = None


class Finding(StrictModel):
    schema_version: str = "wsp.finding/1.0"
    detector_id: str
    detector_version: str
    kind: str
    value: Any = None
    unit: str | None = None
    severity: Severity | None = None
    explanation: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectorSelection(StrictModel):
    detector_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class DetectorRun(StrictModel):
    detector_id: str
    detector_version: str
    status: Literal["ok", "skipped", "error"]
    findings: list[Finding] = Field(default_factory=list)
    measurements: dict[str, Any] = Field(default_factory=dict)
    skip_reason: str | None = None
    error: str | None = None
    limitations: list[str] = Field(default_factory=list)


class ScanResult(StrictModel):
    schema_version: str = "wsp.scan/1.0"
    scan_id: str = Field(pattern=r"^sc_[0-9a-f]{16}$")
    trace_id: str
    created_at: str
    detectors: list[str] = Field(default_factory=list)
    runs: list[DetectorRun] = Field(default_factory=list)
    findings_count: int = Field(default=0, ge=0)
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class EvaluatorSpec(StrictModel):
    evaluator_id: str
    version: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class EvaluationOutcome(StrictModel):
    schema_version: str = "wsp.evaluation/1.0"
    evaluation_id: str = Field(pattern=r"^ev_[0-9a-f]{16}$")
    created_at: str
    evaluator_id: str
    evaluator_version: str
    status: EvaluationStatus
    value: Any = None
    explanation: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reviewer: str | None = None
    trace_id: str | None = None
    references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JsonContract(StrictModel):
    """A deliberately small JSON contract, not a general JSON Schema engine."""

    required_keys: list[str] = Field(default_factory=list)
    properties: dict[str, Literal["string", "number", "integer", "boolean", "object", "array", "null"]] = Field(
        default_factory=dict
    )
    allow_extra: bool = True
    allow_code_fence: bool = True


class ScriptedToken(StrictModel):
    text: str
    token_id: int | None = None


class ScriptedFixture(StrictModel):
    input_equals: str | None = None
    input_contains: str | None = None
    output_text: str
    output_tokens: list[ScriptedToken] | None = None
    token_logprobs: list[float] | None = None
    termination: TerminationReason = "eos"
    termination_detail: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "ScriptedFixture":
        if self.input_equals is not None and self.input_contains is not None:
            raise ValueError("provide at most one of input_equals and input_contains")
        if self.token_logprobs is not None:
            if self.output_tokens is None:
                raise ValueError("token_logprobs require output_tokens")
            if len(self.token_logprobs) != len(self.output_tokens):
                raise ValueError("token_logprobs must align with output_tokens")
        return self


class ScriptedBackendConfig(StrictModel):
    fixtures: list[ScriptedFixture] = Field(default_factory=list)
    default_fixture: ScriptedFixture | None = None
    model_name: str = "scripted/fixture"
    model_revision: str | None = None
    tokenizer_name: str = "scripted/whitespace"
    vocab_size: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _has_fallback(self) -> "ScriptedBackendConfig":
        if not self.fixtures and self.default_fixture is None:
            raise ValueError("provide at least one fixture or a default fixture")
        return self


class HuggingFaceBackendConfig(StrictModel):
    model_path: str
    tokenizer_path: str | None = None
    device: str = "cpu"
    dtype: str | None = None
    model_name: str | None = None
    compute_file_sha256: bool = False


class BackendSpec(StrictModel):
    kind: Literal["scripted", "huggingface"]
    scripted: ScriptedBackendConfig | None = None
    huggingface: HuggingFaceBackendConfig | None = None

    @model_validator(mode="after")
    def _one_config(self) -> "BackendSpec":
        if self.kind == "scripted" and self.scripted is None:
            raise ValueError("scripted backend requires a scripted configuration")
        if self.kind == "huggingface" and self.huggingface is None:
            raise ValueError("huggingface backend requires a huggingface configuration")
        if self.kind == "scripted" and self.huggingface is not None:
            raise ValueError("unused huggingface configuration")
        if self.kind == "huggingface" and self.scripted is not None:
            raise ValueError("unused scripted configuration")
        return self


class GenerationRequest(StrictModel):
    input_text: str
    system: str | None = None
    params: GenerationParams = Field(default_factory=GenerationParams)
    seed: int | None = None


class ContrastVariant(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    input_text: str
    system: str | None = None
    notes: str | None = None


class MetricSpec(StrictModel):
    name: Literal[
        "generated_tokens",
        "repetition_score",
        "reconsideration_markers",
        "termination_is_length_limit",
    ]
    direction: Literal["unchanged", "increase", "decrease"]
    tolerance: float = Field(default=0.0, ge=0.0)


class ContrastExpectation(StrictModel):
    kind: Literal[
        "exact_match",
        "reference_agreement",
        "format",
        "metric_change",
        "property_preserved",
    ]
    description: str = Field(min_length=1)
    normalization: Literal["none", "whitespace", "casefold_whitespace"] = "whitespace"
    reference_answers: list[str] = Field(default_factory=list)
    contract: JsonContract | None = None
    metric: MetricSpec | None = None
    evaluator: EvaluatorSpec | None = None

    @model_validator(mode="after")
    def _required_parts(self) -> "ContrastExpectation":
        if self.kind == "reference_agreement" and not self.reference_answers:
            if self.evaluator is None:
                raise ValueError("reference_agreement requires reference answers or an evaluator")
        if self.kind == "format" and self.contract is None:
            raise ValueError("format expectation requires a contract")
        if self.kind == "metric_change" and self.metric is None:
            raise ValueError("metric_change requires a metric")
        if self.kind == "property_preserved" and self.evaluator is None:
            raise ValueError("property_preserved requires an evaluator")
        return self


class ContrastTest(StrictModel):
    schema_version: str = "wsp.contrast/1.0"
    test_id: str = Field(pattern=r"^ct_[0-9a-f]{16}$")
    name: str = Field(min_length=1)
    description: str | None = None
    baseline: ContrastVariant
    variants: list[ContrastVariant] = Field(min_length=1)
    expectation: ContrastExpectation
    backend: BackendSpec
    params: GenerationParams = Field(default_factory=GenerationParams)
    seed: int | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _variant_names(self) -> "ContrastTest":
        names = [variant.name for variant in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("variant names must be unique")
        if self.baseline.name in names:
            raise ValueError("baseline name must not be reused by a variant")
        return self


class VariantOutcome(StrictModel):
    name: str
    role: Literal["baseline", "variation"]
    input_text: str
    trace_id: str | None = None
    output_text: str | None = None
    output_sha256: str | None = None
    status: Literal["observed", "pass", "fail", "inconclusive", "abstain", "error"]
    explanation: str | None = None
    observed: dict[str, Any] = Field(default_factory=dict)
    evaluation: EvaluationOutcome | None = None


class ContrastResult(StrictModel):
    schema_version: str = "wsp.contrast-result/1.0"
    result_id: str = Field(pattern=r"^cr_[0-9a-f]{16}$")
    test: ContrastTest
    created_at: str
    baseline: VariantOutcome
    outcomes: list[VariantOutcome] = Field(default_factory=list)
    overall_status: EvaluationStatus
    summary: str
    trace_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CaseAnnotation(StrictModel):
    annotation_id: str = Field(pattern=r"^an_[0-9a-f]{16}$")
    created_at: str
    author: str | None = None
    kind: Literal["note", "review", "reproduction", "other"] = "note"
    text: str


class CaseHypothesis(StrictModel):
    hypothesis_id: str = Field(pattern=r"^hy_[0-9a-f]{16}$")
    created_at: str
    statement: str = Field(min_length=1)
    rationale: str | None = None
    status: Literal["proposed", "testing", "supported", "refuted", "inconclusive"] = "proposed"
    evidence_refs: list[str] = Field(default_factory=list)


class CaseReview(StrictModel):
    status: Literal["open", "triage", "confirmed", "rejected", "archived"] = "open"
    reviewers: list[str] = Field(default_factory=list)
    notes: str | None = None
    updated_at: str | None = None


class ReproInstructions(StrictModel):
    backend: str | None = None
    command: str | None = None
    notes: str | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    params: GenerationParams | None = None
    seed: int | None = None


class PatchReference(StrictModel):
    patch_id: str | None = None
    patchset_id: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> "PatchReference":
        if (self.patch_id is None) == (self.patchset_id is None):
            raise ValueError("provide exactly one of patch_id or patchset_id")
        return self


class ContrastReference(StrictModel):
    result_id: str
    artifact_id: str | None = None
    test_name: str | None = None
    overall_status: EvaluationStatus | None = None


class FailureCase(StrictModel):
    schema_version: str = "wsp.case/1.0"
    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    title: str = Field(min_length=1)
    summary: str | None = None
    created_at: str
    updated_at: str
    revision: int = Field(default=1, ge=1)
    review: CaseReview = Field(default_factory=CaseReview)
    model: ModelRef = Field(default_factory=ModelRef)
    inference: GenerationParams = Field(default_factory=GenerationParams)
    observations: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    contrasts: list[ContrastReference] = Field(default_factory=list)
    evaluations: list[EvaluationOutcome] = Field(default_factory=list)
    reproduction: ReproInstructions = Field(default_factory=ReproInstructions)
    annotations: list[CaseAnnotation] = Field(default_factory=list)
    hypotheses: list[CaseHypothesis] = Field(default_factory=list)
    activation_captures: list[ArtifactRef] = Field(default_factory=list)
    patches: list[PatchReference] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coherent(self) -> "FailureCase":
        if not (self.observations or self.controls or self.findings):
            raise ValueError("a case requires an observation trace, a control trace, or a finding")
        if len(set(self.observations)) != len(self.observations):
            raise ValueError("observation trace ids must be unique")
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("control trace ids must be unique")
        overlap = set(self.observations) & set(self.controls)
        if overlap:
            raise ValueError("a trace cannot be both an observation and a control")
        annotation_ids = [item.annotation_id for item in self.annotations]
        if len(set(annotation_ids)) != len(annotation_ids):
            raise ValueError("annotation ids must be unique")
        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError("hypothesis ids must be unique")
        return self


class CaseSummary(StrictModel):
    case_id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    revision: int
    observation_count: int
    control_count: int
    finding_count: int
    hypothesis_count: int
    tags: list[str]
    model_name: str | None = None
