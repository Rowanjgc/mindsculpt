"""Tool input schemas for the Behavioral Observatory."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wsp.behavior.models import (
    BackendSpec,
    ContrastReference,
    ContrastVariant,
    ContrastExpectation,
    DetectorSelection,
    EvaluatorSpec,
    EvaluationOutcome,
    Finding,
    GenerationParams,
    ModelRef,
    PatchReference,
    ReproInstructions,
    TracePrivacy,
)
from wsp.tools.schemas import ToolInput


class RecordTraceInput(ToolInput):
    input_text: str = Field(min_length=1)
    system: str | None = None
    input_artifact_id: str | None = None
    params: GenerationParams = Field(default_factory=GenerationParams)
    seed: int | None = None
    backend: BackendSpec
    privacy: TracePrivacy = Field(default_factory=TracePrivacy)


class ImportTraceInput(ToolInput):
    format: Literal["wsp", "inif", "inspect"]
    path: str
    sample_ids: list[str] = Field(default_factory=list)
    persist: bool = True


class InspectTraceInput(ToolInput):
    trace_id: str
    include_text: bool = True
    include_tokens: bool = True
    token_offset: int = Field(default=0, ge=0)
    token_limit: int = Field(default=64, ge=1, le=2_000)
    include_metadata: bool = False


class ListTracesInput(ToolInput):
    limit: int = Field(default=50, ge=1, le=1_000)


class ExportTraceInput(ToolInput):
    trace_id: str
    format: Literal["wsp", "inif"] = "wsp"
    output_path: str
    overwrite: bool = False
    redact_text: bool = False


class ScanTraceInput(ToolInput):
    trace_id: str
    detectors: list[DetectorSelection] = Field(default_factory=list)


class ListDetectorsInput(ToolInput):
    pass


class CompareTracesInput(ToolInput):
    left_trace_id: str
    right_trace_id: str


class RunContrastInput(ToolInput):
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


class EvaluateBehaviorInput(ToolInput):
    evaluator: EvaluatorSpec
    trace_id: str | None = None
    text: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> "EvaluateBehaviorInput":
        if (self.trace_id is None) == (self.text is None):
            raise ValueError("provide exactly one of trace_id or text")
        return self


class ListEvaluatorsInput(ToolInput):
    pass


class NewAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    author: str | None = None
    kind: Literal["note", "review", "reproduction", "other"] = "note"


class NewHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1)
    rationale: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class HypothesisUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    status: Literal["proposed", "testing", "supported", "refuted", "inconclusive"]


class CreateFailureCaseInput(ToolInput):
    title: str = Field(min_length=1)
    summary: str | None = None
    model: ModelRef | None = None
    inference: GenerationParams | None = None
    observations: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    contrasts: list[ContrastReference] = Field(default_factory=list)
    evaluations: list[EvaluationOutcome] = Field(default_factory=list)
    reproduction: ReproInstructions = Field(default_factory=ReproInstructions)
    annotations: list[NewAnnotation] = Field(default_factory=list)
    hypotheses: list[NewHypothesis] = Field(default_factory=list)
    activation_capture_ids: list[str] = Field(default_factory=list)
    patches: list[PatchReference] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InspectFailureCaseInput(ToolInput):
    case_id: str


class ListFailureCasesInput(ToolInput):
    status: Literal["open", "triage", "confirmed", "rejected", "archived"] | None = None
    tag: str | None = None
    limit: int = Field(default=100, ge=1, le=1_000)


class UpdateFailureCaseInput(ToolInput):
    case_id: str
    review_status: Literal["open", "triage", "confirmed", "rejected", "archived"] | None = None
    review_notes: str | None = None
    reviewer: str | None = None
    add_annotation: NewAnnotation | None = None
    add_hypothesis: NewHypothesis | None = None
    hypothesis_updates: list[HypothesisUpdate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _at_least_one(self) -> "UpdateFailureCaseInput":
        if (
            self.review_status is None
            and self.review_notes is None
            and self.reviewer is None
            and self.add_annotation is None
            and self.add_hypothesis is None
            and not self.hypothesis_updates
        ):
            raise ValueError("provide at least one update")
        return self


class ExportFailureCaseInput(ToolInput):
    case_id: str
    output_path: str
    overwrite: bool = False
    include_traces: bool = True
    redact_text: bool = False


class ImportFailureCaseInput(ToolInput):
    path: str
    overwrite: bool = False
