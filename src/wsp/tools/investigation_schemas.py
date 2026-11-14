"""Tool input schemas for activation comparison and investigation evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wsp.tools.schemas import ToolInput

_Metric = Literal["l2", "rel_l2", "cosine", "dot", "max_abs", "mean_abs", "rms", "projection"]
_Strategy = Literal["identical_positions", "paired_positions", "matching_prefix", "semantic_event"]


class _Nested(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PositionPairInput(_Nested):
    """One explicit left/right position pairing."""

    left_position: int = Field(ge=0)
    right_position: int = Field(ge=0)
    left_token_index: int | None = Field(default=None, ge=0)
    right_token_index: int | None = Field(default=None, ge=0)
    note: str | None = None


class AlignmentInput(_Nested):
    strategy: _Strategy = "identical_positions"
    pairs: list[PositionPairInput] = Field(default_factory=list)
    reviewed: bool = False
    note: str | None = None


class CompareActivationCapturesInput(ToolInput):
    """Compare two stored capture records numerically."""

    left_capture_artifact_id: str
    right_capture_artifact_id: str
    metrics: list[_Metric] | None = None
    alignment: AlignmentInput = Field(default_factory=AlignmentInput)
    modules: list[str] | None = None
    direction: list[float] | None = None
    allow_different_models: bool = False
    atol: float = Field(default=1e-6, ge=0)
    rtol: float = Field(default=1e-5, ge=0)
    max_entries: int = Field(default=200, gt=0, le=2000)
    save: bool = True

    @model_validator(mode="after")
    def _coherent(self) -> "CompareActivationCapturesInput":
        if self.left_capture_artifact_id == self.right_capture_artifact_id:
            raise ValueError(
                "a comparison needs two different captures; comparing a capture with itself measures nothing"
            )
        if (
            self.alignment.strategy in ("paired_positions", "semantic_event")
            and not self.alignment.reviewed
        ):
            raise ValueError(
                f"alignment strategy '{self.alignment.strategy}' must be marked reviewed; "
                "the workbench cannot decide that two contexts represent the same event"
            )
        if self.alignment.strategy in ("paired_positions", "semantic_event") and not self.alignment.pairs:
            raise ValueError(
                f"alignment strategy '{self.alignment.strategy}' requires explicit position pairs"
            )
        if self.direction is not None and "projection" not in (self.metrics or []):
            raise ValueError("supply metrics=['projection'] when a direction is given")
        if "projection" in (self.metrics or []) and self.direction is None:
            raise ValueError(
                "metrics=['projection'] requires an explicit direction; the workbench does not invent one"
            )
        if self.direction is not None and not self.direction:
            raise ValueError("direction must not be empty")
        return self


class NewEvidenceHypothesis(_Nested):
    statement: str = Field(min_length=1)
    rationale: str | None = None
    proposed_by: str | None = None
    related_capture_ids: list[str] = Field(default_factory=list)
    related_comparison_ids: list[str] = Field(default_factory=list)

    @field_validator("statement")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a hypothesis statement must not be blank")
        return value


class AttachInvestigationEvidenceInput(ToolInput):
    """Build an investigation record from stored measurements and attach it to a case."""

    case_id: str | None = None
    title: str = Field(min_length=1)
    summary: str | None = None
    observation_trace_id: str | None = None
    control_trace_ids: list[str] = Field(default_factory=list)
    finding_refs: list[dict] = Field(default_factory=list)
    replay_artifact_ids: list[str] = Field(default_factory=list)
    alignment_artifact_ids: list[str] = Field(default_factory=list)
    span_artifact_ids: list[str] = Field(default_factory=list)
    capture_artifact_ids: list[str] = Field(default_factory=list)
    comparison_artifact_ids: list[str] = Field(default_factory=list)
    hypotheses: list[NewEvidenceHypothesis] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    save: bool = True

    @model_validator(mode="after")
    def _coherent(self) -> "AttachInvestigationEvidenceInput":
        measurements = (
            self.replay_artifact_ids
            or self.capture_artifact_ids
            or self.comparison_artifact_ids
        )
        if not measurements and not self.finding_refs:
            raise ValueError(
                "an investigation needs at least one finding reference or one measurement artifact"
            )
        if self.case_id is not None and not self.save:
            raise ValueError("save=False cannot attach evidence to a case")
        return self


class InspectInvestigationEvidenceInput(ToolInput):
    artifact_id: str
    include_records: bool = True
