"""Tool schemas for temporal plans, compositions, and staged generation.

Every operation takes a validated, versioned record and returns bounded output.
Nothing here accepts a Python predicate as a schedule: a schedule is a declarative
record, so a plan can be stored, diffed, and re-executed by a later caller.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from wsp.replay.sources import ReplayModelSource
from wsp.tools.schemas import ToolInput

ExecutionMode = Literal["teacher-forced", "uncached-generation", "staged-generation"]
PositionMode = Literal["absolute", "last"]
Metric = Literal[
    "mean_generated_logprob",
    "final_token_logprob",
    "logit_l2_vs_baseline",
    "generated_length",
]
Isolation = Literal["recompute", "clone"]


class ScheduledInterventionInput(ToolInput):
    """One intervention plus the time axis it fires on."""

    intervention: dict[str, Any] | None = None
    intervention_artifact_id: str | None = None
    label: str | None = None
    order: int | None = Field(default=None, ge=0)
    schedule: dict[str, Any] | int | list[int] | None = None
    position_mode: PositionMode = "absolute"
    on_step_call_indices: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=16)
    allow_overlap: bool = False

    @model_validator(mode="after")
    def _source(self) -> "ScheduledInterventionInput":
        if (self.intervention is None) == (self.intervention_artifact_id is None):
            raise ValueError(
                "supply exactly one of intervention (inline) or intervention_artifact_id"
            )
        return self


class CreateInterventionPlanInput(ToolInput):
    name: str | None = None
    interventions: list[ScheduledInterventionInput] = Field(min_length=1, max_length=32)
    model: dict[str, Any] | None = None
    prompt_token_ids: list[int] | None = Field(default=None, min_length=1, max_length=4096)
    trace_id: str | None = None
    execution_mode: ExecutionMode = "uncached-generation"
    max_new_tokens: int = Field(default=8, ge=1, le=4096)
    max_sequence_length: int = Field(default=2048, ge=2)
    eos_token_id: int | None = Field(default=None, ge=0)
    stop_on_eos: bool = True
    limits: dict[str, Any] | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    save: bool = True

    @model_validator(mode="after")
    def _prompt(self) -> "CreateInterventionPlanInput":
        if not self.prompt_token_ids and self.trace_id is None:
            raise ValueError(
                "supply prompt_token_ids or a trace_id: a plan records its prompt length "
                "and the prompt digest it was built against"
            )
        orders = [item.order for item in self.interventions if item.order is not None]
        if len(set(orders)) != len(orders):
            raise ValueError("explicit intervention orders must be unique")
        return self


class _PlanRequest(ToolInput):
    plan: dict[str, Any] | None = None
    plan_artifact_id: str | None = None

    @model_validator(mode="after")
    def _source(self) -> "_PlanRequest":
        if (self.plan is None) == (self.plan_artifact_id is None):
            raise ValueError("supply exactly one of plan (inline) or plan_artifact_id")
        return self


class _PlanExecutionRequest(_PlanRequest):
    source: ReplayModelSource
    token_ids: list[int] | None = Field(default=None, min_length=1, max_length=4096)
    trace_id: str | None = None
    save: bool = True
    case_id: str | None = None


class ValidateInterventionPlanInput(_PlanRequest):
    source: ReplayModelSource | None = None
    token_ids: list[int] | None = None
    trace_id: str | None = None


class PreviewInterventionPlanInput(_PlanExecutionRequest):
    pass


class RunTemporalInterventionInput(_PlanExecutionRequest):
    identity: bool = True


class RunCompositionalComparisonInput(ToolInput):
    plan_a: dict[str, Any] | None = None
    plan_b: dict[str, Any] | None = None
    plan_ab: dict[str, Any] | None = None
    plan_a_artifact_id: str | None = None
    plan_b_artifact_id: str | None = None
    plan_ab_artifact_id: str | None = None
    negative_control_artifact_id: str | None = None
    source: ReplayModelSource
    token_ids: list[int] | None = Field(default=None, min_length=1, max_length=4096)
    trace_id: str | None = None
    metric: Metric = "mean_generated_logprob"
    identity: bool = True
    isolation: Isolation = "recompute"
    save: bool = True
    case_id: str | None = None

    @model_validator(mode="after")
    def _plans(self) -> "RunCompositionalComparisonInput":
        for name in ("a", "b", "ab"):
            inline = getattr(self, f"plan_{name}")
            artifact = getattr(self, f"plan_{name}_artifact_id")
            if (inline is None) == (artifact is None):
                raise ValueError(
                    f"supply exactly one of plan_{name} or plan_{name}_artifact_id"
                )
        return self


class InspectInterventionEventsInput(ToolInput):
    event_log_artifact_id: str | None = None
    run_artifact_id: str | None = None
    include_events: bool = True
    max_entries: int = Field(default=50, ge=1, le=500)

    @model_validator(mode="after")
    def _source(self) -> "InspectInterventionEventsInput":
        if (self.event_log_artifact_id is None) == (self.run_artifact_id is None):
            raise ValueError(
                "supply exactly one of event_log_artifact_id or run_artifact_id"
            )
        return self


class PrepareStagedGenerationInput(ToolInput):
    source: ReplayModelSource
    token_ids: list[int] = Field(min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=8, ge=1, le=4096)


class VerifyStagedGenerationInput(_PlanExecutionRequest):
    identity: bool = True
    isolation: Isolation = "recompute"
    atol: float = Field(default=1e-5, ge=0)
    rtol: float = Field(default=1e-5, ge=0)


class RunStagedInterventionInput(_PlanExecutionRequest):
    identity: bool = True
    isolation: Isolation = "recompute"
    atol: float = Field(default=1e-5, ge=0)
    rtol: float = Field(default=1e-5, ge=0)


class AttachTemporalEvidenceInput(ToolInput):
    case_id: str | None = None
    title: str = Field(min_length=1)
    summary: str | None = None
    observation_trace_id: str | None = None
    plan_artifact_ids: list[str] = Field(default_factory=list)
    event_log_artifact_ids: list[str] = Field(default_factory=list)
    run_artifact_ids: list[str] = Field(default_factory=list)
    composition_artifact_ids: list[str] = Field(default_factory=list)
    staged_artifact_ids: list[str] = Field(default_factory=list)
    verification_artifact_ids: list[str] = Field(default_factory=list)
    capture_artifact_ids: list[str] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    save: bool = True

    @model_validator(mode="after")
    def _has_evidence(self) -> "AttachTemporalEvidenceInput":
        if not (
            self.plan_artifact_ids
            or self.event_log_artifact_ids
            or self.run_artifact_ids
            or self.composition_artifact_ids
            or self.staged_artifact_ids
            or self.verification_artifact_ids
        ):
            raise ValueError(
                "at least one temporal plan, event log, run, composition, or staged record "
                "is required"
            )
        return self


__all__ = [
    "AttachTemporalEvidenceInput",
    "CreateInterventionPlanInput",
    "InspectInterventionEventsInput",
    "PrepareStagedGenerationInput",
    "PreviewInterventionPlanInput",
    "RunCompositionalComparisonInput",
    "RunStagedInterventionInput",
    "RunTemporalInterventionInput",
    "ScheduledInterventionInput",
    "ValidateInterventionPlanInput",
    "VerifyStagedGenerationInput",
]
