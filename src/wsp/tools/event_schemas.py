"""Tool schemas for events, fixed-context replay, and dossiers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from wsp.replay.sources import ReplayModelSource
from wsp.tools.schemas import ToolInput


class CreateEventTriggerInput(ToolInput):
    trigger: dict[str, Any]
    save: bool = True


class ListEventTriggersInput(ToolInput):
    limit: int = Field(default=50, ge=1, le=500)


class DetectBehavioralEventsInput(ToolInput):
    trace_id: str
    event_definition: dict[str, Any] | None = None
    finding_index: int | None = Field(default=None, ge=0)
    finding: dict[str, Any] | None = None
    text_pattern: str | None = None
    manual_position: int | None = Field(default=None, ge=0)
    manual_start: int | None = Field(default=None, ge=0)
    manual_end: int | None = Field(default=None, ge=0)
    manual_note: str | None = None
    observation_mode: Literal["online", "reconstructed"] = "reconstructed"
    max_events: int = Field(default=8, ge=1, le=256)
    save: bool = True

    @model_validator(mode="after")
    def _one_source(self) -> "DetectBehavioralEventsInput":
        manual = self.manual_position is not None or (
            self.manual_start is not None and self.manual_end is not None
        )
        sources = [
            self.event_definition is not None,
            self.finding_index is not None or self.finding is not None,
            self.text_pattern is not None,
            manual,
        ]
        if sum(1 for item in sources if item) != 1:
            raise ValueError(
                "provide exactly one event source: event_definition, finding, "
                "text_pattern, or manual position/span"
            )
        return self


class ResolveEventPositionInput(ToolInput):
    event_artifact_id: str


class PrepareFixedContextReplayInput(ToolInput):
    trace_id: str
    event_artifact_ids: list[str] = Field(default_factory=list)
    replay_step: int | None = Field(default=None, ge=0)
    target_position: int | None = Field(default=None, ge=0)
    execution: dict[str, Any] | None = None
    save: bool = True

    @model_validator(mode="after")
    def _one_target(self) -> "PrepareFixedContextReplayInput":
        if self.replay_step is not None and self.target_position is not None:
            raise ValueError("supply replay_step or target_position, not both")
        if self.event_artifact_ids and (
            self.replay_step is not None or self.target_position is not None
        ):
            raise ValueError(
                "an event selects its own consumed position; do not also supply replay_step "
                "or target_position"
            )
        return self


class RunFixedContextComparisonInput(ToolInput):
    replay_artifact_id: str
    intervention: dict[str, Any] | None = None
    intervention_artifact_id: str | None = None
    source: ReplayModelSource
    logit_positions: list[int] | None = Field(default=None, max_length=64)
    measure_modules: list[str] = Field(default_factory=list, max_length=32)
    measure_positions: list[int] | None = Field(default=None, max_length=256)
    continuation: dict[str, Any] | None = None
    identity: bool = True
    case_id: str | None = None
    save: bool = True

    @model_validator(mode="after")
    def _intervention(self) -> "RunFixedContextComparisonInput":
        if (self.intervention is None) == (self.intervention_artifact_id is None):
            raise ValueError(
                "supply exactly one of intervention (inline) or intervention_artifact_id"
            )
        return self


class RunEventTriggeredInterventionInput(ToolInput):
    trigger: dict[str, Any] | None = None
    trigger_artifact_id: str | None = None
    intervention: dict[str, Any] | None = None
    intervention_artifact_id: str | None = None
    source: ReplayModelSource
    trace_id: str | None = None
    token_ids: list[int] | None = Field(default=None, min_length=1, max_length=4096)
    event_artifact_id: str | None = None
    max_new_tokens: int = Field(default=8, ge=1, le=4096)
    eos_token_id: int | None = Field(default=None, ge=0)
    identity: bool = True
    case_id: str | None = None
    save: bool = True

    @model_validator(mode="after")
    def _sources(self) -> "RunEventTriggeredInterventionInput":
        if (self.trigger is None) == (self.trigger_artifact_id is None):
            raise ValueError("supply exactly one of trigger or trigger_artifact_id")
        if (self.intervention is None) == (self.intervention_artifact_id is None):
            raise ValueError(
                "supply exactly one of intervention or intervention_artifact_id"
            )
        if self.trace_id is None and not self.token_ids:
            raise ValueError("supply trace_id or token_ids for the generation prefix")
        return self


class CreateInvestigationDossierInput(ToolInput):
    title: str = Field(min_length=1)
    case_id: str | None = None
    trace_id: str | None = None
    event_artifact_ids: list[str] = Field(default_factory=list)
    fixed_context_comparison_artifact_id: str | None = None
    counterfactual_comparison_artifact_id: str | None = None
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    extra_limitations: list[str] = Field(default_factory=list)
    summary: str | None = None
    save: bool = True


class InspectInvestigationDossierInput(ToolInput):
    dossier_artifact_id: str
    include_measurements: bool = True
    max_entries: int = Field(default=50, ge=1, le=500)


class AttachEventEvidenceInput(ToolInput):
    case_id: str | None = None
    title: str = Field(min_length=1)
    summary: str | None = None
    trace_id: str | None = None
    event_artifact_ids: list[str] = Field(default_factory=list)
    fixed_context_replay_artifact_ids: list[str] = Field(default_factory=list)
    fixed_context_comparison_artifact_ids: list[str] = Field(default_factory=list)
    recipe_run_artifact_ids: list[str] = Field(default_factory=list)
    dossier_artifact_ids: list[str] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    save: bool = True


class DiscoverCapabilitiesInput(ToolInput):
    query: str | None = Field(default=None, max_length=400)
    group: str | None = None
    detail: Literal["compact", "full"] = "compact"
    include_recipes: bool = True
    max_operations: int = Field(default=12, ge=1, le=100)


__all__ = [
    "AttachEventEvidenceInput",
    "CreateEventTriggerInput",
    "CreateInvestigationDossierInput",
    "DetectBehavioralEventsInput",
    "DiscoverCapabilitiesInput",
    "InspectInvestigationDossierInput",
    "ListEventTriggersInput",
    "PrepareFixedContextReplayInput",
    "ResolveEventPositionInput",
    "RunEventTriggeredInterventionInput",
    "RunFixedContextComparisonInput",
]
