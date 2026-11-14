"""Tool input schemas for replay and (later) alignment and capture."""

from __future__ import annotations

from pydantic import Field, model_validator

from wsp.replay.models import ReplayExecutionConfig, ReplayTolerances
from wsp.replay.sources import ReplayModelSource
from wsp.tools.schemas import ToolInput


class PrepareReplayInput(ToolInput):
    trace_id: str
    execution: ReplayExecutionConfig = Field(default_factory=ReplayExecutionConfig)
    tolerances: ReplayTolerances = Field(default_factory=ReplayTolerances)
    save: bool = True
    notes: str | None = None


class VerifyReplayInput(ToolInput):
    """Execute a teacher-forced replay and record measured agreement.

    Provide either ``trace_id`` (a plan is prepared on the fly, or reused when
    ``plan_artifact_id`` is also given) or ``plan_artifact_id`` alone.
    """

    trace_id: str | None = None
    plan_artifact_id: str | None = None
    model: ReplayModelSource
    execution: ReplayExecutionConfig | None = None
    tolerances: ReplayTolerances | None = None
    include_per_token: bool = True
    save: bool = True

    @model_validator(mode="after")
    def _source(self) -> "VerifyReplayInput":
        if self.trace_id is None and self.plan_artifact_id is None:
            raise ValueError("provide trace_id or plan_artifact_id")
        if self.plan_artifact_id is not None and self.trace_id is not None:
            raise ValueError(
                "provide either trace_id or plan_artifact_id; a stored plan already names its trace"
            )
        return self


__all__ = ["PrepareReplayInput", "VerifyReplayInput"]
