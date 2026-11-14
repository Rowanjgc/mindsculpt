"""Tool input schemas for targeted activation capture."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from wsp.investigation.models import CaptureMode, PositionSemantics
from wsp.replay.models import ReplayExecutionConfig, ReplayTolerances
from wsp.replay.sources import ReplayModelSource
from wsp.tools.schemas import ToolInput

_Role = Literal["prompt", "generation"]
_Instrumentation = Literal["torch_hooks", "nnterp"]


class ListCapturePointsInput(ToolInput):
    """Discover addressable modules in a locally available model."""

    model: ReplayModelSource
    include_submodules: bool = True
    kinds: list[str] | None = None
    limit: int = Field(default=200, gt=0, le=5000)
    instrumentation: _Instrumentation = "torch_hooks"


class CaptureTraceActivationsInput(ToolInput):
    """Measure selected internal states for a recorded trace.

    Positions are full-sequence token indices.  ``position_semantics`` states
    which state is measured for those tokens: ``consuming`` measures the token
    position itself, ``predicting`` measures the preceding position, which is
    the state that predicts the token.
    """

    trace_id: str
    model: ReplayModelSource
    modules: list[str] = Field(min_length=1, max_length=32)
    mode: CaptureMode = "summary"
    instrumentation: _Instrumentation = "torch_hooks"
    positions: list[int] | None = None
    role: _Role | None = None
    role_start: int | None = Field(default=None, ge=0)
    role_end: int | None = Field(default=None, ge=0)
    span_artifact_id: str | None = None
    alignment_artifact_id: str | None = None
    special_token_ids: list[int] | None = None
    position_semantics: PositionSemantics = "consuming"
    include_inputs: bool = False
    first_call_only: bool = True
    include_values: bool = False
    execution: ReplayExecutionConfig = Field(default_factory=ReplayExecutionConfig)
    tolerances: ReplayTolerances = Field(default_factory=ReplayTolerances)
    save: bool = True

    @model_validator(mode="after")
    def _one_position_source(self) -> "CaptureTraceActivationsInput":
        explicit = self.positions is not None
        span = self.span_artifact_id is not None
        ranged = self.role is not None or self.role_start is not None or self.role_end is not None
        chosen = [name for name, flag in (("positions", explicit), ("span_artifact_id", span), ("role range", ranged)) if flag]
        if len(chosen) != 1:
            raise ValueError(
                "provide exactly one position source: 'positions', 'span_artifact_id', or role/role_start/role_end"
            )
        if ranged:
            if self.role is None or self.role_start is None or self.role_end is None:
                raise ValueError("a role range requires role, role_start, and role_end")
            if self.role_end <= self.role_start:
                raise ValueError("role_end must be greater than role_start")
        if explicit and not self.positions:
            raise ValueError("positions must not be empty")
        if explicit and any(index < 0 for index in self.positions or []):
            raise ValueError("positions must be non-negative full-sequence indices")
        if self.include_values and self.mode != "selected":
            raise ValueError("include_values requires mode='selected'")
        return self


class InspectTraceCaptureInput(ToolInput):
    artifact_id: str
    include_measurements: bool = True
    measurement_offset: int = Field(default=0, ge=0)
    measurement_limit: int = Field(default=50, gt=0, le=500)
    include_values: bool = False
