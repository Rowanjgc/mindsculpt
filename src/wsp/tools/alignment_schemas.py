"""Tool schemas for token alignment and span resolution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from wsp.replay.sources import ReplayModelSource
from wsp.tools.schemas import ToolInput


class AlignTraceTokensInput(ToolInput):
    trace_id: str
    model: ReplayModelSource | None = Field(
        default=None,
        description="Optional local model whose tokenizer supplies offsets and special ids.",
    )
    special_token_ids: list[int] | None = None
    include_positions: bool = True
    position_offset: int = Field(default=0, ge=0)
    position_limit: int = Field(default=64, ge=1, le=5_000)
    save: bool = True


class ResolveFindingSpanInput(ToolInput):
    trace_id: str | None = None
    alignment_artifact_id: str | None = None
    model: ReplayModelSource | None = None
    special_token_ids: list[int] | None = None
    kind: Literal["token_range", "text_span", "word_range", "finding_evidence"] = "token_range"
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    role: Literal["prompt", "generation"] | None = None
    name: str | None = None
    detail: dict[str, Any] | None = None
    window: dict[str, int] | None = None
    save: bool = True

    @model_validator(mode="after")
    def _source(self) -> "ResolveFindingSpanInput":
        if self.trace_id is None and self.alignment_artifact_id is None:
            raise ValueError("provide trace_id or alignment_artifact_id")
        if self.kind == "finding_evidence" and self.detail is None:
            raise ValueError("finding_evidence requires the original evidence in 'detail'")
        if self.kind != "finding_evidence" and (self.start is None or self.end is None):
            raise ValueError("provide start and end for this span kind")
        return self
