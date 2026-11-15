"""Tool schemas for investigation recipes."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from wsp.tools.schemas import ToolInput


class ListInvestigationRecipesInput(ToolInput):
    group: str | None = None


class PreviewInvestigationRecipeInput(ToolInput):
    recipe_id: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class RunInvestigationRecipeInput(ToolInput):
    recipe_id: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    save: bool = True


class InspectRecipeRunInput(ToolInput):
    run_artifact_id: str
    include_steps: bool = True
    max_entries: int = Field(default=50, ge=1, le=500)


__all__ = [
    "InspectRecipeRunInput",
    "ListInvestigationRecipesInput",
    "PreviewInvestigationRecipeInput",
    "RunInvestigationRecipeInput",
]
