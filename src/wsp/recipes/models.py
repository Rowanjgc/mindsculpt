"""Declarative investigation recipes.

A recipe is a versioned, inspectable description of a multi-step investigation.
It composes existing service operations; it does not bypass their validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from wsp.investigation.models import StrictModel

SCHEMA_RECIPE = "wsp.investigation-recipe/1.0"
SCHEMA_RECIPE_RUN = "wsp.recipe-run/1.0"

RecipeStatus = Literal["planned", "completed", "failed", "partial"]
StepStatus = Literal["planned", "completed", "failed", "skipped"]


class RecipeInput(StrictModel):
    name: str
    type: str = "object"
    required: bool = True
    description: str
    default: Any | None = None


class RecipeStep(StrictModel):
    index: int = Field(ge=0)
    name: str
    operation: str
    description: str
    required: bool = True
    produces: list[str] = Field(default_factory=list)
    est_cost: str = "low"
    forward_passes: int = 0


class RecipeDefinition(StrictModel):
    schema_version: Literal[SCHEMA_RECIPE] = SCHEMA_RECIPE
    recipe_id: str
    version: str
    title: str
    summary: str
    purpose: str
    group: str = "investigate"
    inputs: list[RecipeInput] = Field(default_factory=list)
    steps: list[RecipeStep] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    privacy_implications: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)


class RecipeStepRecord(StrictModel):
    index: int = Field(ge=0)
    name: str
    operation: str
    status: StepStatus
    artifact_ids: list[str] = Field(default_factory=list)
    detail: str | None = None
    error_code: str | None = None
    error: str | None = None
    elapsed_ms: float | None = Field(default=None, ge=0)


class RecipeRun(StrictModel):
    schema_version: Literal[SCHEMA_RECIPE_RUN] = SCHEMA_RECIPE_RUN
    run_id: str = Field(pattern=r"^rr_[0-9a-f]{16}$")
    created_at: str
    recipe_id: str
    recipe_version: str
    status: RecipeStatus
    case_id: str | None = None
    trace_id: str | None = None
    inputs_sha256: str | None = None
    steps: list[RecipeStepRecord] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    dossier_id: str | None = None
    validity: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)


class RecipePreview(StrictModel):
    schema_version: Literal["wsp.recipe-preview/1.0"] = "wsp.recipe-preview/1.0"
    recipe: RecipeDefinition
    inputs_valid: bool
    provided_inputs: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    missing_prerequisites: list[str] = Field(default_factory=list)
    planned_steps: list[RecipeStepRecord] = Field(default_factory=list)
    resource_estimates: dict[str, Any] = Field(default_factory=dict)
    expected_artifacts: list[str] = Field(default_factory=list)
    privacy_implications: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    executable: bool = False


__all__ = [
    "SCHEMA_RECIPE",
    "SCHEMA_RECIPE_RUN",
    "RecipeDefinition",
    "RecipeInput",
    "RecipePreview",
    "RecipeRun",
    "RecipeStatus",
    "RecipeStep",
    "RecipeStepRecord",
    "StepStatus",
]
