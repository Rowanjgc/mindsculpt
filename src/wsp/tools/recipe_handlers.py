"""Tool handlers for high-level investigation recipes."""

from __future__ import annotations

from typing import Any

from wsp.core.budget import Budget
from wsp.core.errors import ArtifactError, InputError
from wsp.core.models import ArtifactRef
from wsp.recipes.models import SCHEMA_RECIPE_RUN, RecipeRun
from wsp.recipes.registry import get_recipe, list_recipes
from wsp.recipes.runner import preview_recipe, recipe_run_sha256, run_recipe
from wsp.tools.recipe_schemas import (
    InspectRecipeRunInput,
    ListInvestigationRecipesInput,
    PreviewInvestigationRecipeInput,
    RunInvestigationRecipeInput,
)

RECIPE_RUN_ARTIFACT_KIND = "recipe_run"


def _result(data: dict[str, Any], artifacts: list[Any] | None = None, warnings: list[str] | None = None):
    from wsp.tools.service import ToolResult

    return ToolResult(data=data, artifacts=list(artifacts or []), warnings=list(warnings or []))


def _require_workspace(service: Any) -> Any:
    if service.workspace is None:
        raise InputError(
            "this operation requires a workspace; run 'wsp init' or pass --workspace",
            details={"hint": "recipe runs and artifacts are stored in the workspace"},
        )
    return service.workspace


def list_investigation_recipes(
    service: Any, request: ListInvestigationRecipesInput, budget: Budget, run: Any
):
    recipes = [
        {
            **recipe.model_dump(mode="json"),
            "preview_operation": "preview_investigation_recipe",
            "run_operation": "run_investigation_recipe",
        }
        for recipe in list_recipes(request.group)
    ]
    return _result(
        {
            "recipes": recipes,
            "count": len(recipes),
            "note": (
                "Recipes compose existing validated operations. Preview a recipe before "
                "running it; a recipe does not add scientific conclusions."
            ),
        }
    )


def preview_investigation_recipe(
    service: Any, request: PreviewInvestigationRecipeInput, budget: Budget, run: Any
):
    preview = preview_recipe(
        request.recipe_id,
        request.inputs,
        has_workspace=service.workspace is not None,
    )
    data = preview.model_dump(mode="json")
    data["recipe_id"] = request.recipe_id
    data["will_write_artifacts"] = service.workspace is not None
    return _result(data)


def run_investigation_recipe(
    service: Any, request: RunInvestigationRecipeInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    # Ensure the recipe id is known before any nested operation is called.
    get_recipe(request.recipe_id)
    record = run_recipe(
        request.recipe_id,
        dict(request.inputs),
        service,
        has_workspace=True,
        budget=budget,
    )
    if len(record.artifacts) > budget.max_recipe_artifacts:
        from wsp.core.errors import ResourceLimitError

        raise ResourceLimitError(
            "the recipe produced more artifacts than the configured limit",
            details={"artifacts": len(record.artifacts), "limit": budget.max_recipe_artifacts},
        )
    artifacts: list[ArtifactRef] = []
    ref = None
    if request.save:
        record.checksums["run_sha256"] = recipe_run_sha256(record)
        ref = workspace.artifacts.put_json(
            RECIPE_RUN_ARTIFACT_KIND,
            record.model_dump(mode="json"),
            metadata={
                "schema_version": record.schema_version,
                "run_id": record.run_id,
                "recipe_id": record.recipe_id,
                "status": record.status,
                "steps": len(record.steps),
            },
        )
        artifacts.append(ref)
    data = record.model_dump(mode="json")
    data["artifact_id"] = ref.artifact_id if ref else None
    data["summary"] = {
        "status": record.status,
        "completed_steps": sum(1 for item in record.steps if item.status == "completed"),
        "failed_steps": sum(1 for item in record.steps if item.status == "failed"),
        "artifacts": len(record.artifacts),
        "dossier_id": record.dossier_id,
    }
    return _result(data, artifacts, record.warnings)


def inspect_recipe_run(
    service: Any, request: InspectRecipeRunInput, budget: Budget, run: Any
):
    workspace = _require_workspace(service)
    ref, _path = workspace.artifacts.get(request.run_artifact_id)
    if ref.kind != RECIPE_RUN_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a recipe run",
            details={"artifact_id": request.run_artifact_id, "kind": ref.kind},
        )
    record = RecipeRun.model_validate(workspace.artifacts.read_json(ref.artifact_id))
    if record.schema_version != SCHEMA_RECIPE_RUN:
        raise ArtifactError(
            "unsupported recipe-run schema",
            details={"schema_version": record.schema_version},
        )
    data = record.model_dump(mode="json")
    if not request.include_steps:
        data["steps"] = []
    else:
        data["steps"] = data["steps"][: request.max_entries]
    data["artifact_id"] = ref.artifact_id
    data["counts"] = {
        "steps": len(record.steps),
        "completed": sum(1 for item in record.steps if item.status == "completed"),
        "failed": sum(1 for item in record.steps if item.status == "failed"),
        "skipped": sum(1 for item in record.steps if item.status == "skipped"),
    }
    return _result(data)


def register_recipe_tools(service: Any) -> None:
    from wsp.tools.service import ToolDefinition

    def bind(function: Any) -> Any:
        def handler(request: Any, budget: Budget, run: Any):
            return function(service, request, budget, run)

        return handler

    definitions = [
        (
            "list_investigation_recipes",
            "List high-level investigation recipes with their inputs, steps, prerequisites, and limitations.",
            ListInvestigationRecipesInput,
            list_investigation_recipes,
            "A bounded recipe listing with preview/run operation names.",
            False,
        ),
        (
            "preview_investigation_recipe",
            "Inspect a recipe's planned steps, missing inputs or prerequisites, resource estimates, expected artifacts, and privacy implications without running it.",
            PreviewInvestigationRecipeInput,
            preview_investigation_recipe,
            "A recipe preview that states whether the recipe is executable.",
            False,
        ),
        (
            "run_investigation_recipe",
            "Run a bounded multi-step investigation composed from existing validated workbench operations.",
            RunInvestigationRecipeInput,
            run_investigation_recipe,
            "A recipe-run record with per-step status, preserved artifacts, validity, and warnings.",
            True,
        ),
        (
            "inspect_recipe_run",
            "Read a stored recipe run with its completed, failed, and skipped steps.",
            InspectRecipeRunInput,
            inspect_recipe_run,
            "The recipe run with a bounded step page.",
            True,
        ),
    ]
    for name, description, schema, function, output, needs_workspace in definitions:
        service.register_tool(
            ToolDefinition(
                name=name,
                description=description,
                input_model=schema,
                handler=bind(function),
                output_description=output,
                requires_workspace=needs_workspace,
            )
        )


__all__ = ["RECIPE_RUN_ARTIFACT_KIND", "register_recipe_tools"]
