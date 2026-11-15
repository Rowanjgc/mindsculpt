"""Execution of declarative investigation recipes.

The runner is intentionally small.  Each step calls an existing structured
operation; validation, budgets, and artifact storage remain owned by those
operations.  A failure records the failing step and preserves the artifacts that
were already produced.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from wsp.core.errors import InputError, WspError
from wsp.core.hashing import sha256_bytes
from wsp.core.time import utc_now_iso
from wsp.recipes.models import (
    RecipeDefinition,
    RecipePreview,
    RecipeRun,
    RecipeStepRecord,
)
from wsp.recipes.registry import get_recipe

RecipeExecutor = Callable[[Any, dict[str, Any]], RecipeRun]


def _new(prefix: str) -> str:
    return f"{prefix}_{__import__('secrets').token_hex(8)}"


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def recipe_run_sha256(run: RecipeRun) -> str:
    payload = run.model_dump(mode="json")
    payload.get("checksums", {}).pop("run_sha256", None)
    return "sha256:" + sha256_bytes(_canonical(payload).encode("utf-8")).split(":", 1)[1]


def _required_missing(recipe: RecipeDefinition, inputs: dict[str, Any]) -> list[str]:
    return [
        item.name
        for item in recipe.inputs
        if item.required and inputs.get(item.name) in (None, "")
    ]


def _resource_estimate(recipe: RecipeDefinition, inputs: dict[str, Any]) -> dict[str, Any]:
    forwards = sum(step.forward_passes for step in recipe.steps if step.required)
    optional_forwards = sum(step.forward_passes for step in recipe.steps if not step.required)
    return {
        "cost_class": "low" if forwards <= 3 else "medium",
        "estimated_forward_passes": forwards,
        "optional_forward_passes": optional_forwards,
        "estimated_steps": len([step for step in recipe.steps if step.required]),
        "max_new_tokens": inputs.get("max_new_tokens"),
        "capture_modules": len((inputs.get("capture") or {}).get("modules", []))
        if isinstance(inputs.get("capture"), dict)
        else 0,
    }


def preview_recipe(
    recipe_id: str,
    inputs: dict[str, Any] | None = None,
    *,
    has_workspace: bool = True,
) -> RecipePreview:
    recipe = get_recipe(recipe_id)
    provided = {key for key, value in (inputs or {}).items() if value is not None}
    missing = _required_missing(recipe, inputs or {})
    prerequisites: list[str] = []
    if not has_workspace:
        prerequisites.append("an initialized workspace for artifacts")
    if "model" in missing:
        prerequisites.append("a replay model source (synthetic or local_hf)")
    if recipe.recipe_id in ("test_activation_hypothesis", "investigate_repetition_event"):
        if not (inputs or {}).get("intervention") and not (inputs or {}).get(
            "intervention_artifact_id"
        ):
            prerequisites.append("an activation intervention specification")
    steps = [
        RecipeStepRecord(
            index=step.index,
            name=step.name,
            operation=step.operation,
            status="planned",
            detail=step.description,
        )
        for step in recipe.steps
    ]
    warnings = list(recipe.limitations[:2])
    return RecipePreview(
        recipe=recipe,
        inputs_valid=not missing and not prerequisites,
        provided_inputs=sorted(provided),
        missing_inputs=missing,
        missing_prerequisites=prerequisites,
        planned_steps=steps,
        resource_estimates=_resource_estimate(recipe, inputs or {}),
        expected_artifacts=list(recipe.expected_artifacts),
        privacy_implications=list(recipe.privacy_implications),
        warnings=warnings,
        executable=not missing and not prerequisites,
    )


class _RecipeFailure(Exception):
    def __init__(self, record: RecipeStepRecord) -> None:
        self.record = record


class _Context:
    def __init__(self, service: Any, recipe: RecipeDefinition, inputs: dict[str, Any]) -> None:
        self.service = service
        self.recipe = recipe
        self.inputs = inputs
        self.steps: list[RecipeStepRecord] = []
        self.artifacts: list[dict[str, Any]] = []
        self.data: dict[str, Any] = {}
        self.warnings: list[str] = []

    def call(
        self,
        index: int,
        name: str,
        operation: str,
        payload: dict[str, Any],
        *,
        required: bool = True,
    ) -> dict[str, Any] | None:
        started = time.perf_counter()
        try:
            envelope = self.service.call(operation, payload, raise_errors=True)
        except WspError as exc:
            if not required:
                self.steps.append(
                    RecipeStepRecord(
                        index=index,
                        name=name,
                        operation=operation,
                        status="skipped",
                        error_code=exc.code,
                        error=str(exc),
                        elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                    )
                )
                self.warnings.append(f"optional step '{name}' was skipped: {exc.message}")
                return None
            record = RecipeStepRecord(
                index=index,
                name=name,
                operation=operation,
                status="failed",
                error_code=exc.code,
                error=str(exc),
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            self.steps.append(record)
            raise _RecipeFailure(record) from exc
        except Exception as exc:  # pragma: no cover - defensive
            record = RecipeStepRecord(
                index=index,
                name=name,
                operation=operation,
                status="failed",
                error_code="internal_error",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            self.steps.append(record)
            raise _RecipeFailure(record) from exc
        artifacts = list(envelope.get("artifacts") or [])
        self.artifacts.extend(artifacts)
        self.steps.append(
            RecipeStepRecord(
                index=index,
                name=name,
                operation=operation,
                status="completed",
                artifact_ids=[item.get("artifact_id") for item in artifacts if item.get("artifact_id")],
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
        )
        data = envelope.get("data") or {}
        self.data[name] = data
        for warning in envelope.get("warnings") or []:
            if warning not in self.warnings:
                self.warnings.append(str(warning))
        return data


def _finish(
    ctx: _Context,
    *,
    status: str,
    case_id: str | None,
    trace_id: str | None,
    dossier_id: str | None = None,
    validity: dict[str, Any] | None = None,
) -> RecipeRun:
    run = RecipeRun(
        run_id=_new("rr"),
        created_at=utc_now_iso(),
        recipe_id=ctx.recipe.recipe_id,
        recipe_version=ctx.recipe.version,
        status=status,  # type: ignore[arg-type]
        case_id=case_id,
        trace_id=trace_id,
        inputs_sha256="sha256:"
        + sha256_bytes(_canonical(ctx.inputs).encode("utf-8")).split(":", 1)[1],
        steps=ctx.steps,
        artifacts=ctx.artifacts,
        dossier_id=dossier_id,
        validity=validity or {},
        warnings=list(ctx.warnings),
        limitations=list(ctx.recipe.limitations),
        provenance={"executed_by": "wsp.recipes"},
    )
    run.checksums["run_sha256"] = recipe_run_sha256(run)
    return run


def _apply_common_config(ctx: _Context, payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("replay_step", "logit_positions", "measure_modules", "measure_positions"):
        value = ctx.inputs.get(key)
        if value is not None:
            payload[key] = value
    return payload


def _exec_finding(ctx: _Context) -> RecipeRun:
    inputs = ctx.inputs
    trace_id = inputs["trace_id"]
    ctx.call(0, "inspect_trace", "inspect_trace", {"trace_id": trace_id})
    scan = ctx.call(
        1,
        "scan",
        "scan_trace",
        {
            "trace_id": trace_id,
            "detectors": [
                {
                    "detector_id": inputs.get("detector_id", "core.repetition"),
                    "parameters": inputs.get("detector_parameters") or {},
                }
            ],
        },
    )
    findings = (scan or {}).get("findings") or []
    if not findings:
        raise _RecipeFailure(
            RecipeStepRecord(
                index=1,
                name="scan",
                operation="scan_trace",
                status="failed",
                error="the detector produced no findings",
            )
        )
    event_data = ctx.call(
        2,
        "resolve_event",
        "detect_behavioral_events",
        {
            "trace_id": trace_id,
            "finding_index": int(inputs.get("finding_index", 0)),
            "save": True,
        },
    )
    event_id = (event_data or {}).get("event_id")
    event_artifact = (event_data or {}).get("event_artifact_id")
    replay = ctx.call(
        3,
        "prepare_replay",
        "prepare_fixed_context_replay",
        _apply_common_config(
            ctx,
            {
                "trace_id": trace_id,
                "event_artifact_ids": [event_artifact] if event_artifact else [],
                "save": True,
            },
        ),
    )
    capture_config = inputs.get("capture")
    if capture_config and event_data:
        positions = capture_config.get("positions") or [
            (event_data.get("alignment") or {}).get("consumed_position")
        ]
        positions = [int(value) for value in positions if value is not None]
        if positions:
            payload = {
                "trace_id": trace_id,
                "model": inputs["model"],
                "modules": list(capture_config.get("modules") or []),
                "positions": positions,
                "mode": capture_config.get("mode", "summary"),
                "position_semantics": capture_config.get("position_semantics", "consuming"),
                "save": True,
            }
            if payload["modules"]:
                ctx.call(4, "capture", "capture_trace_activations", payload, required=False)
    title = inputs.get("title") or f"Behavioral finding in {trace_id}"
    dossier = ctx.call(
        5,
        "dossier",
        "create_investigation_dossier",
        {
            "title": title,
            "case_id": inputs.get("case_id"),
            "trace_id": trace_id,
            "event_artifact_ids": [event_artifact] if event_artifact else [],
            "summary": "Resolved a detector finding to an exact token event and reconstructed its prefix.",
            "save": True,
        },
    )
    dossier_id = (dossier or {}).get("dossier_id")
    if inputs.get("case_id"):
        ctx.call(
            6,
            "attach",
            "attach_event_evidence",
            {
                "case_id": inputs["case_id"],
                "title": title,
                "summary": "Behavioral finding resolved to an exact event.",
                "trace_id": trace_id,
                "event_artifact_ids": [event_artifact] if event_artifact else [],
                "fixed_context_replay_artifact_ids": [
                    (replay or {}).get("artifact_id")
                ]
                if (replay or {}).get("artifact_id")
                else [],
                "dossier_artifact_ids": [(dossier or {}).get("artifact_id")]
                if (dossier or {}).get("artifact_id")
                else [],
                "save": True,
            },
            required=False,
        )
    return _finish(
        ctx,
        status="completed",
        case_id=inputs.get("case_id"),
        trace_id=trace_id,
        dossier_id=dossier_id,
        validity={
            "event_alignment": (event_data or {}).get("alignment", {}).get("status"),
            "identity_control": None,
        },
    )


def _exec_hypothesis(ctx: _Context) -> RecipeRun:
    inputs = ctx.inputs
    trace_id = inputs["trace_id"]
    intervention = inputs.get("intervention")
    intervention_id = inputs.get("intervention_artifact_id")
    common = {
        "source": inputs["model"],
        "trace_id": trace_id,
        "save": True,
    }
    if intervention is not None:
        common["intervention"] = intervention
    else:
        common["intervention_artifact_id"] = intervention_id
    ctx.call(0, "validate", "validate_activation_intervention", common)
    target_position = inputs.get("target_position")
    if target_position is None and isinstance(intervention, dict):
        positions = (
            ((intervention.get("target") or {}).get("selector") or {}).get("positions")
            or []
        )
        if positions:
            target_position = int(positions[0])
    replay_payload: dict[str, Any] = {"trace_id": trace_id, "save": True}
    if target_position is not None:
        replay_payload["target_position"] = int(target_position)
    replay = ctx.call(
        1,
        "prepare_replay",
        "prepare_fixed_context_replay",
        _apply_common_config(ctx, replay_payload),
    )
    compare_payload = _apply_common_config(
        ctx,
        {
            "replay_artifact_id": (replay or {}).get("artifact_id"),
            "source": inputs["model"],
            "identity": True,
            "save": True,
            "case_id": inputs.get("case_id"),
        },
    )
    if intervention is not None:
        compare_payload["intervention"] = intervention
    else:
        compare_payload["intervention_artifact_id"] = intervention_id
    comparison = ctx.call(2, "compare", "run_fixed_context_comparison", compare_payload)
    title = inputs.get("title") or f"Activation hypothesis for {trace_id}"
    dossier = ctx.call(
        3,
        "dossier",
        "create_investigation_dossier",
        {
            "title": title,
            "case_id": inputs.get("case_id"),
            "trace_id": trace_id,
            "fixed_context_comparison_artifact_id": (comparison or {}).get("artifact_id"),
            "summary": "Ran a fixed-context baseline, identity control, and activation intervention.",
            "save": True,
        },
    )
    dossier_id = (dossier or {}).get("dossier_id")
    if inputs.get("case_id"):
        ctx.call(
            4,
            "attach",
            "attach_event_evidence",
            {
                "case_id": inputs["case_id"],
                "title": title,
                "summary": "Fixed-context activation comparison.",
                "trace_id": trace_id,
                "fixed_context_comparison_artifact_ids": [
                    (comparison or {}).get("artifact_id")
                ]
                if (comparison or {}).get("artifact_id")
                else [],
                "dossier_artifact_ids": [(dossier or {}).get("artifact_id")]
                if (dossier or {}).get("artifact_id")
                else [],
                "save": True,
            },
            required=False,
        )
    validity = (comparison or {}).get("validity") or {}
    return _finish(
        ctx,
        status="completed",
        case_id=inputs.get("case_id"),
        trace_id=trace_id,
        dossier_id=dossier_id,
        validity={
            "status": validity.get("status"),
            "identity_control": ((comparison or {}).get("identity_control") or {}).get(
                "agreement"
            ),
        },
    )


def _exec_repetition(ctx: _Context) -> RecipeRun:
    inputs = ctx.inputs
    trace_id = inputs["trace_id"]
    event_data = ctx.call(
        0,
        "detect_repetition",
        "detect_behavioral_events",
        {
            "trace_id": trace_id,
            "event_definition": {
                "kind": "first_repetition",
                "ngram_size": int(inputs.get("ngram_size", 2)),
                "min_occurrences": int(inputs.get("min_occurrences", 2)),
                "consecutive": bool(inputs.get("consecutive", False)),
                "lookback": inputs.get("lookback"),
            },
            "save": True,
        },
    )
    replay = ctx.call(
        1,
        "prepare_replay",
        "prepare_fixed_context_replay",
        _apply_common_config(
            ctx,
            {
                "trace_id": trace_id,
                "event_artifact_ids": [(event_data or {}).get("event_artifact_id")]
                if (event_data or {}).get("event_artifact_id")
                else [],
                "save": True,
            },
        ),
    )
    compare_payload = _apply_common_config(
        ctx,
        {
            "replay_artifact_id": (replay or {}).get("artifact_id"),
            "source": inputs["model"],
            "identity": True,
            "save": True,
            "case_id": inputs.get("case_id"),
        },
    )
    if inputs.get("intervention") is not None:
        compare_payload["intervention"] = inputs["intervention"]
    else:
        compare_payload["intervention_artifact_id"] = inputs.get(
            "intervention_artifact_id"
        )
    comparison = ctx.call(2, "compare", "run_fixed_context_comparison", compare_payload)
    title = inputs.get("title") or f"Repetition event in {trace_id}"
    dossier = ctx.call(
        3,
        "dossier",
        "create_investigation_dossier",
        {
            "title": title,
            "case_id": inputs.get("case_id"),
            "trace_id": trace_id,
            "event_artifact_ids": [(event_data or {}).get("event_artifact_id")]
            if (event_data or {}).get("event_artifact_id")
            else [],
            "fixed_context_comparison_artifact_id": (comparison or {}).get("artifact_id"),
            "summary": "Measured the earliest configured repetition and a fixed-context intervention.",
            "extra_limitations": [
                "The repetition is a token-level fixture measurement, not a diagnosis."
            ],
            "save": True,
        },
    )
    dossier_id = (dossier or {}).get("dossier_id")
    if inputs.get("case_id"):
        ctx.call(
            4,
            "attach",
            "attach_event_evidence",
            {
                "case_id": inputs["case_id"],
                "title": title,
                "summary": "Repetition event and fixed-context comparison.",
                "trace_id": trace_id,
                "event_artifact_ids": [(event_data or {}).get("event_artifact_id")]
                if (event_data or {}).get("event_artifact_id")
                else [],
                "fixed_context_comparison_artifact_ids": [
                    (comparison or {}).get("artifact_id")
                ]
                if (comparison or {}).get("artifact_id")
                else [],
                "dossier_artifact_ids": [(dossier or {}).get("artifact_id")]
                if (dossier or {}).get("artifact_id")
                else [],
                "save": True,
            },
            required=False,
        )
    validity = (comparison or {}).get("validity") or {}
    return _finish(
        ctx,
        status="completed",
        case_id=inputs.get("case_id"),
        trace_id=trace_id,
        dossier_id=dossier_id,
        validity={
            "status": validity.get("status"),
            "identity_control": ((comparison or {}).get("identity_control") or {}).get(
                "agreement"
            ),
            "event_alignment": (event_data or {}).get("alignment", {}).get("status"),
        },
    )


_EXECUTORS: dict[str, Callable[[_Context], RecipeRun]] = {
    "investigate_behavioral_finding": _exec_finding,
    "test_activation_hypothesis": _exec_hypothesis,
    "investigate_repetition_event": _exec_repetition,
}


def run_recipe(
    recipe_id: str,
    inputs: dict[str, Any],
    service: Any,
    *,
    has_workspace: bool = True,
    budget: Any | None = None,
) -> RecipeRun:
    from wsp.core.budget import Budget
    from wsp.core.errors import ResourceLimitError

    budget = budget or Budget()
    recipe = get_recipe(recipe_id)
    missing = _required_missing(recipe, inputs)
    if not has_workspace:
        raise InputError(
            "running an investigation recipe requires a workspace",
            details={"hint": "run 'wsp init' or pass --workspace"},
        )
    if missing:
        raise InputError(
            "the recipe is missing required inputs",
            details={"missing": missing, "recipe_id": recipe_id},
        )
    if len(recipe.steps) > budget.max_recipe_steps:
        raise ResourceLimitError(
            "the recipe has more steps than the configured limit",
            details={"steps": len(recipe.steps), "limit": budget.max_recipe_steps},
        )
    estimated_forwards = sum(step.forward_passes for step in recipe.steps)
    if estimated_forwards > budget.max_recipe_forward_passes:
        raise ResourceLimitError(
            "the recipe could exceed the configured forward-pass budget",
            details={
                "estimated_forward_passes": estimated_forwards,
                "limit": budget.max_recipe_forward_passes,
            },
        )
    executor = _EXECUTORS.get(recipe_id)
    if executor is None:  # pragma: no cover - registry/executor drift guard
        raise InputError(f"recipe {recipe_id!r} has no registered executor")
    ctx = _Context(service, recipe, inputs)
    try:
        return executor(ctx)
    except _RecipeFailure as failure:
        # Preserve every completed artifact and the failing step.  A failed run is
        # never reported as completed.
        return _finish(
            ctx,
            status="failed",
            case_id=inputs.get("case_id"),
            trace_id=inputs.get("trace_id"),
            validity={"failed_step": failure.record.model_dump(mode="json")},
        )


__all__ = [
    "preview_recipe",
    "recipe_run_sha256",
    "run_recipe",
]
