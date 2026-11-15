"""Recipe planning/execution and deterministic capability discovery."""

from __future__ import annotations

import pytest

from support.temporal import make_spec
from support.tiny_lm import build_runtime, make_trace
from wsp.behavior.trace import TraceStore
from wsp.core.budget import Budget
from wsp.recipes.models import SCHEMA_RECIPE, SCHEMA_RECIPE_RUN
from wsp.tools.discovery import GROUPS


@pytest.fixture
def prepared_trace(workspace):
    runtime = build_runtime(seed=7)
    trace = make_trace(runtime, "loop", max_new_tokens=10)
    TraceStore(workspace).save(trace)
    prompt_length = len(trace.input.token_ids or [])
    tokens = [int(value) for value in (trace.input.token_ids or [])] + [
        int(token.token_id) for token in trace.tokens
    ]
    from wsp.behavior.events import EventTriggerDefinition, detect_events

    event = detect_events(
        tokens,
        definition=EventTriggerDefinition(
            kind="first_repetition", ngram_size=2, min_occurrences=2
        ),
        prompt_length=prompt_length,
        trace_id=trace.trace_id,
    )[0]
    return runtime, trace, event


def _spec(runtime, trace, event):
    return make_spec(
        "blocks.0.mlp",
        prompt_length=len(trace.input.token_ids or []) + len(trace.tokens),
        positions=[event.alignment.consumed_position],
        operation="zero",
        execution_mode="teacher-forced",
    )


def test_registry_lists_three_recipes():
    from wsp.recipes.registry import list_recipes

    recipes = list_recipes()
    assert len(recipes) >= 3
    assert {item.recipe_id for item in recipes} >= {
        "investigate_behavioral_finding",
        "test_activation_hypothesis",
        "investigate_repetition_event",
    }
    for recipe in recipes:
        assert recipe.schema_version == SCHEMA_RECIPE
        assert recipe.steps
        assert recipe.limitations
        assert recipe.prerequisites


def test_preview_reports_missing_inputs_and_costs(service):
    preview = service.call(
        "preview_investigation_recipe",
        {"recipe_id": "test_activation_hypothesis", "inputs": {}},
    )["data"]
    assert preview["executable"] is False
    assert "trace_id" in preview["missing_inputs"]
    assert preview["missing_prerequisites"]
    assert preview["resource_estimates"]["estimated_forward_passes"] >= 3
    assert preview["privacy_implications"]


def test_recipe_run_completes_and_records_every_step(service, prepared_trace):
    runtime, trace, event = prepared_trace
    spec = _spec(runtime, trace, event)
    run = service.call(
        "run_investigation_recipe",
        {
            "recipe_id": "test_activation_hypothesis",
            "inputs": {
                "trace_id": trace.trace_id,
                "model": {"kind": "synthetic"},
                "intervention": spec.model_dump(mode="json"),
            },
        },
    )["data"]
    assert run["schema_version"] == SCHEMA_RECIPE_RUN
    assert run["status"] == "completed"
    assert [step["status"] for step in run["steps"]] == ["completed"] * 4
    assert run["dossier_id"]
    assert run["artifacts"]
    inspected = service.call("inspect_recipe_run", {"run_artifact_id": run["artifact_id"]})["data"]
    assert inspected["counts"]["completed"] == 4


def test_recipe_failure_preserves_partial_progress(service, prepared_trace):
    runtime, trace, event = prepared_trace
    spec = _spec(runtime, trace, event)
    payload = spec.model_dump(mode="json")
    payload["target"]["selector"]["positions"] = [0]
    run = service.call(
        "run_investigation_recipe",
        {
            "recipe_id": "investigate_repetition_event",
            "inputs": {
                "trace_id": trace.trace_id,
                "model": {"kind": "synthetic"},
                "ngram_size": 2,
                "min_occurrences": 2,
                "intervention": payload,
            },
        },
    )["data"]
    assert run["status"] == "failed"
    failed = [step for step in run["steps"] if step["status"] == "failed"]
    assert failed
    assert failed[0]["name"] == "compare"
    assert run["dossier_id"] is None
    assert run["artifacts"]
    assert any(step["status"] == "completed" for step in run["steps"])


def test_behavioral_finding_recipe_does_not_claim_mechanism(service, prepared_trace):
    runtime, trace, event = prepared_trace
    run = service.call(
        "run_investigation_recipe",
        {
            "recipe_id": "investigate_behavioral_finding",
            "inputs": {
                "trace_id": trace.trace_id,
                "model": {"kind": "synthetic"},
                "detector_id": "core.repetition",
                "detector_parameters": {"ngram_size": 2, "min_occurrences": 2},
            },
        },
    )["data"]
    assert run["status"] == "completed"
    dossier = service.call(
        "inspect_investigation_dossier",
        {"dossier_artifact_id": run["artifacts"][-1]["artifact_id"]},
    )
    # The last artifact may be the dossier; locate it explicitly if needed.
    data = dossier["data"]
    assert data["limitations"]
    assert any("not" in item.lower() or "does not" in item.lower() for item in data["limitations"])


def test_discovery_is_progressive_and_deterministic(service):
    overview = service.call("discover_capabilities", {})["data"]
    assert overview["schema_version"] == "wsp.capabilities/1.0"
    assert len(overview["operations"]) <= 12
    assert {item["name"] for item in overview["groups"]} == set(GROUPS)
    assert overview["recipes"]
    repeat = service.call("discover_capabilities", {"query": "repetition"})["data"]
    names = {item["name"] for item in repeat["operations"]}
    assert "detect_behavioral_events" in names
    assert any(item["recipe_id"] == "investigate_repetition_event" for item in repeat["recipes"])


def test_every_tool_has_discovery_metadata(service):
    registry = service.describe(detail="compact")
    for entry in registry["operations"]:
        assert entry["group"] in GROUPS
        assert entry["purpose"]
        assert isinstance(entry["prerequisites"], list)
        assert isinstance(entry["artifact_kinds"], list)
        assert entry["cost_class"] in {"low", "medium", "high"}
