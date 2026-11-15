"""Tier-2 offline integration: event -> replay -> comparison -> dossier -> export."""

from __future__ import annotations

from pathlib import Path

import pytest

from wsp.behavior.events import EventTriggerDefinition
from wsp.behavior.models import ModelRef
from wsp.interventions.models import (
    ActivationIntervention,
    ExecutionIdentity,
    InterventionTarget,
    TensorSelector,
)
from wsp.replay.engine import parameter_fingerprint
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace


def _spec(sequence_length: int, position: int) -> ActivationIntervention:
    return ActivationIntervention(
        intervention_id="iv_0123456789abcdef",
        created_at="2026-01-01T00:00:00Z",
        name="offline fixture intervention",
        model=ModelRef(name="tiny-causal-lm"),
        execution=ExecutionIdentity(
            execution_mode="teacher-forced", sequence_length=sequence_length
        ),
        target=InterventionTarget(
            module="blocks.0.mlp",
            tensor_role="output",
            position_semantics="consuming",
            selector=TensorSelector(positions=[position]),
        ),
        operation="zero",
    )


def test_complete_offline_event_investigation(workspace, tmp_path: Path):
    service = ToolService(workspace, raise_errors=True)
    recorded = service.call(
        "record_trace",
        {
            "input_text": "loop",
            "backend": {"kind": "synthetic"},
            "params": {"max_new_tokens": 12},
        },
    )["data"]
    trace_id = recorded["trace_id"]
    trace = service.call(
        "inspect_trace", {"trace_id": trace_id, "token_limit": 200}
    )["data"]
    prompt_length = len(trace["input"]["token_ids"])
    full_length = prompt_length + len(trace["tokens"])

    event_data = service.call(
        "detect_behavioral_events",
        {
            "trace_id": trace_id,
            "event_definition": EventTriggerDefinition(
                kind="first_repetition", ngram_size=2, min_occurrences=2
            ).model_dump(mode="json"),
        },
    )["data"]
    event = event_data["events"][0]
    assert event["alignment"]["status"] == "exact"
    assert event["alignment"]["consumed_position"] < full_length
    event_artifact = event_data["event_artifact_id"]

    replay = service.call(
        "prepare_fixed_context_replay",
        {"trace_id": trace_id, "event_artifact_ids": [event_artifact]},
    )["data"]
    assert replay["sequence"]["full_length"] == full_length

    spec = _spec(full_length, event["alignment"]["consumed_position"])
    comparison = service.call(
        "run_fixed_context_comparison",
        {
            "replay_artifact_id": replay["artifact_id"],
            "intervention": spec.model_dump(mode="json"),
            "source": {"kind": "synthetic"},
            "continuation": {"max_new_tokens": 3},
        },
    )["data"]
    assert comparison["identity_control"]["performed"] is True
    assert comparison["identity_control"]["passed"] is True
    assert comparison["validity"]["status"] == "valid"
    assert comparison["identity_control"]["max_abs_logits_diff"] == 0.0

    case = service.call(
        "create_failure_case",
        {"title": "fixture repetition", "observations": [trace_id]},
    )["data"]["case"]
    dossier = service.call(
        "create_investigation_dossier",
        {
            "title": "fixture dossier",
            "case_id": case["case_id"],
            "trace_id": trace_id,
            "event_artifact_ids": [event_artifact],
            "fixed_context_comparison_artifact_id": comparison["artifact_id"],
        },
    )["data"]
    evidence = service.call(
        "attach_event_evidence",
        {
            "case_id": case["case_id"],
            "title": "fixture evidence",
            "trace_id": trace_id,
            "event_artifact_ids": [event_artifact],
            "fixed_context_comparison_artifact_ids": [comparison["artifact_id"]],
            "dossier_artifact_ids": [dossier["artifact_id"]],
        },
    )["data"]
    assert evidence["fixed_context_comparisons"]
    assert evidence["dossiers"]
    updated_case = service.call("inspect_failure_case", {"case_id": case["case_id"]})["data"]
    assert updated_case["case"]["investigations"]

    private_path = tmp_path / "case.wspcase"
    service.call(
        "export_failure_case",
        {
            "case_id": case["case_id"],
            "output_path": str(private_path),
            "policy": {"mode": "private"},
        },
    )
    fresh = Workspace(tmp_path / "fresh", create=True, name="fresh")
    fresh_service = ToolService(fresh, raise_errors=True)
    imported = fresh_service.call("import_failure_case", {"path": str(private_path)})["data"]
    assert imported["case"]["case_id"] == case["case_id"]
    imported_dossier = fresh_service.call(
        "inspect_investigation_dossier",
        {"dossier_artifact_id": dossier["artifact_id"]},
    )["data"]
    assert imported_dossier["dossier_id"] == dossier["dossier_id"]
    imported_comparison = fresh.artifacts.read_json(comparison["artifact_id"])
    assert imported_comparison["checksums"]["comparison_sha256"]

    redacted_path = tmp_path / "case-redacted.wspcase"
    service.call(
        "export_failure_case",
        {
            "case_id": case["case_id"],
            "output_path": str(redacted_path),
            "policy": {"mode": "redacted"},
        },
    )
    assert b"looploop" not in redacted_path.read_bytes()

    # The comparison record is immutable and the original trace is unchanged.
    stored = service.call("inspect_trace", {"trace_id": trace_id})["data"]
    assert stored["trace_id"] == trace_id
    repeated = service.call(
        "run_fixed_context_comparison",
        {
            "replay_artifact_id": replay["artifact_id"],
            "intervention": spec.model_dump(mode="json"),
            "source": {"kind": "synthetic"},
        },
    )["data"]
    assert repeated["identity_control"]["passed"] is True


def test_online_trigger_never_fires_before_observation(service):
    recorded = service.call(
        "record_trace",
        {
            "input_text": "loop",
            "backend": {"kind": "synthetic"},
            "params": {"max_new_tokens": 10},
        },
    )["data"]
    trace_id = recorded["trace_id"]
    trace = service.call("inspect_trace", {"trace_id": trace_id})["data"]
    generated = [token["token_id"] for token in trace["tokens"]]
    spec = ActivationIntervention(
        intervention_id="iv_0123456789abcdef",
        created_at="2026-01-01T00:00:00Z",
        name="online fixture",
        model=ModelRef(name="tiny-causal-lm"),
        execution=ExecutionIdentity(
            execution_mode="uncached-continuation",
            sequence_length=len(trace["input"]["token_ids"]),
        ),
        target=InterventionTarget(
            module="blocks.0.mlp",
            tensor_role="output",
            position_semantics="consuming",
            selector=TensorSelector(positions=[0]),
        ),
        operation="zero",
    )
    run = service.call(
        "run_event_triggered_intervention",
        {
            "trigger": {
                "trigger_id": "et_0123456789abcdef",
                "mode": "online",
                "definition": {"kind": "token_occurrence", "token_id": generated[0]},
                "max_triggers": 1,
                "fire_on": "once",
                "position_mode": "last",
            },
            "intervention": spec.model_dump(mode="json"),
            "source": {"kind": "synthetic"},
            "token_ids": trace["input"]["token_ids"],
            "max_new_tokens": 8,
        },
    )["data"]
    fired = [item for item in run["events"] if item["state"] == "fired"]
    assert fired
    for item in fired:
        assert min(item["fired_steps"]) > item["observation_step"]
    assert run["validity"]["status"] == "valid"
