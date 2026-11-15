"""Tier 3: optional real-model event investigation.

Runs only when ``WSP_REAL_MODEL`` names a local checkpoint directory. No model is
downloaded. The test validates real token events, fixed-context replay, identity
agreement, a bounded intervention, dossier persistence, and export/reload.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wsp.behavior.models import ModelRef
from wsp.core.budget import Budget
from wsp.interventions.models import (
    ActivationIntervention,
    ExecutionIdentity,
    InterventionTarget,
    TensorSelector,
)
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace

pytestmark = pytest.mark.real_model

REAL_MODEL = os.environ.get("WSP_REAL_MODEL")

if not REAL_MODEL:
    pytest.skip(
        "set WSP_REAL_MODEL to a local checkpoint directory to run tier-3 tests",
        allow_module_level=True,
    )

checkpoint = Path(REAL_MODEL).expanduser()
if not checkpoint.is_dir():
    pytest.skip(f"WSP_REAL_MODEL is not a directory: {checkpoint}", allow_module_level=True)

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> ToolService:
    root = tmp_path_factory.mktemp("real-event")
    workspace = Workspace(root / "workspace", create=True, name="real-event")
    return ToolService(
        workspace=workspace,
        raise_errors=True,
        budget=Budget(max_replay_sequence_tokens=512),
    )


def test_real_token_event_fixed_context_and_export(service, tmp_path: Path):
    source = {"kind": "local_hf", "path": str(checkpoint), "compute_file_sha256": False}
    recorded = service.call(
        "record_trace",
        {
            "input_text": "The capital of France is",
            "params": {"max_new_tokens": 6},
            "backend": {
                "kind": "huggingface",
                "huggingface": {
                    "model_path": str(checkpoint),
                    "compute_file_sha256": False,
                },
            },
        },
    )
    trace_id = recorded["data"]["trace_id"]
    trace = service.call("inspect_trace", {"trace_id": trace_id})["data"]
    generated = [token["token_id"] for token in trace["tokens"]]
    assert generated

    event_data = service.call(
        "detect_behavioral_events",
        {
            "trace_id": trace_id,
            "event_definition": {"kind": "token_occurrence", "token_id": generated[0]},
        },
    )["data"]
    event = event_data["events"][0]
    assert event["alignment"]["status"] == "exact"
    assert event["alignment"]["consumed_position"] is not None
    event_artifact = event_data["event_artifact_id"]

    replay = service.call(
        "prepare_fixed_context_replay",
        {"trace_id": trace_id, "event_artifact_ids": [event_artifact]},
    )["data"]
    full_length = replay["sequence"]["full_length"]

    points = service.call(
        "list_capture_points", {"model": source, "limit": 500}
    )["data"]["points"]
    candidates = [
        point["path"]
        for point in points
        if point["kind"] == "mlp" and point["path"].endswith(".mlp")
    ]
    assert candidates
    module = candidates[0]

    spec = ActivationIntervention(
        intervention_id="iv_0123456789abcdef",
        created_at="2026-01-01T00:00:00Z",
        name="real-model bounded zero ablation",
        model=ModelRef(name=trace["model"].get("name")),
        execution=ExecutionIdentity(
            execution_mode="teacher-forced", sequence_length=full_length
        ),
        target=InterventionTarget(
            module=module,
            tensor_role="output",
            position_semantics="consuming",
            selector=TensorSelector(positions=[event["alignment"]["consumed_position"]]),
        ),
        operation="zero",
    )
    comparison = service.call(
        "run_fixed_context_comparison",
        {
            "replay_artifact_id": replay["artifact_id"],
            "intervention": spec.model_dump(mode="json"),
            "source": source,
        },
    )["data"]
    assert comparison["identity_control"]["performed"] is True
    assert comparison["identity_control"]["passed"] is True
    assert comparison["validity"]["status"] in ("valid", "limited")
    assert comparison["measurements"]

    case = service.call(
        "create_failure_case",
        {"title": "real token event", "observations": [trace_id]},
    )["data"]["case"]
    dossier = service.call(
        "create_investigation_dossier",
        {
            "title": "real model event dossier",
            "case_id": case["case_id"],
            "trace_id": trace_id,
            "event_artifact_ids": [event_artifact],
            "fixed_context_comparison_artifact_id": comparison["artifact_id"],
        },
    )["data"]

    exported = tmp_path / "real-event.wspcase"
    service.call(
        "export_failure_case",
        {
            "case_id": case["case_id"],
            "output_path": str(exported),
            "policy": {"mode": "private"},
        },
    )
    fresh = Workspace(tmp_path / "fresh", create=True, name="fresh")
    fresh_service = ToolService(fresh, raise_errors=True)
    imported = fresh_service.call("import_failure_case", {"path": str(exported)})["data"]
    assert imported["case"]["case_id"] == case["case_id"]
    reloaded = fresh_service.call(
        "inspect_investigation_dossier",
        {"dossier_artifact_id": dossier["artifact_id"]},
    )["data"]
    assert reloaded["dossier_id"] == dossier["dossier_id"]
