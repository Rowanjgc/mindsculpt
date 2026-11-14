"""Workspace, artifact, service, and resource-safety tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from wsp.core.budget import Budget
from wsp.core.errors import ArtifactError, NotFoundError, SafetyError
from wsp.core.hashing import sha256_file
from wsp.execution.builtin_models import TinyMLP
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace


def _hash(path: Path) -> str:
    return sha256_file(path)


def test_workspace_creation_is_explicit_and_safe(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    workspace = Workspace(root, create=True, name="lab")
    assert workspace.config.name == "lab"
    assert (root / "workspace.json").exists()
    with pytest.raises(SafetyError):
        Workspace(tmp_path, create=True)


def test_artifacts_are_content_addressed_and_verified(workspace: Workspace) -> None:
    first = workspace.artifacts.put_bytes("notes", b"hello", format="txt")
    second = workspace.artifacts.put_bytes("notes", b"hello", format="txt")
    assert first.artifact_id == second.artifact_id
    assert first.sha256 == "sha256:" + hashlib.sha256(b"hello").hexdigest()
    assert first.path.startswith("artifacts/")
    assert (workspace.root / first.path).exists()
    assert not (workspace.root / "artifacts" / "artifacts").exists()
    assert workspace.artifacts.read_bytes(first.artifact_id) == b"hello"
    assert workspace.artifacts.verify(first.artifact_id)
    with pytest.raises((ArtifactError, NotFoundError)):
        workspace.artifacts.get("../workspace.json")


def test_artifact_array_and_json_round_trip(workspace: Workspace) -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    ref = workspace.artifacts.put_array("probe", array, metadata={"case": "unit"})
    restored = workspace.artifacts.read_array(ref.artifact_id)
    np.testing.assert_array_equal(restored, array)
    json_ref = workspace.artifacts.put_json("probe-json", {"a": [1, 2, 3]})
    assert workspace.artifacts.read_json(json_ref.artifact_id)["a"] == [1, 2, 3]


def test_run_record_contains_replay_provenance(
    workspace: Workspace, mlp_checkpoint: Path
) -> None:
    service = ToolService(workspace=workspace, raise_errors=True)
    envelope = service.call(
        "inspect_model",
        {"path": str(mlp_checkpoint), "compute_file_sha256": True, "save_manifest": True},
    )
    assert envelope["ok"]
    run_id = envelope["run_id"]
    assert run_id is not None
    record = workspace.load_run(run_id)
    assert record.status == "succeeded"
    assert record.operation == "inspect_model"
    assert record.model_identity is not None
    assert record.environment["dependencies"]["numpy"]
    assert record.replay["operation"] == "inspect_model"
    assert record.replay["parameters"]["path"] == str(mlp_checkpoint)
    assert record.replay["budget"]["max_tensor_numel"] == Budget().max_tensor_numel
    assert record.completed_at is not None
    assert workspace.load_manifest(record.model_identity.structure_fingerprint).identity.tensor_count == 4


def test_experiment_groups_runs(workspace: Workspace, mlp_checkpoint: Path) -> None:
    service = ToolService(workspace=workspace, raise_errors=True)
    envelope = service.call("list_tensors", {"path": str(mlp_checkpoint)})
    experiment = workspace.create_experiment("inspection", tags=["unit"])
    updated = workspace.add_run_to_experiment(experiment.experiment_id, envelope["run_id"])
    assert updated.run_ids == [envelope["run_id"]]
    assert workspace.list_experiments()[0].name == "inspection"


def test_service_describes_schemas() -> None:
    service = ToolService(discover_workspace=False)
    registry = service.describe()
    names = {entry["name"] for entry in registry["operations"]}
    assert {
        "inspect_model",
        "create_patch",
        "apply_patch",
        "run_forward",
        "capture_activations",
        "compare_forward",
    } <= names
    patch_schema = service.describe("create_patch")["operations"][0]["input_schema"]
    assert "params" in patch_schema["properties"]
    assert patch_schema.get("additionalProperties") is False


def test_service_returns_machine_readable_validation_error() -> None:
    service = ToolService(discover_workspace=False)
    envelope = service.call("inspect_tensor", {"path": "x", "tensor": 5})
    assert envelope["schema_version"] == "wsp.tool/1.0"
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "invalid_input"
    assert envelope["error"]["details"]["errors"]


def test_service_resource_limit_is_reported(
    workspace: Workspace, mlp_checkpoint: Path
) -> None:
    service = ToolService(workspace=workspace, raise_errors=True)
    with pytest.raises(Exception) as excinfo:
        service.call(
            "inspect_model",
            {"path": str(mlp_checkpoint), "budget": {"max_output_bytes": 10}},
        )
    assert "response" in str(excinfo.value)


def test_patch_payload_externalization(workspace: Workspace, tmp_path: Path) -> None:
    model = TinyMLP(input_dim=128, hidden_dim=64)
    checkpoint = tmp_path / "large-mlp.safetensors"
    from safetensors.torch import save_file

    save_file(
        {key: value.detach().clone() for key, value in model.state_dict().items()},
        str(checkpoint),
    )
    service = ToolService(workspace=workspace, raise_errors=True)
    delta = [[0.01] * 128 for _ in range(64)]
    envelope = service.call(
        "create_patch",
        {
            "path": str(checkpoint),
            "target": "fc1.weight",
            "params": {"kind": "add", "delta": {"values": delta}},
            "externalize_payloads": True,
        },
    )
    assert envelope["ok"]
    patch = envelope["data"]["patch"]
    assert patch["params"]["delta"]["artifact_id"]
    assert envelope["artifacts"][0]["kind"] == "patch_payload"
    assert workspace.artifacts.verify(envelope["artifacts"][0]["artifact_id"])


def test_saving_equivalent_patches_is_idempotent(
    workspace: Workspace, mlp_checkpoint: Path
) -> None:
    service = ToolService(workspace=workspace, raise_errors=True)
    first = service.call(
        "create_patch",
        {
            "path": str(mlp_checkpoint),
            "target": "fc1.bias",
            "params": {"kind": "add", "delta": {"values": [0.125] * 16}},
            "notes": "first note",
        },
    )
    second = service.call(
        "create_patch",
        {
            "path": str(mlp_checkpoint),
            "target": "fc1.bias",
            "params": {"kind": "add", "delta": {"values": [0.125] * 16}},
            "notes": "a later note",
        },
    )
    assert first["ok"] and second["ok"]
    assert first["data"]["patch"]["patch_id"] == second["data"]["patch"]["patch_id"]
    assert len(workspace.list_patches()) == 1


def test_equivalent_patchset_reapplication_refreshes_output_identity(
    workspace: Workspace, mlp_checkpoint: Path, tmp_path: Path
) -> None:
    from wsp.core.selection import Selection
    from wsp.inspection.factory import open_checkpoint
    from wsp.patches.engine import apply_patchset, build_patchset, create_patch
    from wsp.patches.spec import AddParams, PayloadRef

    budget = Budget()
    with open_checkpoint(mlp_checkpoint) as source:
        patch = create_patch(
            source,
            target="fc1.bias",
            selection=Selection(),
            params=AddParams(
                delta=PayloadRef(
                    values=[0.125] * 16,
                    shape=[16],
                    dtype="float64",
                )
            ),
            budget=budget,
        )
    output = tmp_path / "patched.safetensors"
    with open_checkpoint(mlp_checkpoint) as source:
        patchset = build_patchset(source, [patch], budget=budget, output_path=output)
        first = apply_patchset(source, patchset, output, budget=budget)
    workspace.save_patchset(first)

    equivalent = first.model_copy(deep=True)
    equivalent.created_at = "2000-01-01T00:00:00Z"
    assert equivalent.output_identity is not None
    original_hash = equivalent.output_identity.file_sha256
    equivalent.output_identity.file_sha256 = "sha256:" + "0" * 64
    workspace.save_patchset(equivalent)

    loaded = workspace.load_patchset(first.patchset_id)
    assert loaded.output_identity is not None
    assert loaded.output_identity.file_sha256 == "sha256:" + "0" * 64
    assert loaded.output_identity.file_sha256 != original_hash
    assert loaded.created_at == first.created_at

    conflicting = equivalent.model_copy(deep=True)
    conflicting.status = "pending"
    with pytest.raises(ArtifactError):
        workspace.save_patchset(conflicting)


def test_incompatible_patch_is_rejected_by_service(
    workspace: Workspace, mlp_checkpoint: Path, transformer_checkpoint: Path
) -> None:
    service = ToolService(workspace=workspace, raise_errors=True)
    created = service.call(
        "create_patch",
        {
            "path": str(mlp_checkpoint),
            "target": "fc1.bias",
            "params": {"kind": "scale", "factor": 0.5},
        },
    )
    patch_id = created["data"]["patch"]["patch_id"]
    with pytest.raises(Exception) as excinfo:
        service.call(
            "preview_patch", {"path": str(transformer_checkpoint), "patch_id": patch_id}
        )
    assert "incompatible" in str(excinfo.value).lower()


def test_forward_run_saves_artifacts_and_replays(workspace: Workspace) -> None:
    service = ToolService(workspace=workspace, raise_errors=True)
    envelope = service.call(
        "run_forward",
        {
            "model": {"kind": "mlp", "seed": 4},
            "inputs": {"x": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]},
            "seed": 9,
            "capture": {"module_names": ["fc1"], "include_values": True},
        },
    )
    assert envelope["ok"]
    data = envelope["data"]
    assert data["capture_artifact"]
    assert data["output_artifacts"]["output"]
    assert len(envelope["artifacts"]) >= 2
    record = workspace.load_run(envelope["run_id"])
    assert record.seed == 9
    assert record.device == "cpu"
    assert record.result["forward"]["outputs"][0]["key"] == "output"
    assert record.artifacts
    assert all(workspace.artifacts.verify(ref.artifact_id) for ref in record.artifacts)


def test_capture_artifact_comparison(workspace: Workspace) -> None:
    service = ToolService(workspace=workspace, raise_errors=True)
    arguments = {
        "model": {"kind": "mlp", "seed": 4},
        "inputs": {"x": [[0.1] * 8]},
        "seed": 4,
        "capture": {"module_names": ["fc1"], "include_values": True},
    }
    first = service.call("capture_activations", arguments)
    second = service.call("capture_activations", arguments)
    comparison = service.call(
        "compare_activations",
        {
            "left_capture_artifact": first["data"]["capture_artifact"],
            "right_capture_artifact": second["data"]["capture_artifact"],
        },
    )
    assert comparison["ok"]
    assert comparison["data"]["comparison"]["identical_count"] >= 1
    assert comparison["data"]["comparison"]["different_count"] == 0


def test_source_is_unchanged_by_preview_and_apply(
    workspace: Workspace, mlp_checkpoint: Path, tmp_path: Path
) -> None:
    before = _hash(mlp_checkpoint)
    service = ToolService(workspace=workspace, raise_errors=True)
    created = service.call(
        "create_patch",
        {
            "path": str(mlp_checkpoint),
            "target": "fc2.bias",
            "params": {"kind": "add", "delta": {"values": [0.125] * 4}},
        },
    )
    patch_id = created["data"]["patch"]["patch_id"]
    service.call("preview_patch", {"path": str(mlp_checkpoint), "patch_id": patch_id})
    assert _hash(mlp_checkpoint) == before
    service.call(
        "apply_patch",
        {
            "path": str(mlp_checkpoint),
            "output_path": str(tmp_path / "derived.safetensors"),
            "patch_ids": [patch_id],
        },
    )
    assert _hash(mlp_checkpoint) == before


def test_export_refuses_to_overwrite(tmp_path: Path, mlp_checkpoint: Path) -> None:
    service = ToolService(discover_workspace=False)
    first = service.call(
        "export_checkpoint",
        {
            "path": str(mlp_checkpoint),
            "output_path": str(tmp_path / "copy.safetensors"),
        },
    )
    assert first["ok"]
    second = service.call(
        "export_checkpoint",
        {
            "path": str(mlp_checkpoint),
            "output_path": str(tmp_path / "copy.safetensors"),
        },
    )
    assert second["ok"] is False
    assert second["error"]["code"] == "safety_violation"
