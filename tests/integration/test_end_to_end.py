"""End-to-end V1 workflow on a small deterministic model, fully offline."""

from __future__ import annotations

import numpy as np
import torch
from safetensors.torch import load_file

from wsp.core.hashing import sha256_file
from wsp.execution.builtin_models import TinyMLP
from wsp.inspection.factory import open_checkpoint
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace


INPUT = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]


def _forward(service: ToolService, checkpoint) -> dict:
    return service.call(
        "run_forward",
        {
            "model": {"kind": "mlp", "seed": 3, "checkpoint_path": str(checkpoint)},
            "inputs": {"x": INPUT},
            "seed": 11,
            "capture": {"module_names": ["fc1", "fc2"], "include_values": True},
        },
    )


def test_v1_acceptance_workflow(tmp_path, mlp_checkpoint) -> None:
    workspace = Workspace(tmp_path / "workspace", create=True, name="acceptance")
    service = ToolService(workspace=workspace, raise_errors=True)
    original_hash = sha256_file(mlp_checkpoint)

    # 1. Inspect a checkpoint without loading a model.
    inspected = service.call(
        "inspect_model",
        {
            "path": str(mlp_checkpoint),
            "include_tensors": True,
            "compute_file_sha256": True,
            "save_manifest": True,
        },
    )
    manifest = inspected["data"]["manifest"]
    assert manifest["identity"]["file_sha256"] == original_hash
    assert manifest["identity"]["tensor_count"] == 4
    assert {tensor["name"] for tensor in manifest["tensors"]} == {
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
    }

    # 2. Tensor summaries match reference NumPy/Torch calculations.
    tensor = service.call(
        "inspect_tensor",
        {
            "path": str(mlp_checkpoint),
            "tensor": "fc1.weight",
            "include_fingerprint": True,
        },
    )["data"]
    with open_checkpoint(mlp_checkpoint) as source:
        values = source.read_full("fc1.weight", max_elements=1_000_000)
    assert np.isclose(tensor["stats"]["mean"], float(np.mean(values)), rtol=1e-6)
    assert np.isclose(tensor["stats"]["std"], float(np.std(values)), rtol=1e-6)
    assert np.isclose(tensor["stats"]["min"], float(np.min(values)), rtol=1e-6)
    assert np.isclose(tensor["stats"]["max"], float(np.max(values)), rtol=1e-6)
    assert tensor["fingerprint"].startswith("sha256:")

    # 3. Capture baseline activations and outputs.
    baseline = _forward(service, mlp_checkpoint)
    assert baseline["ok"]
    assert baseline["data"]["capture_artifact"]
    baseline_run = baseline["run_id"]

    # 4. Create and preview a precise, bounded edit without changing behavior.
    created = service.call(
        "create_patch",
        {
            "path": str(mlp_checkpoint),
            "target": "fc1.bias",
            "params": {"kind": "add", "delta": {"values": [0.125] * 16}},
            "notes": "acceptance test",
        },
    )
    patch = created["data"]["patch"]
    patch_id = patch["patch_id"]
    assert patch["parameter_count"] == 16
    assert patch["effect"]["changed_count"] == 16

    preview = service.call(
        "preview_patch", {"path": str(mlp_checkpoint), "patch_id": patch_id}
    )["data"]["preview"]
    assert preview["compatible"] is True
    assert preview["effect"]["changed_count"] == 16
    assert sha256_file(mlp_checkpoint) == original_hash

    after_preview_baseline = _forward(service, mlp_checkpoint)
    compared_unchanged = service.call(
        "compare_forward",
        {
            "left_run_id": baseline_run,
            "right_run_id": after_preview_baseline["run_id"],
            "compare_activations": True,
        },
    )["data"]["comparison"]
    assert compared_unchanged["outputs_identical"] is True
    assert compared_unchanged["activation_comparison"]["different_count"] == 0

    # 5. Apply the patch to a derived checkpoint, never the source.
    derived = tmp_path / "derived.safetensors"
    applied = service.call(
        "apply_patch",
        {
            "path": str(mlp_checkpoint),
            "output_path": str(derived),
            "patch_ids": [patch_id],
        },
    )["data"]["patchset"]
    assert applied["status"] == "applied"
    assert applied["output_identity"]["structure_fingerprint"] == manifest["identity"]["structure_fingerprint"]
    assert sha256_file(mlp_checkpoint) == original_hash

    # 6. The exported checkpoint reloads and has exactly the expected values.
    reloaded = load_file(str(derived), device="cpu")
    original_values = load_file(str(mlp_checkpoint), device="cpu")
    expected = original_values["fc1.bias"] + torch.tensor(0.125, dtype=torch.float32)
    assert torch.equal(reloaded["fc1.bias"], expected)
    assert torch.equal(reloaded["fc2.weight"], original_values["fc2.weight"])
    with open_checkpoint(derived, compute_file_sha256=True) as source:
        assert source.identity().tensor_count == 4
        assert source.read_tensor("fc1.bias", max_elements=16).shape == (16,)

    # 7. Same input comparison measures a real output and activation difference.
    patched = _forward(service, derived)
    assert patched["ok"]
    compared = service.call(
        "compare_forward",
        {
            "left_run_id": baseline_run,
            "right_run_id": patched["run_id"],
            "compare_activations": True,
        },
    )["data"]["comparison"]
    assert compared["outputs_identical"] is False
    assert compared["entries"][0]["diff"]["max_abs"] > 0
    assert compared["activation_comparison"]["different_count"] >= 1

    # 8. Reversion selects the immutable source; original behavior returns.
    reverted = service.call(
        "revert_patch",
        {"path": str(mlp_checkpoint), "patchset_id": applied["patchset_id"]},
    )["data"]
    assert reverted["restored_by_source"] is True
    restored = _forward(service, mlp_checkpoint)
    restored_comparison = service.call(
        "compare_forward",
        {
            "left_run_id": baseline_run,
            "right_run_id": restored["run_id"],
            "compare_activations": True,
        },
    )["data"]["comparison"]
    assert restored_comparison["outputs_identical"] is True

    # 9. Provenance is sufficient to replay the operation.
    run = workspace.load_run(baseline_run)
    assert run.status == "succeeded"
    assert run.replay["operation"] == "run_forward"
    assert run.replay["seed"] == 11
    assert run.replay["parameters"]["model"]["checkpoint_path"] == str(mlp_checkpoint)
    assert run.environment["dependencies"]["torch"]
    assert run.artifacts
    assert workspace.artifacts.verify(run.artifacts[0].artifact_id)
    assert workspace.load_patch(patch_id).provenance.notes == "acceptance test"
    assert workspace.load_patchset(applied["patchset_id"]).status == "applied"
    assert sha256_file(mlp_checkpoint) == original_hash


def test_invalid_requests_fail_without_touching_source(tmp_path, mlp_checkpoint) -> None:
    service = ToolService(discover_workspace=False)
    before = sha256_file(mlp_checkpoint)
    invalid = service.call(
        "create_patch",
        {
            "path": str(mlp_checkpoint),
            "target": "fc1.weight",
            "params": {"kind": "add", "delta": {"values": [[0.0] * 3]}},
        },
    )
    assert invalid["ok"] is False
    assert invalid["error"]["code"] == "invalid_input"
    missing = service.call(
        "inspect_tensor", {"path": str(mlp_checkpoint), "tensor": "not.a.tensor"}
    )
    assert missing["ok"] is False
    assert missing["error"]["code"] == "not_found"
    non_finite = service.call(
        "create_patch",
        {
            "path": str(mlp_checkpoint),
            "target": "fc1.bias",
            "params": {"kind": "scale", "factor": 1e400},
        },
    )
    assert non_finite["ok"] is False
    assert sha256_file(mlp_checkpoint) == before
