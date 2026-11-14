"""Offline end-to-end demonstration of the WSP Workbench instruments.

The script:

1. inspects a SafeTensors checkpoint without loading a model;
2. captures activations from a deterministic forward pass;
3. creates and previews an immutable weight patch;
4. applies the patch to a derived checkpoint (never the source);
5. reloads the derived checkpoint and compares forward results;
6. verifies that rollback selects the untouched source;
7. saves the manifest, patch, patch set, and run metadata in a workspace.

Run ``examples/make_tiny_model.py`` first, or pass ``--checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from wsp.activations.capture import ActivationCapture
from wsp.core.budget import Budget
from wsp.core.hashing import sha256_file
from wsp.core.json import to_jsonable
from wsp.core.selection import Selection
from wsp.execution.compare import compare_forward
from wsp.execution.forward import run_forward
from wsp.execution.loaders import load_state_dict_into_module
from wsp.inspection.factory import open_checkpoint
from wsp.inspection.manifest import build_manifest
from wsp.patches.engine import (
    apply_patchset,
    build_patchset,
    create_patch,
    preview_patch,
    revert_patchset,
)
from wsp.patches.spec import AddParams, PayloadRef
from wsp.tensors.stats import basic_stats
from wsp.workspace.store import Workspace


def _fresh_model(checkpoint: Path):
    from wsp.execution.builtin_models import TinyMLP

    model = TinyMLP(input_dim=8, hidden_dim=16, output_dim=4)
    load_state_dict_into_module(model, checkpoint)
    return model


def _forward(model, x: torch.Tensor):
    capture = ActivationCapture(
        model,
        ["fc1"],
        record_inputs=True,
        include_values=True,
        max_elements_per_tensor=4096,
    )
    execution = run_forward(
        model,
        inputs={"x": x},
        capture=capture,
        seed=0,
    )
    return execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=here / "out" / "tiny_mlp.safetensors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "out" / "tiny_mlp.patched.safetensors",
        help="derived checkpoint; the source is never modified",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=here / "out" / "workspace",
        help="local workspace for manifests, patches, and run records",
    )
    parser.add_argument("--force", action="store_true", help="allow replacing the derived checkpoint")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    budget = Budget()

    if not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        print("run examples/make_tiny_model.py first", file=sys.stderr)
        return 2

    source_hash_before = sha256_file(checkpoint)

    # ---------------------------------------------------------------- inspect
    with open_checkpoint(checkpoint, compute_file_sha256=True) as source:
        manifest = build_manifest(
            source, include_tensors=True, max_tensors=budget.max_manifest_tensors
        )
        bias = source.read_full("fc1.bias", max_elements=budget.max_tensor_numel)
        bias_info = source.tensor_info("fc1.bias")
    bias_stats = basic_stats(bias, storage_dtype=bias_info.dtype)

    workspace = Workspace(args.workspace, create=not (args.workspace / "workspace.json").exists())
    manifest_path = workspace.save_manifest(manifest)

    print(
        json.dumps(
            to_jsonable(
                {
                    "step": "inspect",
                    "path": str(checkpoint),
                    "file_sha256": manifest.identity.file_sha256,
                    "structure_fingerprint": manifest.identity.structure_fingerprint,
                    "tensor_count": manifest.totals.tensor_count,
                    "parameter_count": manifest.totals.parameter_count,
                    "total_bytes": manifest.totals.total_bytes,
                    "fc1.bias": {
                        "shape": list(bias.shape),
                        "dtype": bias_info.dtype,
                        "mean": bias_stats.mean,
                        "std": bias_stats.std,
                    },
                    "manifest_path": str(manifest_path),
                }
            ),
            indent=2,
        )
    )

    # ------------------------------------------------------- baseline forward
    x = torch.tensor([[0.1 * (index + 1) for index in range(8)]], dtype=torch.float32)
    baseline = _forward(_fresh_model(checkpoint), x)

    # ---------------------------------------------------------- create patch
    delta = np.full(bias.shape, 0.125, dtype=np.float64)
    params = AddParams(
        delta=PayloadRef(
            values=delta.tolist(),
            shape=[int(dimension) for dimension in delta.shape],
            dtype="float64",
        )
    )
    with open_checkpoint(checkpoint) as source:
        patch = create_patch(
            source,
            target="fc1.bias",
            selection=Selection(),
            params=params,
            budget=budget,
            notes="offline example: add 0.125 to every fc1 bias",
        )
    with open_checkpoint(checkpoint) as source:
        preview = preview_patch(source, patch, budget=budget)

    patch_path = workspace.save_patch(patch)
    print(
        json.dumps(
            to_jsonable(
                {
                    "step": "create_and_preview_patch",
                    "patch_id": patch.patch_id,
                    "preview_compatible": preview.compatible,
                    "checks": preview.checks,
                    "parameter_count": preview.parameter_count,
                    "changed_count": preview.effect.changed_count,
                    "max_abs_change": preview.effect.max_abs_change,
                    "storage_dtype": preview.precision.storage_dtype,
                    "cast_applied": preview.precision.cast_applied,
                    "patch_path": str(patch_path),
                }
            ),
            indent=2,
        )
    )

    # ------------------------------------------------------------ apply patch
    with open_checkpoint(checkpoint) as source:
        patchset = build_patchset(
            source, [patch], budget=budget, output_path=output
        )
        applied = apply_patchset(
            source,
            patchset,
            output,
            budget=budget,
            overwrite=args.force,
        )
    patchset_path = workspace.save_patchset(applied)
    assert applied.status == "applied"
    assert applied.output_identity is not None
    print(
        json.dumps(
            to_jsonable(
                {
                    "step": "apply_patch",
                    "patchset_id": applied.patchset_id,
                    "output_path": applied.output_path,
                    "output_file_sha256": applied.output_identity.file_sha256,
                    "patchset_path": str(patchset_path),
                }
            ),
            indent=2,
        )
    )

    # --------------------------------------------------- verify and compare
    patched_model = _fresh_model(output)
    patched = _forward(patched_model, x)
    comparison = compare_forward(baseline, patched)
    patched_bias = patched_model.fc1.bias.detach().cpu().numpy()
    expected_bias = bias.astype(np.float64) + 0.125
    exact_delta = bool(np.array_equal(patched_bias, expected_bias.astype(np.float32)))

    # --------------------------------------------------------------- rollback
    with open_checkpoint(checkpoint) as source:
        restored_identity = revert_patchset(source, applied)
    restored = _forward(_fresh_model(checkpoint), x)
    restored_comparison = compare_forward(baseline, restored)
    source_hash_after = sha256_file(checkpoint)

    result = {
        "step": "verify_and_revert",
        "patched_outputs_identical": comparison.outputs_identical,
        "patched_output_max_abs_delta": comparison.entries[0].diff.max_abs,
        "patched_activation_different_count": (
            comparison.activation_comparison.different_count
            if comparison.activation_comparison
            else None
        ),
        "patched_bias_equals_source_plus_delta": exact_delta,
        "restored_outputs_identical": restored_comparison.outputs_identical,
        "restored_structure_fingerprint": restored_identity.structure_fingerprint,
        "source_hash_unchanged": source_hash_before == source_hash_after,
        "source": str(checkpoint),
        "derived": str(output),
    }
    print(json.dumps(to_jsonable(result), indent=2))

    ok = (
        preview.compatible
        and comparison.outputs_identical is False
        and exact_delta
        and restored_comparison.outputs_identical
        and source_hash_before == source_hash_after
    )
    if not ok:
        print("example verification failed", file=sys.stderr)
        return 1
    print("\nOK: source untouched, derived checkpoint verified, rollback by source selection verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
