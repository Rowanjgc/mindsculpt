from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import save_file

from wsp.core.budget import Budget
from wsp.core.errors import CompatibilityError, InputError, NotFoundError, SafetyError
from wsp.core.selection import AxisSlice, Selection
from wsp.inspection.factory import open_checkpoint
from wsp.patches.engine import (
    apply_patchset,
    build_patchset,
    create_patch,
    export_checkpoint,
    preview_patch,
    revert_patchset,
)
from wsp.patches.operations import compute_effect
from wsp.patches.safetensors_export import copy_checkpoint
from wsp.patches.spec import (
    AddParams,
    PayloadRef,
    ProjectionParams,
    RankOneParams,
    ReplaceParams,
    ScaleParams,
)


def _payload(values) -> PayloadRef:
    return PayloadRef(values=values)


@pytest.mark.parametrize(
    "params, expected_transform",
    [
        (lambda rows: ReplaceParams(values=_payload(np.ones((2, 8)).tolist())), lambda base: np.ones((2, 8))),
        (lambda rows: AddParams(delta=_payload(np.full((2, 8), 0.25).tolist())), lambda base: base + 0.25),
        (lambda rows: ScaleParams(factor=2.0), lambda base: base * 2.0),
        (
            lambda rows: RankOneParams(
                u=_payload(np.ones(2).tolist()),
                v=_payload(np.full(8, 0.5).tolist()),
                scale=0.5,
            ),
            lambda base: base + 0.25 * np.ones((2, 8)),
        ),
        (
            lambda rows: ProjectionParams(
                direction=_payload(np.ones(8).tolist()), axis=1, mode="remove"
            ),
            lambda base: base - np.outer(base @ (np.ones(8) / np.sqrt(8)), np.ones(8) / np.sqrt(8)),
        ),
    ],
)
def test_patch_operations_match_expected_math(mlp_checkpoint: Path, tmp_path: Path, params, expected_transform) -> None:
    selection = Selection(slices=[AxisSlice(axis=0, start=0, stop=2)])
    with open_checkpoint(mlp_checkpoint) as source:
        base = source.read_tensor("fc1.weight", selection, max_elements=1000).astype(np.float64)
        patch = create_patch(
            source,
            target="fc1.weight",
            selection=selection,
            params=params(2),
            budget=Budget(),
        )
        applied = apply_patchset(
            source,
            build_patchset(source, [patch], budget=Budget()),
            tmp_path / "operation.safetensors",
            budget=Budget(),
        )
    expected = expected_transform(base).astype(np.float32).astype(np.float64)
    assert patch.after.finite_count == 16
    assert patch.parameter_count == 16
    assert patch.precision.storage_dtype == "float32"
    with open_checkpoint(applied.output_path) as patched:
        actual = patched.read_tensor("fc1.weight", selection, max_elements=1000).astype(np.float64)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert patch.effect.changed_count == int(np.count_nonzero(expected != base))


def test_create_preview_apply_and_revert_workflow(mlp_checkpoint: Path, tmp_path: Path) -> None:
    output = tmp_path / "patched.safetensors"
    selection = Selection(slices=[AxisSlice(axis=0, start=0, stop=2)])
    with open_checkpoint(mlp_checkpoint) as source:
        base = source.read_full("fc1.weight", max_elements=1000)
        patch = create_patch(
            source,
            target="fc1.weight",
            selection=selection,
            params=AddParams(delta=_payload(np.full((2, 8), 0.5).tolist())),
            budget=Budget(),
        )
        preview = preview_patch(source, patch, budget=Budget())
        assert preview.compatible
        assert preview.effect.changed_count == 16
        patchset = build_patchset(source, [patch], budget=Budget())
        applied = apply_patchset(source, patchset, output, budget=Budget())
        source_identity = source.identity()
        restored = revert_patchset(source, applied)
    assert restored.structure_fingerprint == source_identity.structure_fingerprint
    assert applied.status == "applied"
    assert applied.output_identity is not None
    assert applied.output_identity.file_sha256

    with open_checkpoint(output) as patched:
        values = patched.read_full("fc1.weight", max_elements=1000)
    expected = base.copy()
    expected[0:2, :] = (base[0:2, :].astype(np.float64) + 0.5).astype(np.float32)
    np.testing.assert_array_equal(values, expected)
    # The original file is byte-identical to before the patch.
    with open_checkpoint(mlp_checkpoint) as original:
        np.testing.assert_array_equal(
            original.read_full("fc1.weight", max_elements=1000), base
        )


def test_reapplying_same_patch_is_deterministic(mlp_checkpoint: Path, tmp_path: Path) -> None:
    selection = Selection(slices=[AxisSlice(axis=0, start=0, stop=1)])
    outputs = []
    with open_checkpoint(mlp_checkpoint) as source:
        patch = create_patch(
            source,
            target="fc2.bias",
            selection=Selection(),
            params=ScaleParams(factor=1.5),
            budget=Budget(),
        )
        for index in range(2):
            patchset = build_patchset(source, [patch], budget=Budget())
            target = tmp_path / f"out-{index}.safetensors"
            applied = apply_patchset(source, patchset, target, budget=Budget())
            outputs.append(applied.output_identity.file_sha256)
    assert outputs[0] == outputs[1]


def test_chained_patches_on_distinct_targets(mlp_checkpoint: Path, tmp_path: Path) -> None:
    output = tmp_path / "chained.safetensors"
    with open_checkpoint(mlp_checkpoint) as source:
        first = create_patch(
            source,
            target="fc1.bias",
            selection=Selection(),
            params=AddParams(delta=_payload(np.full(16, 0.125).tolist())),
            budget=Budget(),
        )
        second = create_patch(
            source,
            target="fc2.bias",
            selection=Selection(),
            params=ScaleParams(factor=0.5),
            budget=Budget(),
        )
        patchset = build_patchset(source, [first, second], budget=Budget())
        applied = apply_patchset(source, patchset, output, budget=Budget())
    assert applied.patch_ids == [first.patch_id, second.patch_id]
    with open_checkpoint(output) as patched, open_checkpoint(mlp_checkpoint) as original:
        assert not np.allclose(
            patched.read_full("fc1.bias", max_elements=100),
            original.read_full("fc1.bias", max_elements=100),
        )
        assert not np.allclose(
            patched.read_full("fc2.bias", max_elements=100),
            original.read_full("fc2.bias", max_elements=100),
        )


def test_wrong_fingerprint_is_rejected(mlp_checkpoint: Path, tmp_path: Path) -> None:
    with open_checkpoint(mlp_checkpoint) as source:
        patch = create_patch(
            source,
            target="fc1.weight",
            selection=Selection(),
            params=AddParams(delta=_payload(np.full((16, 8), 0.01).tolist())),
            budget=Budget(),
        )
    torch.manual_seed(9999)
    from wsp.execution.builtin_models import TinyMLP

    other = TinyMLP()
    other_path = tmp_path / "other.safetensors"
    save_file({k: v.contiguous() for k, v in other.state_dict().items()}, str(other_path))
    with open_checkpoint(other_path) as source:
        with pytest.raises(CompatibilityError):
            preview_patch(source, patch, budget=Budget())


def test_missing_target_and_bad_payloads_are_rejected(mlp_checkpoint: Path) -> None:
    with open_checkpoint(mlp_checkpoint) as source:
        with pytest.raises(NotFoundError):
            create_patch(
                source,
                target="missing.weight",
                selection=Selection(),
                params=ScaleParams(factor=1.0),
                budget=Budget(),
            )
        with pytest.raises(InputError):
            create_patch(
                source,
                target="fc1.weight",
                selection=Selection(),
                params=AddParams(delta=_payload(np.zeros((1, 1)).tolist())),
                budget=Budget(),
            )
        with pytest.raises(InputError):
            create_patch(
                source,
                target="fc1.weight",
                selection=Selection(),
                params=AddParams(delta=_payload(np.full((16, 8), np.nan).tolist())),
                budget=Budget(),
            )
        with pytest.raises(InputError):
            create_patch(
                source,
                target="fc1.weight",
                selection=Selection(),
                params=ProjectionParams(
                    direction=_payload(np.zeros(8).tolist()), axis=1
                ),
                budget=Budget(),
            )
        with pytest.raises(ValidationError):
            ScaleParams(factor=float("inf"))


def test_arithmetic_on_integer_tensor_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ints.safetensors"
    save_file({"x": torch.arange(6, dtype=torch.int64).reshape(2, 3)}, str(path))
    with open_checkpoint(path) as source:
        with pytest.raises(InputError):
            create_patch(
                source,
                target="x",
                selection=Selection(),
                params=AddParams(delta=_payload(np.ones((2, 3)).tolist())),
                budget=Budget(),
            )
        patch = create_patch(
            source,
            target="x",
            selection=Selection(),
            params=ReplaceParams(values=_payload([[1, 2, 3], [4, 5, 6]])),
            budget=Budget(),
        )
        assert patch.operation == "replace"


def test_bf16_patch_reports_casting(tmp_path: Path) -> None:
    path = tmp_path / "bf16.safetensors"
    save_file(
        {
            "x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16),
            "keep": torch.tensor([9.5], dtype=torch.bfloat16),
        },
        str(path),
    )
    output = tmp_path / "bf16-patched.safetensors"
    with open_checkpoint(path) as source:
        patch = create_patch(
            source,
            target="x",
            selection=Selection(),
            params=AddParams(delta=_payload([0.1, 0.2, 0.3])),
            budget=Budget(),
        )
        assert patch.precision.cast_applied is True
        assert patch.precision.storage_dtype == "bfloat16"
        applied = apply_patchset(
            source,
            build_patchset(source, [patch], budget=Budget()),
            output,
            budget=Budget(),
        )
    with open_checkpoint(output) as patched:
        assert patched.tensor_info("x").dtype == "bfloat16"
        assert patched.tensor_info("keep").dtype == "bfloat16"
        kept = patched.read_full("keep", max_elements=10)
    assert kept.tolist() == [9.5]


def test_export_copy_is_streamed_and_refuses_overwrite(mlp_checkpoint: Path, tmp_path: Path) -> None:
    output = tmp_path / "copy.safetensors"
    identity = export_checkpoint(mlp_checkpoint, output)
    assert identity.tensor_count == 4
    with open_checkpoint(mlp_checkpoint) as source, open_checkpoint(output) as copied:
        assert source.identity().structure_fingerprint == copied.identity().structure_fingerprint
    with pytest.raises(SafetyError):
        copy_checkpoint(mlp_checkpoint, output, overwrite=False)
    with pytest.raises(SafetyError):
        copy_checkpoint(mlp_checkpoint, mlp_checkpoint, overwrite=True)


def test_effect_metrics_are_finite_for_nan_free_arrays() -> None:
    before = np.array([1.0, 2.0, 3.0])
    after = np.array([1.0, 2.5, 3.0])
    effect = compute_effect(before, after)
    assert effect["changed_count"] == 1
    assert effect["unchanged_count"] == 2
    assert effect["max_abs_change"] == pytest.approx(0.5)
