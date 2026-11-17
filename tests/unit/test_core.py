from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from wsp.core.budget import Budget
from wsp.core.dtypes import normalize_dtype_name, torch_to_numpy
from wsp.core.errors import InputError, ResourceLimitError, UnsupportedError
from wsp.core.selection import AxisSlice, Selection
from wsp.inspection.factory import open_checkpoint
from wsp.inspection.manifest import build_manifest
from wsp.tensors.analysis import analyze_array
from wsp.tensors.diff import compare_arrays
from wsp.tensors.preview import preview_array
from wsp.tensors.stats import basic_stats, sampled_selection


def test_selection_full_and_rectangular_slice() -> None:
    full = Selection().resolve([4, 6])
    assert full.selected_shape == [4, 6]
    assert full.numel == 24
    assert full.is_full()
    assert full.to_index() == (slice(0, 4, 1), slice(0, 6, 1))

    selection = Selection(
        slices=[AxisSlice(axis=0, start=1, stop=4, step=2), AxisSlice(axis=1, start=2)]
    )
    resolved = selection.resolve([4, 6])
    assert resolved.selected_shape == [2, 4]
    assert resolved.to_index() == (slice(1, 4, 2), slice(2, 6, 1))
    assert resolved.numel == 8


@pytest.mark.parametrize(
    "selection, shape",
    [
        (Selection(slices=[AxisSlice(axis=2)]), [4, 4]),
        (Selection(slices=[AxisSlice(axis=0, start=2, stop=99)]), [4, 4]),
        (Selection(slices=[AxisSlice(axis=0, start=3, stop=1)]), [4, 4]),
        (
            Selection(slices=[AxisSlice(axis=0), AxisSlice(axis=0, start=1)]),
            [4, 4],
        ),
    ],
)
def test_invalid_selections_are_rejected(selection: Selection, shape: list[int]) -> None:
    with pytest.raises(InputError):
        selection.resolve(shape)


def test_stats_match_numpy_and_report_non_finite() -> None:
    values = np.array([[1.0, 2.0], [3.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    stats = basic_stats(values, storage_dtype="float32")
    finite = np.array([1.0, 2.0, 3.0])
    assert stats.count == 6
    assert stats.finite_count == 3
    assert stats.nan_count == 1
    assert stats.posinf_count == 1
    assert stats.neginf_count == 1
    assert stats.mean == pytest.approx(float(finite.mean()))
    assert stats.std == pytest.approx(float(finite.std(ddof=0)))
    assert stats.min == 1.0
    assert stats.max == 3.0
    assert stats.abs_max == 3.0


def test_stats_all_non_finite_has_null_scalars() -> None:
    stats = basic_stats(np.array([np.nan, np.inf], dtype=np.float64))
    assert stats.finite_count == 0
    assert stats.mean is None
    assert stats.std is None


def test_preview_prefix_and_sampling() -> None:
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    prefix = preview_array(values[:1, :5], max_elements=10, storage_dtype="float32")
    assert prefix.returned == 5
    assert prefix.truncated is False
    assert prefix.sampled is False
    assert prefix.values == [0.0, 1.0, 2.0, 3.0, 4.0]

    sampled = preview_array(values, max_elements=4, storage_dtype="float32")
    assert sampled.sampled is True
    assert sampled.returned <= 4
    assert sampled.stride and sampled.stride > 1


def test_diff_metrics_and_shape_mismatch() -> None:
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([1.0, 4.0, 3.0])
    diff = compare_arrays(left, right, atol=0, rtol=0)
    assert diff.compatible is True
    assert diff.allclose is False
    assert diff.max_abs == 2.0
    assert diff.mean_abs == pytest.approx(2.0 / 3.0)
    assert diff.non_finite_pairs == 0

    mismatch = compare_arrays(left, np.zeros((3, 1)))
    assert mismatch.compatible is False
    assert mismatch.allclose is False


def test_analysis_statistics_quantiles_histogram_norms_svd() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(12, 8)).astype(np.float32)
    analysis = analyze_array(
        matrix,
        storage_dtype="float32",
        operations=["stats", "quantiles", "histogram", "norms", "svd", "rank"],
        budget=Budget(),
        bins=8,
        top_k=4,
    )
    assert analysis.stats.count == matrix.size
    assert analysis.stats.mean == pytest.approx(float(matrix.astype(np.float64).mean()))
    assert len(analysis.operations["quantiles"]["values"]) == 9
    assert int(np.sum(analysis.operations["histogram"]["counts"])) == matrix.size
    fro = float(np.linalg.norm(matrix.astype(np.float64), ord="fro"))
    assert analysis.operations["norms"]["matrix"]["frobenius"] == pytest.approx(fro)
    singular = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
    assert analysis.operations["svd"]["singular_values"][0] == pytest.approx(float(singular[0]))
    assert analysis.operations["rank"]["rank"] == int(np.linalg.matrix_rank(matrix))


def test_analysis_rejects_unknown_operation() -> None:
    with pytest.raises(InputError):
        analyze_array(np.ones(4), operations=["interpret_neurons"], budget=Budget())


def test_sampled_selection_is_bounded_and_labeled() -> None:
    resolved = Selection().resolve([100, 100])
    sampled = sampled_selection(resolved, 100)
    assert sampled is not None
    sampled_resolved = sampled.resolve([100, 100])
    assert sampled_resolved.numel <= 100
    stats = basic_stats(np.arange(10, dtype=np.float32), sampled=True)
    assert stats.sampled is True


def test_safetensors_manifest_does_not_read_tensor_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mlp_checkpoint: Path
) -> None:
    from wsp.inspection.safetensors_source import SafeTensorsSource

    def explode(*args, **kwargs):
        raise AssertionError("tensor data was read")

    monkeypatch.setattr(SafeTensorsSource, "read_full", explode)
    monkeypatch.setattr(SafeTensorsSource, "read_tensor", explode)
    with open_checkpoint(mlp_checkpoint) as source:
        manifest = build_manifest(source)
    assert manifest.tensor_count == 4
    assert manifest.identity.parameter_count == 212
    assert manifest.totals.by_dtype["float32"] == 212
    assert manifest.tensors


def test_bf16_reads_are_widened_losslessly(tmp_path: Path) -> None:
    path = tmp_path / "bf16.safetensors"
    save_file(
        {"x": torch.tensor([1.5, -2.25, 0.0], dtype=torch.bfloat16)},
        str(path),
    )
    with open_checkpoint(path) as source:
        values = source.read_full("x", max_elements=10)
        assert values.dtype == np.float32
        assert values.tolist() == [1.5, -2.25, 0.0]


def test_pickle_and_unknown_checkpoints_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "model.bin"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(UnsupportedError):
        open_checkpoint(bad)


def test_directory_checkpoint_path_gets_actionable_error(tmp_path: Path) -> None:
    # Real failure seen during the MCP agent trial: passing a Hugging Face
    # snapshot directory instead of the weights file inside it produced a
    # misleading "pickle-based checkpoints are not supported" error.
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    with pytest.raises(InputError) as info:
        open_checkpoint(snapshot)
    assert "directory" in str(info.value)
    assert "model.safetensors" in str(info.value)


def test_float8_numeric_read_is_explicitly_unsupported(tmp_path: Path) -> None:
    from safetensors.numpy import save_file as save_numpy

    path = tmp_path / "fp8.safetensors"
    # safetensors accepts the dtype name in a hand-written header; use the
    # library's float8 torch dtype when available.
    try:
        save_file(
            {"x": torch.zeros(2, dtype=torch.float8_e4m3fn)}, str(path)
        )
    except Exception:
        pytest.skip("this torch build cannot write float8 SafeTensors")
    with open_checkpoint(path) as source:
        with pytest.raises(UnsupportedError):
            source.read_full("x", max_elements=10)


def test_dtype_normalization() -> None:
    assert normalize_dtype_name("torch.float32") == "float32"
    assert normalize_dtype_name("F32") == "float32"
    tensor = torch.tensor([1.0], dtype=torch.bfloat16)
    assert torch_to_numpy(tensor, decode_bfloat16=True).dtype == np.float32
