from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from wsp.activations.capture import ActivationCapture
from wsp.activations.compare import compare_activations
from wsp.core.budget import Budget
from wsp.core.errors import InputError, ResourceLimitError
from wsp.execution.builtin_models import TinyMLP, TinyTransformer
from wsp.execution.compare import compare_forward
from wsp.execution.forward import run_forward


def test_capture_records_inputs_outputs_and_removes_hooks(input_x) -> None:
    model = TinyMLP()
    x = torch.tensor(input_x)
    capture = ActivationCapture(
        model, ["fc1", "act", "fc2"], record_inputs=True, include_values=True
    )
    with capture:
        output = model(x)
        assert output.shape == (1, 4)
    assert len(capture.records) == 6
    kinds = {(record.module, record.kind) for record in capture.records}
    assert ("fc1", "input[0]") in kinds
    assert ("fc1", "output") in kinds
    assert ("fc2", "output") in kinds
    assert all(record.values is not None for record in capture.records)
    for module in (model.fc1, model.act, model.fc2):
        assert len(module._forward_hooks) == 0


def test_hooks_are_removed_when_forward_raises() -> None:
    class Broken(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(3, 3)

        def forward(self, x):
            self.fc(x)
            raise RuntimeError("intentional failure")

    model = Broken()
    capture = ActivationCapture(model, ["fc"], include_values=False)
    with pytest.raises(RuntimeError):
        with capture:
            model(torch.ones(2, 3))
    assert len(model.fc._forward_hooks) == 0


def test_summary_only_capture_does_not_consume_value_budget(input_x) -> None:
    model = TinyMLP()
    capture = ActivationCapture(
        model,
        ["fc1", "fc2"],
        include_values=False,
        max_elements_per_tensor=1,
        max_elements_total=1,
    )
    with capture:
        model(torch.tensor(input_x))
    assert len(capture.records) == 2
    assert all(record.values is None for record in capture.records)
    summaries = capture.summaries(include_values=True)
    assert all(summary.retained is False for summary in summaries)
    assert all(summary.values is None for summary in summaries)


def test_capture_enforces_total_budget() -> None:
    model = TinyMLP()
    capture = ActivationCapture(
        model,
        ["fc1", "fc2"],
        include_values=True,
        max_elements_per_tensor=1000,
        max_elements_total=10,
    )
    with pytest.raises(ResourceLimitError):
        with capture:
            model(torch.ones(1, 8))
    assert len(model.fc1._forward_hooks) == 0


def test_transformer_token_position_is_explicit() -> None:
    model = TinyTransformer(vocab_size=32, model_dim=16, num_heads=4, num_layers=1)
    ids = torch.tensor([[1, 2, 3, 4]])
    capture = ActivationCapture(model, ["blocks.0"], token_position=-1, include_values=True)
    with capture:
        model(ids)
    record = next(record for record in capture.records if record.kind == "output")
    assert record.token_position == -1
    assert record.shape == [1, 4, 16]
    assert record.values.shape == (16,)


def test_forward_runs_are_deterministic_and_capture_activations(input_x) -> None:
    model = TinyMLP()
    capture = ActivationCapture(model, ["fc1", "fc2"], include_values=True)
    left = run_forward(
        model,
        inputs={"x": torch.tensor(input_x)},
        seed=7,
        capture=capture,
        budget=Budget(),
    )
    capture = ActivationCapture(model, ["fc1", "fc2"], include_values=True)
    right = run_forward(
        model,
        inputs={"x": torch.tensor(input_x)},
        seed=7,
        capture=capture,
        budget=Budget(),
    )
    comparison = compare_forward(left, right)
    assert comparison.outputs_identical is True
    assert comparison.activation_comparison is not None
    assert comparison.activation_comparison.identical_count == 2
    assert left.output_values["output"].shape == (4,)


def test_changed_weights_produce_a_measured_output_difference(input_x) -> None:
    model = TinyMLP()
    left = run_forward(model, inputs={"x": torch.tensor(input_x)}, seed=3, budget=Budget())
    with torch.no_grad():
        model.fc2.weight.add_(0.25)
    right = run_forward(model, inputs={"x": torch.tensor(input_x)}, seed=3, budget=Budget())
    comparison = compare_forward(left, right)
    assert comparison.outputs_identical is False
    assert comparison.entries[0].diff.max_abs > 0
    assert comparison.entries[0].diff.compatible is True


def test_model_training_mode_is_restored(input_x) -> None:
    model = TinyMLP()
    model.train()
    result = run_forward(model, inputs={"x": torch.tensor(input_x)}, budget=Budget())
    assert model.training is True
    assert result.result.training_restored is True
    assert result.result.inference_mode is True


def test_capture_comparison_of_different_inputs(input_x) -> None:
    model = TinyMLP()
    first = ActivationCapture(model, ["fc2"], include_values=True)
    with first:
        model(torch.tensor(input_x))
    second = ActivationCapture(model, ["fc2"], include_values=True)
    with second:
        model(torch.tensor(input_x) * 2)
    comparison = compare_activations(first, second)
    assert comparison.identical_count == 0
    assert comparison.different_count == 1
    assert comparison.entries[0].diff is not None
    assert comparison.entries[0].diff.max_abs > 0


def test_unknown_module_is_rejected() -> None:
    with pytest.raises(InputError):
        ActivationCapture(TinyMLP(), ["not_a_module"])
