"""Targeted activation capture: positions, modes, budgets, and cleanup."""

from __future__ import annotations

import json

import pytest
import torch

from wsp.core.budget import Budget
from wsp.core.errors import InputError, ResourceLimitError
from wsp.execution.synthetic_lm import build_runtime
from wsp.investigation.collector import PositionedActivationCapture
from wsp.investigation.models import SCHEMA_CAPTURE
from wsp.investigation.points import discover_points
from wsp.tools.service import ToolService


def _record(service: ToolService, text: str = "the cat sat on the mat", max_new_tokens: int = 6) -> str:
    envelope = service.call(
        "record_trace",
        {
            "input_text": text,
            "params": {"max_new_tokens": max_new_tokens},
            "backend": {"kind": "synthetic", "synthetic": {"model": {}}},
        },
    )
    assert envelope["ok"], envelope
    return envelope["data"]["trace"]["trace_id"]


def _align(service: ToolService, trace_id: str) -> dict:
    envelope = service.call("align_trace_tokens", {"trace_id": trace_id})
    assert envelope["ok"], envelope
    return envelope["data"]


def _capture(service: ToolService, trace_id: str, **overrides):
    payload = {
        "trace_id": trace_id,
        "model": {"kind": "synthetic"},
        "modules": ["blocks.0", "norm"],
        "positions": [1, 2],
    }
    payload.update(overrides)
    return service.call("capture_trace_activations", payload)


def test_list_capture_points_classifies_modules():
    runtime = build_runtime()
    listing = discover_points(runtime.model)
    paths = {point.path: point for point in listing.points}

    assert "<root>" in paths
    assert paths["blocks.0.attn"].kind == "attention"
    assert paths["blocks.0.mlp"].kind == "mlp"
    assert paths["norm"].kind == "norm"
    assert paths["head"].kind == "other"
    assert all(point.parameter_count >= 0 for point in listing.points)
    assert listing.total_modules == len(paths)

    root_only = discover_points(runtime.model, include_submodules=False)
    assert [point.path for point in root_only.points] == ["<root>"]


def test_list_capture_points_through_the_service(service: ToolService):
    envelope = service.call("list_capture_points", {"model": {"kind": "synthetic"}})
    assert envelope["ok"], envelope
    data = envelope["data"]
    assert data["schema_version"] == "wsp.capture-points/1.0"
    assert data["source"]["kind"] == "synthetic"
    assert any(point["path"] == "blocks.1" for point in data["points"])
    assert "attention" in data["kind_labels"]


def test_empty_kind_filter_reports_canonical_labels():
    # Real friction from the autonomous agent trial: guessed kind labels
    # ("linear_attention") matched nothing silently. An empty filter result
    # must say so and name the canonical labels.
    runtime = build_runtime()
    listing = discover_points(runtime.model, kinds=["linear_attention"])
    assert listing.points == []
    joined = " ".join(listing.notes)
    assert "no capture points matched kinds linear_attention" in joined
    assert "canonical kind labels are" in joined
    assert "linear-attention" in joined

    root_only = discover_points(runtime.model, include_submodules=False, kinds=["attention"])
    assert root_only.points == []
    assert any("include_submodules=false lists only the model root" in note for note in root_only.notes)


def test_summary_capture_measures_requested_positions(service: ToolService):
    trace_id = _record(service)
    alignment = _align(service, trace_id)
    prompt_length = alignment["prompt_length"]
    positions = [prompt_length, prompt_length + 1]

    envelope = _capture(service, trace_id, positions=positions)
    assert envelope["ok"], envelope
    data = envelope["data"]

    assert data["schema_version"] == SCHEMA_CAPTURE
    assert data["mode"] == "summary"
    assert data["position_semantics"] == "consuming"
    assert data["positions"] == positions
    assert data["requested_tokens"] == positions
    assert data["sequence_length"] == alignment["full_length"]
    assert data["measurement_total"] == 4  # two modules x two positions
    assert data["replay_verification"]["numerical_agreement"] == "verified"
    assert data["replay_verification"]["generation_reproducibility"] == "not-claimed"
    assert data["unavailable"]["elementwise_comparison"]

    for measurement in data["measurements"]:
        assert measurement["position"] in positions
        assert measurement["retained"] is False
        assert measurement["value_artifact_id"] is None
        assert measurement["stats"]["mean"] is not None
        assert measurement["norm"] is not None
        assert measurement["stats"]["count"] == 32  # model_dim
        assert measurement["slice_shape"] == [1, 32]
    assert all("values were not retained" in note for note in data["measurements"][0]["notes"])

    capture_ref = service.workspace.artifacts.get(data["capture_artifact_id"])[0]
    assert capture_ref.kind == "trace_capture"
    assert capture_ref.metadata["capture_id"] == data["capture_id"]
    assert service.workspace.artifacts.get(data["replay_artifact_id"])[0].kind == "replay_result"


def test_predicting_semantics_measures_the_previous_position(service: ToolService):
    trace_id = _record(service)
    prompt_length = _align(service, trace_id)["prompt_length"]

    envelope = _capture(
        service,
        trace_id,
        positions=[prompt_length, prompt_length + 1],
        position_semantics="predicting",
    )
    assert envelope["ok"], envelope
    data = envelope["data"]

    assert data["position_semantics"] == "predicting"
    assert data["positions"] == [prompt_length - 1, prompt_length]
    assert data["requested_tokens"] == [prompt_length, prompt_length + 1]
    assert "predicting" in data["position_semantics_note"]


def test_predicting_semantics_drops_the_first_token_with_a_warning(service: ToolService):
    trace_id = _record(service)
    envelope = _capture(service, trace_id, positions=[0, 2], position_semantics="predicting")
    assert envelope["ok"], envelope
    data = envelope["data"]
    assert data["positions"] == [1]
    assert data["dropped_tokens"] == [0]
    assert any("no predicting position" in warning for warning in data["warnings"])


def test_selected_capture_retains_vectors_and_records_artifacts(service: ToolService):
    trace_id = _record(service)
    prompt_length = _align(service, trace_id)["prompt_length"]

    envelope = _capture(
        service,
        trace_id,
        positions=[prompt_length],
        mode="selected",
        modules=["blocks.0.attn"],
    )
    assert envelope["ok"], envelope
    data = envelope["data"]
    measurement = data["measurements"][0]

    assert measurement["retained"] is True
    assert measurement["retained_numel"] == 32
    assert measurement["value_artifact_id"]
    assert measurement["value_sha256"]
    assert data["resources"]["elements_retained"] == 32
    vector = service.workspace.artifacts.read_array(measurement["value_artifact_id"])
    assert vector.shape == (32,)
    ref = service.workspace.artifacts.get(measurement["value_artifact_id"])[0]
    assert ref.kind == "activation_vector"
    assert ref.metadata["position"] == prompt_length
    assert ref.metadata["source_trace"] == trace_id

    inspected = service.call(
        "inspect_activation_capture",
        {"artifact_id": data["capture_artifact_id"], "include_values": True},
    )
    assert inspected["ok"], inspected
    inspected_data = inspected["data"]
    assert inspected_data["measurement_total"] == 1
    assert inspected_data["integrity"]["missing_vectors"] == []
    assert len(inspected_data["measurements"][0]["values"]) == 32
    assert inspected_data["measurements"][0]["values"] == pytest.approx(vector.tolist())


def test_capture_accepts_a_resolved_span_artifact(service: ToolService):
    trace_id = _record(service, max_new_tokens=12)
    alignment = _align(service, trace_id)
    prompt_length = alignment["prompt_length"]

    span = service.call(
        "resolve_finding_span",
        {
            "trace_id": trace_id,
            "kind": "token_range",
            "role": "generation",
            "start": 0,
            "end": 3,
        },
    )
    assert span["ok"], span
    span_data = span["data"]
    assert span_data["full_token_indices"] == [prompt_length, prompt_length + 1, prompt_length + 2]

    envelope = _capture(
        service,
        trace_id,
        positions=None,
        span_artifact_id=span_data["span_artifact_id"],
        modules=["norm"],
    )
    assert envelope["ok"], envelope
    data = envelope["data"]
    assert data["positions"] == span_data["full_token_indices"]
    assert data["span_artifact_id"] == span_data["span_artifact_id"]


def test_capture_from_a_role_range(service: ToolService):
    trace_id = _record(service, max_new_tokens=12)
    prompt_length = _align(service, trace_id)["prompt_length"]

    envelope = _capture(
        service,
        trace_id,
        positions=None,
        role="generation",
        role_start=1,
        role_end=4,
        modules=["norm"],
    )
    assert envelope["ok"], envelope
    data = envelope["data"]
    assert data["positions"] == [prompt_length + 1, prompt_length + 2, prompt_length + 3]
    assert data["span_resolutions"][0]["status"] == "resolved"


def test_capture_accepts_a_stored_alignment(service: ToolService):
    trace_id = _record(service)
    alignment = _align(service, trace_id)
    prompt_length = alignment["prompt_length"]

    envelope = _capture(
        service,
        trace_id,
        positions=[prompt_length],
        alignment_artifact_id=alignment["alignment_artifact_id"],
    )
    assert envelope["ok"], envelope
    assert envelope["data"]["alignment_id"] == alignment["alignment_id"]


def test_include_inputs_records_module_inputs(service: ToolService):
    trace_id = _record(service)
    envelope = _capture(
        service,
        trace_id,
        positions=[1],
        modules=["blocks.0.attn"],
        include_inputs=True,
    )
    assert envelope["ok"], envelope
    roles = {measurement["tensor_role"] for measurement in envelope["data"]["measurements"]}
    assert roles == {"input", "output"}


def test_unknown_module_is_rejected_with_a_hint(service: ToolService):
    trace_id = _record(service)
    with pytest.raises(InputError) as failure:
        _capture(service, trace_id, modules=["blocks.99.attn"])
    assert "list_capture_points" in json.dumps(failure.value.details)


def test_capture_enforces_the_total_element_budget(service: ToolService):
    trace_id = _record(service)
    prompt_length = _align(service, trace_id)["prompt_length"]
    with pytest.raises(ResourceLimitError):
        _capture(
            service,
            trace_id,
            positions=[prompt_length],
            mode="selected",
            budget={"max_activation_elements_total": 8},
        )


def test_capture_position_count_is_bounded(service: ToolService):
    trace_id = _record(service)
    with pytest.raises(InputError):
        _capture(service, trace_id, positions=[1, 2], budget={"max_capture_positions": 1})


def test_capture_leaves_model_parameters_untouched():
    from wsp.investigation.capture import run_capture
    from wsp.replay.plan import prepare_plan
    from support.tiny_lm import make_trace

    runtime = build_runtime()
    trace = make_trace(runtime, "the cat sat on the mat", max_new_tokens=3)
    plan = prepare_plan(trace, budget=Budget())
    before = {name: value.clone() for name, value in runtime.model.state_dict().items()}

    outcome = run_capture(
        plan,
        runtime.replay_model(),
        alignment=None,
        requested_tokens=[plan.sequence.prompt_length],
        positions=[plan.sequence.prompt_length],
        modules=["blocks.0"],
        mode="selected",
    )
    assert outcome.record.measurements
    after = runtime.model.state_dict()
    assert set(before) == set(after)
    for name, value in before.items():
        assert torch.equal(value, after[name])


def test_hooks_are_removed_when_the_forward_pass_raises():
    class Exploding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = torch.nn.Linear(4, 4)

        def forward(self, values):
            self.inner(values)
            raise RuntimeError("boom")

    module = Exploding()
    collector = PositionedActivationCapture(
        module,
        ["inner"],
        [1],
        sequence_length=4,
        mode="selected",
    )
    assert not collector.installed
    with pytest.raises(RuntimeError):
        with collector:
            module(torch.zeros(1, 4, 4))
    assert not collector.installed
    for child in module.modules():
        assert len(getattr(child, "_forward_hooks", {})) == 0


def test_collector_rejects_positions_outside_the_sequence():
    runtime = build_runtime()
    with pytest.raises(InputError):
        PositionedActivationCapture(
            runtime.model, ["norm"], [64], sequence_length=8, mode="summary"
        )


def test_summary_capture_of_a_large_tensor_stays_bounded():
    runtime = build_runtime()

    class Squeeze(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(32, 512)

        def forward(self, input_ids):  # pragma: no cover - architecture shim
            hidden = torch.randn(1, input_ids.shape[1], 32)
            return self.linear(hidden)

    module = Squeeze()
    collector = PositionedActivationCapture(
        module,
        ["linear"],
        [0, 1],
        sequence_length=4,
        mode="selected",
        max_elements_per_tensor=64,
        budget=Budget(max_activation_elements_total=10_000),
    )
    with collector:
        module(torch.zeros(1, 4, dtype=torch.long))
    measurements = collector.ordered_measurements()
    assert len(measurements) == 2
    assert all(item.retained for item in measurements)
    assert all(item.sampled for item in measurements)
    assert all(item.retained_numel == 64 for item in measurements)
    assert all(item.stride and item.stride > 1 for item in measurements)
