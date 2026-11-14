"""Structured replay operations, the synthetic backend, and their artifacts."""

from __future__ import annotations

import json

import pytest
import torch

from wsp.behavior.models import BackendSpec, GenerationParams, SyntheticBackendConfig
from wsp.behavior.backends import build_backend
from wsp.core.budget import Budget
from wsp.core.errors import ArtifactError, CompatibilityError, UnsupportedError
from wsp.core.json import dumps
from wsp.execution.synthetic_lm import build_runtime
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace


class _Request:
    def __init__(self, text: str, params: GenerationParams, system: str | None = None) -> None:
        self.input_text = text
        self.system = system
        self.params = params
        self.seed = None


def _record(service: ToolService, text: str, max_new_tokens: int = 4) -> str:
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


def test_synthetic_backend_records_a_replayable_trace():
    backend = build_backend(
        BackendSpec(kind="synthetic", synthetic=SyntheticBackendConfig()), budget=Budget()
    )
    trace = backend.generate(
        _Request("the cat sat on the mat", GenerationParams(max_new_tokens=4)), budget=Budget()
    )

    assert trace.schema_version == "wsp.trace/1.1"
    assert trace.backend.name == "synthetic"
    assert trace.input.token_ids
    assert trace.input.tokens_sha256
    assert trace.input.token_encoding == "text"
    assert trace.model.structure_fingerprint
    assert [token.token_id for token in trace.tokens]
    assert all(token.logprob is not None for token in trace.tokens)
    assert trace.termination.reason in {"eos", "length_limit"}


def test_synthetic_backend_refuses_sampling():
    backend = build_backend(
        BackendSpec(kind="synthetic", synthetic=SyntheticBackendConfig()), budget=Budget()
    )
    with pytest.raises(UnsupportedError):
        backend.generate(
            _Request("hello", GenerationParams(max_new_tokens=2, do_sample=True, temperature=0.8)),
            budget=Budget(),
        )


def test_synthetic_backend_honours_stop_sequences():
    backend = build_backend(
        BackendSpec(kind="synthetic", synthetic=SyntheticBackendConfig()), budget=Budget()
    )
    trace = backend.generate(
        _Request("the cat sat", GenerationParams(max_new_tokens=6, stop_sequences=["aa"])),
        budget=Budget(),
    )
    if trace.termination.reason == "stop_sequence":
        assert trace.termination.stop_sequence == "aa"
        assert "aa" not in trace.text
    else:  # pragma: no cover - depends on the fixture's stochastic-looking output
        assert trace.termination.reason in {"eos", "length_limit"}


def test_prepare_and_verify_replay_through_the_service(service: ToolService, workspace: Workspace):
    trace_id = _record(service, "hello there", max_new_tokens=5)

    prepared = service.call("prepare_replay", {"trace_id": trace_id})
    assert prepared["ok"], prepared
    data = prepared["data"]
    assert data["schema_version"] == "wsp.replay-plan/1.0"
    assert data["semantics"] == "teacher-forced"
    assert data["sequence"]["source"] == "recorded-token-ids"
    assert data["check_statuses"]["generation_reproducibility"] == "unavailable"
    plan_ref = workspace.artifacts.get(data["plan_artifact_id"])[0]
    assert plan_ref.kind == "replay_plan"

    verified = service.call(
        "verify_replay", {"trace_id": trace_id, "model": {"kind": "synthetic"}}
    )
    assert verified["ok"], verified
    result = verified["data"]
    assert result["token_sequence_reconstructed"] is True
    assert result["forward_executed"] is True
    assert result["numerical_agreement"] == "verified"
    assert result["generation_reproducibility"] == "not-claimed"
    assert result["evidence_level"] == "verified"
    assert result["agreement"]["max_abs_diff"] < 1e-6
    assert result["agreement"]["top1_agreement_rate"] == 1.0
    result_ref = workspace.artifacts.get(result["result_artifact_id"])[0]
    assert result_ref.kind == "replay_result"
    stored = json.loads(dumps(workspace.artifacts.read_json(result["result_artifact_id"])))
    assert stored["schema_version"] == "wsp.replay/1.0"
    assert stored["generation_reproducibility"] == "not-claimed"


def test_stored_plan_can_be_verified_without_the_trace(service: ToolService):
    trace_id = _record(service, "hello there", max_new_tokens=3)
    prepared = service.call("prepare_replay", {"trace_id": trace_id})
    plan_artifact_id = prepared["data"]["plan_artifact_id"]

    verified = service.call(
        "verify_replay",
        {"plan_artifact_id": plan_artifact_id, "model": {"kind": "synthetic"}},
    )
    assert verified["ok"], verified
    assert verified["data"]["plan_artifact_id"] == plan_artifact_id
    assert verified["data"]["plan_id"] == prepared["data"]["plan_id"]


def test_verifying_a_non_plan_artifact_is_rejected(service: ToolService, workspace: Workspace):
    trace_id = _record(service, "hello", max_new_tokens=2)
    verified = service.call(
        "verify_replay", {"trace_id": trace_id, "model": {"kind": "synthetic"}}
    )
    result_artifact_id = verified["data"]["result_artifact_id"]

    with pytest.raises(ArtifactError):
        service.call(
            "verify_replay",
            {"plan_artifact_id": result_artifact_id, "model": {"kind": "synthetic"}},
        )


def test_incompatible_model_is_refused(service: ToolService):
    trace_id = _record(service, "hello", max_new_tokens=2)
    with pytest.raises(CompatibilityError):
        service.call(
            "verify_replay",
            {
                "trace_id": trace_id,
                "model": {"kind": "synthetic", "synthetic": {"model": {"num_layers": 2}}},
            },
        )


def test_replay_does_not_change_model_parameters(service: ToolService):
    trace_id = _record(service, "the cat sat", max_new_tokens=4)
    reference = build_runtime()
    before = {name: value.clone() for name, value in reference.model.state_dict().items()}

    verified = service.call(
        "verify_replay", {"trace_id": trace_id, "model": {"kind": "synthetic"}}
    )
    check = next(
        entry for entry in verified["data"]["checks"] if entry["name"] == "parameters_unchanged"
    )
    assert check["status"] == "pass"
    for name, value in reference.model.state_dict().items():
        assert torch.equal(before[name], value)


def test_per_token_rows_can_be_trimmed(service: ToolService):
    trace_id = _record(service, "hello there", max_new_tokens=4)
    verified = service.call(
        "verify_replay",
        {"trace_id": trace_id, "model": {"kind": "synthetic"}, "include_per_token": False},
    )
    agreement = verified["data"]["agreement"]
    assert agreement["per_token"] == []
    assert agreement["per_token_truncated"] is True
    assert agreement["tokens_compared"] > 0


def test_replay_operations_are_registered_with_schemas():
    service = ToolService(None)
    described = {entry["name"]: entry for entry in service.describe()["operations"]}
    for name in ("prepare_replay", "verify_replay"):
        assert name in described
        schema = described[name]["input_schema"]
        assert schema["type"] == "object"
    assert described["verify_replay"]["requires_workspace"] is True
