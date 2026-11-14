"""Tier 3: optional integration test against a real local checkpoint.

This test never downloads weights.  It runs only when ``WSP_REAL_MODEL`` names a
local checkpoint directory (and the optional ``transformers`` dependency is
installed)::

    WSP_REAL_MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<rev> \
        .venv/bin/python -m pytest tests/integration/test_real_checkpoint.py -q

Without that variable the test is reported as skipped, not passed.  A skip means
real-model compatibility is unverified for this machine; the synthetic hybrid
fixture in ``tests/unit/test_nnsight_backend.py`` is not a substitute.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wsp.core.budget import Budget
from wsp.core.errors import CompatibilityError, InputError, ResourceLimitError
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

transformers = pytest.importorskip("transformers")

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> ToolService:
    root = tmp_path_factory.mktemp("real-model")
    workspace = Workspace(root / "workspace", create=True, name="real-model")
    return ToolService(
        workspace=workspace,
        raise_errors=True,
        budget=Budget(max_replay_sequence_tokens=512),
    )


@pytest.fixture(scope="module")
def trace_id(service: ToolService) -> str:
    envelope = service.call(
        "record_trace",
        {
            "input_text": PROMPT,
            "params": {"max_new_tokens": 4},
            "backend": {
                "kind": "huggingface",
                "huggingface": {
                    "model_path": str(checkpoint),
                    "compute_file_sha256": False,
                },
            },
        },
    )
    assert envelope["ok"], envelope
    return envelope["data"]["trace"]["trace_id"]


def test_generation_records_replayable_tokens(service: ToolService, trace_id: str):
    trace = service.call("inspect_trace", {"trace_id": trace_id})["data"]

    assert trace["schema_version"].startswith("wsp.trace/1.1")
    assert trace["input"]["token_ids"], "the recorder must store prompt token ids"
    assert trace["input"]["token_encoding"] in ("chat_template", "text")
    assert trace["input"].get("tokens_sha256")
    assert trace["token_count"] >= 1


def test_teacher_forced_replay_and_capture(service: ToolService, trace_id: str):
    plan = service.call("prepare_replay", {"trace_id": trace_id})["data"]
    assert plan["sequence"]["source"] == "recorded-token-ids"

    replay = service.call(
        "verify_replay",
        {"plan_artifact_id": plan["plan_artifact_id"], "model": {"kind": "local_hf", "path": str(checkpoint)}},
    )["data"]
    assert replay["forward_executed"] is True
    assert replay["token_sequence_reconstructed"] is True
    assert replay["generation_reproducibility"] == "not-claimed"
    # Teacher-forced replay of recorded tokens on CPU should reproduce the
    # recorded log-probabilities; if it does not, the workbench must say so.
    assert replay["numerical_agreement"] in ("verified", "partial", "mismatch")
    if replay["numerical_agreement"] != "verified":
        assert replay["evidence_level"] == "exploratory"
        assert replay["exploratory_reason"]

    points = service.call(
        "list_capture_points",
        {"model": {"kind": "local_hf", "path": str(checkpoint)}, "limit": 500},
    )["data"]
    blocks = [point["path"] for point in points["points"] if point["kind"] == "block"]
    assert blocks, "a transformer checkpoint must expose block paths"
    module = blocks[len(blocks) // 2]

    capture = service.call(
        "capture_trace_activations",
        {
            "trace_id": trace_id,
            "model": {"kind": "local_hf", "path": str(checkpoint)},
            "modules": [module],
            "positions": [plan["sequence"]["prompt_length"] - 1],
            "position_semantics": "predicting",
            "mode": "summary",
        },
    )["data"]
    assert capture["measurements"], capture
    assert capture["replay_verification"]["forward_executed"] is True
    assert capture["mode"] == "summary"

    stored = service.call(
        "inspect_activation_capture", {"artifact_id": capture["capture_artifact_id"]}
    )["data"]
    assert stored["backend"], "the capture record must name its instrumentation backend"
    assert stored["position_semantics"] == "predicting"
    assert stored["mode"] == "summary"


def test_capture_is_bounded(service: ToolService, trace_id: str):
    with pytest.raises((InputError, CompatibilityError, ResourceLimitError)):
        service.call(
            "capture_trace_activations",
            {
                "trace_id": trace_id,
                "model": {"kind": "local_hf", "path": str(checkpoint)},
                "modules": ["does.not.exist"],
                "positions": [0],
            },
        )


@pytest.fixture(scope="module")
def control_trace_id(service: ToolService) -> str:
    envelope = service.call(
        "record_trace",
        {
            "input_text": "The capital of Germany is",
            "params": {"max_new_tokens": 4},
            "backend": {
                "kind": "huggingface",
                "huggingface": {
                    "model_path": str(checkpoint),
                    "compute_file_sha256": False,
                },
            },
        },
    )
    assert envelope["ok"], envelope
    return envelope["data"]["trace"]["trace_id"]


@pytest.fixture(scope="module")
def staged(service: ToolService, trace_id: str, control_trace_id: str) -> dict:
    plan = service.call("prepare_replay", {"trace_id": trace_id})["data"]
    stored = service.workspace.artifacts.read_json(plan["plan_artifact_id"])
    tokens = list(stored["sequence"]["full_token_ids"])

    points = service.call(
        "list_capture_points",
        {"model": {"kind": "local_hf", "path": str(checkpoint)}, "limit": 500},
    )["data"]
    blocks = [point["path"] for point in points["points"] if point["kind"] == "block"]
    module = blocks[len(blocks) // 2]

    positions = sorted({max(len(tokens) - 2, 0), len(tokens) - 1})
    capture = service.call(
        "capture_trace_activations",
        {
            "trace_id": trace_id,
            "model": {"kind": "local_hf", "path": str(checkpoint)},
            "modules": [module],
            "positions": positions,
            "position_semantics": "consuming",
            "mode": "selected",
        },
    )["data"]
    donor = service.call(
        "capture_trace_activations",
        {
            "trace_id": control_trace_id,
            "model": {"kind": "local_hf", "path": str(checkpoint)},
            "modules": [module],
            "positions": positions,
            "position_semantics": "consuming",
            "mode": "selected",
        },
    )["data"]
    return {
        "tokens": tokens,
        "module": module,
        "positions": positions,
        "capture": capture,
        "donor": donor,
        "plan": plan,
    }


def test_identity_control_on_real_checkpoint(service: ToolService, staged: dict):
    spec = service.call(
        "create_activation_intervention",
        {
            "name": "real-model identity control",
            "capture_artifact_id": staged["capture"]["capture_artifact_id"],
            "module": staged["module"],
            "positions": staged["positions"],
            "operation": "identity",
        },
    )["data"]

    validation = service.call(
        "validate_activation_intervention",
        {
            "intervention_artifact_id": spec["artifact_id"],
            "source": {"kind": "local_hf", "path": str(checkpoint)},
            "token_ids": staged["tokens"],
        },
    )["data"]
    assert validation["valid"] is True, validation

    paired = service.call(
        "run_paired_intervention",
        {
            "intervention_artifact_id": spec["artifact_id"],
            "source": {"kind": "local_hf", "path": str(checkpoint)},
            "token_ids": staged["tokens"],
        },
    )["data"]
    assert paired["identity_control"]["passed"] is True
    assert paired["validity"]["status"] in ("valid", "limited")
    assert paired["effect"]["logits"]["max_abs"] <= 1e-5


def test_bounded_zero_ablation_and_donor_patch(service: ToolService, staged: dict):
    zero = service.call(
        "create_activation_intervention",
        {
            "name": "real-model zero ablation",
            "capture_artifact_id": staged["capture"]["capture_artifact_id"],
            "module": staged["module"],
            "positions": staged["positions"],
            "operation": "zero",
        },
    )["data"]
    zero_pair = service.call(
        "run_paired_intervention",
        {
            "intervention_artifact_id": zero["artifact_id"],
            "source": {"kind": "local_hf", "path": str(checkpoint)},
            "token_ids": staged["tokens"],
            "logit_positions": [staged["tokens"].__len__() - 1],
        },
    )["data"]
    assert zero_pair["identity_control"]["passed"] is True
    assert zero_pair["validity"]["status"] in ("valid", "limited")
    assert zero_pair["effect"]["target_activation"]["after"]["l2_norm"] == 0.0

    patch = service.call(
        "create_activation_intervention",
        {
            "name": "real-model donor patch",
            "capture_artifact_id": staged["capture"]["capture_artifact_id"],
            "module": staged["module"],
            "positions": staged["positions"],
            "operation": "patch",
            "donor_capture_artifact_id": staged["donor"]["capture_artifact_id"],
            "donor_positions": staged["positions"],
            "donor_condition": "alternative-input",
        },
    )["data"]
    patch_pair = service.call(
        "run_paired_intervention",
        {
            "intervention_artifact_id": patch["artifact_id"],
            "source": {"kind": "local_hf", "path": str(checkpoint)},
            "token_ids": staged["tokens"],
        },
    )["data"]
    assert patch_pair["identity_control"]["passed"] is True
    assert patch_pair["effect"]["target_activation"]["max_abs"] > 0


def test_parameters_unchanged_and_continuation_is_bounded(
    service: ToolService, staged: dict
):
    from wsp.replay.sources import ReplayModelSource, load_replay_source

    model = load_replay_source(
        ReplayModelSource(kind="local_hf", path=str(checkpoint), compute_file_sha256=False)
    )
    module = model.module if hasattr(model, "module") else model.model
    before = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }

    prefix_length = min(4, len(staged["tokens"]))
    prefix = staged["tokens"][:prefix_length]
    spec = service.call(
        "create_activation_intervention",
        {
            "name": "real-model continuation ablation",
            "module": staged["module"],
            "positions": [prefix_length - 1],
            "operation": "zero",
            "model": {"name": "local-hf"},
            "sequence_length": prefix_length,
        },
    )["data"]
    continuation = service.call(
        "run_counterfactual_continuation",
        {
            "intervention_artifact_id": spec["artifact_id"],
            "source": {"kind": "local_hf", "path": str(checkpoint)},
            "token_ids": prefix,
            "max_new_tokens": 2,
        },
    )["data"]
    assert continuation["comparison"]["identity_agreement"] == "exact"
    assert continuation["validity"]["status"] in ("valid", "limited")
    baseline_condition = next(
        item for item in continuation["conditions"] if item["condition"] == "baseline"
    )
    assert baseline_condition["termination_reason"] in (
        "eos",
        "max_new_tokens",
        "max_sequence_length",
    )

    after = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }
    assert before.keys() == after.keys()
    for name in before:
        assert before[name].equal(after[name]), name
