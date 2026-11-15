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


# --------------------------------------------------------------------------- #
# V1.4: temporal schedules, multi-site plans, and staged compatibility
# --------------------------------------------------------------------------- #

SOURCE = {"kind": "local_hf", "path": None, "compute_file_sha256": False}


@pytest.fixture(scope="module")
def temporal(service: ToolService, trace_id: str) -> dict:
    plan = service.call("prepare_replay", {"trace_id": trace_id})["data"]
    stored = service.workspace.artifacts.read_json(plan["plan_artifact_id"])
    full_tokens = list(stored["sequence"]["full_token_ids"])
    prompt = full_tokens[: min(6, len(full_tokens))]

    points = service.call(
        "list_capture_points",
        {"model": {"kind": "local_hf", "path": str(checkpoint)}, "limit": 500},
    )["data"]
    blocks = [point["path"] for point in points["points"] if point["kind"] == "block"]
    assert len(blocks) >= 3, blocks
    return {
        "prompt": prompt,
        "module_a": blocks[len(blocks) // 3],
        "module_b": blocks[(2 * len(blocks)) // 3],
        "source": {
            "kind": "local_hf",
            "path": str(checkpoint),
            "compute_file_sha256": False,
        },
    }


def _scheduled_spec(service: ToolService, module: str, prompt: list[int]) -> str:
    spec = service.call(
        "create_activation_intervention",
        {
            "name": f"temporal ablation {module}",
            "module": module,
            "positions": [0],
            "operation": "zero",
            "position_semantics": "consuming",
            "model": {"name": "local-hf"},
            "sequence_length": len(prompt),
        },
    )["data"]
    return spec["artifact_id"]


def _plan_entry(spec_id: str, *, order: int, step: int, label: str) -> dict:
    return {
        "intervention_artifact_id": spec_id,
        "label": label,
        "order": order,
        "schedule": {"kind": "generation-steps", "steps": [step]},
        "position_mode": "last",
        "on_step_call_indices": [0],
    }


def test_scheduled_uncached_intervention_fires_at_the_requested_step(
    service: ToolService, temporal: dict
):
    prompt = temporal["prompt"]
    source = temporal["source"]
    spec_id = _scheduled_spec(service, temporal["module_a"], prompt)
    created = service.call(
        "create_intervention_plan",
        {
            "name": "real-model scheduled ablation",
            "interventions": [_plan_entry(spec_id, order=0, step=2, label="A")],
            "prompt_token_ids": prompt,
            "execution_mode": "uncached-generation",
            "max_new_tokens": 3,
        },
    )["data"]
    assert created["valid"] is True, created["validation"]

    validation = service.call(
        "validate_intervention_plan",
        {
            "plan_artifact_id": created["artifact_id"],
            "source": source,
            "token_ids": prompt,
        },
    )["data"]
    assert validation["valid"] is True, validation["errors"]

    preview = service.call(
        "preview_intervention_plan",
        {
            "plan_artifact_id": created["artifact_id"],
            "source": source,
            "token_ids": prompt,
        },
    )["data"]
    assert [firing["generation_step"] for firing in preview["firings"]] == [2]
    assert preview["firings"][0]["module"] == temporal["module_a"]

    run = service.call(
        "run_temporal_intervention",
        {
            "plan_artifact_id": created["artifact_id"],
            "source": source,
            "token_ids": prompt,
            "identity": True,
        },
    )["data"]
    assert run["validity"]["status"] == "valid", run["validity"]
    assert run["identity_control"]["agreement"] == "exact"
    assert run["summary"]["events_fired"] == 1
    assert run["summary"]["events_missing"] == 0

    inspected = service.call(
        "inspect_intervention_events",
        {"run_artifact_id": run["artifact_id"], "include_events": True},
    )["data"]
    plan_log = next(log for log in inspected["logs"] if log["condition"] == "plan")
    assert plan_log["by_step"] == {"2": 1}
    event = plan_log["events"][0]
    assert event["generation_step"] == 2
    assert event["module"] == temporal["module_a"]
    assert event["after"]["l2_norm"] == 0.0

    checks = {check["name"]: check["status"] for check in run["validity"]["checks"]}
    assert checks["parameters_unchanged"] == "pass"
    assert checks["mutable_state_unchanged"] == "pass"


def test_multi_site_plan_fires_on_every_targeted_module(
    service: ToolService, temporal: dict
):
    prompt = temporal["prompt"]
    source = temporal["source"]
    spec_a = _scheduled_spec(service, temporal["module_a"], prompt)
    spec_b = _scheduled_spec(service, temporal["module_b"], prompt)
    created = service.call(
        "create_intervention_plan",
        {
            "name": "real-model multi-site plan",
            "interventions": [
                _plan_entry(spec_a, order=0, step=1, label="A"),
                _plan_entry(spec_b, order=1, step=1, label="B"),
            ],
            "prompt_token_ids": prompt,
            "execution_mode": "uncached-generation",
            "max_new_tokens": 2,
        },
    )["data"]
    assert created["valid"] is True, created["validation"]

    run = service.call(
        "run_temporal_intervention",
        {
            "plan_artifact_id": created["artifact_id"],
            "source": source,
            "token_ids": prompt,
        },
    )["data"]
    assert run["validity"]["status"] == "valid", run["validity"]
    assert run["summary"]["events_fired"] == 2
    log_id = run["event_log_artifact_ids"]["plan"]
    inspected = service.call(
        "inspect_intervention_events",
        {"event_log_artifact_id": log_id, "include_events": True},
    )["data"]
    events = inspected["logs"][0]["events"]
    assert {event["module"] for event in events} == {
        temporal["module_a"],
        temporal["module_b"],
    }
    assert {event["generation_step"] for event in events} == {1}


def test_staged_generation_support_is_reported_not_assumed(
    service: ToolService, temporal: dict
):
    """Staged support is architecture-specific and must fail closed.

    A hybrid checkpoint (for example Qwen3.5 with Gated DeltaNet layers) has
    generation state beyond a key/value cache.  This test accepts either a
    fully verified staged path or an explicit rejection; it never treats a
    skip, a warning, or an unverified run as staged support.
    """

    from wsp.core.errors import UnsupportedError
    from wsp.execution.staged import build_staged_decoder, classify_hf_architecture
    from wsp.replay.sources import ReplayModelSource, load_replay_source

    prompt = temporal["prompt"]
    source = temporal["source"]
    model = load_replay_source(
        ReplayModelSource(kind="local_hf", path=str(checkpoint), compute_file_sha256=False)
    )
    module = model.module if hasattr(model, "module") else model.model
    architecture, reason = classify_hf_architecture(module)

    prepared = service.call(
        "prepare_staged_generation",
        {"source": source, "token_ids": prompt, "max_new_tokens": 2},
    )
    if architecture == "hybrid-recurrent":
        assert prepared["ok"] is False, prepared
        assert prepared["error"]["code"] == "unsupported", prepared
        with pytest.raises(UnsupportedError):
            build_staged_decoder(module)
        return

    assert prepared["ok"] is True, prepared
    assert prepared["data"]["supported"] is True

    spec_id = _scheduled_spec(service, temporal["module_a"], prompt)
    plan = service.call(
        "create_intervention_plan",
        {
            "name": "real-model staged ablation",
            "interventions": [_plan_entry(spec_id, order=0, step=1, label="A")],
            "prompt_token_ids": prompt,
            "execution_mode": "staged-generation",
            "max_new_tokens": 2,
        },
    )["data"]
    assert plan["valid"] is True, plan["validation"]

    verification = service.call(
        "verify_staged_generation",
        {
            "plan_artifact_id": plan["artifact_id"],
            "source": source,
            "token_ids": prompt,
        },
    )["data"]
    assert verification["agreement"] in ("exact", "within_tolerance"), verification
    assert verification["token_sequence_match"] is True
    assert verification["identity_control_agreement"] == "exact"
    assert verification["architecture_class"] == architecture
    assert verification["isolation_verified"] is True

    staged = service.call(
        "run_staged_intervention",
        {
            "plan_artifact_id": plan["artifact_id"],
            "source": source,
            "token_ids": prompt,
        },
    )["data"]
    assert staged["verification"]["agreement"] in ("exact", "within_tolerance")
    assert staged["validity"]["status"] == "valid", staged["validity"]
    assert staged["performance"]["cache_state_bytes"] is not None
