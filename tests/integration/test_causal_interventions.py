"""Tier 2: controlled causal experiments on the tiny transformer.

The test is deliberately end-to-end and offline.  It builds a trace, replays it,
captures an activation, creates identity/zero/donor interventions, runs paired
counterfactual measurements, runs a deterministic continuation, attaches the
experiment to a failure case, exports the case, and reloads it into a fresh
workspace.

The point is instrument correctness, not a scientific claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from support.tiny_lm import build_runtime
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace

MARKER = "CANARY-V13-private-prompt"
MODULE = "blocks.1"


def _payload(service: ToolService, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    envelope = service.call(operation, payload)
    assert envelope["ok"], json.dumps(envelope.get("error"), indent=2)
    return envelope["data"]


def _record(service: ToolService, text: str) -> dict[str, Any]:
    return _payload(
        service,
        "record_trace",
        {
            "input_text": text,
            "params": {"max_new_tokens": 6},
            "backend": {"kind": "synthetic", "synthetic": {"model": {}}},
        },
    )["trace"]


def _capture(service: ToolService, trace_id: str, positions: list[int]) -> dict[str, Any]:
    return _payload(
        service,
        "capture_trace_activations",
        {
            "trace_id": trace_id,
            "model": {"kind": "synthetic"},
            "modules": [MODULE],
            "mode": "selected",
            "positions": positions,
            "position_semantics": "consuming",
        },
    )


def _create(service: ToolService, **fields: Any) -> dict[str, Any]:
    return _payload(service, "create_activation_intervention", fields)


def _intervention_id(spec: dict[str, Any]) -> str:
    return spec["intervention_id"]


def test_causal_experiment_round_trip(service: ToolService, workspace: Workspace, tmp_path: Path):
    runtime = build_runtime()
    before = {
        name: parameter.detach().clone()
        for name, parameter in runtime.model.named_parameters()
    }

    observation = _record(service, f"{MARKER} the cat sat on the mat")
    control = _record(service, "the dog sat on the mat")

    plan = _payload(service, "prepare_replay", {"trace_id": observation["trace_id"]})
    replay = _payload(
        service,
        "verify_replay",
        {"plan_artifact_id": plan["plan_artifact_id"], "model": {"kind": "synthetic"}},
    )
    assert replay["numerical_agreement"] in ("verified", "partial")
    stored_plan = workspace.artifacts.read_json(plan["plan_artifact_id"])
    full_tokens = list(stored_plan["sequence"]["full_token_ids"])

    capture = _capture(service, observation["trace_id"], [1, 2, 3])
    assert capture["mode"] == "selected"
    assert capture["capture_artifact_id"]

    # ---------------------------------------------------------------- #
    # identity control: the mandatory negative control
    # ---------------------------------------------------------------- #
    identity_spec = _create(
        service,
        name="identity control",
        capture_artifact_id=capture["capture_artifact_id"],
        module=MODULE,
        positions=[1, 2, 3],
        operation="identity",
    )
    validation = _payload(
        service,
        "validate_activation_intervention",
        {
            "intervention_artifact_id": identity_spec["artifact_id"],
            "source": {"kind": "synthetic"},
            "token_ids": full_tokens,
        },
    )
    assert validation["valid"] is True
    assert validation["scope"]["estimated_elements"] > 0

    identity_pair = _payload(
        service,
        "run_paired_intervention",
        {
            "intervention_artifact_id": identity_spec["artifact_id"],
            "source": {"kind": "synthetic"},
            "token_ids": full_tokens,
            "trace_id": observation["trace_id"],
        },
    )
    assert identity_pair["identity_control"]["agreement"] == "exact"
    assert identity_pair["identity_control"]["passed"] is True
    assert identity_pair["validity"]["status"] == "valid"
    assert identity_pair["effect"]["logits"]["max_abs"] == 0.0

    # ---------------------------------------------------------------- #
    # zero ablation: a real, bounded change
    # ---------------------------------------------------------------- #
    zero_spec = _create(
        service,
        name="zero ablation",
        capture_artifact_id=capture["capture_artifact_id"],
        module=MODULE,
        positions=[1, 2, 3],
        operation="zero",
    )
    zero_pair = _payload(
        service,
        "run_paired_intervention",
        {
            "intervention_artifact_id": zero_spec["artifact_id"],
            "source": {"kind": "synthetic"},
            "token_ids": full_tokens,
            "measure_modules": ["blocks.2"],
            "logit_positions": [1, 2, 3],
        },
    )
    assert zero_pair["identity_control"]["agreement"] == "exact"
    assert zero_pair["validity"]["status"] == "valid"
    assert zero_pair["effect"]["logits"]["max_abs"] > 0
    assert zero_pair["effect"]["target_activation"]["after"]["l2_norm"] == 0.0
    assert zero_pair["summary"]["downstream_modules"] == ["blocks.2"]

    # ---------------------------------------------------------------- #
    # donor activation patch from a controlled execution
    # ---------------------------------------------------------------- #
    donor_capture = _capture(service, control["trace_id"], [1, 2, 3])
    patch_spec = _create(
        service,
        name="donor patch",
        capture_artifact_id=capture["capture_artifact_id"],
        module=MODULE,
        positions=[1, 2, 3],
        operation="patch",
        donor_capture_artifact_id=donor_capture["capture_artifact_id"],
        donor_trace_id=control["trace_id"],
        donor_positions=[1, 2, 3],
        donor_condition="healthy-control",
    )
    patch_pair = _payload(
        service,
        "run_paired_intervention",
        {
            "intervention_artifact_id": patch_spec["artifact_id"],
            "source": {"kind": "synthetic"},
            "token_ids": full_tokens,
        },
    )
    assert patch_pair["identity_control"]["agreement"] == "exact"
    assert patch_pair["validity"]["status"] == "valid"
    # The donor came from a different input, so the target must change.
    assert patch_pair["effect"]["target_activation"]["max_abs"] > 0

    # ---------------------------------------------------------------- #
    # continuation: first divergence under a deliberate patch
    # ---------------------------------------------------------------- #
    prefix = full_tokens[:4]
    width = int(runtime.model.norm.normalized_shape[0])
    alternating = [(-10.0 if index % 2 == 0 else 10.0) for index in range(width)]
    continuation_spec = _create(
        service,
        name="continuation patch",
        module="norm",
        positions=[len(prefix) - 1],
        operation="patch",
        donor_values=alternating,
        donor_shape=[1, width],
        donor_dtype="float32",
        allow_dtype_cast=True,
        model={"name": "tiny"},
        sequence_length=len(prefix),
    )
    continuation = _payload(
        service,
        "run_counterfactual_continuation",
        {
            "intervention_artifact_id": continuation_spec["artifact_id"],
            "source": {"kind": "synthetic"},
            "token_ids": prefix,
            "max_new_tokens": 4,
        },
    )
    assert continuation["comparison"]["identity_agreement"] == "exact"
    assert continuation["comparison"]["first_divergence"] is not None
    assert (
        continuation["comparison"]["longest_common_prefix"]
        == continuation["comparison"]["first_divergence"]
    )
    assert continuation["summary"]["interpretation"]

    # ---------------------------------------------------------------- #
    # case integration and export/reload
    # ---------------------------------------------------------------- #
    case = _payload(
        service,
        "create_failure_case",
        {
            "title": "causal instrument demonstration",
            "summary": "an offline correctness fixture",
            "observations": [observation["trace_id"]],
            "controls": [control["trace_id"]],
            "findings": [
                {
                    "detector_id": "manual.fixture",
                    "detector_version": "1",
                    "kind": "manual",
                    "severity": "low",
                    "explanation": "a fixture finding for case integration",
                }
            ],
        },
    )["case"]

    evidence = _payload(
        service,
        "attach_intervention_evidence",
        {
            "case_id": case["case_id"],
            "title": "paired intervention evidence",
            "observation_trace_id": observation["trace_id"],
            "control_trace_ids": [control["trace_id"]],
            "intervention_artifact_ids": [
                identity_spec["artifact_id"],
                zero_spec["artifact_id"],
                patch_spec["artifact_id"],
                continuation_spec["artifact_id"],
            ],
            "comparison_artifact_ids": [
                zero_pair["artifact_id"],
                patch_pair["artifact_id"],
            ],
            "continuation_artifact_ids": [continuation["artifact_id"]],
            "hypotheses": [
                {
                    "statement": "the selected block activation contributes to the next token",
                    "rationale": "observed under the tested conditions only",
                }
            ],
        },
    )
    assert evidence["interventions"]
    assert evidence["continuations"]

    private_path = tmp_path / "case-private.json"
    _payload(
        service,
        "export_failure_case",
        {"case_id": case["case_id"], "output_path": str(private_path)},
    )
    fresh_workspace = Workspace(tmp_path / "fresh", create=True, name="fresh")
    fresh = ToolService(workspace=fresh_workspace, raise_errors=True)
    imported = _payload(fresh, "import_failure_case", {"path": str(private_path)})
    assert imported["case"]["case_id"] == case["case_id"]
    loaded = _payload(fresh, "inspect_failure_case", {"case_id": case["case_id"]})
    assert loaded["case"]["investigations"]
    evidence_artifact = loaded["case"]["investigations"][0]["artifact_id"]
    fresh_workspace.artifacts.verify(evidence_artifact)

    # Redacted export must not retain the private prompt, donor values, or
    # generated tokens, and must still reload.
    redacted_path = tmp_path / "case-redacted.json"
    redacted = _payload(
        service,
        "export_failure_case",
        {
            "case_id": case["case_id"],
            "output_path": str(redacted_path),
            "policy": {"mode": "redacted", "sensitive_markers": [MARKER]},
        },
    )
    assert redacted["privacy"]["leak_scan"] == "clean"
    raw = redacted_path.read_bytes()
    assert MARKER.encode() not in raw
    assert b"the cat sat" not in raw
    assert b"the dog sat" not in raw
    fresh_redacted = ToolService(
        workspace=Workspace(tmp_path / "fresh-redacted", create=True),
        raise_errors=True,
    )
    reloaded = _payload(fresh_redacted, "import_failure_case", {"path": str(redacted_path)})
    assert reloaded["case"]["case_id"] == case["case_id"]
    assert reloaded["case"]["redaction"]["mode"] == "redacted"

    # ------------------------------------------------------------------ #
    # the model itself is unchanged
    # ------------------------------------------------------------------ #
    for name, parameter in runtime.model.named_parameters():
        assert torch.equal(parameter, before[name]), name
