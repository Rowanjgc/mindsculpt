#!/usr/bin/env python
"""WSP Workbench V1.3 demonstration: controlled causal intervention instruments.

This script is a correctness fixture, not a scientific claim.  It walks the
complete V1.3 chain on a small deterministic transformer built in process:

    observe -> replay -> capture -> identity control -> zero ablation
    -> donor patch -> paired measurement -> greedy continuation
    -> first divergence -> case evidence -> export -> reload -> redacted export

Every step is explicit.  The script checks that the identity control agrees with
the baseline exactly, that the deliberate interventions change downstream
numbers, that the model parameters never change, and that a redacted export
does not retain the private prompt or donor activations.

Usage::

    python examples/causal_investigation.py
    python examples/causal_investigation.py --workspace /tmp/wsp-causal-demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from wsp.behavior.cases import case_sha256
from wsp.behavior.models import FailureCase
from wsp.execution.synthetic_lm import build_runtime
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace

MARKER = "CANARY-V13-private-prompt"
MODULE = "blocks.1"


def _say(step: str, detail: str = "") -> None:
    print(f"[{step}] {detail}".rstrip())


def _data(envelope: dict[str, Any], what: str) -> dict[str, Any]:
    if not envelope.get("ok"):
        raise SystemExit(f"{what} failed: {json.dumps(envelope.get('error'), indent=2)}")
    return envelope["data"]


def _parameter_digest(module: torch.nn.Module) -> str:
    """Digest names, dtypes, shapes, and values, so mutation is detectable."""

    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _record(service: ToolService, text: str) -> dict[str, Any]:
    return _data(
        service.call(
            "record_trace",
            {
                "input_text": text,
                "params": {"max_new_tokens": 6},
                "backend": {"kind": "synthetic", "synthetic": {"model": {}}},
            },
        ),
        "record_trace",
    )["trace"]


def _capture(service: ToolService, trace_id: str, positions: list[int]) -> dict[str, Any]:
    return _data(
        service.call(
            "capture_trace_activations",
            {
                "trace_id": trace_id,
                "model": {"kind": "synthetic"},
                "modules": [MODULE],
                "mode": "selected",
                "positions": positions,
                "position_semantics": "consuming",
            },
        ),
        "capture_trace_activations",
    )


def _create(service: ToolService, **fields: Any) -> dict[str, Any]:
    return _data(service.call("create_activation_intervention", fields), "create")


def _paired(service: ToolService, spec: dict[str, Any], tokens: list[int], **extra: Any):
    return _data(
        service.call(
            "run_paired_intervention",
            {
                "intervention_artifact_id": spec["artifact_id"],
                "source": {"kind": "synthetic"},
                "token_ids": tokens,
                **extra,
            },
        ),
        "run_paired_intervention",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Parent directory for the demo workspace (default: a temporary directory).",
    )
    args = parser.parse_args(argv)

    if args.workspace is not None:
        root = args.workspace.expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            if not (root / "workspace.json").exists():
                raise SystemExit(
                    f"refusing to reuse non-empty directory {root}; "
                    "pass an empty or new directory"
                )
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(tempfile.mkdtemp(prefix="wsp-causal-demo-"))
    workspace = Workspace(root, create=True, name="causal-demo")
    service = ToolService(workspace=workspace, raise_errors=True)

    # 1 ---------------------------------------------------------------- #
    runtime = build_runtime()
    before = _parameter_digest(runtime.model)
    _say(
        "1",
        f"deterministic transformer ({sum(p.numel() for p in runtime.model.parameters())} parameters)",
    )

    # 2 ---------------------------------------------------------------- #
    observation = _record(service, f"{MARKER} the cat sat on the mat")
    control = _record(service, "the dog sat on the mat")
    _say("2", f"behavioral traces {observation['trace_id']} / {control['trace_id']}")

    # 3 ---------------------------------------------------------------- #
    plan = _data(service.call("prepare_replay", {"trace_id": observation["trace_id"]}), "plan")
    replay = _data(
        service.call(
            "verify_replay",
            {"plan_artifact_id": plan["plan_artifact_id"], "model": {"kind": "synthetic"}},
        ),
        "replay",
    )
    stored_plan = workspace.artifacts.read_json(plan["plan_artifact_id"])
    full_tokens = list(stored_plan["sequence"]["full_token_ids"])
    _say(
        "3",
        f"replay {replay['replay_id']} agreement={replay['numerical_agreement']} "
        f"tokens={len(full_tokens)}",
    )

    # 4 ---------------------------------------------------------------- #
    target = _capture(service, observation["trace_id"], [1, 2, 3])
    _say(
        "4",
        f"captured {target['capture_artifact_id']} module={MODULE} positions=[1, 2, 3] "
        f"mode={target['mode']}",
    )

    # 5 ---------------------------------------------------------------- #
    identity = _create(
        service,
        name="identity control",
        capture_artifact_id=target["capture_artifact_id"],
        module=MODULE,
        positions=[1, 2, 3],
        operation="identity",
    )
    validation = _data(
        service.call(
            "validate_activation_intervention",
            {
                "intervention_artifact_id": identity["artifact_id"],
                "source": {"kind": "synthetic"},
                "token_ids": full_tokens,
            },
        ),
        "validate",
    )
    assert validation["valid"] is True
    _say("5", f"identity intervention {identity['intervention_id']} validated")

    # 6 ---------------------------------------------------------------- #
    identity_pair = _paired(service, identity, full_tokens)
    assert identity_pair["identity_control"]["agreement"] == "exact"
    assert identity_pair["identity_control"]["passed"] is True
    assert identity_pair["validity"]["status"] == "valid"
    assert identity_pair["effect"]["logits"]["max_abs"] == 0.0
    _say(
        "6",
        "identity control exact; baseline and identity logits are bitwise equal "
        f"(max_abs={identity_pair['effect']['logits']['max_abs']:.1e})",
    )

    # 7 ---------------------------------------------------------------- #
    zero = _create(
        service,
        name="zero ablation",
        capture_artifact_id=target["capture_artifact_id"],
        module=MODULE,
        positions=[1, 2, 3],
        operation="zero",
    )
    _say("7", f"zero-ablation intervention {zero['intervention_id']}")

    # 8 ---------------------------------------------------------------- #
    zero_pair = _paired(
        service,
        zero,
        full_tokens,
        measure_modules=["blocks.2"],
        logit_positions=[1, 2, 3],
    )
    assert zero_pair["identity_control"]["agreement"] == "exact"
    assert zero_pair["validity"]["status"] == "valid"
    assert zero_pair["effect"]["target_activation"]["after"]["l2_norm"] == 0.0
    _say(
        "8",
        f"paired measurement: logits max_abs={zero_pair['effect']['logits']['max_abs']:.4f} "
        f"target_l2_after=0.0 downstream={zero_pair['summary']['downstream_modules']}",
    )

    # 9 ---------------------------------------------------------------- #
    donor = _capture(service, control["trace_id"], [1, 2, 3])
    _say("9", f"donor activation captured from the control input ({donor['capture_artifact_id']})")

    # 10 --------------------------------------------------------------- #
    patch = _create(
        service,
        name="donor patch",
        capture_artifact_id=target["capture_artifact_id"],
        module=MODULE,
        positions=[1, 2, 3],
        operation="patch",
        donor_capture_artifact_id=donor["capture_artifact_id"],
        donor_trace_id=control["trace_id"],
        donor_positions=[1, 2, 3],
        donor_condition="healthy-control",
    )
    patch_pair = _paired(service, patch, full_tokens)
    assert patch_pair["identity_control"]["agreement"] == "exact"
    assert patch_pair["validity"]["status"] == "valid"
    _say(
        "10",
        f"donor patch {patch['intervention_id']} applied; "
        f"target max_abs={patch_pair['effect']['target_activation']['max_abs']:.4f}",
    )

    # 11 --------------------------------------------------------------- #
    _say(
        "11",
        "downstream logits and selected token log-probabilities: "
        + json.dumps(
            {
                "logits_max_abs": round(zero_pair["effect"]["logits"]["max_abs"], 6),
                "logits_cosine": round(zero_pair["effect"]["logits"]["cosine"], 6),
                "token_deltas": [
                    {
                        "position": item["position"],
                        "baseline": item["baseline"],
                        "intervention": item["intervention"],
                        "delta": item["intervention_delta"],
                    }
                    for item in zero_pair["effect"]["token_logprobs"]
                ],
            }
        ),
    )

    # 12 --------------------------------------------------------------- #
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
    continuation = _data(
        service.call(
            "run_counterfactual_continuation",
            {
                "intervention_artifact_id": continuation_spec["artifact_id"],
                "source": {"kind": "synthetic"},
                "token_ids": prefix,
                "max_new_tokens": 4,
            },
        ),
        "continuation",
    )
    assert continuation["comparison"]["identity_agreement"] == "exact"
    _say(
        "12",
        "greedy continuations "
        + json.dumps(continuation["summary"]["generated"])
        + " (uncached full-prefix, intervention on the first prefix forward)",
    )

    # 13 --------------------------------------------------------------- #
    diff = continuation["comparison"]
    assert diff["first_divergence"] is not None
    assert diff["longest_common_prefix"] == diff["first_divergence"]
    _say(
        "13",
        f"first divergence at index {diff['first_divergence']} "
        f"(longest common prefix {diff['longest_common_prefix']}, "
        f"agreement rate {diff['token_agreement_rate']:.3f})",
    )

    # 14 --------------------------------------------------------------- #
    case = _data(
        service.call(
            "create_failure_case",
            {
                "title": "causal instrument demonstration",
                "summary": "an offline correctness fixture, not a scientific result",
                "observations": [observation["trace_id"]],
                "controls": [control["trace_id"]],
                "findings": [
                    {
                        "detector_id": "manual.fixture",
                        "detector_version": "1",
                        "kind": "manual",
                        "severity": "low",
                        "explanation": "fixture finding used to anchor the case",
                    }
                ],
            },
        ),
        "create case",
    )["case"]
    evidence = _data(
        service.call(
            "attach_intervention_evidence",
            {
                "case_id": case["case_id"],
                "title": "paired intervention evidence",
                "observation_trace_id": observation["trace_id"],
                "control_trace_ids": [control["trace_id"]],
                "intervention_artifact_ids": [
                    identity["artifact_id"],
                    zero["artifact_id"],
                    patch["artifact_id"],
                    continuation_spec["artifact_id"],
                ],
                "comparison_artifact_ids": [
                    zero_pair["artifact_id"],
                    patch_pair["artifact_id"],
                ],
                "continuation_artifact_ids": [continuation["artifact_id"]],
                "hypotheses": [
                    {
                        "statement": (
                            "the selected block activation contributes to the next-token "
                            "distribution under the tested conditions"
                        ),
                        "rationale": "observed under the tested conditions only",
                    }
                ],
            },
        ),
        "attach evidence",
    )
    _say(
        "14",
        f"case {case['case_id']} now references evidence {evidence['evidence_id']} "
        f"({len(evidence['interventions'])} interventions, "
        f"{len(evidence['continuations'])} continuation)",
    )

    # 15 --------------------------------------------------------------- #
    private_path = root / "case-private.json"
    exported = _data(
        service.call(
            "export_failure_case",
            {"case_id": case["case_id"], "output_path": str(private_path)},
        ),
        "export",
    )
    _say("15", f"private export {private_path.name} ({private_path.stat().st_size} bytes)")

    # 16 --------------------------------------------------------------- #
    fresh_workspace = Workspace(root / "fresh", create=True, name="fresh")
    fresh = ToolService(workspace=fresh_workspace, raise_errors=True)
    imported = _data(
        fresh.call("import_failure_case", {"path": str(private_path)}), "import"
    )
    _say("16", f"reloaded into a fresh workspace as case {imported['case']['case_id']}")

    # 17 --------------------------------------------------------------- #
    loaded = _data(
        fresh.call("inspect_failure_case", {"case_id": case["case_id"]}), "inspect case"
    )
    assert loaded["case"]["investigations"], "evidence link was lost in export"
    link = loaded["case"]["investigations"][0]
    fresh_workspace.artifacts.verify(link["artifact_id"])
    for artifact_id in [*link["capture_ids"], *link["comparison_ids"]]:
        if artifact_id:
            fresh_workspace.artifacts.verify(artifact_id)
    assert exported["case_sha256"] == case_sha256(
        FailureCase.model_validate(loaded["case"])
    )
    _say(
        "17",
        "checksums and references verified: evidence, captures, and comparisons all resolve",
    )

    # 18 --------------------------------------------------------------- #
    after = _parameter_digest(runtime.model)
    assert after == before, "model parameters changed during the experiment"
    _say("18", f"parameter digest unchanged ({before[:16]}...): no weights were edited")

    # 19 --------------------------------------------------------------- #
    redacted_path = root / "case-redacted.json"
    redacted = _data(
        service.call(
            "export_failure_case",
            {
                "case_id": case["case_id"],
                "output_path": str(redacted_path),
                "policy": {"mode": "redacted", "sensitive_markers": [MARKER]},
            },
        ),
        "redacted export",
    )
    raw = redacted_path.read_bytes()
    assert redacted["privacy"]["leak_scan"] == "clean"
    for needle in (MARKER, "the cat sat", "the dog sat"):
        assert needle.encode() not in raw, f"redacted export retained {needle!r}"
    redacted_fresh = ToolService(
        workspace=Workspace(root / "fresh-redacted", create=True), raise_errors=True
    )
    reloaded = _data(
        redacted_fresh.call("import_failure_case", {"path": str(redacted_path)}),
        "reload redacted",
    )
    assert reloaded["case"]["redaction"]["mode"] == "redacted"
    _say(
        "19",
        f"redacted export {redacted_path.name} ({redacted_path.stat().st_size} bytes) "
        f"leak_scan={redacted['privacy']['leak_scan']}; reloaded with redaction record",
    )

    print()
    print("artifacts:", root)
    print(
        "result: instruments agree. An observed intervention effect is evidence under "
        "the tested conditions, not proof of a causal circuit."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
