#!/usr/bin/env python
"""WSP Workbench V1.4 demonstration: temporal and compositional instruments.

This script is a correctness fixture, not a scientific claim.  It walks the
complete V1.4 chain on a small deterministic transformer built in process:

    observe -> choose a generation step -> scheduled plan -> validate -> preview
    -> identity plan -> control agreement -> scheduled ablation -> event log
    -> downstream comparison -> second site -> A / B / A+B
    -> interaction contrast -> staged prefill -> staged verification
    -> staged intervention -> evidence -> case -> export -> reload
    -> redacted export -> state preservation

Every step is explicit, and every claim in the output is a measurement of the
instrument itself.  The script does not claim to have discovered a mechanism.

Usage::

    python examples/temporal_intervention_walkthrough.py
    python examples/temporal_intervention_walkthrough.py --workspace /tmp/wsp-temporal-demo
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

from wsp.execution.synthetic_lm import build_runtime
from wsp.interventions.plan import InterventionPlan, identity_control_plan
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace

MARKER = "CANARY-V14-private-prompt"
PROMPT = "abc"
MODULE_A = "blocks.1"
MODULE_B = "blocks.2"
SELECTED_STEP = 2
SECOND_STEP = 3
SOURCE: dict[str, Any] = {
    "kind": "synthetic",
    "synthetic": {
        "name": "tiny-causal-lm",
        "model": {"seed": 7, "model_dim": 32, "num_layers": 3, "max_seq_len": 96},
    },
}


def _say(step: str, detail: str = "") -> None:
    print(f"[{step}] {detail}".rstrip())


def _data(envelope: dict[str, Any], what: str) -> dict[str, Any]:
    if not envelope.get("ok"):
        raise SystemExit(f"{what} failed: {json.dumps(envelope.get('error'), indent=2)}")
    return envelope["data"]


def _digest(module: torch.nn.Module) -> str:
    """Digest names, dtypes, shapes, and values, so mutation is detectable."""

    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _buffer_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, buffer in sorted(module.named_buffers()):
        digest.update(name.encode())
        digest.update(str(buffer.dtype).encode())
        digest.update(str(tuple(buffer.shape)).encode())
        digest.update(buffer.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _spec(service: ToolService, module: str, *, prompt_length: int, label: str) -> dict[str, Any]:
    """One position-free intervention specification stored as an artifact."""

    created = _data(
        service.call(
            "create_activation_intervention",
            {
                "name": label,
                "module": module,
                "positions": [0],
                "operation": "zero",
                "position_semantics": "consuming",
                "model": {"name": "tiny-causal-lm"},
                "sequence_length": prompt_length,
                "save": True,
            },
        ),
        "create_activation_intervention",
    )
    assert created["artifact_id"], created
    return created


def _item(
    spec: dict[str, Any],
    *,
    order: int,
    steps: list[int],
    label: str,
    position_mode: str = "last",
) -> dict[str, Any]:
    return {
        "intervention_artifact_id": spec["artifact_id"],
        "label": label,
        "order": order,
        "schedule": {"kind": "generation-steps", "steps": steps},
        "position_mode": position_mode,
        "on_step_call_indices": [0],
    }


def _plan(
    service: ToolService,
    name: str,
    items: list[dict[str, Any]],
    *,
    tokens: list[int],
    execution_mode: str = "uncached-generation",
) -> dict[str, Any]:
    return _data(
        service.call(
            "create_intervention_plan",
            {
                "name": name,
                "interventions": items,
                "model": {"name": "tiny-causal-lm"},
                "prompt_token_ids": tokens,
                "execution_mode": execution_mode,
                "max_new_tokens": 5,
                "save": True,
            },
        ),
        f"create_intervention_plan({name})",
    )


def _run(
    service: ToolService,
    plan: dict[str, Any],
    *,
    tokens: list[int],
    identity: bool = True,
    staged: bool = False,
) -> dict[str, Any]:
    if staged:
        return _data(
            service.call(
                "run_staged_intervention",
                {
                    "plan_artifact_id": plan["artifact_id"],
                    "source": SOURCE,
                    "token_ids": tokens,
                    "identity": identity,
                    "isolation": "clone",
                    "save": True,
                },
            ),
            "run_staged_intervention",
        )
    return _data(
        service.call(
            "run_temporal_intervention",
            {
                "plan_artifact_id": plan["artifact_id"],
                "source": SOURCE,
                "token_ids": tokens,
                "identity": identity,
                "save": True,
            },
        ),
        "run_temporal_intervention",
    )


def _log_events(service: ToolService, event_log_artifact_id: str) -> list[dict[str, Any]]:
    inspected = _data(
        service.call(
            "inspect_intervention_events",
            {"event_log_artifact_id": event_log_artifact_id, "include_events": True},
        ),
        "inspect_intervention_events",
    )
    return [event for log in inspected["logs"] for event in log.get("events", [])]


def _loaded_plan(workspace: Workspace, artifact_id: str) -> InterventionPlan:
    return InterventionPlan.model_validate(workspace.artifacts.read_json(artifact_id))


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
        root = Path(tempfile.mkdtemp(prefix="wsp-temporal-demo-"))
    workspace = Workspace(root, create=True, name="temporal-demo")
    service = ToolService(workspace=workspace, raise_errors=True)

    # 1 ---------------------------------------------------------------- #
    runtime = build_runtime()
    parameters_before = _digest(runtime.model)
    buffers_before = _buffer_digest(runtime.model)
    tokens = list(runtime.tokenizer.encode(PROMPT, add_special_tokens=True))
    _say(
        "1",
        f"deterministic transformer ({sum(p.numel() for p in runtime.model.parameters())} "
        f"parameters), prompt {PROMPT!r} -> {tokens}",
    )

    # 2 ---------------------------------------------------------------- #
    trace = _data(
        service.call(
            "record_trace",
            {
                "input_text": f"{MARKER} {PROMPT}",
                "params": {"max_new_tokens": 6},
                "backend": {"kind": "synthetic", "synthetic": {"model": {}}},
            },
        ),
        "record_trace",
    )["trace"]
    _say(
        "2",
        f"baseline trace {trace['trace_id']} generated {trace['text']!r} "
        f"({trace['termination']['generated_tokens']} tokens, "
        f"termination={trace['termination']['reason']})",
    )

    # 3 ---------------------------------------------------------------- #
    _say(
        "3",
        f"selected generation step {SELECTED_STEP}: the forward that consumes "
        f"{len(tokens) + SELECTED_STEP} tokens and predicts generated token {SELECTED_STEP}",
    )

    # 4 ---------------------------------------------------------------- #
    spec_a = _spec(
        service,
        MODULE_A,
        prompt_length=len(tokens),
        label=f"scheduled zero ablation at step {SELECTED_STEP}",
    )
    plan_a = _plan(
        service,
        "scheduled ablation A",
        [_item(spec_a, order=0, steps=[SELECTED_STEP], label="A")],
        tokens=tokens,
    )
    assert plan_a["valid"] is True, plan_a["validation"]
    _say(
        "4",
        f"plan {plan_a['plan_id']} created with one intervention scheduled for step "
        f"{SELECTED_STEP} (position_mode=last)",
    )

    # 5 ---------------------------------------------------------------- #
    validation = _data(
        service.call(
            "validate_intervention_plan",
            {"plan_artifact_id": plan_a["artifact_id"], "source": SOURCE, "token_ids": tokens},
        ),
        "validate_intervention_plan",
    )
    preview = _data(
        service.call(
            "preview_intervention_plan",
            {"plan_artifact_id": plan_a["artifact_id"], "source": SOURCE, "token_ids": tokens},
        ),
        "preview_intervention_plan",
    )
    assert validation["valid"] is True, validation["errors"]
    assert [firing["generation_step"] for firing in preview["firings"]] == [SELECTED_STEP]
    firing = preview["firings"][0]
    assert firing["module"] == MODULE_A
    _say(
        "5",
        f"validated ({len(validation['checks'])} checks) and previewed: "
        f"step={firing['generation_step']} module={firing['module']} "
        f"rows={firing['rows']} shape={firing['tensor_shape']} "
        f"token_indices={firing['token_indices']}",
    )

    # 6 ---------------------------------------------------------------- #
    source_plan = _loaded_plan(workspace, plan_a["artifact_id"])
    identity_plan = identity_control_plan(source_plan)
    assert identity_plan.plan_kind == "identity-control"
    assert identity_plan.interventions[0].intervention.operation == "identity"
    assert (
        identity_plan.interventions[0].schedule.resolve()
        == source_plan.interventions[0].schedule.resolve()
    )
    assert (
        identity_plan.interventions[0].on_step_call_indices
        == source_plan.interventions[0].on_step_call_indices
    )
    _say(
        "6",
        f"identity plan {identity_plan.plan_id} keeps the same module, schedule, and call "
        "indices with operation=identity",
    )

    # 7 ---------------------------------------------------------------- #
    identity_run = _data(
        service.call(
            "run_temporal_intervention",
            {
                "plan": identity_plan.model_dump(mode="json"),
                "source": SOURCE,
                "token_ids": tokens,
                "identity": True,
                "save": True,
            },
        ),
        "run_temporal_intervention(identity plan)",
    )
    control_conditions = {item["condition"]: item for item in identity_run["conditions"]}
    assert set(control_conditions) == {"baseline", "identity", "plan"}
    _say(
        "7",
        "baseline and identity conditions run: baseline "
        f"{control_conditions['baseline']['generated_token_ids']} identity "
        f"{control_conditions['identity']['generated_token_ids']}",
    )

    # 8 ---------------------------------------------------------------- #
    control = identity_run["identity_control"]
    assert control["agreement"] == "exact", control
    assert identity_run["validity"]["status"] == "valid"
    assert control_conditions["identity"]["events_applied"] == 0
    assert control_conditions["identity"]["events_identity"] == 1
    _say(
        "8",
        f"identity control {control['agreement']}: attaching hooks at the scheduled site "
        "changed nothing, so attribution at this site is licensed",
    )

    # 9 ---------------------------------------------------------------- #
    run_a = _run(service, plan_a, tokens=tokens)
    assert run_a["validity"]["status"] == "valid", run_a["validity"]
    _say(
        "9",
        f"scheduled intervention executed: {run_a['summary']['events_fired']} event(s) fired, "
        f"{run_a['summary']['events_missing']} missing, identity "
        f"{run_a['summary']['identity_agreement']}",
    )

    # 10 --------------------------------------------------------------- #
    plan_events = [
        event
        for event in _log_events(service, run_a["event_log_artifact_ids"]["plan"])
        if event["execution_result"] == "applied"
    ]
    assert len(plan_events) == 1, plan_events
    event = plan_events[0]
    assert event["generation_step"] == SELECTED_STEP
    assert event["module"] == MODULE_A
    assert event["after"]["l2_norm"] == 0.0
    _say(
        "10",
        f"event log: intervention {event['intervention_id']} fired at generation step "
        f"{event['generation_step']} (forward invocation {event['forward_invocation']}), "
        f"module {event['module']}, rows {event['rows']}, "
        f"elements {event['elements_modified']}, after_l2={event['after']['l2_norm']:.1f}",
    )

    # 11 --------------------------------------------------------------- #
    comparison = run_a["comparison"]
    plan_condition = [
        item for item in run_a["conditions"] if item["condition"] == "plan"
    ][0]
    _say(
        "11",
        f"downstream: first divergence at {comparison['first_divergence']} "
        f"(longest common prefix {comparison['longest_common_prefix']}), token agreement "
        f"{comparison['token_agreement_rate']:.3f}, tokens "
        f"{control_conditions['baseline']['generated_token_ids']} -> "
        f"{plan_condition['generated_token_ids']}",
    )

    # 12 --------------------------------------------------------------- #
    spec_b = _spec(
        service,
        MODULE_B,
        prompt_length=len(tokens),
        label=f"second site ablation at step {SECOND_STEP}",
    )
    plan_b = _plan(
        service,
        "scheduled ablation B",
        [_item(spec_b, order=0, steps=[SECOND_STEP], label="B")],
        tokens=tokens,
    )
    assert plan_b["valid"] is True, plan_b["validation"]
    _say(
        "12",
        f"second plan {plan_b['plan_id']} targets {MODULE_B} at generation step "
        f"{SECOND_STEP} (a different module and a different step)",
    )

    # 13 --------------------------------------------------------------- #
    run_b = _run(service, plan_b, tokens=tokens)
    assert run_b["validity"]["status"] == "valid", run_b["validity"]
    _say(
        "13",
        f"individual effects measured: A run {run_a['run_id']}, B run {run_b['run_id']} "
        f"(each with its own baseline and identity condition)",
    )

    # 14 --------------------------------------------------------------- #
    plan_ab = _plan(
        service,
        "combined plan A+B",
        [
            _item(spec_a, order=0, steps=[SELECTED_STEP], label="A"),
            _item(spec_b, order=1, steps=[SECOND_STEP], label="B"),
        ],
        tokens=tokens,
    )
    assert plan_ab["valid"] is True, plan_ab["validation"]
    run_ab = _run(service, plan_ab, tokens=tokens)
    assert run_ab["validity"]["status"] == "valid", run_ab["validity"]
    modules_fired = sorted(
        {
            event["module"]
            for event in _log_events(service, run_ab["event_log_artifact_ids"]["plan"])
            if event["execution_result"] == "applied"
        }
    )
    assert modules_fired == [MODULE_A, MODULE_B], modules_fired
    _say(
        "14",
        f"combined plan executed: both sites fired (steps {SELECTED_STEP} and {SECOND_STEP}), "
        f"modules {modules_fired}, recorded in actual model execution order",
    )

    # 15 --------------------------------------------------------------- #
    composition = _data(
        service.call(
            "run_compositional_comparison",
            {
                "plan_a_artifact_id": plan_a["artifact_id"],
                "plan_b_artifact_id": plan_b["artifact_id"],
                "plan_ab_artifact_id": plan_ab["artifact_id"],
                "source": SOURCE,
                "token_ids": tokens,
                "metric": "mean_generated_logprob",
                "save": True,
            },
        ),
        "run_compositional_comparison",
    )
    assert composition["validity"]["status"] == "valid", composition["validity"]
    assert [item["condition"] for item in composition["conditions"]] == [
        "baseline",
        "identity",
        "A",
        "B",
        "AB",
    ]
    _say(
        "15",
        "control comparison: baseline, identity, A, B, and A+B measured in one execution "
        f"mode ({composition['execution_mode']}) with the same prompt and precision",
    )

    # 16 --------------------------------------------------------------- #
    contrast = composition["interactions"][0]
    additive = contrast["y_ab"] - contrast["y_baseline"]
    predicted_additive = contrast["delta_a"] + contrast["delta_b"]
    assert abs(contrast["interaction"] - (additive - predicted_additive)) < 1e-12
    _say(
        "16",
        "interaction contrast on mean_generated_logprob: "
        f"Y0={contrast['y_baseline']:.6f} YA={contrast['y_a']:.6f} "
        f"YB={contrast['y_b']:.6f} YAB={contrast['y_ab']:.6f} "
        f"interaction={contrast['interaction']:+.6f} "
        f"(additive prediction {predicted_additive:+.6f})",
    )
    _say(
        "16b",
        "that number is non-additivity on the chosen scale; it is not a discovered "
        "circuit and no mechanistic interaction is claimed",
    )

    # 17 --------------------------------------------------------------- #
    prepared = _data(
        service.call(
            "prepare_staged_generation",
            {"source": SOURCE, "token_ids": tokens, "max_new_tokens": 5},
        ),
        "prepare_staged_generation",
    )
    assert prepared["supported"] is True, prepared
    _say(
        "17",
        f"staged prefill prepared: architecture={prepared['architecture_class']} "
        f"state={prepared['state_kind']} bytes={prepared['state_bytes']} "
        f"hash={prepared['state_hash'][:16]}... prefill={prepared['prefill_ms']:.3f} ms",
    )

    # 18 --------------------------------------------------------------- #
    staged_plan = _plan(
        service,
        "staged ablation A",
        [_item(spec_a, order=0, steps=[1, 2], label="A staged")],
        tokens=tokens,
        execution_mode="staged-generation",
    )
    assert staged_plan["valid"] is True, staged_plan["validation"]
    verification = _data(
        service.call(
            "verify_staged_generation",
            {
                "plan_artifact_id": staged_plan["artifact_id"],
                "source": SOURCE,
                "token_ids": tokens,
                "isolation": "clone",
                "save": True,
            },
        ),
        "verify_staged_generation",
    )
    assert verification["agreement"] in ("exact", "within_tolerance"), verification
    assert verification["token_sequence_match"] is True
    assert verification["identity_control_agreement"] == "exact"
    _say(
        "18",
        f"staged verification: agreement={verification['agreement']} "
        f"max_abs_logits={verification['max_abs_logits']:.3e} "
        f"atol={verification['atol']:.0e} tokens_match={verification['token_sequence_match']} "
        f"identity={verification['identity_control_agreement']} "
        f"checks={[check['name'] for check in verification['checks']]}",
    )

    # 19 --------------------------------------------------------------- #
    staged_run = _run(service, staged_plan, tokens=tokens, staged=True)
    assert staged_run["validity"]["status"] == "valid", staged_run["validity"]
    staged_baseline = [
        item for item in staged_run["conditions"] if item["condition"] == "baseline"
    ][0]["generated_token_ids"]
    uncached_baseline = control_conditions["baseline"]["generated_token_ids"]
    assert staged_baseline == uncached_baseline, (staged_baseline, uncached_baseline)
    staged_fired = [
        event["generation_step"]
        for event in _log_events(service, staged_run["event_log_artifact_ids"]["plan"])
        if event["execution_result"] == "applied"
    ]
    assert staged_fired == [1, 2], staged_fired
    performance = staged_run["performance"]
    _say(
        "19",
        f"staged intervention through the model's own cache: events at steps {staged_fired}, "
        f"baseline tokens equal to the uncached reference, "
        f"prefill={performance['prefill_ms']:.3f} ms decode={performance['decode_ms_total']:.3f} ms "
        f"uncached={performance['uncached_reference_ms']:.3f} ms "
        f"speedup={performance['speedup_ratio']:.2f}x "
        f"state={performance['cache_state_bytes']} bytes",
    )

    # 20 --------------------------------------------------------------- #
    case = _data(
        service.call(
            "create_failure_case",
            {
                "title": "temporal intervention instrument demonstration",
                "summary": "an offline correctness fixture, not a scientific result",
                "observations": [trace["trace_id"]],
            },
        ),
        "create_failure_case",
    )["case"]
    event_log_ids = [
        *run_a["event_log_artifact_ids"].values(),
        *run_b["event_log_artifact_ids"].values(),
        *run_ab["event_log_artifact_ids"].values(),
        *staged_run["event_log_artifact_ids"].values(),
    ]
    evidence = _data(
        service.call(
            "attach_temporal_evidence",
            {
                "case_id": case["case_id"],
                "title": "temporal and compositional evidence",
                "summary": (
                    "scheduled ablations at steps "
                    f"{SELECTED_STEP} and {SECOND_STEP} and one verified staged run"
                ),
                "observation_trace_id": trace["trace_id"],
                "plan_artifact_ids": [
                    plan_a["artifact_id"],
                    plan_b["artifact_id"],
                    plan_ab["artifact_id"],
                    staged_plan["artifact_id"],
                ],
                "event_log_artifact_ids": event_log_ids,
                "run_artifact_ids": [
                    run_a["artifact_id"],
                    run_b["artifact_id"],
                    run_ab["artifact_id"],
                ],
                "composition_artifact_ids": [composition["artifact_id"]],
                "staged_artifact_ids": [staged_run["artifact_id"]],
                "verification_artifact_ids": [verification["artifact_id"]],
                "hypotheses": [
                    {
                        "statement": (
                            "the selected block activation at generation step "
                            f"{SELECTED_STEP} contributes to the next-token distribution "
                            "under the tested conditions"
                        ),
                        "rationale": "observed under the tested conditions only",
                    }
                ],
                "notes": ["instrument demonstration; no model capability claim"],
            },
        ),
        "attach_temporal_evidence",
    )
    summary = evidence["summary"]
    _say(
        "20",
        f"case {case['case_id']} references evidence {evidence['evidence_id']} "
        f"({summary['plans']} plans, {summary['event_logs']} event logs, "
        f"{summary['temporal_runs']} runs, {summary['compositions']} comparison, "
        f"{summary['staged']} staged record)",
    )

    # 21 --------------------------------------------------------------- #
    private_path = root / "case-private.json"
    exported = _data(
        service.call(
            "export_failure_case",
            {"case_id": case["case_id"], "output_path": str(private_path)},
        ),
        "export_failure_case",
    )
    _say("21", f"private export {private_path.name} ({private_path.stat().st_size} bytes)")

    # 22 --------------------------------------------------------------- #
    fresh_workspace = Workspace(root / "fresh", create=True, name="fresh")
    fresh = ToolService(workspace=fresh_workspace, raise_errors=True)
    imported = _data(fresh.call("import_failure_case", {"path": str(private_path)}), "import")
    loaded = _data(
        fresh.call("inspect_failure_case", {"case_id": case["case_id"]}),
        "inspect_failure_case",
    )
    link = loaded["case"]["investigations"][0]
    assert link["plan_ids"] and link["temporal_run_ids"] and link["composition_ids"]
    assert link["staged_ids"]
    integrity = _data(
        fresh.call(
            "inspect_investigation_evidence", {"artifact_id": link["artifact_id"]}
        ),
        "inspect_investigation_evidence",
    )
    assert integrity["integrity"]["missing_artifacts"] == [], integrity["integrity"]
    referenced = [
        link["artifact_id"],
        *[ref["artifact_id"] for ref in integrity["plans"]],
        *[ref["artifact_id"] for ref in integrity["event_logs"]],
        *[ref["artifact_id"] for ref in integrity["temporal_runs"]],
        *[ref["artifact_id"] for ref in integrity["compositions"]],
        *[ref["artifact_id"] for ref in integrity["staged_continuations"]],
    ]
    for artifact_id in referenced:
        fresh_workspace.artifacts.verify(artifact_id)
    assert exported["case_sha256"] == loaded["case_sha256"]
    _say(
        "22",
        f"reloaded into a fresh workspace as case {imported['case']['case_id']}; "
        f"{len(referenced)} artifact references and their checksums verified",
    )

    # 23 --------------------------------------------------------------- #
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
    for needle in (MARKER, "abcabc"):
        assert needle.encode() not in raw, f"redacted export retained {needle!r}"
    redacted_fresh = ToolService(
        workspace=Workspace(root / "fresh-redacted", create=True), raise_errors=True
    )
    reloaded = _data(
        redacted_fresh.call("import_failure_case", {"path": str(redacted_path)}),
        "import redacted",
    )
    assert reloaded["case"]["redaction"]["mode"] == "redacted"
    _say(
        "23",
        f"redacted export {redacted_path.name} ({redacted_path.stat().st_size} bytes) "
        f"leak_scan={redacted['privacy']['leak_scan']}; reloaded with a redaction record",
    )

    # 24 --------------------------------------------------------------- #
    checks = {check["name"]: check["status"] for check in run_ab["validity"]["checks"]}
    for name in ("parameters_unchanged", "mutable_state_unchanged", "plan_valid"):
        assert checks[name] == "pass", checks
    assert _digest(runtime.model) == parameters_before, "model parameters changed"
    assert _buffer_digest(runtime.model) == buffers_before, "model buffers changed"
    _say(
        "24",
        f"model state preserved: parameter digest unchanged ({parameters_before[:16]}...), "
        "buffers unchanged, mutable-state check pass, and every hook was removed after "
        "each run",
    )

    print()
    print("artifacts:", root)
    print(
        "result: the temporal and compositional instruments agree with their controls. "
        "A scheduled event is a measurement under the recorded conditions, not proof of "
        "a causal circuit."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
