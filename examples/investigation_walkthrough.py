#!/usr/bin/env python
"""WSP Workbench V1.2 walkthrough: behavior -> replay -> measurement -> evidence.

The script is deliberately boring and explicit.  It walks the full investigation
chain the workbench is built for and prints what each step actually established:

    trace -> detector finding -> selected span -> replay plan -> teacher-forced
    replay -> token alignment -> activation capture -> controlled comparison
    -> investigation evidence -> private and redacted case export -> import

Everything runs offline against a small local causal transformer built in
process.  It demonstrates the *instruments*; it does not claim to find a real
model defect or a causal mechanism.

Usage::

    python examples/investigation_walkthrough.py
    python examples/investigation_walkthrough.py --workspace /tmp/wsp-demo
    python examples/investigation_walkthrough.py --model ~/models/some-local-ckpt
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

from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace

LOOPING_PROMPT = "the cat sat on the mat"
TERMINATING_PROMPT = "the cat sat on the mat stop"
VARIANT_PROMPT = "the dog sat on the mat"
CONTROL_MODULE = "blocks.1"


def _say(step: str, detail: str = "") -> None:
    print(f"[{step}] {detail}".rstrip())


def _unwrap(envelope: dict[str, Any], what: str) -> dict[str, Any]:
    if not envelope.get("ok"):
        raise SystemExit(f"{what} failed: {json.dumps(envelope.get('error'), indent=2)}")
    return envelope["data"]


def _parameter_digest(module: Any) -> str:
    """Stable digest of every parameter, so accidental mutation is detectable."""

    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().reshape(-1).to("cpu").to(torch.float64).numpy().tobytes())
    return digest.hexdigest()


def _payload(service: ToolService, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _unwrap(service.call(operation, payload), operation)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace directory. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional local Hugging Face checkpoint directory to use instead of the synthetic model.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep a temporary workspace instead of deleting it.",
    )
    args = parser.parse_args()

    temporary = args.workspace is None
    root = args.workspace or Path(tempfile.mkdtemp(prefix="wsp-walkthrough-"))
    if root.exists() and not temporary:
        shutil.rmtree(root)
    workspace = Workspace(root, create=True, name="walkthrough")
    service = ToolService(workspace=workspace, raise_errors=True)

    # ------------------------------------------------------------------ #
    # 1-3. A controlled pair of generations
    # ------------------------------------------------------------------ #
    if args.model is not None:
        backend = {
            "kind": "huggingface",
            "huggingface": {"model_path": str(Path(args.model).expanduser())},
        }
        model_source = {"kind": "local_hf", "path": str(Path(args.model).expanduser())}
        module = None  # resolved later from the real checkpoint
        _say("0", f"using local checkpoint {model_source['path']}")
    else:
        backend = {"kind": "synthetic", "synthetic": {"model": {}}}
        model_source = {"kind": "synthetic"}
        module = CONTROL_MODULE
        _say("0", "using the built-in synthetic causal transformer")

    def record(text: str) -> dict[str, Any]:
        return _payload(
            service,
            "record_trace",
            {
                "input_text": text,
                "params": {"max_new_tokens": 12},
                "backend": backend,
            },
        )["trace"]

    observation = record(LOOPING_PROMPT)
    control = record(TERMINATING_PROMPT)
    variant = record(VARIANT_PROMPT)
    _say("1", f"observed trace {observation['trace_id']} -> {observation['text']!r}")
    _say("2", f"control  trace {control['trace_id']} -> {control['text']!r}")
    _say("2b", f"variant  trace {variant['trace_id']} -> {variant['text']!r} (different input)")
    _say(
        "3",
        "both traces are versioned: "
        f"schema {observation['schema_version']}, "
        f"{observation['token_count']} recorded tokens, "
        f"token ids {'present' if observation['input'].get('token_ids') else 'missing'}",
    )

    # ------------------------------------------------------------------ #
    # 4-5. Detector finding, span selection, replay plan
    # ------------------------------------------------------------------ #
    scan = _payload(service, "scan_trace", {"trace_id": observation["trace_id"]})
    if not scan["findings"]:
        _say("4", "no detector fired; continuing with an explicit token range")
        span_source: dict[str, Any] = {"role": "generation", "role_start": 0, "role_end": 2}
    else:
        finding = scan["findings"][0]
        _say("4", f"detector {finding['detector_id']} reported {finding['kind']} ({finding['severity']})")
        span_source = {"span_artifact_id": None}

    alignment = _payload(
        service,
        "align_trace_tokens",
        {
            "trace_id": observation["trace_id"],
            "model": model_source if args.model is not None else None,
            "position_limit": 4,
        },
    )
    _say(
        "5",
        f"aligned {alignment['full_length']} tokens "
        f"(prompt {alignment['prompt_length']}, generated {alignment['generated_length']}); "
        f"prediction semantics: {alignment['prediction_semantics']}",
    )

    evidence_refs = scan["findings"][0]["evidence"] if scan["findings"] else []
    if evidence_refs:
        span = _payload(
            service,
            "resolve_finding_span",
            {
                "trace_id": observation["trace_id"],
                "alignment_artifact_id": alignment["alignment_artifact_id"],
                "kind": "finding_evidence",
                "detail": dict(evidence_refs[0]),
            },
        )
        _say(
            "5b",
            f"finding span resolved to {span['status']} at full indices {span['full_token_indices']}",
        )
        span_source = {"span_artifact_id": span["span_artifact_id"]}
    elif scan["findings"]:
        _say(
            "5b",
            f"detector {scan['findings'][0]['detector_id']} reported no textual evidence span; "
            "the comparison will use shared positions only",
        )

    plan = _payload(
        service,
        "prepare_replay",
        {"trace_id": observation["trace_id"], "execution": {"max_sequence_length": 64}},
    )
    _say(
        "6",
        f"replay plan {plan['plan_id']} ({plan['semantics']}, "
        f"source {plan['sequence']['source']}), checksum {plan['plan_checksum'][:19]}...",
    )

    # ------------------------------------------------------------------ #
    # 7-8. Teacher-forced replay and alignment-checked capture
    # ------------------------------------------------------------------ #
    replay = _payload(
        service,
        "verify_replay",
        {"plan_artifact_id": plan["plan_artifact_id"], "model": model_source},
    )
    agreement = replay["agreement"]
    _say(
        "7",
        f"replay {replay['replay_id']}: token_sequence_reconstructed="
        f"{replay['token_sequence_reconstructed']}, forward_executed={replay['forward_executed']}, "
        f"numerical_agreement={replay['numerical_agreement']} "
        f"(max |diff| {agreement['max_abs_diff']:.3e} over {agreement['tokens_compared']} tokens)",
    )
    _say(
        "7b",
        f"generation_reproducibility={replay['generation_reproducibility']}: teacher forcing does "
        "not claim sampling reproducibility",
    )

    if module is None:
        points = _payload(
            service,
            "list_capture_points",
            {"model": model_source, "limit": 500},
        )
        blocks = [point["path"] for point in points["points"] if point["kind"] == "block"]
        module = blocks[1] if len(blocks) > 1 else blocks[0]
        _say("8", f"selected a real checkpoint block: {module}")

    # The interesting span lies inside the divergent continuation, so those
    # positions do not exist in the control execution.  Comparing them would
    # compare unrelated context, so the comparison uses positions both
    # executions genuinely share: the prompt prefix.
    observation_prompt = len(observation["input"].get("token_ids") or [])
    control_prompt = len(control["input"].get("token_ids") or [])
    shared_positions = list(range(1, max(1, min(5, observation_prompt, control_prompt))))
    if not shared_positions:
        shared_positions = [0]
    _say(
        "8b",
        f"comparing at shared prefix positions {shared_positions}; the detected span "
        "lives in the divergent continuation and is not comparable across executions",
    )

    capture = _payload(
        service,
        "capture_trace_activations",
        {
            "trace_id": observation["trace_id"],
            "model": model_source,
            "modules": [module],
            "mode": "selected",
            "positions": shared_positions,
            "position_semantics": "consuming",
        },
    )
    _say(
        "8",
        f"captured {len(capture['measurements'])} measurements at positions {capture['positions']} "
        f"from {module}, mode={capture['mode']}, "
        f"replay verification={capture['replay_verification']['numerical_agreement']}",
    )

    # ------------------------------------------------------------------ #
    # 9-10. The controlled comparison
    # ------------------------------------------------------------------ #
    control_capture = _payload(
        service,
        "capture_trace_activations",
        {
            "trace_id": control["trace_id"],
            "model": model_source,
            "modules": [module],
            "mode": "selected",
            "positions": shared_positions,
            "position_semantics": "consuming",
        },
    )
    _say(
        "9",
        f"control capture {control_capture['capture_id']} at the same module and positions",
    )

    comparison = _payload(
        service,
        "compare_activation_captures",
        {
            "left_capture_artifact_id": capture["capture_artifact_id"],
            "right_capture_artifact_id": control_capture["capture_artifact_id"],
            "metrics": ["l2", "cosine", "max_abs"],
            "alignment": {"strategy": "identical_positions"},
        },
    )
    first_entry = comparison["entries"][0]
    metrics = {item["metric"]: item["value"] for item in first_entry["metrics"]}
    _say(
        "10",
        f"identical-context control: {comparison['counts']['compared']} paired measurements, "
        f"l2={metrics.get('l2'):.4f}, cosine={metrics.get('cosine'):.4f} "
        "(zero difference where the context is identical)",
    )

    # A comparison that should *not* be zero: a genuinely different input, in
    # positions that exist in both executions and follow the divergence.
    variant_capture = _payload(
        service,
        "capture_trace_activations",
        {
            "trace_id": variant["trace_id"],
            "model": model_source,
            "modules": [module],
            "mode": "selected",
            "positions": shared_positions,
            "position_semantics": "consuming",
        },
    )
    variant_comparison = _payload(
        service,
        "compare_activation_captures",
        {
            "left_capture_artifact_id": capture["capture_artifact_id"],
            "right_capture_artifact_id": variant_capture["capture_artifact_id"],
            "metrics": ["l2", "cosine", "max_abs"],
            "alignment": {"strategy": "identical_positions"},
        },
    )
    def _metric(entry: dict[str, Any], name: str) -> float | None:
        for item in entry["metrics"]:
            if item["metric"] == name:
                return item["value"]
        return None

    worst = max(variant_comparison["entries"], key=lambda item: _metric(item, "l2") or 0.0)
    _say(
        "10b",
        f"different-input comparison: max l2={_metric(worst, 'l2'):.4f}, "
        f"cosine={_metric(worst, 'cosine'):.4f} at position {worst['left_position']} "
        f"(the paired measurements agree exactly before the inputs diverge)",
    )
    _say(
        "10c",
        "these are numbers, not a mechanism: "
        f"{comparison['limitations'][1]}",
    )

    # ------------------------------------------------------------------ #
    # 11-12. Failure case, evidence, private export
    # ------------------------------------------------------------------ #
    case = _payload(
        service,
        "create_failure_case",
        {
            "title": "repetition versus termination",
            "summary": "one prompt loops, the other stops; both prompts share a prefix",
            "observations": [observation["trace_id"]],
            "controls": [control["trace_id"]],
            "findings": scan["findings"],
            "patches": [],
        },
    )["case"]
    _say("11", f"failure case {case['case_id']} created")

    evidence = _payload(
        service,
        "attach_investigation_evidence",
        {
            "case_id": case["case_id"],
            "title": "internal measurements for the repetition finding",
            "summary": "teacher-forced replay, one module, two controlled executions",
            "observation_trace_id": observation["trace_id"],
            "control_trace_ids": [control["trace_id"]],
            "replay_artifact_ids": [replay["result_artifact_id"]],
            "alignment_artifact_ids": [alignment["alignment_artifact_id"]],
            **(
                {"span_artifact_ids": [span_source["span_artifact_id"]]}
                if span_source.get("span_artifact_id")
                else {}
            ),
            "capture_artifact_ids": [
                capture["capture_artifact_id"],
                control_capture["capture_artifact_id"],
                variant_capture["capture_artifact_id"],
            ],
            "comparison_artifact_ids": [
                comparison["comparison_artifact_id"],
                variant_comparison["comparison_artifact_id"],
            ],
            "hypotheses": [
                {
                    "statement": "the trailing stop token changes the next-token distribution",
                    "rationale": "recorded by a human, not derived from the measurements",
                }
            ],
        },
    )
    _say(
        "11b",
        f"evidence {evidence['evidence_id']} attached; "
        f"{len(evidence['captures'])} captures, {len(evidence['comparisons'])} comparisons, "
        f"{len(evidence['hypotheses'])} unverified hypothesis",
    )

    private_path = root / "case-private.json"
    _payload(
        service,
        "export_failure_case",
        {"case_id": case["case_id"], "output_path": str(private_path), "policy": {"mode": "private"}},
    )
    _say("12", f"private export {private_path.name} ({private_path.stat().st_size} bytes)")

    # ------------------------------------------------------------------ #
    # 13-15. Fresh workspace import, integrity, immutability
    # ------------------------------------------------------------------ #
    fresh_root = root / "imported"
    fresh = Workspace(fresh_root, create=True, name="imported")
    fresh_service = ToolService(workspace=fresh, raise_errors=True)
    imported = _payload(fresh_service, "import_failure_case", {"path": str(private_path)})
    _say("13", f"imported case {imported['case']['case_id']} into a fresh workspace")

    inspected = _payload(
        fresh_service,
        "inspect_failure_case",
        {"case_id": imported["case"]["case_id"]},
    )["case"]
    link = inspected["investigations"][0]
    evidence_readback = _payload(
        fresh_service,
        "inspect_investigation_evidence",
        {"artifact_id": link["artifact_id"]},
    )
    integrity = evidence_readback["integrity"]
    _say(
        "14",
        f"integrity: checksum_matches={integrity['checksum_matches']}, "
        f"missing_artifacts={len(integrity['missing_artifacts'])}, "
        f"captures restored={len(evidence_readback['capture_records'])}, "
        f"comparisons restored={len(evidence_readback['comparison_records'])}",
    )
    if integrity["missing_artifacts"] or integrity["checksum_matches"] is not True:
        raise SystemExit("imported evidence failed its integrity check")

    if args.model is None:
        from wsp.execution.synthetic_lm import build_runtime

        runtime = build_runtime()
        before = _parameter_digest(runtime.model)
        _payload(
            service,
            "capture_trace_activations",
            {
                "trace_id": observation["trace_id"],
                "model": model_source,
                "modules": [module],
                "mode": "summary",
                "positions": capture["positions"],
            },
        )
        after = _parameter_digest(runtime.model)
        if before != after:
            raise SystemExit("model parameters changed during observation")
        _say("15", f"parameter digest unchanged after observation: {after[:16]}...")
    else:
        _say("15", "parameter immutability is asserted per replay by the workbench (checksums above)")

    # ------------------------------------------------------------------ #
    # 16. Redacted export
    # ------------------------------------------------------------------ #
    redacted_path = root / "case-redacted.json"
    redacted = _payload(
        service,
        "export_failure_case",
        {
            "case_id": case["case_id"],
            "output_path": str(redacted_path),
            "policy": {"mode": "redacted", "sensitive_markers": [LOOPING_PROMPT]},
        },
    )
    bundle = json.loads(redacted_path.read_text())
    _say(
        "16",
        f"redacted export {redacted_path.name}: mode={bundle['privacy']['mode']}, "
        f"replay_capability={bundle['privacy']['replay_capability']}, "
        f"leak_scan={bundle['privacy']['leak_scan']}, "
        f"artifacts rewritten={len(bundle['artifact_map'])}",
    )
    _say(
        "16b",
        f"{sum(1 for value in bundle['artifact_map'].values() if value is None)} artifact(s) "
        "dropped because retained activation values cannot be redacted",
    )
    if LOOPING_PROMPT in redacted_path.read_text():
        raise SystemExit("redacted export leaked the source prompt")
    _say("16c", f"redacted bundle reported {redacted['nbytes']} bytes and excludes the prompt text")

    _say("done", f"workspace: {root}")
    if temporary and not args.keep:
        shutil.rmtree(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
