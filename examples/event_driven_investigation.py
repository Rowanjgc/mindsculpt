"""Offline V1.5 event-driven investigation demonstration.

This example builds a deterministic tiny language model, records a trace that
contains a known token repetition, resolves the earliest repetition event,
demonstrates online trigger timing, reconstructs the exact prefix, runs a
fixed-context baseline/identity/intervention comparison, produces a dossier,
attaches it to a failure case, and exports/imports the evidence.

The repetition is a constructed fixture. Nothing here claims that the model has
a reasoning loop, a defect, or a mechanism. The instruments are the subject.

Run from the repository root:

    python examples/event_driven_investigation.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from wsp.behavior.models import ModelRef
from wsp.core.hashing import sha256_file
from wsp.execution.synthetic_lm import build_runtime
from wsp.interventions.models import (
    ActivationIntervention,
    ExecutionIdentity,
    InterventionTarget,
    TensorSelector,
)
from wsp.replay.engine import parameter_fingerprint
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace


def _spec(*, sequence_length: int, positions: list[int], mode: str) -> ActivationIntervention:
    return ActivationIntervention(
        intervention_id="iv_0123456789abcdef",
        created_at="2026-01-01T00:00:00Z",
        name="fixture zero-ablation at the selected position",
        model=ModelRef(name="tiny-causal-lm"),
        execution=ExecutionIdentity(
            execution_mode=mode,
            sequence_length=sequence_length,
        ),
        target=InterventionTarget(
            module="blocks.0.mlp",
            tensor_role="output",
            position_semantics="consuming",
            selector=TensorSelector(positions=positions),
        ),
        operation="zero",
        notes="fixture intervention; not a hypothesis about a real circuit",
    )


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="wsp-v15-example-"))
    workspace = Workspace(root / "workspace", create=True, name="event-demo")
    service = ToolService(workspace, raise_errors=True)

    # 1. A deterministic tiny model generates a trace with a known repetition.
    recorded = service.call(
        "record_trace",
        {
            "input_text": "loop",
            "backend": {"kind": "synthetic"},
            "params": {"max_new_tokens": 12},
        },
    )["data"]
    trace_id = recorded["trace_id"]
    inspection = service.call(
        "inspect_trace",
        {"trace_id": trace_id, "include_tokens": True, "token_limit": 200},
    )["data"]
    trace = inspection
    prompt_length = len(trace["input"]["token_ids"])
    generated_ids = [token["token_id"] for token in trace["tokens"]]
    full_length = prompt_length + len(generated_ids)
    print(f"1. trace {trace_id}: prompt={trace['input']['token_ids']} generated={generated_ids}")

    # 2. Resolve the earliest deterministic repetition event.
    event_data = service.call(
        "detect_behavioral_events",
        {
            "trace_id": trace_id,
            "event_definition": {
                "kind": "first_repetition",
                "ngram_size": 2,
                "min_occurrences": 2,
            },
        },
    )["data"]
    event = event_data["events"][0]
    event_artifact_id = event_data["event_artifact_id"]
    alignment = event["alignment"]
    print(
        f"2. event {event['event_id']} type={event['event_type']} "
        f"token_position={alignment['token_position']} "
        f"consumed_position={alignment['consumed_position']} "
        f"generation_step={alignment['generation_step']} "
        f"earliest_online_step={alignment['earliest_online_step']}"
    )

    # 3. Online trigger: the intervention can fire no earlier than the next forward.
    online_spec = _spec(sequence_length=prompt_length, positions=[0], mode="uncached-continuation")
    online = service.call(
        "run_event_triggered_intervention",
        {
            "trigger": {
                "trigger_id": "et_0123456789abcdef",
                "mode": "online",
                "definition": {
                    "kind": "first_repetition",
                    "ngram_size": 2,
                    "min_occurrences": 2,
                },
                "delay_steps": 0,
                "steps_after_event": 1,
                "max_triggers": 1,
                "fire_on": "once",
                "position_mode": "last",
            },
            "intervention": online_spec.model_dump(mode="json"),
            "source": {"kind": "synthetic"},
            "token_ids": trace["input"]["token_ids"],
            "max_new_tokens": 10,
        },
    )["data"]
    fired = [item for item in online["events"] if item["state"] == "fired"]
    if fired:
        item = fired[0]
        print(
            f"3. online trigger: observed at step {item['observation_step']}, "
            f"earliest eligible {item['earliest_eligible_step']}, "
            f"fired at {item['fired_steps']} (never before observation)"
        )
    else:
        print("3. online trigger: no event led to an eligible forward")

    # 4. Retrospective fixed-context investigation of the recorded event.
    replay = service.call(
        "prepare_fixed_context_replay",
        {"trace_id": trace_id, "event_artifact_ids": [event_artifact_id]},
    )["data"]
    print(
        f"4. replay {replay['replay_id']}: step={replay['replay_step']} "
        f"consumed={replay['consumed_position']} tokens={replay['sequence']['full_length']} "
        f"sha256={replay['sequence']['sequence_sha256']}"
    )

    fixed_spec = _spec(
        sequence_length=full_length,
        positions=[alignment["consumed_position"]],
        mode="teacher-forced",
    )
    runtime = build_runtime()
    fingerprint_before = parameter_fingerprint(runtime.model)
    comparison = service.call(
        "run_fixed_context_comparison",
        {
            "replay_artifact_id": replay["artifact_id"],
            "intervention": fixed_spec.model_dump(mode="json"),
            "source": {"kind": "synthetic"},
            "continuation": {"max_new_tokens": 4},
        },
    )["data"]
    fingerprint_after = parameter_fingerprint(runtime.model)
    if fingerprint_before != fingerprint_after:
        raise SystemExit("model parameters changed during the example")
    print(
        f"5. fixed-context comparison {comparison['comparison_id']}: "
        f"validity={comparison['validity']['status']} "
        f"identity={comparison['identity_control']['agreement']} "
        f"measurements={len(comparison['measurements'])}"
    )
    for measurement in comparison["measurements"]:
        print(
            f"   position {measurement['position']}: "
            f"baseline={measurement['baseline_logprob']:.6f} "
            f"identity={measurement['identity_logprob']:.6f} "
            f"intervention={measurement['intervention_logprob']:.6f}"
        )
    if comparison.get("continuation"):
        diff = comparison["continuation"]["diff"]
        print(
            "   continuation (separate from fixed context): "
            f"common_prefix={diff['longest_common_prefix']} "
            f"first_divergence={diff['first_divergence']}"
        )

    # 5. A failure case, dossier, and evidence record.
    case = service.call(
        "create_failure_case",
        {
            "title": "Constructed repetition fixture",
            "observations": [trace_id],
            "summary": "A deliberately constructed token repetition for instrument validation.",
        },
    )["data"]["case"]
    dossier = service.call(
        "create_investigation_dossier",
        {
            "title": "Repetition event: fixed-context instrument check",
            "case_id": case["case_id"],
            "trace_id": trace_id,
            "event_artifact_ids": [event_artifact_id],
            "fixed_context_comparison_artifact_id": comparison["artifact_id"],
            "summary": "Observations and a controlled numerical comparison; no mechanism claimed.",
        },
    )["data"]
    evidence = service.call(
        "attach_event_evidence",
        {
            "case_id": case["case_id"],
            "title": "Event-driven investigation evidence",
            "trace_id": trace_id,
            "event_artifact_ids": [event_artifact_id],
            "fixed_context_comparison_artifact_ids": [comparison["artifact_id"]],
            "dossier_artifact_ids": [dossier["artifact_id"]],
        },
    )["data"]
    print(
        f"6. dossier {dossier['dossier_id']} attached to case {case['case_id']}; "
        f"evidence {evidence['evidence_id']}"
    )

    # 6. Export privately, import into a fresh workspace, verify checksums.
    private_path = root / "investigation.wspcase"
    service.call(
        "export_failure_case",
        {
            "case_id": case["case_id"],
            "output_path": str(private_path),
            "policy": {"mode": "private"},
        },
    )
    fresh = Workspace(root / "imported", create=True, name="imported")
    fresh_service = ToolService(fresh, raise_errors=True)
    imported = fresh_service.call(
        "import_failure_case", {"path": str(private_path)}
    )["data"]
    fresh_case_id = imported["case"]["case_id"]
    imported_dossier = fresh_service.call(
        "inspect_investigation_dossier",
        {"dossier_artifact_id": dossier["artifact_id"]},
    )["data"]
    imported_comparison = fresh.artifacts.read_json(comparison["artifact_id"])
    print(
        f"7. imported case {fresh_case_id}; dossier {imported_dossier['dossier_id']} "
        f"reloaded; comparison checksum "
        f"{'present' if imported_comparison.get('checksums') else 'missing'}"
    )

    # 7. Redacted export must also succeed and must not expose the fixture text.
    redacted_path = root / "investigation-redacted.wspcase"
    service.call(
        "export_failure_case",
        {
            "case_id": case["case_id"],
            "output_path": str(redacted_path),
            "policy": {"mode": "redacted"},
        },
    )
    if b"looploop" in redacted_path.read_bytes():
        raise SystemExit("redacted export contains generated fixture text")
    print(f"8. redacted export written to {redacted_path}")

    print(
        "9. model parameters unchanged; source checkpoint was never opened for writing. "
        f"workspace={root}"
    )
    print(
        "10. Reminder: the repetition is a fixture. The comparison is a numerical "
        "observation, not a discovered mechanism or a repair."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
