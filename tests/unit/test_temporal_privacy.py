"""Privacy of temporal intervention evidence.

V1.4 adds six artifact kinds and five evidence reference lists.  A redacted
export must transform every one of them, and it must fail closed when a payload
does not match the schema its transformer was written against.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from wsp.core.errors import SafetyError
from wsp.core.json import canonical_bytes
from wsp.tools.service import ToolService
from wsp.workspace.privacy import (
    ExportPolicy,
    RemovedText,
    transform_artifact,
)
from wsp.workspace.store import Workspace

MARKER = "CANARY-8891-temporal-only"

SOURCE = {
    "kind": "synthetic",
    "synthetic": {"name": "tiny", "model": {"seed": 7, "model_dim": 32, "num_layers": 3, "max_seq_len": 96}},
}
TOKENS = [1, 10, 11, 12]


def _spec(module: str, *, label: str = "spec") -> dict:
    from support.temporal import make_spec

    return make_spec(module, prompt_length=len(TOKENS), label=label).model_dump(mode="json")


def _plan(service: ToolService, modules: list[str], *, staged: bool = False) -> dict:
    envelope = service.call(
        "create_intervention_plan",
        {
            "name": f"scheduled plan {MARKER}",
            "notes": f"internal reasoning {MARKER}",
            "provenance": {"author": f"investigator {MARKER}"},
            "prompt_token_ids": TOKENS,
            "execution_mode": "staged-generation" if staged else "uncached-generation",
            "max_new_tokens": 4,
            "interventions": [
                {
                    "intervention": _spec(module, label=f"label {MARKER} {index}"),
                    "label": f"site {MARKER} {index}",
                    "order": index,
                    "schedule": [1] if staged else [1, 2],
                    "position_mode": "last",
                }
                for index, module in enumerate(modules)
            ],
        },
    )
    assert envelope["ok"], envelope
    return envelope["data"]


def _temporal_case(service: ToolService) -> dict:
    plan = _plan(service, ["blocks.1"])
    run = service.call(
        "run_temporal_intervention",
        {
            "plan_artifact_id": plan["artifact_id"],
            "source": SOURCE,
            "token_ids": TOKENS,
            "save": True,
        },
    )
    assert run["ok"], run

    plan_a = _plan(service, ["blocks.1"])
    plan_b = _plan(service, ["blocks.2"])
    plan_ab = service.call(
        "create_intervention_plan",
        {
            "name": f"combined plan {MARKER}",
            "prompt_token_ids": TOKENS,
            "max_new_tokens": 4,
            "interventions": [
                {"intervention": _spec("blocks.1"), "order": 0, "schedule": [1], "position_mode": "last"},
                {"intervention": _spec("blocks.2"), "order": 1, "schedule": [1], "position_mode": "last"},
            ],
        },
    )
    assert plan_ab["ok"], plan_ab
    composition = service.call(
        "run_compositional_comparison",
        {
            "plan_a_artifact_id": plan_a["artifact_id"],
            "plan_b_artifact_id": plan_b["artifact_id"],
            "plan_ab_artifact_id": plan_ab["data"]["artifact_id"],
            "source": SOURCE,
            "token_ids": TOKENS,
        },
    )
    assert composition["ok"], composition

    staged_plan = _plan(service, ["blocks.1"], staged=True)
    staged = service.call(
        "run_staged_intervention",
        {
            "plan_artifact_id": staged_plan["artifact_id"],
            "source": SOURCE,
            "token_ids": TOKENS,
        },
    )
    assert staged["ok"], staged

    trace = service.call(
        "record_trace",
        {
            "input_text": f"the loop begins here {MARKER}",
            "params": {"max_new_tokens": 8},
            "backend": {"kind": "synthetic", "synthetic": {"model": {}}},
        },
    )
    assert trace["ok"], trace
    trace_id = trace["data"]["trace"]["trace_id"]

    case = service.call(
        "create_failure_case",
        {"title": f"temporal loop {MARKER}", "observations": [trace_id]},
    )
    assert case["ok"], case
    case_id = case["data"]["case"]["case_id"]

    evidence = service.call(
        "attach_temporal_evidence",
        {
            "case_id": case_id,
            "title": f"temporal evidence {MARKER}",
            "summary": f"scheduled ablations {MARKER}",
            "observation_trace_id": trace_id,
            "plan_artifact_ids": [
                plan["artifact_id"],
                plan_a["artifact_id"],
                plan_b["artifact_id"],
                plan_ab["data"]["artifact_id"],
                staged_plan["artifact_id"],
            ],
            "event_log_artifact_ids": list(run["data"]["event_log_artifact_ids"].values())
            + list(composition["data"]["event_log_artifact_ids"].values())
            + list(staged["data"]["event_log_artifact_ids"].values()),
            "run_artifact_ids": [run["data"]["artifact_id"]],
            "composition_artifact_ids": [composition["data"]["artifact_id"]],
            "staged_artifact_ids": [staged["data"]["artifact_id"]],
            "verification_artifact_ids": [staged["data"]["verification_artifact_id"]],
            "hypotheses": [
                {
                    "statement": f"blocks.1 matters at step 2 {MARKER}",
                    "rationale": f"because the logits moved {MARKER}",
                }
            ],
            "notes": [f"temporal note {MARKER}"],
            "metadata": {"note": f"internal {MARKER}"},
        },
    )
    assert evidence["ok"], evidence
    return {
        "case_id": case_id,
        "evidence_artifact_id": evidence["data"]["evidence_artifact_id"],
        "plan_artifact_id": plan["artifact_id"],
        "run_artifact_id": run["data"]["artifact_id"],
        "composition_artifact_id": composition["data"]["artifact_id"],
        "staged_artifact_id": staged["data"]["artifact_id"],
        "verification_artifact_id": staged["data"]["verification_artifact_id"],
    }


def _export(service: ToolService, case_id: str, output: Path, policy: dict) -> dict:
    return service.call(
        "export_failure_case",
        {"case_id": case_id, "output_path": str(output), "policy": policy},
    )


def _decode(entry: dict) -> dict:
    if "data_base64" in entry:
        return json.loads(base64.b64decode(entry["data_base64"]))
    return entry["payload"]


def _by_kind(bundle: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for entry in bundle["investigation_artifacts"]:
        grouped.setdefault(entry["kind"], []).append(_decode(entry))
    return grouped


def test_redacted_export_removes_temporal_markers(service, tmp_path):
    ids = _temporal_case(service)
    output = tmp_path / "case.json"
    envelope = _export(service, ids["case_id"], output, {"mode": "redacted"})
    assert envelope["ok"], envelope

    text = output.read_text()
    assert MARKER not in text
    bundle = json.loads(text)
    assert bundle["privacy"]["leak_scan"] == "clean"
    assert bundle["privacy"]["mode"] == "redacted"

    grouped = _by_kind(bundle)
    assert {
        "intervention_plan",
        "intervention_event_log",
        "temporal_intervention",
        "compositional_comparison",
        "staged_continuation",
        "investigation_evidence",
    } <= set(grouped)

    for plan_payload in grouped["intervention_plan"]:
        assert plan_payload["name"] == "[redacted]"
        assert plan_payload["prompt_token_ids"] == []
        assert plan_payload["prompt_sha256"] is None
        assert plan_payload["notes"] in (None, "", "[redacted]")
        assert plan_payload["checksums"] == {}
        # Target structure survives: the schedule is the instrument reading.
        for item in plan_payload["interventions"]:
            assert item["label"] is None or MARKER not in str(item["label"])
            assert item["intervention"]["schema_version"] == "wsp.activation-intervention/1.0"
            assert item["intervention"]["target"]["module"].startswith("blocks.")
            assert item["schedule"]["kind"] in ("generation-steps", "generation-range")

    logs = grouped["intervention_event_log"]
    assert any(log["events"] for log in logs), "firing structure must survive redaction"
    for log in logs:
        for event in log["events"]:
            assert event["module"].startswith("blocks.")
            assert event["generation_step"] >= 0
            assert event["label"] is None or MARKER not in str(event["label"])
        assert log["checksums"] == {}

    for run in grouped["temporal_intervention"]:
        for condition in run["conditions"]:
            assert condition["generated_token_ids"] == []
            assert condition["generated_text"] is None
            assert condition["top1_token_ids"] == []
        assert run["prompt_sha256"] is None
        assert run["validity"]["status"] == "valid"
        assert run["identity_control"]["agreement"] == "exact"

    for composition in grouped["compositional_comparison"]:
        assert composition["metric"]
        for effect in composition["pairwise"]:
            for step in effect["stepwise"]:
                assert step["token_uncached"] is None
                assert step["token_staged"] is None

    for staged in grouped["staged_continuation"]:
        assert staged["conditions"]
        assert all(item["generated_token_ids"] == [] for item in staged["conditions"])
        assert staged["verification"]["agreement"] in ("within_tolerance", "exact")
        assert staged["performance"]["total_ms"] is not None

    evidence = grouped["investigation_evidence"][0]
    assert evidence["title"] == "[redacted]"
    assert evidence["summary"] is None
    assert evidence["hypotheses"][0]["statement"] == "[redacted]"
    assert evidence["hypotheses"][0]["rationale"] is None
    assert all(note in ("[redacted]", "") for note in evidence["notes"])
    assert evidence["checksums"] == {}
    assert len(evidence["plans"]) == 5
    assert evidence["event_logs"]
    assert len(evidence["temporal_runs"]) == 1
    assert len(evidence["compositions"]) == 1
    assert len(evidence["staged_continuations"]) == 2  # verification + staged run


def test_redacted_temporal_export_reimports_with_intact_references(service, tmp_path):
    ids = _temporal_case(service)
    output = tmp_path / "case.json"
    envelope = _export(service, ids["case_id"], output, {"mode": "redacted"})
    assert envelope["ok"], envelope

    fresh = Workspace(tmp_path / "fresh", create=True, name="fresh")
    target = ToolService(workspace=fresh, raise_errors=True)
    imported = target.call("import_failure_case", {"path": str(output)})
    assert imported["ok"], imported

    case = target.call("inspect_failure_case", {"case_id": ids["case_id"]})["data"]["case"]
    link = case["investigations"][0]
    assert link["plan_ids"] or link["temporal_run_ids"]

    stored = target.call(
        "inspect_investigation_evidence", {"artifact_id": link["artifact_id"]}
    )["data"]
    assert stored["integrity"]["checksum_present"] is False
    assert stored["integrity"]["missing_artifacts"] == []
    assert len(stored["plans"]) == 5
    assert stored["temporal_runs"][0]["validity_status"] == "valid"
    assert stored["temporal_runs"][0]["events_missing"] == 0
    assert all(ref["agreement"] != "failed" for ref in stored["staged_continuations"])
    assert MARKER not in json.dumps(stored)

    # Every referenced temporal artifact resolves to a stored artifact.
    for ref in (
        stored["plans"]
        + stored["event_logs"]
        + stored["temporal_runs"]
        + stored["compositions"]
        + stored["staged_continuations"]
    ):
        fresh.artifacts.get(ref["artifact_id"])


def test_private_export_keeps_temporal_evidence_intact(service, tmp_path):
    ids = _temporal_case(service)
    output = tmp_path / "case.json"
    envelope = _export(service, ids["case_id"], output, {"mode": "private"})
    assert envelope["ok"], envelope

    text = output.read_text()
    assert MARKER in text
    bundle = json.loads(text)
    assert bundle["artifact_map"] == {}

    grouped = _by_kind(bundle)
    run = grouped["temporal_intervention"][0]
    assert run["conditions"][0]["generated_token_ids"]
    evidence = grouped["investigation_evidence"][0]
    assert evidence["checksums"]
    assert evidence["title"].startswith("temporal evidence")
    assert evidence["hypotheses"][0]["rationale"].startswith("because the logits moved")


def test_temporal_redaction_fails_closed_on_unknown_fields(service, tmp_path):
    ids = _temporal_case(service)
    payload = service.workspace.artifacts.read_json(ids["plan_artifact_id"])
    payload["unexpected_field"] = f"unscanned text {MARKER}"
    data = canonical_bytes(payload)

    policy = ExportPolicy(mode="redacted")
    with pytest.raises(SafetyError) as excinfo:
        transform_artifact(
            kind="intervention_plan",
            format_name="json",
            data=data,
            policy=policy,
            removed=RemovedText(),
        )
    assert "does not match its schema" in str(excinfo.value)
    assert service.workspace.artifacts.read_json(ids["plan_artifact_id"])["name"]


def test_temporal_redaction_fails_closed_on_unknown_kind():
    policy = ExportPolicy(mode="redacted")
    with pytest.raises(SafetyError):
        transform_artifact(
            kind="unregistered_temporal_record",
            format_name="npy",
            data=b"\x00\x01",
            policy=policy,
            removed=RemovedText(),
        )
