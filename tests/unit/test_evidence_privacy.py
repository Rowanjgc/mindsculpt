"""Privacy of investigation evidence in case exports.

The marker is embedded at several depths: trace text, capture metadata, a
hypothesis, an evidence title and summary, and a detector finding.  A redacted
export must not contain it anywhere, and a private export must stay complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wsp.core.errors import SafetyError
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace

MARKER = "CANARY-4417-internal-only"


def _record(service: ToolService, text: str) -> str:
    envelope = service.call(
        "record_trace",
        {
            "input_text": text,
            "params": {"max_new_tokens": 8},
            "backend": {"kind": "synthetic", "synthetic": {"model": {}}},
        },
    )
    assert envelope["ok"], envelope
    return envelope["data"]["trace"]["trace_id"]


def _capture(service: ToolService, trace_id: str, *, positions=(3, 4)) -> dict:
    envelope = service.call(
        "capture_trace_activations",
        {
            "trace_id": trace_id,
            "model": {"kind": "synthetic"},
            "modules": ["blocks.1"],
            "positions": list(positions),
            "mode": "selected",
        },
    )
    assert envelope["ok"], envelope
    return envelope["data"]


def _case_with_evidence(service: ToolService) -> dict[str, str]:
    observation = _record(service, f"the cat sat on the mat {MARKER}")
    control = _record(service, "the dog sat on the mat")
    left = _capture(service, observation)
    right = _capture(service, control)
    comparison = service.call(
        "compare_activation_captures",
        {
            "left_capture_artifact_id": left["capture_artifact_id"],
            "right_capture_artifact_id": right["capture_artifact_id"],
        },
    )
    assert comparison["ok"], comparison

    case = service.call(
        "create_failure_case",
        {
            "title": f"loop versus stop {MARKER}",
            "observations": [observation],
            "controls": [control],
        },
    )
    assert case["ok"], case
    case_id = case["data"]["case"]["case_id"]

    evidence = service.call(
        "attach_investigation_evidence",
        {
            "case_id": case_id,
            "title": f"internal comparison {MARKER}",
            "summary": f"same prompt, different continuation {MARKER}",
            "observation_trace_id": observation,
            "control_trace_ids": [control],
            "capture_artifact_ids": [
                left["capture_artifact_id"],
                right["capture_artifact_id"],
            ],
            "comparison_artifact_ids": [comparison["data"]["comparison_artifact_id"]],
            "hypotheses": [
                {
                    "statement": f"the stop token matters {MARKER}",
                    "rationale": f"because {MARKER}",
                }
            ],
            "metadata": {"note": f"private note {MARKER}"},
        },
    )
    assert evidence["ok"], evidence
    return {
        "case_id": case_id,
        "evidence_artifact_id": evidence["data"]["evidence_artifact_id"],
        "comparison_artifact_id": comparison["data"]["comparison_artifact_id"],
        "left_capture_artifact_id": left["capture_artifact_id"],
        "right_capture_artifact_id": right["capture_artifact_id"],
        "left_value_artifact_id": left["measurements"][0]["value_artifact_id"],
    }


def _export(service: ToolService, case_id: str, output: Path, policy: dict) -> dict:
    envelope = service.call(
        "export_failure_case",
        {"case_id": case_id, "output_path": str(output), "policy": policy},
    )
    return envelope


def _decode(entry: dict) -> dict:
    import base64

    if "data_base64" in entry:
        return json.loads(base64.b64decode(entry["data_base64"]))
    return entry["payload"]


def test_redacted_export_removes_markers_from_investigation_evidence(service, tmp_path):
    ids = _case_with_evidence(service)
    output = tmp_path / "case.json"
    envelope = _export(
        service,
        ids["case_id"],
        output,
        {"mode": "redacted", "sensitive_markers": [MARKER]},
    )
    assert envelope["ok"], envelope

    text = output.read_text()
    assert MARKER not in text
    bundle = json.loads(text)

    assert bundle["privacy"]["mode"] == "redacted"
    assert bundle["privacy"]["leak_scan"] == "clean"
    assert bundle["privacy"]["replay_capability"] == "unavailable"
    assert any("retained numeric" in item for item in bundle["privacy"]["limitations"])

    kinds = {entry["kind"] for entry in bundle["investigation_artifacts"]}
    assert kinds == {"trace_capture", "capture_comparison", "investigation_evidence"}

    evidence = next(
        _decode(entry)
        for entry in bundle["investigation_artifacts"]
        if entry["kind"] == "investigation_evidence"
    )
    assert evidence["title"] == "[redacted]"
    assert evidence["summary"] is None
    assert evidence["alignments"] == []
    assert evidence["spans"] == []
    assert all(replay["plan_artifact_id"] is None for replay in evidence["replays"])
    assert all(replay["result_artifact_id"] is None for replay in evidence["replays"])
    assert evidence["hypotheses"][0]["statement"] == "[redacted]"
    assert evidence["hypotheses"][0]["rationale"] is None
    assert evidence["checksums"] == {}
    assert any("retained activation vectors were removed" in item for item in evidence["limitations"])

    captures = [
        _decode(entry)
        for entry in bundle["investigation_artifacts"]
        if entry["kind"] == "trace_capture"
    ]
    for capture in captures:
        assert capture["model_name"] == "[redacted]"
        assert capture["checksums"] == {}
        assert capture["alignment_artifact_id"] is None
        assert capture["replay_artifact_id"] is None
        assert capture["resources"]["artifacts"] == 0
        assert capture["resources"]["elements_retained"] == 0
        for measurement in capture["measurements"]:
            assert measurement["retained"] is False
            assert measurement["value_artifact_id"] is None
            assert measurement["value_sha256"] is None
            assert measurement["values"] is None
            assert measurement["retained_numel"] == 0
            # Summary statistics survive: they are the measurement.
            assert measurement["norm"] is not None
        assert capture["replay_verification"]["agreement"]["per_token"] == []

    comparison = next(
        _decode(entry)
        for entry in bundle["investigation_artifacts"]
        if entry["kind"] == "capture_comparison"
    )
    assert comparison["direction"] is None
    assert comparison["counts"]["compared"] >= 1
    for entry in comparison["entries"]:
        assert all(metric["value"] is not None for metric in entry["metrics"])

    # The retained activation vectors are dropped and recorded as such.
    assert bundle["artifact_map"][ids["left_value_artifact_id"]] is None


def test_redacted_evidence_round_trips_into_a_fresh_workspace(service, tmp_path):
    ids = _case_with_evidence(service)
    output = tmp_path / "case.json"
    envelope = _export(service, ids["case_id"], output, {"mode": "redacted"})
    assert envelope["ok"], envelope

    fresh = Workspace(tmp_path / "fresh", create=True, name="fresh")
    target = ToolService(workspace=fresh, raise_errors=True)
    imported = target.call("import_failure_case", {"path": str(output)})
    assert imported["ok"], imported

    case = target.call("inspect_failure_case", {"case_id": ids["case_id"]})["data"]["case"]
    assert len(case["investigations"]) == 1
    link = case["investigations"][0]

    stored = target.call(
        "inspect_investigation_evidence", {"artifact_id": link["artifact_id"]}
    )["data"]
    assert stored["integrity"]["checksum_present"] is False
    assert stored["integrity"]["checksum_matches"] is None
    assert stored["integrity"]["missing_artifacts"] == []
    assert len(stored["capture_records"]) == 2
    assert stored["capture_records"][0]["measurement_total"] >= 1
    assert stored["comparison_records"][0]["counts"]["compared"] >= 1
    assert MARKER not in json.dumps(stored)


def test_private_export_keeps_evidence_intact(service, tmp_path):
    ids = _case_with_evidence(service)
    output = tmp_path / "case.json"
    envelope = _export(service, ids["case_id"], output, {"mode": "private"})
    assert envelope["ok"], envelope

    text = output.read_text()
    assert MARKER in text
    bundle = json.loads(text)
    assert bundle["artifact_map"] == {}
    kinds = {entry["kind"] for entry in bundle["investigation_artifacts"]}
    assert "activation_vector" in kinds
    assert "token_alignment" in kinds
    assert "replay_result" in kinds

    fresh = Workspace(tmp_path / "fresh", create=True, name="fresh")
    target = ToolService(workspace=fresh, raise_errors=True)
    imported = target.call("import_failure_case", {"path": str(output)})
    assert imported["ok"], imported

    case = target.call("inspect_failure_case", {"case_id": ids["case_id"]})["data"]["case"]
    stored = target.call(
        "inspect_investigation_evidence",
        {"artifact_id": case["investigations"][0]["artifact_id"]},
    )["data"]
    assert stored["integrity"]["checksum_matches"] is True
    assert stored["integrity"]["missing_artifacts"] == []
    assert stored["capture_records"][0]["mode"] == "selected"

    # The retained vectors survive a private round trip byte for byte.
    capture = target.workspace.artifacts.read_json(
        stored["capture_records"][0]["artifact_id"]
    )
    measurement = capture["measurements"][0]
    vector = target.workspace.artifacts.read_array(measurement["value_artifact_id"])
    original = service.workspace.artifacts.read_array(ids["left_value_artifact_id"])
    assert vector.shape == original.shape
    assert (vector == original).all()


def test_declared_marker_in_a_structural_field_still_blocks_export(service, tmp_path):
    """A marker the policy was told about must never survive, even in an identifier."""

    observation = _record(service, "the cat sat on the mat")
    left = _capture(service, observation)
    case = service.call(
        "create_failure_case",
        {"title": "marker case", "observations": [observation]},
    )["data"]["case"]
    attached = service.call(
        "attach_investigation_evidence",
        {
            "case_id": case["case_id"],
            "title": "evidence",
            "observation_trace_id": observation,
            "capture_artifact_ids": [left["capture_artifact_id"]],
            # `proposed_by` is treated as an identifier, not free text.
            "hypotheses": [{"statement": "a claim", "proposed_by": MARKER}],
        },
    )
    assert attached["ok"], attached

    output = tmp_path / "case.json"
    with pytest.raises(SafetyError) as error:
        _export(
            service,
            case["case_id"],
            output,
            {"mode": "redacted", "sensitive_markers": [MARKER]},
        )
    assert "redacted case export" in str(error.value)
    assert not output.exists()
