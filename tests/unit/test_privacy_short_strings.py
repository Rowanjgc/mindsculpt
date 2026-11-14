"""M0 regression tests: short and non-ASCII sensitive markers.

V1.2 removed short strings from redacted exports but deliberately skipped them
in the post-serialisation leak scan, because a three-character substring matches
unrelated structural text.  V1.3 closes that gap:

* every explicitly declared marker is searched as a substring at any length,
* short *removed* values are checked by exact equality inside decoded JSON
  values and keys (and by a delimiter-anchored pattern outside JSON),
* values are still removed structurally, so the scan is a second line of
  defence rather than the only one.

These tests keep both halves honest: no known marker survives, and the export
refuses when its policy cannot prove the guarantee.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from wsp.core.errors import SafetyError
from wsp.tools.service import ToolService
from wsp.workspace.privacy import (
    ExportPolicy,
    RemovedText,
    enforce_no_leaks,
    scan_for_leaks,
    transform_artifact,
)
from wsp.workspace.store import Workspace

SHORT = "Zq7"
UNICODE = "秘密昵称"
LONG = "CANARY-8871-do-not-export"


def _case_with_markers(service: ToolService, workspace: Workspace, tmp_path: Path) -> str:
    """A case whose text and nested artifacts carry every marker shape."""

    observation = service.call(
        "record_trace",
        {
            "input_text": f"{SHORT} {UNICODE} {LONG} please repeat",
            "system": f"you are {UNICODE}",
            "backend": {
                "kind": "scripted",
                "scripted": {
                    "default_fixture": {
                        "output_text": f"{SHORT} but wait",
                        "output_tokens": [
                            {"text": f"{SHORT}", "token_id": 2},
                            {"text": UNICODE, "token_id": 3},
                        ],
                        "tool_calls": [
                            {
                                "index": 0,
                                "name": "lookup",
                                "arguments": {"query": f"{SHORT} argument"},
                                "result_summary": f"{UNICODE} result",
                            }
                        ],
                    }
                },
            },
        },
    )["data"]["trace_id"]
    control = service.call(
        "record_trace",
        {
            "input_text": "healthy prompt",
            "backend": {
                "kind": "scripted",
                "scripted": {"default_fixture": {"output_text": "Canberra"}},
            },
        },
    )["data"]["trace_id"]

    # A nested capture artifact: marker in a nested list, a mapping key, and a
    # base64-encoded inner record.
    nested = {
        "schema_version": "wsp.activation-capture/1.0",
        "summaries": [
            {
                "module": "layer.0",
                "note": f"{SHORT} nested {UNICODE}",
                f"{SHORT} key": f"{UNICODE} value",
                "encoded": base64.b64encode(
                    json.dumps({"inner": f"{SHORT}{UNICODE}"}).encode()
                ).decode(),
            }
        ],
        "notes": f"{LONG} top-level",
    }
    capture = workspace.artifacts.put_json("activation_capture", nested)

    created = service.call(
        "create_failure_case",
        {
            "title": f"{SHORT} title {UNICODE}",
            "summary": f"{LONG} summary",
            "observations": [observation],
            "controls": [control],
            "findings": [
                {
                    "detector_id": "manual.finding",
                    "detector_version": "1",
                    "kind": "manual",
                    "severity": "medium",
                    "explanation": f"{SHORT} explanation",
                    "value": {f"{SHORT} key": UNICODE, "count": 1},
                    "parameters": {"note": f"{UNICODE} parameter"},
                    "metadata": {"origin": f"{SHORT} metadata"},
                }
            ],
            "annotations": [{"text": f"{SHORT} annotation", "author": UNICODE}],
            "hypotheses": [
                {"statement": f"{UNICODE} hypothesis", "rationale": f"{SHORT} rationale"}
            ],
            "activation_capture_ids": [capture.artifact_id],
            "tags": [SHORT],
            "metadata": {
                "source_path": f"/home/user/{SHORT}/{UNICODE}.json",
                "counter": 3,
            },
        },
    )["data"]
    return created["case"]["case_id"]


def _survives(data: bytes) -> list[str]:
    return [marker for marker in (SHORT, UNICODE, LONG) if marker.encode() in data]


def test_redacted_export_removes_short_and_unicode_markers(
    service: ToolService, workspace: Workspace, tmp_path: Path
):
    case_id = _case_with_markers(service, workspace, tmp_path)
    out = tmp_path / "redacted.json"
    result = service.call(
        "export_failure_case",
        {
            "case_id": case_id,
            "output_path": str(out),
            "policy": {
                "mode": "redacted",
                "sensitive_markers": [SHORT, UNICODE, LONG],
            },
        },
    )["data"]

    data = out.read_bytes()
    assert _survives(data) == []
    assert result["privacy"]["leak_scan"] == "clean"
    # The original workspace still holds the private artifacts untouched.
    assert any(
        SHORT.encode() in path.read_bytes()
        for path in workspace.root.rglob("*")
        if path.is_file()
    )


def test_nested_and_base64_payloads_are_scrubbed(
    service: ToolService, workspace: Workspace, tmp_path: Path
):
    case_id = _case_with_markers(service, workspace, tmp_path)
    out = tmp_path / "redacted.json"
    service.call(
        "export_failure_case",
        {
            "case_id": case_id,
            "output_path": str(out),
            "policy": {"mode": "redacted", "sensitive_markers": [SHORT, UNICODE]},
        },
    )
    bundle = json.loads(out.read_text(encoding="utf-8"))
    for entry in bundle["activation_captures"]:
        payload = json.loads(base64.b64decode(entry["data_base64"]))
        text = json.dumps(payload, ensure_ascii=False)
        assert SHORT not in text
        assert UNICODE not in text
        # The base64-encoded inner record was dropped as an opaque string.
        assert payload["summaries"][0]["encoded"] == "[redacted]"


def test_explicit_short_marker_is_searched_as_a_substring():
    with pytest.raises(SafetyError) as excinfo:
        enforce_no_leaks("a cpu-bound export", [], markers=["cpu"], what="test export")
    assert "cpu" not in str(excinfo.value)
    assert "test export" in str(excinfo.value)


def test_short_removed_value_is_found_by_exact_equality_inside_json():
    text = json.dumps({"note": "A&B", "other": "xA&By"}, ensure_ascii=False)
    assert scan_for_leaks(text, ["A&B"])
    # An unrelated longer string that merely contains the short value is not a
    # match, which is what keeps the scan from crying wolf.
    assert scan_for_leaks(json.dumps({"note": "xA&By"}), ["A&B"]) == []


def test_short_structural_needle_is_only_checked_when_declared():
    # Auto-collected needles that look like identifiers are skipped on purpose.
    assert scan_for_leaks(json.dumps({"dtype": "float32"}), ["float32"]) == []
    # Callers who really mean it must declare the marker explicitly.
    assert scan_for_leaks(json.dumps({"dtype": "float32"}), [], markers=["float32"])


def test_non_json_short_values_use_a_delimiter_anchored_pattern():
    with pytest.raises(SafetyError):
        enforce_no_leaks("value=A&B; next", ["A&B"], what="test export")
    with pytest.raises(SafetyError):
        enforce_no_leaks("value=AB; next", [], markers=["AB"], what="test export")


def test_declared_short_structural_marker_in_an_artifact_refuses_the_export(
    service: ToolService, workspace: Workspace, tmp_path: Path
):
    case_id = _case_with_markers(service, workspace, tmp_path)
    # ``count`` survives as a structural key, so declaring it sensitive must
    # fail closed rather than ship it.
    with pytest.raises(SafetyError):
        service.call(
            "export_failure_case",
            {
                "case_id": case_id,
                "output_path": str(tmp_path / "refused.json"),
                "policy": {"mode": "redacted", "sensitive_markers": ["count"]},
            },
        )
    assert not (tmp_path / "refused.json").exists()


def test_policy_documents_the_short_string_guarantee():
    limitations = ExportPolicy(mode="redacted").limitations()
    joined = " ".join(limitations)
    assert "short" in joined
    assert "exact equality" in joined
    assert "marker" in joined


def test_intervention_artifact_transform_drops_donor_values():
    """New V1.3 artifacts go through the same registry, not a second exporter."""

    policy = ExportPolicy(mode="redacted")
    removed = RemovedText()
    payload = {
        "schema_version": "wsp.activation-intervention/1.0",
        "intervention_id": "iv_0123456789abcdef",
        "created_at": "2026-01-01T00:00:00Z",
        "name": f"{SHORT} donor patch",
        "model": {"name": f"{SHORT} model", "checkpoint_path": f"/tmp/{SHORT}.safetensors"},
        "execution": {"sequence_length": 4, "dtype": "float32"},
        "target": {
            "module": "blocks.1",
            "position_semantics": "consuming",
            "selector": {"positions": [1]},
        },
        "operation": "patch",
        "donor": {
            "module": "blocks.1",
            "values": [1.0, 2.0, 3.0],
            "shape": [1, 3],
            "dtype": "float32",
            "note": f"{SHORT} donor note",
        },
        "notes": f"{UNICODE} intervention note",
        "checksums": {"spec_sha256": "abc"},
    }
    kind, fmt, data = transform_artifact(
        kind="activation_intervention",
        format_name="json",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        policy=policy,
        removed=removed,
    )
    assert kind == "activation_intervention"
    assert fmt == "json"
    text = data.decode()
    assert SHORT not in text
    assert UNICODE not in text
    decoded = json.loads(text)
    # Structure survives; private values do not.
    assert decoded["operation"] == "patch"
    assert decoded["target"]["module"] == "blocks.1"
    assert decoded["donor"]["values"] is None
    assert decoded["checksums"] == {}
    # The removed strings were recorded, so the outer export can scan them.
    assert any(SHORT in value for value in removed.values)


def test_intervention_transform_rejects_a_payload_that_does_not_match_schema():
    with pytest.raises(SafetyError):
        transform_artifact(
            kind="activation_intervention",
            format_name="json",
            data=json.dumps({"intervention_id": f"iv_{SHORT}", "notes": SHORT}).encode(),
            policy=ExportPolicy(mode="redacted"),
            removed=RemovedText(),
        )
