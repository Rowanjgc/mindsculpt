"""Privacy regression tests for redacted exports.

The marker is embedded at every serialisation depth the case model can reach:
trace text, token text, tool calls, prompt metadata, detector values, case prose,
annotations, hypotheses, tags, metadata, absolute paths, filenames, contrast
inputs and outputs, and patch provenance.  A redacted export must not contain it
anywhere, including inside base64 artifact payloads and nested JSON strings.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from wsp.core.errors import InputError, SafetyError
from wsp.patches.operations import payload_sha256
from wsp.tools.service import ToolService
from wsp.workspace.privacy import ExportPolicy, RemovedText, transform_artifact
from wsp.workspace.store import Workspace

MARKER = "CANARY-7731-do-not-export"


def _marker_count(data: bytes) -> int:
    return data.count(MARKER.encode("utf-8"))


def _checkpoint(tmp_path: Path, name: str, model) -> Path:
    from safetensors.torch import save_file

    path = tmp_path / name
    save_file(
        {
            key: value.detach().to("cpu").contiguous()
            for key, value in model.state_dict().items()
        },
        str(path),
    )
    return path


def _rich_case(service, workspace, tmp_path: Path) -> tuple[str, Path]:
    checkpoint = _checkpoint(tmp_path, f"{MARKER}-checkpoint.safetensors", _tiny_mlp())

    observation = service.call(
        "record_trace",
        {
            "input_text": f"{MARKER} please loop forever",
            "system": f"you are {MARKER}",
            "backend": {
                "kind": "scripted",
                "scripted": {
                    "default_fixture": {
                        "output_text": f"{MARKER} but wait but wait",
                        "output_tokens": [
                            {"text": f"{MARKER}", "token_id": 2},
                            {"text": " wait", "token_id": 3},
                        ],
                        "tool_calls": [
                            {
                                "index": 0,
                                "name": "lookup",
                                "arguments": {"query": f"{MARKER} argument"},
                                "result_summary": f"{MARKER} result",
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
                "scripted": {
                    "default_fixture": {
                        "output_text": "Canberra",
                        "output_tokens": [{"text": "Canberra", "token_id": 1}],
                    }
                },
            },
        },
    )["data"]["trace_id"]

    contrast = service.call(
        "run_contrast",
        {
            "name": f"{MARKER} contrast",
            "baseline": {"name": "baseline", "input_text": f"{MARKER} baseline input"},
            "variants": [
                {"name": "marked_variant", "input_text": f"{MARKER} variant input"}
            ],
            "expectation": {
                "kind": "reference_agreement",
                "description": f"{MARKER} expectation",
                "reference_answers": [f"{MARKER} answer"],
            },
            "backend": {
                "kind": "scripted",
                "scripted": {
                    "fixtures": [
                        {
                            "input_equals": f"{MARKER} baseline input",
                            "output_text": f"{MARKER} baseline output",
                            "output_tokens": [{"text": f"{MARKER}", "token_id": 4}],
                        },
                        {
                            "input_equals": f"{MARKER} variant input",
                            "output_text": f"{MARKER} variant output",
                            "output_tokens": [{"text": f"{MARKER}", "token_id": 5}],
                        },
                    ]
                },
            },
        },
    )["data"]

    capture = workspace.artifacts.put_json(
        "activation_capture",
        {
            "schema_version": "wsp.activation-capture/1.0",
            "notes": f"{MARKER} capture note",
            "summaries": [{"module": "layer.0", "note": f"{MARKER} nested"}],
        },
    )
    delta = np.zeros((16, 8), dtype=np.float64)
    delta_ref = workspace.artifacts.put_array("patch_payload", delta)
    patch = service.call(
        "create_patch",
        {
            "path": str(checkpoint),
            "target": "fc1.weight",
            "notes": f"{MARKER} patch provenance",
            "params": {
                "kind": "add",
                "delta": {
                    "artifact_id": delta_ref.artifact_id,
                    "shape": [16, 8],
                    "dtype": "float64",
                    "numel": 128,
                    "sha256": payload_sha256(delta),
                },
            },
        },
    )["data"]["patch"]

    created = service.call(
        "create_failure_case",
        {
            "title": f"{MARKER} title",
            "summary": f"{MARKER} summary",
            "observations": [observation],
            "controls": [control],
            "findings": [
                {
                    "detector_id": "manual.finding",
                    "detector_version": "1",
                    "kind": "manual",
                    "severity": "medium",
                    "explanation": f"{MARKER} explanation",
                    "value": {f"{MARKER} key": 1, "count": f"{MARKER} count"},
                    "parameters": {"note": f"{MARKER} parameter"},
                    "limitations": [f"{MARKER} limitation"],
                    "metadata": {"origin": f"{MARKER} metadata"},
                    "evidence": [
                        {
                            "kind": "text_span",
                            "trace_id": observation,
                            "start": 0,
                            "end": 3,
                            "name": f"{MARKER} evidence",
                            "detail": f"{MARKER} detail",
                        }
                    ],
                }
            ],
            "contrasts": [
                {
                    "result_id": contrast["result_id"],
                    "artifact_id": contrast["artifact_id"],
                    "test_name": f"{MARKER} test",
                    "overall_status": contrast["overall_status"],
                }
            ],
            "annotations": [{"text": f"{MARKER} annotation", "author": f"{MARKER} author"}],
            "hypotheses": [
                {
                    "statement": f"{MARKER} hypothesis",
                    "rationale": f"{MARKER} rationale",
                }
            ],
            "activation_capture_ids": [capture.artifact_id],
            "patches": [{"patch_id": patch["patch_id"], "note": f"{MARKER} patch note"}],
            "tags": [MARKER],
            "metadata": {
                "source_path": f"/home/user/{MARKER}.json",
                "note": f"{MARKER} metadata value",
                "internal_counter_key_7731": 5,
            },
            "reproduction": {
                "backend": "scripted",
                "command": f"wsp replay --path /tmp/{MARKER}",
                "notes": f"{MARKER} reproduction notes",
                "environment": {"host": f"{MARKER} host"},
            },
        },
    )["data"]["case"]["case_id"]
    return created, tmp_path / "export.json"


def _tiny_mlp():
    from wsp.execution.builtin_models import TinyMLP

    import torch

    torch.manual_seed(5)
    model = TinyMLP(input_dim=8, hidden_dim=16, output_dim=4)
    model.eval()
    return model


def test_redacted_case_export_contains_no_marker_anywhere(
    service, workspace, tmp_path, monkeypatch
):
    case_id, out = _rich_case(service, workspace, tmp_path)
    # Sanity check: the source workspace really does contain the marker in many
    # places, otherwise the assertion below could pass trivially.
    marker_bearing = [
        path
        for path in workspace.root.rglob("*")
        if path.is_file() and _marker_count(path.read_bytes())
    ]
    assert len(marker_bearing) >= 5

    result = service.call(
        "export_failure_case",
        {
            "case_id": case_id,
            "output_path": str(out),
            "policy": {"mode": "redacted", "sensitive_markers": [MARKER]},
        },
    )["data"]

    data = out.read_bytes()
    assert _marker_count(data) == 0
    assert result["redacted"] is True
    assert result["privacy"]["replay_capability"] == "unavailable"
    assert result["privacy"]["leak_scan"] == "clean"
    assert result["privacy"]["limitations"]

    bundle = json.loads(data)
    assert bundle["schema_version"] == "wsp.case-export/1.1"
    assert bundle["privacy"]["mode"] == "redacted"
    assert bundle["case"]["redaction"]["replay_capability"] == "unavailable"
    assert bundle["case"]["title"].startswith("redacted case")
    assert bundle["case"]["summary"] is None

    # Nested JSON inside base64 artifact payloads must be transformed too.
    for section in ("contrast_results", "activation_captures", "payload_artifacts"):
        for entry in bundle[section]:
            raw = base64.b64decode(entry["data_base64"])
            assert _marker_count(raw) == 0

    # Reference integrity survives the transformation.
    case = bundle["case"]
    for reference in case["activation_captures"]:
        assert reference["artifact_id"] in bundle["artifact_map"].values()
    assert all(
        new_id is None or bundle["artifact_map"][old_id] == new_id
        for old_id, new_id in bundle["artifact_map"].items()
    )


def test_redacted_export_reimports_with_reduced_replay(service, workspace, tmp_path):
    case_id, out = _rich_case(service, workspace, tmp_path)
    service.call(
        "export_failure_case",
        {
            "case_id": case_id,
            "output_path": str(out),
            "policy": {"mode": "redacted", "sensitive_markers": [MARKER]},
        },
    )
    fresh = ToolService(
        workspace=Workspace(tmp_path / "fresh", create=True), raise_errors=True
    )
    imported = fresh.call("import_failure_case", {"path": str(out)})["data"]
    assert imported["case"]["case_id"] == case_id
    assert imported["case"]["redaction"]["replay_capability"] == "unavailable"
    assert _marker_count(out.read_bytes()) == 0

    fresh_workspace = fresh.workspace
    assert fresh_workspace is not None
    bundle = json.loads(out.read_text(encoding="utf-8"))
    for section in ("contrast_results", "activation_captures", "payload_artifacts"):
        for entry in bundle[section]:
            assert fresh_workspace.artifacts.verify(entry["artifact_id"])
    # The case still loads and validates its (rewritten) references.
    inspected = fresh.call("inspect_failure_case", {"case_id": case_id})["data"]
    assert inspected["case"]["review"]["status"] == "open"


def test_redacted_trace_keeps_token_ids_only_when_requested(service, workspace, tmp_path):
    case_id, _out = _rich_case(service, workspace, tmp_path)
    bundle = service.call(
        "export_failure_case",
        {
            "case_id": case_id,
            "output_path": str(tmp_path / "reduced.json"),
            "policy": {
                "mode": "redacted",
                "sensitive_markers": [MARKER],
                "keep_token_ids": True,
            },
        },
    )["data"]
    assert bundle["privacy"]["replay_capability"] == "reduced"
    exported = json.loads((tmp_path / "reduced.json").read_text(encoding="utf-8"))
    assert exported["case"]["redaction"]["token_ids_retained"] is True
    assert any(
        token["token_id"] is not None
        for entry in exported["traces"]
        for token in entry["trace"]["tokens"]
    )
    assert _marker_count((tmp_path / "reduced.json").read_bytes()) == 0


def test_private_export_is_byte_faithful(service, workspace, tmp_path):
    case_id, out = _rich_case(service, workspace, tmp_path)
    created = service.call("inspect_failure_case", {"case_id": case_id})["data"]
    result = service.call(
        "export_failure_case", {"case_id": case_id, "output_path": str(out)}
    )["data"]
    assert result["redacted"] is False
    assert result["privacy"]["mode"] == "private"
    assert result["case_sha256"] == created["case_sha256"]
    assert _marker_count(out.read_bytes()) > 0
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["case"]["title"].startswith(MARKER)
    assert bundle["artifact_map"] == {}


def test_undeclared_marker_in_structural_position_is_refused(
    service, workspace, tmp_path
):
    """When the marker survives only as a preserved identifier key, refuse.

    Structural mapping keys are kept on purpose so a redacted finding still has
    readable shape.  The leak scan is the backstop: if the caller declares the
    marker sensitive, the export fails loudly instead of shipping it.
    """

    case_id, _out = _rich_case(service, workspace, tmp_path)
    with pytest.raises(SafetyError) as excinfo:
        service.call(
            "export_failure_case",
            {
                "case_id": case_id,
                "output_path": str(tmp_path / "refused.json"),
                "policy": {
                    "mode": "redacted",
                    "sensitive_markers": ["internal_counter_key_7731"],
                },
            },
        )
    assert "redacted case export" in str(excinfo.value)
    assert not (tmp_path / "refused.json").exists()


def test_untracked_binary_artifact_is_rejected_unless_omitted(
    service, workspace, tmp_path
):
    case_id, _out = _rich_case(service, workspace, tmp_path)
    blob = workspace.artifacts.put_bytes(
        "activation_capture", b"\x00\x01binary-not-json", format="bin"
    )
    case = service.call("inspect_failure_case", {"case_id": case_id})["data"]["case"]
    from wsp.behavior.cases import CaseStore
    from wsp.behavior.trace import TraceStore

    store = CaseStore(workspace, traces=TraceStore(workspace))
    loaded = store.load(case["case_id"])
    loaded.activation_captures.append(blob)
    store.save(loaded)

    target = tmp_path / "rejected.json"
    with pytest.raises(SafetyError):
        service.call(
            "export_failure_case",
            {
                "case_id": case_id,
                "output_path": str(target),
                "policy": {"mode": "redacted"},
            },
        )
    assert not target.exists()

    omitted = tmp_path / "omitted.json"
    result = service.call(
        "export_failure_case",
        {
            "case_id": case_id,
            "output_path": str(omitted),
            "policy": {
                "mode": "redacted",
                "on_untransformable_artifact": "omit",
            },
        },
    )["data"]
    assert result["artifact_map"][blob.artifact_id] is None
    bundle = json.loads(omitted.read_text(encoding="utf-8"))
    assert any("omitted" in limitation for limitation in bundle["privacy"]["limitations"])


def test_transform_artifact_never_passes_untransformed_binary_data():
    policy = ExportPolicy(mode="redacted")
    removed = RemovedText()
    with pytest.raises(SafetyError):
        transform_artifact(
            kind="mystery_blob",
            format_name="bin",
            data=b"\x00secret",
            policy=policy,
            removed=removed,
        )
    # Unknown JSON artifacts keep their structure but lose every string.
    kind, fmt, data = transform_artifact(
        kind="mystery_blob",
        format_name="json",
        data=json.dumps({"input_text": "secret text", "count": 1}).encode(),
        policy=policy,
        removed=removed,
    )
    assert kind == "mystery_blob"
    assert fmt == "json"
    assert b"secret text" not in data
    assert b'"count":1' in data.replace(b" ", b"")
    # A contrast result that does not match its schema is refused, because the
    # transformer cannot prove it covered every field.
    with pytest.raises(SafetyError):
        transform_artifact(
            kind="contrast_result",
            format_name="json",
            data=json.dumps({"input_text": "secret text", "value": 1}).encode(),
            policy=policy,
            removed=RemovedText(),
        )


def test_export_policy_alias_conflict_is_rejected(service, workspace, tmp_path):
    case_id, out = _rich_case(service, workspace, tmp_path)
    with pytest.raises(InputError):
        service.call(
            "export_failure_case",
            {
                "case_id": case_id,
                "output_path": str(out),
                "redact_text": False,
                "policy": {"mode": "redacted"},
            },
        )
