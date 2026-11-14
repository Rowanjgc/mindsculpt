from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wsp.core.errors import ArtifactError, InputError, NotFoundError
from wsp.patches.operations import payload_sha256
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace


def _scripted(text: str, *, input_text: str, tokens: list[str] | None = None) -> dict:
    fixture: dict = {"output_text": text}
    if tokens is not None:
        fixture["output_tokens"] = [
            {"text": token, "token_id": index + 1} for index, token in enumerate(tokens)
        ]
    return {
        "kind": "scripted",
        "scripted": {"fixtures": [{"input_equals": input_text, **fixture}]},
    }


def _case_payload(observations: list[str], controls: list[str] | None = None) -> dict:
    return {
        "title": "Repetitive loop investigation",
        "summary": "A reproducible repeated sequence was observed in one controlled run.",
        "observations": observations,
        "controls": controls or [],
        "tags": ["repetition", "synthetic"],
        "hypotheses": [
            {
                "statement": "Hypothesis only: the loop may be associated with a decoding configuration.",
                "rationale": "The two runs differ only in the prompt.",
            }
        ],
        "annotations": [{"text": "Observed twice with the same seed.", "author": "tester"}],
        "reproduction": {
            "backend": "scripted",
            "command": "wsp inspect-trace <id>",
            "notes": "No model download is required.",
        },
    }


def _make_contrast(service) -> dict:
    payload = {
        "name": "prompt perturbation",
        "baseline": {"name": "baseline", "input_text": "healthy"},
        "variants": [{"name": "loop", "input_text": "loop"}],
        "expectation": {
            "kind": "reference_agreement",
            "description": "The answer should remain Canberra.",
            "reference_answers": ["Canberra"],
        },
        "backend": {
            "kind": "scripted",
            "scripted": {
                "fixtures": [
                    {
                        "input_equals": "healthy",
                        "output_text": "Canberra",
                        "output_tokens": [{"text": "Canberra", "token_id": 1}],
                    },
                    {
                        "input_equals": "loop",
                        "output_text": "but wait but wait but wait",
                        "output_tokens": [
                            {"text": "but", "token_id": 2},
                            {"text": " wait", "token_id": 3},
                            {"text": "but", "token_id": 2},
                            {"text": " wait", "token_id": 3},
                            {"text": "but", "token_id": 2},
                            {"text": " wait", "token_id": 3},
                        ],
                    },
                ]
            },
        },
    }
    return service.call("run_contrast", payload)["data"]


def _make_case(service, workspace, tmp_path: Path) -> tuple[str, dict, Path, str]:
    observation = service.call(
        "record_trace",
        {
            "input_text": "loop",
            "backend": {
                "kind": "scripted",
                "scripted": {
                    "default_fixture": {
                        "output_text": "but wait " * 8,
                        "output_tokens": [
                            {"text": "but", "token_id": 2},
                            {"text": " wait", "token_id": 3},
                        ]
                        * 8,
                    }
                },
            },
        },
    )["data"]["trace_id"]
    control = service.call(
        "record_trace",
        {
            "input_text": "healthy",
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
    scan = service.call(
        "scan_trace",
        {"trace_id": observation, "detectors": [{"detector_id": "core.reconsideration"}]},
    )["data"]
    contrast = _make_contrast(service)
    capture = workspace.artifacts.put_json(
        "activation_capture",
        {"schema_version": "wsp.activation-capture/1.0", "notes": "synthetic"},
    )
    delta = np.zeros((16, 8), dtype=np.float64)
    delta_ref = workspace.artifacts.put_array("patch_payload", delta)
    patch = service.call(
        "create_patch",
        {
            "path": str(tmp_path / "tiny-mlp.safetensors"),
            "target": "fc1.weight",
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
    )
    payload = _case_payload([observation], [control])
    payload.update(
        {
            "findings": scan["findings"],
            "contrasts": [
                {
                    "result_id": contrast["result_id"],
                    "artifact_id": contrast["artifact_id"],
                    "test_name": contrast["test"]["name"],
                    "overall_status": contrast["overall_status"],
                }
            ],
            "activation_capture_ids": [capture.artifact_id],
            "patches": [{"patch_id": patch["data"]["patch"]["patch_id"]}],
        }
    )
    created = service.call("create_failure_case", payload)["data"]
    return created["case"]["case_id"], created, tmp_path / "case_export.json", contrast["artifact_id"]


def test_case_creation_listing_and_inspection(service, workspace, tmp_path, mlp_checkpoint):
    # mlp_checkpoint is created in tmp_path by the fixture, matching _make_case's path.
    case_id, created, _out, _artifact = _make_case(service, workspace, tmp_path)
    case = created["case"]
    assert case_id.startswith("case_")
    assert case["review"]["status"] == "open"
    assert case["observations"] and case["controls"]
    assert case["findings"]
    assert case["hypotheses"][0]["status"] == "proposed"
    assert case["activation_captures"]
    assert case["patches"][0]["patch_id"]
    assert created["case_sha256"].startswith("sha256:")
    assert Path(created["path"]).exists()

    listed = service.call("list_failure_cases", {})["data"]
    assert listed["returned"] == 1
    assert listed["cases"][0]["observation_count"] == 1
    inspected = service.call("inspect_failure_case", {"case_id": case_id})["data"]
    assert inspected["case"]["title"] == "Repetitive loop investigation"
    assert inspected["case_sha256"] == created["case_sha256"]


def test_case_update_review_annotations_and_hypotheses(service, workspace, tmp_path, mlp_checkpoint):
    case_id, created, _out, _artifact = _make_case(service, workspace, tmp_path)
    updated = service.call(
        "update_failure_case",
        {
            "case_id": case_id,
            "review_status": "triage",
            "review_notes": "Needs a larger controlled comparison.",
            "reviewer": "reviewer-1",
            "add_annotation": {"text": "Second observation matched.", "kind": "review"},
            "add_hypothesis": {"statement": "Hypothesis: prompt-specific loop."},
        },
    )["data"]["case"]
    assert updated["revision"] == 2
    assert updated["review"]["status"] == "triage"
    assert updated["review"]["reviewers"] == ["reviewer-1"]
    assert len(updated["annotations"]) == 2
    assert len(updated["hypotheses"]) == 2
    hypothesis_id = updated["hypotheses"][1]["hypothesis_id"]
    updated_again = service.call(
        "update_failure_case",
        {
            "case_id": case_id,
            "hypothesis_updates": [
                {"hypothesis_id": hypothesis_id, "status": "testing"}
            ],
        },
    )["data"]["case"]
    assert updated_again["hypotheses"][1]["status"] == "testing"
    assert updated_again["revision"] == 3
    filtered = service.call(
        "list_failure_cases", {"status": "triage", "tag": "repetition"}
    )["data"]
    assert filtered["returned"] == 1


def test_case_export_and_reload_in_fresh_workspace(
    service, workspace, tmp_path, mlp_checkpoint
):
    case_id, created, out, contrast_artifact = _make_case(service, workspace, tmp_path)
    exported = service.call(
        "export_failure_case", {"case_id": case_id, "output_path": str(out)}
    )["data"]
    assert Path(exported["output_path"]).exists()
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "wsp.case-export/1.1"
    assert bundle["case_sha256"] == created["case_sha256"]
    assert len(bundle["traces"]) == 2
    assert bundle["contrast_results"][0]["artifact_id"] == contrast_artifact
    assert bundle["activation_captures"][0]["data_base64"]
    assert bundle["patches"][0]["kind"] == "patch"
    assert bundle["payload_artifacts"]

    fresh = ToolService(
        workspace=Workspace(tmp_path / "fresh", create=True), raise_errors=True
    )
    imported = fresh.call("import_failure_case", {"path": str(out)})["data"]
    assert imported["case"]["case_id"] == case_id
    assert fresh.call("list_traces", {})["data"]["returned"] == 2
    assert fresh.call("list_failure_cases", {})["data"]["returned"] == 1
    inspected = fresh.call("inspect_failure_case", {"case_id": case_id})["data"]
    assert inspected["case"] == created["case"]
    fresh_workspace = fresh.workspace
    assert fresh_workspace is not None
    assert fresh_workspace.artifacts.verify(contrast_artifact)
    assert fresh_workspace.load_patch(created["case"]["patches"][0]["patch_id"])
    for entry in bundle["payload_artifacts"]:
        assert fresh_workspace.artifacts.verify(entry["artifact_id"])


def test_redacted_export_removes_trace_text(
    service, workspace, tmp_path, mlp_checkpoint
):
    case_id, _created, out, _artifact = _make_case(service, workspace, tmp_path)
    result = service.call(
        "export_failure_case",
        {"case_id": case_id, "output_path": str(out), "redact_text": True},
    )
    assert result["data"]["redacted"] is True
    bundle = json.loads(out.read_text(encoding="utf-8"))
    for entry in bundle["traces"]:
        trace = entry["trace"]
        assert trace["input"]["text"] is None
        assert trace["text"] is None
        assert all(token["text"] is None for token in trace["tokens"])
        assert trace["privacy"]["redacted"] is True
    assert bundle["privacy"]["mode"] == "redacted"
    assert bundle["privacy"]["limitations"]
    assert bundle["case"]["redaction"]["leak_scan"] == "clean"
    assert any("redacted export" in warning for warning in bundle["warnings"])


def test_export_checksum_tampering_is_rejected(
    service, workspace, tmp_path, mlp_checkpoint
):
    case_id, _created, out, _artifact = _make_case(service, workspace, tmp_path)
    service.call("export_failure_case", {"case_id": case_id, "output_path": str(out)})
    bundle = json.loads(out.read_text(encoding="utf-8"))
    bundle["case"]["title"] = "tampered"
    out.write_text(json.dumps(bundle), encoding="utf-8")
    fresh = ToolService(
        workspace=Workspace(tmp_path / "fresh2", create=True), raise_errors=True
    )
    with pytest.raises(ArtifactError):
        fresh.call("import_failure_case", {"path": str(out)})


def test_case_rejects_missing_or_wrong_references(service, workspace, tmp_path, mlp_checkpoint):
    with pytest.raises(NotFoundError):
        service.call(
            "create_failure_case",
            {
                "title": "missing trace",
                "observations": ["tr_0000000000000000"],
            },
        )
    note = workspace.artifacts.put_json("notes", {"text": "not a capture"})
    with pytest.raises(InputError):
        service.call(
            "create_failure_case",
            {
                "title": "wrong artifact kind",
                "observations": [],
                "findings": [],
                "activation_capture_ids": [note.artifact_id],
            },
        )
