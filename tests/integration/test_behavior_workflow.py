from __future__ import annotations

import json
from pathlib import Path

from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace


def _fixture(input_text: str, output_text: str, *, tokens: list[str] | None = None) -> dict:
    fixture: dict = {"input_equals": input_text, "output_text": output_text}
    if tokens is not None:
        fixture["output_tokens"] = [
            {"text": token, "token_id": index + 1} for index, token in enumerate(tokens)
        ]
    return fixture


def test_offline_behavioral_workflow_end_to_end(service, workspace: Workspace, tmp_path: Path):
    backend = {
        "kind": "scripted",
        "scripted": {
            "fixtures": [
                _fixture(
                    "healthy",
                    "The capital of Australia is Canberra.",
                    tokens=["The", " capital", " of", " Australia", " is", " Canberra", "."],
                ),
                _fixture(
                    "loop",
                    "The answer is unclear. " + "but wait I should check. " * 8,
                    tokens=(
                        "The answer is unclear. " + "but wait I should check. " * 8
                    ).split(),
                ),
            ]
        },
    }
    looping = service.call(
        "record_trace",
        {"input_text": "loop", "backend": backend, "params": {"max_new_tokens": 128}, "seed": 7},
    )["data"]["trace_id"]
    healthy = service.call(
        "record_trace",
        {"input_text": "healthy", "backend": backend, "params": {"max_new_tokens": 128}, "seed": 7},
    )["data"]["trace_id"]

    scan = service.call("scan_trace", {"trace_id": looping})["data"]
    kinds = {finding["kind"] for finding in scan["findings"]}
    assert "repeated_ngram" in kinds
    assert "reconsideration_markers" in kinds
    repetition = next(
        finding for finding in scan["findings"] if finding["kind"] == "repeated_ngram"
    )
    assert repetition["value"]["count"] >= 2
    assert "does not diagnose" in repetition["explanation"]

    comparison = service.call(
        "compare_traces", {"left_trace_id": healthy, "right_trace_id": looping}
    )["data"]
    assert comparison["output"]["identical"] is False
    assert comparison["tokens"]["identity_basis"] == "token_id"
    assert comparison["tokens"]["first_divergence"] is not None

    contrast_payload = {
        "name": "factual stability under user suggestion",
        "description": "The answer should remain Canberra under a misleading suggestion.",
        "baseline": {"name": "baseline", "input_text": "healthy"},
        "variants": [
            {
                "name": "suggestion",
                "input_text": "loop",
            }
        ],
        "expectation": {
            "kind": "reference_agreement",
            "description": "The factual answer should remain Canberra.",
            "reference_answers": ["Canberra"],
        },
        "backend": backend,
        "params": {"max_new_tokens": 128},
        "seed": 7,
    }
    contrast = service.call("run_contrast", contrast_payload)["data"]
    assert contrast["overall_status"] == "fail"
    assert contrast["test"]["baseline"]["input_text"] == "healthy"
    assert contrast["test"]["variants"][0]["input_text"] == "loop"
    assert contrast["test"]["expectation"]["description"] == "The factual answer should remain Canberra."

    capture = workspace.artifacts.put_json(
        "activation_capture",
        {"schema_version": "wsp.activation-capture/1.0", "note": "future capture reference"},
    )
    created = service.call(
        "create_failure_case",
        {
            "title": "Repetitive reconsideration under a misleading prompt",
            "summary": "A synthetic trace shows repeated reconsideration markers and repetition.",
            "observations": [looping],
            "controls": [healthy],
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
            "tags": ["synthetic", "repetition"],
            "reproduction": {
                "backend": "scripted",
                "command": "wsp inspect-trace " + looping,
                "seed": 7,
            },
        },
    )["data"]
    case = created["case"]
    assert case["review"]["status"] == "open"
    assert case["findings"]
    assert case["contrasts"][0]["artifact_id"] == contrast["artifact_id"]
    assert case["activation_captures"][0]["artifact_id"] == capture.artifact_id
    assert all(hypothesis["status"] == "proposed" for hypothesis in case["hypotheses"])

    out = tmp_path / "investigation.json"
    service.call(
        "export_failure_case",
        {"case_id": case["case_id"], "output_path": str(out)},
    )
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "wsp.case-export/1.1"
    assert len(bundle["traces"]) == 2
    assert bundle["traces"][0]["trace"]["trace_id"] in {looping, healthy}

    fresh_root = tmp_path / "restored"
    fresh = ToolService(workspace=Workspace(fresh_root, create=True), raise_errors=True)
    imported = fresh.call("import_failure_case", {"path": str(out)})["data"]
    assert imported["case"]["case_id"] == case["case_id"]
    restored_scan = fresh.call("scan_trace", {"trace_id": looping})["data"]
    assert any(
        finding["kind"] == "reconsideration_markers"
        for finding in restored_scan["findings"]
    )
    assert fresh.call("workspace_status", {})["data"]["traces"] == 2
    assert fresh.call("workspace_status", {})["data"]["cases"] == 1

    # Observations remain measurements. The case does not contain an automatic
    # causal conclusion or a claimed model improvement.
    explanations = " ".join(
        finding["explanation"] for finding in case["findings"]
    ).lower()
    assert "defect" not in explanations
    assert "cause" not in explanations
