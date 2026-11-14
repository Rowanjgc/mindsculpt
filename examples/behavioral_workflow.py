"""Offline Behavioral Observatory demonstration (V1.1).

This is an infrastructure demonstration, not a scientific result. It uses only
synthetic scripted generations and requires no model download, network access,
or GPU.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from wsp.behavior.trace import TraceStore
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace


def fixture(input_text: str, output_text: str, tokens: list[str]) -> dict:
    return {
        "input_equals": input_text,
        "output_text": output_text,
        "output_tokens": [
            {"text": token, "token_id": index + 1}
            for index, token in enumerate(tokens)
        ],
    }


def main() -> None:
    out = Path(__file__).parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="wsp-behavior-", dir=out))
    workspace = Workspace(root, create=True, name="behavior-demo")
    service = ToolService(workspace=workspace, raise_errors=True)

    healthy_text = "The capital of Australia is Canberra."
    looping_text = "The answer is unclear. " + "but wait I should check. " * 8
    backend = {
        "kind": "scripted",
        "scripted": {
            "model_name": "synthetic/demo",
            "fixtures": [
                fixture("healthy", healthy_text, healthy_text.split()),
                fixture("loop", looping_text, looping_text.split()),
            ],
        },
    }

    looping = service.call(
        "record_trace",
        {
            "input_text": "loop",
            "backend": backend,
            "params": {"max_new_tokens": 128},
            "seed": 7,
        },
    )["data"]["trace_id"]
    healthy = service.call(
        "record_trace",
        {
            "input_text": "healthy",
            "backend": backend,
            "params": {"max_new_tokens": 128},
            "seed": 7,
        },
    )["data"]["trace_id"]

    scan = service.call("scan_trace", {"trace_id": looping})["data"]
    comparison = service.call(
        "compare_traces", {"left_trace_id": healthy, "right_trace_id": looping}
    )["data"]
    contrast = service.call(
        "run_contrast",
        {
            "name": "factual stability under a misleading prompt",
            "baseline": {"name": "baseline", "input_text": "healthy"},
            "variants": [{"name": "suggestion", "input_text": "loop"}],
            "expectation": {
                "kind": "reference_agreement",
                "description": "The factual answer should remain Canberra.",
                "reference_answers": ["Canberra"],
            },
            "backend": backend,
            "params": {"max_new_tokens": 128},
            "seed": 7,
        },
    )["data"]

    capture = workspace.artifacts.put_json(
        "activation_capture",
        {"schema_version": "wsp.activation-capture/1.0", "note": "future evidence"},
    )
    case = service.call(
        "create_failure_case",
        {
            "title": "Synthetic repetitive reconsideration",
            "summary": "Synthetic traces used to demonstrate evidence plumbing.",
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
            "reproduction": {"backend": "scripted", "seed": 7},
        },
    )["data"]

    export_path = root / "case_export.json"
    service.call(
        "export_failure_case",
        {"case_id": case["case"]["case_id"], "output_path": str(export_path)},
    )

    restored_workspace = Workspace(root / "restored", create=True, name="restored")
    restored = ToolService(workspace=restored_workspace, raise_errors=True)
    reloaded = restored.call(
        "import_failure_case", {"path": str(export_path)}
    )["data"]
    restored_scan = restored.call("scan_trace", {"trace_id": looping})["data"]

    print(
        json.dumps(
            {
                "workspace": str(root),
                "traces": {
                    "healthy": healthy,
                    "looping": looping,
                },
                "finding_kinds": sorted(
                    {finding["kind"] for finding in scan["findings"]}
                ),
                "comparison": {
                    "outputs_identical": comparison["output"]["identical"],
                    "first_token_divergence": comparison["tokens"]["first_divergence"],
                },
                "contrast": {
                    "overall_status": contrast["overall_status"],
                    "summary": contrast["summary"],
                    "artifact_id": contrast["artifact_id"],
                },
                "case": {
                    "case_id": case["case"]["case_id"],
                    "status": case["case"]["review"]["status"],
                    "revision": case["case"]["revision"],
                    "export_path": str(export_path),
                },
                "reloaded": {
                    "case_id": reloaded["case"]["case_id"],
                    "trace_count": len(TraceStore(restored_workspace).list()),
                    "restored_finding_kinds": sorted(
                        {finding["kind"] for finding in restored_scan["findings"]}
                    ),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
