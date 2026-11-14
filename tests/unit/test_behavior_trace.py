from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from wsp.behavior.trace import TraceStore
from wsp.core.budget import Budget
from wsp.core.errors import InputError, ResourceLimitError
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace


def test_record_trace_is_immutable_and_inspectable(service, workspace: Workspace):
    trace_id = service.call(
        "record_trace",
        {
            "input_text": "What is the capital of Australia?",
            "system": "Answer briefly.",
            "backend": {
                "kind": "scripted",
                "scripted": {
                    "default_fixture": {
                        "output_text": "Canberra.",
                        "output_tokens": [
                            {"text": "Canberra", "token_id": 10},
                            {"text": ".", "token_id": 11},
                        ],
                        "token_logprobs": [-0.1, -0.2],
                        "termination": "eos",
                    },
                },
            },
            "params": {"max_new_tokens": 8, "temperature": 0.0},
            "seed": 4321,
        },
    )["data"]
    assert trace_id["trace_id"].startswith("tr_")
    record = TraceStore(workspace).get_record(trace_id["trace_id"])
    assert workspace.artifacts.verify(record.artifact)
    assert record.artifact.kind == "generation_trace"

    inspected = service.call("inspect_trace", {"trace_id": trace_id["trace_id"]})["data"]
    assert inspected["text"] == "Canberra."
    assert inspected["token_count"] == 2
    assert inspected["tokens"][0]["token_id"] == 10
    assert inspected["tokens"][0]["logprob"] == -0.1
    assert inspected["termination"]["reason"] == "eos"
    assert inspected["input"]["system"] == "Answer briefly."
    assert inspected["seed"] == 4321
    assert inspected["checksums"]["output_sha256"].startswith("sha256:")
    assert inspected["unavailable"] == {}

    listed = service.call("list_traces", {})["data"]
    assert listed["returned"] == 1
    assert listed["traces"][0]["trace_id"] == trace_id["trace_id"]
    assert workspace.status()["traces"] == 1


def test_text_only_trace_reports_missing_measurements(service):
    data = service.call(
        "record_trace",
        {
            "input_text": "prompt",
            "backend": {
                "kind": "scripted",
                "scripted": {
                    "default_fixture": {
                        "output_text": "plain text without token records",
                        "termination": "eos",
                    }
                },
            },
        },
    )["data"]
    inspected = service.call("inspect_trace", {"trace_id": data["trace_id"]})["data"]
    assert inspected["token_count"] == 0
    assert inspected["termination"]["generated_tokens"] is None
    assert "tokens" in inspected["unavailable"]
    assert "token_logprobs" in inspected["unavailable"]


def test_privacy_redaction_strips_text_but_keeps_structure(service):
    trace_id = service.call(
        "record_trace",
        {
            "input_text": "secret prompt",
            "backend": {
                "kind": "scripted",
                "scripted": {
                    "default_fixture": {
                        "output_text": "secret output",
                        "output_tokens": [
                            {"text": "secret", "token_id": 1},
                            {"text": " output", "token_id": 2},
                        ],
                    }
                },
            },
            "privacy": {"store_input_text": False, "store_output_text": False},
        },
    )["data"]["trace_id"]
    inspected = service.call("inspect_trace", {"trace_id": trace_id})["data"]
    assert inspected["input"]["text"] is None
    assert inspected["text"] is None
    assert all(token["text"] is None for token in inspected["tokens"])
    assert [token["token_id"] for token in inspected["tokens"]] == [1, 2]
    assert inspected["privacy"]["redacted"] is True
    assert "input_text" in inspected["unavailable"]
    assert "output_text" in inspected["unavailable"]


def test_input_artifact_reference_is_preserved(service, workspace: Workspace):
    ref = workspace.artifacts.put_bytes("prompt", b"exact prompt bytes", format="txt")
    trace_id = service.call(
        "record_trace",
        {
            "input_text": "exact prompt bytes",
            "input_artifact_id": ref.artifact_id,
            "backend": {
                "kind": "scripted",
                "scripted": {"default_fixture": {"output_text": "ok"}},
            },
        },
    )["data"]["trace_id"]
    inspected = service.call("inspect_trace", {"trace_id": trace_id})["data"]
    assert inspected["input"]["artifact_id"] == ref.artifact_id
    assert inspected["input"]["sha256"] == ref.sha256


def test_trace_limits_are_enforced(workspace: Workspace):
    bounded = ToolService(
        workspace=workspace,
        budget=Budget(max_trace_tokens=2),
        raise_errors=True,
    )
    with pytest.raises(ResourceLimitError):
        bounded.call(
            "record_trace",
            {
                "input_text": "prompt",
                "backend": {
                    "kind": "scripted",
                    "scripted": {
                        "default_fixture": {
                            "output_text": "a b c",
                            "output_tokens": [
                                {"text": "a", "token_id": 1},
                                {"text": "b", "token_id": 2},
                                {"text": "c", "token_id": 3},
                            ],
                        }
                    },
                },
            },
        )


def test_wsp_trace_export_and_import(service, workspace: Workspace, tmp_path: Path):
    trace_id = service.call(
        "record_trace",
        {
            "input_text": "prompt",
            "backend": {
                "kind": "scripted",
                "scripted": {
                    "default_fixture": {
                        "output_text": "answer",
                        "output_tokens": [{"text": "answer", "token_id": 4}],
                    }
                },
            },
        },
    )["data"]["trace_id"]
    path = tmp_path / "trace.json"
    exported = service.call(
        "export_trace",
        {"trace_id": trace_id, "format": "wsp", "output_path": str(path)},
    )["data"]
    assert path.exists()
    assert exported["sha256"].startswith("sha256:")

    fresh_root = tmp_path / "fresh"
    fresh = ToolService(workspace=Workspace(fresh_root, create=True), raise_errors=True)
    imported = fresh.call("import_trace", {"format": "wsp", "path": str(path)})["data"]
    assert imported["imported"] == 1
    inspected = fresh.call(
        "inspect_trace", {"trace_id": imported["trace_ids"][0]}
    )["data"]
    assert inspected["text"] == "answer"
    assert inspected["tokens"][0]["token_id"] == 4
    assert inspected["provenance"]["origin"] == "generated"

    detached = ToolService(discover_workspace=False, raise_errors=True)
    inline = detached.call(
        "import_trace", {"format": "wsp", "path": str(path), "persist": False}
    )["data"]
    assert inline["persisted"] is False
    assert inline["traces"][0]["text"] == "answer"


def test_import_failure_is_isolated(service, tmp_path: Path):
    bad = tmp_path / "bad.eval"
    bad.write_text("this is not a zip archive", encoding="utf-8")
    envelope = service.call(
        "import_trace", {"format": "inspect", "path": str(bad)}, raise_errors=False
    )
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "invalid_input"
    assert service.call("list_detectors", {})["ok"] is True


def test_inif_import_preserves_sequences_extras_and_metadata(service, tmp_path: Path):
    document = {
        "metadata": {
            "model": {
                "name": "tiny",
                "revision": "r1",
                "huggingface_id": "org/tiny",
                "generation_config": {"max_tokens": 8, "temperature": 0.0},
                "loading_config": {"dtype": "float32"},
            },
            "inif_version": "0.1",
            "packages": {},
            "sources": ["test"],
            "created_at": "2026-11-14T00:00:00Z",
            "extra": {"kept": True},
        },
        "sequences": [
            {
                "id": "seq_a",
                "n_tokens": 2,
                "tokens": [
                    {"token": "but", "id": 10},
                    {"token": " wait", "id": 11},
                ],
            }
        ],
        "samples": [
            {
                "id": "s1",
                "tokens": [
                    {"token": "Q", "id": 1},
                    {"token": "seq_a", "id": None},
                    {"token": "seq_a", "id": None},
                    {"token": " done", "id": 12, "logprob": -0.2, "custom": "keep"},
                ],
                "texts": [
                    {"name": "user_0", "value": "prompt", "start": 0, "end": 1},
                    {
                        "name": "assistant_0",
                        "value": "but waitbut wait done",
                        "start": 1,
                        "end": 4,
                    },
                ],
                "annotations": [{"name": "loop", "ranges": [[1, 4]]}],
                "spans": [],
                "scores": [{"scorer": "match", "value": "C"}],
                "target": "done",
                "metadata": {"note": "kept"},
                "custom_field": "preserve-me",
                "total_time": 0.5,
            }
        ],
    }
    path = tmp_path / "doc.inif.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    imported = service.call("import_trace", {"format": "inif", "path": str(path)})["data"]
    trace_id = imported["trace_ids"][0]
    inspected = service.call(
        "inspect_trace", {"trace_id": trace_id, "include_metadata": True}
    )["data"]
    assert [token["token_id"] for token in inspected["tokens"]] == [10, 11, 10, 11, 12]
    assert [token["text"] for token in inspected["tokens"]] == [
        "but",
        " wait",
        "but",
        " wait",
        " done",
    ]
    assert inspected["tokens"][-1]["logprob"] == -0.2
    assert inspected["text"] == "but waitbut wait done"
    assert inspected["annotations"] == [
        {"name": "loop", "start": 0, "end": 5, "metadata": {}}
    ]
    assert inspected["expected_answers"] == ["done"]
    assert inspected["model"]["name"] == "tiny"
    assert inspected["params"]["max_new_tokens"] == 8
    assert inspected["metadata"]["inif_unmapped_sample_fields"] == {
        "custom_field": "preserve-me"
    }
    assert inspected["metadata"]["inif_token_extras"] == {"4": {"custom": "keep"}}
    assert inspected["metadata"]["inif_scores"] == [{"scorer": "match", "value": "C"}]
    assert "termination" in inspected["unavailable"]

    out = tmp_path / "out.inif.json"
    service.call(
        "export_trace",
        {"trace_id": trace_id, "format": "inif", "output_path": str(out)},
    )
    exported = json.loads(out.read_text(encoding="utf-8"))
    assert exported["metadata"]["inif_version"] == "0.1"
    sample = exported["samples"][0]
    assert [token["id"] for token in sample["tokens"]] == [10, 11, 10, 11, 12]
    assert sample["target"] == "done"
    assert sample["texts"][-1]["name"] == "assistant_0"


def test_inif_invalid_document_is_rejected(service, tmp_path: Path):
    path = tmp_path / "bad.inif.json"
    path.write_text(json.dumps({"metadata": {}, "samples": "not-a-list"}), encoding="utf-8")
    with pytest.raises(InputError):
        service.call("import_trace", {"format": "inif", "path": str(path)})


def test_inspect_json_log_import(service, tmp_path: Path):
    payload = {
        "version": "0.3.289",
        "status": "success",
        "eval": {
            "model": "mockllm/model",
            "model_generate_config": {"max_tokens": 8, "temperature": 0.0, "seed": 7},
            "packages": {"inspect-ai": "0.3.289"},
            "task": "geography",
            "task_id": "task-1",
            "eval_id": "eval-1",
            "run_id": "run-1",
        },
        "samples": [
            {
                "id": "sample-1",
                "input": "What is the capital of Australia?",
                "target": "Canberra",
                "started_at": "2026-11-14T00:00:00Z",
                "completed_at": "2026-11-14T00:00:01Z",
                "total_time": 0.25,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of Australia?",
                        "source": "input",
                    },
                    {"role": "assistant", "content": "Canberra", "source": "generate"},
                ],
                "output": {
                    "completion": "Canberra",
                    "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
                    "choices": [{"message": {"role": "assistant"}, "stop_reason": "max_tokens"}],
                },
                "scores": {"match": {"value": "C", "answer": "Canberra"}},
            }
        ],
    }
    path = tmp_path / "log.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    imported = service.call("import_trace", {"format": "inspect", "path": str(path)})["data"]
    inspected = service.call(
        "inspect_trace", {"trace_id": imported["trace_ids"][0], "include_metadata": True}
    )["data"]
    assert inspected["model"]["name"] == "mockllm/model"
    assert inspected["text"] == "Canberra"
    assert inspected["input"]["text"] == "What is the capital of Australia?"
    assert inspected["termination"]["reason"] == "length_limit"
    assert inspected["token_count"] == 0
    assert "token_ids" in inspected["unavailable"]
    assert inspected["expected_answers"] == ["Canberra"]
    assert inspected["metadata"]["inspect_scores"]["match"]["value"] == "C"
    assert inspected["params"]["max_new_tokens"] == 8
    assert inspected["seed"] == 7
    assert inspected["provenance"]["source_system"] == "inspect_ai"


def test_inspect_eval_zip_import(service, tmp_path: Path):
    path = tmp_path / "log.eval"
    header = {
        "version": "0.3.289",
        "eval": {"model": "mockllm/model", "model_generate_config": {"max_tokens": 4}},
    }
    sample = {
        "id": "s1",
        "input": "Q",
        "output": {"completion": "A", "choices": [{"stop_reason": "stop"}]},
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("header.json", json.dumps(header))
        archive.writestr("samples/1_epoch_1.json", json.dumps(sample))
    imported = service.call("import_trace", {"format": "inspect", "path": str(path)})["data"]
    inspected = service.call("inspect_trace", {"trace_id": imported["trace_ids"][0]})["data"]
    assert inspected["text"] == "A"
    assert inspected["termination"]["reason"] == "eos"
