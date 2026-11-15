"""Redacted exports of V1.5 event artifacts."""

from __future__ import annotations

import json

import pytest

from support.tiny_lm import build_runtime, make_trace
from wsp.behavior.events import EventTriggerDefinition, detect_events
from wsp.behavior.trace import TraceStore
from wsp.core.errors import SafetyError
from wsp.interventions.fixed_context import (
    prepare_fixed_context,
    run_fixed_context_comparison,
)
from wsp.investigation.dossier import build_dossier
from wsp.workspace import privacy
from wsp.workspace.privacy import ExportPolicy, RemovedText, transform_artifact
from support.temporal import make_spec


def _payload(kind: str, model, data: dict) -> tuple[str, bytes]:
    return kind, privacy.canonical_bytes(model.model_dump(mode="json"))


def _redact(kind: str, model, data: dict):
    _kind, raw = _payload(kind, model, data)
    removed = RemovedText()
    new_kind, new_format, output = transform_artifact(
        kind=kind,
        format_name="json",
        data=raw,
        policy=ExportPolicy(mode="redacted"),
        removed=removed,
    )
    assert new_kind == kind
    assert new_format == "json"
    return json.loads(output), removed


def test_event_replay_comparison_and_dossier_redaction(workspace):
    runtime = build_runtime(seed=7)
    trace = make_trace(runtime, "loop", max_new_tokens=10)
    record = TraceStore(workspace).save(trace)
    prompt_length = len(trace.input.token_ids or [])
    tokens = [int(v) for v in (trace.input.token_ids or [])] + [
        int(t.token_id) for t in trace.tokens
    ]
    event = detect_events(
        tokens,
        definition=EventTriggerDefinition(
            kind="first_repetition", ngram_size=2, min_occurrences=2
        ),
        prompt_length=prompt_length,
        trace_id=trace.trace_id,
        trace_artifact_id=record.artifact.artifact_id,
    )[0]
    replay = prepare_fixed_context(
        trace,
        trace_artifact_id=record.artifact.artifact_id,
        events=[event],
    )
    spec = make_spec(
        "blocks.0.mlp",
        prompt_length=replay.sequence.full_length,
        positions=[replay.consumed_position],
        operation="zero",
        execution_mode="teacher-forced",
    )
    comparison = run_fixed_context_comparison(
        replay, spec, runtime.replay_model()
    )
    dossier = build_dossier(
        title="sensitive title",
        trace_id=trace.trace_id,
        events=[event],
        comparison=comparison,
        comparison_artifact_id="a_fixture",
        summary="sensitive summary",
    )

    event_payload, _ = _redact("behavioral_event", event, None)
    assert event_payload["token_ids"] == []
    assert event_payload["token_span"]  # positions survive
    assert event_payload["checksums"] == {}

    replay_payload, _ = _redact("fixed_context_replay", replay, None)
    assert replay_payload["sequence"]["full_token_ids"] == []
    assert replay_payload["sequence"]["sequence_sha256"] is None
    assert replay_payload["consumed_position"] is None

    comparison_payload, _ = _redact("fixed_context_comparison", comparison, None)
    assert comparison_payload["token_ids_sha256"] == ""
    assert all(item["token_id"] is None for item in comparison_payload["measurements"])
    assert comparison_payload["measurements"]  # numeric evidence survives
    assert comparison_payload["checksums"] == {}

    dossier_payload, _ = _redact("investigation_dossier", dossier, None)
    assert dossier_payload["title"] == "[redacted]"
    assert dossier_payload["summary"] is None
    assert all(item["token_id"] is None for item in dossier_payload["measurements"])
    assert dossier_payload["limitations"]


def test_unknown_numeric_artifact_fails_closed():
    removed = RemovedText()
    with pytest.raises(SafetyError):
        transform_artifact(
            kind="mystery_array",
            format_name="npy",
            data=b"not-a-real-npy",
            policy=ExportPolicy(mode="redacted"),
            removed=removed,
        )
