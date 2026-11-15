"""Fixed-context counterfactual replay and mandatory controls."""

from __future__ import annotations

import pytest

from support.temporal import make_spec
from support.tiny_lm import build_runtime, make_trace
from wsp.behavior.events import EventTriggerDefinition, detect_events, manual_event
from wsp.behavior.trace import TraceStore
from wsp.core.budget import Budget
from wsp.core.errors import CompatibilityError, InputError
from wsp.interventions.fixed_context import (
    SCHEMA_FIXED_COMPARISON,
    SCHEMA_FIXED_REPLAY,
    prepare_fixed_context,
    run_fixed_context_comparison,
)
from wsp.replay.engine import parameter_fingerprint
from wsp.replay.plan import sequence_checksum


def _fixture(workspace):
    runtime = build_runtime(seed=7)
    trace = make_trace(runtime, "loop", max_new_tokens=10)
    record = TraceStore(workspace).save(trace)
    prompt_length = len(trace.input.token_ids or [])
    tokens = [int(value) for value in (trace.input.token_ids or [])] + [
        int(token.token_id) for token in trace.tokens
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
        budget=Budget(),
    )
    return runtime, trace, event, replay


def test_prepare_reconstructs_exact_prefix(workspace):
    runtime, trace, event, replay = _fixture(workspace)
    assert replay.schema_version == SCHEMA_FIXED_REPLAY
    assert replay.replay_step == event.generation_step
    assert replay.consumed_position == event.alignment.consumed_position
    assert replay.sequence.sequence_sha256 == sequence_checksum(
        replay.sequence.input_token_ids, replay.sequence.generated_token_ids
    )
    assert replay.checksums["replay_sha256"]


def test_fixed_context_identity_control_and_measurements(workspace):
    runtime, trace, event, replay = _fixture(workspace)
    spec = make_spec(
        "blocks.0.mlp",
        prompt_length=replay.sequence.full_length,
        positions=[replay.consumed_position],
        operation="zero",
        execution_mode="teacher-forced",
    )
    before = parameter_fingerprint(runtime.model)
    comparison = run_fixed_context_comparison(
        replay,
        spec,
        runtime.replay_model(),
        budget=Budget(),
    )
    after = parameter_fingerprint(runtime.model)
    assert before == after
    assert comparison.schema_version == SCHEMA_FIXED_COMPARISON
    assert comparison.identity_control.performed is True
    assert comparison.identity_control.passed is True
    assert comparison.identity_control.agreement == "exact"
    assert comparison.validity.status == "valid"
    assert comparison.measurements
    assert comparison.measurements[0].position == replay.consumed_position
    assert comparison.measurements[0].identity_delta == 0.0
    assert comparison.measurements[0].intervention_delta is not None
    assert comparison.token_ids_sha256 == replay.sequence.sequence_sha256
    assert comparison.checksums["comparison_sha256"]


def test_target_position_must_match_the_selected_replay(workspace):
    runtime, trace, event, replay = _fixture(workspace)
    other = 0 if replay.consumed_position != 0 else 1
    spec = make_spec(
        "blocks.0.mlp",
        prompt_length=replay.sequence.full_length,
        positions=[other],
        operation="zero",
        execution_mode="teacher-forced",
    )
    with pytest.raises(CompatibilityError):
        run_fixed_context_comparison(replay, spec, runtime.replay_model())


def test_incompatible_model_structure_is_rejected(workspace):
    runtime, trace, event, replay = _fixture(workspace)
    other = build_runtime(seed=7, model_dim=16, num_layers=2)
    spec = make_spec(
        "blocks.0.mlp",
        prompt_length=replay.sequence.full_length,
        positions=[replay.consumed_position],
        operation="zero",
        execution_mode="teacher-forced",
    )
    with pytest.raises(CompatibilityError):
        run_fixed_context_comparison(replay, spec, other.replay_model())


def test_ambiguous_event_cannot_be_prepared_for_targeting(workspace):
    runtime = build_runtime(seed=7)
    trace = make_trace(runtime, "loop", max_new_tokens=8)
    prompt_length = len(trace.input.token_ids or [])
    tokens = [int(value) for value in (trace.input.token_ids or [])] + [
        int(token.token_id) for token in trace.tokens
    ]
    event = manual_event(tokens, prompt_length=prompt_length, position=prompt_length + 2)
    event.alignment.status = "ambiguous"
    with pytest.raises(InputError):
        prepare_fixed_context(trace, events=[event])


def test_optional_continuation_is_reported_separately(workspace):
    runtime, trace, event, replay = _fixture(workspace)
    spec = make_spec(
        "blocks.0.mlp",
        prompt_length=replay.sequence.full_length,
        positions=[replay.consumed_position],
        operation="zero",
        execution_mode="teacher-forced",
    )
    comparison = run_fixed_context_comparison(
        replay,
        spec,
        runtime.replay_model(),
        budget=Budget(),
        continuation={"max_new_tokens": 3},
    )
    assert comparison.continuation is not None
    assert comparison.continuation.diff.longest_common_prefix >= 0
    assert comparison.continuation.intervention_applied_steps == [0]
    assert "conditioned on diverging histories" in comparison.continuation.note
    assert any("continuation" in item.lower() for item in comparison.limitations)
