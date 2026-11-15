"""Deterministic behavioral events: schema, detection, alignment, timing."""

from __future__ import annotations

import pytest

from wsp.behavior.events import (
    SCHEMA_EVENT,
    EventTriggerDefinition,
    TokenSpan,
    detect_events,
    detect_text_pattern,
    earliest_repetition,
    event_sha256,
    manual_event,
    repeated_sequence_positions,
    resolve_finding_event,
    token_occurrences,
)
from wsp.behavior.models import EvidenceRef, Finding
from wsp.core.budget import Budget
from wsp.core.errors import InputError, ResourceLimitError
from support.tiny_lm import build_runtime, make_trace


def _definition(**overrides):
    payload = {"kind": "first_repetition", "ngram_size": 2, "min_occurrences": 2}
    payload.update(overrides)
    return EventTriggerDefinition.model_validate(payload)


def test_token_occurrence_reports_absolute_positions():
    tokens = [10, 20, 21, 20, 21, 30]
    found = token_occurrences(tokens, prompt_length=2, token_id=20)
    assert found == [3]
    found = token_occurrences(tokens, prompt_length=2, token_sequence=[20, 21])
    assert found == [4]


def test_rejected_definition_is_not_ambiguous():
    with pytest.raises(ValueError):
        EventTriggerDefinition(kind="token_occurrence")
    with pytest.raises(ValueError):
        EventTriggerDefinition(kind="first_repetition", ngram_size=2)
    with pytest.raises(ValueError):
        EventTriggerDefinition(
            kind="repeated_token_sequence",
            ngram_size=2,
            min_occurrences=2,
            consecutive=True,
            lookback=10,
        )


def test_first_repetition_earliest_absolute_position():
    tokens = [9, 9, 5, 5, 5, 5]
    result = earliest_repetition(
        tokens, prompt_length=2, ngram_size=2, min_occurrences=2
    )
    assert result is not None
    end, count, span = result
    assert end == 3
    assert count == 2
    assert span == TokenSpan(start=2, end=4)


def test_consecutive_repetition_requires_adjacent_copies():
    # The two 5,5 windows overlap but are not adjacent copies.
    tokens = [0, 1, 5, 5, 5]
    found = repeated_sequence_positions(
        tokens,
        prompt_length=2,
        ngram_size=2,
        min_occurrences=2,
        consecutive=True,
    )
    assert found == []
    found = repeated_sequence_positions(
        tokens,
        prompt_length=2,
        ngram_size=2,
        min_occurrences=2,
        consecutive=False,
    )
    assert found


def test_event_records_consumed_position_and_online_earliest_step():
    tokens = [1, 8, 4, 4, 4]
    events = detect_events(
        tokens,
        definition=_definition(),
        prompt_length=2,
        observation_mode="online",
    )
    assert len(events) == 1
    event = events[0]
    assert event.alignment.status == "exact"
    assert event.token_position == 3
    assert event.alignment.consumed_position == 2
    assert event.generation_step == 1
    assert event.alignment.earliest_online_step == 2
    assert event.observation_mode == "online"
    assert event.schema_version == SCHEMA_EVENT
    assert event.checksums["event_sha256"] == event_sha256(event)


def test_future_tokens_cannot_change_an_earlier_observation():
    base = [1, 8, 4, 4, 4]
    extended = base + [4, 4, 7]
    first = detect_events(base, definition=_definition(), prompt_length=2, max_events=1)
    second = detect_events(extended, definition=_definition(), prompt_length=2, max_events=1)
    assert first[0].token_position == second[0].token_position == 3
    assert first[0].alignment.generation_step == second[0].alignment.generation_step


def test_first_repetition_is_single_event():
    tokens = [1, 2, 5, 5, 5, 5, 5]
    events = detect_events(
        tokens,
        definition=_definition(kind="first_repetition"),
        prompt_length=2,
    )
    assert len(events) == 1


def test_manual_event_requires_generated_position():
    with pytest.raises(InputError):
        manual_event([1, 2, 3], prompt_length=2, position=1)
    event = manual_event([1, 2, 3, 4], prompt_length=2, position=3, note="inspected")
    assert event.event_type == "manual"
    assert event.token_position == 3
    assert event.alignment.status == "exact"


def test_budget_limits_lookback():
    with pytest.raises(ResourceLimitError):
        detect_events(
            [1, 2, 3],
            definition=_definition(lookback=10_000),
            prompt_length=1,
            budget=Budget(max_event_lookback=16),
        )


def test_text_pattern_reports_exact_and_ambiguous_alignment():
    runtime = build_runtime(seed=7)
    trace = make_trace(runtime, "loop", max_new_tokens=8)
    prompt_length = len(trace.input.token_ids or [])
    exact = detect_text_pattern(trace, "loop", prompt_length=prompt_length)
    assert exact and exact[0].alignment.status == "exact"
    ambiguous = detect_text_pattern(trace, "loo", prompt_length=prompt_length)
    assert ambiguous and ambiguous[0].alignment.status == "ambiguous"
    assert ambiguous[0].token_position is None or ambiguous[0].alignment.status == "ambiguous"


def test_detector_finding_resolution_uses_prompt_offset():
    runtime = build_runtime(seed=7)
    trace = make_trace(runtime, "loop", max_new_tokens=8)
    prompt_length = len(trace.input.token_ids or [])
    finding = Finding(
        detector_id="core.repetition",
        detector_version="1.0.0",
        kind="repeated_ngram",
        explanation="fixture",
        evidence=[
            EvidenceRef(
                kind="token_range",
                trace_id=trace.trace_id,
                start=0,
                end=2,
            )
        ],
    )
    event = resolve_finding_event(trace, finding, trace_artifact_id="a_fixture")
    assert event.alignment.status == "exact"
    assert event.token_span == TokenSpan(start=prompt_length, end=prompt_length + 2)
    assert event.metadata["detector_offset"] == prompt_length


def test_detector_finding_without_token_range_is_ambiguous():
    runtime = build_runtime(seed=7)
    trace = make_trace(runtime, "loop", max_new_tokens=8)
    finding = Finding(
        detector_id="core.repetition",
        detector_version="1.0.0",
        kind="repeated_ngram",
        explanation="fixture",
        evidence=[EvidenceRef(kind="word_range", start=0, end=2)],
    )
    event = resolve_finding_event(trace, finding)
    assert event.alignment.status == "ambiguous"
    assert event.token_position is None
    assert "no token-range" in (event.alignment.detail or "")
