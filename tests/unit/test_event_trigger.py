"""Online event-trigger scheduling and causal ordering."""

from __future__ import annotations

import pytest

from support.temporal import make_spec
from support.tiny_lm import build_runtime
from wsp.execution.synthetic_lm import EOS_ID, greedy_generate
from wsp.behavior.events import EventTriggerDefinition
from wsp.core.budget import Budget
from wsp.core.errors import InputError, ResourceLimitError
from wsp.interventions.event_trigger import (
    SCHEMA_EVENT_INTERVENTION,
    EventTrigger,
    event_run_sha256,
    retrospective_trigger,
    run_online_event_trigger,
    trigger_sha256,
    validate_trigger_for_run,
)


def _online(**overrides) -> EventTrigger:
    payload = {
        "trigger_id": "et_0123456789abcdef",
        "mode": "online",
        "definition": {
            "kind": "token_occurrence",
            "token_id": 1,
        },
        "delay_steps": 0,
        "steps_after_event": 1,
        "max_triggers": 1,
        "fire_on": "once",
        "position_mode": "last",
    }
    payload.update(overrides)
    return EventTrigger.model_validate(payload)


def _setup():
    runtime = build_runtime(seed=7)
    prompt = runtime.tokenizer.encode("loop", add_special_tokens=True)
    generated, _ = greedy_generate(runtime, prompt, max_new_tokens=10, eos_id=EOS_ID)
    spec = make_spec(
        "blocks.0.mlp",
        prompt_length=len(prompt),
        positions=[0],
        operation="zero",
        execution_mode="uncached-continuation",
    )
    return runtime, prompt, generated, spec


def test_trigger_requires_mode_coherence():
    with pytest.raises(ValueError):
        _online(event_id="ev_0123456789abcdef")
    with pytest.raises(ValueError):
        EventTrigger(
            trigger_id="et_0123456789abcdef",
            mode="retrospective",
            definition=EventTriggerDefinition(kind="token_occurrence", token_id=1),
            replay_step=2,
            delay_steps=1,
        )
    with pytest.raises(ValueError):
        _online(max_triggers=2, fire_on="once")


def test_online_event_at_final_step_is_not_reached():
    runtime, prompt, generated, spec = _setup()
    trigger = _online(
        definition=EventTriggerDefinition(
            kind="token_occurrence", token_id=generated[0]
        ),
        delay_steps=10,
        steps_after_event=1,
    )
    record = run_online_event_trigger(
        trigger,
        spec,
        runtime.replay_model(),
        prompt,
        max_new_tokens=10,
        budget=Budget(),
    )
    assert record.schema_version == SCHEMA_EVENT_INTERVENTION
    assert record.observed_events >= 1
    assert record.fired_events == 0
    assert record.not_reached_events >= 1
    assert all(item.state != "fired" for item in record.events)
    assert record.events[0].no_subsequent_forward is True
    assert record.events[0].fired_steps == []
    assert record.validity.status == "limited"
    assert any("no eligible subsequent forward" in item for item in record.warnings)


def test_online_intervention_fires_after_observation_only():
    runtime, prompt, generated, spec = _setup()
    trigger = _online(
        definition=EventTriggerDefinition(
            kind="token_occurrence", token_id=generated[0]
        )
    )
    record = run_online_event_trigger(
        trigger,
        spec,
        runtime.replay_model(),
        prompt,
        max_new_tokens=8,
        budget=Budget(),
    )
    fired = [item for item in record.events if item.state == "fired"]
    assert len(fired) == 1
    event = fired[0]
    assert event.observation_step == 0
    assert event.earliest_eligible_step == 1
    assert event.fired_steps == [1]
    assert min(event.fired_steps) > event.observation_step
    assert event.applications and event.applications[0].elements_modified > 0
    assert record.identity_control.performed is True
    assert record.identity_control.passed is True
    assert record.validity.status == "valid"
    assert record.checksums["run_sha256"]


def test_delay_advances_earliest_eligible_step():
    runtime, prompt, generated, spec = _setup()
    trigger = _online(
        definition=EventTriggerDefinition(
            kind="token_occurrence", token_id=generated[0]
        ),
        delay_steps=2,
    )
    record = run_online_event_trigger(
        trigger,
        spec,
        runtime.replay_model(),
        prompt,
        max_new_tokens=8,
    )
    event = next(item for item in record.events if item.state == "fired")
    assert event.earliest_eligible_step == 3
    assert event.fired_steps == [3]


def test_execution_rejects_unbounded_trigger_budget():
    runtime, prompt, _generated, spec = _setup()
    trigger = _online(max_triggers=32, fire_on="every_occurrence")
    with pytest.raises(ResourceLimitError):
        run_online_event_trigger(
            trigger,
            spec,
            runtime.replay_model(),
            prompt,
            max_new_tokens=4,
            budget=Budget(max_trigger_occurrences=8),
        )


def test_retrospective_trigger_is_labelled_and_can_precede_event():
    runtime = build_runtime(seed=7)
    from support.tiny_lm import make_trace
    from wsp.behavior.events import detect_events

    trace = make_trace(runtime, "loop", max_new_tokens=10)
    prompt_length = len(trace.input.token_ids or [])
    tokens = [int(v) for v in (trace.input.token_ids or [])] + [
        int(t.token_id) for t in trace.tokens
    ]
    events = detect_events(
        tokens,
        definition=EventTriggerDefinition(
            kind="first_repetition", ngram_size=2, min_occurrences=2
        ),
        prompt_length=prompt_length,
    )
    event = events[0]
    trigger = retrospective_trigger(event=event, replay_step=0)
    assert trigger.mode == "retrospective"
    assert trigger.event_id == event.event_id
    assert trigger.replay_step == 0
    assert trigger_sha256(trigger)
    validate_trigger_for_run(trigger, available_event=event)
    with pytest.raises(InputError):
        validate_trigger_for_run(trigger, available_event=None)
