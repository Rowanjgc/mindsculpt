"""Tier 1: temporal schedule representation and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wsp.core.errors import InputError
from wsp.interventions.schedule import (
    SCHEMA_SCHEDULE,
    TIMING_CONVENTIONS,
    InterventionSchedule,
    parse_schedule,
    resolve_schedule,
    schedule_from_step,
)


def test_first_prefix_forward_resolves_to_step_zero():
    schedule = InterventionSchedule()
    assert schedule.kind == "first-prefix-forward"
    assert schedule.resolve() == [0]
    assert schedule.one_time is True
    assert schedule.bounded_event_count == 1
    assert "first prefix" in schedule.describe()


def test_explicit_steps_are_one_time_or_sustained():
    one = InterventionSchedule(kind="generation-steps", steps=[3])
    sustained = InterventionSchedule(kind="generation-steps", steps=[1, 2, 3])
    assert one.resolve() == [3]
    assert one.one_time is True
    assert sustained.resolve() == [1, 2, 3]
    assert sustained.one_time is False
    assert sustained.bounded_event_count == 3


def test_inclusive_range_convention_is_documented_and_applied():
    inclusive = InterventionSchedule(
        kind="generation-range", start=2, end=4, end_inclusive=True
    )
    half_open = InterventionSchedule(
        kind="generation-range", start=2, end=4, end_inclusive=False
    )
    assert inclusive.resolve() == [2, 3, 4]
    assert half_open.resolve() == [2, 3]
    assert "inclusive end" in inclusive.describe()
    assert "exclusive end" in half_open.describe()


def test_range_stride_is_applied():
    schedule = InterventionSchedule(kind="generation-range", start=1, end=9, every=3)
    assert schedule.resolve() == [1, 4, 7]


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "generation-steps", "steps": []},
        {"kind": "generation-steps", "steps": [-1]},
        {"kind": "generation-steps", "steps": [2, 1]},
        {"kind": "generation-steps", "steps": [1, 1]},
        {"kind": "generation-steps", "steps": [1, 2], "every": 2},
        {"kind": "generation-steps", "steps": [1], "start": 0},
        {"kind": "generation-range"},
        {"kind": "generation-range", "start": 4, "end": 2},
        {"kind": "generation-range", "start": 0, "end": 1, "steps": [0]},
        {"kind": "first-prefix-forward", "steps": [0]},
        {"kind": "generation-range", "start": 0, "end": 1, "every": 0},
    ],
)
def test_malformed_schedules_are_rejected(payload):
    with pytest.raises(ValidationError):
        InterventionSchedule(**payload)


def test_unbounded_sweeps_are_rejected():
    with pytest.raises(ValidationError):
        InterventionSchedule(kind="generation-range", start=0, end=50, max_events=10)
    with pytest.raises(ValidationError):
        InterventionSchedule(
            kind="generation-steps", steps=list(range(0, 20)), max_events=8
        )
    with pytest.raises(ValidationError):
        InterventionSchedule(
            kind="generation-range", start=0, end=500, max_steps=100
        )


def test_step_beyond_configured_maximum_is_rejected():
    with pytest.raises(ValidationError):
        InterventionSchedule(kind="generation-steps", steps=[12], max_steps=8)


def test_parse_schedule_accepts_ints_and_lists():
    assert parse_schedule(4).resolve() == [4]
    assert parse_schedule([1, 2, 5]).resolve() == [1, 2, 5]
    assert parse_schedule({"kind": "generation-range", "start": 1, "end": 2}).resolve() == [
        1,
        2,
    ]
    assert parse_schedule(InterventionSchedule()).resolve() == [0]
    assert parse_schedule(schedule_from_step(2)).resolve() == [2]


def test_parse_schedule_rejects_unknown_shapes():
    with pytest.raises(InputError):
        parse_schedule(object())
    with pytest.raises(InputError):
        parse_schedule("later")


def test_schedule_round_trips_through_json():
    schedule = InterventionSchedule(kind="generation-range", start=1, end=4, every=2)
    payload = schedule.model_dump(mode="json")
    assert payload["schema_version"] == SCHEMA_SCHEDULE
    assert InterventionSchedule.model_validate(payload) == schedule


def test_resolve_schedule_separates_not_reached_from_lost():
    schedule = InterventionSchedule(kind="generation-steps", steps=[0, 2, 5])
    resolved = resolve_schedule(schedule, steps_executed=3)
    assert resolved.fired_steps == [0, 2]
    assert resolved.not_reached_steps == [5]
    assert resolved.one_time is False


def test_resolve_schedule_records_a_fired_step_outside_the_schedule():
    from wsp.interventions.schedule import ResolvedSchedule

    with pytest.raises(ValidationError):
        ResolvedSchedule(
            schedule=InterventionSchedule(kind="generation-steps", steps=[1]),
            steps=[1],
            fired_steps=[3],
            not_reached_steps=[],
        )


def test_timing_conventions_are_explicit():
    # The distinction between a generation step and a token position is the
    # single most error-prone part of temporal scheduling, so it is asserted.
    assert "generation_step" in TIMING_CONVENTIONS
    assert "full_sequence_index" in TIMING_CONVENTIONS
    text = TIMING_CONVENTIONS["generation_step"]
    assert "0" in text and "prefix" in text
