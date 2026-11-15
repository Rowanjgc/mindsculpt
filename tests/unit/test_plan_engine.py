"""Tier 1/2: the scheduled executor, event accounting, and state protection."""

from __future__ import annotations

import pytest
import torch

from support.temporal import (
    make_plan,
    make_plan_intervention,
    make_spec,
    one_plan,
)
from wsp.core.errors import ExecutionError
from wsp.execution.synthetic_lm import TinyCausalLM
from wsp.interventions.plan_engine import (
    installed_hook_count,
    mutable_state_fingerprint,
    preview_plan,
    run_temporal_intervention,
)
from wsp.interventions.engine import parameter_fingerprint

PROMPT = "abc"


def _prompt(runtime) -> list[int]:
    return runtime.tokenizer.encode(PROMPT, add_special_tokens=True)


def _run(plan, runtime, *, identity=True):
    return run_temporal_intervention(
        plan, runtime.replay_model(), _prompt(runtime), identity=identity
    )


def test_baseline_and_identity_agree_exactly(tiny_lm):
    plan = one_plan("blocks.1", prompt_length=4, steps=[1])
    outcome = _run(plan, tiny_lm)
    record = outcome.record
    assert record.validity.status == "valid"
    assert record.identity_control.performed is True
    assert record.identity_control.agreement == "exact"
    baseline, identity, executed = record.conditions
    assert identity.generated_token_ids == baseline.generated_token_ids
    assert identity.events_applied == 0
    assert identity.events_identity == 1


def test_scheduled_event_fires_exactly_once_at_the_requested_step(tiny_lm):
    plan = one_plan("blocks.1", prompt_length=4, steps=[2])
    outcome = _run(plan, tiny_lm)
    log = outcome.event_logs["plan"]
    assert log.all_fired
    assert log.missing_events == []
    assert log.not_reached_events == []
    assert len(log.events) == 1
    event = log.events[0]
    assert event.generation_step == 2
    assert event.forward_invocation == 0
    assert event.execution_result == "applied"
    assert event.module == "blocks.1"
    assert event.tensor_shape[1] == 4 + 2  # prefix plus two generated tokens
    assert event.rows == [4 + 2 - 1]  # position_mode="last"
    assert event.token_indices == [5]


def test_one_time_intervention_does_not_persist_into_later_steps(tiny_lm):
    # A one-time intervention changes the output token, but the next uncached
    # forward recomputes the prefix from scratch; only the token sequence carries
    # the change forward.  The event log is what states this explicitly.
    plan = one_plan("blocks.1", prompt_length=4, steps=[1], max_new_tokens=4)
    outcome = _run(plan, tiny_lm)
    log = outcome.event_logs["plan"]
    assert len(log.events) == 1
    assert log.metadata["one_time_interventions"] == [
        plan.interventions[0].plan_intervention_id
    ]
    assert "not present in the next" in outcome.record.metadata["uncached_note"]


def test_repeated_intervention_fires_on_every_scheduled_step(tiny_lm):
    plan = one_plan("blocks.1", prompt_length=4, steps=[1, 2, 3], max_new_tokens=4)
    outcome = _run(plan, tiny_lm)
    log = outcome.event_logs["plan"]
    assert [event.generation_step for event in log.events] == [1, 2, 3]
    assert outcome.record.metadata["sustained_interventions"]


def test_events_beyond_the_continuation_are_reported_as_not_reached(tiny_lm):
    plan = one_plan("blocks.1", prompt_length=4, steps=[0, 5], max_new_tokens=3)
    outcome = _run(plan, tiny_lm)
    log = outcome.event_logs["plan"]
    assert [event.plan_intervention_id for event in log.not_reached_events] == [
        plan.interventions[0].plan_intervention_id
    ] * 1
    assert log.not_reached_events[0].generation_step == 5
    assert any("not reached" in item for item in outcome.record.warnings)
    # Not reached is not the same as lost, so the experiment stays valid.
    assert outcome.record.validity.status == "valid"


def test_a_scheduled_event_that_never_fires_invalidates_the_run(tiny_lm):
    # The module is invoked once per forward, so call index 1 never occurs.
    spec = make_spec("blocks.1", prompt_length=4, label="second-call")
    item = make_plan_intervention(
        spec, order=0, schedule=[1], position_mode="last", on_step_call_indices=[1]
    )
    plan = make_plan([item], prompt_length=4, max_new_tokens=3)
    outcome = _run(plan, tiny_lm)
    log = outcome.event_logs["plan"]
    assert log.events == []
    assert len(log.missing_events) == 1
    assert log.missing_events[0].generation_step == 1
    assert outcome.record.validity.status == "invalid"
    assert any("did not fire" in item for item in outcome.record.validity.reasons)


def test_identity_can_be_disabled_and_marks_the_run_limited(tiny_lm):
    plan = one_plan("blocks.1", prompt_length=4, steps=[1])
    outcome = _run(plan, tiny_lm, identity=False)
    assert outcome.record.identity_control.performed is False
    assert outcome.record.validity.status == "limited"
    assert "identity" not in outcome.event_logs
    assert "identity" not in outcome.record.metadata["event_log_ids"]


def test_parameters_and_mutable_state_are_preserved(tiny_lm):
    model = tiny_lm.model
    params_before = parameter_fingerprint(model)
    state_before = mutable_state_fingerprint(model)
    hooks_before = installed_hook_count(model)
    plan = one_plan("blocks.1", prompt_length=4, steps=[0, 1], max_new_tokens=3)
    _run(plan, tiny_lm)
    assert parameter_fingerprint(model) == params_before
    assert mutable_state_fingerprint(model) == state_before
    assert installed_hook_count(model) == hooks_before
    checks = {check.name: check.status for check in (
        _run(plan, tiny_lm).record.validity.checks
    )}
    assert checks["parameters_unchanged"] == "pass"
    assert checks["mutable_state_unchanged"] == "pass"


def test_a_mutated_parameter_fails_the_protection_check(tiny_lm):
    plan = one_plan("blocks.1", prompt_length=4, steps=[0])
    model = tiny_lm.model
    before = parameter_fingerprint(model)

    original = model.blocks[1].mlp[-1].weight
    saved = original.detach().clone()

    # Simulate a hook that illegally writes a parameter.  The executor must
    # notice and refuse to call the run valid.
    handle = model.blocks[1].register_forward_hook(
        lambda module, args, output: original.data.add_(1e-3)
    )
    try:
        with pytest.raises(ExecutionError):
            _run(plan, tiny_lm)
    finally:
        handle.remove()
        with torch.no_grad():
            original.data.copy_(saved)
    assert parameter_fingerprint(model) == before


def test_multiple_interventions_on_different_modules_both_fire(tiny_lm):
    spec_a = make_spec("blocks.1", prompt_length=4, label="a")
    spec_b = make_spec("blocks.2", prompt_length=4, label="b")
    plan = make_plan(
        [
            make_plan_intervention(spec_a, order=0, schedule=[1], position_mode="last"),
            make_plan_intervention(spec_b, order=1, schedule=[1], position_mode="last"),
        ],
        prompt_length=4,
        max_new_tokens=3,
    )
    log = _run(plan, tiny_lm).event_logs["plan"]
    assert {(event.module, event.generation_step) for event in log.events} == {
        ("blocks.1", 1),
        ("blocks.2", 1),
    }


def test_same_module_composition_records_the_explicit_order(tiny_lm):
    zero = make_spec("blocks.1", prompt_length=4, label="zero", allow_overlap=True)
    patch = make_spec(
        "blocks.1", prompt_length=4, label="identity-after-zero", allow_overlap=True
    )
    plan = make_plan(
        [
            make_plan_intervention(zero, order=0, schedule=[1], position_mode="last"),
            make_plan_intervention(
                patch, order=1, schedule=[1], position_mode="last", label="second"
            ),
        ],
        prompt_length=4,
        max_new_tokens=3,
    )
    log = _run(plan, tiny_lm).event_logs["plan"]
    ordered = sorted(log.events, key=lambda event: event.event_index)
    assert [event.plan_intervention_id for event in ordered] == [
        item.plan_intervention_id for item in plan.ordered()
    ]
    # The second operation sees the tensor produced by the first.
    assert ordered[1].max_abs is not None


def test_preview_resolves_positions_without_changing_the_model(tiny_lm):
    plan = one_plan("blocks.1", prompt_length=4, steps=[1, 2], max_new_tokens=4)
    before = parameter_fingerprint(tiny_lm.model)
    preview = preview_plan(plan, tiny_lm.replay_model(), _prompt(tiny_lm))
    assert preview.valid
    assert [firing.generation_step for firing in preview.firings] == [1, 2]
    assert [firing.rows for firing in preview.firings] == [[4], [5]]
    assert preview.firings[0].tensor_shape == [1, 5, 32]
    assert any("resolved firings" in line for line in preview.human_readable)
    assert parameter_fingerprint(tiny_lm.model) == before


def test_preview_of_an_unreachable_schedule_says_so(tiny_lm):
    plan = one_plan("blocks.1", prompt_length=4, steps=[0, 9], max_new_tokens=2)
    preview = preview_plan(plan, tiny_lm.replay_model(), _prompt(tiny_lm))
    assert len(preview.firings) == 1
    assert any("not reached" in item for item in preview.warnings)
    assert any("not reached:" in line for line in preview.human_readable)


def test_stepwise_logits_are_bounded_by_the_observation_limit(tiny_lm):
    plan = one_plan("blocks.1", prompt_length=4, steps=[0], max_new_tokens=6)
    from wsp.interventions.plan import TemporalLimits

    plan = plan.model_copy(
        update={
            "limits": TemporalLimits(max_observed_logit_rows=2),
        }
    )
    outcome = _run(plan, tiny_lm)
    assert len(outcome.record.stepwise_logits) <= 2
    assert any("truncated" in item for item in outcome.record.warnings)


def test_absolute_position_targets_a_prompt_token(tiny_lm):
    spec = make_spec("blocks.1", prompt_length=4, positions=[1], label="absolute")
    plan = make_plan(
        [make_plan_intervention(spec, order=0, schedule=[0], position_mode="absolute")],
        prompt_length=4,
        max_new_tokens=2,
    )
    log = _run(plan, tiny_lm).event_logs["plan"]
    event = log.events[0]
    assert event.generation_step == 0
    assert event.position_mode == "absolute"
    assert event.rows == [1]
    assert event.token_indices == [1]


def test_execution_mode_guard_rejects_staged_plans(tiny_lm):
    plan = one_plan(
        "blocks.1",
        prompt_length=4,
        steps=[0],
        execution_mode="staged-generation",
    )
    with pytest.raises(Exception) as excinfo:
        _run(plan, tiny_lm)
    assert "staged" in str(excinfo.value)


def test_a_zero_ablation_changes_logits_and_is_recorded(tiny_lm):
    plan = one_plan("blocks.2", prompt_length=4, steps=[1], max_new_tokens=3)
    outcome = _run(plan, tiny_lm)
    log = outcome.event_logs["plan"]
    event = log.events[0]
    assert event.operation == "zero"
    assert event.elements_modified == 32
    assert event.after.l2_norm == 0.0
    assert event.before.l2_norm is not None and event.before.l2_norm > 0
    assert outcome.record.comparison.baseline_identity_match is True
