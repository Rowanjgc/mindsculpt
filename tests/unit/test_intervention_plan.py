"""Tier 1: multi-site intervention plans, ordering, conflicts, identity control."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from support.temporal import make_plan, make_plan_intervention, make_spec, one_plan
from wsp.interventions.plan import (
    SCHEMA_PLAN,
    PlanContinuationConfig,
    TemporalLimits,
    find_conflicts,
    identity_control_plan,
    plan_sha256,
    renumber,
    validate_plan,
)


def test_plan_serializes_with_the_expected_schema():
    plan = one_plan("blocks.1", prompt_length=4, steps=[1])
    payload = plan.model_dump(mode="json")
    assert payload["schema_version"] == SCHEMA_PLAN
    assert payload["plan_id"].startswith("pl_")
    assert payload["timing_conventions"]["generation_step"]


def test_plan_checksum_ignores_creation_time():
    plan = one_plan("blocks.1", prompt_length=4, steps=[1])
    digest = plan_sha256(plan)
    changed = plan.model_copy(update={"created_at": "2030-01-01T00:00:00Z"})
    assert plan_sha256(changed) == digest
    changed_name = plan.model_copy(update={"name": "another name"})
    assert plan_sha256(changed_name) != digest


def test_identity_control_plan_preserves_sites_and_schedule():
    plan = one_plan("blocks.1", prompt_length=5, steps=[1, 3])
    control = identity_control_plan(plan)
    assert control.plan_kind == "identity-control"
    assert control.plan_id != plan.plan_id
    assert [item.order for item in control.ordered()] == [item.order for item in plan.ordered()]
    assert [item.schedule.resolve() for item in control.ordered()] == [[1, 3]]
    for item, original in zip(control.ordered(), plan.ordered()):
        assert item.operation == "identity"
        assert item.position_mode == original.position_mode
        assert item.on_step_call_indices == original.on_step_call_indices
        assert item.intervention.intervention_id != original.intervention.intervention_id
        assert item.intervention.mean is None
        assert item.intervention.donor is None
    assert control.metadata["derived_from_plan_id"] == plan.plan_id


def test_identity_plan_is_distinct_from_an_empty_plan():
    plan = one_plan("blocks.1", prompt_length=4, steps=[0])
    control = identity_control_plan(plan)
    # An identity plan keeps the instrumentation; an empty plan would not.
    assert len(control.interventions) == len(plan.interventions)
    with pytest.raises(ValidationError):
        make_plan([], prompt_length=4)


def test_intervention_orders_must_be_unique():
    spec_a = make_spec("blocks.1", prompt_length=4, label="a")
    spec_b = make_spec("blocks.2", prompt_length=4, label="b")
    with pytest.raises(ValidationError):
        make_plan(
            [
                make_plan_intervention(spec_a, order=0),
                make_plan_intervention(spec_b, order=0),
            ],
            prompt_length=4,
        )


def test_plan_rejects_a_spec_recorded_against_a_different_prompt_length():
    spec = make_spec("blocks.1", prompt_length=4)
    item = make_plan_intervention(spec, order=0)
    with pytest.raises(ValidationError):
        make_plan([item], prompt_length=6)


def test_plan_rejects_a_spec_with_the_wrong_execution_mode():
    spec = make_spec("blocks.1", prompt_length=4, execution_mode="teacher-forced")
    item = make_plan_intervention(spec, order=0)
    with pytest.raises(ValidationError):
        make_plan([item], prompt_length=4, execution_mode="uncached-generation")


def test_plan_rejects_absolute_positions_outside_the_prefix():
    # The nested record refuses a position beyond its sequence length, so the
    # plan can never be built from it in the first place.
    with pytest.raises(ValidationError):
        make_spec("blocks.1", prompt_length=4, positions=[4])


def test_last_position_mode_requires_the_documented_sentinel():
    spec = make_spec("blocks.1", prompt_length=4, positions=[2])
    with pytest.raises(ValidationError):
        make_plan_intervention(spec, order=0, position_mode="last")


def test_interventions_at_distinct_modules_do_not_conflict():
    spec_a = make_spec("blocks.1", prompt_length=4, label="a")
    spec_b = make_spec("blocks.2", prompt_length=4, label="b")
    plan = make_plan(
        [
            make_plan_intervention(spec_a, order=0, schedule=[1]),
            make_plan_intervention(spec_b, order=1, schedule=[1]),
        ],
        prompt_length=4,
    )
    assert find_conflicts(plan) == []
    assert validate_plan(plan).valid


def test_interventions_on_the_same_region_and_step_conflict():
    spec_a = make_spec("blocks.1", prompt_length=4, label="a")
    spec_b = make_spec("blocks.1", prompt_length=4, label="b")
    plan = make_plan(
        [
            make_plan_intervention(spec_a, order=0, schedule=[1], position_mode="last"),
            make_plan_intervention(spec_b, order=1, schedule=[1], position_mode="last"),
        ],
        prompt_length=4,
    )
    conflicts = find_conflicts(plan)
    assert len(conflicts) == 1
    assert conflicts[0].kind == "duplicate-site"
    validation = validate_plan(plan)
    assert validation.valid is False
    assert any("overlap" in item for item in validation.errors)


def test_partially_overlapping_rows_report_overlapping_region():
    spec_a = make_spec("blocks.1", prompt_length=4, positions=[0, 1], label="a")
    spec_b = make_spec("blocks.1", prompt_length=4, positions=[1, 2], label="b")
    plan = make_plan(
        [
            make_plan_intervention(spec_a, order=0, schedule=[0], position_mode="absolute"),
            make_plan_intervention(spec_b, order=1, schedule=[0], position_mode="absolute"),
        ],
        prompt_length=4,
    )
    conflicts = find_conflicts(plan)
    assert len(conflicts) == 1
    assert conflicts[0].kind == "overlapping-region"


def test_disjoint_rows_on_the_same_module_do_not_conflict():
    spec_a = make_spec(
        "blocks.1", prompt_length=4, positions=[0], label="a", feature_start=0, feature_end=4
    )
    spec_b = make_spec(
        "blocks.1", prompt_length=4, positions=[1], label="b", feature_start=4, feature_end=8
    )
    plan = make_plan(
        [
            make_plan_intervention(spec_a, order=0, schedule=[0], position_mode="absolute"),
            make_plan_intervention(spec_b, order=1, schedule=[0], position_mode="absolute"),
        ],
        prompt_length=4,
    )
    assert find_conflicts(plan) == []


def test_disjoint_steps_do_not_conflict_on_the_same_region():
    spec_a = make_spec("blocks.1", prompt_length=4, label="a")
    spec_b = make_spec("blocks.1", prompt_length=4, label="b")
    plan = make_plan(
        [
            make_plan_intervention(spec_a, order=0, schedule=[0], position_mode="last"),
            make_plan_intervention(spec_b, order=1, schedule=[1], position_mode="last"),
        ],
        prompt_length=4,
    )
    assert find_conflicts(plan) == []


def test_disjoint_features_do_not_conflict_on_the_same_module():
    spec_a = make_spec(
        "blocks.1", prompt_length=4, label="a", feature_start=0, feature_end=4
    )
    spec_b = make_spec(
        "blocks.1", prompt_length=4, label="b", feature_start=4, feature_end=8
    )
    plan = make_plan(
        [
            make_plan_intervention(spec_a, order=0, schedule=[0], position_mode="last"),
            make_plan_intervention(spec_b, order=1, schedule=[0], position_mode="last"),
        ],
        prompt_length=4,
    )
    assert find_conflicts(plan) == []


def test_overlap_is_allowed_only_with_explicit_intent():
    spec_a = make_spec("blocks.1", prompt_length=4, label="a", allow_overlap=True)
    spec_b = make_spec("blocks.1", prompt_length=4, label="b", allow_overlap=True)
    plan = make_plan(
        [
            make_plan_intervention(spec_a, order=0, schedule=[1], position_mode="last"),
            make_plan_intervention(spec_b, order=1, schedule=[1], position_mode="last"),
        ],
        prompt_length=4,
    )
    assert find_conflicts(plan) == []
    validation = validate_plan(plan)
    assert validation.valid is True
    # The order is explicit and recorded, not inferred.
    assert [item.order for item in plan.ordered()] == [0, 1]


def test_partial_overlap_intent_still_conflicts():
    spec_a = make_spec("blocks.1", prompt_length=4, label="a", allow_overlap=True)
    spec_b = make_spec("blocks.1", prompt_length=4, label="b")
    plan = make_plan(
        [
            make_plan_intervention(spec_a, order=0, schedule=[1], position_mode="last"),
            make_plan_intervention(spec_b, order=1, schedule=[1], position_mode="last"),
        ],
        prompt_length=4,
    )
    assert len(find_conflicts(plan)) == 1


def test_distinct_call_indices_do_not_conflict():
    spec_a = make_spec("blocks.1", prompt_length=4, label="a")
    spec_b = make_spec("blocks.1", prompt_length=4, label="b")
    plan = make_plan(
        [
            make_plan_intervention(
                spec_a, order=0, schedule=[1], position_mode="last", on_step_call_indices=[0]
            ),
            make_plan_intervention(
                spec_b, order=1, schedule=[1], position_mode="last", on_step_call_indices=[1]
            ),
        ],
        prompt_length=4,
    )
    assert find_conflicts(plan) == []


def test_plan_limits_reject_excessive_events():
    limits = TemporalLimits(max_scheduled_events=2)
    spec = make_spec("blocks.1", prompt_length=4)
    item = make_plan_intervention(spec, order=0, schedule=[0, 1, 2])
    with pytest.raises(ValidationError):
        make_plan([item], prompt_length=4, limits=limits)


def test_plan_limits_reject_too_many_interventions():
    limits = TemporalLimits(max_interventions=1)
    spec_a = make_spec("blocks.1", prompt_length=4, label="a")
    spec_b = make_spec("blocks.2", prompt_length=4, label="b")
    with pytest.raises(ValidationError):
        make_plan(
            [
                make_plan_intervention(spec_a, order=0),
                make_plan_intervention(spec_b, order=1),
            ],
            prompt_length=4,
            limits=limits,
        )


def test_staged_plans_reject_later_absolute_positions():
    plan = make_plan(
        [
            make_plan_intervention(
                make_spec("blocks.1", prompt_length=4, positions=[0]),
                order=0,
                schedule=[2],
                position_mode="absolute",
            )
        ],
        prompt_length=4,
        execution_mode="staged-generation",
    )
    validation = validate_plan(plan)
    assert validation.valid is False
    assert any("staged" in item for item in validation.errors)


def test_staged_plan_accepts_absolute_positions_at_step_zero():
    plan = make_plan(
        [
            make_plan_intervention(
                make_spec("blocks.1", prompt_length=4, positions=[0]),
                order=0,
                schedule=[0],
                position_mode="absolute",
            )
        ],
        prompt_length=4,
        execution_mode="staged-generation",
    )
    assert validate_plan(plan).valid


def test_continuation_budget_is_enforced():
    plan = make_plan(
        [
            make_plan_intervention(
                make_spec("blocks.1", prompt_length=4), order=0, schedule=[0]
            )
        ],
        prompt_length=4,
        max_new_tokens=128,
        limits=TemporalLimits(max_continuation_tokens=128),
    )
    validation = validate_plan(plan)
    assert validation.valid is False
    assert any("token budget" in item for item in validation.errors)


def test_cost_estimate_accounts_for_uncached_recomputation():
    plan = one_plan(
        "blocks.1", prompt_length=4, steps=[0], max_new_tokens=6
    )
    validation = validate_plan(plan)
    # Six forwards over a prefix that grows by one token each time.
    assert validation.estimated_forward_calls == 6
    assert validation.estimated_token_positions == sum(range(4, 10))


def test_validate_resolves_modules_against_a_live_model():
    from support.tiny_lm import build_runtime

    runtime = build_runtime()
    plan = one_plan("blocks.1", prompt_length=4, steps=[0])
    validation = validate_plan(plan, root=runtime.model)
    assert validation.valid
    assert validation.resolved[0]["module_class"] == "TinyBlock"


def test_validate_reports_an_unknown_module():
    from support.tiny_lm import build_runtime

    runtime = build_runtime()
    plan = one_plan("blocks.99", prompt_length=4, steps=[0])
    validation = validate_plan(plan, root=runtime.model)
    assert validation.valid is False
    assert any("blocks.99" in item for item in validation.errors)


def test_renumber_recomputes_the_checksum_without_changing_identity():
    plan = one_plan("blocks.1", prompt_length=4, steps=[1])
    rebuilt = renumber(plan)
    assert rebuilt.plan_id == plan.plan_id
    assert rebuilt.checksums["plan_sha256"] == plan_sha256(rebuilt)


def test_prompt_token_ids_must_match_prompt_length():
    plan = one_plan("blocks.1", prompt_length=4, steps=[0])
    with pytest.raises(ValidationError):
        plan.model_copy(update={"prompt_token_ids": [1, 2]}).model_validate(
            {**plan.model_dump(), "prompt_token_ids": [1, 2]}
        )


def test_plan_continuation_is_greedy_and_uncached():
    config = PlanContinuationConfig()
    assert config.strategy == "greedy"
    assert config.do_sample is False
    assert config.temperature == 0.0
