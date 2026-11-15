"""Tier 1/2: individual-versus-combined comparisons and interaction contrasts."""

from __future__ import annotations

import math

import pytest

from support.temporal import make_plan, make_plan_intervention, make_spec, one_plan
from wsp.core.errors import CompatibilityError
from wsp.interventions.composition import run_compositional_comparison
from wsp.interventions.models import IdentityControl


def _prompt(runtime) -> list[int]:
    return runtime.tokenizer.encode("abc", add_special_tokens=True)


def _plans(prompt_length: int = 4, *, steps=(1, 2), max_new_tokens: int = 4):
    zero = make_spec("blocks.1", prompt_length=prompt_length, label="a")
    patch = make_spec("blocks.2", prompt_length=prompt_length, label="b")
    plan_a = make_plan(
        [make_plan_intervention(zero, order=0, schedule=list(steps), position_mode="last")],
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        seed="a",
    )
    plan_b = make_plan(
        [make_plan_intervention(patch, order=0, schedule=list(steps), position_mode="last")],
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        seed="b",
    )
    plan_ab = make_plan(
        [
            make_plan_intervention(zero, order=0, schedule=list(steps), position_mode="last"),
            make_plan_intervention(patch, order=1, schedule=list(steps), position_mode="last"),
        ],
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        seed="ab",
    )
    return plan_a, plan_b, plan_ab


def test_comparison_records_baseline_identity_a_b_and_ab(tiny_lm):
    plan_a, plan_b, plan_ab = _plans()
    result = run_compositional_comparison(
        plan_a, plan_b, plan_ab, tiny_lm.replay_model(), _prompt(tiny_lm)
    )
    record = result.record
    names = [condition.condition for condition in record.conditions]
    assert names == ["baseline", "identity", "A", "B", "AB"]
    assert record.identity_control.performed is True
    assert record.identity_control.agreement == "exact"
    assert record.validity.status == "valid"
    assert set(result.event_logs) == {"baseline", "identity", "A", "B", "AB"}
    for name in ("A", "B", "AB"):
        assert result.event_logs[name].all_fired


def test_interaction_is_exactly_the_declared_contrast(tiny_lm):
    plan_a, plan_b, plan_ab = _plans()
    record = run_compositional_comparison(
        plan_a, plan_b, plan_ab, tiny_lm.replay_model(), _prompt(tiny_lm)
    ).record
    primary = record.interactions[0]
    assert primary.metric == "mean_generated_logprob"
    assert primary.continuous is True
    expected = (primary.y_ab - primary.y_baseline) - (
        primary.y_a - primary.y_baseline
    ) - (primary.y_b - primary.y_baseline)
    assert primary.interaction == pytest.approx(expected, abs=1e-12)
    assert primary.delta_a == pytest.approx(primary.y_a - primary.y_baseline)
    assert primary.delta_ab == pytest.approx(primary.y_ab - primary.y_baseline)


def test_interaction_does_not_claim_a_circuit(tiny_lm):
    plan_a, plan_b, plan_ab = _plans()
    record = run_compositional_comparison(
        plan_a, plan_b, plan_ab, tiny_lm.replay_model(), _prompt(tiny_lm)
    ).record
    assert "does not establish a functional circuit" in record.metadata["note"]
    assert "mechanism" in record.interpretation.lower()
    assert "not a discovered mechanism" in record.interpretation.lower()


def test_pairwise_measurements_include_divergence_and_termination(tiny_lm):
    plan_a, plan_b, plan_ab = _plans()
    record = run_compositional_comparison(
        plan_a, plan_b, plan_ab, tiny_lm.replay_model(), _prompt(tiny_lm)
    ).record
    labels = {(pair.condition, pair.reference) for pair in record.pairwise}
    assert {("A", "baseline"), ("B", "baseline"), ("AB", "baseline")} <= labels
    assert {("AB", "A"), ("AB", "B")} <= labels
    for pair in record.pairwise:
        assert pair.token_agreement_rate is None or 0.0 <= pair.token_agreement_rate <= 1.0
        assert pair.logit_l2_vs_reference is not None
        assert pair.stepwise  # bounded per-step logit rows


def test_discrete_metric_is_marked_as_discrete(tiny_lm):
    plan_a, plan_b, plan_ab = _plans()
    record = run_compositional_comparison(
        plan_a,
        plan_b,
        plan_ab,
        tiny_lm.replay_model(),
        _prompt(tiny_lm),
        metric="generated_length",
    ).record
    primary = record.interactions[0]
    assert primary.metric == "generated_length"
    assert primary.continuous is False
    # A continuous secondary scale is still available for interpretation.
    assert any(item.metric == "mean_generated_logprob" for item in record.interactions)
    assert any("discrete" in item for item in record.warnings)


def test_logit_l2_metric_is_zero_at_baseline(tiny_lm):
    plan_a, plan_b, plan_ab = _plans()
    record = run_compositional_comparison(
        plan_a,
        plan_b,
        plan_ab,
        tiny_lm.replay_model(),
        _prompt(tiny_lm),
        metric="logit_l2_vs_baseline",
    ).record
    assert record.interactions[0].y_baseline == 0.0


def test_a_failed_identity_control_invalidates_the_comparison(tiny_lm, monkeypatch):
    plan_a, plan_b, plan_ab = _plans()
    import wsp.interventions.composition as composition

    def failed(baseline, identity):
        return IdentityControl(
            performed=True,
            passed=False,
            agreement="failed",
            max_abs_logits_diff=1.0,
            detail="simulated instrumentation defect",
        )

    monkeypatch.setattr(composition, "_identity_from_uncached", failed)
    record = run_compositional_comparison(
        plan_a, plan_b, plan_ab, tiny_lm.replay_model(), _prompt(tiny_lm)
    ).record
    assert record.validity.status == "invalid"
    assert any("identity control" in item for item in record.validity.reasons)


def test_without_an_identity_control_the_comparison_is_limited(tiny_lm):
    plan_a, plan_b, plan_ab = _plans()
    record = run_compositional_comparison(
        plan_a,
        plan_b,
        plan_ab,
        tiny_lm.replay_model(),
        _prompt(tiny_lm),
        identity=False,
    ).record
    assert record.validity.status == "limited"
    assert "identity" not in [condition.condition for condition in record.conditions]


def test_plans_from_different_prompts_are_rejected(tiny_lm):
    plan_a, plan_b, plan_ab = _plans()
    other = one_plan("blocks.1", prompt_length=6, steps=[0], max_new_tokens=2, seed="other")
    with pytest.raises(CompatibilityError):
        run_compositional_comparison(
            other, plan_b, plan_ab, tiny_lm.replay_model(), _prompt(tiny_lm)
        )


def test_plans_from_different_execution_modes_are_rejected(tiny_lm):
    plan_a, plan_b, plan_ab = _plans()
    staged = one_plan(
        "blocks.1",
        prompt_length=4,
        steps=[0],
        execution_mode="staged-generation",
        max_new_tokens=2,
        seed="staged",
    )
    with pytest.raises(CompatibilityError):
        run_compositional_comparison(
            staged, plan_b, plan_ab, tiny_lm.replay_model(), _prompt(tiny_lm)
        )


def test_negative_control_is_recorded_without_interpretation(tiny_lm):
    plan_a, plan_b, plan_ab = _plans()
    control = one_plan(
        "blocks.1", prompt_length=4, steps=[1, 2], max_new_tokens=4, seed="control"
    )
    record = run_compositional_comparison(
        plan_a,
        plan_b,
        plan_ab,
        tiny_lm.replay_model(),
        _prompt(tiny_lm),
        negative_control=control,
    ).record
    names = [condition.condition for condition in record.conditions]
    assert "negative-control" in names
    assert record.negative_control_plan_id == control.plan_id
    assert any("negative control" in item for item in record.warnings)


def test_comparison_checksum_is_stable_over_time(tiny_lm):
    plan_a, plan_b, plan_ab = _plans()
    record = run_compositional_comparison(
        plan_a, plan_b, plan_ab, tiny_lm.replay_model(), _prompt(tiny_lm)
    ).record
    from wsp.interventions.plan_models import composition_sha256

    stored = record.checksums["composition_sha256"]
    assert stored == composition_sha256(record)
