"""Individual versus combined intervention measurements.

The measurement this module produces is descriptive.  For one explicitly named
scalar outcome it reports

    I_AB = (Y_AB - Y_0) - (Y_A - Y_0) - (Y_B - Y_0)

which is non-additivity on that scale.  It is not a circuit discovery, and a
value near zero is not evidence of independence.  Discrete outcomes such as
generated tokens are compared as sequences and reported as divergence, never
averaged into a continuous quantity.

Every condition is executed from the same prefix, with the same decoding
settings, the same model object, and the same numerical precision.  The
execution mode is recorded once and applied to all conditions; conditions from
different modes are never combined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from wsp.core.budget import Budget
from wsp.core.errors import CompatibilityError, ExecutionError, ResourceLimitError, UnsupportedError
from wsp.core.hashing import sha256_bytes
from wsp.core.time import utc_now_iso
from wsp.execution.staged import StagedDecoder, build_staged_decoder
from wsp.tensors.diff import compare_arrays

from .engine import describe_model, torch_module
from .models import IdentityControl, ValidationCheck, ValidityReport
from .plan import InterventionPlan, identity_control_plan, plan_sha256, validate_plan
from .plan_engine import (
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    PlanRunOutcome,
    _condition_outcome,
    _continuation_diff,
    _event_log_for,
    _generate_uncached,
    _checked_outcome,
)
from .plan_models import (
    CompositionalComparison,
    InteractionContrast,
    InteractionMetric,
    PairwiseEffect,
    StepwiseLogitDiff,
    composition_sha256,
)
from .staged import _generate_staged, _identity_from_staged, verify_staged_generation

DISCRETE_METRICS = frozenset({"generated_length"})


@dataclass
class CompositionOutcome:
    """The record plus the per-condition event logs that belong with it."""

    record: CompositionalComparison
    event_logs: dict[str, Any] = field(default_factory=dict)


def _metric_value(metric: InteractionMetric, outcome: PlanRunOutcome) -> float | None:
    if metric == "mean_generated_logprob":
        return (
            float(sum(outcome.logprobs) / len(outcome.logprobs)) if outcome.logprobs else None
        )
    if metric == "final_token_logprob":
        return float(outcome.logprobs[-1]) if outcome.logprobs else None
    if metric == "generated_length":
        return float(len(outcome.generated_token_ids))
    return None


def _logit_l2(outcome: PlanRunOutcome, reference: PlanRunOutcome) -> float | None:
    if outcome.step_logits is None or reference.step_logits is None:
        return None
    steps = min(int(outcome.step_logits.shape[0]), int(reference.step_logits.shape[0]))
    if steps == 0:
        return None
    delta = outcome.step_logits[:steps] - reference.step_logits[:steps]
    denominator = float(torch.linalg.vector_norm(reference.step_logits[:steps]).item())
    numerator = float(torch.linalg.vector_norm(delta).item())
    if denominator == 0:
        return None
    return numerator / denominator


def _pairwise(
    name: str,
    reference_name: str,
    outcome: PlanRunOutcome,
    reference: PlanRunOutcome,
    *,
    budget: Budget,
) -> PairwiseEffect:
    diff = _continuation_diff(
        reference.generated_token_ids,
        outcome.generated_token_ids,
        baseline_termination=reference.termination,
        other_termination=outcome.termination,
    )
    stats = None
    stepwise: list[StepwiseLogitDiff] = []
    if outcome.step_logits is not None and reference.step_logits is not None:
        steps = min(int(outcome.step_logits.shape[0]), int(reference.step_logits.shape[0]))
        if steps:
            left = reference.step_logits[:steps]
            right = outcome.step_logits[:steps]
            elements = int(left.numel())
            if elements > budget.max_comparison_elements:
                raise ResourceLimitError(
                    "the stepwise logit comparison exceeds the configured budget",
                    details={"elements": elements, "limit": budget.max_comparison_elements},
                )
            stats = compare_arrays(
                left.detach().to("cpu", torch.float64).numpy(),
                right.detach().to("cpu", torch.float64).numpy(),
                atol=IDENTITY_ATOL,
                rtol=IDENTITY_RTOL,
            )
            for step in range(steps):
                row_left = left[step]
                row_right = right[step]
                stepwise.append(
                    StepwiseLogitDiff(
                        step=step,
                        token_id=(
                            outcome.generated_token_ids[step]
                            if step < len(outcome.generated_token_ids)
                            else (
                                reference.generated_token_ids[step]
                                if step < len(reference.generated_token_ids)
                                else 0
                            )
                        ),
                        token_matches=(
                            step < len(reference.generated_token_ids)
                            and step < len(outcome.generated_token_ids)
                            and reference.generated_token_ids[step] == outcome.generated_token_ids[step]
                        ),
                        max_abs=float((row_right - row_left).abs().max().item()),
                        mean_abs=float((row_right - row_left).abs().mean().item()),
                        allclose=bool(
                            torch.allclose(row_right, row_left, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)
                        ),
                    )
                )
    return PairwiseEffect(
        condition=name,
        reference=reference_name,
        first_divergence=diff.first_divergence,
        longest_common_prefix=diff.longest_common_prefix,
        token_agreement_rate=diff.token_agreement_rate,
        length_delta=diff.length_delta,
        termination_changed=diff.termination_changed,
        logits=stats,
        logit_l2_vs_reference=_logit_l2(outcome, reference),
        stepwise=stepwise,
    )


def run_compositional_comparison(
    plan_a: InterventionPlan,
    plan_b: InterventionPlan,
    plan_ab: InterventionPlan,
    model: Any,
    prompt_token_ids: Sequence[int],
    *,
    metric: InteractionMetric = "mean_generated_logprob",
    store: Any | None = None,
    budget: Budget | None = None,
    case_id: str | None = None,
    plan_artifact_ids: dict[str, str] | None = None,
    identity: bool = True,
    negative_control: InterventionPlan | None = None,
    decoder: StagedDecoder | None = None,
    isolation: str = "recompute",
) -> CompositionalComparison:
    """Measure baseline, identity, A, B, and A+B over one prefix."""

    budget = budget or Budget()
    prompt = [int(item) for item in prompt_token_ids]
    plans = {"A": plan_a, "B": plan_b, "AB": plan_ab}
    for label, plan in plans.items():
        if len(prompt) != plan.prompt_length:
            raise CompatibilityError(
                f"plan {label} records a different prompt length",
                details={"plan": label, "prompt_length": plan.prompt_length, "prefix": len(prompt)},
            )
        if plan.execution_mode != plan_ab.execution_mode:
            raise CompatibilityError(
                "conditions executed in different execution modes must not be combined",
                details={
                    "plan": label,
                    "execution_mode": plan.execution_mode,
                    "comparison_mode": plan_ab.execution_mode,
                },
            )
        validation = validate_plan(plan, root=torch_module(model), budget=budget)
        if not validation.valid:
            raise CompatibilityError(
                f"plan {label} failed validation",
                details={"errors": validation.errors[:8]},
            )
    staged = plan_ab.execution_mode == "staged-generation"
    if staged:
        decoder = decoder or build_staged_decoder(model)
        verification, _timings = verify_staged_generation(
            plan_ab,
            model,
            prompt,
            budget=budget,
            decoder=decoder,
            identity=identity,
            isolation=isolation,
        )
        if verification.agreement not in ("exact", "within_tolerance"):
            raise ExecutionError(
                "staged execution did not reproduce the uncached reference; the "
                "compositional comparison would not be interpretable",
                details={"agreement": verification.agreement},
            )
    elif plan_ab.execution_mode != "uncached-generation":
        raise UnsupportedError(
            "compositional comparison supports uncached generation and verified staged "
            "generation in this release",
            details={"execution_mode": plan_ab.execution_mode},
        )

    def run(plan: InterventionPlan, *, identity: bool, apply: bool) -> PlanRunOutcome:
        if staged:
            run_result = _generate_staged(
                plan,
                decoder,  # type: ignore[arg-type]
                prompt,
                store=store,
                budget=budget,
                identity=identity,
                apply=apply,
                isolation=isolation,
            )
            return run_result.outcome
        return _generate_uncached(
            plan, model, prompt, store=store, budget=budget, identity=identity, apply=apply
        )

    baseline = run(plan_ab, identity=False, apply=False)
    _checked_outcome(baseline)
    if baseline.termination == "error":
        raise ExecutionError(
            "the baseline condition failed, so the comparison is not measurable",
            details={"errors": baseline.errors},
        )
    conditions = []
    outcomes: dict[str, PlanRunOutcome] = {"baseline": baseline}
    condition_plans: dict[str, InterventionPlan] = {"baseline": plan_ab}
    logs: dict[str, Any] = {}

    def record(name: str, plan: InterventionPlan, outcome: PlanRunOutcome) -> None:
        _checked_outcome(outcome)
        log = _event_log_for(
            outcome,
            condition=name,
            execution_mode=plan.execution_mode,
            backend=plan.backend,
            steps=outcome.steps,
        )
        logs[name] = log
        conditions.append(_condition_outcome(name, plan, outcome, log, model=model))

    record("baseline", plan_ab, baseline)

    identity_control = IdentityControl(performed=False)
    control_plan = identity_control_plan(plan_ab)
    if identity:
        identity_outcome = run(control_plan, identity=True, apply=True)
        record("identity", control_plan, identity_outcome)
        if staged:
            identity_control = _identity_from_staged(baseline, identity_outcome)
        else:
            identity_control = _identity_from_uncached(baseline, identity_outcome)
    else:
        identity_outcome = None

    outcomes["A"] = run(plan_a, identity=False, apply=True)
    condition_plans["A"] = plan_a
    outcomes["B"] = run(plan_b, identity=False, apply=True)
    condition_plans["B"] = plan_b
    outcomes["AB"] = run(plan_ab, identity=False, apply=True)
    condition_plans["AB"] = plan_ab
    for name in ("A", "B", "AB"):
        record(name, condition_plans[name], outcomes[name])

    negative_outcome = None
    if negative_control is not None:
        if negative_control.prompt_length != len(prompt):
            raise CompatibilityError(
                "the negative control was recorded against a different prompt length"
            )
        negative_outcome = run(negative_control, identity=False, apply=True)
        record("negative-control", negative_control, negative_outcome)

    pairwise = [
        _pairwise("A", "baseline", outcomes["A"], baseline, budget=budget),
        _pairwise("B", "baseline", outcomes["B"], baseline, budget=budget),
        _pairwise("AB", "baseline", outcomes["AB"], baseline, budget=budget),
        _pairwise("AB", "A", outcomes["AB"], outcomes["A"], budget=budget),
        _pairwise("AB", "B", outcomes["AB"], outcomes["B"], budget=budget),
    ]
    if negative_outcome is not None:
        pairwise.append(
            _pairwise("negative-control", "baseline", negative_outcome, baseline, budget=budget)
        )

    metric_values = {name: _metric_value(metric, outcome) for name, outcome in outcomes.items()}
    if metric == "logit_l2_vs_baseline":
        for name, outcome in outcomes.items():
            metric_values[name] = _logit_l2(outcome, baseline) if name != "baseline" else 0.0
    interactions: list[InteractionContrast] = []
    if all(metric_values.get(name) is not None for name in ("baseline", "A", "B", "AB")):
        y0 = float(metric_values["baseline"])  # type: ignore[arg-type]
        ya = float(metric_values["A"])  # type: ignore[arg-type]
        yb = float(metric_values["B"])  # type: ignore[arg-type]
        yab = float(metric_values["AB"])  # type: ignore[arg-type]
        delta_a = ya - y0
        delta_b = yb - y0
        delta_ab = yab - y0
        interactions.append(
            InteractionContrast(
                metric=metric,
                y_baseline=y0,
                y_a=ya,
                y_b=yb,
                y_ab=yab,
                delta_a=delta_a,
                delta_b=delta_b,
                delta_ab=delta_ab,
                interaction=delta_ab - delta_a - delta_b,
                continuous=metric not in DISCRETE_METRICS,
            )
        )
    else:
        interactions.append(
            InteractionContrast(metric=metric, continuous=metric not in DISCRETE_METRICS)
        )

    # A second, always-available continuous scale, so a discrete primary metric
    # does not leave the record without any non-additivity measurement.
    if metric != "mean_generated_logprob" and all(
        outcomes[name].logprobs for name in ("baseline", "A", "B", "AB")
    ):
        values = {
            name: float(sum(outcome.logprobs) / len(outcome.logprobs))
            for name, outcome in outcomes.items()
        }
        y0, ya, yb, yab = (
            values["baseline"],
            values["A"],
            values["B"],
            values["AB"],
        )
        interactions.append(
            InteractionContrast(
                metric="mean_generated_logprob",
                y_baseline=y0,
                y_a=ya,
                y_b=yb,
                y_ab=yab,
                delta_a=ya - y0,
                delta_b=yb - y0,
                delta_ab=yab - y0,
                interaction=(yab - y0) - (ya - y0) - (yb - y0),
                continuous=True,
            )
        )

    checks = [
        ValidationCheck(
            name="identity_control",
            status=(
                "pass" if identity_control.passed else "fail"
            ) if identity_control.performed else "skipped",
            detail=identity_control.detail,
        ),
        ValidationCheck(
            name="all_events_fired",
            status=(
                "pass"
                if all(log.all_fired for name, log in logs.items() if name not in ("baseline", "identity"))
                else "fail"
            ),
            detail="every scheduled event in A, B, and AB fired",
        ),
    ]
    reasons: list[str] = []
    status = "valid"
    if identity_control.performed and not identity_control.passed:
        status = "invalid"
        reasons.append(
            "the identity control disagreed with the baseline; individual and combined "
            "differences cannot be attributed to the interventions"
        )
    for name, log in logs.items():
        if name in ("baseline", "identity"):
            continue
        if log.missing_events or log.rejected_events:
            status = "invalid"
            reasons.append(f"condition '{name}' did not apply its full schedule")
    for name, outcome in outcomes.items():
        if outcome.termination == "error":
            status = "invalid"
            reasons.append(f"condition '{name}' failed: {'; '.join(outcome.errors)}")
    if not identity_control.performed:
        status = "limited" if status == "valid" else status
        reasons.append("no identity control was performed")

    warnings: list[str] = []
    if metric in DISCRETE_METRICS:
        warnings.append(
            f"metric '{metric}' is discrete; its interaction value is reported but should "
            "be read as an observation, not a continuous effect size"
        )
    if negative_control is not None:
        warnings.append(
            "a negative control was executed; its result is recorded, not interpreted"
        )
    if any(outcome.steps == 0 for outcome in outcomes.values()):
        warnings.append("a condition produced no generated tokens")

    record_artifact_ids = dict(plan_artifact_ids or {})
    comparison = CompositionalComparison(
        comparison_id=_new_composition_id(),
        created_at=utc_now_iso(),
        case_id=case_id,
        trace_id=plan_ab.provenance.get("trace_id") if isinstance(plan_ab.provenance, dict) else None,
        plan_a_id=plan_a.plan_id,
        plan_b_id=plan_b.plan_id,
        plan_ab_id=plan_ab.plan_id,
        plan_sha256={
            "A": plan_sha256(plan_a),
            "B": plan_sha256(plan_b),
            "AB": plan_sha256(plan_ab),
        },
        plan_artifact_ids=record_artifact_ids,
        identity_plan_id=control_plan.plan_id if identity_outcome is not None else None,
        negative_control_plan_id=(
            negative_control.plan_id if negative_control is not None else None
        ),
        model=describe_model(model),
        execution_mode=plan_ab.execution_mode,
        backend=plan_ab.backend,
        prompt_length=len(prompt),
        prompt_sha256=sha256_bytes((",".join(str(item) for item in prompt)).encode("utf-8")),
        config=plan_ab.continuation.model_dump(mode="json"),
        metric=metric,
        conditions=conditions,
        identity_control=identity_control,
        interactions=interactions,
        pairwise=pairwise,
        validity=ValidityReport(status=status, checks=checks, reasons=reasons),
        warnings=warnings,
        metadata={
            "event_log_ids": {name: log.log_id for name, log in logs.items()},
            "staged_verified": staged,
            "metric_values": {name: value for name, value in metric_values.items()},
            "note": (
                "non-additivity on one scale; it does not establish a functional circuit"
            ),
        },
    )
    comparison.checksums["composition_sha256"] = composition_sha256(comparison)
    return CompositionOutcome(record=comparison, event_logs=dict(logs))


def _identity_from_uncached(
    baseline: PlanRunOutcome, identity: PlanRunOutcome
) -> IdentityControl:
    tokens_match = baseline.generated_token_ids == identity.generated_token_ids
    if baseline.step_logits is None or identity.step_logits is None:
        return IdentityControl(
            performed=True,
            passed=tokens_match,
            agreement="exact" if tokens_match else "failed",
            detail="logit rows were not retained; agreement is based on greedy tokens only",
        )
    steps = min(int(baseline.step_logits.shape[0]), int(identity.step_logits.shape[0]))
    delta = (baseline.step_logits[:steps] - identity.step_logits[:steps]).abs()
    max_abs = float(delta.max().item()) if steps else None
    mean_abs = float(delta.mean().item()) if steps else None
    exact = bool(
        tokens_match
        and int(baseline.step_logits.shape[0]) == int(identity.step_logits.shape[0])
        and torch.equal(baseline.step_logits, identity.step_logits)
    )
    allclose = bool(
        torch.allclose(
            baseline.step_logits[:steps],
            identity.step_logits[:steps],
            atol=IDENTITY_ATOL,
            rtol=IDENTITY_RTOL,
        )
    )
    if exact:
        agreement, passed = "exact", True
        detail = "baseline and identity step logits are bitwise identical"
    elif allclose and tokens_match:
        agreement, passed = "within_tolerance", True
        detail = "baseline and identity agree within tolerance"
    else:
        agreement, passed = "failed", False
        detail = "multi-site instrumentation changed the computation"
    return IdentityControl(
        performed=True,
        passed=passed,
        agreement=agreement,  # type: ignore[arg-type]
        max_abs_logits_diff=max_abs,
        mean_abs_logits_diff=mean_abs,
        atol=IDENTITY_ATOL,
        rtol=IDENTITY_RTOL,
        detail=detail,
    )


def _new_composition_id() -> str:
    from .ids import new_composition_id

    return new_composition_id()


__all__ = [
    "DISCRETE_METRICS",
    "CompositionOutcome",
    "run_compositional_comparison",
]
