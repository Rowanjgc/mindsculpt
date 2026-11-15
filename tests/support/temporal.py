"""Helpers for building temporal intervention plans in tests.

These are test-only conveniences.  The library itself never fabricates a plan
from less information than :class:`~wsp.interventions.plan.InterventionPlan`
declares.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from wsp.behavior.models import ModelRef
from wsp.core.hashing import sha256_bytes
from wsp.core.time import utc_now_iso
from wsp.interventions.models import (
    ActivationIntervention,
    ExecutionIdentity,
    InterventionTarget,
    TensorSelector,
)
from wsp.interventions.plan import (
    InterventionPlan,
    PlanContinuationConfig,
    PlanIntervention,
    TemporalLimits,
)
from wsp.interventions.schedule import parse_schedule


def _identifier(prefix: str, seed: str) -> str:
    return f"{prefix}_{sha256_bytes(seed.encode()).split(':', 1)[1][:16]}"


def make_spec(
    module: str,
    *,
    prompt_length: int,
    positions: Sequence[int] = (0,),
    operation: str = "zero",
    execution_mode: str = "uncached-continuation",
    feature_start: int | None = None,
    feature_end: int | None = None,
    feature_indices: Sequence[int] = (),
    tensor_index: int | None = None,
    tensor_name: str | None = None,
    tensor_role: str = "output",
    position_semantics: str = "consuming",
    label: str = "spec",
    allow_overlap: bool = False,
) -> ActivationIntervention:
    """Build a minimal, valid intervention specification for a fixture model."""

    selector = TensorSelector(
        positions=list(positions),
        feature_start=feature_start,
        feature_end=feature_end,
        feature_indices=list(feature_indices),
        tensor_index=tensor_index,
        tensor_name=tensor_name,
    )
    return ActivationIntervention(
        intervention_id=_identifier("iv", f"{module}:{label}:{positions}:{operation}"),
        created_at=utc_now_iso(),
        name=label,
        model=ModelRef(name="tiny-causal-lm"),
        execution=ExecutionIdentity(
            sequence_length=prompt_length,
            execution_mode=execution_mode,
        ),
        target=InterventionTarget(
            module=module,
            tensor_role=tensor_role,
            position_semantics=position_semantics,
            selector=selector,
        ),
        operation=operation,
        metadata={"allow_overlap": True} if allow_overlap else {},
    )


def make_plan_intervention(
    spec: ActivationIntervention,
    *,
    order: int = 0,
    schedule: Any = None,
    position_mode: str = "absolute",
    on_step_call_indices: Sequence[int] = (0,),
    label: str | None = None,
) -> PlanIntervention:
    return PlanIntervention(
        plan_intervention_id=_identifier("pi", f"{spec.intervention_id}:{order}:{label}"),
        label=label,
        order=order,
        intervention=spec,
        schedule=parse_schedule(schedule if schedule is not None else {"kind": "first-prefix-forward"}),
        position_mode=position_mode,
        on_step_call_indices=list(on_step_call_indices),
    )


def make_plan(
    interventions: Iterable[PlanIntervention],
    *,
    prompt_length: int,
    prompt_token_ids: Sequence[int] | None = None,
    execution_mode: str = "uncached-generation",
    max_new_tokens: int = 4,
    max_sequence_length: int = 256,
    plan_kind: str = "intervention",
    name: str | None = "test plan",
    limits: TemporalLimits | None = None,
    seed: str = "plan",
) -> InterventionPlan:
    items = list(interventions)
    return InterventionPlan(
        plan_id=_identifier("pl", f"{seed}:{prompt_length}:{execution_mode}:{len(items)}"),
        created_at=utc_now_iso(),
        name=name,
        plan_kind=plan_kind,
        model=ModelRef(name="tiny-causal-lm"),
        execution_mode=execution_mode,
        prompt_length=prompt_length,
        prompt_token_ids=list(prompt_token_ids or []),
        continuation=PlanContinuationConfig(
            max_new_tokens=max_new_tokens,
            max_sequence_length=max_sequence_length,
        ),
        interventions=items,
        limits=limits or TemporalLimits(),
    )


def one_plan(
    module: str,
    *,
    prompt_length: int,
    steps: Any = None,
    positions: Sequence[int] = (0,),
    operation: str = "zero",
    position_mode: str = "last",
    execution_mode: str = "uncached-generation",
    max_new_tokens: int = 4,
    seed: str | None = None,
    **kwargs: Any,
) -> InterventionPlan:
    """Convenience: a one-intervention plan targeting ``module``."""

    spec_mode = (
        "teacher-forced" if execution_mode == "teacher-forced" else "uncached-continuation"
    )
    spec = make_spec(
        module,
        prompt_length=prompt_length,
        positions=positions,
        operation=operation,
        execution_mode=spec_mode,
        label=kwargs.pop("label", "spec"),
        allow_overlap=kwargs.pop("allow_overlap", False),
        **kwargs,
    )
    item = make_plan_intervention(
        spec,
        order=0,
        schedule=steps,
        position_mode=position_mode,
    )
    return make_plan(
        [item],
        prompt_length=prompt_length,
        execution_mode=execution_mode,
        max_new_tokens=max_new_tokens,
        seed=seed or module,
    )


__all__ = [
    "make_plan",
    "make_plan_intervention",
    "make_spec",
    "one_plan",
]
