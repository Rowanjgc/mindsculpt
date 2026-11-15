"""First-class, versioned intervention plans.

A plan is the portable description of a *multi-site, scheduled* experiment.  It
reuses :class:`wsp.interventions.models.ActivationIntervention` for every
individual operation, so the numerical semantics of ``zero``, ``mean``,
``patch``, and ``identity`` are defined in exactly one place.

What the plan adds is time and composition:

* a generation-step schedule per intervention, including sustained schedules;
* the position mode that resolves activation rows at each firing;
* an explicit order for interventions that share a module;
* a deterministic conflict policy that rejects ambiguity instead of merging it;
* resource bounds that are checked before any model executes.

Ordering is never inferred from layer names.  Distinct modules fire in the order
the model actually invokes them; interventions inside one module fire in
ascending ``order``.  A plan that cannot be ordered deterministically fails
validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from wsp.behavior.models import ModelRef
from wsp.core.hashing import sha256_bytes
from wsp.core.json import canonical_bytes
from wsp.core.time import utc_now_iso
from wsp.investigation.models import StrictModel
from wsp.interventions.models import (
    ActivationIntervention,
    ValidationCheck,
)
from wsp.interventions.schedule import (
    InterventionSchedule,
    TIMING_CONVENTIONS,
)

SCHEMA_PLAN = "wsp.intervention-plan/1.0"

PlanKind = Literal["intervention", "identity-control"]
PlanExecutionMode = Literal["teacher-forced", "uncached-generation", "staged-generation"]
PositionMode = Literal["absolute", "last"]
OrderingPolicy = Literal["explicit-order"]

PLAN_LIMITATIONS = [
    "a plan states what will be executed and where; it makes no claim about mechanism",
    "a scheduled firing is instrumentation evidence, not evidence of a causal role",
    "uncached recomputation means a one-time intervention is not carried forward into "
    "later forwards; a sustained schedule is a different experiment",
    "results are specific to the recorded model, dtype, device, backend, and token sequence",
]


class TemporalLimits(StrictModel):
    """Bounds that keep a schedule from becoming an unbounded sweep.

    Full-prefix uncached decoding costs ``O(steps^2)`` token positions, so the
    step budget is also a compute budget, not only a record-size bound.
    """

    max_generation_steps: int = Field(default=64, ge=1)
    max_scheduled_events: int = Field(default=64, ge=1)
    max_total_scheduled_events: int = Field(default=256, ge=1)
    max_interventions: int = Field(default=8, ge=1)
    max_events_per_step: int = Field(default=8, ge=1)
    max_targeted_modules: int = Field(default=16, ge=1)
    max_event_records: int = Field(default=512, ge=1)
    max_observed_logit_rows: int = Field(default=64, ge=1)
    max_selected_elements: int = Field(default=500_000, ge=1)
    max_donor_elements: int = Field(default=500_000, ge=1)
    max_continuation_tokens: int = Field(default=64, ge=1)


class PlanContinuationConfig(StrictModel):
    """Deterministic decoding settings for a scheduled run.

    Kept separate from V1.3 ``ContinuationConfig`` because that record pins the
    intervention to ``first-prefix-forward`` and ``use_cache=False``.  Reusing it
    would silently misdescribe a V1.4 schedule.
    """

    strategy: Literal["greedy"] = "greedy"
    max_new_tokens: int = Field(default=8, ge=1, le=4096)
    max_sequence_length: int = Field(default=2048, ge=2)
    eos_token_id: int | None = Field(default=None, ge=0)
    stop_on_eos: bool = True
    seed: int | None = None
    do_sample: Literal[False] = False
    temperature: Literal[0.0] = 0.0


class PlanIntervention(StrictModel):
    """One scheduled intervention inside a plan."""

    plan_intervention_id: str = Field(pattern=r"^pi_[0-9a-f]{16}$")
    label: str | None = None
    order: int = Field(ge=0)
    intervention: ActivationIntervention
    schedule: InterventionSchedule = Field(
        default_factory=lambda: InterventionSchedule(kind="first-prefix-forward")
    )
    position_mode: PositionMode = "absolute"
    on_step_call_indices: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=16)

    @model_validator(mode="after")
    def _coherent(self) -> "PlanIntervention":
        if any(isinstance(item, bool) or item < 0 for item in self.on_step_call_indices):
            raise ValueError("on_step_call_indices must be non-negative integers")
        if len(set(self.on_step_call_indices)) != len(self.on_step_call_indices):
            raise ValueError("on_step_call_indices must be unique")
        if self.on_step_call_indices != sorted(self.on_step_call_indices):
            raise ValueError("on_step_call_indices must be ascending")
        positions = self.intervention.target.selector.positions
        if self.position_mode == "last" and positions != [0]:
            raise ValueError(
                "position_mode 'last' resolves to the newest token of the current forward; "
                "write positions=[0] as the explicit sentinel rather than reusing an "
                "absolute index"
            )
        return self

    @property
    def module(self) -> str:
        return self.intervention.target.module

    @property
    def tensor_role(self) -> str:
        return self.intervention.target.tensor_role

    @property
    def operation(self) -> str:
        return self.intervention.operation


class InterventionPlan(StrictModel):
    """A multi-site, temporally scheduled intervention experiment."""

    schema_version: Literal[SCHEMA_PLAN] = SCHEMA_PLAN
    plan_id: str = Field(pattern=r"^pl_[0-9a-f]{16}$")
    created_at: str
    name: str | None = None
    plan_kind: PlanKind = "intervention"
    model: ModelRef
    execution_mode: PlanExecutionMode = "uncached-generation"
    backend: Literal["torch_hooks"] = "torch_hooks"
    prompt_length: int = Field(ge=1)
    prompt_sha256: str | None = None
    prompt_token_ids: list[int] = Field(default_factory=list)
    continuation: PlanContinuationConfig = Field(default_factory=PlanContinuationConfig)
    interventions: list[PlanIntervention] = Field(min_length=1, max_length=32)
    ordering_policy: OrderingPolicy = "explicit-order"
    limits: TemporalLimits = Field(default_factory=TemporalLimits)
    provenance: dict[str, Any] = Field(default_factory=dict)
    timing_conventions: dict[str, str] = Field(default_factory=lambda: dict(TIMING_CONVENTIONS))
    notes: str | None = None
    limitations: list[str] = Field(default_factory=lambda: list(PLAN_LIMITATIONS))
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> "InterventionPlan":
        ids = [item.plan_intervention_id for item in self.interventions]
        if len(set(ids)) != len(ids):
            raise ValueError("plan_intervention_id values must be unique")
        orders = [item.order for item in self.interventions]
        if len(set(orders)) != len(orders):
            raise ValueError(
                "intervention order values must be unique; identical order plus identical "
                "region is exactly the ambiguity this plan refuses to resolve silently"
            )
        if len(self.interventions) > self.limits.max_interventions:
            raise ValueError("the plan exceeds limits.max_interventions")

        for item in self.interventions:
            spec = item.intervention
            expected_mode = (
                "teacher-forced"
                if self.execution_mode == "teacher-forced"
                else "uncached-continuation"
            )
            if spec.execution.execution_mode != expected_mode:
                raise ValueError(
                    f"intervention '{item.plan_intervention_id}' is marked "
                    f"execution_mode={spec.execution.execution_mode!r} but the plan is "
                    f"{self.execution_mode!r}"
                )
            if spec.execution.backend != self.backend:
                raise ValueError("every intervention must use the plan's backend")
            if spec.execution.sequence_length != self.prompt_length:
                raise ValueError(
                    f"intervention '{item.plan_intervention_id}' was recorded against a "
                    f"sequence length of {spec.execution.sequence_length}, but the plan "
                    f"prompt length is {self.prompt_length}"
                )
            if self.model.file_sha256 is not None and spec.model.file_sha256 is not None:
                if self.model.file_sha256 != spec.model.file_sha256:
                    raise ValueError("an intervention belongs to a different checkpoint file")
            if (
                self.model.structure_fingerprint is not None
                and spec.model.structure_fingerprint is not None
                and self.model.structure_fingerprint != spec.model.structure_fingerprint
            ):
                raise ValueError("an intervention belongs to a different model structure")
            if item.position_mode == "absolute":
                for position in spec.target.selector.positions:
                    if position >= self.prompt_length:
                        raise ValueError(
                            f"absolute position {position} is not a prompt position "
                            f"(prompt_length={self.prompt_length}); use position_mode='last' "
                            "to target the newest token of a later forward"
                        )
            resolved = item.schedule.resolve()
            if len(resolved) > self.limits.max_scheduled_events:
                raise ValueError(
                    f"intervention '{item.plan_intervention_id}' schedules {len(resolved)} "
                    "events, exceeding limits.max_scheduled_events"
                )
            if resolved[-1] >= self.limits.max_generation_steps:
                raise ValueError(
                    f"intervention '{item.plan_intervention_id}' schedules generation step "
                    f"{resolved[-1]}, beyond limits.max_generation_steps="
                    f"{self.limits.max_generation_steps}"
                )

        modules = {item.module for item in self.interventions}
        if len(modules) > self.limits.max_targeted_modules:
            raise ValueError("the plan exceeds limits.max_targeted_modules")
        total = sum(item.schedule.bounded_event_count for item in self.interventions)
        if total > self.limits.max_total_scheduled_events:
            raise ValueError(
                f"the plan schedules {total} events in total, exceeding "
                f"limits.max_total_scheduled_events={self.limits.max_total_scheduled_events}"
            )
        per_step: dict[int, int] = {}
        for item in self.interventions:
            for step in item.schedule.resolve():
                per_step[step] = per_step.get(step, 0) + 1
        if per_step and max(per_step.values()) > self.limits.max_events_per_step:
            raise ValueError(
                "a single generation step schedules more interventions than "
                f"limits.max_events_per_step={self.limits.max_events_per_step}"
            )
        if len(self.prompt_token_ids) not in (0, self.prompt_length):
            raise ValueError("prompt_token_ids must be empty or match prompt_length")
        if self.limits.max_continuation_tokens < self.continuation.max_new_tokens:
            raise ValueError(
                "continuation.max_new_tokens exceeds limits.max_continuation_tokens"
            )
        if self.plan_kind == "identity-control":
            for item in self.interventions:
                if item.intervention.operation != "identity":
                    raise ValueError(
                        "an identity-control plan must attach identity operations only"
                    )
                if item.intervention.mean is not None or item.intervention.donor is not None:
                    raise ValueError("an identity-control intervention takes no replacement")
        return self

    # ------------------------------------------------------------------ helpers

    def ordered(self) -> list[PlanIntervention]:
        return sorted(self.interventions, key=lambda item: item.order)

    def module_paths(self) -> list[str]:
        return sorted({item.module for item in self.interventions})

    def expected_event_keys(self, *, steps_executed: int) -> list[tuple[str, int, int]]:
        """``(plan_intervention_id, generation_step, on_step_call_index)`` triples."""

        keys: list[tuple[str, int, int]] = []
        for item in self.interventions:
            for step in item.schedule.resolve():
                if step >= steps_executed:
                    continue
                for call_index in item.on_step_call_indices:
                    keys.append((item.plan_intervention_id, step, call_index))
        return keys


# --------------------------------------------------------------------------- #
# conflicts
# --------------------------------------------------------------------------- #


class PlanConflict(StrictModel):
    kind: Literal["overlapping-region", "duplicate-site"]
    plan_intervention_ids: list[str]
    module: str
    generation_steps: list[int]
    detail: str


def _rows_key(item: PlanIntervention) -> str:
    if item.position_mode == "last":
        return "<last-token>"
    return ",".join(str(value) for value in item.intervention.target.selector.positions)


def _feature_overlap(left: PlanIntervention, right: PlanIntervention) -> bool:
    a = left.intervention.target.selector
    b = right.intervention.target.selector
    if a.feature_indices and b.feature_indices:
        return bool(set(a.feature_indices) & set(b.feature_indices))
    a_all = not a.feature_indices and a.feature_start is None
    b_all = not b.feature_indices and b.feature_start is None
    if a_all or b_all:
        return True
    if a.feature_indices and b.feature_start is not None:
        return any(b.feature_start <= index < (b.feature_end or 0) for index in a.feature_indices)
    if b.feature_indices and a.feature_start is not None:
        return any(a.feature_start <= index < (a.feature_end or 0) for index in b.feature_indices)
    a_range = (a.feature_start or 0, a.feature_end or 0)
    b_range = (b.feature_start or 0, b.feature_end or 0)
    return max(a_range[0], b_range[0]) < min(a_range[1], b_range[1])


def _regions_overlap(left: PlanIntervention, right: PlanIntervention) -> bool:
    """Whether two interventions can write the same activation element.

    A ``last`` selector resolves per step, so an absolute selector cannot be
    proven disjoint from it without running.  The workbench treats that pair as
    overlapping and requires explicit authorization rather than assuming
    separation it cannot verify.
    """

    if left.position_mode == "last" and right.position_mode == "last":
        # Both resolve to the newest row of the forward, so the row sets are
        # identical and only the feature regions can separate them.
        return _feature_overlap(left, right)
    if left.position_mode == "last" or right.position_mode == "last":
        return True
    if not set(left.intervention.target.selector.positions) & set(
        right.intervention.target.selector.positions
    ):
        return False
    return _feature_overlap(left, right)


def find_conflicts(plan: InterventionPlan) -> list[PlanConflict]:
    """Regions that more than one intervention would write on the same step.

    The workbench rejects these unless the investigator explicitly authorized
    overlap on both sides.  Overlap is computed on the resolved step sets, so
    two interventions that target the same module at *different* times are not
    a conflict: they never touch the same activation.
    """

    conflicts: list[PlanConflict] = []
    items = plan.ordered()
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left.module != right.module or left.tensor_role != right.tensor_role:
                continue
            shared = sorted(
                set(left.schedule.resolve()) & set(right.schedule.resolve())
            )
            if not shared:
                continue
            shared_calls = sorted(
                set(left.on_step_call_indices) & set(right.on_step_call_indices)
            )
            if not shared_calls:
                continue
            if not _regions_overlap(left, right):
                continue
            authorized = (
                bool(left.intervention.metadata.get("allow_overlap"))
                and bool(right.intervention.metadata.get("allow_overlap"))
            )
            if authorized:
                continue
            conflicts.append(
                PlanConflict(
                    kind="duplicate-site"
                    if _rows_key(left) == _rows_key(right)
                    else "overlapping-region",
                    plan_intervention_ids=[
                        left.plan_intervention_id,
                        right.plan_intervention_id,
                    ],
                    module=left.module,
                    generation_steps=shared,
                    detail=(
                        f"'{left.plan_intervention_id}' (order {left.order}) and "
                        f"'{right.plan_intervention_id}' (order {right.order}) both write "
                        f"{left.module} [{left.tensor_role}] at steps {shared}; "
                        "set metadata.allow_overlap=true on both interventions to accept "
                        "an ordered composition"
                    ),
                )
            )
    return conflicts


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


class PlanValidation(StrictModel):
    valid: bool
    plan_id: str
    checks: list[ValidationCheck] = Field(default_factory=list)
    conflicts: list[PlanConflict] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    resolved: list[dict[str, Any]] = Field(default_factory=list)
    estimated_forward_calls: int = Field(default=0, ge=0)
    estimated_token_positions: int = Field(default=0, ge=0)
    human_readable: list[str] = Field(default_factory=list)


def _absolute_schedule_end(plan: InterventionPlan) -> int:
    return max(item.schedule.resolve()[-1] for item in plan.interventions)


def _cost_estimate(plan: InterventionPlan) -> tuple[int, int]:
    """Forwards and token positions for the full continuation.

    Uncached decoding re-runs the whole prefix at every step, so the count is
    quadratic in the sequence length.  Recording the estimate makes the cost
    visible before the run rather than after it.
    """

    steps = max(plan.continuation.max_new_tokens, _absolute_schedule_end(plan) + 1)
    forwards = steps
    positions = sum(plan.prompt_length + index for index in range(steps))
    return forwards, positions


def validate_plan(
    plan: InterventionPlan,
    *,
    root: Any | None = None,
    budget: Any | None = None,
) -> PlanValidation:
    """Static validation, with optional module resolution against a real model."""

    from wsp.core.budget import Budget

    budget = budget or Budget()
    checks: list[ValidationCheck] = []
    errors: list[str] = []
    warnings: list[str] = []
    resolved: list[dict[str, Any]] = []
    conflicts: list[PlanConflict] = []

    plan_events = sum(item.schedule.bounded_event_count for item in plan.interventions)
    per_step: dict[int, list[str]] = {}
    for item in plan.interventions:
        for step in item.schedule.resolve():
            per_step.setdefault(step, []).append(item.plan_intervention_id)

    if plan.continuation.max_new_tokens > budget.max_continuation_tokens:
        checks.append(
            ValidationCheck(
                name="continuation_budget",
                status="fail",
                detail=(
                    f"max_new_tokens={plan.continuation.max_new_tokens} exceeds the "
                    f"budget of {budget.max_continuation_tokens}"
                ),
            )
        )
        errors.append("the continuation exceeds the configured token budget")
    else:
        checks.append(ValidationCheck(name="continuation_budget", status="pass"))

    if plan.continuation.max_new_tokens > budget.max_generation_steps:
        checks.append(
            ValidationCheck(
                name="generation_steps_budget",
                status="fail",
                detail=(
                    f"max_new_tokens={plan.continuation.max_new_tokens} exceeds the "
                    f"budget of {budget.max_generation_steps} generation steps"
                ),
            )
        )
        errors.append("the continuation exceeds the generation-step budget")
    else:
        checks.append(ValidationCheck(name="generation_steps_budget", status="pass"))

    if plan_events > budget.max_scheduled_events:
        checks.append(
            ValidationCheck(
                name="scheduled_events_budget",
                status="fail",
                detail=(
                    f"the plan schedules {plan_events} events, exceeding the budget of "
                    f"{budget.max_scheduled_events}"
                ),
            )
        )
        errors.append("the plan schedules more events than the budget allows")
    else:
        checks.append(ValidationCheck(name="scheduled_events_budget", status="pass"))

    if len(plan.interventions) > budget.max_temporal_interventions:
        checks.append(
            ValidationCheck(
                name="intervention_count_budget",
                status="fail",
                detail=(
                    f"the plan holds {len(plan.interventions)} interventions, exceeding the "
                    f"budget of {budget.max_temporal_interventions}"
                ),
            )
        )
        errors.append("the plan holds more interventions than the budget allows")
    else:
        checks.append(ValidationCheck(name="intervention_count_budget", status="pass"))

    total_sequence = plan.prompt_length + plan.continuation.max_new_tokens
    if total_sequence > budget.max_continuation_sequence_tokens:
        checks.append(
            ValidationCheck(
                name="sequence_budget",
                status="fail",
                detail=(
                    f"the longest sequence is {total_sequence} tokens, exceeding "
                    f"{budget.max_continuation_sequence_tokens}"
                ),
            )
        )
        errors.append("the continuation exceeds the sequence-length budget")

    if total_sequence > plan.continuation.max_sequence_length:
        checks.append(
            ValidationCheck(
                name="config_sequence_length",
                status="fail",
                detail="the continuation exceeds config.max_sequence_length",
            )
        )
        errors.append("the continuation exceeds config.max_sequence_length")

    scheduled_end = _absolute_schedule_end(plan)
    if scheduled_end >= plan.continuation.max_new_tokens:
        checks.append(
            ValidationCheck(
                name="schedule_reachable",
                status="warning",
                detail=(
                    f"the latest scheduled step is {scheduled_end}, but only "
                    f"{plan.continuation.max_new_tokens} generation steps will run; those "
                    "firings are reported as not-reached rather than missing"
                ),
            )
        )
        warnings.append(
            "part of the schedule lies beyond the configured continuation length; those "
            "events will be recorded as not reached"
        )
    else:
        checks.append(ValidationCheck(name="schedule_reachable", status="pass"))

    over_limit = [
        step for step, items in per_step.items() if len(items) > plan.limits.max_events_per_step
    ]
    if over_limit:
        errors.append(
            "steps "
            + ", ".join(str(step) for step in sorted(over_limit))
            + " schedule more interventions than limits.max_events_per_step"
        )

    conflicts = find_conflicts(plan)
    if conflicts:
        checks.append(
            ValidationCheck(
                name="composition_conflicts",
                status="fail",
                detail=f"{len(conflicts)} overlapping region(s) without explicit authorization",
            )
        )
        errors.extend(item.detail for item in conflicts)
    else:
        checks.append(ValidationCheck(name="composition_conflicts", status="pass"))

    if plan.execution_mode == "staged-generation":
        unsupported = [
            item.plan_intervention_id
            for item in plan.interventions
            if item.position_mode == "absolute" and item.schedule.resolve() != [0]
        ]
        if unsupported:
            checks.append(
                ValidationCheck(
                    name="staged_position_semantics",
                    status="fail",
                    detail="a staged decode forward carries only the newest activation row",
                )
            )
            errors.append(
                "staged plans can address absolute prompt positions only at step 0; "
                "these interventions schedule later steps and must use position_mode='last': "
                + ", ".join(unsupported)
            )
        else:
            checks.append(ValidationCheck(name="staged_position_semantics", status="pass"))

    for item in plan.ordered():
        entry: dict[str, Any] = {
            "plan_intervention_id": item.plan_intervention_id,
            "label": item.label,
            "order": item.order,
            "module": item.module,
            "tensor_role": item.tensor_role,
            "operation": item.operation,
            "position_mode": item.position_mode,
            "schedule": item.schedule.describe(),
            "steps": item.schedule.resolve(),
            "one_time": item.schedule.one_time,
            "on_step_call_indices": list(item.on_step_call_indices),
        }
        if root is not None:
            try:
                from wsp.interventions.engine import _module_lookup

                module = _module_lookup(root, item.module)
                entry["module_class"] = type(module).__name__
            except Exception as exc:  # reported, never swallowed
                errors.append(f"{item.plan_intervention_id}: {exc}")
                entry["module_class"] = None
        resolved.append(entry)

    if errors:
        checks.append(ValidationCheck(name="plan", status="fail", detail="; ".join(errors[:3])))
    else:
        checks.append(ValidationCheck(name="plan", status="pass"))

    forwards, positions = _cost_estimate(plan)
    lines = [
        f"plan: {plan.plan_id} ({plan.plan_kind})",
        f"execution: {plan.execution_mode}, prompt_length={plan.prompt_length}, "
        f"continuation={plan.continuation.max_new_tokens}",
        f"interventions: {len(plan.interventions)}, scheduled events: {plan_events}",
        f"estimated forwards: {forwards}, token positions: {positions}"
        + (
            " (uncached full-prefix decoding recomputes the prefix on every step)"
            if plan.execution_mode == "uncached-generation"
            else ""
        ),
    ]
    for entry in resolved:
        lines.append(
            f"  [{entry['order']}] {entry['plan_intervention_id']} {entry['operation']} "
            f"{entry['module']} [{entry['tensor_role']}] mode={entry['position_mode']} "
            f"{entry['schedule']}"
        )
    lines.extend(f"error: {item}" for item in errors)
    lines.extend(f"warning: {item}" for item in warnings)
    lines.append(f"valid: {not errors}")
    return PlanValidation(
        valid=not errors,
        plan_id=plan.plan_id,
        checks=checks,
        conflicts=conflicts,
        errors=errors,
        warnings=warnings,
        resolved=resolved,
        estimated_forward_calls=forwards,
        estimated_token_positions=positions,
        human_readable=lines,
    )


# --------------------------------------------------------------------------- #
# hashing, identity control
# --------------------------------------------------------------------------- #


def plan_sha256(plan: InterventionPlan) -> str:
    payload = plan.model_dump(mode="json")
    payload.pop("created_at", None)
    payload.pop("checksums", None)
    return sha256_bytes(canonical_bytes(payload))


def identity_control_plan(plan: InterventionPlan) -> InterventionPlan:
    """The same schedule and attachment sites with no numerical change.

    An identity plan is not an empty plan.  Every hook, every module, every
    step, and every call index is preserved, so a disagreement between the
    baseline and the identity run is evidence that *instrumentation* changed the
    computation.  That check is what licenses attribution in later conditions.
    """

    from wsp.interventions.ids import new_plan_intervention_id

    interventions: list[PlanIntervention] = []
    for item in plan.ordered():
        payload = item.intervention.model_dump(mode="python")
        payload["intervention_id"] = _identity_intervention_id(item.intervention)
        payload["operation"] = "identity"
        payload["mean"] = None
        payload["donor"] = None
        payload["checksums"] = {}
        identity_spec = ActivationIntervention.model_validate(payload)
        interventions.append(
            PlanIntervention(
                plan_intervention_id=new_plan_intervention_id(),
                label=(item.label or item.plan_intervention_id) + " (identity control)",
                order=item.order,
                intervention=identity_spec,
                schedule=item.schedule.model_copy(deep=True),
                position_mode=item.position_mode,
                on_step_call_indices=list(item.on_step_call_indices),
            )
        )
    payload = plan.model_dump(mode="python")
    payload.update(
        {
            "plan_id": _new_plan_id(),
            "created_at": utc_now_iso(),
            "name": (plan.name or plan.plan_id) + " (identity control)",
            "plan_kind": "identity-control",
            "interventions": [item.model_dump(mode="python") for item in interventions],
            "checksums": {},
            "metadata": {
                **plan.metadata,
                "derived_from_plan_id": plan.plan_id,
                "derived_from_plan_sha256": plan_sha256(plan),
                "control_kind": "identity",
            },
        }
    )
    return InterventionPlan.model_validate(payload)


def _identity_intervention_id(spec: ActivationIntervention) -> str:
    from wsp.interventions.ids import new_intervention_id

    # A distinct identifier keeps the identity operation from being confused
    # with the original specification when both are stored in one workspace.
    del spec
    return new_intervention_id()


def _new_plan_id() -> str:
    from wsp.interventions.ids import new_plan_id

    return new_plan_id()


def renumber(plan: InterventionPlan) -> InterventionPlan:
    """Return a copy with a recomputed plan checksum.

    The plan identity is deliberately preserved: renumbering records that the
    checksum was rebuilt, not that a different experiment exists.
    """

    checksums = dict(plan.checksums)
    checksums["plan_sha256"] = plan_sha256(plan)
    return plan.model_copy(update={"checksums": checksums})


__all__ = [
    "InterventionPlan",
    "PLAN_LIMITATIONS",
    "PlanConflict",
    "PlanContinuationConfig",
    "PlanExecutionMode",
    "PlanIntervention",
    "PlanKind",
    "PlanValidation",
    "PositionMode",
    "SCHEMA_PLAN",
    "TemporalLimits",
    "find_conflicts",
    "identity_control_plan",
    "plan_sha256",
    "renumber",
    "validate_plan",
]
