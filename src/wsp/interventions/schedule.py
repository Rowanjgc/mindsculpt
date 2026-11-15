"""Temporal scheduling semantics for activation interventions.

V1.3 applied an intervention during one forward pass selected by its *module
call index*.  That is enough to intervene at the prompt, and it is not enough to
intervene at generation step 500.  This module defines the missing time axis.

The convention, used unchanged by replay, capture, intervention, and generation
-------------------------------------------------------------------------------

Let ``P`` be the number of prefix (prompt) tokens supplied by the investigator.

* **Prompt token index** -- ``0 .. P-1``.  Fixed for a whole experiment.
* **Generation step** ``k`` -- ``0-based``.  Step ``k`` is the forward pass whose
  input is the prefix followed by the ``k`` previously generated tokens, and
  whose next-token distribution predicts *generated token* ``k``.  Step ``0`` is
  therefore the first-prefix-forward event, and it is the only event V1.3 could
  address.
* **Generated token index** -- ``0-based``, one per generation step.
* **Full-sequence token index** -- ``0 .. P+k-1`` at step ``k``.  Prompt tokens
  keep their indices; generated token ``j`` has index ``P+j``.
* **Forward invocation index** -- how many times a *module* has been invoked so
  far in the whole run.  It is a module property, not a time axis.
* **Activation consumption position** -- the sequence row of an activation
  tensor.  In an uncached full-prefix forward over ``N = P+k`` tokens, row ``p``
  is the state that consumed token index ``p``.  In a staged decode forward the
  input has length ``1``; row ``0`` is the state that consumed full-sequence
  index ``P+k-1``.
* **Next-token prediction position** -- the token index that a row predicts.  A
  row at index ``p`` predicts token ``p+1``.  The workbench never shifts a
  position because of this label: ``position_semantics`` records the
  investigator's intent, and ``positions`` always index rows.

Two modes resolve a selector's ``positions`` at each firing:

``absolute``
    The rows are literal token indices inside the current forward.  They must be
    ``< P``, because generated positions do not exist at every step.  This is
    the default and reproduces V1.3 behaviour exactly.

``last``
    The selector carries the sentinel ``positions=[0]`` and resolves to the
    final row of the current forward: full index ``P+k-1`` for ``k >= 1``, and
    ``P-1`` at step ``0``.  This is how an investigation targets "the token the
    model just produced".

Uncached recomputation
----------------------

Uncached full-prefix decoding recomputes every earlier position on every step,
so an activation modified at step ``3`` is *not* present in the step ``4``
forward unless the schedule fires again.  A one-time schedule and a sustained
schedule are therefore different experiments, not different spellings of the
same one.  Every firing is recorded, so the difference is visible in the event
log rather than inferred from the result.

Schedules are declarative records.  Arbitrary executable predicates are never
accepted; ``trigger`` reserves the shape for future token-event triggers and
only ``forward`` is executable in this release.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from wsp.core.errors import InputError
from wsp.investigation.models import StrictModel

SCHEMA_SCHEDULE = "wsp.intervention-schedule/1.0"

ScheduleKind = Literal["first-prefix-forward", "generation-steps", "generation-range"]
ScheduleTrigger = Literal["forward"]

#: Everything a reader needs to interpret a recorded event without guessing.
TIMING_CONVENTIONS: dict[str, str] = {
    "prompt_length": "P: number of supplied prefix tokens; fixed for the experiment",
    "generation_step": (
        "0-based; step k consumes prefix + k tokens and predicts generated token k"
    ),
    "first_prefix_forward": "generation step 0; the forward over the untouched prefix",
    "generated_token_index": "0-based index of a produced token; token j has full index P+j",
    "full_sequence_index": "0..P+k-1 at step k; prompt indices never shift",
    "forward_invocation_index": (
        "per-module count of invocations within one forward pass; identifies the firing "
        "when a module is called repeatedly"
    ),
    "activation_consumption_position": (
        "sequence row of the activation; uncached row p consumed full index p, staged "
        "decode row 0 consumed full index P+k-1"
    ),
    "next_token_prediction_position": (
        "a row at index p predicts token p+1; the label never shifts selection indices"
    ),
}

DEFAULT_MAX_GENERATION_STEPS = 4096
DEFAULT_MAX_SCHEDULED_EVENTS = 64


class InterventionSchedule(StrictModel):
    """When an intervention fires, expressed in generation steps.

    Exactly one of the three kinds is used:

    ``first-prefix-forward``
        Step ``0`` only.
    ``generation-steps``
        An explicit ascending list of steps.  One entry is a one-time
        intervention; several entries are the explicit way to request a
        sustained intervention.
    ``generation-range``
        A bounded, contiguous or strided range.  ``end_inclusive`` states the
        convention unambiguously instead of leaving readers to guess.
    """

    schema_version: Literal[SCHEMA_SCHEDULE] = SCHEMA_SCHEDULE
    kind: ScheduleKind = "first-prefix-forward"
    trigger: ScheduleTrigger = "forward"
    steps: list[int] = Field(default_factory=list, max_length=4096)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    end_inclusive: bool = True
    every: int = Field(default=1, ge=1)
    max_steps: int = Field(default=DEFAULT_MAX_GENERATION_STEPS, ge=1)
    max_events: int = Field(default=DEFAULT_MAX_SCHEDULED_EVENTS, ge=1)

    @model_validator(mode="after")
    def _well_defined(self) -> "InterventionSchedule":
        if self.kind == "first-prefix-forward":
            if self.steps or self.start is not None or self.end is not None:
                raise ValueError(
                    "a first-prefix-forward schedule takes no steps, start, or end"
                )
            return self
        if self.kind == "generation-steps":
            if not self.steps:
                raise ValueError("a generation-steps schedule requires at least one step")
            if self.start is not None or self.end is not None:
                raise ValueError("a generation-steps schedule does not take start or end")
            if any(isinstance(item, bool) or item < 0 for item in self.steps):
                raise ValueError("generation steps must be non-negative integers")
            if len(set(self.steps)) != len(self.steps):
                raise ValueError("generation steps must be unique")
            if self.steps != sorted(self.steps):
                raise ValueError("generation steps must be ascending")
            if self.every != 1:
                raise ValueError(
                    "a generation-steps schedule is already explicit; 'every' is only "
                    "used by generation-range"
                )
            if len(self.steps) > self.max_events:
                raise ValueError(
                    f"the schedule lists {len(self.steps)} events but max_events is "
                    f"{self.max_events}; an intervention sweep must stay bounded"
                )
            if self.steps[-1] >= self.max_steps:
                raise ValueError(
                    f"generation step {self.steps[-1]} is outside the configured maximum "
                    f"of {self.max_steps} steps"
                )
            return self
        # generation-range
        if self.steps:
            raise ValueError("a generation-range schedule does not take an explicit step list")
        if self.start is None or self.end is None:
            raise ValueError("a generation-range schedule requires start and end")
        if self.end < self.start:
            raise ValueError("end must not be earlier than start")
        if self.end >= self.max_steps:
            raise ValueError(
                f"generation step {self.end} is outside the configured maximum of "
                f"{self.max_steps} steps"
            )
        if len(self.resolve()) > self.max_events:
            raise ValueError(
                f"the range resolves to more than max_events={self.max_events} events; "
                "an intervention sweep must stay bounded"
            )
        return self

    # ------------------------------------------------------------------ resolve

    def resolve(self) -> list[int]:
        """The generation steps this schedule fires on, ascending."""

        if self.kind == "first-prefix-forward":
            return [0]
        if self.kind == "generation-steps":
            return list(self.steps)
        assert self.start is not None and self.end is not None  # validated
        stop = self.end + 1 if self.end_inclusive else self.end
        return list(range(self.start, stop, self.every))

    @property
    def one_time(self) -> bool:
        return len(self.resolve()) == 1

    @property
    def bounded_event_count(self) -> int:
        return len(self.resolve())

    def describe(self) -> str:
        steps = self.resolve()
        if self.kind == "first-prefix-forward":
            return "generation step 0 (first prefix forward) only"
        if self.kind == "generation-steps":
            return "generation steps " + ", ".join(str(item) for item in steps)
        convention = "inclusive" if self.end_inclusive else "exclusive"
        stride = "" if self.every == 1 else f" with stride {self.every}"
        return f"generation steps {self.start}..{self.end} ({convention} end){stride}"


class ResolvedSchedule(StrictModel):
    """The concrete steps a schedule addressed during one recorded execution."""

    schedule: InterventionSchedule
    steps: list[int] = Field(default_factory=list)
    fired_steps: list[int] = Field(default_factory=list)
    not_reached_steps: list[int] = Field(default_factory=list)
    one_time: bool = True

    @model_validator(mode="after")
    def _partition(self) -> "ResolvedSchedule":
        resolved = self.schedule.resolve()
        unknown = [step for step in self.fired_steps if step not in resolved]
        if unknown:
            raise ValueError(
                "a fired step is not part of the recorded schedule: "
                + ", ".join(str(item) for item in unknown)
            )
        return self


def resolve_schedule(
    schedule: InterventionSchedule, *, steps_executed: int
) -> ResolvedSchedule:
    """Partition a schedule into steps that could fire and steps not reached.

    ``steps_executed`` is the number of forwards actually performed.  A step at
    or beyond it was never reached, which is different from a step that was
    reached and did not fire.  The distinction is what keeps "the experiment
    stopped early" separate from "the intervention was lost".
    """

    if steps_executed < 0:
        raise InputError("steps_executed must be non-negative")
    resolved = schedule.resolve()
    reached = [step for step in resolved if step < steps_executed]
    pending = [step for step in resolved if step >= steps_executed]
    return ResolvedSchedule(
        schedule=schedule,
        steps=resolved,
        fired_steps=reached,
        not_reached_steps=pending,
        one_time=schedule.one_time,
    )


def schedule_from_step(step: int, *, max_events: int = 1) -> InterventionSchedule:
    """Convenience for a single explicit step."""

    return InterventionSchedule(
        kind="generation-steps",
        steps=[int(step)],
        max_events=max(max_events, 1),
    )


def parse_schedule(payload: Any) -> InterventionSchedule:
    """Validate a schedule supplied as a record or a bare step list."""

    if isinstance(payload, InterventionSchedule):
        return payload
    if isinstance(payload, int):
        return schedule_from_step(payload)
    if isinstance(payload, (list, tuple)):
        return InterventionSchedule(kind="generation-steps", steps=[int(item) for item in payload])
    if isinstance(payload, dict):
        return InterventionSchedule.model_validate(payload)
    raise InputError(
        "a schedule must be a schedule record, an integer step, or a list of steps",
        details={"type": type(payload).__name__},
    )


__all__ = [
    "DEFAULT_MAX_GENERATION_STEPS",
    "DEFAULT_MAX_SCHEDULED_EVENTS",
    "InterventionSchedule",
    "ResolvedSchedule",
    "SCHEMA_SCHEDULE",
    "TIMING_CONVENTIONS",
    "parse_schedule",
    "resolve_schedule",
    "schedule_from_step",
]
