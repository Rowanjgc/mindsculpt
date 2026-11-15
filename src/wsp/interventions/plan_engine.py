"""Deterministic execution of scheduled, multi-site intervention plans.

The executor installs exactly one hook per ``(module, tensor_role)`` pair and
resolves every scheduled intervention inside that hook.  This keeps the number
of installed hooks proportional to the number of targeted modules, not to the
number of scheduled events, and it means a module that is called repeatedly is
handled by the module's own invocation counter rather than by hook order.

Fail-closed rules
-----------------

* a scheduled firing that the execution reached but that never happened is a
  **missing event** and invalidates the run;
* a firing that raises is recorded as **rejected** and aborts the run;
* model parameters, mutable buffers, module training flags, and the installed
  hook count are compared before and after every execution;
* a plan is never reported as executed merely because generation finished.

The executor never mutates a parameter, never writes in place, and never keeps
hidden-state tensors: events store bounded statistics plus a replacement digest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from pydantic import Field

from wsp.core.budget import Budget
from wsp.core.dtypes import torch_dtype_name
from wsp.core.errors import (
    CompatibilityError,
    ExecutionError,
    InputError,
    ResourceLimitError,
    UnsupportedError,
    WspError,
)
from wsp.core.hashing import sha256_bytes
from wsp.core.time import utc_now_iso
from wsp.investigation.models import StrictModel
from wsp.replay.engine import parameter_fingerprint

from .engine import (
    _check_logits,
    _module_lookup,
    _replacement_tensor,
    _spec_variant,
    describe_model,
    torch_module,
)
from .models import (
    ContinuationDiff,
    IdentityControl,
    TensorSelector,
    ValidationCheck,
    ValidityReport,
)
from .operations import apply_operation, compare_slices, extract_activation, rebuild_activation, resolve_selection
from .plan import (
    InterventionPlan,
    PlanIntervention,
    identity_control_plan,
    plan_sha256,
    validate_plan,
)
from .plan_models import (
    ExpectedEvent,
    InterventionEventLog,
    InterventionEventRecord,
    StepwiseLogitDiff,
    TemporalInterventionRun,
    event_log_sha256,
    temporal_run_sha256,
)
from .schedule import resolve_schedule

IDENTITY_ATOL = 1e-6
IDENTITY_RTOL = 1e-5


# --------------------------------------------------------------------------- #
# model state protection
# --------------------------------------------------------------------------- #


def mutable_state_fingerprint(module: torch.nn.Module) -> str:
    """Digest buffers, training flags, and hook counts.

    A parameter-only fingerprint does not establish that all model state stayed
    unchanged.  Normalisation running statistics, rotary caches, and module
    training flags are mutable state too, so they are sampled and hashed here.
    """

    parts: list[str] = []
    for name, buffer in sorted(module.named_buffers()):
        tensor = buffer.detach().reshape(-1)
        if tensor.numel() > 64:
            step = max(1, tensor.numel() // 64)
            sample = tensor[::step][:64]
        else:
            sample = tensor
        try:
            total = float(sample.to(torch.float64).sum().item())
        except (RuntimeError, TypeError):  # pragma: no cover - exotic dtypes
            total = 0.0
        parts.append(f"b:{name}|{tuple(int(d) for d in buffer.shape)}|{buffer.dtype}|{total:.17g}")
    for name, child in module.named_modules():
        parts.append(f"t:{name or '<root>'}:{int(bool(child.training))}")
    return sha256_bytes("\n".join(parts).encode("utf-8"))


def installed_hook_count(module: torch.nn.Module) -> int:
    total = 0
    for child in module.modules():
        total += len(child._forward_hooks)  # noqa: SLF001 - intentional residue check
        total += len(child._forward_pre_hooks)  # noqa: SLF001
        total += len(child._backward_hooks)  # noqa: SLF001
    return total


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #


class PreviewFiring(StrictModel):
    plan_intervention_id: str
    generation_step: int
    forward_invocation: int
    module: str
    tensor_role: str
    position_mode: str
    rows: list[int] = Field(default_factory=list)
    token_indices: list[int] = Field(default_factory=list)
    tensor_shape: list[int] = Field(default_factory=list)
    dtype: str | None = None
    axis: int | None = None
    elements: int = Field(default=0, ge=0)
    note: str | None = None


class PlanPreview(StrictModel):
    plan_id: str
    valid: bool
    steps_executed: int = Field(ge=0)
    firings: list[PreviewFiring] = Field(default_factory=list)
    human_readable: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# the session
# --------------------------------------------------------------------------- #


@dataclass
class TemporalOutcome:
    record: TemporalInterventionRun
    event_logs: dict[str, InterventionEventLog] = field(default_factory=dict)


@dataclass
class PlanRunOutcome:
    plan: InterventionPlan
    generated_token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    step_logits: torch.Tensor | None = None
    step_top1: list[int] = field(default_factory=list)
    termination: str = "max_new_tokens"
    steps: int = 0
    identity: bool = False
    applied: bool = False
    events: list[InterventionEventRecord] = field(default_factory=list)
    rejected: list[InterventionEventRecord] = field(default_factory=list)
    firings: list[PreviewFiring] = field(default_factory=list)
    truncated: bool = False
    total_observed: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    fingerprint_before: str = ""
    fingerprint_after: str = ""
    state_before: str = ""
    state_after: str = ""
    hook_residue: bool = False
    stepwise_truncated: bool = False
    log: InterventionEventLog | None = None


class PlanSession:
    """Install one reversible hook per targeted module and fire on schedule."""

    def __init__(
        self,
        root: torch.nn.Module,
        plan: InterventionPlan,
        *,
        prompt_length: int,
        store: Any | None = None,
        budget: Budget | None = None,
        identity: bool = False,
        apply: bool = True,
        dry_run: bool = False,
    ) -> None:
        self.root = root
        self.plan = plan
        self.budget = budget or Budget()
        self.store = store
        self.identity = identity
        self.apply = apply
        self.dry_run = dry_run
        self.prompt_length = int(prompt_length)
        self.current_step = 0
        self.current_length = int(prompt_length)
        self.current_offset = 0
        self._call_counts: dict[tuple[str, str], int] = {}
        self._steps: dict[str, set[int]] = {}
        self.events: list[InterventionEventRecord] = []
        self.rejected: list[InterventionEventRecord] = []
        self.firings: list[PreviewFiring] = []
        self.truncated = False
        self.total_observed = 0
        self._handles: list[Any] = []
        self._closed = False
        self._fired_keys: set[tuple[str, int, int]] = set()

        self._groups: dict[tuple[str, str], list[PlanIntervention]] = {}
        self._replacements: dict[str, torch.Tensor | None] = {}
        for item in plan.ordered():
            key = (item.module, item.tensor_role)
            self._groups.setdefault(key, []).append(item)
            self._steps[item.plan_intervention_id] = set(item.schedule.resolve())
            if not dry_run:
                replacement = None
                if not identity and item.intervention.operation in ("mean", "patch"):
                    replacement = _replacement_tensor(item.intervention, store)
                self._replacements[item.plan_intervention_id] = replacement
        if len(self._groups) > plan.limits.max_targeted_modules:
            raise ResourceLimitError(
                "the plan targets more modules than limits.max_targeted_modules",
                details={"requested": len(self._groups), "limit": plan.limits.max_targeted_modules},
            )
        self._events_limit = min(
            plan.limits.max_event_records, self.budget.max_event_records
        )

    # ------------------------------------------------------------------ hooks

    def __enter__(self) -> "PlanSession":
        self.install()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def install(self) -> None:
        if self._handles:
            raise InputError("the plan session is already installed")
        for (path, role), _items in sorted(self._groups.items()):
            module = _module_lookup(self.root, "" if path in ("", "<root>") else path)
            if role == "input":
                self._handles.append(
                    module.register_forward_pre_hook(self._pre_hook(path), with_kwargs=True)
                )
            else:
                self._handles.append(module.register_forward_hook(self._post_hook(path)))

    def close(self) -> None:
        if self._closed:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._closed = True

    def begin_step(self, step: int, length: int, offset: int) -> None:
        """Declare the temporal identity of the forward that is about to run."""

        if step != self.current_step:
            self._call_counts.clear()
        self.current_step = int(step)
        self.current_length = int(length)
        self.current_offset = int(offset)
        if step > 0:
            # Invocation counters are per forward pass, never cumulative.
            self._call_counts.clear()

    def _post_hook(self, path: str) -> Callable[..., Any]:
        def hook(module: torch.nn.Module, args: Any, output: Any) -> Any:
            return self._process(path, "output", output, output)

        return hook

    def _pre_hook(self, path: str) -> Callable[..., Any]:
        def hook(module: torch.nn.Module, args: Any, kwargs: Any) -> Any:
            if len(args) == 1:
                value = args[0]
                single = True
            else:
                value = tuple(args)
                single = False
            updated = self._process(path, "input", value, args)
            if updated is value:
                return None
            if single:
                return ((updated,) + tuple(args[1:]), kwargs)
            return (tuple(updated), kwargs)

        return hook

    # --------------------------------------------------------------- processing

    def _matching(self, path: str, role: str, call_index: int) -> list[PlanIntervention]:
        items = []
        for item in self._groups.get((path, role), ()):
            if self.current_step not in self._steps[item.plan_intervention_id]:
                continue
            if call_index not in item.on_step_call_indices:
                continue
            items.append(item)
        return sorted(items, key=lambda item: item.order)

    def _rows(self, item: PlanIntervention) -> list[int]:
        if item.position_mode == "last":
            return [self.current_length - 1]
        rows = list(item.intervention.target.selector.positions)
        if any(row >= self.current_length for row in rows):
            raise CompatibilityError(
                "an absolute activation position is not addressable in this forward",
                details={
                    "plan_intervention_id": item.plan_intervention_id,
                    "positions": rows,
                    "forward_length": self.current_length,
                    "generation_step": self.current_step,
                    "hint": (
                        "use position_mode='last' for a generated token, because a staged "
                        "decode forward contains only the newest position"
                    ),
                },
            )
        return rows

    def _selector(self, item: PlanIntervention, rows: Sequence[int]) -> TensorSelector:
        payload = item.intervention.target.selector.model_dump(mode="python")
        payload["positions"] = [int(row) for row in rows]
        return TensorSelector.model_validate(payload)

    def _process(self, path: str, role: str, value: Any, original: Any) -> Any:
        key = (path, role)
        call_index = self._call_counts.get(key, 0)
        self._call_counts[key] = call_index + 1
        items = self._matching(path, role, call_index)
        if not items:
            return value
        current = value
        for item in items:
            try:
                current = self._apply_one(current, item, call_index)
            except WspError as exc:
                self._record_rejected(item, call_index, exc)
                raise
        return current

    def _apply_one(self, value: Any, item: PlanIntervention, call_index: int) -> Any:
        rows = self._rows(item)
        selector = self._selector(item, rows)
        spec = item.intervention
        tensor, location = extract_activation(value, selector, role=spec.target.tensor_role)
        selection = resolve_selection(tensor, selector, sequence_length=self.current_length)
        token_indices = [row + self.current_offset for row in rows]
        if self.dry_run:
            self.firings.append(
                PreviewFiring(
                    plan_intervention_id=item.plan_intervention_id,
                    generation_step=self.current_step,
                    forward_invocation=call_index,
                    module=item.module or "<root>",
                    tensor_role=spec.target.tensor_role,
                    position_mode=item.position_mode,
                    rows=rows,
                    token_indices=token_indices,
                    tensor_shape=[int(size) for size in tensor.shape],
                    dtype=torch_dtype_name(tensor),
                    axis=selection.axis,
                    elements=len(rows) * selection.width,
                )
            )
            return value

        effective = _spec_variant(spec, "identity") if self.identity else spec
        replacement = None if self.identity else self._replacements[item.plan_intervention_id]
        updated, elements, dtype_cast, before, after = apply_operation(
            tensor,
            selection,
            operation=effective.operation,
            replacement=replacement,
            allow_dtype_cast=effective.allow_dtype_cast,
            allow_position_broadcast=bool(
                effective.donor is not None and effective.donor.allow_position_broadcast
            ),
        )
        before_stats, after_stats, max_abs, mean_abs, _relative, _cosine = compare_slices(
            before, after, rows
        )
        payload = after.detach().to("cpu", torch.float64).contiguous().numpy().tobytes()
        record = InterventionEventRecord(
            event_index=len(self.events) + len(self.rejected),
            plan_id=self.plan.plan_id,
            plan_intervention_id=item.plan_intervention_id,
            intervention_id=spec.intervention_id,
            label=item.label,
            generation_step=self.current_step,
            forward_invocation=call_index,
            module=item.module or "<root>",
            tensor_role=spec.target.tensor_role,
            position_mode=item.position_mode,
            position_semantics=spec.target.position_semantics,
            rows=rows,
            token_indices=token_indices,
            feature_start=selection.feature_start,
            feature_end=selection.feature_end,
            feature_indices=list(selection.feature_indices),
            operation=effective.operation,
            tensor_shape=[int(size) for size in tensor.shape],
            dtype=torch_dtype_name(tensor),
            elements_modified=elements,
            dtype_cast=dtype_cast,
            replacement_sha256=sha256_bytes(payload) if elements else None,
            before=before_stats,
            after=after_stats,
            max_abs=max_abs,
            mean_abs=mean_abs,
            # An identity plan keeps the hook sites live but changes nothing
            # numerically, so its events are recorded as identity events rather
            # than as applied changes.
            execution_result=(
                "identity" if self.identity or effective.operation == "identity" else "applied"
            ),
            provenance={
                "plan_id": self.plan.plan_id,
                "intervention_id": spec.intervention_id,
            },
        )
        self._fired_keys.add((item.plan_intervention_id, self.current_step, call_index))
        self.total_observed += 1
        if len(self.events) < self._events_limit:
            self.events.append(record)
        else:
            self.truncated = True
        return rebuild_activation(value, location, updated)

    def _record_rejected(self, item: PlanIntervention, call_index: int, exc: WspError) -> None:
        record = InterventionEventRecord(
            event_index=len(self.events) + len(self.rejected),
            plan_id=self.plan.plan_id,
            plan_intervention_id=item.plan_intervention_id,
            intervention_id=item.intervention.intervention_id,
            label=item.label,
            generation_step=self.current_step,
            forward_invocation=call_index,
            module=item.module or "<root>",
            tensor_role=item.intervention.target.tensor_role,
            position_mode=item.position_mode,
            position_semantics=item.intervention.target.position_semantics,
            operation=item.intervention.operation,
            execution_result="rejected",
            rejection_reason=str(exc)[:400],
        )
        self.rejected.append(record)

    # ------------------------------------------------------------- accounting

    def event_log(
        self,
        *,
        steps_executed: int,
        condition: str,
        execution_mode: str,
        backend: str,
    ) -> InterventionEventLog:
        expected = [
            ExpectedEvent(
                plan_intervention_id=pi_id,
                generation_step=step,
                on_step_call_index=call_index,
                module=self._module_of(pi_id),
            )
            for pi_id, step, call_index in self.plan.expected_event_keys(steps_executed=steps_executed)
        ]
        missing = [
            item
            for item in expected
            if (item.plan_intervention_id, item.generation_step, item.on_step_call_index)
            not in self._fired_keys
        ]
        not_reached: list[ExpectedEvent] = []
        for item in self.plan.ordered():
            for step in resolve_schedule(item.schedule, steps_executed=steps_executed).not_reached_steps:
                for call_index in item.on_step_call_indices:
                    not_reached.append(
                        ExpectedEvent(
                            plan_intervention_id=item.plan_intervention_id,
                            generation_step=step,
                            on_step_call_index=call_index,
                            module=item.module or "<root>",
                        )
                    )
        log = InterventionEventLog(
            log_id=_new_event_log_id(),
            created_at=utc_now_iso(),
            plan_id=self.plan.plan_id,
            plan_sha256=plan_sha256(self.plan),
            plan_kind=self.plan.plan_kind,
            execution_mode=execution_mode,
            backend=backend,
            condition=condition,
            prompt_length=self.prompt_length,
            steps_executed=steps_executed,
            events=list(self.events),
            expected_events=expected,
            missing_events=missing,
            not_reached_events=not_reached,
            rejected_events=list(self.rejected),
            event_limit=self._events_limit,
            truncated=self.truncated,
            total_events_observed=self.total_observed,
            metadata={
                "identity_run": self.identity,
                "dry_run": self.dry_run,
                "one_time_interventions": [
                    item.plan_intervention_id for item in self.plan.ordered() if item.schedule.one_time
                ],
            },
        )
        log.checksums["event_log_sha256"] = event_log_sha256(log)
        return log

    def _module_of(self, plan_intervention_id: str) -> str:
        for item in self.plan.interventions:
            if item.plan_intervention_id == plan_intervention_id:
                return item.module or "<root>"
        return "<root>"


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


def _logprob_row(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.to(torch.float64), dim=-1)


def _generate_uncached(
    plan: InterventionPlan,
    model: Any,
    prompt: Sequence[int],
    *,
    store: Any | None,
    budget: Budget,
    identity: bool,
    apply: bool,
    dry_run: bool = False,
) -> PlanRunOutcome:
    tokens = [int(item) for item in prompt]
    config = plan.continuation
    if config.max_new_tokens > budget.max_generation_steps:
        raise ResourceLimitError(
            "the continuation exceeds the generation-step budget",
            details={
                "requested": config.max_new_tokens,
                "limit": budget.max_generation_steps,
            },
        )
    if len(plan.interventions) > budget.max_temporal_interventions:
        raise ResourceLimitError(
            "the plan holds more interventions than the budget allows",
            details={
                "requested": len(plan.interventions),
                "limit": budget.max_temporal_interventions,
            },
        )
    root = torch_module(model)
    outcome = PlanRunOutcome(plan=plan, identity=identity, applied=apply)
    session = PlanSession(
        root,
        plan,
        prompt_length=len(tokens),
        store=store,
        budget=budget,
        identity=identity,
        apply=apply,
        dry_run=dry_run,
    ) if (apply or dry_run) else None
    was_training = root.training
    hooks_before = installed_hook_count(root)
    outcome.fingerprint_before = parameter_fingerprint(root)
    outcome.state_before = mutable_state_fingerprint(root)
    rows: list[torch.Tensor] = []
    started = time.perf_counter()
    root.eval()
    try:
        with torch.inference_mode():
            if session is not None:
                session.install()
            try:
                sequence = list(tokens)
                for step in range(config.max_new_tokens):
                    if session is not None:
                        session.begin_step(step, len(sequence), 0)
                    logits = model.forward_token_ids(sequence)
                    checked = _check_logits(logits, spec=plan.interventions[0].intervention, budget=budget)
                    row = checked[0, -1].to(torch.float64)
                    token_id = int(torch.argmax(row).item())
                    outcome.logprobs.append(float(_logprob_row(row)[token_id].item()))
                    outcome.step_top1.append(token_id)
                    if len(rows) < plan.limits.max_observed_logit_rows:
                        rows.append(row.clone())
                    else:
                        outcome.stepwise_truncated = True
                    outcome.generated_token_ids.append(token_id)
                    outcome.steps += 1
                    sequence.append(token_id)
                    if config.stop_on_eos and config.eos_token_id is not None:
                        if token_id == config.eos_token_id:
                            outcome.termination = "eos"
                            break
                    if len(sequence) >= config.max_sequence_length:
                        outcome.termination = "max_sequence_length"
                        break
            finally:
                if session is not None:
                    session.close()
    except WspError as exc:
        outcome.termination = "error"
        outcome.errors.append(str(exc))
    except Exception as exc:  # pragma: no cover - model-specific failure
        outcome.termination = "error"
        outcome.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        root.train(was_training)
        outcome.elapsed_ms = (time.perf_counter() - started) * 1000.0
        outcome.fingerprint_after = parameter_fingerprint(root)
        outcome.state_after = mutable_state_fingerprint(root)
        outcome.hook_residue = installed_hook_count(root) != hooks_before
        if session is not None:
            outcome.events = list(session.events)
            outcome.rejected = list(session.rejected)
            outcome.firings = list(session.firings)
            outcome.truncated = session.truncated
            outcome.total_observed = session.total_observed
            outcome.log = session.event_log(
                steps_executed=outcome.steps,
                condition="identity" if identity else "plan",
                execution_mode=plan.execution_mode,
                backend=plan.backend,
            )
    if rows:
        outcome.step_logits = torch.stack(rows)
    return outcome


def _checked_outcome(outcome: PlanRunOutcome) -> None:
    if outcome.fingerprint_before != outcome.fingerprint_after:
        raise ExecutionError(
            "model parameters changed during the scheduled run; no result should be recorded",
            details={"plan_id": outcome.plan.plan_id},
        )
    if outcome.state_before != outcome.state_after:
        raise ExecutionError(
            "model buffers, training flags, or execution state changed during the run; "
            "the comparison is contaminated",
            details={"plan_id": outcome.plan.plan_id},
        )
    if outcome.hook_residue:
        raise ExecutionError("intervention hooks were not fully removed after the run")


def _diff_between(
    reference: torch.Tensor | None,
    other: torch.Tensor | None,
    *,
    atol: float,
    rtol: float,
    reference_tokens: Sequence[int] | None = None,
    other_tokens: Sequence[int] | None = None,
) -> tuple[ContinuationDiff, list[StepwiseLogitDiff]]:
    """Per-step logit agreement plus the discrete divergence summary.

    ``token_matches`` records whether the greedy choice at each step agreed with
    the reference.  Logit agreement and token agreement are reported separately
    because a real logit change need not move the argmax.
    """

    stepwise: list[StepwiseLogitDiff] = []
    if reference is None or other is None:
        return ContinuationDiff(), stepwise
    steps = min(int(reference.shape[0]), int(other.shape[0]))
    allclose_all = True
    max_abs_all = 0.0
    for step in range(steps):
        left = reference[step]
        right = other[step]
        delta = (right - left).abs()
        max_abs = float(delta.max().item())
        mean_abs = float(delta.mean().item())
        close = bool(torch.allclose(right, left, atol=atol, rtol=rtol))
        allclose_all = allclose_all and close
        max_abs_all = max(max_abs_all, max_abs)
        matches: bool | None = None
        if reference_tokens is not None and other_tokens is not None:
            if step < len(reference_tokens) and step < len(other_tokens):
                matches = int(reference_tokens[step]) == int(other_tokens[step])
        stepwise.append(
            StepwiseLogitDiff(
                step=step,
                max_abs=max_abs,
                mean_abs=mean_abs,
                allclose=close,
                token_matches=matches,
                token_id=(
                    int(other_tokens[step])
                    if other_tokens is not None and step < len(other_tokens)
                    else None
                ),
            )
        )
    return ContinuationDiff(), stepwise


def _continuation_diff(
    baseline_tokens: Sequence[int],
    other_tokens: Sequence[int],
    *,
    baseline_termination: str,
    other_termination: str,
) -> ContinuationDiff:
    common = 0
    for left, right in zip(baseline_tokens, other_tokens):
        if left != right:
            break
        common += 1
    divergence = None if list(baseline_tokens) == list(other_tokens) else common
    longest = max(len(baseline_tokens), len(other_tokens))
    return ContinuationDiff(
        longest_common_prefix=common,
        first_divergence=divergence,
        length_delta=len(other_tokens) - len(baseline_tokens),
        termination_changed=baseline_termination != other_termination,
        token_agreement_rate=(common / longest) if longest else None,
    )


def _decode_text(model: Any, token_ids: Sequence[int]) -> str | None:
    support = getattr(model, "tokenizer_support", None)
    if not callable(support):
        return None
    try:
        tokenizer = support()
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            text = decode(list(token_ids))
            if isinstance(text, str):
                return text
    except Exception:  # pragma: no cover - tokenizer specific
        return None
    return None


def run_temporal_intervention(
    plan: InterventionPlan,
    model: Any,
    prompt_token_ids: Sequence[int],
    *,
    store: Any | None = None,
    budget: Budget | None = None,
    case_id: str | None = None,
    plan_artifact_id: str | None = None,
    identity: bool = True,
    identity_plan_artifact_id: str | None = None,
    identity_plan: InterventionPlan | None = None,
) -> TemporalInterventionRun:
    """Run untouched, identity-control, and scheduled-plan conditions."""

    budget = budget or Budget()
    prompt = [int(item) for item in prompt_token_ids]
    if len(prompt) != plan.prompt_length:
        raise CompatibilityError(
            "the supplied prefix length does not match the plan prompt length",
            details={"prefix": len(prompt), "plan_prompt_length": plan.prompt_length},
        )
    if any(isinstance(item, bool) or item < 0 for item in prompt):
        raise InputError("prompt token ids must be non-negative integers")
    if plan.execution_mode == "staged-generation":
        raise UnsupportedError(
            "this plan requests staged generation; use run_staged_intervention so the "
            "staged verification is recorded with the result",
            details={"plan_id": plan.plan_id},
        )
    validation = validate_plan(plan, root=torch_module(model), budget=budget)
    if not validation.valid:
        raise CompatibilityError(
            "the intervention plan failed validation and was not executed",
            details={"errors": validation.errors[:8]},
        )

    model_info = describe_model(model)
    warnings: list[str] = []
    conditions = []

    baseline = _generate_uncached(
        plan, model, prompt, store=store, budget=budget, identity=False, apply=False
    )
    _checked_outcome(baseline)
    if baseline.termination == "error":
        raise ExecutionError(
            "the baseline run failed, so the experiment is not measurable",
            details={"errors": baseline.errors},
        )
    baseline_log = _event_log_for(baseline, condition="baseline", execution_mode=plan.execution_mode, backend=plan.backend, steps=baseline.steps)
    # A baseline installs no hooks.  An empty log still records that fact.
    conditions.append(
        _condition_outcome("baseline", plan, baseline, baseline_log, model=model)
    )

    identity_outcome: PlanRunOutcome | None = None
    identity_control = IdentityControl(performed=False)
    control_plan = identity_plan or identity_control_plan(plan)
    identity_artifact_id = identity_plan_artifact_id
    if identity:
        identity_outcome = _generate_uncached(
            control_plan, model, prompt, store=store, budget=budget, identity=True, apply=True
        )
        identity_log = _event_log_for(
            identity_outcome,
            condition="identity",
            execution_mode=plan.execution_mode,
            backend=plan.backend,
            steps=identity_outcome.steps,
        )
        conditions.append(
            _condition_outcome("identity", control_plan, identity_outcome, identity_log, model=model)
        )
        identity_control = _identity_control_from(baseline, identity_outcome)
    else:
        warnings.append(
            "the identity control was not performed; multi-site hook instrumentation is "
            "therefore unverified for this run"
        )

    executed = _generate_uncached(
        plan, model, prompt, store=store, budget=budget, identity=False, apply=True
    )
    _checked_outcome(executed)
    executed_log = _event_log_for(
        executed,
        condition="plan",
        execution_mode=plan.execution_mode,
        backend=plan.backend,
        steps=executed.steps,
    )
    conditions.append(_condition_outcome("plan", plan, executed, executed_log, model=model))

    comparison = _continuation_diff(
        baseline.generated_token_ids,
        executed.generated_token_ids,
        baseline_termination=baseline.termination,
        other_termination=executed.termination,
    )
    comparison.baseline_identity_match = (
        None if identity_outcome is None else identity_outcome.generated_token_ids == baseline.generated_token_ids
    )
    comparison.identity_agreement = (
        "not-performed"
        if identity_outcome is None
        else (
            "exact"
            if identity_outcome.generated_token_ids == baseline.generated_token_ids
            and bool(torch.equal(baseline.step_logits, identity_outcome.step_logits))
            else "failed"
        )
    )
    _, stepwise = _diff_between(
        baseline.step_logits,
        executed.step_logits,
        atol=IDENTITY_ATOL,
        rtol=IDENTITY_RTOL,
        reference_tokens=baseline.generated_token_ids,
        other_tokens=executed.generated_token_ids,
    )

    checks = [
        ValidationCheck(name="plan_valid", status="pass"),
        ValidationCheck(
            name="identity_control",
            status=("pass" if identity_control.passed else "fail") if identity_control.performed else "skipped",
            detail=identity_control.detail,
        ),
        ValidationCheck(
            name="all_scheduled_events_fired",
            status="pass" if executed_log.all_fired else "fail",
            detail=(
                f"{len(executed_log.events)} fired, "
                f"{len(executed_log.missing_events)} missing, "
                f"{len(executed_log.rejected_events)} rejected"
            ),
        ),
        ValidationCheck(
            name="parameters_unchanged",
            status="pass" if executed.fingerprint_before == executed.fingerprint_after else "fail",
        ),
        ValidationCheck(
            name="mutable_state_unchanged",
            status="pass" if executed.state_before == executed.state_after else "fail",
        ),
    ]
    reasons: list[str] = []
    status = "valid"
    if identity_control.performed and not identity_control.passed:
        status = "invalid"
        reasons.append(
            "the identity control disagreed with the baseline, so instrumentation itself "
            "changed the computation"
        )
    if executed_log.missing_events:
        status = "invalid"
        reasons.append(
            f"{len(executed_log.missing_events)} scheduled event(s) did not fire at a step "
            "that executed; the plan was not faithfully applied"
        )
    if executed_log.rejected_events:
        status = "invalid"
        reasons.append("at least one scheduled intervention was rejected during execution")
    if executed.termination == "error":
        status = "invalid"
        reasons.append("the scheduled run failed: " + "; ".join(executed.errors))
    if not identity_control.performed:
        status = "limited" if status == "valid" else status
        reasons.append("no identity control was performed")
    if executed_log.not_reached_events:
        warnings.append(
            f"{len(executed_log.not_reached_events)} scheduled event(s) were beyond the "
            "executed continuation and are reported as not reached"
        )
    if executed.stepwise_truncated:
        warnings.append(
            "stepwise logit comparison was truncated at limits.max_observed_logit_rows"
        )

    record = TemporalInterventionRun(
        run_id=_new_temporal_run_id(),
        created_at=utc_now_iso(),
        case_id=case_id,
        plan_id=plan.plan_id,
        plan_sha256=plan_sha256(plan),
        plan_artifact_id=plan_artifact_id,
        identity_plan_id=control_plan.plan_id if identity_outcome is not None else None,
        identity_plan_artifact_id=identity_artifact_id,
        trace_id=plan.provenance.get("trace_id") if isinstance(plan.provenance, dict) else None,
        model=model_info,
        execution_mode=plan.execution_mode,
        backend=plan.backend,
        prompt_length=len(prompt),
        prompt_sha256=sha256_bytes((",".join(str(item) for item in prompt)).encode("utf-8")),
        config=plan.continuation.model_dump(mode="json"),
        conditions=conditions,
        identity_control=identity_control,
        comparison=comparison,
        stepwise_logits=stepwise,
        validity=ValidityReport(status=status, checks=checks, reasons=reasons),
        warnings=warnings,
        metadata={
            "scheduled_events": len(executed_log.expected_events),
            "events_fired": len(executed_log.events),
            "event_log_ids": {
                "baseline": baseline_log.log_id,
                "plan": executed_log.log_id,
                **({"identity": conditions[1].event_log_id} if identity_outcome is not None else {}),
            },
            "one_time_interventions": [
                item.plan_intervention_id for item in plan.ordered() if item.schedule.one_time
            ],
            "sustained_interventions": [
                item.plan_intervention_id for item in plan.ordered() if not item.schedule.one_time
            ],
            "uncached_note": (
                "each generation step re-runs the whole prefix; an activation modified at "
                "one step is not present in the next unless the schedule fires again"
            ),
        },
    )
    record.checksums["temporal_run_sha256"] = temporal_run_sha256(record)
    logs = {"baseline": baseline_log, "plan": executed_log}
    if identity_outcome is not None:
        logs["identity"] = identity_log
    return TemporalOutcome(record=record, event_logs=logs)


def _event_log_for(
    outcome: PlanRunOutcome,
    *,
    condition: str,
    execution_mode: str,
    backend: str,
    steps: int,
) -> InterventionEventLog:
    """Return the session's own log, or a hook-free baseline log.

    The baseline has no session, so its log is constructed directly.  It still
    records that no intervention was intended, which makes "the plan was not
    applied here" explicit rather than an absent field.
    """

    if outcome.log is not None:
        return outcome.log
    expected: list[ExpectedEvent] = []
    log = InterventionEventLog(
        log_id=_new_event_log_id(),
        created_at=utc_now_iso(),
        plan_id=outcome.plan.plan_id,
        plan_sha256=plan_sha256(outcome.plan),
        plan_kind=outcome.plan.plan_kind,
        execution_mode=execution_mode,
        backend=backend,
        condition=condition,
        prompt_length=outcome.plan.prompt_length,
        steps_executed=steps,
        events=list(outcome.events),
        expected_events=expected,
        # A baseline never intends to fire, so its expected list is empty.
        missing_events=[],
        not_reached_events=[],
        rejected_events=list(outcome.rejected),
        event_limit=outcome.plan.limits.max_event_records,
        truncated=outcome.truncated,
        total_events_observed=outcome.total_observed,
        metadata={"condition": condition, "applied": outcome.applied},
    )
    log.checksums["event_log_sha256"] = event_log_sha256(log)
    return log


def _module_for(plan: InterventionPlan, plan_intervention_id: str) -> str:
    for item in plan.interventions:
        if item.plan_intervention_id == plan_intervention_id:
            return item.module or "<root>"
    return "<root>"


def _condition_outcome(
    name: str,
    plan: InterventionPlan,
    outcome: PlanRunOutcome,
    log: InterventionEventLog,
    *,
    model: Any,
) -> Any:
    from .plan_models import ConditionOutcome

    return ConditionOutcome(
        condition=name,
        plan_id=plan.plan_id,
        plan_kind=plan.plan_kind,
        applied=outcome.applied,
        generated_token_ids=list(outcome.generated_token_ids),
        generated_text=_decode_text(model, outcome.generated_token_ids),
        logprobs=list(outcome.logprobs),
        top1_token_ids=list(outcome.step_top1),
        termination_reason=outcome.termination,  # type: ignore[arg-type]
        steps=outcome.steps,
        events_applied=len([item for item in log.events if item.execution_result == "applied"]),
        events_identity=len([item for item in log.events if item.execution_result == "identity"]),
        events_missing=len(log.missing_events),
        events_not_reached=len(log.not_reached_events),
        event_log_id=log.log_id,
        elapsed_ms=round(outcome.elapsed_ms, 3),
        errors=list(outcome.errors),
    )


def _identity_control_from(
    baseline: PlanRunOutcome, identity: PlanRunOutcome
) -> IdentityControl:
    tokens_match = baseline.generated_token_ids == identity.generated_token_ids
    exact = bool(
        tokens_match
        and baseline.step_logits is not None
        and identity.step_logits is not None
        and torch.equal(baseline.step_logits, identity.step_logits)
    )
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
    allclose = bool(
        torch.allclose(
            baseline.step_logits[:steps], identity.step_logits[:steps],
            atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL,
        )
    )
    if exact and steps == int(baseline.step_logits.shape[0]) == int(identity.step_logits.shape[0]):
        agreement = "exact"
        passed = True
        detail = "baseline and identity step logits are bitwise identical"
    elif allclose and tokens_match:
        agreement = "within_tolerance"
        passed = True
        detail = "baseline and identity agree within tolerance but are not bitwise identical"
    else:
        agreement = "failed"
        passed = False
        detail = (
            "baseline and identity disagree beyond tolerance or produce different tokens; "
            "multi-site instrumentation changed the computation"
        )
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


def preview_plan(
    plan: InterventionPlan,
    model: Any,
    prompt_token_ids: Sequence[int],
    *,
    store: Any | None = None,
    budget: Budget | None = None,
) -> PlanPreview:
    """Resolve every scheduled firing against a real model without changing anything."""

    budget = budget or Budget()
    outcome = _generate_uncached(
        plan,
        model,
        prompt_token_ids,
        store=store,
        budget=budget,
        identity=False,
        apply=False,
        dry_run=True,
    )
    _checked_outcome(outcome)
    lines = [
        f"plan: {plan.plan_id}",
        f"steps executed: {outcome.steps}",
        f"resolved firings: {len(outcome.firings)}",
    ]
    for firing in outcome.firings:
        lines.append(
            f"  step {firing.generation_step} call {firing.forward_invocation}: "
            f"{firing.plan_intervention_id} -> {firing.module} "
            f"shape={firing.tensor_shape} axis={firing.axis} rows={firing.rows} "
            f"tokens={firing.token_indices} elements={firing.elements}"
        )
    warnings: list[str] = []
    if outcome.errors:
        warnings.extend(outcome.errors)
    not_reached = [
        (item.plan_intervention_id, step)
        for item in plan.ordered()
        for step in resolve_schedule(item.schedule, steps_executed=outcome.steps).not_reached_steps
    ]
    for plan_intervention_id, step in not_reached:
        lines.append(
            f"  not reached: {plan_intervention_id} was scheduled for generation step "
            f"{step}, beyond the {outcome.steps} executed steps"
        )
    if not_reached:
        warnings.append(
            f"{len(not_reached)} scheduled event(s) were not reached by this execution; "
            "they are reported as pending rather than resolved"
        )
    if outcome.steps < plan.continuation.max_new_tokens:
        warnings.append(
            "the preview stopped before the requested continuation length; the remaining "
            "schedule was not resolved"
        )
    return PlanPreview(
        plan_id=plan.plan_id,
        valid=not outcome.errors,
        steps_executed=outcome.steps,
        firings=outcome.firings,
        human_readable=lines,
        warnings=warnings,
    )


def _new_event_log_id() -> str:
    from .ids import new_event_log_id

    return new_event_log_id()


def _new_temporal_run_id() -> str:
    from .ids import new_temporal_run_id

    return new_temporal_run_id()


# --------------------------------------------------------------------------- #
# inspection
# --------------------------------------------------------------------------- #


class EventSummary(StrictModel):
    log_id: str
    plan_id: str
    condition: str
    steps_executed: int
    fired: int
    missing: int
    not_reached: int
    rejected: int
    truncated: bool
    by_step: dict[str, int] = Field(default_factory=dict)
    by_module: dict[str, int] = Field(default_factory=dict)
    by_operation: dict[str, int] = Field(default_factory=dict)
    event_log_artifact_id: str | None = None
    human_readable: list[str] = Field(default_factory=list)


def summarise_event_log(log: InterventionEventLog) -> EventSummary:
    by_step: dict[str, int] = {}
    by_module: dict[str, int] = {}
    by_operation: dict[str, int] = {}
    for event in log.events:
        by_step[str(event.generation_step)] = by_step.get(str(event.generation_step), 0) + 1
        by_module[event.module] = by_module.get(event.module, 0) + 1
        by_operation[event.operation] = by_operation.get(event.operation, 0) + 1
    lines = [
        f"log: {log.log_id} ({log.condition}) plan={log.plan_id}",
        f"steps executed: {log.steps_executed}; fired: {len(log.events)}; "
        f"missing: {len(log.missing_events)}; not reached: {len(log.not_reached_events)}; "
        f"rejected: {len(log.rejected_events)}",
    ]
    for event in log.events[:16]:
        lines.append(
            f"  step {event.generation_step} call {event.forward_invocation}: "
            f"{event.plan_intervention_id} {event.operation} {event.module} "
            f"rows={event.rows} tokens={event.token_indices} "
            f"elements={event.elements_modified}"
        )
    if len(log.events) > 16:
        lines.append(f"  ... {len(log.events) - 16} more event(s)")
    return EventSummary(
        log_id=log.log_id,
        plan_id=log.plan_id,
        condition=log.condition,
        steps_executed=log.steps_executed,
        fired=len(log.events),
        missing=len(log.missing_events),
        not_reached=len(log.not_reached_events),
        rejected=len(log.rejected_events),
        truncated=log.truncated,
        by_step=by_step,
        by_module=by_module,
        by_operation=by_operation,
        human_readable=lines,
    )


__all__ = [
    "EventSummary",
    "PlanPreview",
    "PlanRunOutcome",
    "PlanSession",
    "PreviewFiring",
    "installed_hook_count",
    "mutable_state_fingerprint",
    "preview_plan",
    "run_temporal_intervention",
    "summarise_event_log",
]
