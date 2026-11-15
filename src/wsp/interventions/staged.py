"""Verified staged/cached execution under a scheduled intervention plan.

Correctness before speed
------------------------

A staged run is only recorded as causal evidence after an *unmodified* staged
run has been compared against the uncached reference on:

* greedy token choices per step;
* stepwise next-token logits;
* termination decisions;
* an identity-control plan that preserves every hook but changes no number.

Matching generated text is not accepted as evidence of equivalence.  A run whose
tokens or logits disagree is stored as failed, and the containing record can
never be ``valid``.

Cache isolation
---------------

Every condition rebuilds its cache from the prefix by default.  ``clone`` mode
deep-copies the state and then re-hashes the source to prove the branch did not
touch it.  A shallow copy fails that check by construction, which is what the
offline tests exercise.
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from wsp.core.budget import Budget
from wsp.core.errors import (
    CompatibilityError,
    ExecutionError,
    ResourceLimitError,
    UnsupportedError,
    WspError,
)
from wsp.core.hashing import sha256_bytes
from wsp.core.time import utc_now_iso
from wsp.execution.staged import (
    StagedDecoder,
    assert_clone_isolation,
    build_staged_decoder,
)

from .engine import _check_logits, describe_model, torch_module
from .models import IdentityControl, ValidationCheck, ValidityReport
from .plan import InterventionPlan, identity_control_plan, plan_sha256, validate_plan
from .plan_engine import (
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    PlanRunOutcome,
    PlanSession,
    _condition_outcome,
    _event_log_for,
    _generate_uncached,
    _logprob_row,
    _continuation_diff,
    mutable_state_fingerprint,
    installed_hook_count,
    _checked_outcome,
)
from .plan_models import (
    ConditionOutcome,
    StagedContinuation,
    StagedPerformance,
    StagedStep,
    StagedVerification,
    STAGED_VERIFICATION_ATOL,
    STAGED_VERIFICATION_RTOL,
    staged_continuation_sha256,
    staged_verification_sha256,
)
from wsp.replay.engine import parameter_fingerprint


@dataclass
class StagedRun:
    outcome: PlanRunOutcome
    prefill_ms: float = 0.0
    decode_ms: list[float] = field(default_factory=list)
    prefill_state_hash: str | None = None
    state_bytes: int | None = None
    isolation_verified: bool = True

    @property
    def decode_total_ms(self) -> float:
        return float(sum(self.decode_ms))


@dataclass
class StagedOutcome:
    record: StagedContinuation
    event_logs: dict[str, Any] = field(default_factory=dict)
    verification: StagedVerification | None = None


def _generate_staged(
    plan: InterventionPlan,
    decoder: StagedDecoder,
    prompt: Sequence[int],
    *,
    store: Any | None,
    budget: Budget,
    identity: bool,
    apply: bool,
    isolation: str = "recompute",
) -> StagedRun:
    tokens = [int(item) for item in prompt]
    config = plan.continuation
    root = decoder.root()
    outcome = PlanRunOutcome(plan=plan, identity=identity, applied=apply)
    session = (
        PlanSession(
            root,
            plan,
            prompt_length=len(tokens),
            store=store,
            budget=budget,
            identity=identity,
            apply=apply,
        )
        if apply
        else None
    )
    run = StagedRun(outcome=outcome)
    was_training = root.training
    hooks_before = installed_hook_count(root)
    outcome.fingerprint_before = parameter_fingerprint(root)
    outcome.state_before = mutable_state_fingerprint(root)
    rows: list[torch.Tensor] = []
    started = time.perf_counter()
    root.eval()
    state: Any = None
    original_state: Any = None
    original_hash: str | None = None
    try:
        with torch.inference_mode():
            if session is not None:
                session.install()
            try:
                for step in range(config.max_new_tokens):
                    if step == 0:
                        if session is not None:
                            session.begin_step(0, len(tokens), 0)
                        prefill_started = time.perf_counter()
                        logits, state = decoder.prefill(tokens)
                        run.prefill_ms = (time.perf_counter() - prefill_started) * 1000.0
                        run.prefill_state_hash = decoder.state_hash(state)
                        run.state_bytes = decoder.state_bytes(state)
                        if run.state_bytes > budget.max_staged_state_bytes:
                            raise ResourceLimitError(
                                "the generation cache exceeds the configured state budget",
                                details={
                                    "state_bytes": run.state_bytes,
                                    "limit": budget.max_staged_state_bytes,
                                },
                            )
                        if isolation == "clone":
                            original_state = state
                            original_hash = run.prefill_state_hash
                            state = decoder.clone(state)
                            assert_clone_isolation(
                                decoder, original_state, original_hash, action="staged branch"
                            )
                        elif isolation != "recompute":
                            raise CompatibilityError(
                                "unknown cache isolation strategy",
                                details={"isolation": isolation, "allowed": ["recompute", "clone"]},
                            )
                    else:
                        if session is not None:
                            # A decode forward carries exactly one activation row;
                            # its full-sequence index is prompt_length + step - 1.
                            session.begin_step(step, 1, len(tokens) + step - 1)
                        decode_started = time.perf_counter()
                        logits, state = decoder.step(state, outcome.generated_token_ids[-1])
                        run.decode_ms.append((time.perf_counter() - decode_started) * 1000.0)
                    checked = _check_logits(
                        logits, spec=plan.interventions[0].intervention, budget=budget
                    )
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
                    if config.stop_on_eos and config.eos_token_id is not None:
                        if token_id == config.eos_token_id:
                            outcome.termination = "eos"
                            break
                    if len(tokens) + outcome.steps >= config.max_sequence_length:
                        outcome.termination = "max_sequence_length"
                        break
                    if decoder.state_length(state) >= config.max_sequence_length:
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
        if original_state is not None and original_hash is not None:
            try:
                assert_clone_isolation(
                    decoder, original_state, original_hash, action="staged branch"
                )
            except WspError:
                run.isolation_verified = False
    if rows:
        outcome.step_logits = torch.stack(rows)
    return run


def _stepwise(
    reference: PlanRunOutcome, other: PlanRunOutcome, *, atol: float, rtol: float
) -> tuple[list[StagedStep], bool, float | None, float | None, bool]:
    steps: list[StagedStep] = []
    allclose_all = True
    max_abs_all: float | None = None
    mean_abs_sum = 0.0
    compared = 0
    if reference.step_logits is None or other.step_logits is None:
        return steps, False, None, None, False
    total = min(int(reference.step_logits.shape[0]), int(other.step_logits.shape[0]))
    exact = int(reference.step_logits.shape[0]) == int(other.step_logits.shape[0])
    for step in range(total):
        left = reference.step_logits[step]
        right = other.step_logits[step]
        delta = (right - left).abs()
        max_abs = float(delta.max().item())
        mean_abs = float(delta.mean().item())
        close = bool(torch.allclose(right, left, atol=atol, rtol=rtol))
        allclose_all = allclose_all and close
        exact = exact and bool(torch.equal(left, right))
        max_abs_all = max_abs if max_abs_all is None else max(max_abs_all, max_abs)
        mean_abs_sum += mean_abs
        compared += 1
        token_left = (
            reference.generated_token_ids[step]
            if step < len(reference.generated_token_ids)
            else None
        )
        token_right = (
            other.generated_token_ids[step]
            if step < len(other.generated_token_ids)
            else None
        )
        steps.append(
            StagedStep(
                step=step,
                token_uncached=token_left,
                token_staged=token_right,
                token_matches=token_left == token_right,
                max_abs_logits=max_abs,
                mean_abs_logits=mean_abs,
                allclose=close,
            )
        )
    mean_abs_all = (mean_abs_sum / compared) if compared else None
    return steps, allclose_all, max_abs_all, mean_abs_all, exact


def verify_staged_generation(
    plan: InterventionPlan,
    model: Any,
    prompt_token_ids: Sequence[int],
    *,
    budget: Budget | None = None,
    decoder: StagedDecoder | None = None,
    identity: bool = True,
    atol: float = STAGED_VERIFICATION_ATOL,
    rtol: float = STAGED_VERIFICATION_RTOL,
    isolation: str = "recompute",
) -> tuple[StagedVerification, dict[str, float]]:
    """Compare an unmodified staged run with the uncached reference."""

    budget = budget or Budget()
    prompt = [int(item) for item in prompt_token_ids]
    if len(prompt) != plan.prompt_length:
        raise CompatibilityError(
            "the supplied prefix length does not match the plan prompt length",
            details={"prefix": len(prompt), "plan_prompt_length": plan.prompt_length},
        )
    decoder = decoder or build_staged_decoder(model)
    capabilities = decoder.capabilities()
    if not capabilities.supported:
        raise UnsupportedError(
            "staged generation is not supported for this model",
            details={
                "architecture_class": capabilities.architecture_class,
                "reason": capabilities.reason,
            },
        )
    if plan.execution_mode != "staged-generation":
        raise CompatibilityError(
            "verification requires a plan marked for staged generation",
            details={"execution_mode": plan.execution_mode},
        )

    started = time.perf_counter()
    reference = _generate_uncached(
        plan, model, prompt, store=None, budget=budget, identity=False, apply=False
    )
    uncached_ms = (time.perf_counter() - started) * 1000.0
    _checked_outcome(reference)
    staged = _generate_staged(
        plan,
        decoder,
        prompt,
        store=None,
        budget=budget,
        identity=False,
        apply=False,
        isolation=isolation,
    )
    _checked_outcome(staged.outcome)

    steps, allclose_all, max_abs, mean_abs, exact = _stepwise(
        reference, staged.outcome, atol=atol, rtol=rtol
    )
    tokens_match = reference.generated_token_ids == staged.outcome.generated_token_ids
    termination_match = reference.termination == staged.outcome.termination

    identity_agreement: str = "not-performed"
    if identity:
        control_plan = identity_control_plan(plan)
        staged_identity = _generate_staged(
            control_plan,
            decoder,
            prompt,
            store=None,
            budget=budget,
            identity=True,
            apply=True,
            isolation=isolation,
        )
        if staged_identity.outcome.generated_token_ids == staged.outcome.generated_token_ids and (
            staged.outcome.step_logits is not None
            and staged_identity.outcome.step_logits is not None
            and bool(torch.equal(staged.outcome.step_logits, staged_identity.outcome.step_logits))
        ):
            identity_agreement = "exact"
        elif staged_identity.outcome.generated_token_ids == staged.outcome.generated_token_ids:
            identity_agreement = "within_tolerance"
        else:
            identity_agreement = "failed"

    if tokens_match and termination_match and exact:
        agreement = "exact"
    elif tokens_match and termination_match and allclose_all:
        agreement = "within_tolerance"
    else:
        agreement = "failed"

    checks = [
        ValidationCheck(
            name="token_sequence",
            status="pass" if tokens_match else "fail",
            detail="greedy token choices agree with the uncached reference"
            if tokens_match
            else "greedy tokens diverge; text agreement alone is not equivalence",
        ),
        ValidationCheck(
            name="stepwise_logits",
            status="pass" if allclose_all else "fail",
            detail=f"atol={atol}, rtol={rtol}",
        ),
        ValidationCheck(
            name="termination",
            status="pass" if termination_match else "fail",
            detail=f"uncached={reference.termination}, staged={staged.outcome.termination}",
        ),
        ValidationCheck(
            name="identity_control",
            status=(
                "pass"
                if identity_agreement in ("exact", "within_tolerance")
                else ("skipped" if identity_agreement == "not-performed" else "fail")
            ),
            detail=f"staged identity agreement: {identity_agreement}",
        ),
        ValidationCheck(
            name="cache_isolation",
            status="pass" if staged.isolation_verified else "fail",
            detail=f"strategy: {isolation}",
        ),
    ]
    warnings: list[str] = []
    if agreement == "within_tolerance":
        warnings.append(
            "staged and uncached logits agree within the declared tolerance but are not "
            "bitwise identical; cached decoding changes reduction order"
        )
    if identity_agreement == "within_tolerance":
        warnings.append(
            "the staged identity control agrees within tolerance rather than bitwise"
        )

    record = StagedVerification(
        verification_id=_new_verification_id(),
        created_at=utc_now_iso(),
        model=describe_model(model),
        execution_backend=decoder.backend,
        execution_mode="cached-decode",
        architecture_class=capabilities.architecture_class,
        prompt_length=len(prompt),
        prompt_sha256=sha256_bytes((",".join(str(item) for item in prompt)).encode("utf-8")),
        config=plan.continuation.model_dump(mode="json"),
        steps=max(reference.steps, staged.outcome.steps),
        agreement=agreement,  # type: ignore[arg-type]
        exact_logits=exact,
        atol=atol,
        rtol=rtol,
        max_abs_logits=max_abs,
        mean_abs_logits=mean_abs,
        token_sequence_match=tokens_match,
        termination_match=termination_match,
        identity_control_agreement=identity_agreement,  # type: ignore[arg-type]
        position_ids_mode=capabilities.position_ids_mode,
        attention_mask_mode=capabilities.attention_mask_mode,
        state_identity_sha256=staged.prefill_state_hash,
        state_bytes=staged.state_bytes,
        isolation=isolation,  # type: ignore[arg-type]
        isolation_verified=staged.isolation_verified,
        steps_detail=steps,
        checks=checks,
        warnings=warnings,
        metadata={
            "uncached_reference_ms": round(uncached_ms, 3),
            "staged_total_ms": round(staged.outcome.elapsed_ms, 3),
            "prefill_ms": round(staged.prefill_ms, 3),
            "decode_ms_total": round(staged.decode_total_ms, 3),
            "state_kind": capabilities.state_kind,
            "note": "verification runs install no intervention hooks",
        },
    )
    record.checksums["staged_verification_sha256"] = staged_verification_sha256(record)
    timings = {
        "uncached_ms": uncached_ms,
        "staged_total_ms": staged.outcome.elapsed_ms,
        "prefill_ms": staged.prefill_ms,
        "decode_ms_total": staged.decode_total_ms,
    }
    return record, timings


def _peak_rss_bytes() -> int | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception:  # pragma: no cover - platform specific
        return None
    # Linux reports kilobytes, macOS bytes.  The platform is recorded with the
    # measurement, and this release only claims Linux behaviour.
    return int(usage.ru_maxrss) * 1024


def run_staged_intervention(
    plan: InterventionPlan,
    model: Any,
    prompt_token_ids: Sequence[int],
    *,
    store: Any | None = None,
    budget: Budget | None = None,
    case_id: str | None = None,
    plan_artifact_id: str | None = None,
    decoder: StagedDecoder | None = None,
    identity: bool = True,
    identity_plan_artifact_id: str | None = None,
    isolation: str = "recompute",
    atol: float = STAGED_VERIFICATION_ATOL,
    rtol: float = STAGED_VERIFICATION_RTOL,
) -> StagedContinuation:
    """Run a verified staged continuation under a scheduled plan."""

    budget = budget or Budget()
    prompt = [int(item) for item in prompt_token_ids]
    if plan.execution_mode != "staged-generation":
        raise CompatibilityError(
            "run_staged_intervention requires a plan marked for staged generation",
            details={"execution_mode": plan.execution_mode},
        )
    validation = validate_plan(plan, root=torch_module(model), budget=budget)
    if not validation.valid:
        raise CompatibilityError(
            "the staged plan failed validation and was not executed",
            details={"errors": validation.errors[:8]},
        )
    decoder = decoder or build_staged_decoder(model)
    verification, timings = verify_staged_generation(
        plan,
        model,
        prompt,
        budget=budget,
        decoder=decoder,
        identity=identity,
        atol=atol,
        rtol=rtol,
        isolation=isolation,
    )
    if verification.agreement not in ("exact", "within_tolerance"):
        raise ExecutionError(
            "staged execution did not reproduce the uncached reference; no intervention "
            "result from it can be interpreted",
            details={
                "agreement": verification.agreement,
                "tokens_match": verification.token_sequence_match,
                "max_abs_logits": verification.max_abs_logits,
                "verification_id": verification.verification_id,
            },
        )

    conditions: list[ConditionOutcome] = []
    baseline = _generate_staged(
        plan, decoder, prompt, store=store, budget=budget, identity=False, apply=False,
        isolation=isolation,
    )
    _checked_outcome(baseline.outcome)
    baseline_log = _event_log_for(
        baseline.outcome,
        condition="baseline",
        execution_mode=plan.execution_mode,
        backend=plan.backend,
        steps=baseline.outcome.steps,
    )
    conditions.append(
        _condition_outcome("baseline", plan, baseline.outcome, baseline_log, model=model)
    )

    partial = False
    identity_run: StagedRun | None = None
    identity_control = IdentityControl(performed=False)
    control_plan = identity_control_plan(plan)
    if identity:
        identity_run = _generate_staged(
            control_plan, decoder, prompt, store=store, budget=budget, identity=True,
            apply=True, isolation=isolation,
        )
        _checked_outcome(identity_run.outcome)
        identity_log = _event_log_for(
            identity_run.outcome,
            condition="identity",
            execution_mode=plan.execution_mode,
            backend=plan.backend,
            steps=identity_run.outcome.steps,
        )
        conditions.append(
            _condition_outcome("identity", control_plan, identity_run.outcome, identity_log, model=model)
        )
        identity_control = _identity_from_staged(baseline.outcome, identity_run.outcome)
        partial = not identity_run.isolation_verified

    executed = _generate_staged(
        plan, decoder, prompt, store=store, budget=budget, identity=False, apply=True,
        isolation=isolation,
    )
    _checked_outcome(executed.outcome)
    executed_log = _event_log_for(
        executed.outcome,
        condition="plan",
        execution_mode=plan.execution_mode,
        backend=plan.backend,
        steps=executed.outcome.steps,
    )
    conditions.append(
        _condition_outcome("plan", plan, executed.outcome, executed_log, model=model)
    )

    comparison = _continuation_diff(
        baseline.outcome.generated_token_ids,
        executed.outcome.generated_token_ids,
        baseline_termination=baseline.outcome.termination,
        other_termination=executed.outcome.termination,
    )
    comparison.identity_agreement = (
        "not-performed"
        if identity_run is None
        else (
            "exact"
            if identity_run.outcome.generated_token_ids == baseline.outcome.generated_token_ids
            else "failed"
        )
    )
    steps, _allclose, _max, _mean, _exact = _stepwise(
        baseline.outcome, executed.outcome, atol=atol, rtol=rtol
    )

    checks = [
        ValidationCheck(
            name="staged_verification",
            status="pass" if verification.agreement in ("exact", "within_tolerance") else "fail",
            detail=f"agreement: {verification.agreement}",
        ),
        ValidationCheck(
            name="identity_control",
            status=(
                "pass" if identity_control.passed else "fail"
            ) if identity_control.performed else "skipped",
            detail=identity_control.detail,
        ),
        ValidationCheck(
            name="all_scheduled_events_fired",
            status="pass" if executed_log.all_fired else "fail",
            detail=(
                f"{len(executed_log.events)} fired, {len(executed_log.missing_events)} missing, "
                f"{len(executed_log.rejected_events)} rejected"
            ),
        ),
        ValidationCheck(
            name="parameters_unchanged",
            status="pass"
            if executed.outcome.fingerprint_before == executed.outcome.fingerprint_after
            else "fail",
        ),
        ValidationCheck(
            name="mutable_state_unchanged",
            status="pass" if executed.outcome.state_before == executed.outcome.state_after else "fail",
        ),
        ValidationCheck(
            name="cache_isolation",
            status="pass"
            if (executed.isolation_verified and not partial)
            else "fail",
            detail=f"strategy: {isolation}",
        ),
    ]
    reasons: list[str] = []
    status = "valid"
    if verification.agreement not in ("exact", "within_tolerance"):
        status = "invalid"
        reasons.append("the staged decoder did not reproduce the uncached reference")
    if identity_control.performed and not identity_control.passed:
        status = "invalid"
        reasons.append("the staged identity control disagreed with the staged baseline")
    if executed_log.missing_events:
        status = "invalid"
        reasons.append("a scheduled event did not fire at a step that executed")
    if executed_log.rejected_events:
        status = "invalid"
        reasons.append("a scheduled intervention was rejected during staged execution")
    if not executed.isolation_verified or partial:
        status = "invalid"
        reasons.append("cache isolation could not be verified for every branch")
    if not identity_control.performed:
        status = "limited" if status == "valid" else status
        reasons.append("no identity control was performed")
    warnings: list[str] = []
    if verification.agreement == "within_tolerance":
        warnings.append(
            "staged agreement is tolerance bounded; see the verification record"
        )
    if executed_log.not_reached_events:
        warnings.append(
            f"{len(executed_log.not_reached_events)} scheduled event(s) were not reached"
        )

    performance = StagedPerformance(
        prefill_ms=round(executed.prefill_ms, 3),
        decode_ms_total=round(executed.decode_total_ms, 3),
        decode_ms_per_step=round(executed.decode_total_ms / max(len(executed.decode_ms), 1), 3),
        total_ms=round(executed.outcome.elapsed_ms, 3),
        uncached_reference_ms=round(timings.get("uncached_ms", 0.0), 3),
        speedup_ratio=(
            round(timings["uncached_ms"] / executed.outcome.elapsed_ms, 3)
            if timings.get("uncached_ms") and executed.outcome.elapsed_ms > 0
            else None
        ),
        peak_process_rss_bytes=_peak_rss_bytes(),
        cache_state_bytes=executed.state_bytes,
    )

    record = StagedContinuation(
        continuation_id=_new_staged_id(),
        created_at=utc_now_iso(),
        case_id=case_id,
        plan_id=plan.plan_id,
        plan_sha256=plan_sha256(plan),
        plan_artifact_id=plan_artifact_id,
        identity_plan_id=control_plan.plan_id if identity_run is not None else None,
        trace_id=plan.provenance.get("trace_id") if isinstance(plan.provenance, dict) else None,
        model=describe_model(model),
        backend=plan.backend,
        prompt_length=len(prompt),
        prompt_sha256=sha256_bytes((",".join(str(item) for item in prompt)).encode("utf-8")),
        config=plan.continuation.model_dump(mode="json"),
        verification=verification,
        conditions=conditions,
        identity_control=identity_control,
        comparison=comparison,
        stepwise_logits=steps,
        performance=performance,
        validity=ValidityReport(status=status, checks=checks, reasons=reasons),
        warnings=warnings,
        metadata={
            "scheduled_events": len(executed_log.expected_events),
            "events_fired": len(executed_log.events),
            "event_log_ids": {
                "baseline": baseline_log.log_id,
                "plan": executed_log.log_id,
                **({"identity": conditions[1].event_log_id} if identity_run is not None else {}),
            },
            "cache_isolation": isolation,
            "uncached_semantics_note": (
                "staged decode forwards carry one activation row, whose full-sequence index "
                "is prompt_length + step - 1"
            ),
        },
    )
    record.checksums["staged_continuation_sha256"] = staged_continuation_sha256(record)
    logs = {"baseline": baseline_log, "plan": executed_log}
    if identity_run is not None:
        logs["identity"] = identity_log
    return StagedOutcome(record=record, event_logs=logs, verification=verification)


def _identity_from_staged(baseline: PlanRunOutcome, identity: PlanRunOutcome) -> IdentityControl:
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
        detail = "staged baseline and identity logits are bitwise identical"
    elif allclose and tokens_match:
        agreement, passed = "within_tolerance", True
        detail = "staged baseline and identity agree within tolerance"
    else:
        agreement, passed = "failed", False
        detail = "staged instrumentation changed the computation"
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


def _new_verification_id() -> str:
    from .ids import new_staged_verification_id

    return new_staged_verification_id()


def _new_staged_id() -> str:
    from .ids import new_staged_id

    return new_staged_id()


__all__ = [
    "StagedOutcome",
    "StagedRun",
    "run_staged_intervention",
    "verify_staged_generation",
]
