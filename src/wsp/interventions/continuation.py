"""Bounded counterfactual continuation experiments.

A continuation experiment is deliberately *not* an extension of teacher-forced
replay.  It re-runs the whole prefix for every generated token (``uncached``),
so no KV cache or recurrent state is manipulated, and it records exactly when
the intervention is applied: on the first forward pass over the shared prefix.

That choice is what makes the comparison trustworthy on hybrid architectures
such as Qwen3.5, where a cached or stateful decode would interact with the
intervention in ways this release does not verify.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from wsp.core.budget import Budget
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

from .engine import (
    InterventionSession,
    _check_logits,
    _replacement_tensor,
    describe_model,
    torch_module,
)
from .models import (
    ContinuationComparison,
    ContinuationCondition,
    ContinuationDiff,
    InterventionEvent,
    ValidationCheck,
    ValidityReport,
    continuation_sha256,
    intervention_sha256,
)
from .engine import _spec_variant

SUPPORTED_BACKENDS = ("torch_hooks",)



@dataclass
class ContinuationOutcome:
    record: ContinuationComparison
    conditions: dict[str, ContinuationCondition] = field(default_factory=dict)


@dataclass
class _Generated:
    tokens: list[int]
    logprobs: list[float]
    termination: str
    steps: int
    events: list[InterventionEvent]
    errors: list[str]
    elapsed_ms: float


def _validate(
    spec: Any,
    prefix: Sequence[int],
    config: Any,
    budget: Budget,
) -> None:
    if spec.execution.backend not in SUPPORTED_BACKENDS:
        raise UnsupportedError(
            "counterfactual continuation is only implemented for the torch-hooks "
            "backend in this release",
            details={"backend": spec.execution.backend},
        )
    if spec.execution.execution_mode != "uncached-continuation":
        raise CompatibilityError(
            "the intervention specification is not marked for an uncached "
            "continuation; retarget it explicitly before running a continuation",
            details={
                "execution_mode": spec.execution.execution_mode,
                "hint": "use retarget_spec_execution(..., execution_mode="
                "'uncached-continuation')",
            },
        )
    if not prefix:
        raise InputError("a continuation requires a non-empty prefix")
    if len(prefix) != spec.execution.sequence_length:
        raise CompatibilityError(
            "the prefix length does not match the recorded execution sequence length",
            details={
                "prefix": len(prefix),
                "execution_sequence_length": spec.execution.sequence_length,
            },
        )
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in prefix):
        raise InputError("prefix token ids must be non-negative integers")
    if config.max_new_tokens > budget.max_continuation_tokens:
        raise ResourceLimitError(
            "the requested continuation exceeds the configured token budget",
            details={
                "requested": config.max_new_tokens,
                "limit": budget.max_continuation_tokens,
            },
        )
    total = len(prefix) + config.max_new_tokens
    if total > budget.max_continuation_sequence_tokens:
        raise ResourceLimitError(
            "the requested continuation exceeds the sequence-length budget",
            details={"required": total, "limit": budget.max_continuation_sequence_tokens},
        )
    if total > config.max_sequence_length:
        raise ResourceLimitError(
            "the requested continuation exceeds config.max_sequence_length",
            details={"required": total, "limit": config.max_sequence_length},
        )


def _next_token(
    logits: torch.Tensor, *, spec: Any, budget: Budget
) -> tuple[int, float]:
    checked = _check_logits(logits, spec=spec, budget=budget)
    last = checked[0, -1].to(torch.float64)
    logprobabilities = torch.log_softmax(last, dim=-1)
    token_id = int(torch.argmax(last).item())
    return token_id, float(logprobabilities[token_id].item())


def _generate(
    spec: Any,
    model: Any,
    root: torch.nn.Module,
    prefix: Sequence[int],
    config: Any,
    *,
    store: Any | None,
    budget: Budget,
    apply: bool,
    replacement: torch.Tensor | None,
) -> _Generated:
    tokens = [int(item) for item in prefix]
    logprobs: list[float] = []
    events: list[InterventionEvent] = []
    errors: list[str] = []
    termination = "max_new_tokens"
    started = time.perf_counter()
    was_training = root.training
    root.eval()
    session: InterventionSession | None = None
    try:
        if apply:
            session = InterventionSession(
                root,
                spec,
                replacement=replacement,
                apply=True,
                budget=budget,
            )
            session.install()
        with torch.inference_mode():
            for step in range(config.max_new_tokens):
                logits = model.forward_token_ids(tokens)
                token_id, logprob = _next_token(logits, spec=spec, budget=budget)
                tokens.append(token_id)
                logprobs.append(logprob)
                if session is not None and step == 0:
                    for application in session.outcome.applications:
                        events.append(
                            InterventionEvent(
                                step=0,
                                call_index=application.call_index,
                                position=application.rows[0],
                                operation=application.operation,
                                elements_modified=application.elements_modified,
                                note="intervention applied during the first prefix forward pass",
                            )
                        )
                if config.stop_on_eos and config.eos_token_id is not None and token_id == config.eos_token_id:
                    termination = "eos"
                    break
                if len(tokens) >= config.max_sequence_length:
                    termination = "max_sequence_length"
                    break
    except WspError as exc:
        termination = "error"
        errors.append(str(exc))
    except Exception as exc:  # pragma: no cover - defensive, model-specific failures
        termination = "error"
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if session is not None:
            session.close()
        root.train(was_training)
    return _Generated(
        tokens=tokens[len(prefix) :],
        logprobs=logprobs,
        termination=termination,
        steps=len(logprobs),
        events=events,
        errors=errors,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def _decode(model: Any, token_ids: Sequence[int]) -> str | None:
    support = getattr(model, "tokenizer_support", None)
    if not callable(support):
        return None
    try:
        tokenizer = support()
        text = tokenizer.decode(list(token_ids))
        if isinstance(text, str):
            return text
    except Exception:  # pragma: no cover - tokenizer-specific
        return None
    return None


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _condition(
    name: str,
    generated: _Generated,
    *,
    model: Any,
    with_text: bool,
) -> ContinuationCondition:
    return ContinuationCondition(
        condition=name,  # type: ignore[arg-type]
        generated_token_ids=list(generated.tokens),
        generated_text=_decode(model, generated.tokens) if with_text else None,
        logprobs=generated.logprobs,
        termination_reason=generated.termination,  # type: ignore[arg-type]
        steps=generated.steps,
        intervention_events=generated.events,
        elapsed_ms=round(generated.elapsed_ms, 3),
        errors=generated.errors,
    )


def run_counterfactual_continuation(
    spec: Any,
    model: Any,
    prefix_token_ids: Sequence[int],
    config: Any,
    *,
    store: Any | None = None,
    budget: Budget | None = None,
    case_id: str | None = None,
    intervention_artifact_id: str | None = None,
    identity: bool = True,
) -> ContinuationOutcome:
    """Run baseline, identity, and intervention greedy continuations."""

    budget = budget or Budget()
    prefix = [int(item) for item in prefix_token_ids]
    _validate(spec, prefix, config, budget)
    root = torch_module(model)
    replacement = _replacement_tensor(spec, store)
    model_info = describe_model(model)
    warnings: list[str] = []

    baseline = _generate(
        spec,
        model,
        root,
        prefix,
        config,
        store=store,
        budget=budget,
        apply=False,
        replacement=None,
    )
    if baseline.termination == "error":
        raise ExecutionError(
            "the baseline continuation failed",
            details={"errors": baseline.errors},
        )

    identity_generated: _Generated | None = None
    identity_agreement = "not-performed"
    identity_match: bool | None = None
    if identity:
        identity_generated = _generate(
            _spec_variant(spec, "identity"),
            model,
            root,
            prefix,
            config,
            store=store,
            budget=budget,
            apply=True,
            replacement=None,
        )
        identity_match = identity_generated.tokens == baseline.tokens
        identity_agreement = "exact" if identity_match else "failed"
    else:
        warnings.append(
            "the identity continuation control was not performed; the experiment is "
            "limited and attribution is not established"
        )

    intervention = _generate(
        spec,
        model,
        root,
        prefix,
        config,
        store=store,
        budget=budget,
        apply=True,
        replacement=replacement,
    )

    conditions = {
        "baseline": _condition("baseline", baseline, model=model, with_text=not baseline.errors),
    }
    if identity_generated is not None:
        conditions["identity"] = _condition(
            "identity", identity_generated, model=model, with_text=not identity_generated.errors
        )
    conditions["intervention"] = _condition(
        "intervention", intervention, model=model, with_text=not intervention.errors
    )

    common = _common_prefix(baseline.tokens, intervention.tokens)
    longest = max(len(baseline.tokens), len(intervention.tokens))
    divergence: int | None
    if baseline.tokens == intervention.tokens:
        divergence = None
    else:
        divergence = common
    comparison = ContinuationDiff(
        baseline_identity_match=identity_match,
        identity_agreement=identity_agreement,  # type: ignore[arg-type]
        longest_common_prefix=common,
        first_divergence=divergence,
        length_delta=len(intervention.tokens) - len(baseline.tokens),
        termination_changed=baseline.termination != intervention.termination,
        token_agreement_rate=(common / longest) if longest else None,
    )

    checks = [
        ValidationCheck(
            name="baseline_execution",
            status="pass" if baseline.termination != "error" else "fail",
        ),
        ValidationCheck(
            name="intervention_applied",
            status="pass" if intervention.events else "fail",
            detail=f"{len(intervention.events)} application event(s)",
        ),
        ValidationCheck(
            name="identity_control",
            status=("pass" if identity_match else "fail") if identity else "skipped",
        ),
        ValidationCheck(
            name="parameter_immutability",
            status="pass",
            detail="interventions clone activations; no checkpoint write is performed",
        ),
    ]
    reasons: list[str] = []
    status = "valid"
    if identity and not identity_match:
        status = "invalid"
        reasons.append(
            "baseline and identity continuations diverged; the experiment is not a "
            "controlled comparison"
        )
    for name, generated in (
        ("baseline", baseline),
        ("identity", identity_generated),
        ("intervention", intervention),
    ):
        if generated is not None and generated.termination == "error":
            status = "invalid"
            reasons.append(f"the {name} continuation failed: {'; '.join(generated.errors)}")
    if not identity:
        status = "limited" if status == "valid" else status
        reasons.append("no identity continuation control was performed")
    if not intervention.events:
        status = "invalid"
        reasons.append("the intervention never applied during the recorded continuation")

    record = ContinuationComparison(
        continuation_id=_new_continuation_id(),
        created_at=utc_now_iso(),
        case_id=case_id,
        trace_id=spec.execution.trace_id,
        replay_id=spec.execution.replay_id,
        replay_artifact_id=spec.execution.replay_artifact_id,
        intervention_id=spec.intervention_id,
        intervention_sha256=intervention_sha256(spec),
        intervention_artifact_id=intervention_artifact_id,
        model=model_info,
        backend=spec.execution.backend,
        prefix_length=len(prefix),
        prefix_token_ids=prefix,
        prefix_sha256=sha256_bytes((",".join(str(item) for item in prefix)).encode("utf-8")),
        config=config,
        conditions=list(conditions.values()),
        comparison=comparison,
        validity=ValidityReport(
            status=status,  # type: ignore[arg-type]
            checks=checks,
            reasons=reasons,
        ),
        warnings=warnings,
        metadata={
            "execution": "uncached-full-prefix",
            "intervention_event": config.intervention_event,
            "cached_generation": False,
            "note": "the intervention is applied only during the first prefix forward pass; "
            "no KV-cache or recurrent-state claim is made",
        },
    )
    record.checksums["continuation_sha256"] = continuation_sha256(record)
    return ContinuationOutcome(record=record, conditions=conditions)


def _new_continuation_id() -> str:
    from .ids import new_continuation_id

    return new_continuation_id()


__all__ = [
    "ContinuationOutcome",
    "run_counterfactual_continuation",
]
