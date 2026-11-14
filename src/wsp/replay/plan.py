"""Plan a teacher-forced replay of one recorded token sequence."""

from __future__ import annotations

from typing import Any

from wsp.behavior.models import GenerationTrace
from wsp.core.budget import Budget
from wsp.core.errors import CompatibilityError, InputError, ResourceLimitError
from wsp.core.hashing import sha256_bytes
from wsp.core.json import canonical_bytes
from wsp.core.time import utc_now_iso
from wsp.replay.ids import new_plan_id
from wsp.replay.models import (
    TEACHER_FORCING_LIMITATIONS,
    ReplayCheck,
    ReplayExecutionConfig,
    ReplayExpectation,
    ReplayPlan,
    ReplayTokenSequence,
    ReplayTolerances,
)


def sequence_checksum(input_token_ids: list[int], generated_token_ids: list[int]) -> str:
    return sha256_bytes(
        canonical_bytes(
            {"input_token_ids": input_token_ids, "generated_token_ids": generated_token_ids}
        )
    )


def plan_checksum(plan: ReplayPlan) -> str:
    return sha256_bytes(canonical_bytes(plan.model_dump(mode="json")))


def _check(name: str, status: str, detail: str, expected: Any = None, observed: Any = None) -> ReplayCheck:
    return ReplayCheck(
        name=name,
        status=status,  # type: ignore[arg-type]
        detail=detail,
        expected=expected,
        observed=observed,
    )


def _recorded_generated_ids(trace: GenerationTrace) -> list[int] | None:
    if not trace.tokens:
        return []
    if any(token.token_id is None for token in trace.tokens):
        return None
    return [int(token.token_id) for token in trace.tokens if token.token_id is not None]


def prepare_plan(
    trace: GenerationTrace,
    *,
    trace_artifact_id: str | None = None,
    trace_artifact_sha256: str | None = None,
    execution: ReplayExecutionConfig | None = None,
    tolerances: ReplayTolerances | None = None,
    budget: Budget | None = None,
    encoder: Any | None = None,
    notes: list[str] | None = None,
) -> ReplayPlan:
    """Resolve a trace into an explicit, bounded replay plan.

    Token identifiers are the only representation replay trusts.  When a trace
    recorded them they are used verbatim.  When they are missing, identifiers
    are re-derived from stored text *only* if the caller supplies the tokenizer
    that produced the trace, and the plan says so.  Guessing is never an
    option: a plan without an exact sequence raises.
    """

    budget = budget or Budget()
    execution = execution or ReplayExecutionConfig()
    tolerances = tolerances or ReplayTolerances()
    warnings: list[str] = []
    checks: list[ReplayCheck] = []

    generated_ids = _recorded_generated_ids(trace)
    input_ids: list[int] | None = None
    source = "recorded-token-ids"
    sequence_warnings: list[str] = []

    if trace.input.token_ids:
        input_ids = [int(value) for value in trace.input.token_ids]
        checks.append(
            _check(
                "input_token_ids",
                "pass",
                "the trace recorded the exact input identifiers",
                observed=len(input_ids),
            )
        )
        if trace.input.tokens_sha256:
            actual = sha256_bytes(canonical_bytes(input_ids))
            if actual == trace.input.tokens_sha256:
                checks.append(_check("input_tokens_checksum", "pass", "input token checksum matches"))
            else:
                checks.append(
                    _check(
                        "input_tokens_checksum",
                        "fail",
                        "recorded input token identifiers do not match their checksum",
                        expected=trace.input.tokens_sha256,
                        observed=actual,
                    )
                )
                raise CompatibilityError(
                    "recorded input token identifiers do not match their checksum",
                    details={"trace_id": trace.trace_id},
                )
        else:
            checks.append(
                _check(
                    "input_tokens_checksum",
                    "warning",
                    "the trace did not record a checksum for its input token identifiers",
                )
            )
    else:
        encoding = trace.input.token_encoding or "unknown"
        if trace.input.text is None:
            raise InputError(
                "the trace has no input token identifiers and no stored input text",
                details={
                    "trace_id": trace.trace_id,
                    "hint": "record traces with token identifiers, or unredacted text plus a tokenizer",
                },
            )
        if encoding == "chat_template":
            raise InputError(
                "the prompt was rendered with a chat template, so its token identifiers "
                "cannot be re-derived from the stored user text alone",
                details={
                    "trace_id": trace.trace_id,
                    "hint": "re-record the trace with token identifiers; a chat template is not reconstructable",
                },
            )
        if encoder is None:
            raise InputError(
                "the trace did not record input token identifiers; supply the tokenizer "
                "used for the trace to re-derive them, or replay cannot proceed",
                details={"trace_id": trace.trace_id},
            )
        add_special = trace.input.add_special_tokens
        try:
            input_ids = [int(value) for value in encoder(trace.input.text, bool(add_special))]
        except TypeError:
            input_ids = [int(value) for value in encoder(trace.input.text)]
        source = "re-derived-from-text"
        sequence_warnings.append(
            "input identifiers were re-derived from stored text with the supplied tokenizer; "
            "this is a reconstruction, not a recorded sequence"
        )
        checks.append(
            _check(
                "input_token_ids",
                "warning",
                "input identifiers were re-derived from stored text",
                observed=len(input_ids),
            )
        )

    if generated_ids is None:
        raise InputError(
            "the trace does not record token identifiers for its generated tokens",
            details={
                "trace_id": trace.trace_id,
                "hint": "log-probability and identifier logging must be enabled when recording",
            },
        )
    checks.append(
        _check(
            "generated_token_ids",
            "pass" if generated_ids else "warning",
            "the trace recorded identifier-level generation output",
            observed=len(generated_ids),
        )
    )

    full_ids = list(input_ids) + list(generated_ids)
    if len(full_ids) > budget.max_replay_sequence_tokens:
        raise ResourceLimitError(
            "replay sequence exceeds the configured token limit",
            details={
                "tokens": len(full_ids),
                "limit": budget.max_replay_sequence_tokens,
                "hint": "raise budget.max_replay_sequence_tokens or replay a shorter trace",
            },
        )
    if len(full_ids) > execution.max_sequence_length:
        raise ResourceLimitError(
            "replay sequence exceeds the plan's maximum sequence length",
            details={
                "tokens": len(full_ids),
                "limit": execution.max_sequence_length,
                "hint": "raise max_sequence_length in the replay plan",
            },
        )
    checks.append(
        _check(
            "sequence_length",
            "pass",
            "the full sequence fits the configured limit",
            expected={"max_sequence_length": execution.max_sequence_length},
            observed=len(full_ids),
        )
    )

    recorded_logprobs = [token.logprob for token in trace.tokens]
    if any(value is not None for value in recorded_logprobs):
        checks.append(
            _check(
                "recorded_logprobs",
                "pass",
                "the trace recorded token log-probabilities for numerical comparison",
                observed=sum(1 for value in recorded_logprobs if value is not None),
            )
        )
    else:
        checks.append(
            _check(
                "recorded_logprobs",
                "unavailable",
                "no token log-probabilities were recorded, so numerical agreement cannot be assessed",
            )
        )

    if trace.tokenizer is None:
        checks.append(
            _check(
                "tokenizer_identity",
                "warning",
                "the trace did not record tokenizer identity; identifier agreement cannot be checked",
            )
        )
    else:
        checks.append(
            _check(
                "tokenizer_identity",
                "pass",
                "the trace recorded tokenizer identity",
                observed=trace.tokenizer.fingerprint,
            )
        )
    if trace.model.file_sha256 is None and trace.model.structure_fingerprint is None:
        checks.append(
            _check(
                "model_identity",
                "warning",
                "the trace did not record a model file checksum or structure fingerprint",
            )
        )
    else:
        checks.append(
            _check(
                "model_identity",
                "pass",
                "the trace recorded a model identity anchor",
                observed=trace.model.file_sha256 or trace.model.structure_fingerprint,
            )
        )

    checks.append(
        _check(
            "special_tokens",
            "unavailable",
            "special-token positions are resolved against the live tokenizer at execution time",
        )
    )
    checks.append(
        _check(
            "attention_mask",
            "pass",
            "replay presents one unpadded sequence, so the attention mask is all ones",
        )
    )
    checks.append(
        _check(
            "position_ids",
            "pass",
            "positions are contiguous and implicit, starting at zero",
        )
    )
    checks.append(
        _check(
            "generation_reproducibility",
            "unavailable",
            "teacher forcing does not attempt, and cannot establish, generation reproducibility",
        )
    )

    if trace.params.sampling_enabled():
        warnings.append(
            "the recorded generation used sampling; replay presents its tokens but does not "
            "reproduce the sampling process"
        )

    if trace.checksums.get("output_sha256") is None:
        checks.append(
            _check(
                "recorded_output_checksum",
                "unavailable",
                "the trace did not record an output text checksum",
            )
        )

    sequence = ReplayTokenSequence(
        input_token_ids=list(input_ids),
        generated_token_ids=list(generated_ids),
        full_token_ids=full_ids,
        prompt_length=len(input_ids),
        generated_length=len(generated_ids),
        full_length=len(full_ids),
        sequence_sha256=sequence_checksum(list(input_ids), list(generated_ids)),
        token_encoding=trace.input.token_encoding or "unknown",
        add_special_tokens=trace.input.add_special_tokens,
        source=source,  # type: ignore[arg-type]
        warnings=sequence_warnings,
    )
    expectations = ReplayExpectation(
        recorded_generated_token_ids=list(generated_ids),
        recorded_logprobs=[value for value in recorded_logprobs],
        recorded_output_sha256=trace.checksums.get("output_sha256"),
        recorded_text_chars=len(trace.text) if trace.text is not None else None,
        recorded_termination_reason=trace.termination.reason,
        recorded_params=trace.params.model_dump(mode="json", exclude_none=True),
        recorded_seed=trace.seed,
    )

    limitations = list(TEACHER_FORCING_LIMITATIONS)
    if trace.model.file_sha256 is None:
        limitations.append(
            "The trace did not record a checkpoint checksum, so model identity is only as "
            "strong as its name and parameter count."
        )

    return ReplayPlan(
        plan_id=new_plan_id(),
        created_at=utc_now_iso(),
        trace_id=trace.trace_id,
        trace_artifact_id=trace_artifact_id,
        trace_artifact_sha256=trace_artifact_sha256,
        model=trace.model,
        tokenizer=trace.tokenizer,
        sequence=sequence,
        execution=execution,
        tolerances=tolerances,
        expectations=expectations,
        checks=checks,
        warnings=warnings,
        limitations=limitations,
        notes=list(notes or []),
    )
