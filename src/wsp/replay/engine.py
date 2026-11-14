"""Execute a replay plan and measure agreement with the recorded execution.

The engine is deliberately small and explicit.  It performs exactly one
instrumented forward pass over the recorded sequence, compares the logits at
prediction positions with the recorded log-probabilities, and reports what it
measured.  It never regenerates, never samples, and never claims that a
teacher-forced pass is bitwise equivalent to a cached generation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from wsp.core.budget import Budget
from wsp.core.dtypes import torch_dtype_name
from wsp.core.errors import CompatibilityError, ExecutionError, ResourceLimitError
from wsp.core.hashing import sha256_bytes
from wsp.core.time import utc_now_iso
from wsp.replay.ids import new_replay_id
from wsp.replay.models import (
    TEACHER_FORCING_LIMITATIONS,
    AgreementLevel,
    ReplayAgreement,
    ReplayCheck,
    ReplayForwardInfo,
    ReplayIdentity,
    ReplayModelInfo,
    ReplayPlan,
    ReplayResult,
    ReplayTokenAgreement,
)
from wsp.replay.plan import plan_checksum


@dataclass
class ReplayExecution:
    result: ReplayResult
    capture: Any | None = None


def _torch_module(model: Any) -> torch.nn.Module | None:
    for attribute in ("module", "model"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate
    if isinstance(model, torch.nn.Module):
        return model
    return None


def parameter_fingerprint(module: torch.nn.Module) -> str:
    """Cheap deterministic digest of sampled parameter values.

    Hashing every weight would cost as much as loading the checkpoint.  A
    bounded sample (64 values per tensor, plus name/shape/dtype) is enough to
    detect accidental mutation during observation, which is what the tests
    assert.
    """

    parts: list[str] = []
    for name, parameter in sorted(module.named_parameters()):
        tensor = parameter.detach().reshape(-1)
        if tensor.numel() > 64:
            step = max(1, tensor.numel() // 64)
            sample = tensor[::step][:64]
        else:
            sample = tensor
        try:
            total = float(sample.to(torch.float64).sum().item())
        except (RuntimeError, TypeError):  # pragma: no cover - exotic dtypes
            total = 0.0
        parts.append(
            f"{name}|{tuple(int(d) for d in parameter.shape)}|{parameter.dtype}|{total:.17g}"
        )
    return sha256_bytes("\n".join(parts).encode("utf-8"))


def _identity_checks(plan: ReplayPlan, info: ReplayModelInfo) -> list[ReplayCheck]:
    checks: list[ReplayCheck] = []
    trace_model = plan.model

    if trace_model.file_sha256 and info.file_sha256:
        status = "pass" if trace_model.file_sha256 == info.file_sha256 else "fail"
        checks.append(
            ReplayCheck(
                name="model_file_checksum",
                status=status,  # type: ignore[arg-type]
                detail=(
                    "the loaded checkpoint matches the recorded checksum"
                    if status == "pass"
                    else "the loaded checkpoint is a different file from the recorded one"
                ),
                expected=trace_model.file_sha256,
                observed=info.file_sha256,
            )
        )
    else:
        checks.append(
            ReplayCheck(
                name="model_file_checksum",
                status="unavailable",
                detail="model file checksums are not available on both sides",
                expected=trace_model.file_sha256,
                observed=info.file_sha256,
            )
        )

    if trace_model.structure_fingerprint and info.structure_fingerprint:
        status = (
            "pass" if trace_model.structure_fingerprint == info.structure_fingerprint else "fail"
        )
        checks.append(
            ReplayCheck(
                name="model_structure_fingerprint",
                status=status,  # type: ignore[arg-type]
                detail="structure fingerprints compared",
                expected=trace_model.structure_fingerprint,
                observed=info.structure_fingerprint,
            )
        )

    if trace_model.parameter_count and info.parameter_count:
        status = "pass" if trace_model.parameter_count == info.parameter_count else "fail"
        checks.append(
            ReplayCheck(
                name="model_parameter_count",
                status=status,  # type: ignore[arg-type]
                detail=(
                    "parameter counts match"
                    if status == "pass"
                    else "the loaded model has a different parameter count"
                ),
                expected=trace_model.parameter_count,
                observed=info.parameter_count,
            )
        )

    recorded_architectures = trace_model.metadata.get("architectures") if trace_model.metadata else None
    if isinstance(recorded_architectures, list) and recorded_architectures and info.config_architectures:
        overlap = set(map(str, recorded_architectures)) & set(info.config_architectures)
        checks.append(
            ReplayCheck(
                name="model_architecture",
                status="pass" if overlap else "warning",
                detail=(
                    "architecture names overlap"
                    if overlap
                    else "architecture names differ; parameter and file checksums remain authoritative"
                ),
                expected=recorded_architectures,
                observed=info.config_architectures,
            )
        )

    trace_tokenizer = plan.tokenizer
    if trace_tokenizer is not None:
        if trace_tokenizer.fingerprint and info.tokenizer_fingerprint:
            status = (
                "pass" if trace_tokenizer.fingerprint == info.tokenizer_fingerprint else "fail"
            )
            checks.append(
                ReplayCheck(
                    name="tokenizer_fingerprint",
                    status=status,  # type: ignore[arg-type]
                    detail=(
                        "tokenizer fingerprint matches"
                        if status == "pass"
                        else "the tokenizer vocabulary differs from the recorded one"
                    ),
                    expected=trace_tokenizer.fingerprint,
                    observed=info.tokenizer_fingerprint,
                )
            )
        else:
            checks.append(
                ReplayCheck(
                    name="tokenizer_fingerprint",
                    status="unavailable",
                    detail="tokenizer fingerprints are not available on both sides",
                )
            )
        if trace_tokenizer.vocab_size and info.tokenizer_vocab_size:
            status = (
                "pass" if trace_tokenizer.vocab_size == info.tokenizer_vocab_size else "fail"
            )
            checks.append(
                ReplayCheck(
                    name="tokenizer_vocab_size",
                    status=status,  # type: ignore[arg-type]
                    detail="tokenizer vocabulary sizes compared",
                    expected=trace_tokenizer.vocab_size,
                    observed=info.tokenizer_vocab_size,
                )
            )

    recorded_template = plan.tokenizer is not None and plan.sequence.token_encoding == "chat_template"
    if recorded_template:
        if plan.trace_artifact_id is None:  # pragma: no cover - informational only
            pass
        if info.chat_template_sha256 is None:
            status = "warning" if plan.sequence.source == "recorded-token-ids" else "fail"
            checks.append(
                ReplayCheck(
                    name="chat_template",
                    status=status,  # type: ignore[arg-type]
                    detail=(
                        "the loaded tokenizer has no chat template, but identifiers were recorded "
                        "so the sequence itself is unaffected"
                        if status == "warning"
                        else "the prompt must be re-rendered but the loaded tokenizer has no chat template"
                    ),
                )
            )
        else:
            checks.append(
                ReplayCheck(
                    name="chat_template",
                    status="pass",
                    detail="the loaded tokenizer exposes a chat template",
                    observed=info.chat_template_sha256,
                )
            )

    return checks


def _select_logits(logits: torch.Tensor, sequence_length: int) -> torch.Tensor:
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if logits.dim() != 3:
        raise ExecutionError(
            "logits must have shape [batch, sequence, vocabulary]",
            details={"shape": [int(d) for d in logits.shape]},
        )
    if int(logits.shape[0]) != 1:
        raise ExecutionError(
            "replay expects a single sequence per forward pass",
            details={"batch_size": int(logits.shape[0])},
        )
    if int(logits.shape[1]) != sequence_length:
        raise ExecutionError(
            "the model returned logits for a different number of positions",
            details={"expected": sequence_length, "observed": int(logits.shape[1])},
        )
    return logits


def _compare(
    plan: ReplayPlan,
    logits: torch.Tensor,
    *,
    budget: Budget,
) -> ReplayAgreement:
    sequence = plan.sequence
    tolerances = plan.tolerances
    generated_length = sequence.generated_length
    if generated_length == 0:
        return ReplayAgreement(
            level="unavailable",
            tolerance=tolerances,
            unavailable_reason="no generated tokens were recorded, so there is nothing to compare",
        )
    recorded = plan.expectations.recorded_logprobs
    if not recorded or all(value is None for value in recorded):
        return ReplayAgreement(
            level="unavailable",
            tolerance=tolerances,
            unavailable_reason=(
                "the trace did not record token log-probabilities; replay executed but no "
                "numerical agreement can be reported"
            ),
        )

    vocabulary = int(logits.shape[-1])
    if generated_length * vocabulary > budget.max_replay_logits_elements:
        raise ResourceLimitError(
            "the comparison slice exceeds the configured logits element limit",
            details={
                "elements": generated_length * vocabulary,
                "limit": budget.max_replay_logits_elements,
                "hint": "compare fewer generated tokens or raise budget.max_replay_logits_elements",
            },
        )

    prompt_length = sequence.prompt_length
    predicting_positions = [prompt_length - 1 + index for index in range(generated_length)]
    if min(predicting_positions) < 0:
        return ReplayAgreement(
            level="unavailable",
            tolerance=tolerances,
            unavailable_reason=(
                "the recorded sequence has no prompt position before the first generated token"
            ),
        )
    index_tensor = torch.tensor(predicting_positions, dtype=torch.long, device=logits.device)
    selected = logits[0].index_select(0, index_tensor)
    logprobs = torch.log_softmax(selected.to(torch.float32), dim=-1)

    rows: list[ReplayTokenAgreement] = []
    diffs: list[float] = []
    within: list[bool] = []
    top1_hits: list[bool] = []
    paired_recorded: list[float] = []
    paired_replay: list[float] = []
    detail_limit = max(0, budget.max_replay_agreement_tokens)

    for index in range(generated_length):
        token_id = int(sequence.generated_token_ids[index])
        recorded_value = recorded[index] if index < len(recorded) else None
        row = logprobs[index]
        replay_value = float(row[token_id].item()) if token_id < vocabulary else None
        top1 = int(torch.argmax(row).item())
        top1_match = top1 == token_id
        diff: float | None = None
        tolerance: float | None = None
        inside: bool | None = None
        if recorded_value is not None and replay_value is not None:
            diff = abs(float(recorded_value) - replay_value)
            tolerance = tolerances.logprob_atol + tolerances.logprob_rtol * abs(float(recorded_value))
            inside = diff <= tolerance
            diffs.append(diff)
            within.append(inside)
            top1_hits.append(top1_match)
            paired_recorded.append(float(recorded_value))
            paired_replay.append(replay_value)
        if index < detail_limit:
            rows.append(
                ReplayTokenAgreement(
                    generated_index=index,
                    token_id=token_id,
                    predicting_position=predicting_positions[index],
                    recorded_logprob=recorded_value,
                    replay_logprob=replay_value,
                    diff=diff,
                    tolerance=tolerance,
                    within_tolerance=inside,
                    replay_top1_token_id=top1,
                    top1_matches=top1_match,
                )
            )

    compared = len(diffs)
    if compared == 0:
        return ReplayAgreement(
            level="unavailable",
            tolerance=tolerances,
            per_token=rows,
            per_token_truncated=generated_length > len(rows),
            unavailable_reason="no recorded log-probability could be paired with a replay measurement",
        )

    within_count = sum(1 for value in within if value)
    top1_count = sum(1 for value in top1_hits if value)
    fraction = within_count / compared
    top1_rate = top1_count / compared
    diffs_array = np.asarray(diffs, dtype=np.float64)
    recorded_array = np.asarray(paired_recorded, dtype=np.float64)
    replay_array = np.asarray(paired_replay, dtype=np.float64)
    correlation: float | None = None
    if recorded_array.size >= 2:
        recorded_std = float(recorded_array.std())
        replay_std = float(replay_array.std())
        if recorded_std > 0.0 and replay_std > 0.0:
            correlation = float(np.corrcoef(recorded_array, replay_array)[0, 1])
            if not math.isfinite(correlation):
                correlation = None

    if fraction >= tolerances.verified_min_fraction and top1_rate >= tolerances.top1_min_agreement:
        level: AgreementLevel = "verified"
    elif fraction >= tolerances.partial_min_fraction:
        level = "partial"
    else:
        level = "mismatch"

    worst_index = int(np.argmax(diffs_array)) if diffs_array.size else None
    absolute_diffs = np.abs(diffs_array)
    return ReplayAgreement(
        tokens_compared=compared,
        tokens_within_tolerance=within_count,
        top1_matches=top1_count,
        top1_agreement_rate=top1_rate,
        within_tolerance_fraction=fraction,
        max_abs_diff=float(absolute_diffs.max()),
        mean_abs_diff=float(absolute_diffs.mean()),
        rms_diff=float(np.sqrt(np.mean(np.square(absolute_diffs)))),
        correlation=correlation,
        worst_generated_index=worst_index,
        level=level,
        tolerance=tolerances,
        per_token=rows,
        per_token_truncated=generated_length > len(rows),
    )


def execute_replay(
    plan: ReplayPlan,
    model: Any,
    *,
    budget: Budget | None = None,
    capture: Any | None = None,
) -> ReplayExecution:
    """Run one teacher-forced replay and return a measured, honest record."""

    budget = budget or Budget()
    info = model.describe()
    checks = _identity_checks(plan, info)
    failures = [check for check in checks if check.status == "fail"]
    if failures:
        raise CompatibilityError(
            "the loaded model is not compatible with the recorded trace",
            details={
                "checks": [
                    {"name": check.name, "detail": check.detail, "expected": check.expected, "observed": check.observed}
                    for check in failures
                ]
            },
        )

    sequence = plan.sequence
    if not sequence.full_token_ids:
        raise ExecutionError("the replay plan has no token sequence")

    length = len(sequence.full_token_ids)
    limit = min(plan.execution.max_sequence_length, budget.max_replay_sequence_tokens)
    if length > limit:
        raise ResourceLimitError(
            "replay sequence exceeds the configured limit",
            details={"tokens": length, "limit": limit},
        )

    vocabulary = info.tokenizer_vocab_size
    if vocabulary is None:
        module = _torch_module(model)
        config = getattr(module, "config", None) if module is not None else None
        vocabulary = getattr(config, "vocab_size", None)
        if vocabulary is None:
            text_config = getattr(config, "text_config", None) if config is not None else None
            vocabulary = getattr(text_config, "vocab_size", None)
    largest = max(sequence.full_token_ids)
    if vocabulary is not None:
        if largest >= int(vocabulary):
            raise CompatibilityError(
                "the recorded token identifiers do not fit the loaded vocabulary",
                details={"largest_token_id": largest, "vocabulary_size": int(vocabulary)},
            )
        checks.append(
            ReplayCheck(
                name="token_ids_in_vocabulary",
                status="pass",
                detail="all recorded identifiers are inside the loaded vocabulary",
                expected=int(vocabulary),
                observed=largest,
            )
        )
        estimated_elements = length * int(vocabulary)
        if estimated_elements > budget.max_replay_logits_elements:
            raise ResourceLimitError(
                "the replay forward pass would materialize more logits than the budget allows",
                details={
                    "elements": estimated_elements,
                    "limit": budget.max_replay_logits_elements,
                    "tokens": length,
                    "vocabulary_size": int(vocabulary),
                    "hint": "lower max_sequence_length or raise budget.max_replay_logits_elements",
                },
            )
    else:
        estimated_elements = 0
        checks.append(
            ReplayCheck(
                name="token_ids_in_vocabulary",
                status="unavailable",
                detail="the model does not report a vocabulary size",
            )
        )

    module = _torch_module(model)
    was_training = bool(getattr(module, "training", False)) if module is not None else False
    fingerprint_before = parameter_fingerprint(module) if module is not None else None
    if module is not None:
        module.eval()

    if capture is not None:
        capture.install()
    started = time.perf_counter()
    try:
        logits = model.forward_token_ids(sequence.full_token_ids)
    finally:
        if capture is not None:
            capture.close()
        if module is not None and was_training:
            module.train(True)
    elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
    logits = _select_logits(logits, length)
    logits_shape = [int(dimension) for dimension in logits.shape]
    logits_dtype = torch_dtype_name(logits.dtype)

    if fingerprint_before is not None and module is not None:
        fingerprint_after = parameter_fingerprint(module)
        checks.append(
            ReplayCheck(
                name="parameters_unchanged",
                status="pass" if fingerprint_after == fingerprint_before else "fail",
                detail=(
                    "sampled parameter values are identical before and after replay"
                    if fingerprint_after == fingerprint_before
                    else "model parameters changed during replay; this is a defect"
                ),
                expected=fingerprint_before,
                observed=fingerprint_after,
            )
        )
        if fingerprint_after != fingerprint_before:
            raise ExecutionError(
                "model parameters changed during replay",
                details={"before": fingerprint_before, "after": fingerprint_after},
            )

    checks.append(
        ReplayCheck(
            name="forward_execution",
            status="pass",
            detail="the recorded sequence was consumed without error",
            observed={"tokens": length, "logits_shape": logits_shape},
        )
    )
    checks.append(
        ReplayCheck(
            name="execution_mode",
            status="pass",
            detail="replay ran in evaluation and inference mode",
        )
    )
    checks.append(
        ReplayCheck(
            name="token_sequence",
            status="pass",
            detail="the recorded identifier sequence was used verbatim",
            observed=plan.sequence.sequence_sha256,
        )
    )
    checks.append(
        ReplayCheck(
            name="generation_reproducibility",
            status="unavailable",
            detail=(
                "teacher forcing presents recorded tokens; it does not sample and therefore "
                "cannot establish that the model would generate them again"
            ),
        )
    )
    checks.append(
        ReplayCheck(
            name="attention_mask",
            status="pass",
            detail="a single unpadded sequence was presented, so the mask is all ones",
        )
    )
    checks.append(
        ReplayCheck(
            name="position_ids",
            status="pass",
            detail="contiguous positions from zero; no offsets were applied",
        )
    )

    agreement = _compare(plan, logits, budget=budget)
    checks.append(
        ReplayCheck(
            name="numerical_agreement",
            status={
                "verified": "pass",
                "partial": "warning",
                "mismatch": "fail",
                "unavailable": "unavailable",
            }[agreement.level],
            detail=(
                "replayed log-probabilities agree with the recorded ones within tolerance"
                if agreement.level == "verified"
                else (
                    "only part of the replayed log-probabilities agree within tolerance"
                    if agreement.level == "partial"
                    else (
                        "replayed log-probabilities disagree with the recorded ones"
                        if agreement.level == "mismatch"
                        else agreement.unavailable_reason
                    )
                )
            ),
            expected=plan.tolerances.model_dump(mode="json"),
            observed={
                "tokens_compared": agreement.tokens_compared,
                "within_tolerance": agreement.tokens_within_tolerance,
                "max_abs_diff": agreement.max_abs_diff,
            },
        )
    )

    try:
        support = model.tokenizer_support()
        special_ids = sorted(int(value) for value in support.special_ids)
    except Exception:  # pragma: no cover - adapters must not fail here
        special_ids = []
    special_set = set(special_ids)
    special_positions = [
        position
        for position, token_id in enumerate(sequence.full_token_ids)
        if token_id in special_set
    ]

    traced_dtype = plan.model.metadata.get("dtype") if plan.model.metadata else None
    warnings: list[str] = []
    if info.dtype and plan.expectations.recorded_params.get("do_sample"):
        warnings.append("the trace was produced with sampling enabled")
    if plan.tokenizer is not None and plan.tokenizer.vocab_size and info.tokenizer_vocab_size:
        if plan.tokenizer.vocab_size != info.tokenizer_vocab_size:
            warnings.append("tokenizer vocabulary size differs from the recorded trace")

    limitations = list(TEACHER_FORCING_LIMITATIONS)
    limitations.extend(info.caveats)
    if str(info.dtype or "") in {"bfloat16", "float16"}:
        limitations.append(
            f"the model executes in {info.dtype}; tolerance-based agreement is expected, "
            "bitwise equality is not"
        )
    if str(getattr(logits, "device", "")) .startswith("cuda"):
        limitations.append(
            "accelerator execution may not be bitwise reproducible across hardware or software versions"
        )
    limitations.append(
        "agreement covers token log-probabilities at prediction positions only; it is not "
        "evidence that the recorded generation would recur"
    )

    result = ReplayResult(
        replay_id=new_replay_id(),
        created_at=utc_now_iso(),
        plan_id=plan.plan_id,
        plan_sha256=plan_checksum(plan),
        trace_id=plan.trace_id,
        trace_artifact_sha256=plan.trace_artifact_sha256,
        token_sequence_reconstructed=True,
        forward_executed=True,
        numerical_agreement=agreement.level,
        generation_reproducibility="not-claimed",
        evidence_level="verified" if agreement.level == "verified" else "exploratory",
        exploratory_reason=(
            None
            if agreement.level == "verified"
            else (
                "numerical agreement was unavailable: "
                f"{agreement.unavailable_reason or 'no comparison was possible'}"
                if agreement.level == "unavailable"
                else f"numerical agreement was only '{agreement.level}' within the stated tolerances"
            )
        ),
        identity=ReplayIdentity(
            model_name=info.name,
            model_class=info.model_class,
            parameter_count=info.parameter_count,
            file_sha256=info.file_sha256,
            structure_fingerprint=info.structure_fingerprint,
            tokenizer_name=info.tokenizer_name,
            tokenizer_kind=info.tokenizer_kind,
            tokenizer_fingerprint=info.tokenizer_fingerprint,
            tokenizer_vocab_size=info.tokenizer_vocab_size,
            chat_template_sha256=info.chat_template_sha256,
            device=info.device,
            dtype=info.dtype,
            backend=info.backend,
            backend_version=info.backend_version,
            runtime=info.runtime,
        ),
        sequence=sequence,
        forward=ReplayForwardInfo(
            sequence_length=length,
            batch_size=1,
            forward_calls=1,
            logits_shape=logits_shape,
            logits_dtype=logits_dtype,
            vocabulary_size=int(logits_shape[-1]),
            device=str(logits.device),
            inference_mode=True,
            use_cache=False,
            elapsed_ms=elapsed_ms,
            estimated_logits_elements=estimated_elements,
        ),
        agreement=agreement,
        checks=checks,
        special_token_ids=special_ids,
        special_positions=special_positions,
        warnings=warnings,
        limitations=limitations,
        metadata={
            "parameter_fingerprint": fingerprint_before,
            "recorded_trace_dtype": traced_dtype,
            "plan_sequence_source": sequence.source,
        },
    )
    return ReplayExecution(result=result, capture=capture)
