"""Paired behavioral comparisons with a declared expectation."""

from __future__ import annotations

from typing import Any

from wsp.behavior.backends import GenerationBackend
from wsp.behavior.detectors.reconsideration import ReconsiderationDetector
from wsp.behavior.detectors.repetition import RepetitionDetector
from wsp.behavior.evaluators import EvaluatorRegistry, default_evaluators
from wsp.behavior.ids import new_contrast_result_id, new_evaluation_id
from wsp.behavior.models import (
    ContrastResult,
    ContrastTest,
    EvaluationOutcome,
    EvaluationStatus,
    EvaluatorSpec,
    GenerationRequest,
    GenerationTrace,
    VariantOutcome,
)
from wsp.behavior.trace import TraceStore, sha256_text
from wsp.core.budget import Budget
from wsp.core.errors import InputError, WspError
from wsp.core.time import utc_now_iso

_PREVIEW_CHARS = 4000


def _preview(text: str | None) -> tuple[str | None, bool]:
    if text is None:
        return None, False
    if len(text) <= _PREVIEW_CHARS:
        return text, False
    return text[:_PREVIEW_CHARS], True


def _metric_outcome(
    *,
    metric: str,
    status: EvaluationStatus,
    baseline: float,
    variant: float,
    direction: str,
    tolerance: float,
    trace_id: str | None,
    explanation: str,
) -> EvaluationOutcome:
    return EvaluationOutcome(
        evaluation_id=new_evaluation_id(),
        created_at=utc_now_iso(),
        evaluator_id="core.metric_change",
        evaluator_version="1.0.0",
        status=status,
        value={
            "metric": metric,
            "baseline": baseline,
            "variant": variant,
            "delta": variant - baseline,
            "direction": direction,
            "tolerance": tolerance,
        },
        explanation=explanation,
        trace_id=trace_id,
    )


def _metric_value(
    trace: GenerationTrace,
    name: str,
    budget: Budget,
) -> tuple[float | None, dict[str, Any]]:
    if name == "generated_tokens":
        if trace.tokens:
            return float(len(trace.tokens)), {"source": "token_records"}
        if trace.termination.generated_tokens is not None:
            return float(trace.termination.generated_tokens), {"source": "termination"}
        return None, {"reason": "token count unavailable"}
    if name == "repetition_score":
        detector = RepetitionDetector()
        run = detector.scan(trace, detector.validate_parameters({}), budget)
        if run.status != "ok":
            return None, {"reason": run.skip_reason or "detector did not run"}
        return float(run.measurements.get("max_occurrences", 0)), {
            "source": "repetition_detector",
            "units": run.measurements.get("unit"),
        }
    if name == "reconsideration_markers":
        detector = ReconsiderationDetector()
        run = detector.scan(trace, detector.validate_parameters({}), budget)
        if run.status != "ok":
            return None, {"reason": run.skip_reason or "detector did not run"}
        return float(run.measurements.get("total_markers", 0)), {"source": "reconsideration_detector"}
    if name == "termination_is_length_limit":
        return (1.0 if trace.termination.reason == "length_limit" else 0.0), {
            "source": "termination",
            "reason": trace.termination.reason,
        }
    raise InputError(f"unknown contrast metric {name!r}")


def _evaluate_variant(
    *,
    test: ContrastTest,
    baseline: GenerationTrace,
    variant: GenerationTrace,
    registry: EvaluatorRegistry,
    budget: Budget,
) -> tuple[EvaluationStatus, EvaluationOutcome | None, str | None]:
    expectation = test.expectation
    kind = expectation.kind
    if kind == "exact_match":
        if baseline.text is None:
            return "abstain", None, "baseline has no output text"
        spec = EvaluatorSpec(
            evaluator_id="core.exact_match",
            parameters={
                "expected": baseline.text,
                "normalization": expectation.normalization,
            },
        )
        outcome = registry.evaluate(spec, text=variant.text, trace=variant)
        return outcome.status, outcome, outcome.explanation
    if kind == "reference_agreement":
        if expectation.evaluator is not None:
            spec = expectation.evaluator
        else:
            spec = EvaluatorSpec(
                evaluator_id="core.reference_answer",
                parameters={
                    "answers": expectation.reference_answers,
                    "normalization": expectation.normalization,
                },
            )
        baseline_outcome = registry.evaluate(spec, text=baseline.text, trace=baseline)
        variant_outcome = registry.evaluate(spec, text=variant.text, trace=variant)
        if baseline_outcome.status != "pass":
            return (
                "inconclusive",
                variant_outcome,
                "the baseline did not establish the expected property, so agreement cannot be assessed",
            )
        return variant_outcome.status, variant_outcome, variant_outcome.explanation
    if kind == "format":
        assert expectation.contract is not None
        spec = EvaluatorSpec(
            evaluator_id="core.json_contract",
            parameters={"contract": expectation.contract.model_dump(mode="json")},
        )
        baseline_outcome = registry.evaluate(spec, text=baseline.text, trace=baseline)
        variant_outcome = registry.evaluate(spec, text=variant.text, trace=variant)
        if baseline_outcome.status != "pass":
            return (
                "inconclusive",
                variant_outcome,
                "the baseline did not satisfy the required format, so preservation cannot be assessed",
            )
        return variant_outcome.status, variant_outcome, variant_outcome.explanation
    if kind == "property_preserved":
        assert expectation.evaluator is not None
        baseline_outcome = registry.evaluate(expectation.evaluator, text=baseline.text, trace=baseline)
        variant_outcome = registry.evaluate(expectation.evaluator, text=variant.text, trace=variant)
        if baseline_outcome.status != "pass":
            return (
                "inconclusive",
                variant_outcome,
                "the baseline did not satisfy the declared property, so preservation cannot be assessed",
            )
        return variant_outcome.status, variant_outcome, variant_outcome.explanation
    if kind == "metric_change":
        assert expectation.metric is not None
        metric = expectation.metric
        baseline_value, baseline_detail = _metric_value(baseline, metric.name, budget)
        variant_value, variant_detail = _metric_value(variant, metric.name, budget)
        if baseline_value is None or variant_value is None:
            outcome = _metric_outcome(
                metric=metric.name,
                status="abstain",
                baseline=baseline_value or 0.0,
                variant=variant_value or 0.0,
                direction=metric.direction,
                tolerance=metric.tolerance,
                trace_id=variant.trace_id,
                explanation="the metric could not be measured for both traces",
            )
            outcome.metadata = {"baseline": baseline_detail, "variant": variant_detail}
            return "abstain", outcome, outcome.explanation
        delta = variant_value - baseline_value
        if metric.direction == "unchanged":
            passed = abs(delta) <= metric.tolerance
        elif metric.direction == "increase":
            passed = delta > metric.tolerance
        else:
            passed = delta < -metric.tolerance
        outcome = _metric_outcome(
            metric=metric.name,
            status="pass" if passed else "fail",
            baseline=baseline_value,
            variant=variant_value,
            direction=metric.direction,
            tolerance=metric.tolerance,
            trace_id=variant.trace_id,
            explanation=(
                f"{metric.name} changed from {baseline_value:g} to {variant_value:g} "
                f"(delta {delta:+g}); the declared relationship was {metric.direction!r} "
                f"with tolerance {metric.tolerance:g}."
            ),
        )
        outcome.metadata = {"baseline": baseline_detail, "variant": variant_detail}
        return outcome.status, outcome, outcome.explanation
    raise InputError(f"unknown contrast expectation {kind!r}")


def run_contrast(
    test: ContrastTest,
    *,
    backend: GenerationBackend,
    trace_store: TraceStore,
    budget: Budget,
    evaluators: EvaluatorRegistry | None = None,
) -> ContrastResult:
    registry = evaluators or default_evaluators()
    if len(test.variants) > budget.max_contrast_variants:
        raise InputError(
            "contrast has more variants than the configured limit",
            details={"variants": len(test.variants), "limit": budget.max_contrast_variants},
        )
    warnings: list[str] = []
    baseline_trace = backend.generate(
        GenerationRequest(
            input_text=test.baseline.input_text,
            system=test.baseline.system,
            params=test.params,
            seed=test.seed,
        ),
        budget=budget,
    )
    baseline_record = trace_store.save(baseline_trace)
    warnings.extend(baseline_trace.warnings)
    baseline_preview, baseline_truncated = _preview(baseline_trace.text)
    baseline_outcome = VariantOutcome(
        name=test.baseline.name,
        role="baseline",
        input_text=test.baseline.input_text,
        trace_id=baseline_trace.trace_id,
        output_text=baseline_preview,
        output_sha256=sha256_text(baseline_trace.text),
        status="observed",
        explanation="baseline reference observation; not judged by itself",
        observed={
            "output_truncated": baseline_truncated,
            "output_chars": len(baseline_trace.text or ""),
            "termination": baseline_trace.termination.reason,
            "artifact_sha256": baseline_record.artifact.sha256,
        },
    )

    outcomes: list[VariantOutcome] = []
    for variant in test.variants:
        try:
            trace = backend.generate(
                GenerationRequest(
                    input_text=variant.input_text,
                    system=variant.system,
                    params=test.params,
                    seed=test.seed,
                ),
                budget=budget,
            )
        except WspError as exc:
            outcomes.append(
                VariantOutcome(
                    name=variant.name,
                    role="variation",
                    input_text=variant.input_text,
                    status="error",
                    explanation=f"generation failed: {exc}",
                    observed={"error_code": exc.code, "error_details": exc.details},
                )
            )
            warnings.append(f"variant {variant.name!r} failed: {exc}")
            continue
        except Exception as exc:
            outcomes.append(
                VariantOutcome(
                    name=variant.name,
                    role="variation",
                    input_text=variant.input_text,
                    status="error",
                    explanation=f"generation failed: {type(exc).__name__}: {exc}",
                )
            )
            warnings.append(f"variant {variant.name!r} failed: {exc}")
            continue
        record = trace_store.save(trace)
        warnings.extend(trace.warnings)
        status, evaluation, explanation = _evaluate_variant(
            test=test,
            baseline=baseline_trace,
            variant=trace,
            registry=registry,
            budget=budget,
        )
        preview, truncated = _preview(trace.text)
        outcomes.append(
            VariantOutcome(
                name=variant.name,
                role="variation",
                input_text=variant.input_text,
                trace_id=trace.trace_id,
                output_text=preview,
                output_sha256=sha256_text(trace.text),
                status=status,  # type: ignore[arg-type]
                explanation=explanation,
                observed={
                    "output_truncated": truncated,
                    "output_chars": len(trace.text or ""),
                    "termination": trace.termination.reason,
                    "artifact_sha256": record.artifact.sha256,
                },
                evaluation=evaluation,
            )
        )

    statuses = [outcome.status for outcome in outcomes]
    if "fail" in statuses:
        overall: EvaluationStatus = "fail"
    elif "error" in statuses:
        overall = "error"
    elif any(status in {"inconclusive", "abstain"} for status in statuses):
        overall = "inconclusive"
    elif statuses and all(status == "pass" for status in statuses):
        overall = "pass"
    else:
        overall = "inconclusive"

    passed = sum(1 for status in statuses if status == "pass")
    failed = sum(1 for status in statuses if status == "fail")
    if overall == "pass":
        summary = (
            f"All {len(outcomes)} variation(s) preserved the declared invariant "
            f"({test.expectation.kind}): {test.expectation.description}"
        )
    elif overall == "fail":
        summary = (
            f"{failed} of {len(outcomes)} variation(s) violated the declared invariant "
            f"({test.expectation.kind}): {test.expectation.description}"
        )
    elif overall == "error":
        summary = "At least one variation could not be executed; outcomes are incomplete."
    else:
        summary = (
            f"{passed} of {len(outcomes)} variation(s) satisfied the declared invariant, but "
            "the result is inconclusive (abstention, uncertainty, or a baseline that did not "
            "establish the property)."
        )

    return ContrastResult(
        result_id=new_contrast_result_id(),
        test=test,
        created_at=utc_now_iso(),
        baseline=baseline_outcome,
        outcomes=outcomes,
        overall_status=overall,
        summary=summary,
        trace_ids=[baseline_trace.trace_id]
        + [outcome.trace_id for outcome in outcomes if outcome.trace_id],
        warnings=list(dict.fromkeys(warnings)),
    )
