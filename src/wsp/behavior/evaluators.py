"""Evaluator abstraction and deterministic built-in evaluators.

An evaluation is a judgement, not an observation.  Evaluators may pass, fail,
be inconclusive, abstain, or report an error.  The distinction matters: an
evaluator that cannot judge a case must not be reported as a failure.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from wsp.behavior.ids import new_evaluation_id
from wsp.behavior.models import (
    EvaluationOutcome,
    EvaluationStatus,
    EvaluatorSpec,
    GenerationTrace,
    JsonContract,
)
from wsp.core.errors import InputError, NotFoundError
from wsp.core.time import utc_now_iso

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def normalize_text(text: str, mode: str) -> str:
    if mode == "none":
        return text
    collapsed = " ".join(text.split())
    return collapsed.casefold() if mode == "casefold_whitespace" else collapsed


def _json_payload(text: str, contract: JsonContract) -> tuple[str, bool]:
    if contract.allow_code_fence:
        match = _FENCE.search(text)
        if match:
            return match.group(1).strip(), True
    return text.strip(), False


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _TYPES[expected])


class BaseEvaluator:
    evaluator_id: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = ""
    parameters_model: ClassVar[type[BaseModel]]

    def default_parameters(self) -> dict[str, Any]:
        return self.parameters_model().model_dump(mode="json")

    def validate_parameters(self, payload: dict[str, Any] | None) -> BaseModel:
        try:
            return self.parameters_model.model_validate(payload or {})
        except ValidationError as exc:
            raise InputError(
                f"invalid parameters for evaluator {self.evaluator_id}",
                details={
                    "errors": [
                        {"loc": list(item.get("loc", [])), "msg": item.get("msg")}
                        for item in exc.errors(include_url=False)
                    ]
                },
            ) from exc

    def _outcome(
        self,
        *,
        status: EvaluationStatus,
        value: Any = None,
        explanation: str | None = None,
        confidence: float | None = None,
        reviewer: str | None = None,
        trace_id: str | None = None,
        references: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationOutcome:
        return EvaluationOutcome(
            evaluation_id=new_evaluation_id(),
            created_at=utc_now_iso(),
            evaluator_id=self.evaluator_id,
            evaluator_version=self.version,
            status=status,
            value=value,
            explanation=explanation,
            confidence=confidence,
            reviewer=reviewer,
            trace_id=trace_id,
            references=list(references or []),
            metadata=metadata or {},
        )

    def evaluate(
        self,
        *,
        text: str | None,
        trace: GenerationTrace | None,
        parameters: BaseModel,
    ) -> EvaluationOutcome:  # pragma: no cover - interface
        raise NotImplementedError


class ExactMatchParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: str
    normalization: str = Field(default="whitespace", pattern=r"^(none|whitespace|casefold_whitespace)$")


class ExactMatchEvaluator(BaseEvaluator):
    evaluator_id: ClassVar[str] = "core.exact_match"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = "Compares the output text to one expected string after optional normalization."
    parameters_model: ClassVar[type[BaseModel]] = ExactMatchParameters

    def evaluate(self, *, text, trace, parameters):
        assert isinstance(parameters, ExactMatchParameters)
        if text is None:
            return self._outcome(
                status="abstain",
                explanation="No output text was available to compare.",
                trace_id=trace.trace_id if trace else None,
            )
        observed = normalize_text(text, parameters.normalization)
        expected = normalize_text(parameters.expected, parameters.normalization)
        matched = observed == expected
        return self._outcome(
            status="pass" if matched else "fail",
            value={"matched": matched, "expected": parameters.expected},
            explanation=(
                "The output matches the expected text after normalization."
                if matched
                else "The output differs from the expected text after normalization."
            ),
            trace_id=trace.trace_id if trace else None,
        )


class ReferenceAnswerParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[str] = Field(min_length=1)
    normalization: str = Field(default="whitespace", pattern=r"^(none|whitespace|casefold_whitespace)$")

    @model_validator(mode="after")
    def _answers(self) -> "ReferenceAnswerParameters":
        self.answers = [answer for answer in self.answers if answer.strip()]
        if not self.answers:
            raise ValueError("provide at least one non-empty reference answer")
        return self


class ReferenceAnswerEvaluator(BaseEvaluator):
    evaluator_id: ClassVar[str] = "core.reference_answer"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = (
        "Checks whether the output contains one of the supplied reference answers after normalization. "
        "This is string matching, not semantic entailment."
    )
    parameters_model: ClassVar[type[BaseModel]] = ReferenceAnswerParameters

    def evaluate(self, *, text, trace, parameters):
        assert isinstance(parameters, ReferenceAnswerParameters)
        if text is None:
            return self._outcome(
                status="abstain",
                explanation="No output text was available to evaluate.",
                trace_id=trace.trace_id if trace else None,
            )
        observed = normalize_text(text, parameters.normalization)
        matches = [
            answer
            for answer in parameters.answers
            if normalize_text(answer, parameters.normalization) in observed
        ]
        return self._outcome(
            status="pass" if matches else "fail",
            value={"matched_answers": matches, "answers": parameters.answers},
            explanation=(
                "The output contains a reference answer."
                if matches
                else "The output does not contain any supplied reference answer."
            ),
            confidence=None,
            trace_id=trace.trace_id if trace else None,
        )


class JsonContractParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: JsonContract = Field(default_factory=JsonContract)


class JsonContractEvaluator(BaseEvaluator):
    evaluator_id: ClassVar[str] = "core.json_contract"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = "Validates the output against the small JSON contract supported by the workbench."
    parameters_model: ClassVar[type[BaseModel]] = JsonContractParameters

    def evaluate(self, *, text, trace, parameters):
        assert isinstance(parameters, JsonContractParameters)
        if text is None:
            return self._outcome(
                status="abstain",
                explanation="No output text was available to validate.",
                trace_id=trace.trace_id if trace else None,
            )
        payload, used_fence = _json_payload(text, parameters.contract)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            return self._outcome(
                status="fail",
                value={"error": str(exc), "used_code_fence": used_fence},
                explanation="The output is not valid JSON.",
                trace_id=trace.trace_id if trace else None,
            )
        errors: list[dict[str, Any]] = []
        contract = parameters.contract
        if not isinstance(parsed, dict):
            errors.append({"kind": "wrong_root_type", "observed": type(parsed).__name__})
        else:
            for key in contract.required_keys:
                if key not in parsed:
                    errors.append({"kind": "missing_required_key", "key": key})
            if not contract.allow_extra:
                for key in parsed:
                    if key not in contract.properties and key not in contract.required_keys:
                        errors.append({"kind": "unexpected_key", "key": key})
            for key, expected in contract.properties.items():
                if key in parsed and not _type_ok(parsed[key], expected):
                    errors.append(
                        {
                            "kind": "type_mismatch",
                            "key": key,
                            "expected": expected,
                            "observed": type(parsed[key]).__name__,
                        }
                    )
        return self._outcome(
            status="pass" if not errors else "fail",
            value={"errors": errors, "used_code_fence": used_fence},
            explanation=(
                "The output satisfies the JSON contract."
                if not errors
                else f"The output violates {len(errors)} contract condition(s)."
            ),
            trace_id=trace.trace_id if trace else None,
        )


class HumanJudgmentParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgment: EvaluationStatus
    explanation: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reviewer: str | None = None


class HumanJudgmentEvaluator(BaseEvaluator):
    evaluator_id: ClassVar[str] = "core.human_judgment"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = (
        "Records a judgment supplied by a human reviewer, including uncertainty and abstention."
    )
    parameters_model: ClassVar[type[BaseModel]] = HumanJudgmentParameters

    def evaluate(self, *, text, trace, parameters):
        assert isinstance(parameters, HumanJudgmentParameters)
        metadata: dict[str, Any] = {}
        if parameters.reviewer is None:
            metadata["reviewer_provenance"] = "not supplied"
        return self._outcome(
            status=parameters.judgment,
            value=None,
            explanation=parameters.explanation,
            confidence=parameters.confidence,
            reviewer=parameters.reviewer,
            trace_id=trace.trace_id if trace else None,
            metadata=metadata,
        )


class EvaluatorRegistry:
    def __init__(self) -> None:
        self._evaluators: dict[str, BaseEvaluator] = {}

    def register(self, evaluator: BaseEvaluator, *, replace: bool = False) -> None:
        if not evaluator.evaluator_id or "/" in evaluator.evaluator_id:
            raise InputError("evaluator ids must be simple identifiers")
        if evaluator.evaluator_id in self._evaluators and not replace:
            raise InputError(f"evaluator {evaluator.evaluator_id!r} is already registered")
        self._evaluators[evaluator.evaluator_id] = evaluator

    def get(self, evaluator_id: str) -> BaseEvaluator:
        evaluator = self._evaluators.get(evaluator_id)
        if evaluator is None:
            raise NotFoundError(
                f"unknown evaluator {evaluator_id!r}",
                details={"available": sorted(self._evaluators)},
            )
        return evaluator

    def list(self) -> list[BaseEvaluator]:
        return [self._evaluators[key] for key in sorted(self._evaluators)]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "evaluator_id": evaluator.evaluator_id,
                "version": evaluator.version,
                "description": evaluator.description,
                "parameters_schema": evaluator.parameters_model.model_json_schema(),
                "default_parameters": evaluator.default_parameters(),
            }
            for evaluator in self.list()
        ]

    def evaluate(
        self,
        spec: EvaluatorSpec,
        *,
        text: str | None,
        trace: GenerationTrace | None = None,
    ) -> EvaluationOutcome:
        evaluator = self.get(spec.evaluator_id)
        parameters = evaluator.validate_parameters(spec.parameters)
        try:
            return evaluator.evaluate(text=text, trace=trace, parameters=parameters)
        except Exception as exc:
            return evaluator._outcome(
                status="error",
                explanation=f"evaluator failed: {type(exc).__name__}: {exc}",
                trace_id=trace.trace_id if trace else None,
            )


_default_registry: EvaluatorRegistry | None = None


def default_evaluators() -> EvaluatorRegistry:
    global _default_registry
    if _default_registry is None:
        registry = EvaluatorRegistry()
        for evaluator in (
            ExactMatchEvaluator(),
            ReferenceAnswerEvaluator(),
            JsonContractEvaluator(),
            HumanJudgmentEvaluator(),
        ):
            registry.register(evaluator)
        _default_registry = registry
    return _default_registry
