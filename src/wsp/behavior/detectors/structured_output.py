"""Structured output validation for JSON contracts."""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from wsp.behavior.detectors.base import BaseDetector
from wsp.behavior.models import DetectorRun, EvidenceRef, GenerationTrace, JsonContract
from wsp.core.budget import Budget

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


class StructuredOutputParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str = Field(default="json", pattern=r"^json$")
    contract: JsonContract = Field(default_factory=JsonContract)
    max_key_findings: int = Field(default=20, ge=1, le=100)


class StructuredOutputDetector(BaseDetector):
    detector_id: ClassVar[str] = "core.structured_output"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = (
        "Validates that the output is JSON and optionally checks required keys and value types "
        "against a small explicit contract."
    )
    required_fields: ClassVar[tuple[str, ...]] = ("text",)
    limitations: ClassVar[tuple[str, ...]] = (
        "A valid JSON document can still be semantically wrong.",
        "The contract engine supports required keys and top-level types only; it is not a general JSON Schema validator.",
    )
    parameters_model: ClassVar[type[BaseModel]] = StructuredOutputParameters
    default_enabled: ClassVar[bool] = False

    def _extract(self, text: str, contract: JsonContract) -> tuple[str, bool]:
        if contract.allow_code_fence:
            match = _FENCE.search(text)
            if match:
                return match.group(1).strip(), True
        return text.strip(), False

    def _type_ok(self, value: Any, expected: str) -> bool:
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return isinstance(value, _TYPES[expected])

    def scan(
        self,
        trace: GenerationTrace,
        parameters: BaseModel,
        budget: Budget,
    ) -> DetectorRun:
        assert isinstance(parameters, StructuredOutputParameters)
        if trace.text is None:
            return self._empty_run(parameters, status="skipped", reason="trace has no output text")
        payload, used_fence = self._extract(trace.text, parameters.contract)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            return DetectorRun(
                detector_id=self.detector_id,
                detector_version=self.version,
                status="ok",
                findings=[
                    self._finding(
                        parameters,
                        kind="invalid_json",
                        value={"error": str(exc), "line": exc.lineno, "column": exc.colno},
                        severity="medium",
                        explanation="The output is not valid JSON under the configured extraction rules.",
                        evidence=[
                            EvidenceRef(
                                kind="text_span",
                                trace_id=trace.trace_id,
                                start=0,
                                end=len(trace.text),
                            )
                        ],
                    )
                ],
                measurements={"parsed": False, "used_code_fence": used_fence},
                limitations=list(self.limitations),
            )

        findings = []
        contract = parameters.contract
        if isinstance(parsed, dict):
            missing = [key for key in contract.required_keys if key not in parsed]
            for key in missing[: parameters.max_key_findings]:
                findings.append(
                    self._finding(
                        parameters,
                        kind="missing_required_key",
                        value=key,
                        severity="medium",
                        explanation=f"The required key {key!r} is absent.",
                    )
                )
            if not contract.allow_extra:
                extras = [key for key in parsed if key not in contract.properties and key not in contract.required_keys]
                for key in extras[: parameters.max_key_findings]:
                    findings.append(
                        self._finding(
                            parameters,
                            kind="unexpected_key",
                            value=key,
                            severity="low",
                            explanation=f"The key {key!r} is not allowed by the contract.",
                        )
                    )
            for key, expected in contract.properties.items():
                if key in parsed and not self._type_ok(parsed[key], expected):
                    findings.append(
                        self._finding(
                            parameters,
                            kind="type_mismatch",
                            value={"key": key, "expected": expected, "observed": type(parsed[key]).__name__},
                            severity="medium",
                            explanation=f"The key {key!r} does not have the required JSON type.",
                        )
                    )
        elif contract.required_keys or contract.properties:
            findings.append(
                self._finding(
                    parameters,
                    kind="wrong_root_type",
                    value={"expected": "object", "observed": type(parsed).__name__},
                    severity="medium",
                    explanation="The contract describes an object, but the output is not a JSON object.",
                )
            )
        return DetectorRun(
            detector_id=self.detector_id,
            detector_version=self.version,
            status="ok",
            findings=findings[: budget.max_scan_findings],
            measurements={
                "parsed": True,
                "used_code_fence": used_fence,
                "root_type": type(parsed).__name__,
                "keys": sorted(parsed)[:100] if isinstance(parsed, dict) else None,
                "valid": not findings,
            },
            limitations=list(self.limitations),
        )
