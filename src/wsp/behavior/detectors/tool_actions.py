"""Repeated equivalent tool calls, with retry ambiguity preserved."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from wsp.behavior.detectors.base import BaseDetector
from wsp.behavior.models import DetectorRun, EvidenceRef, GenerationTrace
from wsp.core.budget import Budget
from wsp.core.hashing import sha256_bytes
from wsp.core.json import canonical_bytes, to_jsonable

_NOTE = (
    "Repeated equivalent calls can be legitimate retries, polling, or idempotent probes. "
    "This detector measures call repetition; it does not label the behavior as an error."
)


class ToolActionParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_repeats: int = Field(default=2, ge=2, le=1000)
    max_findings: int = Field(default=10, ge=1, le=100)
    max_evidence: int = Field(default=20, ge=1, le=1000)


class ToolActionDetector(BaseDetector):
    detector_id: ClassVar[str] = "core.tool_actions"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = (
        "Groups tool calls by name and canonical argument hash and reports repeated groups."
    )
    required_fields: ClassVar[tuple[str, ...]] = ("tool_calls",)
    limitations: ClassVar[tuple[str, ...]] = (
        _NOTE,
        "Only syntactically equivalent arguments are grouped; semantically equivalent calls are not detected.",
    )
    parameters_model: ClassVar[type[BaseModel]] = ToolActionParameters

    def _argument_hash(self, call: Any) -> str:
        if call.arguments_sha256:
            return call.arguments_sha256
        if call.arguments is None:
            return sha256_bytes(b"null")
        return sha256_bytes(canonical_bytes(to_jsonable(call.arguments)))

    def scan(
        self,
        trace: GenerationTrace,
        parameters: BaseModel,
        budget: Budget,
    ) -> DetectorRun:
        assert isinstance(parameters, ToolActionParameters)
        if not trace.tool_calls:
            return self._empty_run(parameters, status="skipped", reason="trace has no tool-call records")
        groups: dict[tuple[str, str], list[Any]] = {}
        for call in trace.tool_calls:
            groups.setdefault((call.name, self._argument_hash(call)), []).append(call)
        repeated = [(key, calls) for key, calls in groups.items() if len(calls) >= parameters.min_repeats]
        repeated.sort(key=lambda item: (-len(item[1]), item[1][0].index))
        findings = []
        for (name, digest), calls in repeated[: parameters.max_findings]:
            statuses = [call.status for call in calls]
            retry_like = "error" in statuses
            evidence = [
                EvidenceRef(
                    kind="tool_call",
                    trace_id=trace.trace_id,
                    start=call.index,
                    end=call.index + 1,
                    name=call.name,
                    detail={"status": call.status},
                )
                for call in calls[: parameters.max_evidence]
            ]
            findings.append(
                self._finding(
                    parameters,
                    kind="repeated_tool_call",
                    value={
                        "name": name,
                        "count": len(calls),
                        "call_indices": [call.index for call in calls],
                        "argument_sha256": digest,
                        "statuses": statuses,
                    },
                    unit="calls",
                    severity="low",
                    explanation=(
                        f"Equivalent {name!r} calls occurred {len(calls)} times. "
                        + ("An earlier call in the group reported an error, so retries are plausible. " if retry_like else "")
                        + _NOTE
                    ),
                    evidence=evidence,
                    metadata={"retry_like": retry_like},
                )
            )
        return DetectorRun(
            detector_id=self.detector_id,
            detector_version=self.version,
            status="ok",
            findings=findings,
            measurements={
                "calls": len(trace.tool_calls),
                "distinct_groups": len(groups),
                "repeated_groups": len(repeated),
                "retry_like_groups": sum(
                    1 for _key, calls in repeated if any(call.status == "error" for call in calls)
                ),
            },
            limitations=list(self.limitations),
        )
