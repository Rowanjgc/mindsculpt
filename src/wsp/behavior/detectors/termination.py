"""Termination observation without cause attribution."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from wsp.behavior.detectors.base import BaseDetector
from wsp.behavior.models import DetectorRun, GenerationTrace
from wsp.core.budget import Budget


class TerminationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_unknown: bool = True
    report_stop_sequence: bool = False


class TerminationDetector(BaseDetector):
    detector_id: ClassVar[str] = "core.termination"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = (
        "Reports how generation ended, distinguishing EOS, stop sequences, tool handoff, "
        "length limits, execution errors, and unknown termination."
    )
    required_fields: ClassVar[tuple[str, ...]] = ("termination",)
    limitations: ClassVar[tuple[str, ...]] = (
        "A length-limit termination is an observation about this run, not evidence that the model cannot terminate.",
        "The backend may not be able to distinguish a natural EOS from a configured stop sequence.",
    )
    parameters_model: ClassVar[type[BaseModel]] = TerminationParameters

    def scan(
        self,
        trace: GenerationTrace,
        parameters: BaseModel,
        budget: Budget,
    ) -> DetectorRun:
        assert isinstance(parameters, TerminationParameters)
        termination = trace.termination
        findings = []
        if termination.reason == "length_limit":
            findings.append(
                self._finding(
                    parameters,
                    kind="length_limit_termination",
                    value={
                        "generated_tokens": termination.generated_tokens,
                        "max_new_tokens": trace.params.max_new_tokens,
                    },
                    unit="tokens",
                    severity="medium",
                    explanation=(
                        "Generation stopped at the configured token limit. This is a termination "
                        "observation; it does not show why the generation did not end on its own."
                    ),
                )
            )
        elif termination.reason == "error":
            findings.append(
                self._finding(
                    parameters,
                    kind="generation_error",
                    value=termination.detail,
                    severity="high",
                    explanation="The backend reported an execution error during generation.",
                )
            )
        elif termination.reason == "unknown" and parameters.report_unknown:
            findings.append(
                self._finding(
                    parameters,
                    kind="unknown_termination",
                    value=termination.detail,
                    severity="info",
                    explanation=(
                        "The backend did not report a recognized termination reason. "
                        "No cause is inferred."
                    ),
                )
            )
        elif termination.reason == "stop_sequence" and parameters.report_stop_sequence:
            findings.append(
                self._finding(
                    parameters,
                    kind="stop_sequence_termination",
                    value=termination.stop_sequence,
                    severity="info",
                    explanation="Generation stopped because a configured stop sequence appeared.",
                )
            )
        return DetectorRun(
            detector_id=self.detector_id,
            detector_version=self.version,
            status="ok",
            findings=findings,
            measurements={
                "reason": termination.reason,
                "detail": termination.detail,
                "stop_sequence": termination.stop_sequence,
                "generated_tokens": termination.generated_tokens,
                "max_new_tokens": trace.params.max_new_tokens,
                "reached_length_limit": termination.reason == "length_limit",
            },
            limitations=list(self.limitations),
        )
