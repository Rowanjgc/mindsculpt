"""Token- and word-level repetition measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from wsp.behavior.detectors.base import BaseDetector
from wsp.behavior.models import DetectorRun, EvidenceRef, GenerationTrace
from wsp.core.budget import Budget
from wsp.core.errors import ResourceLimitError

_SAFE_REPETITION_NOTE = (
    "Repeated text can be intentional (lists, code, verse, tables, or deliberate emphasis). "
    "This detector measures repetition; it does not diagnose a reasoning failure."
)


class RepetitionParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ngram_size: int = Field(default=8, ge=2, le=256)
    min_occurrences: int = Field(default=2, ge=2, le=1000)
    max_findings: int = Field(default=5, ge=1, le=50)
    ignore_case: bool = True
    run_length_threshold: int = Field(default=8, ge=2, le=10_000)


@dataclass
class _Unit:
    text: str
    index: int
    start: int | None
    end: int | None
    kind: str


class RepetitionDetector(BaseDetector):
    detector_id: ClassVar[str] = "core.repetition"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = (
        "Counts repeated n-gram sequences and long runs of one repeated unit in the "
        "generated text or token stream."
    )
    required_fields: ClassVar[tuple[str, ...]] = ()
    limitations: ClassVar[tuple[str, ...]] = (_SAFE_REPETITION_NOTE,)
    parameters_model: ClassVar[type[BaseModel]] = RepetitionParameters

    def _units(self, trace: GenerationTrace, parameters: RepetitionParameters) -> tuple[list[_Unit], str]:
        if trace.tokens and all(token.text is not None for token in trace.tokens):
            units = [
                _Unit(
                    text=token.text or "",
                    index=index,
                    start=index,
                    end=index + 1,
                    kind="token_range",
                )
                for index, token in enumerate(trace.tokens)
            ]
            unit_kind = "tokens"
        elif trace.text:
            units = []
            import re

            for index, match in enumerate(re.finditer(r"\S+", trace.text)):
                units.append(
                    _Unit(
                        text=match.group(0),
                        index=index,
                        start=match.start(),
                        end=match.end(),
                        kind="word_range",
                    )
                )
            unit_kind = "words"
        else:
            return [], "none"
        if len(units) > 0 and parameters.ignore_case:
            for unit in units:
                unit.text = unit.text.casefold()
        return units, unit_kind

    def scan(
        self,
        trace: GenerationTrace,
        parameters: BaseModel,
        budget: Budget,
    ) -> DetectorRun:
        assert isinstance(parameters, RepetitionParameters)
        units, unit_kind = self._units(trace, parameters)
        if not units:
            return self._empty_run(parameters, status="skipped", reason="trace has neither tokens nor text")
        if len(units) > budget.max_scan_tokens:
            raise ResourceLimitError(
                "repetition analysis exceeds the configured token limit",
                details={"units": len(units), "limit": budget.max_scan_tokens},
            )
        n = parameters.ngram_size
        groups: dict[tuple[str, ...], list[int]] = {}
        for index in range(max(0, len(units) - n + 1)):
            key = tuple(unit.text for unit in units[index : index + n])
            if not any(key):
                continue
            groups.setdefault(key, []).append(index)
        repeated = [
            (key, positions)
            for key, positions in groups.items()
            if len(positions) >= parameters.min_occurrences
        ]
        repeated.sort(key=lambda item: (-len(item[1]), item[1][0]))
        coverage: set[int] = set()
        for _key, positions in repeated:
            for start in positions:
                coverage.update(range(start, start + n))

        runs: list[tuple[int, int]] = []
        run_start = 0
        for index in range(1, len(units) + 1):
            if index == len(units) or units[index].text != units[run_start].text:
                if index - run_start >= parameters.run_length_threshold:
                    runs.append((run_start, index))
                run_start = index

        findings = []
        for key, positions in repeated[: parameters.max_findings]:
            evidence: list[EvidenceRef] = []
            for start in positions[:10]:
                first, last = units[start], units[start + n - 1]
                evidence.append(
                    EvidenceRef(
                        kind=first.kind,  # type: ignore[arg-type]
                        trace_id=trace.trace_id,
                        start=first.start,
                        end=last.end,
                        detail={"unit_index_start": first.index, "unit_index_end": last.index + 1},
                    )
                )
            count = len(positions)
            severity = "low" if count == 2 else ("medium" if count <= 4 else "high")
            findings.append(
                self._finding(
                    parameters,
                    kind="repeated_ngram",
                    value={"sequence": list(key), "count": count},
                    unit="occurrences",
                    severity=severity,
                    explanation=(
                        f"A {n}-{unit_kind[:-1] if unit_kind != 'none' else 'unit'} sequence occurs "
                        f"{count} times. {_SAFE_REPETITION_NOTE}"
                    ),
                    evidence=evidence,
                )
            )

        for start, end in runs[: parameters.max_findings]:
            first, last = units[start], units[end - 1]
            length = end - start
            findings.append(
                self._finding(
                    parameters,
                    kind="repeated_token_run",
                    value={"text": first.text, "length": length},
                    unit="consecutive_repeats",
                    severity="medium" if length < parameters.run_length_threshold * 2 else "high",
                    explanation=(
                        f"The same unit occurs {length} times consecutively. "
                        f"{_SAFE_REPETITION_NOTE}"
                    ),
                    evidence=[
                        EvidenceRef(
                            kind=first.kind,  # type: ignore[arg-type]
                            trace_id=trace.trace_id,
                            start=first.start,
                            end=last.end,
                        )
                    ],
                )
            )

        return DetectorRun(
            detector_id=self.detector_id,
            detector_version=self.version,
            status="ok",
            findings=findings[: budget.max_scan_findings],
            measurements={
                "unit": unit_kind,
                "units_analyzed": len(units),
                "ngram_size": n,
                "distinct_ngrams": len(groups),
                "repeated_ngrams": len(repeated),
                "max_occurrences": max((len(item[1]) for item in repeated), default=0),
                "repeat_coverage": (len(coverage) / len(units)) if units else 0.0,
                "longest_run": max((end - start for start, end in runs), default=0),
            },
            limitations=list(self.limitations),
        )
