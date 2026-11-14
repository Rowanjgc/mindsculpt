"""Reconsideration-marker measurement (evidence, not diagnosis)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wsp.behavior.detectors.base import BaseDetector
from wsp.behavior.models import DetectorRun, EvidenceRef, GenerationTrace
from wsp.core.budget import Budget

_DEFAULT_MARKERS = [
    "but wait",
    "wait,",
    "on second thought",
    "actually,",
    "let me reconsider",
    "i need to reconsider",
    "that's not right",
    "hmm",
]

_NOTE = (
    "A marker count is textual evidence, not a diagnosis. The same phrases can be "
    "productive self-correction, stylistic variation, or quoted material."
)


class ReconsiderationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    markers: list[str] = Field(default_factory=lambda: list(_DEFAULT_MARKERS))
    min_markers: int = Field(default=3, ge=1, le=10_000)
    min_cluster_size: int = Field(default=3, ge=2, le=100)
    cluster_gap_chars: int = Field(default=200, ge=0, le=100_000)
    case_sensitive: bool = False
    max_findings: int = Field(default=10, ge=1, le=50)
    max_evidence: int = Field(default=20, ge=1, le=200)

    @model_validator(mode="after")
    def _markers(self) -> "ReconsiderationParameters":
        cleaned = [marker for marker in self.markers if marker.strip()]
        if not cleaned:
            raise ValueError("provide at least one non-empty marker")
        self.markers = cleaned
        return self


class ReconsiderationDetector(BaseDetector):
    detector_id: ClassVar[str] = "core.reconsideration"
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = (
        "Counts configured reconsideration markers and clusters of markers in the output text."
    )
    required_fields: ClassVar[tuple[str, ...]] = ("text",)
    limitations: ClassVar[tuple[str, ...]] = (
        _NOTE,
        "Markers are literal strings; paraphrases and other languages are not detected unless configured.",
    )
    parameters_model: ClassVar[type[BaseModel]] = ReconsiderationParameters

    def _occurrences(self, text: str, parameters: ReconsiderationParameters) -> list[tuple[int, int, str]]:
        haystack = text if parameters.case_sensitive else text.casefold()
        found: list[tuple[int, int, str]] = []
        for marker in parameters.markers:
            needle = marker if parameters.case_sensitive else marker.casefold()
            start = haystack.find(needle)
            while start != -1:
                found.append((start, start + len(needle), marker))
                start = haystack.find(needle, start + max(1, len(needle)))
        found.sort(key=lambda item: (item[0], item[1]))
        return found

    def scan(
        self,
        trace: GenerationTrace,
        parameters: BaseModel,
        budget: Budget,
    ) -> DetectorRun:
        assert isinstance(parameters, ReconsiderationParameters)
        if trace.text is None:
            return self._empty_run(parameters, status="skipped", reason="trace has no output text")
        occurrences = self._occurrences(trace.text, parameters)
        counts: dict[str, int] = {}
        for _start, _end, marker in occurrences:
            counts[marker] = counts.get(marker, 0) + 1
        clusters: list[list[tuple[int, int, str]]] = []
        for occurrence in occurrences:
            if not clusters:
                clusters.append([occurrence])
                continue
            previous_end = clusters[-1][-1][1]
            if occurrence[0] - previous_end <= parameters.cluster_gap_chars:
                clusters[-1].append(occurrence)
            else:
                clusters.append([occurrence])
        large_clusters = [cluster for cluster in clusters if len(cluster) >= parameters.min_cluster_size]

        findings = []
        total = len(occurrences)
        if total >= parameters.min_markers:
            evidence = [
                EvidenceRef(
                    kind="text_span",
                    trace_id=trace.trace_id,
                    start=start,
                    end=end,
                    name=marker,
                )
                for start, end, marker in occurrences[: parameters.max_evidence]
            ]
            severity = "low" if total < parameters.min_markers * 2 else "medium"
            findings.append(
                self._finding(
                    parameters,
                    kind="reconsideration_markers",
                    value={"count": total, "markers": counts},
                    unit="markers",
                    severity=severity,
                    explanation=(
                        f"Configured reconsideration markers occurred {total} times. {_NOTE}"
                    ),
                    evidence=evidence,
                )
            )
        for marker, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if count >= 3:
                findings.append(
                    self._finding(
                        parameters,
                        kind="repeated_reconsideration_marker",
                        value={"marker": marker, "count": count},
                        unit="occurrences",
                        severity="low" if count < 6 else "medium",
                        explanation=f"The marker {marker!r} occurs {count} times. {_NOTE}",
                    )
                )
        for cluster in large_clusters[: parameters.max_findings]:
            start = cluster[0][0]
            end = cluster[-1][1]
            findings.append(
                self._finding(
                    parameters,
                    kind="reconsideration_cluster",
                    value={
                        "markers": len(cluster),
                        "char_span": [start, end],
                        "distinct_markers": sorted({item[2] for item in cluster}),
                    },
                    unit="markers",
                    severity="low" if len(cluster) < parameters.min_cluster_size * 2 else "medium",
                    explanation=(
                        f"{len(cluster)} configured markers occur within {parameters.cluster_gap_chars} "
                        f"characters of each other. {_NOTE}"
                    ),
                    evidence=[
                        EvidenceRef(
                            kind="text_span",
                            trace_id=trace.trace_id,
                            start=start,
                            end=end,
                        )
                    ],
                )
            )
        return DetectorRun(
            detector_id=self.detector_id,
            detector_version=self.version,
            status="ok",
            findings=findings[: parameters.max_findings],
            measurements={
                "total_markers": total,
                "marker_counts": counts,
                "clusters": len(clusters),
                "large_clusters": len(large_clusters),
                "characters": len(trace.text),
            },
            limitations=list(self.limitations),
        )
