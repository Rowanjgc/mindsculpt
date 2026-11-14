"""Detector protocol, registry, and bounded scan orchestration."""

from __future__ import annotations

from typing import Any, ClassVar, Iterable

from pydantic import BaseModel, ValidationError

from wsp.behavior.ids import new_scan_id
from wsp.behavior.models import (
    DetectorRun,
    DetectorSelection,
    Finding,
    GenerationTrace,
    ScanResult,
)
from wsp.core.budget import Budget
from wsp.core.errors import InputError, NotFoundError, ResourceLimitError, WspError
from wsp.core.json import to_jsonable
from wsp.core.time import utc_now_iso


class BaseDetector:
    """A deterministic measurement over one trace.

    Implementations must not infer causes.  A detector returns measurements
    and, when a configured condition is met, one or more findings.  Findings
    carry their parameters and limitations so that they remain interpretable
    after the code changes.
    """

    detector_id: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = ""
    required_fields: ClassVar[tuple[str, ...]] = ()
    limitations: ClassVar[tuple[str, ...]] = ()
    parameters_model: ClassVar[type[BaseModel]]
    default_enabled: ClassVar[bool] = True

    def default_parameters(self) -> dict[str, Any]:
        return self.parameters_model().model_dump(mode="json")

    def validate_parameters(self, payload: dict[str, Any] | None) -> BaseModel:
        try:
            return self.parameters_model.model_validate(payload or {})
        except ValidationError as exc:
            raise InputError(
                f"invalid parameters for detector {self.detector_id}",
                details={
                    "errors": [
                        {"loc": list(item.get("loc", [])), "msg": item.get("msg")}
                        for item in exc.errors(include_url=False)
                    ]
                },
            ) from exc

    def missing_fields(self, trace: GenerationTrace) -> list[str]:
        missing: list[str] = []
        for field in self.required_fields:
            if getattr(trace, field, None) in (None, [], {}):
                missing.append(field)
        return missing

    def scan(
        self,
        trace: GenerationTrace,
        parameters: BaseModel,
        budget: Budget,
    ) -> DetectorRun:  # pragma: no cover - interface
        raise NotImplementedError

    def _empty_run(self, parameters: BaseModel, *, status: str = "ok", reason: str | None = None) -> DetectorRun:
        return DetectorRun(
            detector_id=self.detector_id,
            detector_version=self.version,
            status=status,  # type: ignore[arg-type]
            skip_reason=reason,
            limitations=list(self.limitations),
        )

    def _finding(
        self,
        parameters: BaseModel,
        *,
        kind: str,
        value: Any,
        explanation: str,
        unit: str | None = None,
        severity: str | None = None,
        evidence: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Finding:
        return Finding(
            detector_id=self.detector_id,
            detector_version=self.version,
            kind=kind,
            value=value,
            unit=unit,
            severity=severity,  # type: ignore[arg-type]
            explanation=explanation,
            evidence=list(evidence or []),
            parameters=parameters.model_dump(mode="json"),
            limitations=list(self.limitations),
            metadata=metadata or {},
        )


class DetectorRegistry:
    def __init__(self) -> None:
        self._detectors: dict[str, BaseDetector] = {}

    def register(self, detector: BaseDetector, *, replace: bool = False) -> None:
        if not detector.detector_id or "/" in detector.detector_id:
            raise InputError("detector ids must be simple identifiers")
        if detector.detector_id in self._detectors and not replace:
            raise InputError(f"detector {detector.detector_id!r} is already registered")
        self._detectors[detector.detector_id] = detector

    def get(self, detector_id: str) -> BaseDetector:
        detector = self._detectors.get(detector_id)
        if detector is None:
            raise NotFoundError(
                f"unknown detector {detector_id!r}",
                details={"available": sorted(self._detectors)},
            )
        return detector

    def list(self) -> list[BaseDetector]:
        return [self._detectors[key] for key in sorted(self._detectors)]

    def describe(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for detector in self.list():
            entries.append(
                {
                    "detector_id": detector.detector_id,
                    "version": detector.version,
                    "description": detector.description,
                    "required_fields": list(detector.required_fields),
                    "parameters_schema": detector.parameters_model.model_json_schema(),
                    "default_parameters": detector.default_parameters(),
                    "default_enabled": detector.default_enabled,
                    "limitations": list(detector.limitations),
                }
            )
        return entries


def scan_trace(
    trace: GenerationTrace,
    *,
    budget: Budget,
    selections: Iterable[DetectorSelection] | None = None,
    registry: DetectorRegistry | None = None,
) -> ScanResult:
    registry = registry or default_registry()
    if selections is None:
        selections = [
            DetectorSelection(detector_id=detector.detector_id)
            for detector in registry.list()
            if detector.default_enabled
        ]
    selections = list(selections)
    if not selections:
        raise InputError("select at least one detector")
    seen: set[str] = set()
    for selection in selections:
        if selection.detector_id in seen:
            raise InputError(
                "a detector may appear only once per scan",
                details={"detector_id": selection.detector_id},
            )
        seen.add(selection.detector_id)

    runs: list[DetectorRun] = []
    warnings: list[str] = []
    total = 0
    truncated = False
    for selection in selections:
        detector = registry.get(selection.detector_id)
        parameters = detector.validate_parameters(selection.parameters)
        missing = detector.missing_fields(trace)
        if missing:
            runs.append(
                detector._empty_run(
                    parameters,
                    status="skipped",
                    reason="trace does not provide required field(s): " + ", ".join(missing),
                )
            )
            continue
        try:
            run = detector.scan(trace, parameters, budget)
        except (ResourceLimitError, InputError):
            raise
        except WspError as exc:
            runs.append(
                DetectorRun(
                    detector_id=detector.detector_id,
                    detector_version=detector.version,
                    status="error",
                    error=str(exc),
                    limitations=list(detector.limitations),
                )
            )
            warnings.append(f"detector {detector.detector_id} failed: {exc}")
            continue
        except Exception as exc:  # defensive: one detector must not break a scan
            runs.append(
                DetectorRun(
                    detector_id=detector.detector_id,
                    detector_version=detector.version,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    limitations=list(detector.limitations),
                )
            )
            warnings.append(f"detector {detector.detector_id} failed unexpectedly: {exc}")
            continue
        overflow = total + len(run.findings) - budget.max_scan_findings
        if overflow > 0:
            run.findings = run.findings[: max(0, len(run.findings) - overflow)]
            truncated = True
        total += len(run.findings)
        runs.append(run)

    return ScanResult(
        scan_id=new_scan_id(),
        trace_id=trace.trace_id,
        created_at=utc_now_iso(),
        detectors=[selection.detector_id for selection in selections],
        runs=runs,
        findings_count=total,
        truncated=truncated,
        warnings=warnings,
    )


_default_registry: DetectorRegistry | None = None


def default_registry() -> DetectorRegistry:
    global _default_registry
    if _default_registry is None:
        from wsp.behavior.detectors.reconsideration import ReconsiderationDetector
        from wsp.behavior.detectors.repetition import RepetitionDetector
        from wsp.behavior.detectors.structured_output import StructuredOutputDetector
        from wsp.behavior.detectors.termination import TerminationDetector
        from wsp.behavior.detectors.tool_actions import ToolActionDetector

        registry = DetectorRegistry()
        for detector in (
            RepetitionDetector(),
            TerminationDetector(),
            StructuredOutputDetector(),
            ReconsiderationDetector(),
            ToolActionDetector(),
        ):
            registry.register(detector)
        _default_registry = registry
    return _default_registry


def findings_json(result: ScanResult) -> list[dict[str, Any]]:
    """Flatten findings for compact tool output."""

    payload: list[dict[str, Any]] = []
    for run in result.runs:
        for finding in run.findings:
            payload.append(to_jsonable(finding.model_dump(mode="json")))
    return payload
