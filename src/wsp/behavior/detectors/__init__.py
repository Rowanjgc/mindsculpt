"""Built-in detectors and the default registry."""

from __future__ import annotations

from wsp.behavior.detectors.base import (
    BaseDetector,
    DetectorRegistry,
    default_registry,
    scan_trace,
)
from wsp.behavior.detectors.reconsideration import ReconsiderationDetector
from wsp.behavior.detectors.repetition import RepetitionDetector
from wsp.behavior.detectors.structured_output import StructuredOutputDetector
from wsp.behavior.detectors.termination import TerminationDetector
from wsp.behavior.detectors.tool_actions import ToolActionDetector

__all__ = [
    "BaseDetector",
    "DetectorRegistry",
    "ReconsiderationDetector",
    "RepetitionDetector",
    "StructuredOutputDetector",
    "TerminationDetector",
    "ToolActionDetector",
    "default_registry",
    "scan_trace",
]
