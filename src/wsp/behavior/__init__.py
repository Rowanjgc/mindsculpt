"""Behavioral Observatory: traces, detectors, evaluations, contrasts, cases."""

from __future__ import annotations

from wsp.behavior.backends import (
    GenerationBackend,
    HuggingFaceBackend,
    ScriptedBackend,
    build_backend,
)
from wsp.behavior.cases import CaseStore
from wsp.behavior.compare import compare_traces
from wsp.behavior.contrast import run_contrast
from wsp.behavior.detectors import (
    BaseDetector,
    DetectorRegistry,
    default_registry,
    scan_trace,
)
from wsp.behavior.evaluators import (
    BaseEvaluator,
    EvaluatorRegistry,
    default_evaluators,
)
from wsp.behavior.trace import TraceStore, finalize_trace

__all__ = [
    "BaseDetector",
    "BaseEvaluator",
    "CaseStore",
    "DetectorRegistry",
    "EvaluatorRegistry",
    "GenerationBackend",
    "HuggingFaceBackend",
    "ScriptedBackend",
    "TraceStore",
    "build_backend",
    "compare_traces",
    "default_evaluators",
    "default_registry",
    "finalize_trace",
    "run_contrast",
    "scan_trace",
]
