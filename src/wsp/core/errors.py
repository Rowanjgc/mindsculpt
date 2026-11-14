"""Typed errors used across the workbench.

Every user-facing failure should be one of these errors.  The tool layer turns
them into structured error envelopes with stable ``code`` values; tracebacks
are never returned to callers by default.
"""

from __future__ import annotations

from typing import Any


class WspError(Exception):
    """Base class for expected, user-facing workbench errors."""

    code = "wsp_error"
    http_status = 400

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class InputError(WspError):
    """A request is malformed or semantically invalid."""

    code = "invalid_input"


class NotFoundError(WspError):
    """A requested model, tensor, patch, run, or artifact does not exist."""

    code = "not_found"
    http_status = 404


class UnsupportedError(WspError):
    """The request is well-formed but outside the V1 capability envelope."""

    code = "unsupported"


class ResourceLimitError(WspError):
    """An operation would exceed a configured memory, compute, or output budget."""

    code = "resource_limit"
    http_status = 413


class CompatibilityError(WspError):
    """Checkpoints, tensors, patches, or execution inputs are incompatible."""

    code = "incompatible"
    http_status = 409


class SafetyError(WspError):
    """An operation could violate file-safety or trust boundaries."""

    code = "safety_violation"
    http_status = 403


class ArtifactError(WspError):
    """An artifact is missing, malformed, or attached to the wrong workspace."""

    code = "artifact_error"


class ExecutionError(WspError):
    """A model could not be loaded or executed."""

    code = "execution_error"


class InternalError(WspError):
    """An unexpected internal failure.  Details are intentionally limited."""

    code = "internal_error"
    http_status = 500
