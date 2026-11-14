"""Replayable run recording."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from wsp.core.budget import Budget
from wsp.core.environment import dependency_versions, environment_info
from wsp.core.errors import WspError
from wsp.core.json import to_jsonable
from wsp.core.models import ArtifactRef, CheckpointIdentity, SourceRef
from wsp.core.time import utc_now_iso
from wsp.workspace.models import RunRecord


def sanitize_parameters(value: Any) -> Any:
    """Convert parameter objects to JSON-safe summaries.

    Large arrays/tensors are represented by shape/dtype rather than values;
    callers remain responsible for not passing prompt text or secrets.
    """

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a required dependency
        np = None
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a required dependency
        torch = None

    if isinstance(value, BaseModel):
        return sanitize_parameters(value.model_dump(mode="python"))
    if np is not None and isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "dtype": str(value.dtype), "numel": int(value.size)}
    if torch is not None and isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype), "numel": int(value.numel())}
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): sanitize_parameters(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_parameters(item) for item in value]
    return to_jsonable(value)


class RunRecorder:
    """Record one operation with enough metadata for deterministic replay."""

    def __init__(
        self,
        workspace: Any,
        operation: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        model_identity: CheckpointIdentity | None = None,
        source: SourceRef | None = None,
        device: str | None = None,
        dtype: str | None = None,
        seed: int | None = None,
        budget: Budget | None = None,
        record_environment: bool = True,
    ) -> None:
        self.workspace = workspace
        self.operation = operation
        self.parameters = sanitize_parameters(dict(parameters or {}))
        self.model_identity = model_identity
        self.source = source
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.budget = budget or Budget()
        self.record_environment = record_environment
        self.run_id: str | None = None
        self.artifacts: list[ArtifactRef] = []
        self.warnings: list[str] = []
        self.result: dict[str, Any] | None = None
        self._started = 0.0
        self._finished = False
        self._record: RunRecord | None = None

    def __enter__(self) -> "RunRecorder":
        self.run_id = self.workspace.new_run_id()
        self._started = time.perf_counter()
        self._record = self._build(status="running")
        self.workspace.save_run(self._record)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._finished:
            return
        if exc is not None:
            error: dict[str, Any] = {"type": exc_type.__name__, "message": str(exc)}
            if isinstance(exc, WspError):
                error.update({"code": exc.code, "details": exc.details})
            self.finish(status="failed", error=error)
        else:
            self.finish(status="succeeded")

    def add_artifact(self, ref: ArtifactRef) -> None:
        self.artifacts.append(ref)

    def add_json_artifact(
        self, kind: str, payload: Any, *, metadata: Mapping[str, Any] | None = None
    ) -> ArtifactRef:
        ref = self.workspace.artifacts.put_json(kind, payload, metadata=metadata)
        self.add_artifact(ref)
        return ref

    def add_array_artifact(
        self, kind: str, array: Any, *, metadata: Mapping[str, Any] | None = None
    ) -> ArtifactRef:
        ref = self.workspace.artifacts.put_array(kind, array, metadata=metadata)
        self.add_artifact(ref)
        return ref

    def finish(
        self,
        *,
        status: str = "succeeded",
        result: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> RunRecord:
        if self.run_id is None:
            raise RuntimeError("RunRecorder.finish called before entering the context")
        if result is not None:
            self.result = sanitize_parameters(dict(result))
        if warnings:
            self.warnings.extend(warnings)
        record = self._build(status=status, error=dict(error) if error else None)
        self.workspace.save_run(record, overwrite=True)
        self._record = record
        self._finished = True
        return record

    def _build(
        self, *, status: str, error: dict[str, Any] | None = None
    ) -> RunRecord:
        if self.run_id is None:
            raise RuntimeError("run id is not initialized")
        now = utc_now_iso()
        duration_ms = int((time.perf_counter() - self._started) * 1000) if self._started else None
        environment = environment_info() if self.record_environment else {}
        replay = {
            "operation": self.operation,
            "parameters": self.parameters,
            "source": self.source.model_dump() if self.source else None,
            "model_identity": self.model_identity.model_dump() if self.model_identity else None,
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "budget": self.budget.model_dump(),
        }
        return RunRecord(
            run_id=self.run_id,
            created_at=self._record.created_at if self._record else now,
            completed_at=None if status == "running" else now,
            operation=self.operation,
            status=status,
            parameters=self.parameters,
            model_identity=self.model_identity,
            source=self.source,
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
            budget=self.budget.model_dump(),
            environment=environment,
            artifacts=list(self.artifacts),
            result=self.result,
            warnings=list(self.warnings),
            error=error,
            duration_ms=duration_ms,
            replay=replay,
        )
