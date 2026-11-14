"""Versioned workspace and run-record schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from wsp.core.models import ArtifactRef, CheckpointIdentity, SourceRef


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.workspace/1.0"
    workspace_id: str
    name: str
    created_at: str
    description: str | None = None


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.run/1.0"
    run_id: str
    created_at: str
    completed_at: str | None = None
    operation: str
    status: Literal["running", "succeeded", "failed"]
    parameters: dict[str, Any] = Field(default_factory=dict)
    model_identity: CheckpointIdentity | None = None
    source: SourceRef | None = None
    device: str | None = None
    dtype: str | None = None
    seed: int | None = None
    budget: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    duration_ms: int | None = None
    replay: dict[str, Any] = Field(default_factory=dict)


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.experiment/1.0"
    experiment_id: str
    name: str
    description: str | None = None
    created_at: str
    tags: list[str] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
