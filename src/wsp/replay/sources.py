"""How a replay obtains a model.

A replay plan records *identity*; it never records a path that a later process
is expected to trust blindly.  The caller supplies a source that can be loaded
locally, and the engine compares the loaded identity with the recorded one
before a single forward pass is allowed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wsp.behavior.models import SyntheticBackendConfig
from wsp.core.errors import InputError


class ReplayModelSource(BaseModel):
    """A locally available model that can be used for replay.

    Two kinds exist:

    ``synthetic``
        the offline fixture causal LM, rebuilt deterministically from a spec;
    ``local_hf``
        an existing local ``transformers`` model directory.

    Nothing here can download weights or contact a service.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["synthetic", "local_hf"]
    synthetic: SyntheticBackendConfig | None = None
    path: str | None = None
    device: str = "cpu"
    dtype: str | None = None
    compute_file_sha256: bool = True

    @model_validator(mode="after")
    def _consistent(self) -> "ReplayModelSource":
        if self.kind == "synthetic":
            if self.path is not None:
                raise ValueError("a synthetic source does not take a path")
            if self.synthetic is None:
                self.synthetic = SyntheticBackendConfig()
        else:
            if self.synthetic is not None:
                raise ValueError("a local_hf source does not take a synthetic configuration")
            if not self.path:
                raise ValueError("a local_hf source requires a local directory path")
        return self

    def describe(self) -> dict[str, Any]:
        """Safe, path-free description for evidence records."""

        if self.kind == "synthetic":
            assert self.synthetic is not None
            return {
                "kind": "synthetic",
                "name": self.synthetic.name,
                "spec_digest": self.synthetic.model.digest(),
                "device": self.device,
                "dtype": self.dtype,
            }
        return {
            "kind": "local_hf",
            "device": self.device,
            "dtype": self.dtype,
            "compute_file_sha256": self.compute_file_sha256,
        }


def load_replay_source(source: ReplayModelSource) -> Any:
    """Build a :class:`wsp.replay.backends.ReplayModel` from a source."""

    if source.kind == "synthetic":
        from wsp.execution.synthetic_lm import build_runtime

        config = source.synthetic or SyntheticBackendConfig()
        runtime = build_runtime(config.model)
        model = runtime.replay_model()
        model.name = config.name
        return model
    if source.kind == "local_hf":
        from wsp.replay.backends import load_local_replay_model

        if not source.path:
            raise InputError("a local_hf replay source requires a local directory path")
        return load_local_replay_model(
            source.path,
            device=source.device,
            dtype=source.dtype,
            compute_file_sha256=source.compute_file_sha256,
        )
    raise InputError(f"unknown replay source kind {source.kind!r}")


__all__ = ["ReplayModelSource", "load_replay_source"]
