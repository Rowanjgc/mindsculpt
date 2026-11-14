"""Checkpoint source abstraction.

Inspection does not require loading a model.  A source exposes metadata and
bounded tensor reads; concrete sources decide how to satisfy them.  Sources
are read-only by contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from wsp.core.models import CheckpointIdentity, SourceRef, TensorInfo
from wsp.core.selection import Selection


class CheckpointSource(ABC):
    @abstractmethod
    def identity(self) -> CheckpointIdentity: ...

    @abstractmethod
    def source_ref(self) -> SourceRef: ...

    @abstractmethod
    def tensor_infos(self) -> list[TensorInfo]: ...

    @abstractmethod
    def tensor_info(self, name: str) -> TensorInfo: ...

    @abstractmethod
    def read_tensor(
        self,
        name: str,
        selection: Selection | None = None,
        *,
        max_elements: int,
    ) -> np.ndarray[Any, Any]: ...

    @abstractmethod
    def read_full(self, name: str, *, max_elements: int) -> np.ndarray[Any, Any]: ...

    @abstractmethod
    def tensor_fingerprint(
        self,
        name: str,
        selection: Selection | None = None,
        *,
        max_elements: int,
    ) -> str: ...

    @abstractmethod
    def metadata(self) -> dict[str, str]: ...

    def has_tensor(self, name: str) -> bool:
        return any(info.name == name for info in self.tensor_infos())

    def close(self) -> None:
        """Release resources.  Sources are usable again after ``close`` unless documented."""

    def __enter__(self) -> "CheckpointSource":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
