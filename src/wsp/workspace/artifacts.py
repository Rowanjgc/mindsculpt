"""Content-addressed artifact storage inside a workspace."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wsp.core.budget import Budget, enforce_bytes
from wsp.core.errors import ArtifactError, NotFoundError, SafetyError
from wsp.core.hashing import sha256_bytes, sha256_file
from wsp.core.json import atomic_write_bytes, read_json
from wsp.core.models import ArtifactRef
from wsp.core.time import utc_now_iso

_ARTIFACT_ID = re.compile(r"^a_[0-9a-f]{16,64}$")
_FORMATS: dict[str, str] = {
    "json": ".json",
    "npy": ".npy",
    "txt": ".txt",
    "bin": ".bin",
    "safetensors": ".safetensors",
}


class ArtifactStore:
    def __init__(self, root: Path, *, budget: Budget | None = None) -> None:
        self.root = root.resolve()
        self.budget = budget or Budget()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts).resolve()
        if path != self.root and self.root not in path.parents:
            raise SafetyError("artifact path escapes the workspace")
        return path

    def _validate_format(self, format_name: str) -> str:
        if format_name not in _FORMATS:
            raise ArtifactError(
                f"unsupported artifact format {format_name!r}",
                details={"allowed": sorted(_FORMATS)},
            )
        return _FORMATS[format_name]

    def put_bytes(
        self,
        kind: str,
        data: bytes,
        *,
        format: str = "bin",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        if not kind or not re.match(r"^[a-zA-Z0-9_.-]+$", kind):
            raise ArtifactError("artifact kind must be a simple identifier")
        enforce_bytes(len(data), self.budget.max_artifact_bytes, what="artifact")
        extension = self._validate_format(format)
        digest = sha256_bytes(data)
        artifact_id = "a_" + digest[len("sha256:") :][:32]
        relative = Path("artifacts") / artifact_id[2:4] / f"{artifact_id}{extension}"
        path = self._safe_path(str(relative))
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        created_at = utc_now_iso()
        if path.exists():
            existing = sha256_file(path)
            if existing != digest:
                raise ArtifactError(
                    "content-addressed artifact exists with different contents",
                    details={"artifact_id": artifact_id},
                )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(path, data, overwrite=False)
        ref = ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            path=str(relative),
            format=format,
            sha256=digest,
            nbytes=len(data),
            created_at=created_at,
            schema_version="wsp.artifact/1.0",
            metadata=dict(metadata or {}),
        )
        if not meta_path.exists():
            atomic_write_bytes(meta_path, ref.model_dump_json().encode("utf-8"), overwrite=False)
        return ref

    def put_json(
        self,
        kind: str,
        payload: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        from wsp.core.json import canonical_bytes

        return self.put_bytes(kind, canonical_bytes(payload), format="json", metadata=metadata)

    def put_array(
        self,
        kind: str,
        array: np.ndarray[Any, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(array), allow_pickle=False)
        return self.put_bytes(kind, buffer.getvalue(), format="npy", metadata=metadata)

    def _path_for_id(self, artifact_id: str) -> tuple[ArtifactRef, Path]:
        if not _ARTIFACT_ID.match(artifact_id):
            raise ArtifactError(
                "invalid artifact identifier",
                details={"artifact_id": artifact_id},
            )
        pattern = f"artifacts/{artifact_id[2:4]}/{artifact_id}.*.meta.json"
        matches = list(self.root.glob(pattern))
        if not matches:
            raise NotFoundError(
                f"artifact {artifact_id} was not found",
                details={"artifact_id": artifact_id},
            )
        if len(matches) != 1:
            raise ArtifactError(
                "artifact metadata is ambiguous",
                details={"artifact_id": artifact_id, "matches": len(matches)},
            )
        meta_path = matches[0].resolve()
        if self.root not in meta_path.parents:
            raise SafetyError("artifact metadata path escapes the workspace")
        ref = ArtifactRef.model_validate(read_json(meta_path))
        data_path = self._safe_path(*Path(ref.path).parts)
        if not data_path.exists():
            raise ArtifactError(
                "artifact data file is missing",
                details={"artifact_id": artifact_id, "path": ref.path},
            )
        return ref, data_path

    def get(self, artifact_id: str) -> tuple[ArtifactRef, Path]:
        ref, path = self._path_for_id(artifact_id)
        self.verify(ref)
        return ref, path

    def read_bytes(self, artifact_id: str) -> bytes:
        _ref, path = self.get(artifact_id)
        return path.read_bytes()

    def read_json(self, artifact_id: str) -> Any:
        _ref, path = self.get(artifact_id)
        return read_json(path)

    def read_array(self, artifact_id: str) -> np.ndarray[Any, Any]:
        _ref, path = self.get(artifact_id)
        with open(path, "rb") as handle:
            value = np.load(handle, allow_pickle=False)
        if not isinstance(value, np.ndarray):
            raise ArtifactError("artifact does not contain a single numpy array")
        return value

    def verify(self, ref: ArtifactRef | str) -> bool:
        if isinstance(ref, str):
            ref, _path = self._path_for_id(ref)
        path = self._safe_path(*Path(ref.path).parts)
        if not path.exists():
            raise ArtifactError("artifact data is missing", details={"artifact_id": ref.artifact_id})
        if path.stat().st_size != ref.nbytes:
            raise ArtifactError(
                "artifact size does not match its record",
                details={"artifact_id": ref.artifact_id},
            )
        if sha256_file(path) != ref.sha256:
            raise ArtifactError(
                "artifact checksum does not match its record",
                details={"artifact_id": ref.artifact_id},
            )
        return True

    def list(self) -> list[ArtifactRef]:
        refs: list[ArtifactRef] = []
        for meta_path in sorted(self.root.glob("artifacts/*/*.meta.json")):
            try:
                ref = ArtifactRef.model_validate(read_json(meta_path))
            except Exception:
                continue
            refs.append(ref)
        return refs
