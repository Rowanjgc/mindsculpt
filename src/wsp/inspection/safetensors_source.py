"""Streaming SafeTensors inspection.

Only the SafeTensors header is read to enumerate tensors.  Tensor values are
read on demand through slices.  No pickle, ``torch.load``, or model code is
involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from wsp.core.dtypes import (
    bytes_per_element,
    decode_safetensors_array,
    normalize_dtype_name,
)
from wsp.core.errors import InputError, NotFoundError, ResourceLimitError, UnsupportedError
from wsp.core.hashing import sha256_file, structure_fingerprint, tensor_fingerprint
from wsp.core.models import CheckpointIdentity, SourceRef, TensorInfo
from wsp.core.selection import ResolvedSelection, Selection
from wsp.inspection.base import CheckpointSource


class SafeTensorsSource(CheckpointSource):
    """A read-only view over a local SafeTensors file."""

    def __init__(
        self,
        path: str | Path,
        *,
        display_name: str | None = None,
        compute_file_sha256: bool = False,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise NotFoundError(
                f"checkpoint does not exist: {self.path}", details={"path": str(self.path)}
            )
        if not self.path.is_file():
            raise InputError(
                f"checkpoint path is not a regular file: {self.path}",
                details={"path": str(self.path)},
            )
        self._display_name = display_name or self.path.name
        self._infos: list[TensorInfo] = []
        self._by_name: dict[str, TensorInfo] = {}
        self._metadata: dict[str, str] = {}
        self._file_sha256: str | None = sha256_file(self.path) if compute_file_sha256 else None
        self._load_header()

    def _load_header(self) -> None:
        try:
            with safe_open(str(self.path), framework="numpy") as handle:
                metadata = handle.metadata() or {}
                self._metadata = {str(k): str(v) for k, v in metadata.items()}
                names = sorted(handle.keys())
                for name in names:
                    view = handle.get_slice(name)
                    dtype = normalize_dtype_name(view.get_dtype())
                    shape = [int(d) for d in view.get_shape()]
                    numel = 1
                    for dimension in shape:
                        numel *= dimension
                    nbytes = numel * bytes_per_element(dtype)
                    info = TensorInfo(
                        name=name, dtype=dtype, shape=shape, numel=numel, nbytes=nbytes
                    )
                    self._infos.append(info)
                    self._by_name[name] = info
        except (OSError, ValueError) as exc:
            raise InputError(
                f"could not read SafeTensors header from {self.path}: {exc}",
                details={"path": str(self.path)},
            ) from exc

    def identity(self) -> CheckpointIdentity:
        payload = {
            "format": "safetensors",
            "tensors": [
                {"name": info.name, "dtype": info.dtype, "shape": info.shape}
                for info in self._infos
            ],
            "metadata": self._metadata,
        }
        return CheckpointIdentity(
            format="safetensors",
            display_name=self._display_name,
            structure_fingerprint=structure_fingerprint(payload),
            file_sha256=self._file_sha256,
            tensor_count=len(self._infos),
            parameter_count=sum(info.numel for info in self._infos),
            total_bytes=sum(info.nbytes for info in self._infos),
            metadata=dict(self._metadata),
        )

    def source_ref(self) -> SourceRef:
        return SourceRef(
            path=str(self.path),
            display_name=self._display_name,
            format="safetensors",
            size_bytes=self.path.stat().st_size,
        )

    def tensor_infos(self) -> list[TensorInfo]:
        return list(self._infos)

    def tensor_info(self, name: str) -> TensorInfo:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise NotFoundError(
                f"tensor {name!r} is not present in {self._display_name}",
                details={"tensor": name, "available_count": len(self._infos)},
            ) from exc

    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    def _resolved(self, name: str, selection: Selection | None, max_elements: int) -> tuple[TensorInfo, ResolvedSelection]:
        info = self.tensor_info(name)
        resolved = (selection or Selection()).resolve(info.shape)
        if resolved.numel > max_elements:
            raise ResourceLimitError(
                f"selection of {name!r} has {resolved.numel:,} elements, exceeding the limit of {max_elements:,}",
                details={
                    "tensor": name,
                    "selection_numel": resolved.numel,
                    "limit": max_elements,
                    "shape": info.shape,
                },
            )
        return info, resolved

    def read_tensor(
        self,
        name: str,
        selection: Selection | None = None,
        *,
        max_elements: int,
    ) -> np.ndarray[Any, Any]:
        info, resolved = self._resolved(name, selection, max_elements)
        if info.dtype in {"float8_e4m3fn", "float8_e5m2"}:
            raise UnsupportedError(
                f"numerical reads of {info.dtype} tensors are not supported in V1; "
                "copy/export preserves the raw tensor",
                details={"tensor": name, "dtype": info.dtype},
            )
        try:
            if info.dtype == "bfloat16":
                # The numpy SafeTensors framework cannot represent BF16.  Read
                # through torch and widen losslessly to float32.
                from wsp.core.dtypes import torch_to_numpy

                with safe_open(str(self.path), framework="pt") as handle:
                    if not info.shape:
                        tensor = handle.get_tensor(name)
                    else:
                        tensor = handle.get_slice(name)[resolved.to_index()]
                decoded = torch_to_numpy(tensor, decode_bfloat16=True)
            else:
                with safe_open(str(self.path), framework="numpy") as handle:
                    view = handle.get_slice(name)
                    if not info.shape:
                        raw = handle.get_tensor(name)
                    else:
                        raw = view[resolved.to_index()]
                decoded = decode_safetensors_array(np.asarray(raw), info.dtype)
        except (OSError, ValueError) as exc:
            raise InputError(
                f"could not read tensor {name!r}: {exc}",
                details={"tensor": name, "path": str(self.path)},
            ) from exc
        expected_shape = resolved.selected_shape
        if list(decoded.shape) != expected_shape:
            # 0-D tensors and integer scalar indexing can drop dimensions.
            if int(np.prod(expected_shape)) == int(decoded.size):
                decoded = decoded.reshape(expected_shape)
            else:
                raise InputError(
                    f"internal selection shape mismatch for {name!r}",
                    details={
                        "tensor": name,
                        "expected_shape": expected_shape,
                        "actual_shape": list(decoded.shape),
                    },
                )
        return decoded

    def read_full(self, name: str, *, max_elements: int) -> np.ndarray[Any, Any]:
        return self.read_tensor(name, None, max_elements=max_elements)

    def tensor_fingerprint(
        self,
        name: str,
        selection: Selection | None = None,
        *,
        max_elements: int,
    ) -> str:
        info = self.tensor_info(name)
        values = self.read_tensor(name, selection, max_elements=max_elements)
        return tensor_fingerprint(
            values,
            storage_dtype=info.dtype,
            shape=values.shape,
        )

    def close(self) -> None:
        # Header data is in memory; reads open their own handles.  Nothing to release.
        return None
