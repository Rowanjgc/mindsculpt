"""Read-only adapter for ordinary PyTorch modules.

This is a *live* view of a module in the current interpreter.  It is useful for
inspection and activation work.  For immutable patch sources, export or use a
SafeTensors checkpoint instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wsp.core.dtypes import bytes_per_element, torch_dtype_name, torch_to_numpy
from wsp.core.errors import InputError, NotFoundError, ResourceLimitError
from wsp.core.hashing import structure_fingerprint, tensor_fingerprint
from wsp.core.models import CheckpointIdentity, SourceRef, TensorInfo
from wsp.core.selection import ResolvedSelection, Selection
from wsp.inspection.base import CheckpointSource


class TorchModuleSource(CheckpointSource):
    """Inspect a live ``torch.nn.Module`` without copying its full state."""

    def __init__(self, module: Any, *, display_name: str = "torch_module") -> None:
        self.module = module
        self._display_name = display_name
        self._state = module.state_dict()
        self._infos: list[TensorInfo] = []
        self._by_name: dict[str, TensorInfo] = {}
        for name in sorted(self._state):
            tensor = self._state[name]
            try:
                dtype = torch_dtype_name(tensor)
            except Exception as exc:  # pragma: no cover - exotic dtypes
                raise InputError(
                    f"unsupported dtype on module tensor {name!r}: {tensor.dtype}"
                ) from exc
            shape = [int(d) for d in tensor.shape]
            numel = int(tensor.numel())
            info = TensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                numel=numel,
                nbytes=numel * bytes_per_element(dtype),
            )
            self._infos.append(info)
            self._by_name[name] = info

    def identity(self) -> CheckpointIdentity:
        payload = {
            "format": "torch_module",
            "tensors": [
                {"name": info.name, "dtype": info.dtype, "shape": info.shape}
                for info in self._infos
            ],
            "metadata": {},
        }
        return CheckpointIdentity(
            format="torch_module",
            display_name=self._display_name,
            structure_fingerprint=structure_fingerprint(payload),
            file_sha256=None,
            tensor_count=len(self._infos),
            parameter_count=sum(info.numel for info in self._infos),
            total_bytes=sum(info.nbytes for info in self._infos),
            metadata={},
        )

    def source_ref(self) -> SourceRef:
        return SourceRef(display_name=self._display_name, format="torch_module")

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
        return {}

    def _resolved(
        self, name: str, selection: Selection | None, max_elements: int
    ) -> tuple[TensorInfo, ResolvedSelection]:
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
        tensor = self._state[name]
        if info.shape:
            tensor = tensor[resolved.to_index()]
        values = torch_to_numpy(tensor)
        expected_shape = resolved.selected_shape
        if list(values.shape) != expected_shape:
            if int(values.size) == int(np.prod(expected_shape)):
                values = values.reshape(expected_shape)
            else:
                raise InputError(
                    f"selection shape mismatch for {name!r}",
                    details={"expected_shape": expected_shape, "actual_shape": list(values.shape)},
                )
        return values

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
        return tensor_fingerprint(values, storage_dtype=info.dtype, shape=values.shape)
