"""Shared measurement construction for activation capture backends.

Every capture backend (PyTorch forward hooks, or the optional NNsight/nnterp
adapter) ends up holding the same thing: a tensor produced by one module on one
forward position. This module turns that tensor into a :class:`Measurement`
without knowing how it was obtained, so summary and selected-value semantics,
resource accounting, stride sampling, and the statistics themselves stay
identical across backends.

The backend identity is recorded by the caller; nothing here infers it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import torch

from wsp.core.budget import Budget
from wsp.core.dtypes import torch_dtype_name, torch_to_numpy
from wsp.core.errors import InputError, ResourceLimitError
from wsp.investigation.models import CaptureMode, DimensionPeak, Measurement
from wsp.tensors.stats import basic_stats

_TOP_DIMENSIONS = 8


def _json_float(value: float) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _top_dimensions(flat: np.ndarray[Any, Any]) -> list[DimensionPeak]:
    if flat.ndim != 1 or flat.size == 0:
        return []
    finite = np.isfinite(flat)
    if not finite.any():
        return []
    magnitude = np.where(finite, np.abs(flat), -np.inf)
    take = min(_TOP_DIMENSIONS, int(magnitude.size))
    indices = np.argpartition(-magnitude, take - 1)[:take]
    indices = indices[np.argsort(-magnitude[indices])]
    peaks: list[DimensionPeak] = []
    for index in indices:
        value = _json_float(flat[int(index)])
        if value is None:
            continue
        peaks.append(DimensionPeak(index=int(index), value=value))
    return peaks


def _finite_norm(flat: np.ndarray[Any, Any]) -> float | None:
    if flat.size == 0:
        return None
    values = flat[np.isfinite(flat)]
    if values.size == 0:
        return None
    return _json_float(float(np.linalg.norm(values.astype(np.float64, copy=False))))


class MeasurementBuilder:
    """Turn captured tensors into bounded, position-indexed measurements."""

    def __init__(
        self,
        *,
        sequence_length: int,
        positions: Iterable[int],
        mode: CaptureMode = "summary",
        max_elements_per_tensor: int | None = None,
        max_elements_total: int | None = None,
        budget: Budget | None = None,
    ) -> None:
        settings = budget or Budget()
        self.sequence_length = int(sequence_length)
        self.mode: CaptureMode = mode
        self.max_elements_per_tensor = int(
            max_elements_per_tensor or settings.max_activation_elements_per_tensor
        )
        self.max_elements_total = int(
            max_elements_total or settings.max_activation_elements_total
        )
        if self.mode not in ("summary", "selected"):
            raise InputError("capture mode must be 'summary' or 'selected'")
        if self.max_elements_per_tensor <= 0 or self.max_elements_total <= 0:
            raise InputError("activation element budgets must be positive")
        if self.sequence_length <= 0:
            raise InputError("sequence length must be positive")

        self.positions = sorted({int(position) for position in positions})
        for position in self.positions:
            if not 0 <= position < self.sequence_length:
                raise InputError(
                    f"capture position {position} is outside the replayed sequence",
                    details={"sequence_length": self.sequence_length},
                )

        self.measurements: list[Measurement] = []
        self.vectors: dict[tuple[str, int, str, int], np.ndarray[Any, Any]] = {}
        self.notes: list[str] = []
        self.elements_measured = 0
        self.elements_retained = 0
        self._skipped_shapes: set[tuple[str, tuple[int, ...]]] = set()

    # -- shape handling ----------------------------------------------------

    def sequence_dim(self, shape: tuple[int, ...]) -> int | None:
        length = self.sequence_length
        if len(shape) == 3 and shape[1] == length:
            return 1
        if len(shape) == 2:
            if shape[0] == length:
                return 0
            if shape[1] == length:
                return 1
        if len(shape) == 1 and shape[0] == length:
            return 0
        return None

    def skip(self, module: str, shape: tuple[int, ...], note: str) -> None:
        key = (module, shape)
        if key in self._skipped_shapes:
            return
        self._skipped_shapes.add(key)
        self.notes.append(f"{module}: {note}")

    # -- measurement -------------------------------------------------------

    def build(
        self,
        *,
        module: str,
        class_name: str,
        call_index: int,
        tensor_role: str,
        tensor_name: str,
        tensor: Any,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        if not tensor.is_floating_point():
            self.skip(
                module,
                tuple(int(d) for d in tensor.shape),
                "non-floating tensor was not measured",
            )
            return
        dim = self.sequence_dim(tuple(int(d) for d in tensor.shape))
        if dim is None:
            shape = tuple(int(d) for d in tensor.shape)
            self.skip(
                module,
                shape,
                (
                    "tensor does not expose an unambiguous sequence dimension of length "
                    f"{self.sequence_length}; observed shape {list(shape)}; capture skipped "
                    "(set other modules or use a single-token module)"
                ),
            )
            return

        dtype = torch_dtype_name(tensor.dtype)
        device = str(tensor.device)
        full_shape = [int(d) for d in tensor.shape]
        working = tensor.detach()

        for position in self.positions:
            index: list[Any] = [slice(None)] * working.dim()
            index[dim] = position
            sliced = working[tuple(index)]
            slice_shape = [int(d) for d in sliced.shape]
            array = torch_to_numpy(sliced, decode_bfloat16=True)
            numel = int(array.size)
            if self.elements_measured + numel > self.max_elements_total:
                raise ResourceLimitError(
                    "activation capture exceeds the configured total element budget",
                    details={
                        "requested": numel,
                        "already_measured": self.elements_measured,
                        "limit": self.max_elements_total,
                        "module": module,
                        "position": position,
                        "hint": "raise max_activation_elements_total or capture fewer positions",
                    },
                )
            self.elements_measured += numel
            stats = basic_stats(array, storage_dtype=dtype)
            flat = array.reshape(-1)
            notes: list[str] = []
            retained = False
            retained_shape: list[int] | None = None
            retained_numel = 0
            sampled = False
            stride: int | None = None
            vector: np.ndarray[Any, Any] | None = None

            if self.mode == "selected":
                if flat.size > self.max_elements_per_tensor:
                    stride = int(math.ceil(flat.size / self.max_elements_per_tensor))
                    flat = flat[::stride][: self.max_elements_per_tensor]
                    sampled = True
                    notes.append(
                        f"retained a deterministic stride sample (stride {stride}); "
                        "statistics use all values"
                    )
                if self.elements_retained + int(flat.size) > self.max_elements_total:
                    raise ResourceLimitError(
                        "retained activation values exceed the configured total element budget",
                        details={
                            "requested": int(flat.size),
                            "already_retained": self.elements_retained,
                            "limit": self.max_elements_total,
                            "module": module,
                            "position": position,
                        },
                    )
                vector = np.ascontiguousarray(flat, dtype=np.float64)
                self.elements_retained += int(vector.size)
                retained = True
                retained_shape = [int(d) for d in vector.shape]
                retained_numel = int(vector.size)
                self.vectors[(module, call_index, tensor_name, position)] = vector
            else:
                notes.append("summary capture: values were not retained")

            self.measurements.append(
                Measurement(
                    module=module,
                    class_name=class_name,
                    tensor_role=tensor_role,  # type: ignore[arg-type]
                    tensor_name=tensor_name,
                    call_index=call_index,
                    position=position,
                    shape=full_shape,
                    slice_shape=slice_shape,
                    dtype=dtype,
                    device=device,
                    stats=stats,
                    norm=_finite_norm(flat),
                    top_dimensions=_top_dimensions(array),
                    retained=retained,
                    retained_shape=retained_shape,
                    retained_numel=retained_numel,
                    sampled=sampled,
                    stride=stride,
                    notes=notes,
                )
            )

    # -- results -----------------------------------------------------------

    def ordered_measurements(self, module_order: Iterable[str] | None = None) -> list[Measurement]:
        order = {path: index for index, path in enumerate(module_order or ())}
        role_order = {"output": 0, "input": 1}
        return sorted(
            self.measurements,
            key=lambda item: (
                order.get(item.module, len(order)),
                item.call_index,
                role_order.get(item.tensor_role, 2),
                item.tensor_name,
                item.position,
            ),
        )
