"""Position-targeted forward-hook capture.

This reuses the V1 hook machinery (``_iter_tensors``, ``basic_stats``,
``torch_to_numpy``) but adds the two things V1.2 needs: several token positions
per forward pass, and a summary/selected split so a large model does not have to
materialise every hidden state.

Hooks never modify outputs, captured tensors are detached, and every hook is
removed on the way out, including when the forward pass raises.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import numpy as np
import torch

from wsp.activations.capture import _iter_tensors
from wsp.core.budget import Budget
from wsp.core.dtypes import torch_dtype_name, torch_to_numpy
from wsp.core.errors import InputError, ResourceLimitError
from wsp.core.models import StatsSummary
from wsp.core.time import utc_now_iso
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


class PositionedActivationCapture:
    """Capture selected token positions from selected modules.

    The object satisfies the capture protocol used by
    :func:`wsp.replay.engine.execute_replay`: ``install()`` before the forward
    pass, ``close()`` afterwards, and ``measurements`` as the result.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        module_paths: Iterable[str],
        positions: Iterable[int],
        *,
        sequence_length: int,
        mode: CaptureMode = "summary",
        record_inputs: bool = False,
        first_call_only: bool = True,
        max_elements_per_tensor: int | None = None,
        max_elements_total: int | None = None,
        budget: Budget | None = None,
    ) -> None:
        settings = budget or Budget()
        self.module = module
        self.module_paths = list(dict.fromkeys(str(path) for path in module_paths))
        self.sequence_length = int(sequence_length)
        self.mode: CaptureMode = mode
        self.record_inputs = bool(record_inputs)
        self.first_call_only = bool(first_call_only)
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

        named = dict(module.named_modules())
        self._modules: dict[str, torch.nn.Module] = {}
        for path in self.module_paths:
            key = path if path != "<root>" else ""
            if key not in named:
                raise InputError(
                    f"module {path!r} is not present in the model",
                    details={"hint": "call list_capture_points for valid paths"},
                )
            self._modules[path] = named[key]

        self.measurements: list[Measurement] = []
        self.vectors: dict[tuple[str, int, str, int], np.ndarray[Any, Any]] = {}
        self.notes: list[str] = []
        self.started_at = utc_now_iso()
        self.elements_measured = 0
        self.elements_retained = 0
        self._handles: list[Any] = []
        self._call_counts: dict[str, int] = {}
        self._skipped_shapes: set[tuple[str, tuple[int, ...]]] = set()
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "PositionedActivationCapture":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def install(self) -> None:
        if self._handles:
            return
        for path, target in self._modules.items():
            from wsp.investigation.points import _classify

            self._handles.append(
                target.register_forward_hook(self._make_hook(path, _classify(path, type(target).__name__)))
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    # -- hooks -------------------------------------------------------------

    def _make_hook(self, path: str, class_name: str) -> Callable[..., None]:
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            call_index = self._call_counts.get(path, 0)
            self._call_counts[path] = call_index + 1
            if self.first_call_only and call_index > 0:
                return
            if self.record_inputs:
                for argument, item in enumerate(inputs):
                    for tensor_name, tensor in _iter_tensors(item, f"input[{argument}]"):
                        self._capture(path, class_name, call_index, "input", tensor_name, tensor)
            for tensor_name, tensor in _iter_tensors(output, "output"):
                self._capture(path, class_name, call_index, "output", tensor_name, tensor)

        return hook

    def _sequence_dim(self, shape: tuple[int, ...]) -> int | None:
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

    def _capture(
        self,
        path: str,
        class_name: str,
        call_index: int,
        tensor_role: str,
        tensor_name: str,
        tensor: torch.Tensor,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        if not tensor.is_floating_point():
            self._skip(path, tuple(int(d) for d in tensor.shape), "non-floating tensor was not measured")
            return
        dim = self._sequence_dim(tuple(int(d) for d in tensor.shape))
        if dim is None:
            self._skip(
                path,
                tuple(int(d) for d in tensor.shape),
                (
                    "tensor does not expose an unambiguous sequence dimension of length "
                    f"{self.sequence_length}; capture skipped (set other modules or use a "
                    "single-token module)"
                ),
            )
            return

        dtype = torch_dtype_name(tensor.dtype)
        device = str(tensor.device)
        full_shape = [int(d) for d in tensor.shape]
        working = tensor.detach()

        for position in self.positions:
            index = [slice(None)] * working.dim()
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
                        "module": path,
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
                            "module": path,
                            "position": position,
                        },
                    )
                vector = np.ascontiguousarray(flat, dtype=np.float64)
                self.elements_retained += int(vector.size)
                retained = True
                retained_shape = [int(d) for d in vector.shape]
                retained_numel = int(vector.size)
                self.vectors[(path, call_index, tensor_name, position)] = vector
            else:
                notes.append("summary capture: values were not retained")

            self.measurements.append(
                Measurement(
                    module=path,
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

    def _skip(self, path: str, shape: tuple[int, ...], note: str) -> None:
        key = (path, shape)
        if key in self._skipped_shapes:
            return
        self._skipped_shapes.add(key)
        self.notes.append(f"{path}: {note}")

    # -- results -----------------------------------------------------------

    def ordered_measurements(self) -> list[Measurement]:
        order = {path: index for index, path in enumerate(self.module_paths)}
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
