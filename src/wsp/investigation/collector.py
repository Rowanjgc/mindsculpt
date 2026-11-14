"""Position-targeted forward-hook capture.

This reuses the V1 hook machinery (``_iter_tensors``, ``basic_stats``,
``torch_to_numpy``) but adds the two things V1.2 needs: several token positions
per forward pass, and a summary/selected split so a large model does not have to
materialise every hidden state.

Hooks never modify outputs, captured tensors are detached, and every hook is
removed on the way out, including when the forward pass raises.

Measurement construction itself lives in :mod:`wsp.investigation.measure`, so the
optional NNsight/nnterp backend produces byte-identical measurement semantics.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import torch

from wsp.activations.capture import _iter_tensors
from wsp.core.budget import Budget
from wsp.core.errors import InputError
from wsp.investigation.measure import (
    MeasurementBuilder,
    _finite_norm,
    _json_float,
    _top_dimensions,
)
from wsp.investigation.models import CaptureMode, Measurement

__all__ = ["PositionedActivationCapture", "_finite_norm", "_json_float", "_top_dimensions"]


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
        self.module = module
        self.module_paths = list(dict.fromkeys(str(path) for path in module_paths))
        self.sequence_length = int(sequence_length)
        self.mode: CaptureMode = mode
        self.record_inputs = bool(record_inputs)
        self.first_call_only = bool(first_call_only)
        self.builder = MeasurementBuilder(
            sequence_length=self.sequence_length,
            positions=positions,
            mode=mode,
            max_elements_per_tensor=max_elements_per_tensor,
            max_elements_total=max_elements_total,
            budget=budget or Budget(),
        )
        self.positions = self.builder.positions

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

        self._handles: list[Any] = []
        self._call_counts: dict[str, int] = {}
        self._closed = False

    # -- delegated measurement state ---------------------------------------

    @property
    def measurements(self) -> list[Measurement]:
        return self.builder.measurements

    @property
    def vectors(self) -> dict[tuple[str, int, str, int], np.ndarray[Any, Any]]:
        return self.builder.vectors

    @property
    def notes(self) -> list[str]:
        return self.builder.notes

    @property
    def elements_measured(self) -> int:
        return self.builder.elements_measured

    @property
    def elements_retained(self) -> int:
        return self.builder.elements_retained

    @property
    def max_elements_per_tensor(self) -> int:
        return self.builder.max_elements_per_tensor

    @property
    def max_elements_total(self) -> int:
        return self.builder.max_elements_total

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
                        self._record(path, class_name, call_index, "input", tensor_name, tensor)
            for tensor_name, tensor in _iter_tensors(output, "output"):
                self._record(path, class_name, call_index, "output", tensor_name, tensor)

        return hook

    def _record(
        self,
        path: str,
        class_name: str,
        call_index: int,
        tensor_role: str,
        tensor_name: str,
        tensor: torch.Tensor,
    ) -> None:
        self.builder.build(
            module=path,
            class_name=class_name,
            call_index=call_index,
            tensor_role=tensor_role,
            tensor_name=tensor_name,
            tensor=tensor,
        )

    # -- results -----------------------------------------------------------

    def ordered_measurements(self) -> list[Measurement]:
        return self.builder.ordered_measurements(self.module_paths)
