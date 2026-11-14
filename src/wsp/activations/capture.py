"""Bounded forward-hook activation capture."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from wsp.core.budget import Budget
from wsp.core.dtypes import torch_dtype_name, torch_to_numpy
from wsp.core.errors import InputError, InternalError, ResourceLimitError
from wsp.core.models import StatsSummary
from wsp.core.time import utc_now_iso
from wsp.tensors.stats import basic_stats


class ActivationSummary(BaseModel):
    """A structured summary of one captured module input or output."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.activation/1.0"
    capture_id: str
    module: str
    kind: str
    call_index: int
    original_dtype: str
    device: str
    shape: list[int]
    numel: int
    stats: StatsSummary
    retained: bool = False
    retained_shape: list[int] | None = None
    retained_numel: int = 0
    sampled: bool = False
    stride: int | None = None
    token_position: int | None = None
    artifact_id: str | None = None
    values: list[Any] | None = None
    notes: list[str] = Field(default_factory=list)


@dataclass
class ActivationRecord:
    module: str
    kind: str
    call_index: int
    original_dtype: str
    device: str
    shape: list[int]
    stats: StatsSummary
    values: np.ndarray[Any, Any] | None
    token_position: int | None = None
    sampled: bool = False
    stride: int | None = None
    artifact_id: str | None = None
    notes: list[str] = field(default_factory=list)
    _capture_id: str = ""

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def retained_numel(self) -> int:
        return int(self.values.size) if self.values is not None else 0

    def summary(self, *, include_values: bool = False) -> ActivationSummary:
        return ActivationSummary(
            capture_id=self._capture_id,
            module=self.module,
            kind=self.kind,
            call_index=self.call_index,
            original_dtype=self.original_dtype,
            device=self.device,
            shape=self.shape,
            numel=self.numel,
            stats=self.stats,
            retained=self.values is not None,
            retained_shape=list(self.values.shape) if self.values is not None else None,
            retained_numel=self.retained_numel,
            sampled=self.sampled,
            stride=self.stride,
            token_position=self.token_position,
            artifact_id=self.artifact_id,
            values=[_json_value(value) for value in self.values.ravel()] if include_values and self.values is not None else None,
            notes=list(self.notes),
        )


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _iter_tensors(value: Any, prefix: str = "output") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{prefix}.{key}")
    else:
        # Hugging Face ModelOutput objects behave like mappings.
        keys = getattr(value, "keys", None)
        if callable(keys):
            try:
                for key in keys():
                    yield from _iter_tensors(value[key], f"{prefix}.{key}")
            except (KeyError, TypeError):
                return


class ActivationCapture:
    """Attach bounded forward hooks to named modules.

    The object is a context manager.  Hooks are always removed on exit, including
    when the forward pass raises.  Captured tensors are detached and copied to
    CPU; they never participate in autograd.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        module_names: Iterable[str] | None = None,
        *,
        record_inputs: bool = False,
        max_elements_per_tensor: int | None = None,
        max_elements_total: int | None = None,
        token_position: int | None = None,
        budget: Budget | None = None,
        include_values: bool = True,
    ) -> None:
        self.module = module
        self.budget = budget or Budget()
        self.module_names = list(module_names) if module_names is not None else ["<root>"]
        self.record_inputs = bool(record_inputs)
        self.max_elements_per_tensor = int(
            max_elements_per_tensor or self.budget.max_activation_elements_per_tensor
        )
        self.max_elements_total = int(
            max_elements_total or self.budget.max_activation_elements_total
        )
        if self.max_elements_per_tensor <= 0 or self.max_elements_total <= 0:
            raise InputError("activation element limits must be positive")
        self.token_position = token_position
        self.include_values = include_values
        self.capture_id = _capture_id(module_names=self.module_names, token_position=token_position)
        self.records: list[ActivationRecord] = []
        self._handles: list[Any] = []
        self._call_counts: dict[str, int] = {}
        self._retained_elements = 0
        self._modules: dict[str, torch.nn.Module] = {}
        self._closed = False
        self.started_at = utc_now_iso()

        named = dict(module.named_modules())
        if "<root>" in self.module_names:
            self._modules["<root>"] = module
        for name in self.module_names:
            if name == "<root>":
                continue
            if name not in named:
                raise InputError(
                    f"module {name!r} is not present in the model",
                    details={"module": name, "available_count": len(named)},
                )
            self._modules[name] = named[name]

    def __enter__(self) -> "ActivationCapture":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def install(self) -> None:
        if self._handles:
            return
        for name in self.module_names:
            target = self._modules[name]
            self._handles.append(target.register_forward_hook(self._make_hook(name)))

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

    def clear(self) -> None:
        self.records.clear()
        self._retained_elements = 0

    def _make_hook(self, name: str) -> Callable[..., None]:
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            index = self._call_counts.get(name, 0)
            self._call_counts[name] = index + 1
            if self.record_inputs:
                for position, item in enumerate(inputs):
                    for kind, tensor in _iter_tensors(item, f"input[{position}]"):
                        self._capture(name, kind, index, tensor)
            for kind, tensor in _iter_tensors(output, "output"):
                self._capture(name, kind, index, tensor)

        return hook

    def _capture(self, name: str, kind: str, call_index: int, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        original_shape = [int(d) for d in tensor.shape]
        original_dtype = torch_dtype_name(tensor.dtype)
        device = str(tensor.device)
        working = tensor.detach()
        token_position = self.token_position
        if token_position is not None:
            if working.dim() == 3:
                if not -working.shape[1] <= token_position < working.shape[1]:
                    raise InputError(
                        f"token position {token_position} is outside the captured sequence",
                        details={"module": name, "sequence_length": int(working.shape[1])},
                    )
                working = working[:, token_position, :]
            elif working.dim() == 2:
                if not -working.shape[0] <= token_position < working.shape[0]:
                    raise InputError(
                        f"token position {token_position} is outside the captured sequence",
                        details={"module": name, "sequence_length": int(working.shape[0])},
                    )
                working = working[token_position, :]
            else:
                raise InputError(
                    "token_position capture is supported only for 2-D or 3-D tensors",
                    details={"module": name, "shape": original_shape},
                )
        captured_np = torch_to_numpy(working, decode_bfloat16=True)
        stats = basic_stats(captured_np, storage_dtype=original_dtype)
        notes: list[str] = []
        sampled = False
        stride: int | None = None
        flat = captured_np.reshape(-1)
        if not self.include_values:
            notes.append("values were not retained; this is a summary-only capture")
            retained_numel = 0
            values = None
        else:
            if flat.size > self.max_elements_per_tensor:
                stride = int(math.ceil(flat.size / self.max_elements_per_tensor))
                flat = flat[::stride][: self.max_elements_per_tensor]
                sampled = True
                notes.append(
                    f"retained a deterministic stride sample (stride {stride}); statistics use all values"
                )
            retained_numel = int(flat.size)
            if self._retained_elements + retained_numel > self.max_elements_total:
                raise ResourceLimitError(
                    "activation capture exceeds the configured total element budget",
                    details={
                        "requested": retained_numel,
                        "already_retained": self._retained_elements,
                        "limit": self.max_elements_total,
                        "module": name,
                        "kind": kind,
                    },
                )
            values = flat.copy()
            self._retained_elements += retained_numel
        record = ActivationRecord(
            module=name,
            kind=kind,
            call_index=call_index,
            original_dtype=original_dtype,
            device=device,
            shape=original_shape,
            stats=stats,
            values=values,
            token_position=self.token_position,
            sampled=sampled,
            stride=stride,
            notes=notes,
            _capture_id=self.capture_id,
        )
        self.records.append(record)

    def summaries(self, *, include_values: bool = False) -> list[ActivationSummary]:
        return [record.summary(include_values=include_values) for record in self.records]


def _capture_id(*, module_names: list[str], token_position: int | None) -> str:
    from wsp.core.hashing import sha256_bytes
    from wsp.core.json import canonical_bytes

    payload = {
        "schema": "wsp.activation/1.0",
        "modules": module_names,
        "token_position": token_position,
    }
    return "ac_" + sha256_bytes(canonical_bytes(payload))[len("sha256:") :][:16]
