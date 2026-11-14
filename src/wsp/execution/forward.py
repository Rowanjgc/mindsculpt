"""Controlled, deterministic forward execution for compatible PyTorch models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from wsp.activations.capture import ActivationCapture, ActivationSummary, _iter_tensors
from wsp.core.budget import Budget
from wsp.core.dtypes import torch_dtype_name, torch_to_numpy
from wsp.core.errors import ExecutionError, InputError, ResourceLimitError
from wsp.core.models import StatsSummary
from wsp.core.time import utc_now_iso
from wsp.tensors.stats import basic_stats


class InputInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    shape: list[int] | None = None
    dtype: str | None = None
    device: str | None = None
    kind: str = "tensor"
    note: str | None = None


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    shape: list[int]
    dtype: str
    device: str
    numel: int
    stats: StatsSummary
    retained: bool = False
    retained_numel: int = 0
    sampled: bool = False
    stride: int | None = None
    values: list[Any] | None = None


class ForwardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.forward/1.0"
    run_id: str
    created_at: str
    model_class: str
    model_parameter_count: int
    device: str
    dtype: str
    seed: int | None = None
    inference_mode: bool = True
    training_restored: bool = True
    inputs: list[InputInfo] = Field(default_factory=list)
    outputs: list[OutputSpec] = Field(default_factory=list)
    activations: list[ActivationSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class ForwardExecution:
    result: ForwardResult
    output_values: dict[str, np.ndarray[Any, Any]]
    capture: ActivationCapture | None = None


def _move(value: Any, device: torch.device, dtype: torch.dtype | None) -> Any:
    if isinstance(value, torch.Tensor):
        moved = value.to(device)
        if dtype is not None and moved.is_floating_point():
            moved = moved.to(dtype)
        return moved
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).to(device)
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        return tensor
    if isinstance(value, Mapping):
        return {key: _move(item, device, dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device, dtype) for item in value)
    if isinstance(value, list):
        return [_move(item, device, dtype) for item in value]
    return value


def _input_infos(values: Mapping[str, Any]) -> list[InputInfo]:
    infos: list[InputInfo] = []
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            infos.append(
                InputInfo(
                    key=key,
                    shape=[int(d) for d in value.shape],
                    dtype=torch_dtype_name(value.dtype),
                    device=str(value.device),
                )
            )
        else:
            infos.append(InputInfo(key=key, kind=type(value).__name__))
    return infos


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _model_dtype(model: torch.nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _run_id(model_class: str, seed: int | None, device: str) -> str:
    import secrets

    from wsp.core.time import utc_now_compact

    return f"fw_{utc_now_compact()}_{secrets.token_hex(4)}"


def run_forward(
    model: torch.nn.Module,
    *,
    inputs: Mapping[str, Any] | None = None,
    input_args: Iterable[Any] = (),
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    seed: int | None = None,
    capture: ActivationCapture | None = None,
    budget: Budget | None = None,
    include_output_values: bool = False,
    output_preview_elements: int = 256,
) -> ForwardExecution:
    """Run one inference-mode forward pass without modifying model parameters."""

    budget = budget or Budget()
    warning_list: list[str] = []
    target_device = torch.device(device) if device is not None else _model_device(model)
    if isinstance(dtype, str):
        from wsp.core.dtypes import torch_dtype_from_name

        target_dtype: torch.dtype | None = torch_dtype_from_name(dtype)
    else:
        target_dtype = dtype
    if target_dtype is None:
        target_dtype = _model_dtype(model)
    kwargs = {key: _move(value, target_device, target_dtype) for key, value in (inputs or {}).items()}
    args = tuple(_move(value, target_device, target_dtype) for value in input_args)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    was_training = model.training
    model.eval()
    if capture is not None:
        capture.install()
    try:
        with torch.inference_mode():
            raw_output = model(*args, **kwargs)
    except Exception as exc:
        raise ExecutionError(
            f"forward pass failed: {exc}",
            details={"model_class": type(model).__name__},
        ) from exc
    finally:
        if capture is not None:
            capture.close()
        if was_training:
            model.train(was_training)

    tensor_outputs = list(_iter_tensors(raw_output, "output"))
    if not tensor_outputs:
        warning_list.append("model returned no tensor outputs")
    total_retained = 0
    values_by_key: dict[str, np.ndarray[Any, Any]] = {}
    output_specs: list[OutputSpec] = []
    for key, tensor in tensor_outputs:
        shape = [int(d) for d in tensor.shape]
        original_dtype = torch_dtype_name(tensor.dtype)
        array = torch_to_numpy(tensor.detach(), decode_bfloat16=True)
        stats = basic_stats(array, storage_dtype=original_dtype)
        flat = array.reshape(-1)
        stride: int | None = None
        sampled = False
        if flat.size > budget.max_forward_output_elements:
            stride = int(math.ceil(flat.size / budget.max_forward_output_elements))
            flat = flat[::stride][: budget.max_forward_output_elements]
            sampled = True
            warning_list.append(
                f"output {key!r} was retained as a deterministic stride sample"
            )
        total_retained += int(flat.size)
        if total_retained > budget.max_forward_output_elements:
            raise ResourceLimitError(
                "forward output exceeds the configured element budget",
                details={"limit": budget.max_forward_output_elements, "key": key},
            )
        values = flat.copy()
        values_by_key[key] = values
        preview: list[Any] | None = None
        if include_output_values:
            preview = [
                float(value) if isinstance(value, (np.floating, float)) and math.isfinite(float(value)) else None
                for value in values[:output_preview_elements]
            ]
        output_specs.append(
            OutputSpec(
                key=key,
                shape=shape,
                dtype=original_dtype,
                device=str(tensor.device),
                numel=int(array.size),
                stats=stats,
                retained=True,
                retained_numel=int(flat.size),
                sampled=sampled,
                stride=stride,
                values=preview,
            )
        )
    if target_device.type == "cuda":
        warning_list.append(
            "GPU execution may not be bitwise reproducible across hardware or software versions"
        )
    model_class = type(model).__name__
    result = ForwardResult(
        run_id=_run_id(model_class, seed, str(target_device)),
        created_at=utc_now_iso(),
        model_class=model_class,
        model_parameter_count=sum(p.numel() for p in model.parameters()),
        device=str(target_device),
        dtype=str(target_dtype).replace("torch.", ""),
        seed=seed,
        inputs=_input_infos(kwargs),
        outputs=output_specs,
        activations=capture.summaries() if capture is not None else [],
        warnings=warning_list,
    )
    return ForwardExecution(result=result, output_values=values_by_key, capture=capture)
