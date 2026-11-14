"""Explicit model-loading helpers.

Loading a checkpoint into a live module necessarily materializes the model.
These helpers enforce a whole-model element budget and never execute code from
the checkpoint.  They are separate from header-only checkpoint inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wsp.core.budget import Budget
from wsp.core.dtypes import numpy_to_torch
from wsp.core.errors import ResourceLimitError, SafetyError
from wsp.inspection.safetensors_source import SafeTensorsSource
from wsp.inspection.torch_module_source import TorchModuleSource


def load_state_dict_into_module(
    module: Any,
    checkpoint_path: str | Path,
    *,
    budget: Budget | None = None,
    strict: bool = True,
) -> TorchModuleSource:
    """Load a local SafeTensors checkpoint into an already-constructed module."""

    budget = budget or Budget()
    path = Path(checkpoint_path).expanduser()
    if path.suffix.lower() != ".safetensors":
        raise SafetyError(
            "only SafeTensors checkpoints can be loaded; pickle-based loading is not supported"
        )
    with SafeTensorsSource(path, compute_file_sha256=True) as source:
        total = sum(info.numel for info in source.tensor_infos())
        if total > budget.max_model_load_elements:
            raise ResourceLimitError(
                "checkpoint exceeds the whole-model load budget",
                details={
                    "parameter_count": total,
                    "limit": budget.max_model_load_elements,
                },
            )
        state: dict[str, Any] = {}
        for info in source.tensor_infos():
            if info.numel > budget.max_tensor_numel:
                raise ResourceLimitError(
                    f"tensor {info.name!r} exceeds the per-tensor load budget",
                    details={"numel": info.numel, "limit": budget.max_tensor_numel},
                )
            array = source.read_full(info.name, max_elements=budget.max_tensor_numel)
            state[info.name] = numpy_to_torch(array, info.dtype)
        module.load_state_dict(state, strict=strict)
    return TorchModuleSource(module)
