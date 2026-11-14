"""Platform and dependency provenance (no secrets, no telemetry)."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from typing import Any

from wsp import __version__


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_versions() -> dict[str, str]:
    versions = {
        "wsp-workbench": __version__,
        "python": platform.python_version(),
        "numpy": package_version("numpy") or "unknown",
        "safetensors": package_version("safetensors") or "unknown",
        "pydantic": package_version("pydantic") or "unknown",
        "typer": package_version("typer") or "unknown",
        "torch": package_version("torch") or "unknown",
    }
    return versions


def environment_info() -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda_available": cuda_available,
        "cuda_devices": devices,
        "deterministic_hardware": "cpu" if not cuda_available else "gpu_optional",
        "dependencies": dependency_versions(),
    }
