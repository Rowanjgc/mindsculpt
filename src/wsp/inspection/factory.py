"""Checkpoint opening helpers.

Only SafeTensors is opened automatically.  Pickle-based ``.bin``/``.pt``
checkpoints are deliberately not accepted: loading them can execute arbitrary
code.  A trusted Python factory may construct a module explicitly.
"""

from __future__ import annotations

from pathlib import Path

from wsp.core.errors import InputError, UnsupportedError
from wsp.inspection.base import CheckpointSource
from wsp.inspection.safetensors_source import SafeTensorsSource


def open_checkpoint(
    path: str | Path,
    *,
    compute_file_sha256: bool = False,
    display_name: str | None = None,
) -> CheckpointSource:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        found = sorted(item.name for item in candidate.glob("*.safetensors"))
        details: dict[str, object] = {"path": str(candidate)}
        message = (
            "checkpoint path is a directory; pass the .safetensors weights file "
            "inside it (for example model.safetensors)"
        )
        if found:
            details["safetensors_files"] = found
            message += f"; this directory contains: {', '.join(found)}"
        raise InputError(message, details=details)
    if candidate.suffix.lower() != ".safetensors":
        raise UnsupportedError(
            "only .safetensors checkpoints are opened automatically; "
            "pickle-based checkpoints are not supported for safety",
            details={"path": str(candidate)},
        )
    if not candidate.is_file():
        raise InputError(
            f"checkpoint file does not exist: {candidate}",
            details={"path": str(candidate)},
        )
    return SafeTensorsSource(
        candidate, compute_file_sha256=compute_file_sha256, display_name=display_name
    )
