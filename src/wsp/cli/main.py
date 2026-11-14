"""Command-line interface.  Every command returns JSON from the shared service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from wsp.core.budget import Budget
from wsp.core.errors import WspError
from wsp.core.json import to_jsonable
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace

app = typer.Typer(
    name="wsp",
    help="Weight-Space Programming Workbench: inspect, measure, and patch checkpoints.",
    no_args_is_help=True,
    add_completion=False,
)


def _emit(payload: Any, *, exit_code: int = 0) -> None:
    typer.echo(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False))
    if exit_code:
        raise typer.Exit(exit_code)


def _parse_scalar(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = [part for part in dotted.split(".") if part]
    if not parts:
        raise typer.BadParameter("parameter name cannot be empty")
    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            existing = {}
            cursor[part] = existing
        if not isinstance(existing, dict):
            raise typer.BadParameter(f"{part!r} is not an object")
        cursor = existing
    cursor[parts[-1]] = value


def _workspace(ctx: typer.Context, *, create: bool = False, name: str | None = None) -> Workspace | None:
    options = ctx.obj or {}
    raw = options.get("workspace")
    if raw:
        return Workspace(raw, create=create, name=name)
    if create:
        return Workspace(None, create=True, name=name)
    return None


def _budget(ctx: typer.Context) -> Budget:
    options = ctx.obj or {}
    raw = options.get("budget")
    if not raw:
        return Budget()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"--budget must be a JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise typer.BadParameter("--budget must be a JSON object")
    return Budget().merge(data)


@app.callback()
def main(
    ctx: typer.Context,
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory (default: $WSP_WORKSPACE or ./.wsp when initialized).",
    ),
    budget: str | None = typer.Option(
        None,
        "--budget",
        help="JSON object with Budget field overrides, e.g. '{\"max_preview_elements\":64}'.",
    ),
) -> None:
    ctx.obj = {"workspace": workspace, "budget": budget}


@app.command("init")
def init_workspace(
    ctx: typer.Context,
    path: Path | None = typer.Option(None, "--path", help="Workspace directory (default .wsp)."),
    name: str | None = typer.Option(None, "--name", help="Human-readable workspace name."),
) -> None:
    """Create a local experiment workspace (idempotent)."""
    try:
        if path is not None:
            workspace = Workspace(path, create=True, name=name)
        else:
            workspace = _workspace(ctx, create=True, name=name)
        assert workspace is not None
        _emit({"schema_version": "wsp.workspace/1.0", "ok": True, "workspace": workspace.status()})
    except WspError as exc:
        _emit({"schema_version": "wsp.workspace/1.0", "ok": False, "error": exc.to_dict()}, exit_code=1)


@app.command("tools")
def tools(
    ctx: typer.Context,
    operation: str | None = typer.Option(None, "--operation", help="Describe one operation only."),
) -> None:
    """Describe structured operations and their JSON schemas."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    try:
        _emit(service.describe(operation))
    except WspError as exc:
        _emit({"schema_version": "wsp.tool-registry/1.0", "ok": False, "error": exc.to_dict()}, exit_code=1)


@app.command("call")
def call_operation(
    ctx: typer.Context,
    operation: str = typer.Argument(..., help="Operation name, e.g. inspect_model."),
    input_json: str | None = typer.Option(None, "--json", "-j", help="Input payload as a JSON object."),
    input_file: Path | None = typer.Option(None, "--input", "-i", help="Read the input payload from a JSON file."),
    param: list[str] = typer.Option(
        [],
        "--param",
        "-p",
        help="Dotted input parameter, e.g. --param path=model.safetensors. Repeatable.",
    ),
) -> None:
    """Invoke a structured operation and print its JSON envelope."""
    payload: dict[str, Any] = {}
    if input_file is not None:
        try:
            loaded = json.loads(input_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            typer.echo(json.dumps({"schema_version": "wsp.tool/1.0", "ok": False, "operation": operation, "error": {"code": "invalid_input", "message": f"could not read input file: {exc}", "details": {}}}, indent=2))
            raise typer.Exit(1)
        if not isinstance(loaded, dict):
            typer.echo(json.dumps({"schema_version": "wsp.tool/1.0", "ok": False, "operation": operation, "error": {"code": "invalid_input", "message": "--input file must contain a JSON object", "details": {}}}, indent=2))
            raise typer.Exit(1)
        payload.update(loaded)
    if input_json:
        try:
            loaded = json.loads(input_json)
        except json.JSONDecodeError as exc:
            typer.echo(json.dumps({"schema_version": "wsp.tool/1.0", "ok": False, "operation": operation, "error": {"code": "invalid_input", "message": f"--json is not valid JSON: {exc}", "details": {}}}, indent=2))
            raise typer.Exit(1)
        if not isinstance(loaded, dict):
            typer.echo(json.dumps({"schema_version": "wsp.tool/1.0", "ok": False, "operation": operation, "error": {"code": "invalid_input", "message": "--json must be a JSON object", "details": {}}}, indent=2))
            raise typer.Exit(1)
        payload.update(loaded)
    for item in param:
        if "=" not in item:
            raise typer.BadParameter(f"--param must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_dotted(payload, key, _parse_scalar(value))
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(operation, payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("status")
def status(ctx: typer.Context) -> None:
    """Report workspace status."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("workspace_status", {})
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("inspect-model")
def inspect_model(
    ctx: typer.Context,
    path: Path = typer.Argument(..., help="Local SafeTensors checkpoint."),
    include_tensors: bool = typer.Option(True, "--tensors/--no-tensors"),
    file_sha256: bool = typer.Option(False, "--file-sha256", help="Compute the whole-file SHA-256."),
    save_manifest: bool = typer.Option(False, "--save-manifest"),
) -> None:
    """Inspect a checkpoint without materializing all tensors."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "inspect_model",
        {
            "path": str(path),
            "include_tensors": include_tensors,
            "compute_file_sha256": file_sha256,
            "save_manifest": save_manifest,
        },
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("list-tensors")
def list_tensors(
    ctx: typer.Context,
    path: Path = typer.Argument(...),
    name_contains: str | None = typer.Option(None, "--contains"),
    limit: int = typer.Option(100, "--limit"),
    offset: int = typer.Option(0, "--offset"),
) -> None:
    """List parameter metadata."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "list_tensors",
        {"path": str(path), "name_contains": name_contains, "limit": limit, "offset": offset},
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("inspect-tensor")
def inspect_tensor(
    ctx: typer.Context,
    path: Path = typer.Argument(...),
    tensor: str = typer.Argument(...),
    preview_elements: int = typer.Option(32, "--preview-elements"),
    fingerprint: bool = typer.Option(False, "--fingerprint"),
) -> None:
    """Inspect one bounded tensor preview and its statistics."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "inspect_tensor",
        {
            "path": str(path),
            "tensor": tensor,
            "preview_elements": preview_elements,
            "include_fingerprint": fingerprint,
        },
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


if __name__ == "__main__":  # pragma: no cover
    app()
