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


@app.command("mcp")
def mcp(
    ctx: typer.Context,
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory (overrides the global --workspace option).",
    ),
    budget: str | None = typer.Option(
        None,
        "--budget",
        help="Budget JSON object (overrides the global --budget option).",
    ),
    auto_init: bool = typer.Option(
        False,
        "--auto-init",
        help="Create and initialize the workspace if it does not exist yet.",
    ),
) -> None:
    """Serve the tool registry over the MCP stdio transport (optional 'mcp' extra)."""
    try:
        from wsp.mcp.server import run_stdio
    except ModuleNotFoundError as exc:
        if exc.name != "mcp":
            raise
        _emit(
            {
                "schema_version": "wsp.mcp/1.0",
                "ok": False,
                "error": {
                    "code": "unsupported",
                    "message": (
                        "the MCP server requires the optional 'mcp' dependency; "
                        'install it with: pip install -e ".[mcp]"'
                    ),
                    "details": {},
                },
            },
            exit_code=1,
        )
    options = dict(ctx.obj or {})
    if workspace is not None:
        options["workspace"] = workspace
    if budget is not None:
        options["budget"] = budget
    ctx.obj = options
    try:
        service = ToolService(_workspace(ctx, create=auto_init), budget=_budget(ctx))
    except WspError as exc:
        # Fail with a diagnosable message instead of a traceback: MCP clients
        # only see "Connection closed" when the server dies at startup.
        typer.echo(
            json.dumps(
                {
                    "schema_version": "wsp.mcp/1.0",
                    "ok": False,
                    "error": exc.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            err=True,
        )
        raise typer.Exit(1) from exc
    run_stdio(service)


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


@app.command("record-trace")
def record_trace(
    ctx: typer.Context,
    input_text: str | None = typer.Argument(None, help="Exact prompt text."),
    output_text: str | None = typer.Option(None, "--output", help="Deterministic scripted response text."),
    model_name: str = typer.Option("scripted/fixture", "--model"),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens"),
    input_json: str | None = typer.Option(None, "--json", help="Full RecordTraceInput payload."),
) -> None:
    """Store a deterministic scripted trace without loading a model."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        if input_text is None or output_text is None:
            raise typer.BadParameter("provide INPUT_TEXT and --output, or --json")
        payload = {
            "input_text": input_text,
            "params": {"max_new_tokens": max_new_tokens},
            "backend": {
                "kind": "scripted",
                "scripted": {
                    "model_name": model_name,
                    "default_fixture": {"output_text": output_text, "termination": "eos"},
                },
            },
        }
    envelope = service.call("record_trace", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("traces")
def list_traces(
    ctx: typer.Context,
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """List stored behavioral traces."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("list_traces", {"limit": limit})
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("inspect-trace")
def inspect_trace(
    ctx: typer.Context,
    trace_id: str = typer.Argument(...),
    tokens: bool = typer.Option(True, "--tokens/--no-tokens"),
    token_limit: int = typer.Option(64, "--token-limit"),
    metadata: bool = typer.Option(False, "--metadata"),
) -> None:
    """Inspect one trace with explicit unavailable fields."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "inspect_trace",
        {
            "trace_id": trace_id,
            "include_tokens": tokens,
            "token_limit": token_limit,
            "include_metadata": metadata,
        },
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("scan-trace")
def scan_trace(
    ctx: typer.Context,
    trace_id: str = typer.Argument(...),
    detector: list[str] = typer.Option([], "--detector", help="Detector id; repeatable. Defaults to all enabled."),
) -> None:
    """Run deterministic behavioral detectors over a trace."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = {
        "trace_id": trace_id,
        "detectors": [{"detector_id": item} for item in detector],
    }
    envelope = service.call("scan_trace", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("compare-traces")
def compare_traces(
    ctx: typer.Context,
    left: str = typer.Argument(...),
    right: str = typer.Argument(...),
) -> None:
    """Structurally compare two stored traces."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("compare_traces", {"left_trace_id": left, "right_trace_id": right})
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("contrast")
def run_contrast(
    ctx: typer.Context,
    input_file: Path = typer.Option(..., "--input", "-i", help="RunContrastInput JSON file."),
) -> None:
    """Execute a paired comparison with a declared invariant."""
    try:
        payload = json.loads(input_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        typer.echo(json.dumps({"schema_version": "wsp.tool/1.0", "ok": False, "operation": "run_contrast", "error": {"code": "invalid_input", "message": str(exc), "details": {}}}, indent=2))
        raise typer.Exit(1)
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("run_contrast", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("detectors")
def list_detectors(ctx: typer.Context) -> None:
    """Describe detector identifiers, parameters, and limitations."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("list_detectors", {})
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("evaluators")
def list_evaluators(ctx: typer.Context) -> None:
    """Describe evaluator identifiers and parameters."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("list_evaluators", {})
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("cases")
def list_cases(
    ctx: typer.Context,
    status: str | None = typer.Option(None, "--status"),
    tag: str | None = typer.Option(None, "--tag"),
    limit: int = typer.Option(100, "--limit"),
) -> None:
    """List behavioral investigation cases."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "list_failure_cases", {"status": status, "tag": tag, "limit": limit}
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("inspect-case")
def inspect_case(ctx: typer.Context, case_id: str = typer.Argument(...)) -> None:
    """Load a case and its evidence references."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("inspect_failure_case", {"case_id": case_id})
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("export-case")
def export_case(
    ctx: typer.Context,
    case_id: str = typer.Argument(...),
    output: Path = typer.Option(..., "--output"),
    include_traces: bool = typer.Option(True, "--traces/--no-traces"),
    redact: bool = typer.Option(False, "--redact"),
    sensitive_marker: list[str] = typer.Option(
        [],
        "--sensitive-marker",
        help="Extra literal string that must not appear in the export (repeatable).",
    ),
    keep_token_ids: bool = typer.Option(
        False,
        "--keep-token-ids",
        help="Redacted exports keep token ids, enabling reduced replay capability.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Write a self-contained case export bundle."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "export_failure_case",
        {
            "case_id": case_id,
            "output_path": str(output),
            "include_traces": include_traces,
            "overwrite": overwrite,
            "policy": {
                "mode": "redacted" if redact else "private",
                "sensitive_markers": list(sensitive_marker),
                "keep_token_ids": keep_token_ids,
            },
        },
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("import-case")
def import_case(
    ctx: typer.Context,
    path: Path = typer.Argument(...),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Reload a case export, verifying its checksums."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "import_failure_case", {"path": str(path), "overwrite": overwrite}
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)



@app.command("replay-plan")
def replay_plan(
    ctx: typer.Context,
    trace_id: str = typer.Argument(..., help="Stored trace to replay."),
    max_sequence_length: int = typer.Option(2048, "--max-sequence-length"),
    atol: float = typer.Option(0.05, "--logprob-atol"),
    no_save: bool = typer.Option(False, "--no-save"),
    input_json: str | None = typer.Option(None, "--json", help="Full PrepareReplayInput payload."),
) -> None:
    """Turn a trace into an explicit teacher-forced replay plan."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {
            "trace_id": trace_id,
            "execution": {"max_sequence_length": max_sequence_length},
            "tolerances": {"logprob_atol": atol},
            "save": not no_save,
        }
    envelope = service.call("prepare_replay", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("replay")
def replay(
    ctx: typer.Context,
    trace_id: str | None = typer.Argument(None, help="Stored trace to replay."),
    plan: str | None = typer.Option(None, "--plan", help="Stored replay plan artifact id."),
    model: str = typer.Option(
        '{"kind": "synthetic"}',
        "--model",
        help="Model source JSON, for example {\"kind\": \"synthetic\"} or a local_hf path.",
    ),
    full_tokens: bool = typer.Option(
        False, "--tokens/--no-tokens", help="Include per-token agreement rows."
    ),
    no_save: bool = typer.Option(False, "--no-save"),
    input_json: str | None = typer.Option(None, "--json", help="Full VerifyReplayInput payload."),
) -> None:
    """Execute a teacher-forced replay and report measured agreement."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        if trace_id is None and plan is None:
            raise typer.BadParameter("provide TRACE_ID or --plan")
        payload = {
            "trace_id": trace_id,
            "plan_artifact_id": plan,
            "model": json.loads(model),
            "include_per_token": full_tokens,
            "save": not no_save,
        }
    envelope = service.call("verify_replay", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("capture-points")
def capture_points(
    ctx: typer.Context,
    model: str = typer.Option(
        '{"kind": "synthetic"}',
        "--model",
        help="Model source JSON, for example {\"kind\": \"synthetic\"} or a local_hf path.",
    ),
    kind: list[str] | None = typer.Option(None, "--kind", help="Restrict to an architectural role."),
    limit: int = typer.Option(200, "--limit"),
    instrumentation: str = typer.Option(
        "torch_hooks",
        "--instrumentation",
        help="torch_hooks (default, always available) or nnterp (optional dependency).",
    ),
    input_json: str | None = typer.Option(None, "--json", help="Full ListCapturePointsInput payload."),
) -> None:
    """List module paths in a local model that can be targeted by a capture."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {
            "model": json.loads(model),
            "limit": limit,
            "instrumentation": instrumentation,
        }
        if kind:
            payload["kinds"] = list(kind)
    envelope = service.call("list_capture_points", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("capture")
def capture(
    ctx: typer.Context,
    trace_id: str = typer.Argument(..., help="Stored trace to replay and measure."),
    module: list[str] = typer.Option(..., "--module", help="Module path; repeatable."),
    position: list[int] | None = typer.Option(
        None, "--position", help="Full-sequence token index; repeatable."
    ),
    span: str | None = typer.Option(None, "--span", help="Stored span resolution artifact id."),
    mode: str = typer.Option("summary", "--mode", help="summary or selected."),
    semantics: str = typer.Option(
        "consuming", "--semantics", help="consuming or predicting."
    ),
    model: str = typer.Option('{"kind": "synthetic"}', "--model"),
    values: bool = typer.Option(False, "--values/--no-values", help="Echo retained values."),
    instrumentation: str = typer.Option(
        "torch_hooks",
        "--instrumentation",
        help="torch_hooks (default, always available) or nnterp (optional dependency).",
    ),
    no_save: bool = typer.Option(False, "--no-save"),
    input_json: str | None = typer.Option(
        None, "--json", help="Full CaptureTraceActivationsInput payload."
    ),
) -> None:
    """Replay a trace and measure selected internal states."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {
            "trace_id": trace_id,
            "model": json.loads(model),
            "modules": list(module),
            "mode": mode,
            "position_semantics": semantics,
            "include_values": values,
            "instrumentation": instrumentation,
            "save": not no_save,
        }
        if span:
            payload["span_artifact_id"] = span
        elif position:
            payload["positions"] = list(position)
        else:
            raise typer.BadParameter("provide --position or --span")
    envelope = service.call("capture_trace_activations", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("align")
def align_tokens(
    ctx: typer.Context,
    trace_id: str = typer.Argument(..., help="Stored trace to align."),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model source JSON; needed for tokenizer offsets and the special-token set.",
    ),
    limit: int = typer.Option(64, "--limit", help="Positions to echo."),
    no_positions: bool = typer.Option(False, "--no-positions"),
    input_json: str | None = typer.Option(
        None, "--json", help="Full AlignTraceTokensInput payload."
    ),
) -> None:
    """Map a trace's tokens onto forward positions."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {
            "trace_id": trace_id,
            "include_positions": not no_positions,
            "position_limit": limit,
        }
        if model:
            payload["model"] = json.loads(model)
    envelope = service.call("align_trace_tokens", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("resolve-span")
def resolve_span_command(
    ctx: typer.Context,
    trace_id: str = typer.Argument(..., help="Stored trace the span belongs to."),
    kind: str = typer.Option(
        "token_range",
        "--kind",
        help="token_range, text_span, word_range, or finding_evidence.",
    ),
    role: str = typer.Option("generation", "--role", help="prompt or generation."),
    start: int | None = typer.Option(None, "--start"),
    end: int | None = typer.Option(None, "--end"),
    alignment: str | None = typer.Option(None, "--alignment", help="Stored alignment artifact id."),
    detail: str | None = typer.Option(
        None, "--detail", help="JSON EvidenceRef for kind=finding_evidence."
    ),
    model: str | None = typer.Option(None, "--model"),
    input_json: str | None = typer.Option(
        None, "--json", help="Full ResolveFindingSpanInput payload."
    ),
) -> None:
    """Resolve a behavioral span onto token positions."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {"trace_id": trace_id, "kind": kind}
        if detail:
            payload["detail"] = json.loads(detail)
        else:
            if start is None or end is None:
                raise typer.BadParameter("provide --start and --end, or --detail")
            payload["role"] = role
            payload["start"] = start
            payload["end"] = end
        if alignment:
            payload["alignment_artifact_id"] = alignment
        if model:
            payload["model"] = json.loads(model)
    envelope = service.call("resolve_finding_span", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("compare-captures")
def compare_captures(
    ctx: typer.Context,
    left: str = typer.Argument(..., help="Left capture artifact id."),
    right: str = typer.Argument(..., help="Right capture artifact id."),
    metric: list[str] | None = typer.Option(
        None, "--metric", help="l2, rel_l2, cosine, dot, max_abs, mean_abs, rms, projection."
    ),
    module: list[str] | None = typer.Option(None, "--module", help="Restrict to module paths."),
    alignment: str = typer.Option(
        "identical_positions",
        "--alignment",
        help="identical_positions, paired_positions, matching_prefix, or semantic_event.",
    ),
    pair: list[str] | None = typer.Option(
        None,
        "--pair",
        help="Explicit LEFT:RIGHT position pair; requires --reviewed.",
    ),
    reviewed: bool = typer.Option(
        False,
        "--reviewed",
        help="Confirm that explicit pairs or a semantic alignment were reviewed by a human.",
    ),
    direction: str | None = typer.Option(None, "--direction", help="JSON list of floats."),
    allow_different_models: bool = typer.Option(False, "--allow-different-models"),
    input_json: str | None = typer.Option(
        None, "--json", help="Full CompareActivationCapturesInput payload."
    ),
) -> None:
    """Compare two stored activation captures numerically."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {
            "left_capture_artifact_id": left,
            "right_capture_artifact_id": right,
            "alignment": {"strategy": alignment, "reviewed": reviewed},
            "allow_different_models": allow_different_models,
        }
        if metric:
            payload["metrics"] = list(metric)
        if module:
            payload["modules"] = list(module)
        if pair:
            pairs = []
            for item in pair:
                left_position, _, right_position = item.partition(":")
                pairs.append(
                    {
                        "left_position": int(left_position),
                        "right_position": int(right_position),
                    }
                )
            payload["alignment"]["pairs"] = pairs
        if direction:
            payload["direction"] = json.loads(direction)
    envelope = service.call("compare_activation_captures", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("attach-evidence")
def attach_evidence(
    ctx: typer.Context,
    title: str = typer.Option(..., "--title"),
    case_id: str | None = typer.Option(None, "--case", help="Attach the evidence to a case."),
    summary: str | None = typer.Option(None, "--summary"),
    observation: str | None = typer.Option(None, "--observation", help="Observed trace id."),
    control: list[str] | None = typer.Option(None, "--control", help="Control trace id; repeatable."),
    capture: list[str] | None = typer.Option(None, "--capture", help="Capture artifact id; repeatable."),
    comparison: list[str] | None = typer.Option(
        None, "--comparison", help="Comparison artifact id; repeatable."
    ),
    replay: list[str] | None = typer.Option(None, "--replay", help="Replay artifact id; repeatable."),
    span: list[str] | None = typer.Option(None, "--span", help="Span artifact id; repeatable."),
    hypothesis: list[str] | None = typer.Option(
        None, "--hypothesis", help="Statement of an unverified hypothesis; repeatable."
    ),
    note: list[str] | None = typer.Option(None, "--note", help="Free note; repeatable."),
    input_json: str | None = typer.Option(
        None, "--json", help="Full AttachInvestigationEvidenceInput payload."
    ),
) -> None:
    """Record internal measurements as investigation evidence."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {
            "title": title,
            "control_trace_ids": list(control or []),
            "capture_artifact_ids": list(capture or []),
            "comparison_artifact_ids": list(comparison or []),
            "replay_artifact_ids": list(replay or []),
            "span_artifact_ids": list(span or []),
            "hypotheses": [{"statement": item} for item in hypothesis or []],
            "notes": list(note or []),
        }
        if case_id:
            payload["case_id"] = case_id
        if summary:
            payload["summary"] = summary
        if observation:
            payload["observation_trace_id"] = observation
    envelope = service.call("attach_investigation_evidence", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("inspect-evidence")
def inspect_evidence(
    ctx: typer.Context,
    artifact_id: str = typer.Argument(..., help="Investigation evidence artifact id."),
    no_records: bool = typer.Option(False, "--no-records", help="Omit referenced record digests."),
    input_json: str | None = typer.Option(
        None, "--json", help="Full InspectInvestigationEvidenceInput payload."
    ),
) -> None:
    """Read stored investigation evidence and its integrity status."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {"artifact_id": artifact_id, "include_records": not no_records}
    envelope = service.call("inspect_investigation_evidence", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


def _token_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(item) for item in raw.split(",") if item.strip()]


def _execution_payload(
    *,
    intervention: str | None,
    input_json: str | None,
    source: str | None,
    tokens: str | None,
    trace: str | None,
) -> dict[str, Any]:
    if input_json:
        return json.loads(input_json)
    payload: dict[str, Any] = {}
    if intervention:
        payload["intervention_artifact_id"] = intervention
    if source:
        payload["source"] = json.loads(source)
    token_ids = _token_ids(tokens)
    if token_ids is not None:
        payload["token_ids"] = token_ids
    if trace:
        payload["trace_id"] = trace
    return payload


@app.command("create-activation-intervention")
def create_activation_intervention(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name"),
    capture: str | None = typer.Option(
        None, "--capture", help="Capture artifact the target activation was measured in."
    ),
    module: str = typer.Option(..., "--module", help="Module path, e.g. blocks.1."),
    operation: str = typer.Option(
        "identity", "--operation", help="identity, zero, mean, or patch."
    ),
    position: list[int] | None = typer.Option(
        None, "--position", help="Sequence position to intervene on; repeatable."
    ),
    semantics: str = typer.Option(
        "consuming", "--semantics", help="consuming or predicting."
    ),
    role: str = typer.Option("output", "--role", help="input or output."),
    axis: int | None = typer.Option(None, "--axis", help="Tensor axis holding sequence positions."),
    feature_start: int | None = typer.Option(None, "--feature-start"),
    feature_end: int | None = typer.Option(None, "--feature-end"),
    feature_index: list[int] | None = typer.Option(
        None, "--feature-index", help="Explicit feature index; repeatable."
    ),
    tensor_name: str | None = typer.Option(None, "--tensor-name"),
    tensor_index: int | None = typer.Option(None, "--tensor-index"),
    expected_dtype: str | None = typer.Option(None, "--expected-dtype"),
    apply_on_call: list[int] | None = typer.Option(
        None, "--apply-on-call", help="Forward call index; repeatable."
    ),
    allow_dtype_cast: bool = typer.Option(False, "--allow-dtype-cast"),
    donor_capture: str | None = typer.Option(
        None, "--donor-capture", help="Captured donor activation artifact."
    ),
    donor_trace: str | None = typer.Option(None, "--donor-trace"),
    donor_position: list[int] | None = typer.Option(
        None, "--donor-position", help="Donor position; repeatable."
    ),
    donor_condition: str = typer.Option(
        "healthy-control", "--donor-condition", help="healthy-control, alternative-input, or other."
    ),
    donor_values: str | None = typer.Option(None, "--donor-values", help="JSON list of floats."),
    mean_source_capture: str | None = typer.Option(
        None, "--mean-source-capture", help="Capture artifact averaged for mean ablation."
    ),
    mean_position: list[int] | None = typer.Option(
        None, "--mean-position", help="Position used for the mean; repeatable."
    ),
    mean_alignment: str | None = typer.Option(
        None, "--mean-alignment", help="Required explanation of how mean rows align to tokens."
    ),
    mean_values: str | None = typer.Option(None, "--mean-values", help="JSON list of floats."),
    sequence_length: int | None = typer.Option(None, "--sequence-length"),
    save: bool = typer.Option(True, "--save/--no-save"),
    input_json: str | None = typer.Option(
        None, "--json", help="Full CreateActivationInterventionInput payload."
    ),
) -> None:
    """Create a versioned activation-intervention specification."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {
            "name": name,
            "module": module,
            "operation": operation,
            "tensor_role": role,
            "position_semantics": semantics,
            "positions": list(position or []),
            "apply_on_calls": list(apply_on_call or [0]),
            "allow_dtype_cast": allow_dtype_cast,
            "save": save,
        }
        if capture:
            payload["capture_artifact_id"] = capture
        if axis is not None:
            payload["axis"] = axis
        if feature_start is not None:
            payload["feature_start"] = feature_start
        if feature_end is not None:
            payload["feature_end"] = feature_end
        if feature_index:
            payload["feature_indices"] = list(feature_index)
        if tensor_name:
            payload["tensor_name"] = tensor_name
        if tensor_index is not None:
            payload["tensor_index"] = tensor_index
        if expected_dtype:
            payload["expected_dtype"] = expected_dtype
        if sequence_length is not None:
            payload["sequence_length"] = sequence_length
        if donor_capture:
            payload["donor_capture_artifact_id"] = donor_capture
        if donor_trace:
            payload["donor_trace_id"] = donor_trace
        if donor_position:
            payload["donor_positions"] = list(donor_position)
        if donor_values:
            payload["donor_values"] = json.loads(donor_values)
        payload["donor_condition"] = donor_condition
        if mean_source_capture:
            payload["mean_source_capture_artifact_id"] = mean_source_capture
        if mean_position:
            payload["mean_positions"] = list(mean_position)
        if mean_alignment:
            payload["mean_token_alignment"] = mean_alignment
        if mean_values:
            payload["mean_values"] = json.loads(mean_values)
    envelope = service.call("create_activation_intervention", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("validate-activation-intervention")
def validate_activation_intervention(
    ctx: typer.Context,
    intervention: str | None = typer.Option(None, "--intervention", help="Artifact id."),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    tokens: str | None = typer.Option(None, "--tokens", help="Comma-separated token ids."),
    trace: str | None = typer.Option(None, "--trace", help="Trace id to take tokens from."),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Check that an intervention can be applied to a concrete execution."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = _execution_payload(
        intervention=intervention, input_json=input_json, source=source, tokens=tokens, trace=trace
    )
    envelope = service.call("validate_activation_intervention", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("preview-activation-intervention")
def preview_activation_intervention(
    ctx: typer.Context,
    intervention: str | None = typer.Option(None, "--intervention", help="Artifact id."),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    tokens: str | None = typer.Option(None, "--tokens", help="Comma-separated token ids."),
    trace: str | None = typer.Option(None, "--trace", help="Trace id to take tokens from."),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Show the human-readable and structured scope of an intervention."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = _execution_payload(
        intervention=intervention, input_json=input_json, source=source, tokens=tokens, trace=trace
    )
    envelope = service.call("preview_activation_intervention", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


def _run_payload(
    *,
    intervention: str | None,
    input_json: str | None,
    source: str | None,
    tokens: str | None,
    trace: str | None,
    measure_module: list[str] | None,
    measure_position: list[int] | None,
    logit_position: list[int] | None,
) -> dict[str, Any]:
    payload = _execution_payload(
        intervention=intervention, input_json=input_json, source=source, tokens=tokens, trace=trace
    )
    if measure_module:
        payload["measure_modules"] = list(measure_module)
    if measure_position:
        payload["measure_positions"] = list(measure_position)
    if logit_position:
        payload["logit_positions"] = list(logit_position)
    return payload


@app.command("run-activation-intervention")
def run_activation_intervention(
    ctx: typer.Context,
    intervention: str | None = typer.Option(None, "--intervention", help="Artifact id."),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    tokens: str | None = typer.Option(None, "--tokens", help="Comma-separated token ids."),
    trace: str | None = typer.Option(None, "--trace", help="Trace id to take tokens from."),
    measure_module: list[str] | None = typer.Option(None, "--measure-module"),
    measure_position: list[int] | None = typer.Option(None, "--measure-position"),
    logit_position: list[int] | None = typer.Option(None, "--logit-position"),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Run one intervention during a single teacher-forced forward pass."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = _run_payload(
        intervention=intervention,
        input_json=input_json,
        source=source,
        tokens=tokens,
        trace=trace,
        measure_module=measure_module,
        measure_position=measure_position,
        logit_position=logit_position,
    )
    envelope = service.call("run_activation_intervention", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("run-paired-intervention")
def run_paired_intervention(
    ctx: typer.Context,
    intervention: str | None = typer.Option(None, "--intervention", help="Artifact id."),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    tokens: str | None = typer.Option(None, "--tokens", help="Comma-separated token ids."),
    trace: str | None = typer.Option(None, "--trace", help="Trace id to take tokens from."),
    measure_module: list[str] | None = typer.Option(None, "--measure-module"),
    measure_position: list[int] | None = typer.Option(None, "--measure-position"),
    logit_position: list[int] | None = typer.Option(None, "--logit-position"),
    alternative: str | None = typer.Option(
        None, "--alternative", help="Alternative-control intervention artifact id."
    ),
    direction: str | None = typer.Option(None, "--direction", help="JSON list of floats."),
    no_identity: bool = typer.Option(False, "--no-identity", help="Skip the identity control."),
    identity_atol: float = typer.Option(1e-6, "--identity-atol"),
    identity_rtol: float = typer.Option(1e-5, "--identity-rtol"),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Run baseline, identity control, and intervention as one paired experiment."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = _run_payload(
        intervention=intervention,
        input_json=input_json,
        source=source,
        tokens=tokens,
        trace=trace,
        measure_module=measure_module,
        measure_position=measure_position,
        logit_position=logit_position,
    )
    payload["identity"] = not no_identity
    payload["identity_atol"] = identity_atol
    payload["identity_rtol"] = identity_rtol
    if alternative:
        payload["alternative_intervention_artifact_id"] = alternative
    if direction:
        payload["direction"] = json.loads(direction)
    envelope = service.call("run_paired_intervention", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("compare-intervention-results")
def compare_intervention_results(
    ctx: typer.Context,
    artifact_id: str = typer.Argument(..., help="Counterfactual comparison artifact id."),
    no_records: bool = typer.Option(False, "--no-records", help="Omit per-result records."),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Inspect a stored paired intervention comparison."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = (
        json.loads(input_json)
        if input_json
        else {"comparison_artifact_id": artifact_id, "include_result_records": not no_records}
    )
    envelope = service.call("compare_intervention_results", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("run-counterfactual-continuation")
def run_counterfactual_continuation(
    ctx: typer.Context,
    intervention: str | None = typer.Option(None, "--intervention", help="Artifact id."),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    tokens: str | None = typer.Option(None, "--tokens", help="Comma-separated prefix token ids."),
    trace: str | None = typer.Option(None, "--trace", help="Trace id to take prefix tokens from."),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens"),
    max_sequence_length: int = typer.Option(2048, "--max-sequence-length"),
    eos_token_id: int | None = typer.Option(None, "--eos-token-id"),
    seed: int = typer.Option(0, "--seed"),
    no_identity: bool = typer.Option(False, "--no-identity"),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Run baseline, identity, and intervention greedy continuations from a shared prefix."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = _execution_payload(
            intervention=intervention,
            input_json=None,
            source=source,
            tokens=tokens,
            trace=trace,
        )
        payload.update(
            {
                "max_new_tokens": max_new_tokens,
                "max_sequence_length": max_sequence_length,
                "seed": seed,
                "identity": not no_identity,
            }
        )
        if eos_token_id is not None:
            payload["eos_token_id"] = eos_token_id
    envelope = service.call("run_counterfactual_continuation", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("inspect-counterfactual-comparison")
def inspect_counterfactual_comparison(
    ctx: typer.Context,
    artifact_id: str = typer.Argument(..., help="Comparison or continuation artifact id."),
    no_records: bool = typer.Option(False, "--no-records", help="Omit full records."),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Read a paired comparison or continuation comparison with integrity checks."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = (
        json.loads(input_json)
        if input_json
        else {"artifact_id": artifact_id, "include_records": not no_records}
    )
    envelope = service.call("inspect_counterfactual_comparison", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("attach-intervention-evidence")
def attach_intervention_evidence(
    ctx: typer.Context,
    title: str = typer.Option(..., "--title"),
    case_id: str | None = typer.Option(None, "--case", help="Attach the evidence to a case."),
    summary: str | None = typer.Option(None, "--summary"),
    observation: str | None = typer.Option(None, "--observation", help="Observed trace id."),
    control: list[str] | None = typer.Option(None, "--control", help="Control trace id; repeatable."),
    intervention: list[str] | None = typer.Option(None, "--intervention"),
    comparison: list[str] | None = typer.Option(None, "--comparison"),
    continuation: list[str] | None = typer.Option(None, "--continuation"),
    capture: list[str] | None = typer.Option(None, "--capture"),
    hypothesis: list[str] | None = typer.Option(None, "--hypothesis"),
    note: list[str] | None = typer.Option(None, "--note"),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Attach intervention and continuation evidence to a failure case."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {
            "title": title,
            "control_trace_ids": list(control or []),
            "intervention_artifact_ids": list(intervention or []),
            "comparison_artifact_ids": list(comparison or []),
            "continuation_artifact_ids": list(continuation or []),
            "capture_artifact_ids": list(capture or []),
            "hypotheses": [{"statement": item} for item in hypothesis or []],
            "notes": list(note or []),
        }
        if case_id:
            payload["case_id"] = case_id
        if summary:
            payload["summary"] = summary
        if observation:
            payload["observation_trace_id"] = observation
    envelope = service.call("attach_intervention_evidence", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


def _plan_payload(
    *,
    plan: str | None,
    input_json: str | None,
    source: str | None,
    tokens: str | None,
    trace: str | None,
) -> dict[str, Any]:
    if input_json:
        payload = json.loads(input_json)
    else:
        if plan:
            payload = {"plan_artifact_id": plan}
        else:
            payload = {}
    if source and "source" not in payload:
        payload["source"] = json.loads(source)
    if tokens and "token_ids" not in payload:
        payload["token_ids"] = [int(item) for item in tokens.split(",") if item.strip()]
    if trace and "trace_id" not in payload:
        payload["trace_id"] = trace
    return payload


@app.command("create-intervention-plan")
def create_intervention_plan(
    ctx: typer.Context,
    name: str | None = typer.Option(None, "--name"),
    trace: str | None = typer.Option(None, "--trace", help="Trace id providing the prefix."),
    tokens: str | None = typer.Option(None, "--tokens", help="Comma-separated prefix token ids."),
    execution_mode: str = typer.Option("uncached-generation", "--execution-mode"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens"),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Create a versioned, multi-site, generation-step-scheduled plan."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {"execution_mode": execution_mode, "max_new_tokens": max_new_tokens}
        if name:
            payload["name"] = name
        if trace:
            payload["trace_id"] = trace
        if tokens:
            payload["prompt_token_ids"] = [
                int(item) for item in tokens.split(",") if item.strip()
            ]
    envelope = service.call("create_intervention_plan", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("validate-intervention-plan")
def validate_intervention_plan(
    ctx: typer.Context,
    plan: str = typer.Argument(..., help="Plan artifact id."),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
) -> None:
    """Validate a plan's schedule, targeting, ordering, and budgets."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload: dict[str, Any] = {"plan_artifact_id": plan}
    if source:
        payload["source"] = json.loads(source)
    envelope = service.call("validate_intervention_plan", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("preview-intervention-plan")
def preview_intervention_plan(
    ctx: typer.Context,
    plan: str | None = typer.Option(None, "--plan", help="Plan artifact id."),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    tokens: str | None = typer.Option(None, "--tokens", help="Comma-separated prefix token ids."),
    trace: str | None = typer.Option(None, "--trace", help="Trace id providing the prefix."),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Resolve every scheduled firing without changing any activation."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = _plan_payload(
        plan=plan, input_json=input_json, source=source, tokens=tokens, trace=trace
    )
    envelope = service.call("preview_intervention_plan", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("run-temporal-intervention")
def run_temporal_intervention(
    ctx: typer.Context,
    plan: str | None = typer.Option(None, "--plan", help="Plan artifact id."),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    tokens: str | None = typer.Option(None, "--tokens", help="Comma-separated prefix token ids."),
    trace: str | None = typer.Option(None, "--trace", help="Trace id providing the prefix."),
    case: str | None = typer.Option(None, "--case", help="Attach the run to a case id."),
    no_identity: bool = typer.Option(False, "--no-identity"),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Run untouched, identity, and scheduled-plan conditions."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = _plan_payload(
        plan=plan, input_json=input_json, source=source, tokens=tokens, trace=trace
    )
    payload.setdefault("identity", not no_identity)
    if case:
        payload.setdefault("case_id", case)
    envelope = service.call("run_temporal_intervention", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("inspect-intervention-events")
def inspect_intervention_events(
    ctx: typer.Context,
    artifact: str = typer.Argument(..., help="Event-log or temporal-run artifact id."),
    run_level: bool = typer.Option(False, "--run", help="Treat the id as a temporal run."),
    no_events: bool = typer.Option(False, "--no-events", help="Return counts only."),
    max_entries: int = typer.Option(50, "--max-entries"),
) -> None:
    """Inspect applied, missing, not-reached, and rejected intervention events."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload: dict[str, Any] = {
        "include_events": not no_events,
        "max_entries": max_entries,
    }
    if run_level:
        payload["run_artifact_id"] = artifact
    else:
        payload["event_log_artifact_id"] = artifact
    envelope = service.call("inspect_intervention_events", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("run-compositional-comparison")
def run_compositional_comparison(
    ctx: typer.Context,
    plan_a: str = typer.Option(..., "--plan-a"),
    plan_b: str = typer.Option(..., "--plan-b"),
    plan_ab: str = typer.Option(..., "--plan-ab"),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    tokens: str | None = typer.Option(None, "--tokens", help="Comma-separated prefix token ids."),
    trace: str | None = typer.Option(None, "--trace", help="Trace id providing the prefix."),
    metric: str = typer.Option("mean_generated_logprob", "--metric"),
    case: str | None = typer.Option(None, "--case"),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Measure baseline, A, B, and A+B and report descriptive non-additivity."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {
            "plan_a_artifact_id": plan_a,
            "plan_b_artifact_id": plan_b,
            "plan_ab_artifact_id": plan_ab,
            "metric": metric,
        }
        if source:
            payload["source"] = json.loads(source)
        if tokens:
            payload["token_ids"] = [int(item) for item in tokens.split(",") if item.strip()]
        if trace:
            payload["trace_id"] = trace
        if case:
            payload["case_id"] = case
    envelope = service.call("run_compositional_comparison", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("prepare-staged-generation")
def prepare_staged_generation(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="JSON replay model source."),
    tokens: str = typer.Option(..., "--tokens", help="Comma-separated prefix token ids."),
) -> None:
    """Build a cached generation state and report architecture support."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "prepare_staged_generation",
        {
            "source": json.loads(source),
            "token_ids": [int(item) for item in tokens.split(",") if item.strip()],
        },
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("verify-staged-generation")
def verify_staged_generation(
    ctx: typer.Context,
    plan: str | None = typer.Option(None, "--plan"),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    tokens: str | None = typer.Option(None, "--tokens"),
    trace: str | None = typer.Option(None, "--trace"),
    isolation: str = typer.Option("recompute", "--isolation"),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Compare an unmodified staged run with the uncached reference."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = _plan_payload(
        plan=plan, input_json=input_json, source=source, tokens=tokens, trace=trace
    )
    payload.setdefault("isolation", isolation)
    envelope = service.call("verify_staged_generation", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("run-staged-intervention")
def run_staged_intervention(
    ctx: typer.Context,
    plan: str | None = typer.Option(None, "--plan"),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    tokens: str | None = typer.Option(None, "--tokens"),
    trace: str | None = typer.Option(None, "--trace"),
    case: str | None = typer.Option(None, "--case"),
    isolation: str = typer.Option("recompute", "--isolation"),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Run a scheduled plan through a verified staged decoder."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = _plan_payload(
        plan=plan, input_json=input_json, source=source, tokens=tokens, trace=trace
    )
    payload.setdefault("isolation", isolation)
    if case:
        payload.setdefault("case_id", case)
    envelope = service.call("run_staged_intervention", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("attach-temporal-evidence")
def attach_temporal_evidence(
    ctx: typer.Context,
    title: str = typer.Option(..., "--title"),
    case: str | None = typer.Option(None, "--case"),
    summary: str | None = typer.Option(None, "--summary"),
    observation: str | None = typer.Option(None, "--observation"),
    plan: list[str] | None = typer.Option(None, "--plan"),
    event_log: list[str] | None = typer.Option(None, "--event-log"),
    run_id: list[str] | None = typer.Option(None, "--run"),
    composition: list[str] | None = typer.Option(None, "--composition"),
    staged: list[str] | None = typer.Option(None, "--staged"),
    verification: list[str] | None = typer.Option(None, "--verification"),
    hypothesis: list[str] | None = typer.Option(None, "--hypothesis"),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Attach temporal experiments to a failure case."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {
            "title": title,
            "plan_artifact_ids": list(plan or []),
            "event_log_artifact_ids": list(event_log or []),
            "run_artifact_ids": list(run_id or []),
            "composition_artifact_ids": list(composition or []),
            "staged_artifact_ids": list(staged or []),
            "verification_artifact_ids": list(verification or []),
            "hypotheses": [{"statement": item} for item in hypothesis or []],
        }
        if case:
            payload["case_id"] = case
        if summary:
            payload["summary"] = summary
        if observation:
            payload["observation_trace_id"] = observation
    envelope = service.call("attach_temporal_evidence", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("discover")
def discover(
    ctx: typer.Context,
    query: str | None = typer.Option(None, "--query", "-q"),
    group: str | None = typer.Option(None, "--group"),
    detail: str = typer.Option("compact", "--detail"),
    include_recipes: bool = typer.Option(True, "--recipes/--no-recipes"),
    max_operations: int = typer.Option(12, "--max-operations"),
) -> None:
    """Discover relevant operations and recipes without dumping the whole registry."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "discover_capabilities",
        {
            "query": query,
            "group": group,
            "detail": detail,
            "include_recipes": include_recipes,
            "max_operations": max_operations,
        },
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("detect-events")
def detect_events(
    ctx: typer.Context,
    trace: str | None = typer.Option(None, "--trace", help="Trace id."),
    finding_index: int | None = typer.Option(None, "--finding-index"),
    pattern: str | None = typer.Option(None, "--text-pattern"),
    manual_position: int | None = typer.Option(None, "--position"),
    input_json: str | None = typer.Option(None, "--json", help="Full request payload."),
) -> None:
    """Detect deterministic behavioral events in a trace."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload = json.loads(input_json) if input_json else {"trace_id": trace}
    if trace is not None:
        payload.setdefault("trace_id", trace)
    if finding_index is not None:
        payload.setdefault("finding_index", finding_index)
    if pattern is not None:
        payload.setdefault("text_pattern", pattern)
    if manual_position is not None:
        payload.setdefault("manual_position", manual_position)
    envelope = service.call("detect_behavioral_events", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("resolve-event")
def resolve_event(
    ctx: typer.Context,
    artifact: str = typer.Argument(..., help="Behavioral-event artifact id."),
) -> None:
    """Report an event's exact token, generation-step, and online timing."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("resolve_event_position", {"event_artifact_id": artifact})
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("create-event-trigger")
def create_event_trigger(
    ctx: typer.Context,
    input_json: str = typer.Option(..., "--json", help="Trigger payload."),
) -> None:
    """Create a validated declarative event trigger."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("create_event_trigger", {"trigger": json.loads(input_json)})
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("event-triggers")
def event_triggers(
    ctx: typer.Context,
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """List stored event triggers."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("list_event_triggers", {"limit": limit})
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("run-event-intervention")
def run_event_intervention(
    ctx: typer.Context,
    trigger: str | None = typer.Option(None, "--trigger", help="Trigger artifact id."),
    intervention: str | None = typer.Option(None, "--intervention"),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    trace: str | None = typer.Option(None, "--trace"),
    event: str | None = typer.Option(None, "--event", help="Retrospective event artifact id."),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens"),
    no_identity: bool = typer.Option(False, "--no-identity"),
    input_json: str | None = typer.Option(None, "--json"),
) -> None:
    """Run an online trigger or a retrospective event investigation."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    if input_json:
        payload = json.loads(input_json)
    else:
        payload = {
            "intervention_artifact_id": intervention,
            "source": json.loads(source) if source else {"kind": "synthetic"},
            "max_new_tokens": max_new_tokens,
            "identity": not no_identity,
        }
        if trigger:
            payload["trigger_artifact_id"] = trigger
        if trace:
            payload["trace_id"] = trace
        if event:
            payload["event_artifact_id"] = event
    envelope = service.call("run_event_triggered_intervention", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("prepare-fixed-context")
def prepare_fixed_context(
    ctx: typer.Context,
    trace: str = typer.Option(..., "--trace"),
    event: list[str] | None = typer.Option(None, "--event"),
    replay_step: int | None = typer.Option(None, "--replay-step"),
) -> None:
    """Reconstruct an exact token prefix for a fixed-context experiment."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload: dict[str, Any] = {
        "trace_id": trace,
        "event_artifact_ids": list(event or []),
    }
    if replay_step is not None:
        payload["replay_step"] = replay_step
    envelope = service.call("prepare_fixed_context_replay", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("run-fixed-context")
def run_fixed_context(
    ctx: typer.Context,
    replay: str = typer.Option(..., "--replay"),
    intervention: str = typer.Option(..., "--intervention"),
    source: str | None = typer.Option(None, "--source", help="JSON replay model source."),
    case: str | None = typer.Option(None, "--case"),
    continuation: str | None = typer.Option(None, "--continuation", help="JSON continuation config."),
) -> None:
    """Run baseline, identity, and intervention on one fixed token context."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload: dict[str, Any] = {
        "replay_artifact_id": replay,
        "intervention_artifact_id": intervention,
        "source": json.loads(source) if source else {"kind": "synthetic"},
    }
    if case:
        payload["case_id"] = case
    if continuation:
        payload["continuation"] = json.loads(continuation)
    envelope = service.call("run_fixed_context_comparison", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("create-dossier")
def create_dossier(
    ctx: typer.Context,
    title: str = typer.Option(..., "--title"),
    case: str | None = typer.Option(None, "--case"),
    trace: str | None = typer.Option(None, "--trace"),
    event: list[str] | None = typer.Option(None, "--event"),
    comparison: str | None = typer.Option(None, "--comparison"),
    summary: str | None = typer.Option(None, "--summary"),
) -> None:
    """Create an agent-readable investigation dossier."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload: dict[str, Any] = {
        "title": title,
        "event_artifact_ids": list(event or []),
        "fixed_context_comparison_artifact_id": comparison,
    }
    if case:
        payload["case_id"] = case
    if trace:
        payload["trace_id"] = trace
    if summary:
        payload["summary"] = summary
    envelope = service.call("create_investigation_dossier", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("inspect-dossier")
def inspect_dossier(
    ctx: typer.Context,
    artifact: str = typer.Argument(..., help="Dossier artifact id."),
    no_measurements: bool = typer.Option(False, "--no-measurements"),
) -> None:
    """Read an investigation dossier with bounded measurements."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "inspect_investigation_dossier",
        {"dossier_artifact_id": artifact, "include_measurements": not no_measurements},
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("attach-event-evidence")
def attach_event_evidence(
    ctx: typer.Context,
    title: str = typer.Option(..., "--title"),
    case: str | None = typer.Option(None, "--case"),
    trace: str | None = typer.Option(None, "--trace"),
    event: list[str] | None = typer.Option(None, "--event"),
    comparison: list[str] | None = typer.Option(None, "--comparison"),
    dossier: list[str] | None = typer.Option(None, "--dossier"),
) -> None:
    """Attach events, comparisons, and dossiers to a failure case."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    payload: dict[str, Any] = {
        "title": title,
        "event_artifact_ids": list(event or []),
        "fixed_context_comparison_artifact_ids": list(comparison or []),
        "dossier_artifact_ids": list(dossier or []),
    }
    if case:
        payload["case_id"] = case
    if trace:
        payload["trace_id"] = trace
    envelope = service.call("attach_event_evidence", payload)
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("recipes")
def recipes(
    ctx: typer.Context,
    group: str | None = typer.Option(None, "--group"),
) -> None:
    """List investigation recipes."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("list_investigation_recipes", {"group": group})
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("preview-recipe")
def preview_recipe_command(
    ctx: typer.Context,
    recipe_id: str = typer.Argument(..., help="Recipe id."),
    input_json: str | None = typer.Option(None, "--inputs", help="JSON input object."),
) -> None:
    """Preview a recipe before running it."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "preview_investigation_recipe",
        {"recipe_id": recipe_id, "inputs": json.loads(input_json) if input_json else {}},
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("run-recipe")
def run_recipe_command(
    ctx: typer.Context,
    recipe_id: str = typer.Argument(..., help="Recipe id."),
    input_json: str = typer.Option(..., "--inputs", help="JSON input object."),
) -> None:
    """Run a high-level investigation recipe."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call(
        "run_investigation_recipe",
        {"recipe_id": recipe_id, "inputs": json.loads(input_json)},
    )
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


@app.command("inspect-recipe-run")
def inspect_recipe_run(
    ctx: typer.Context,
    artifact: str = typer.Argument(..., help="Recipe-run artifact id."),
) -> None:
    """Read a stored recipe run."""
    service = ToolService(_workspace(ctx), budget=_budget(ctx))
    envelope = service.call("inspect_recipe_run", {"run_artifact_id": artifact})
    _emit(envelope, exit_code=0 if envelope.get("ok") else 1)


if __name__ == "__main__":  # pragma: no cover
    app()
