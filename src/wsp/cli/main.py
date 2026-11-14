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


if __name__ == "__main__":  # pragma: no cover
    app()
