"""CLI tests: machine-readable output through the shared service."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import pytest

from wsp.cli.main import app
from wsp.tools.service import ToolService


def _json(output: str) -> dict:
    return json.loads(output)


def test_cli_mcp_command_lists_in_help() -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["mcp", "--help"])
    assert help_result.exit_code == 0
    assert "MCP" in help_result.stdout


def test_cli_mcp_help_lists_workspace_options() -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["mcp", "--help"])
    assert help_result.exit_code == 0
    for option in ("--workspace", "--budget", "--auto-init"):
        assert option in help_result.stdout, option


def test_cli_mcp_broken_workspace_fails_with_clean_stderr_json(tmp_path: Path) -> None:
    import subprocess
    import sys

    missing = tmp_path / "not-a-workspace"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from wsp.cli.main import app; "
            f"sys.argv = ['wsp', 'mcp', '--workspace', {str(missing)!r}]; app()",
        ],
        input=b"",
        capture_output=True,
        timeout=60,
    )
    # A broken workspace must produce a diagnosable stderr error, not a
    # traceback that MCP clients only see as "Connection closed".
    assert result.returncode == 1
    stderr = result.stderr.decode(errors="replace")
    payload = json.loads(stderr)
    assert payload["schema_version"] == "wsp.mcp/1.0"
    assert payload["ok"] is False
    assert payload["error"]["code"] == "not_found"
    assert "workspace" in payload["error"]["message"]
    assert "Traceback" not in stderr


def test_cli_mcp_auto_init_creates_workspace_then_exits(tmp_path: Path) -> None:
    import subprocess
    import sys

    workspace = tmp_path / "fresh"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from wsp.cli.main import app; "
            f"sys.argv = ['wsp', 'mcp', '--workspace', {str(workspace)!r}, '--auto-init']; app()",
        ],
        input=b"",
        capture_output=True,
        timeout=60,
    )
    # EOF still shuts the server down cleanly, and the workspace was created.
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert (workspace / "workspace.json").is_file()


def test_cli_mcp_server_exits_cleanly_on_eof() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['wsp', 'mcp']; from wsp.cli.main import app; app()",
        ],
        input=b"",
        capture_output=True,
        timeout=60,
    )
    # An immediate EOF (no client) must shut the stdio server down cleanly.
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_cli_help_and_operation_registry(mlp_checkpoint: Path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "Weight-Space Programming" in help_result.stdout

    tools = runner.invoke(app, ["tools", "--operation", "inspect_model"])
    assert tools.exit_code == 0
    payload = _json(tools.stdout)
    assert payload["schema_version"] == "wsp.tool-registry/1.0"
    assert payload["operations"][0]["input_schema"]["additionalProperties"] is False


def test_cli_inspect_model_is_json_and_matches_api(mlp_checkpoint: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["call", "inspect_model", "--param", f"path={mlp_checkpoint}"],
    )
    assert result.exit_code == 0, result.stdout
    cli_payload = _json(result.stdout)
    assert cli_payload["schema_version"] == "wsp.tool/1.0"
    assert cli_payload["ok"] is True
    api_payload = ToolService(discover_workspace=False).call(
        "inspect_model", {"path": str(mlp_checkpoint)}
    )
    assert cli_payload["data"]["manifest"]["identity"] == api_payload["data"]["manifest"]["identity"]


def test_cli_workspace_init_and_status(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "cli-workspace"
    initialized = runner.invoke(
        app, ["--workspace", str(root), "init"], catch_exceptions=False
    )
    assert initialized.exit_code == 0, initialized.stdout
    assert (root / "workspace.json").exists()
    status = runner.invoke(
        app, ["--workspace", str(root), "status"], catch_exceptions=False
    )
    assert status.exit_code == 0
    payload = _json(status.stdout)
    assert payload["data"]["workspace"]["name"] == root.name
    assert payload["ok"] is True


def test_cli_errors_are_json_with_nonzero_exit(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["call", "not_a_tool", "--json", "{}"])
    assert result.exit_code == 1
    payload = _json(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_input"

    bad_json = runner.invoke(app, ["call", "inspect_model", "--json", "[]"])
    assert bad_json.exit_code == 1
    assert _json(bad_json.stdout)["error"]["code"] == "invalid_input"


def test_cli_budget_override_is_validated(
    tmp_path: Path, mlp_checkpoint: Path
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--budget",
            '{"max_preview_elements": 2}',
            "call",
            "inspect_tensor",
            "--param",
            f"path={mlp_checkpoint}",
            "--param",
            "tensor=fc1.weight",
            "--param",
            "preview_elements=100",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = _json(result.stdout)
    assert payload["data"]["preview"]["returned"] <= 2
    assert payload["data"]["preview"]["sampled"] is True


def test_cli_behavioral_workflow(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "cli-behavior"
    initialized = runner.invoke(
        app, ["--workspace", str(root), "init"], catch_exceptions=False
    )
    assert initialized.exit_code == 0, initialized.stdout

    recorded = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "record-trace",
            "prompt",
            "--output",
            "a b c d e f g h a b c d e f g h",
        ],
        catch_exceptions=False,
    )
    assert recorded.exit_code == 0, recorded.stdout
    trace_id = _json(recorded.stdout)["data"]["trace_id"]

    scanned = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "scan-trace",
            trace_id,
            "--detector",
            "core.repetition",
        ],
        catch_exceptions=False,
    )
    assert scanned.exit_code == 0, scanned.stdout
    scan_payload = _json(scanned.stdout)
    assert scan_payload["data"]["scan"]["findings_count"] >= 1

    listed = runner.invoke(
        app, ["--workspace", str(root), "traces"], catch_exceptions=False
    )
    assert listed.exit_code == 0
    assert _json(listed.stdout)["data"]["returned"] == 1

    detectors = runner.invoke(
        app, ["--workspace", str(root), "detectors"], catch_exceptions=False
    )
    assert detectors.exit_code == 0
    assert any(
        item["detector_id"] == "core.repetition"
        for item in _json(detectors.stdout)["data"]["detectors"]
    )

    created = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "call",
            "create_failure_case",
            "--json",
            json.dumps(
                {
                    "title": "CLI-created case",
                    "observations": [trace_id],
                    "tags": ["cli"],
                }
            ),
        ],
        catch_exceptions=False,
    )
    assert created.exit_code == 0, created.stdout
    case_id = _json(created.stdout)["data"]["case"]["case_id"]

    inspected = runner.invoke(
        app,
        ["--workspace", str(root), "inspect-case", case_id],
        catch_exceptions=False,
    )
    assert inspected.exit_code == 0
    assert _json(inspected.stdout)["data"]["case"]["case_id"] == case_id

    out = tmp_path / "cli-case.json"
    exported = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "export-case",
            case_id,
            "--output",
            str(out),
        ],
        catch_exceptions=False,
    )
    assert exported.exit_code == 0, exported.stdout
    assert out.exists()
    assert _json(out.read_text(encoding="utf-8"))["schema_version"] == "wsp.case-export/1.1"


def test_cli_intervention_workflow(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "cli-intervention"
    initialized = runner.invoke(
        app, ["--workspace", str(root), "init"], catch_exceptions=False
    )
    assert initialized.exit_code == 0, initialized.stdout

    created = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "create-activation-intervention",
            "--name",
            "cli zero ablation",
            "--module",
            "blocks.1",
            "--operation",
            "zero",
            "--position",
            "1",
            "--position",
            "2",
            "--sequence-length",
            "4",
            "--json",
            json.dumps(
                {
                    "name": "cli zero ablation",
                    "module": "blocks.1",
                    "operation": "zero",
                    "positions": [1, 2],
                    "position_semantics": "consuming",
                    "tensor_role": "output",
                    "model": {"name": "tiny"},
                    "sequence_length": 4,
                    "apply_on_calls": [0],
                }
            ),
        ],
        catch_exceptions=False,
    )
    assert created.exit_code == 0, created.stdout
    intervention_id = _json(created.stdout)["data"]["artifact_id"]

    previewed = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "preview-activation-intervention",
            "--intervention",
            intervention_id,
            "--source",
            '{"kind": "synthetic"}',
            "--tokens",
            "1,5,6,7",
        ],
        catch_exceptions=False,
    )
    assert previewed.exit_code == 0, previewed.stdout
    assert _json(previewed.stdout)["data"]["valid"] is True

    paired = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "run-paired-intervention",
            "--intervention",
            intervention_id,
            "--source",
            '{"kind": "synthetic"}',
            "--tokens",
            "1,5,6,7",
            "--measure-module",
            "blocks.2",
            "--logit-position",
            "2",
        ],
        catch_exceptions=False,
    )
    assert paired.exit_code == 0, paired.stdout
    payload = _json(paired.stdout)["data"]
    assert payload["identity_control"]["agreement"] == "exact"
    assert payload["validity"]["status"] == "valid"


def test_cli_discover_and_recipes_are_json() -> None:
    runner = CliRunner()
    discover = runner.invoke(
        app, ["discover", "--query", "repetition", "--detail", "compact"]
    )
    assert discover.exit_code == 0, discover.stdout
    payload = _json(discover.stdout)
    assert payload["data"]["schema_version"] == "wsp.capabilities/1.0"
    assert payload["data"]["recipes"]

    recipes = runner.invoke(app, ["recipes"])
    assert recipes.exit_code == 0, recipes.stdout
    listed = _json(recipes.stdout)
    assert listed["data"]["count"] >= 3
    ids = {item["recipe_id"] for item in listed["data"]["recipes"]}
    assert "investigate_repetition_event" in ids


def test_cli_detect_events_and_preview_recipe(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "cli-events"
    runner.invoke(app, ["--workspace", str(root), "init"], catch_exceptions=False)
    recorded = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "record-trace",
            "--json",
            '{"input_text":"loop","backend":{"kind":"synthetic"},"params":{"max_new_tokens":8}}',
        ],
    )
    assert recorded.exit_code == 0, recorded.stdout
    trace_id = _json(recorded.stdout)["data"]["trace_id"]
    detected = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "detect-events",
            "--trace",
            trace_id,
            "--json",
            '{"event_definition":{"kind":"first_repetition","ngram_size":2,"min_occurrences":2}}',
        ],
    )
    assert detected.exit_code == 0, detected.stdout
    assert _json(detected.stdout)["data"]["event_id"]

    preview = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "preview-recipe",
            "test_activation_hypothesis",
            "--inputs",
            '{"trace_id":"tr_missing"}',
        ],
    )
    assert preview.exit_code == 0, preview.stdout
    assert _json(preview.stdout)["data"]["recipe"]["recipe_id"] == "test_activation_hypothesis"
