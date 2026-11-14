"""CLI tests: machine-readable output through the shared service."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from wsp.cli.main import app
from wsp.tools.service import ToolService


def _json(output: str) -> dict:
    return json.loads(output)


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
