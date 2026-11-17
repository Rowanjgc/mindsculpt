"""Offline tests for the SDK-free MCP mapping layer."""

from __future__ import annotations

import json

from wsp.mcp.mapping import (
    MCP_TOOL_NAME_PATTERN,
    McpToolSpec,
    mcp_tool_specs,
    server_instructions,
)
from wsp.tools.service import ToolService


def test_specs_cover_the_registry_one_to_one() -> None:
    service = ToolService()
    specs = mcp_tool_specs(service)
    assert [spec.name for spec in specs] == service.operations()
    assert len(specs) > 40


def test_names_and_schemas_are_mcp_valid() -> None:
    service = ToolService()
    for spec in mcp_tool_specs(service):
        assert MCP_TOOL_NAME_PATTERN.match(spec.name), spec.name
        assert spec.description, spec.name
        assert len(spec.description) <= 2000, spec.name
        assert isinstance(spec.input_schema, dict), spec.name
        assert spec.input_schema.get("type") == "object", spec.name
        # Every schema must be JSON-serializable: MCP transports only move JSON.
        json.dumps(spec.input_schema)


def test_workspace_requirement_is_stated_in_the_description() -> None:
    service = ToolService()
    specs = {spec.name: spec for spec in mcp_tool_specs(service)}
    assert specs["list_patches"].requires_workspace
    assert "workspace" in specs["list_patches"].description.lower()
    assert not specs["inspect_model"].requires_workspace
    assert "workspace" not in specs["inspect_model"].description.lower()


def test_descriptions_declare_bounded_limitations() -> None:
    service = ToolService()
    specs = {spec.name: spec for spec in mcp_tool_specs(service)}
    flagged = [spec for spec in specs.values() if "Limitations:" in spec.description]
    assert flagged, "at least one operation should carry inline limitations"
    for spec in flagged:
        assert spec.description.split("Limitations: ", 1)[1].strip()


def test_schemas_match_the_registry_input_models() -> None:
    service = ToolService()
    registry = {entry["name"]: entry for entry in service.describe(detail="full")["operations"]}
    for spec in mcp_tool_specs(service):
        assert spec.input_schema == registry[spec.name]["input_schema"], spec.name


def test_invalid_operation_name_fails_closed() -> None:
    service = ToolService()

    # Registering through the public API with a name MCP cannot represent must
    # be caught by the projection instead of corrupting the transport surface.
    from wsp.tools.service import ToolDefinition

    class _Input:
        @classmethod
        def model_json_schema(cls) -> dict:
            return {"type": "object"}

        @classmethod
        def model_validate(cls, data: dict) -> object:
            return data

        def model_dump(self) -> dict:
            return {}

    service.register_tool(
        ToolDefinition(
            "bad name!",  # spaces are invalid for MCP
            "deliberately invalid operation name",
            _Input,  # type: ignore[arg-type]
            lambda request, budget, run: None,
            "nothing",
        )
    )
    try:
        mcp_tool_specs(service)
    except ValueError as exc:
        assert "bad name!" in str(exc)
    else:
        raise AssertionError("invalid MCP name must fail closed")


def test_server_instructions_mention_envelope_semantics() -> None:
    text = server_instructions()
    assert "wsp.tool/1.0" in text
    assert "discover_capabilities" in text
