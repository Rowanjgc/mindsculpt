"""Pure mapping from the WSP tool registry to MCP tool specifications.

This module deliberately does not import the optional ``mcp`` package.  The
mapping is derived exclusively from the public registry interface
(``ToolService.describe``), so it stays testable offline and the transport
layer never becomes a second implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from wsp.core.json import to_jsonable

#: MCP tool names must match this pattern (MCP specification).
MCP_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")

#: Bound the number of limitations inlined into one tool description.
_MAX_DESCRIPTION_LIMITATIONS = 3

#: Hard upper bound for a generated tool description.
_MAX_DESCRIPTION_CHARS = 2000

_WORKSPACE_NOTE = (
    "Requires an initialized workspace so artifacts, patches, and run records "
    "can be stored. Start 'wsp mcp' with an initialized workspace, or pass "
    "--auto-init to create one at startup."
)


@dataclass(frozen=True)
class McpToolSpec:
    """An MCP-ready projection of one registered WSP operation."""

    name: str
    description: str
    input_schema: dict[str, Any]
    requires_workspace: bool = False


def _description_for(entry: dict[str, Any]) -> str:
    parts = [str(entry.get("description") or "").strip()]
    if entry.get("requires_workspace"):
        parts.append(_WORKSPACE_NOTE)
    limitations = [item for item in (entry.get("limitations") or []) if item]
    if limitations:
        shown = limitations[:_MAX_DESCRIPTION_LIMITATIONS]
        remaining = len(limitations) - len(shown)
        suffix = f" (+{remaining} more; see discover_capabilities)" if remaining > 0 else ""
        parts.append("Limitations: " + "; ".join(shown) + suffix)
    description = " ".join(part for part in parts if part)
    if len(description) > _MAX_DESCRIPTION_CHARS:
        description = description[: _MAX_DESCRIPTION_CHARS - 1] + "\u2026"
    return description


def _schema_for(entry: dict[str, Any]) -> dict[str, Any]:
    schema = to_jsonable(entry.get("input_schema") or {})
    if not isinstance(schema, dict):
        raise ValueError("operation input schema must be a JSON object")
    schema.setdefault("type", "object")
    return schema


def mcp_tool_specs(service: Any) -> list[McpToolSpec]:
    """Project every registered operation onto an MCP tool specification.

    ``service`` is a :class:`wsp.tools.service.ToolService`.  The projection
    uses only the public ``describe`` interface so the adapter cannot drift
    from the registry.

    Raises:
        ValueError: if any operation name is not a valid MCP tool name, which
            would silently corrupt the transport surface.
    """
    registry = service.describe(detail="full")
    specs: list[McpToolSpec] = []
    for entry in registry["operations"]:
        name = str(entry["name"])
        if not MCP_TOOL_NAME_PATTERN.match(name):
            raise ValueError(
                f"operation {name!r} is not a valid MCP tool name; "
                "MCP names must match ^[a-zA-Z0-9_-]{1,128}$"
            )
        specs.append(
            McpToolSpec(
                name=name,
                description=_description_for(entry),
                input_schema=_schema_for(entry),
                requires_workspace=bool(entry.get("requires_workspace")),
            )
        )
    return specs


def server_instructions() -> str:
    """Static instructions returned to MCP clients during initialization."""
    return (
        "WSP Workbench: bounded instruments for inspecting, measuring, patching, "
        "and reproducing work on neural-network weights and behavior. Every tool "
        "returns a wsp.tool/1.0 envelope with ok, data, artifacts, and warnings; "
        "failed operations return ok=false with a structured error and are never "
        "retried silently. Call discover_capabilities first for progressive "
        "disclosure of operation groups, prerequisites, and limitations. If an "
        "operation reports that a workspace is required, the server must be "
        "started with an initialized workspace (create one with "
        "'wsp init --path <dir>', or restart 'wsp mcp' with --auto-init). "
        "Resource budgets bound every operation; oversized requests fail "
        "cleanly instead of truncating."
    )
