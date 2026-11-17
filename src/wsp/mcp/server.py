"""MCP stdio server exposing the shared WSP ``ToolService`` registry.

Thin transport only.  For every ``tools/call`` the envelope produced by
``ToolService.call`` is passed through verbatim as both text content and
structured content.  Operation-level failures (unknown operation, schema
violation, resource limit) are returned to the agent as ``is_error`` results
carrying the same structured envelope the CLI prints, so agents observe
identical semantics across transports.  The adapter never retries, adjusts
tolerances, or drops fields.

Requires the optional ``mcp`` dependency: ``pip install -e ".[mcp]"``.
"""

from __future__ import annotations

import asyncio
import json
from importlib import metadata
from typing import Any

import mcp.types as types
from mcp.server import stdio as mcp_stdio
from mcp.server.lowlevel import Server

from wsp.core.json import to_jsonable
from wsp.mcp.mapping import McpToolSpec, mcp_tool_specs, server_instructions
from wsp.tools.service import ToolService

SERVER_NAME = "wsp-workbench"


def _package_version() -> str:
    try:
        return metadata.version("wsp-workbench")
    except metadata.PackageNotFoundError:  # pragma: no cover - running from source
        return "0+unknown"


def _error_envelope(operation: str, exc: BaseException) -> dict[str, Any]:
    """Fail-closed envelope for errors that escape the service layer itself."""
    error: dict[str, Any] = {"message": str(exc)}
    if isinstance(exc, Exception) and hasattr(exc, "code"):
        code = getattr(exc, "code", None)
        if isinstance(code, str):
            error["code"] = code
    if not error.get("code"):
        error["code"] = "internal_error"
    return {
        "schema_version": "wsp.tool/1.0",
        "ok": False,
        "operation": operation,
        "data": None,
        "error": error,
        "run_id": None,
        "artifacts": [],
        "warnings": [],
    }


def _call_result(envelope: dict[str, Any]) -> types.CallToolResult:
    payload = to_jsonable(envelope)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structured_content=payload,
        is_error=not bool(envelope.get("ok")),
    )


def build_server(service: ToolService) -> Server:
    """Build an MCP server over ``service`` without running it.

    The tool surface is fixed at construction time from the registry: MCP
    clients observe exactly the operations registered by the service, with
    schemas generated from the same pydantic input models the CLI validates
    against.
    """
    specs: list[McpToolSpec] = mcp_tool_specs(service)

    async def on_list_tools(
        context: Any, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=spec.name,
                    description=spec.description,
                    input_schema=spec.input_schema,
                )
                for spec in specs
            ]
        )

    async def on_call_tool(
        context: Any, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        operation = params.name
        arguments = dict(params.arguments or {})
        try:
            # Handlers are synchronous CPU work; keep the event loop live.
            envelope = await asyncio.to_thread(service.call, operation, arguments)
        except Exception as exc:  # pragma: no cover - service.call is envelope-safe
            envelope = _error_envelope(operation, exc)
        return _call_result(envelope)

    return Server(
        SERVER_NAME,
        version=_package_version(),
        instructions=server_instructions(),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def serve(service: ToolService) -> None:
    """Serve ``service`` over the MCP stdio transport until EOF."""
    server = build_server(service)
    async with mcp_stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def run_stdio(service: ToolService) -> None:
    """Blocking entry point used by ``wsp mcp``."""
    asyncio.run(serve(service))
