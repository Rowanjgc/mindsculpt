"""Offline tests for the MCP stdio server over an in-memory transport.

These tests require the optional ``mcp`` dependency; without it they skip,
which means MCP compatibility is unverified, not passing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

pytest.importorskip("mcp")

pytestmark = pytest.mark.mcp

from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from wsp.mcp.server import build_server
from wsp.tools.service import ToolService


async def _run(service: ToolService, fn: Callable[[ClientSession], Any]) -> Any:
    """Run one client callback against a live in-memory server session."""

    async def _inner() -> Any:
        async with create_client_server_memory_streams() as (
            client_streams,
            server_streams,
        ):
            server = build_server(service)
            task = asyncio.create_task(
                server.run(
                    server_streams[0],
                    server_streams[1],
                    server.create_initialization_options(),
                )
            )
            try:
                async with ClientSession(
                    client_streams[0], client_streams[1]
                ) as session:
                    await session.initialize()
                    return await fn(session)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return await _inner()


def _call(service: ToolService, fn: Callable[[ClientSession], Any]) -> Any:
    """Synchronous wrapper: run one client callback on a fresh event loop."""
    return asyncio.run(_run(service, fn))


def _parse_envelope(raw: str) -> dict[str, Any]:
    envelope = json.loads(raw)
    assert isinstance(envelope, dict)
    assert envelope.get("schema_version") == "wsp.tool/1.0"
    assert "ok" in envelope
    return envelope


def test_list_tools_matches_the_registry() -> None:
    service = ToolService()
    names = _call(
        service, lambda session: _list_names(session)
    )
    assert names == service.operations()


async def _list_names(session: ClientSession) -> list[str]:
    result = await session.list_tools()
    return [tool.name for tool in result.tools]


def test_call_tool_returns_a_success_envelope(mlp_checkpoint: Path) -> None:
    service = ToolService()
    envelope = _call(
        service,
        lambda session: _call_tool(
            session, "inspect_model", {"path": str(mlp_checkpoint)}
        ),
    )
    assert envelope["ok"] is True
    manifest = envelope["data"]["manifest"]
    assert manifest["identity"]["structure_fingerprint"]
    assert envelope["operation"] == "inspect_model"


async def _call_tool(
    session: ClientSession, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    result = await session.call_tool(name, arguments)
    assert result.content, "every result must carry the envelope as text"
    envelope = _parse_envelope(result.content[0].text)
    assert result.structured_content == envelope
    assert result.is_error == (not envelope["ok"])
    return envelope


def test_schema_violations_return_a_structured_error() -> None:
    service = ToolService()
    envelope = _call(
        service,
        lambda session: _call_tool(session, "inspect_model", {}),
    )
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "invalid_input"
    assert envelope["error"]["details"]["errors"]


def test_unknown_operations_return_a_structured_error() -> None:
    service = ToolService()
    envelope = _call(
        service,
        lambda session: _call_tool(session, "definitely_not_an_operation", {}),
    )
    assert envelope["ok"] is False
    assert envelope["operation"] == "definitely_not_an_operation"
    assert "available" in envelope["error"]["details"]


def test_tool_descriptions_carry_workspace_notes() -> None:
    service = ToolService()

    async def _inspect(session: ClientSession) -> dict[str, str]:
        result = await session.list_tools()
        return {tool.name: tool.description or "" for tool in result.tools}

    descriptions = _call(service, _inspect)
    assert "workspace" in descriptions["list_patches"].lower()
    assert "workspace" not in descriptions["inspect_model"].lower()
