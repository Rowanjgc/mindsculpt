"""Provider-neutral structured tools."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {"ToolService", "ToolDefinition", "ToolResult"}:
        from wsp.tools import service

        return getattr(service, name)
    raise AttributeError(name)


__all__ = ["ToolDefinition", "ToolResult", "ToolService"]
