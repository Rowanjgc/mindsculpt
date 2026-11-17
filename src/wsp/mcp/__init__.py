"""MCP transport for the WSP tool registry.

The adapter is a thin, provider-independent transport layer over the shared
:class:`wsp.tools.service.ToolService`.  It adds no behavior, no caching, and
no agent-only operations: every MCP request is validated and executed by the
same service behind the Python API and the ``wsp`` CLI.

- :mod:`wsp.mcp.mapping` is SDK-free and testable offline.
- :mod:`wsp.mcp.server` requires the optional ``mcp`` dependency
  (``pip install -e ".[mcp]"``) and serves the registry over the MCP stdio
  transport.
"""
