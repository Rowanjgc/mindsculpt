# MCP adapter

The workbench registry can be served over the **Model Context Protocol**
(MCP) so that tool-using agents, IDEs, and MCP clients reach the same
operations the Python API and the `wsp` CLI already use.

The adapter is a **thin transport layer**. It registers no new behavior, no
caching, no retries, and no agent-only operations. Every MCP request is
validated and executed by the shared `ToolService`, exactly like a CLI
`wsp call` invocation.

## Install and run

The server requires the optional `mcp` dependency:

```bash
pip install -e ".[mcp]"
```

Run the stdio server:

```bash
wsp mcp
```

The server speaks the MCP stdio transport and exits cleanly at EOF. A client
configuration looks like this (any MCP client):

```json
{
  "mcpServers": {
    "wsp-workbench": {
      "command": ".venv/bin/wsp",
      "args": ["mcp"],
      "cwd": "/path/to/project"
    }
  }
}
```

`--workspace` and `--budget` options pass through to the service layer as in
every other command:

```bash
wsp --workspace ./.wsp --budget '{"max_preview_elements": 32}' mcp
```

## Tool surface

`tools/list` returns one MCP tool per registered operation. The surface is
fixed at server construction:

- **Names** are the operation names (for example `inspect_model`,
  `create_patch`, `discover_capabilities`).
- **Input schemas** are generated from the same pydantic input models the
  CLI validates against; there is no second schema definition.
- **Descriptions** carry the operation description, a workspace
  requirement note when applicable, and up to three declared limitations
  (with a pointer to `discover_capabilities` for the rest).
- **Instructions** tell the agent that every result is a `wsp.tool/1.0`
  envelope and that `discover_capabilities` provides progressive
  disclosure.

## Envelope passthrough

For every `tools/call`, the adapter returns the service envelope verbatim:

- as **text content** (pretty-printed JSON), and
- as **structured content** (the same object), with
  `isError = !envelope.ok`.

Failed operations return `ok: false` with a structured `error` containing a
machine-readable `code` (for example `invalid_input`, `resource_limit`,
`unsupported`) and details. An unknown operation or a schema violation is
therefore an ordinary error result, not a transport exception: agents
observe identical semantics across the Python, CLI, and MCP transports. The
adapter never retries and never adjusts tolerances; that remains the
caller's explicit decision.

## Boundaries and limits

- **Stdio only.** HTTP/streamable transports are not implemented; nothing
  listens on the network.
- **No MCP resources, prompts, sampling, or elicitation** are exposed. The
  registry is tools-only.
- **Workspace side effects.** Operations that store artifacts, patches, or
  run records write to the local workspace exactly as they do through the
  CLI. Point the server at a dedicated workspace directory if that matters
  to you.
- **No sandbox.** The MCP client runs `wsp mcp` as your user with filesystem
  access to the workspace and any checkpoint paths it passes. This is the
  same trust model as the CLI, documented in `docs/limitations.md`; it is
  not a security boundary.
- **Budgets still bound output.** `max_output_bytes` is enforced by the
  service layer, so an oversized MCP response fails cleanly instead of
  flooding the client. Note that a single `tools/list` response carries the
  full schema of every operation (roughly 200 KB at the current registry
  size) as one newline-delimited message; SDK-based clients accept this.
- **Versioned dependency.** The `mcp` extra is pinned to the version this
  release was validated against, like the instrumentation backend.
- **Compatibility tests are gated.** The MCP test tier requires the optional
  dependency and is skipped (not silently passed) when it is absent; the
  SDK-free mapping layer is always tested by the default suite.

## Testing

```bash
python -m pytest tests/unit/test_mcp_mapping.py tests/unit/test_mcp_server.py -q
```

The mapping tests (registry coverage, MCP name rules, schema JSON-safety,
workspace notes) run offline without the `mcp` package. The server tests use
an in-memory MCP transport to verify `tools/list` parity with the registry,
envelope passthrough, and structured error behavior.
