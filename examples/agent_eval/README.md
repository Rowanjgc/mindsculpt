# Autonomous LLM agent evaluation of the MCP tool surface

This directory contains the harness and task files used for the first
autonomous (general-purpose LLM) evaluation of the WSP MCP server, recorded
in [`docs/agent-evaluation.md`](../../docs/agent-evaluation.md).

**This is not an offline example.** It needs network access, an LLM API key,
and real inference (typically a few dollars of hosted-model spend per run).
Every other example in `examples/` runs offline against tiny deterministic
models; this one deliberately exercises a real checkpoint through a real
model-in-the-loop agent.

## What the harness does

`harness.py` connects to `wsp mcp --workspace <ws> --auto-init` over the MCP
stdio transport and hands control to a hosted LLM (any OpenAI-compatible
chat-with-tools endpoint):

- The **system prompt is the server's own MCP instructions** — nothing about
  WSP is hard-coded into the prompt beyond envelope-semantics framing.
- Tools are rendered as OpenAI function definitions. By default the **full
  80-tool surface** is exposed; `--progressive` starts with three entry
  points and widens from `discover_capabilities` responses.
- Every assistant/tool message is recorded into a full JSON transcript with
  per-step usage and cost (`--out`).

## Usage

```bash
# requires: pip install -e ".[dev]" (the mcp extra is in dev), plus OPENCODE_API_KEY
OPENCODE_API_KEY=... .venv/bin/python harness.py \
  --model glm-5.3 \
  --workspace /tmp/wsp-agent-eval \
  --task-file t3_hybrid_trap.txt \
  --out /tmp/t3.json \
  --max-steps 8
```

Task files (prompts, not scripts):

- `t1_investigation.txt` — full investigation arc on a real checkpoint:
  identify the model, record a trace, detect behavioral events, capture
  activations, report artifact IDs and an interpretation.
- `t2_directory_trap.txt` — asks for the model's architecture and parameter
  count from a checkpoint **directory**; tests whether the agent recovers
  from the actionable directory-path error.
- `t3_hybrid_trap.txt` — asks for staged/cached generation on a hybrid
  architecture; tests whether the agent reports the `unsupported` rejection
  honestly instead of claiming success.

Set `WSP_REAL_MODEL`-style local paths inside the task text to point at your
own checkpoint; the tasks never download weights.

## Notes

- Hosted models may emit long hidden reasoning; the harness retries an
  empty assistant message once and leaves the token cap to the endpoint
  default.
- Envelope bodies are truncated to 12000 characters before entering model
  context; transcripts keep the truncation marker.
- The harness is an evaluation instrument, not a supported client: it pins
  the MCP SDK's snake-case attribute quirks and the OpenAI-compatible wire
  format it was run against.
