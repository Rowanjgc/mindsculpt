# Agent evaluation of the tool surface

This document records what happens when the registry is exercised the way a
tool-using agent would exercise it: through the MCP transport only, judging
each step by descriptions, input schemas, server instructions, and result
envelopes — never by reading WSP source.

Status after the first evaluation (2026-11-17): the recovery loop —
**call → read `ok:false` envelope → corrected call** — worked at every failure
point with zero source reading on a real hybrid checkpoint. The honest
caveat up front: the evaluation was **scripted, not autonomous**. A driver
made the choices, so this validates metadata sufficiency and envelope
recoverability, not LLM judgment. Real general-purpose language-model
evaluation remains open.

## 1. Scope and method

- **Transport**: MCP stdio server (`wsp --workspace … mcp`) driven by the
  MCP Python SDK client, exactly as an external MCP client would connect.
- **Model**: a real hybrid checkpoint (Qwen3.5-0.8B; `Qwen3_5ForCausalLM`,
  752,393,024 parameters, 248,044-token Qwen2Tokenizer vocabulary) on CPU,
  loaded through the optional Hugging Face backend (transformers 5.10.2).
- **Naive-agent rules**: the driver may read tool names, descriptions, and
  input schemas from `tools/list`, plus server instructions and any envelope
  returned by a call. It may not read WSP source, docs, or tests. Errors are
  recovered from `error.code` and `error.details` alone.
- **Arc**: initialize → `tools/list` → `discover_capabilities` →
  `inspect_model` → `record_trace` → `detect_behavioral_events` →
  `prepare_fixed_context_replay` → `capture_trace_activations`, plus probes
  of failure paths (`prepare_staged_generation` on the hybrid model,
  unknown operation names, schema violations).

## 2. What worked

### Entry point discoverability

A naive name scan of the 80 registered tools finds exactly one obvious entry
point, `discover_capabilities`, and the server instructions reinforce it
("call discover_capabilities first"). Descriptions are bounded (52–417
characters in this registry) and every input schema is self-contained: `$defs`
references are inlined where the MCP surface requires it, so no additional
fetch is needed to build a valid payload.

### Progressive disclosure

`discover_capabilities` returns 9 operation groups (inspect, observe,
compare, replay, capture, intervene, evaluate, investigate, export),
per-group `next_operations` with reasons, and 3 recipes (for example
`investigate_behavioral_finding`, 7 steps), under schema
`wsp.capabilities/1.0`. An agent can route from intent to operation without
enumerating all 80 tools in detail.

### Error-driven recovery

Three distinct failure loops were recovered purely from envelopes:

- **Unknown operation** → `ok:false` with `error.details.available` listing
  the valid operations; the next call used a listed name.
- **Schema violation** → per-field `loc`/`msg`/`type` details; a guessed
  field name (`source`) was corrected to `path` from the error alone.
- **Unsupported capability** → probing `prepare_staged_generation` on the
  hybrid model returned `error.code "unsupported"` with
  `details.reason "non-attention layers present: ['linear_attention']"` —
  a fail-closed rejection that states *why*, surfaced through MCP as an
  ordinary error result rather than a transport exception.

### A real investigation arc

- `record_trace` with the Hugging Face backend produced a durable trace
  (`tr_…`); provenance via `inspect_trace` includes backend versions, dtype
  (bfloat16), determinism flag, and a tokenizer fingerprint.
- `detect_behavioral_events` with a text pattern found an event (`ev_…`) at a
  specific generation step.
- `prepare_fixed_context_replay` accepted the trace and event artifact IDs
  and reported the replay step.
- `capture_trace_activations` first attempted naive modules and returned
  `ok:true` with actionable skip warnings (the tensors do not expose an
  unambiguous sequence dimension of length N; capture skipped). Picking leaf
  capture points from `list_capture_points` (`…self_attn.q_proj` at two
  layers) produced a clean 12-measurement capture with honest
  `data.unavailable` notes (summary captures do not retain values; use
  `mode="selected"` for elementwise comparison; teacher-forced replay does
  not regenerate tokens).

## 3. Friction findings

Numbered for reference; each notes its disposition.

1. **`--workspace` is not accepted by the subcommand.**
   `wsp mcp --help` shows no `--workspace` option; it is a global option on
   the parent `wsp` command, so `wsp mcp --workspace X` fails. Documented in
   `mcp.md` (client configs must pass it at the top level). Candidate: accept
   the option on the subcommand as well. **Open.**
2. **Uninitialized workspace kills the server silently (as far as the client
   can tell).** The server exits with a raw traceback at startup; an MCP
   client only sees "Connection closed" with zero diagnostics. Candidate:
   a clean startup error, create-on-demand, or a pre-init usable
   `workspace_status`. **Open.**
3. **Workspace notes assume a shell.** Tool descriptions say to
   `run 'wsp init'`, but an MCP agent has no shell. Reword for the MCP
   surface or auto-init. **Open.**
4. **Directory paths to `inspect_model` produced a misleading error.**
   Passing a Hugging Face snapshot directory returned
   `UnsupportedError "…pickle-based checkpoints are not supported for
   safety"`, when the real problem is that a directory was passed.
   **Fixed**: `open_checkpoint` now raises `InputError` ("checkpoint path is
   a directory; pass the .safetensors weights file inside it, for example
   `model.safetensors`"), with a regression test
   (`test_directory_checkpoint_path_gets_actionable_error`).
5. **Deeply nested success data.** `record_trace` puts its summary under
   `summary.trace.artifact.metadata`; discoverable in a couple of probes, but
   the first probe does miss. **Open (minor).**
6. **Naive module choice on hybrid models wastes a call.** Captures over
   non-attention modules are skipped with an actionable warning, but
   `list_capture_points` does not pre-annotate which modules expose a
   sequence dimension suitable for capture. Candidate: annotate capture
   points. **Open.**
7. **Version metadata staleness.** After a version bump, a stale editable
   install still reported the old version through MCP `serverInfo`. Process
   note: refresh the editable install (`uv pip install -e . --no-deps` or
   equivalent) after bumps. **Fixed (process).**

## 4. Honest limits

- **Scripted, not autonomous.** The choices were made by an evaluation
  driver, not a general-purpose LLM. This validates that the metadata,
  instructions, and envelopes are *sufficient in principle*; it does not
  measure LLM judgment, prompt robustness, context-budget behavior, or
  long-horizon drift.
- **Single model, single task, single transport.** One hybrid checkpoint,
  one behavioral-arc recipe shape, stdio only, CPU only, no parallel
  sessions.
- **The MCP-path trial is not automated in CI.** The two real-checkpoint
  integration tests (`tests/integration/test_real_checkpoint.py`,
  `tests/integration/test_real_event_investigation.py`) cover the Python
  path and run locally with `WSP_REAL_MODEL=<checkpoint dir>`; the MCP-path
  trial was executed as ad-hoc scripts against a live stdio server.

## 5. Reproducing the testable portion

The checkpoint-backed sanity tests used by this evaluation never download
weights; point them at a local checkpoint directory:

```bash
WSP_REAL_MODEL=/path/to/local/checkpoint \
  .venv/bin/python -m pytest tests/integration/test_real_checkpoint.py -q
WSP_REAL_MODEL=/path/to/local/checkpoint \
  .venv/bin/python -m pytest tests/integration/test_real_event_investigation.py -q
```

A skipped run means unverified, not passing. See `mcp.md` for how to connect
a real MCP client to the server.
