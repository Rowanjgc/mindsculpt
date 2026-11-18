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
   the parent `wsp` command, so `wsp mcp --workspace X` fails.
   **Fixed**: the subcommand now accepts `--workspace/-w` and `--budget`
   directly (they override the global options).
2. **Uninitialized workspace kills the server silently (as far as the client
   can tell).** The server exits with a raw traceback at startup; an MCP
   client only sees "Connection closed" with zero diagnostics.
   **Fixed**: `wsp mcp` now validates the workspace before starting and, on
   failure, writes the structured `wsp.mcp/1.0` error envelope to stderr and
   exits 1 (clients surface stderr on connection failure). A new
   `--auto-init` flag creates and initializes a missing workspace at
   startup.
3. **Workspace notes assume a shell.** Tool descriptions say to
   `run 'wsp init'`, but an MCP agent has no shell.
   **Fixed**: workspace notes and runtime hints now name the agent-relevant
   actions — start `wsp mcp` with an initialized workspace, or restart with
   `--auto-init` — and two hints that referenced a nonexistent
   `wsp workspace init` command were corrected to `wsp init`.
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
   the first probe does miss. **Documented** (see
   `agent-tool-usage.md`); restructuring the versioned summary envelope is
   out of scope for now. **Open (minor).**
6. **Naive module choice on hybrid models wastes a call.** Captures over
   non-attention modules are skipped with an actionable warning, but
   `list_capture_points` does not pre-annotate which modules expose a
   sequence dimension suitable for capture. **Partially improved**: the skip
   warning now includes the observed tensor shape, which makes the next
   choice concrete. A static suitability annotation is not possible without
   runtime shapes, so `list_capture_points` remains unannotated. **Open
   (candidate).**
7. **Version metadata staleness.** After a version bump, a stale editable
   install still reported the old version through MCP `serverInfo`. Process
   note: refresh the editable install (`uv pip install -e . --no-deps` or
   equivalent) after bumps. **Fixed (process).**
8. **`list_capture_points` kind filter matches nothing silently.** Found in
   the autonomous evaluation: guessed labels (`linear_attention`,
   `gated_delta_net`, `ssm`) returned an empty point list with no
   explanation — and the canonical label is in fact `linear-attention`
   (hyphenated), which is listed nowhere. **Fixed**: canonical kind labels
   are now exposed (`kind_labels` in every response and in the `kinds`
   schema description), and an empty filtered result carries a note naming
   the canonical labels; `include_submodules=false` (root-only listings)
   says so when it produces nothing.
9. **Hosted-model reasoning can exhaust a token cap.** Not a workbench
   defect: the agent model sometimes emitted its entire budget as hidden
   reasoning (`finish_reason=length`, empty content), which the harness now
   retries once and leaves uncapped by default. Recorded here because a
   future evaluation with other models will hit the same artifact.

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

## 6. Autonomous LLM evaluation (second pass)

After the scripted pass, the surface was handed to an actual
general-purpose LLM agent: GLM-5.3 (hosted, temperature 0.1) driving the MCP
stdio server through the harness in
[`examples/agent_eval/`](../examples/agent_eval/). The system prompt is the
server's own MCP instructions plus short envelope-semantics framing; the
full 80-tool surface is exposed; nothing about WSP is hard-coded beyond
that. Full JSON transcripts (every message, per-step usage and cost) were
recorded; the findings below are taken from them.

Tasks (one run each, one hybrid checkpoint
Qwen3.5-0.8B, CPU):

| Task | Steps | Errors | Outcome | Cost |
| --- | --- | --- | --- | --- |
| `t1_investigation` — full arc: identify the model, record a trace, detect `Paris` events, capture activations at the event, report artifact IDs and interpretation | 13 | 1 | **Success.** Complete report with verified replay (100% top-1 agreement, max log-prob diff ≈ 0.009), captured artifact IDs, and observations explicitly separated from interpretation | $0.49 |
| `t2_directory_trap` — architecture + parameter count from the checkpoint *directory* | 4 | 1 | **Success.** Walked into the trap, recovered on the first retry using the weights filename listed by the error envelope, and delivered a full structural inventory (873,438,784 params: 752,393,024 language model + vision tower + MTP head) | $0.13 |
| `t3_hybrid_trap` — generate 8 tokens via the staged/cached path on a hybrid architecture | 4 | 1 | **Success.** The `unsupported` rejection fired; the agent reported it accurately and explained that generation succeeded via the standard deterministic path | $0.13 |

What this pass demonstrates that the scripted pass could not:

- **Judgment-free entry.** The agent chose `discover_capabilities` first
  with a well-formed query on its own — the progressive-disclosure design
  works without being scripted.
- **Error-driven recovery by judgment.** Every designed failure (directory
  path, staged generation on a hybrid) was handled from the envelope alone:
  the agent corrected its own payloads and reported rejections honestly
  instead of claiming success. Finding 4's fix proved itself in practice:
  the directory error now names the real weights file, and both agents
  recovered on the first retry (previously seven failed guesses).
- **Honest reporting.** The t1 report quoted the tool's own limitations
  ("numerical observations from a teacher-forced replay ... not evidence of
  a mechanism"), disclosed a tokenization caveat (chat-template re-encoding
  made character offsets unavailable, token alignment stayed exact), and
  answered the "unsupported operations" probe with "none" plus the one real
  error it had made.
- **Exploration depth.** 13 steps of unprompted structure exploration
  (tensor paging, kind probes, module selection) with per-step cost
  accounting (≈1.1M prompt tokens, 24k completion tokens total across the
  arc).

What remains open:

- One agent model, one checkpoint, three tasks, no repeats: this measures
  the surface against one LLM at one temperature, not agent-population
  behavior.
- The harness truncates envelopes to 12k characters and runs stdio only;
  neither dimension was varied.
- Finding 5 (deeply nested `record_trace` summary) was observed again by
  the autonomous agent and remains open (minor).
- The recipes (multi-step investigation programs) were not the entry path
  in any autonomous run; evaluating recipe-guided behavior with LLM agents
  is still untested.
