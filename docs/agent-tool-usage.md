# Agent tool usage guide

This guide is for a general-purpose language model using the structured tool
interface, and for the people evaluating that workflow. It does not assume
knowledge of PyTorch hooks or intervention internals.

## 1. Discover first

Call `discover_capabilities` with a short question, for example:

```json
{"query": "repetition in a recorded trace", "detail": "compact"}
```

It returns:

- the nine capability groups and their sizes;
- a bounded set of matching operations;
- relevant recipes with required inputs;
- deterministic follow-up suggestions.

Do not read the entire registry unless you need it. Request one operation's
schema with `tools --operation <name>` or `describe(operation)`.

## 2. Choose a recipe before composing low-level calls

| Goal | Recipe |
| --- | --- |
| Resolve a detector finding and capture its context | `investigate_behavioral_finding` |
| Test whether changing an activation changes a measurement | `test_activation_hypothesis` |
| Measure a repetition event and test a target | `investigate_repetition_event` |

Preview it first:

```json
{"recipe_id": "test_activation_hypothesis", "inputs": {"trace_id": "tr_..."}}
```

The preview states missing inputs and prerequisites. If `executable` is false,
do not run it; supply the missing items or choose a different recipe.

## 3. Construct a valid investigation

Minimum viable sequence:

1. `inspect_trace` — confirm the trace and its exact token ids.
2. `detect_behavioral_events` — resolve an event and read its alignment.
3. `prepare_fixed_context_replay` — reconstruct the exact prefix.
4. `run_fixed_context_comparison` — baseline, identity, intervention.
5. `create_investigation_dossier` — concise evidence summary.
6. `attach_event_evidence` — attach to a case when one exists.

An intervention must target the selected replay position. A trace without exact
token identifiers cannot support fixed-context replay.

## 4. Recognise missing prerequisites

Stop and report, rather than guessing, when:

- the trace has no recorded token ids;
- an event alignment is `ambiguous` or `unavailable`;
- the requested module is not present;
- the intervention sequence length or execution mode does not match;
- the replay model structure does not match the trace;
- the budget would be exceeded.

Every tool returns a machine-readable error code and details. `invalid_input`
is not a scientific result.

## 5. Interpret failed controls

- `identity_control.passed = false` means the instrumentation changed the
  computation. The comparison is `invalid`; do not attribute the effect to the
  intervention.
- No identity control means the result is `limited`.
- `not_reached` means the event had no eligible subsequent forward. No
  intervention occurred.
- `rejected` means a scheduled operation failed. The run is invalid for
  interpretation.

Never describe a failed control as a positive finding.

## 6. Separate numerical effects from scientific conclusions

A dossier separates:

- **Observations** — what the model did.
- **Measurements** — numerical values and positions.
- **Controls** — which checks passed.
- **Hypotheses** — what an investigator proposed, if anything.
- **Limitations** — what cannot be concluded.
- **Next operations** — deterministic suggestions.

A log-probability delta, an activation difference, or a changed token is a
measurement. It is not a mechanism, a repair, or a validated weight edit. The
workbench contains no automatic circuit interpretation.

## 7. Request deeper evidence without flooding context

- Use `inspect_*` operations with `max_entries`/`include_*` flags.
- Keep `detail="compact"` for discovery.
- Prefer dossier artifact references over dumping full traces.
- Ask for selected logit positions and selected modules; do not request all
  layers and all tokens.
- Use `max_output_bytes` and operation budgets to bound responses.

## 8. Compact example

```text
discover_capabilities {"query": "repetition", "detail": "compact"}
preview_investigation_recipe {"recipe_id": "investigate_repetition_event", "inputs": {...}}
run_investigation_recipe {"recipe_id": "investigate_repetition_event", "inputs": {...}}
inspect_investigation_dossier {"dossier_artifact_id": "a_..."}
```

If the recipe fails, read `inspect_recipe_run` and report the failing step. Do
not silently retry with weaker validation.

## 9. Where to find result data in envelopes

Success envelopes nest some payloads more deeply than the top-level `data`
object. The most common case: `record_trace` puts its summary under
`data.summary.trace.artifact.metadata`, so the first probe into the envelope
misses it. When a tool returns fewer fields than expected, probe one level
deeper before assuming the data is absent.

## Current status

A first evaluation — scripted, not autonomous — against a real hybrid
checkpoint through the MCP transport is recorded in
[agent-evaluation.md](agent-evaluation.md). The call → error → corrected-call
recovery loop worked at every failure point with zero source reading; the
metadata, recipes, and dossiers supplied here were sufficient in principle.
Evaluation with actual general-purpose language models (autonomous judgment,
prompt robustness, context budgets) remains open.
