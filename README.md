# WSP Workbench

An open-source, model-agnostic **Weight-Space Programming Workbench**: bounded
instruments for observing, inspecting, measuring, patching, executing, and
reproducing work on neural-network weights and behavior.

V1.5 is deliberately **tools first**. It is not an autonomous agent, a training
loop, a benchmark platform, or an interpreter for neurons. It provides reliable
primitives that a person or a future tool-using agent can compose, with the same
operations exposed through Python, a CLI, and JSON tool schemas.

> **Status:** V1.5 complete. CPU-only, offline tests and examples. No LLM, cloud
> service, API key, or network access is required.

## What V1 includes

- **Checkpoint inspection** — SafeTensors metadata, tensor enumeration, bounded
  reads, manifests, fingerprints, and checkpoint comparison without loading a
  whole model.
- **Tensor analysis** — statistics, quantiles/histograms, norms, cosine/dot/
  projection comparisons, matrix rank and singular-value summaries, all with
  explicit sampling labels.
- **Activation instrumentation** — bounded forward hooks, input/output capture,
  summary-only capture, token-position selection, comparison, and cleanup on
  error paths.
- **Controlled weight editing** — immutable, versioned patches; replace/add/
  scale/rank-one/projection operations; preview; disjoint patch sets; streaming
  export to a derived checkpoint; rollback by immutable source.
- **Experiments and artifacts** — content-addressed artifacts, run records with
  environment/replay metadata, manifests, patch storage, and experiment
  grouping.
- **Structured interfaces** — versioned JSON schemas and a shared service layer
  behind both the Python API and the `wsp` CLI.

## What V1.1 adds: the Behavioral Observatory

- **Generation traces** — a versioned `wsp.trace/1.1` record of tokens, text,
  log-probabilities (when available), recorded input token ids, exact inputs,
  settings, termination, timing, provenance, checksums, privacy state, and
  explicitly unavailable fields.
- **Collection** — a deterministic scripted backend and an optional local
  Hugging Face backend; imports for WSP JSON, INIF 0.1, and Inspect AI logs.
- **Detectors** — repetition, generation termination, structured output,
  reconsideration markers, and repeated tool actions, with sparse findings,
  evidence ranges, and declared limitations.
- **Contrastive tests** — explicitly authored baseline/variation pairs with a
  declared invariant, stored traces, and evaluator outcomes.
- **Evaluation** — deterministic, reference-answer, JSON-contract, and human
  judgement evaluators that distinguish pass, fail, inconclusive, abstention,
  and error.
- **Failure cases** — a local registry linking observations, controls, findings,
  contrasts, evaluations, annotations, hypotheses, activation captures, and
  patch references; self-contained export and checksum-verified reload.
- **One export privacy policy** — private exports are full-fidelity evidence;
  redacted exports structurally rebuild every trace, case record, and nested
  artifact, recompute their checksums and identifiers, and then scan the
  serialized bundle for every removed string. An export that cannot be proven
  clean is refused, and reduced replay capability is recorded explicitly.

## What V1.2 adds: internal investigation

V1.2 connects observable behavior to reproducible internal measurements. It is
still an instrument kit: it measures, it does not conclude.

- **Verified replay** — a teacher-forced replay plan that resolves the recorded
  token sequence, audits model/tokenizer/template identity, and executes bounded
  forward passes. Four guarantees are reported separately: token-sequence
  reconstruction, forward execution, numerical agreement with recorded
  log-probabilities, and generation reproducibility (always `not-claimed` for
  teacher forcing).
- **Token alignment** — prompt, generation, and full-sequence index spaces with
  character/byte ranges where they are trustworthy, special-token flags, and
  explicit `consuming` vs `predicting` semantics, so a state is never confused
  with the token it predicts. Ambiguous spans are reported, not invented.
- **Targeted capture** — summary statistics or retained vectors at selected
  module and token positions, bounded by element/position/artifact budgets, with
  hook cleanup on error paths and a parameter-immutability check on every pass.
- **Optional instrumentation backend** — an isolated NNsight/nnterp adapter that
  normalises layer paths, refuses unsupported architectures instead of guessing,
  and records its own backend identity. The plain PyTorch hook backend stays the
  default and needs no extra dependencies.
- **Controlled comparisons** — position-aligned numerical comparison of two
  captures with compatibility validation, shared metric implementations, and
  reviewed pairings. No dimension is ever labelled with a concept.
- **Investigation evidence** — findings, replays, alignments, captures,
  comparisons, and unverified hypotheses linked to a failure case, exported and
  reloaded with rewritten references and explicit omission records.

## What V1.3 adds: controlled causal interventions

V1.3 adds the instruments needed to *intervene* on an activation during an
execution and measure the consequences. It builds the experiment; it does not
claim a discovery.

- **Activation interventions** — a versioned `wsp.activation-intervention/1.0`
  specification naming the model, execution identity, module, capture point,
  tensor role, position semantics, slice, operation, replacement source,
  dtype conversion policy, and resource limits. No executable scripts.
- **Four operations** — identity (the mandatory negative control), zero
  ablation, mean ablation with mandatory provenance, and donor activation
  patching with shape/dtype/model compatibility checks.
- **Reversible execution** — one forward hook modifies only the selected region
  of the selected tensor, clones before writing, preserves structured outputs,
  never edits parameters, and cleans up on success and failure.
- **Paired counterfactual measurement** — baseline, identity control, and
  intervention with identical weights and inputs, plus an optional alternative
  control. Downstream logits, token log-probabilities, activation differences,
  and effect sizes are recorded. A failed identity control invalidates the
  comparison rather than being hidden.
- **Controlled continuations** — a separate, explicitly uncached full-prefix
  greedy mode comparing generated tokens, termination, longest common prefix,
  and first divergence. Cached generation is not claimed.
- **Causal evidence integration** — intervention and continuation references
  attach to existing failure cases and evidence bundles, survive export/reload,
  and are transformed or omitted by the same privacy system as every other
  artifact.

## What V1.4 adds: temporal and compositional interventions

V1.3 can only intervene during the first prefix forward. V1.4 makes *when* an
intervention fires a first-class, reproducible part of the experiment, and lets
several interventions run in one plan.

- **Temporal schedules** — a versioned `wsp.intervention-schedule/1.0` record
  selecting the first prefix forward, explicit generation steps, a bounded
  inclusive or half-open step range, or a bounded repeated stride. Schedules are
  declarative data; arbitrary executable predicates are rejected.
- **Explicit timing conventions** — prompt index, generation step, generated
  token index, full-sequence index, forward invocation, activation consumption
  position, and next-token prediction position are defined once in
  `wsp.interventions.schedule.TIMING_CONVENTIONS` and used consistently by
  replay, capture, intervention, and generation. Generation step `k` is the
  forward that consumes the prefix plus `k` generated tokens and predicts
  generated token `k`.
- **Intervention plans** — `wsp.intervention-plan/1.0` composes several
  interventions, each with its own schedule, site, tensor selection, operation,
  order, and position mode. Deterministic ordering and collision detection are
  part of validation: overlapping same-module regions at overlapping steps are
  rejected unless both specifications explicitly declare `allow_overlap`.
- **Individual versus combined comparisons** — `wsp.compositional-comparison/1.0`
  measures baseline, identity, A, B, and A+B conditions with identical weights,
  inputs, and precision, and reports a descriptive interaction contrast
  `I_AB = (Y_AB - Y_0) - (Y_A - Y_0) - (Y_B - Y_0)` on an explicitly chosen
  scalar. Discrete outcomes such as generated length are marked non-continuous.
- **Verified staged generation** — an optional cached path built on the model's
  own generation interfaces. A staged run is only usable after stepwise logits,
  token choices, termination, and identity control agree with the uncached
  reference within declared tolerances. Hybrid/recurrent architectures are
  rejected explicitly; Qwen3.5-style hybrid state is not claimed.
- **Event logs and evidence** — `wsp.intervention-event-log/1.0`,
  `wsp.temporal-intervention/1.0`, `wsp.staged-verification/1.0`, and
  `wsp.staged-continuation/1.0` record which event fired, where, with what
  before/after summary, and integrate with failure cases, export, and the
  existing privacy policy.

## What V1.5 adds: event-driven investigation

V1.4 could schedule an intervention at a known generation step. V1.5 connects
that machinery to observed behavior and makes the path from observation to a
controlled internal experiment explicit.

- **Deterministic behavioral events** — `wsp.behavioral-event/1.0` records token
  occurrences, repeated sequences, the earliest repetition, detector findings,
  manual marks, and text matches. Events state absolute token positions,
  generation steps, alignment status, prompt offsets, and provenance. A
  repetition is a measurement, not a diagnosis.
- **Online event triggers** — `wsp.event-trigger/1.0` declares a delay, firing
  mode, bounded step window, maximum trigger count, and position mode. An event
  observed after generated token `k` cannot fire before step `k + 1 + delay`.
  Events with no subsequent forward are reported as `not_reached`; the system
  never claims a firing that did not happen.
- **Retrospective triggers** — an existing event may be investigated at an
  explicitly selected replay step, including a step before the event. Every such
  record is labelled retrospective and is distinguished from an online trigger.
- **Fixed-context counterfactuals** — `wsp.fixed-context-replay/1.0` reconstructs
  the exact recorded prefix; `wsp.fixed-context-comparison/1.0` runs baseline,
  mandatory identity control, and one activation intervention on identical token
  identifiers. Measurements include selected log-probabilities, logits, and
  optional downstream activations. Optional continuations are kept separate and
  their divergence is reported.
- **Investigation recipes** — three versioned, previewable recipes compose the
  existing operations: investigate a finding, test an activation hypothesis, or
  investigate a repetition event. A failed recipe preserves completed artifacts
  and records the failing step; it is never reported as completed.
- **Agent-oriented discovery and dossiers** — operations now carry grouping,
  prerequisites, backends, artifact kinds, cost class, side effects, privacy
  notes, limitations, follow-ups, and examples. `discover_capabilities` provides
  deterministic progressive disclosure. `wsp.investigation-dossier/1.0`
  separates observations, measurements, controls, hypotheses, limitations, and
  next operations.
- **Evidence and privacy** — events, replays, comparisons, recipe runs, and
  dossiers are content-addressed artifacts, attach to failure cases, and are
  transformed or omitted by the existing single exporter. New artifact kinds
  are registered explicitly; unregistered kinds fail closed.

The repetition in the offline example is a constructed fixture. V1.5 makes no
claim about discovering a reasoning-loop mechanism.

## Install

Requirements: Python 3.11+ and PyTorch. CPU-only installation is the supported
baseline.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Optional local Hugging Face support:

```bash
pip install -e ".[hf]"
```

ROCm is not required for installation or tests. GPU execution is optional and
not promised to be bitwise reproducible across stacks.

## Quick start

The examples are fully offline and create their own tiny deterministic model.

```bash
python examples/make_tiny_model.py
python examples/inspect_and_patch.py
```

`examples/inspect_and_patch.py` performs the complete workflow:

1. opens the checkpoint without loading the model;
2. prints a manifest and tensor statistics;
3. captures `fc1` activations in a baseline forward pass;
4. creates and previews an `add 0.125` patch on `fc1.bias`;
5. applies it to a derived checkpoint;
6. reloads the derived checkpoint, verifies the expected parameter values, and
   compares outputs and activations;
7. reverts by selecting the untouched source and verifies identical output;
8. stores the manifest, patch, patch set, and run metadata in a workspace.

The source checkpoint hash is checked before and after. It never changes.

### Behavioral workflow (offline)

```bash
python examples/behavioral_workflow.py
```

The example records synthetic healthy and repetitive traces, detects a
repetition anomaly, compares the traces, runs a controlled comparison with an
explicit factual invariant, stores a case, exports it, reloads it in a fresh
workspace, and re-scans the restored trace. It is an infrastructure
demonstration, not a scientific result.

Minimal CLI version:

```bash
wsp init
wsp record-trace "What is the capital of Australia?" --output "Canberra"
wsp traces
wsp scan-trace tr_... --detector core.repetition
wsp detectors
```

### Internal investigation workflow (offline)

```bash
python examples/investigation_walkthrough.py
```

It records a looping generation and a terminating control, resolves the detector
finding onto token positions, plans and verifies a teacher-forced replay,
captures selected states at a chosen module, compares an identical-context
control and a genuinely different input, attaches the measurements to a failure
case as investigation evidence, exports the case, imports it into a fresh
workspace, verifies every reference and checksum, confirms the model parameters
did not change, and finally demonstrates a redacted export. It is an
infrastructure demonstration, not a scientific result.

The same walkthrough runs against a real local checkpoint:

```bash
python examples/investigation_walkthrough.py \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<revision>
```

Tier-3 integration tests are separate from the default suite and never download
weights:

```bash
WSP_REAL_MODEL=<local checkpoint directory> \
    .venv/bin/python -m pytest tests/integration/test_real_checkpoint.py -q
```

A skipped tier-3 run means real-checkpoint compatibility is unverified on that
machine, not that it passed.

### Causal intervention workflow (offline)

```bash
python examples/causal_investigation.py
```

It builds a deterministic transformer, records an observation and a control
trace, verifies a replay, captures the targeted activation, runs an identity
control (exactly equal to the baseline), performs a zero ablation and a donor
patch, compares downstream logits and token log-probabilities, runs a bounded
greedy continuation and identifies the first divergence, attaches the experiment
to a failure case, exports and reloads it in a fresh workspace, verifies
checksums and references, confirms the parameters never changed, and produces a
privacy-safe redacted export.

Minimal CLI version:

```bash
wsp create-activation-intervention --name "zero ablation" \
    --capture a_... --module blocks.1 --operation zero --position 2
wsp preview-activation-intervention --intervention a_... \
    --source '{"kind": "synthetic"}' --tokens 1,5,6,7
wsp run-paired-intervention --intervention a_... \
    --source '{"kind": "synthetic"}' --tokens 1,5,6,7 \
    --measure-module blocks.2
```

### CLI walkthrough

```bash
# A local workspace (defaults to ./.wsp) for manifests, patches, and run records.
wsp init

wsp inspect-model examples/out/tiny_mlp.safetensors --file-sha256
wsp list-tensors examples/out/tiny_mlp.safetensors --contains fc
wsp inspect-tensor examples/out/tiny_mlp.safetensors fc1.bias --preview-elements 8
wsp call analyze_tensor \
  --json '{"path":"examples/out/tiny_mlp.safetensors","tensor":"fc1.weight","operations":["stats","norms"]}'
```

Create a patch using the checked-in input example, then preview and apply it:

```bash
wsp --workspace examples/out/workspace call create_patch \
  --input examples/create_patch.json

# Use the patch_id printed by the previous command.
wsp --workspace examples/out/workspace call preview_patch \
  --json '{"path":"examples/out/tiny_mlp.safetensors","patch_id":"p_..."}'

wsp --workspace examples/out/workspace call apply_patch \
  --json '{"path":"examples/out/tiny_mlp.safetensors","output_path":"examples/out/cli_patched.safetensors","patch_ids":["p_..."]}'
```

Every command prints JSON. A failed operation prints a JSON error and exits
with status 1. Use `wsp tools` for the operation registry and
`wsp tools --operation NAME` for the exact input schema.

### Python API

```python
from wsp.core.budget import Budget
from wsp.inspection.factory import open_checkpoint
from wsp.inspection.manifest import build_manifest
from wsp.tensors.stats import basic_stats

with open_checkpoint("examples/out/tiny_mlp.safetensors", compute_file_sha256=True) as source:
    manifest = build_manifest(source, include_tensors=True)
    info = source.tensor_info("fc1.bias")
    values = source.read_full("fc1.bias", max_elements=Budget().max_tensor_numel)

print(manifest.identity.structure_fingerprint)
print(info.shape, info.dtype)
print(basic_stats(values, storage_dtype=info.dtype))
```

The structured service is also available directly:

```python
from wsp.tools.service import ToolService

service = ToolService(raise_errors=True)
result = service.call("inspect_model", {"path": "examples/out/tiny_mlp.safetensors"})
```

## Safety properties

- Source checkpoints are never modified in place.
- Patch application writes a new checkpoint atomically and refuses to overwrite
  the source or an existing destination unless explicitly allowed.
- Checkpoints are data. Pickle-based loading is rejected; bundled code is never
  executed. A Python model adapter is **not** a sandbox.
- File paths are constrained to explicit destinations and workspaces.
- Measurements are separated from interpretations; sampled/approximate results
  are labeled and never presented as full results.
- CPU is the deterministic baseline. GPU nondeterminism is surfaced, not hidden.
- No telemetry, no automatic downloads, no required cloud services, no API keys.
- Generated data is excluded from Git by default.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — layers, data flow, storage
  formats, budgets, precision, trust boundaries.
- [`docs/behavioral-observatory.md`](docs/behavioral-observatory.md) — trace
  lifecycle, detectors, evaluators, contrasts, cases, and limits.
- [`docs/trace-format.md`](docs/trace-format.md) — the versioned trace schema,
  provenance, privacy, and INIF/Inspect interoperability.
- [`docs/failure-cases.md`](docs/failure-cases.md) — investigation cases,
  review states, hypotheses, and export/reload bundles.
- [`docs/prior-art-and-integration.md`](docs/prior-art-and-integration.md) —
  evaluated upstream projects, tested versions, and integration decisions.
- [`docs/tool-api.md`](docs/tool-api.md) — every structured operation, schemas,
  errors, limits, and examples.
- [`docs/mcp.md`](docs/mcp.md) — serving the registry over the Model Context
  Protocol: a thin stdio transport with envelope passthrough.
- [`docs/agent-evaluation.md`](docs/agent-evaluation.md) — first (scripted)
  evaluation of the tool surface the way a tool-using agent reaches it.
- [`docs/causal-interventions.md`](docs/causal-interventions.md) — intervention
  specs, zero/mean/identity/donor operations, validation, paired measurement,
  and limitations.
- [`docs/counterfactual-continuations.md`](docs/counterfactual-continuations.md) —
  uncached greedy continuations, position semantics, divergence, and limits.
- [`docs/temporal-interventions.md`](docs/temporal-interventions.md) — generation
  steps, token positions, schedules, event logs, and uncached semantics.
- [`docs/event-driven-investigations.md`](docs/event-driven-investigations.md) —
  behavioral events, online/retrospective triggers, timing, and budgets.
- [`docs/fixed-context-counterfactuals.md`](docs/fixed-context-counterfactuals.md) —
  exact prefix reconstruction, baseline/identity/intervention controls, and
  fixed-context measurement.
- [`docs/investigation-recipes.md`](docs/investigation-recipes.md) — the three
  built-in recipes, previews, failure records, and limits.
- [`docs/agent-tool-usage.md`](docs/agent-tool-usage.md) — discovery, recipe
  choice, missing prerequisites, failed controls, and context budgeting.
- [`docs/intervention-plans.md`](docs/intervention-plans.md) — multi-site plans,
  ordering, collisions, identity controls, and compositional comparisons.
- [`docs/staged-generation.md`](docs/staged-generation.md) — staged/cached
  execution, equivalence verification, isolation, and supported architectures.
- [`docs/patch-format.md`](docs/patch-format.md) — the versioned patch format,
  operations, compatibility checks, casting, export, and rollback.
- [`docs/limitations.md`](docs/limitations.md) — supported formats,
  architecture/execution assumptions, safety boundaries, and known gaps.
- [`docs/roadmap.md`](docs/roadmap.md) — completed V1 capabilities and
  deliberately deferred/future work.
- [`examples/README.md`](examples/README.md) — offline examples.

## Development

```bash
python -m pytest -q
```

The test suite is CPU-only and offline. It includes unit tests for core,
patches, activations/execution, workspace/tools, and CLI behavior, plus an
end-to-end integration test covering inspection, patching, export, reload,
comparison, rollback, provenance, and failure safety.

## Research context and attribution

The project was informed by the public Weight-Space Programming research and its
released experiment logs:

- <https://rikiyat.github.io/blog/weight-space-programming/>
- <https://github.com/RikiyaT/lm_understand-logs>

The workbench is an independent, general-purpose implementation of engineering
primitives suggested by that work (bounded model access, explicit tensor
metadata, reversible edits, measurement provenance, execution budgets). No
upstream code or experiment assets are bundled here. Upstream materials have
their own licenses, including mixed content in the logs repository; consult
those sources for their terms. See `docs/limitations.md` for the scope boundary.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
