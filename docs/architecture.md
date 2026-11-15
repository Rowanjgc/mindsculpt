# Architecture

WSP Workbench is a small set of composable layers. Each layer has one job and
can be used without the layers above it. The Python API, structured tool
service, and CLI are three entry points onto the same implementation.

```text
CLI (wsp) ───────────────┐
Structured tool service ─┼─> workspace / run records
Python API ──────────────┘        │
                                   ├─ inspection ──> SafeTensors / torch module
                                   ├─ tensors
                                   ├─ activations ─> execution ─> PyTorch model
                                   └─ patches ─────> streaming SafeTensors export
```

There is no central “workbench” object that owns all behavior. The service is a
thin validation and dispatch layer; the numerical and file-format logic lives
in the domain modules.

## Package map

| Module | Responsibility |
| --- | --- |
| `wsp.core` | Errors, JSON/atomic writes, hashing, dtype handling, selections, budgets, shared result models |
| `wsp.inspection` | Header-only SafeTensors access, live `torch.nn.Module` metadata, manifests, checkpoint comparison |
| `wsp.tensors` | Preview, statistics, difference metrics, quantiles/histograms/norms/SVD/rank |
| `wsp.activations` | Bounded forward hooks, activation summaries, comparison, summary aggregation |
| `wsp.patches` | Immutable patch model, operations, validation, patch sets, streamed export |
| `wsp.execution` | Inference-mode forward passes, model loading, output comparison, offline demo models |
| `wsp.workspace` | Local content-addressed artifacts, run records, experiments, patch/trace/case storage |
| `wsp.behavior` | Trace schema/storage, generation backends, detectors, evaluators, contrasts, failure cases |
| `wsp.replay` | Replay plans, teacher-forced execution, verification, replay model adapters |
| `wsp.alignment` | Token alignment and behavioral span resolution |
| `wsp.investigation` | Capture points, positioned capture backends, measurement building, comparisons, evidence |
| `wsp.interventions` | Intervention specs, schedules, plans, slice selection/application, reversible execution hooks, paired measurement, uncached and staged continuations, compositional comparisons |
| `wsp.tools` | Pydantic tool schemas, operation registry, shared `ToolService` |
| `wsp.cli` | Typer commands; every command prints the service’s JSON envelope |
| `wsp.adapters` | Optional model and format adapters: local Hugging Face, INIF 0.1, Inspect AI logs |

## Data flow

### Inspection

`open_checkpoint()` accepts a local `.safetensors` path and returns a
`CheckpointSource`. `SafeTensorsSource` reads the SafeTensors header and exposes
`TensorInfo` records (name, dtype, shape, element count, byte size, offset).
Tensor payloads are not read unless a tensor operation or bounded slice asks
for them. The manifest records a structural fingerprint over names, dtypes, and
shapes, so two copied checkpoints with different file names can still be
recognized as the same structure.

The fingerprint and the optional whole-file SHA-256 answer different questions:

- `structure_fingerprint`: do these checkpoints contain the same tensor layout?
- `file_sha256`: are these exact files byte-for-byte identical?
- `tensor_fingerprint`: are the decoded numerical values of one tensor identical?

Patch compatibility uses the structure fingerprint plus the target tensor’s
exact value fingerprint.

### Tensor analysis

Selection is resolved against the real tensor shape before any read. Invalid
axes, duplicate axes, out-of-range bounds, and non-positive steps fail before
I/O. Statistics are computed in float64 on the decoded values. If an operation
must sample, the result carries `sampled=True` and the stride; it is never
presented as a full result.

### Activation capture

`ActivationCapture` installs forward hooks on named modules (or the root module)
for the lifetime of one forward pass. Hooks are removed in `finally`, including
when the model raises. Captured tensors are detached, moved to CPU, widened from
BF16 to float32 for analysis, and reduced to a summary plus, when requested, a
deterministic bounded sample of retained values. Summary-only captures do not
consume the retained-value budget.

Module inputs, module outputs, hidden states, and attention tensors are distinct
kinds in the result. A residual-stream state is only reported as such when an
adapter explicitly labels it; the generic hook layer does not guess.

### Forward execution

`run_forward()` puts the model in evaluation mode under `torch.inference_mode()`,
optionally seeds torch/numpy, optionally installs a capture, and restores the
previous training mode. Outputs are summarized and may be sampled under the
output budget. `compare_forward()` compares flattened output arrays and
activation summaries from two executions.

Model-specific input preparation belongs in adapters. The generic executor
accepts tensors or arrays; the structured tool layer builds inputs for the
built-in offline models.

### Patching

Patch creation is a pure read operation on the source:

1. Resolve and validate the target tensor and selection.
2. Read the full target tensor (bounded by `max_tensor_numel`) and compute its
   exact fingerprint.
3. Resolve payloads, validate shapes and finiteness, and apply the operation in
   float64.
4. Cast the result back to the tensor’s storage dtype and report any precision
   effect.
5. Record before/after statistics, effect metrics, expected source identity,
   provenance, and a deterministic patch ID.

Preview repeats validation against the current source without mutating it. A
patch set is a collection of patches with disjoint targets. Application
materializes only targeted tensors, writes all untouched tensors by streaming
copies from the source, and atomically renames the completed output. The source
file is never opened for writing.

Rollback is not an undo log: the source checkpoint was never changed, so
reversion selects and verifies the immutable source identity.

### Behavioral observation

The Behavioral Observatory adds a data layer, not an inference engine:

```text
scripted backend ─┐
local HF backend ─┼─> GenerationTrace ─> immutable artifact + trace index
imported traces ──┘                          │
                                             ├─> detectors ─> findings
                                             ├─> comparison
                                             └─> contrast ─> evaluation + traces
                                                                 │
                                            failure case <───────┘
```

A trace records the exact input, settings, token stream, termination, timing,
provenance, privacy policy, and explicit unavailable fields. `finalize_trace`
is the only place that applies privacy and fills checksums. Missing data is
never guessed.

Detectors are deterministic measurements. Each returns a `DetectorRun` with
measurements and sparse findings; findings carry parameters, evidence ranges,
severity/magnitude when meaningful, and limitations. A finding does not claim a
cause. Detector failures are isolated to their run.

Contrasts execute an authored baseline and variations with the same parameters
and seed, store every trace, and evaluate a declared relationship. A baseline
that does not establish the property makes the result inconclusive; a variation
execution failure is recorded without aborting the other variations.

Failure cases link traces, findings, contrast artifacts, evaluations,
annotations, hypotheses, activation captures, and patch references. They can be
exported as self-contained, checksum-verified bundles and reloaded into a fresh
workspace.

## Storage formats

All schemas are versioned and JSON-compatible.

| Schema | Meaning |
| --- | --- |
| `wsp.identity/1.0` | Checkpoint identity and fingerprints |
| `wsp.manifest/1.0` | Model manifest with tensor metadata and totals |
| `wsp.patch/1.0` | One immutable weight patch |
| `wsp.patchset/1.0` | Compatible patches applied to one derived checkpoint |
| `wsp.preview/1.0` | Revalidated patch preview |
| `wsp.activation/1.0` | One activation summary |
| `wsp.activation-capture/1.0` | Serialized capture artifact payload |
| `wsp.activation-comparison/1.0` | Activation comparison |
| `wsp.forward/1.0` | Forward-pass result |
| `wsp.forward-comparison/1.0` | Forward comparison |
| `wsp.tool/1.0` | Structured tool response envelope |
| `wsp.artifact/1.0` | Content-addressed artifact record |
| `wsp.run/1.0` | Reproduction/run record |
| `wsp.workspace/1.0` | Workspace configuration |
| `wsp.experiment/1.0` | Named experiment grouping runs |
| `wsp.trace/1.1` | One generation observation (1.0 traces still validate) |
| `wsp.trace-index/1.0` | Index entry pointing at an immutable trace artifact |
| `wsp.finding/1.0` | One detector measurement/finding |
| `wsp.scan/1.0` | Detector runs for one trace |
| `wsp.evaluation/1.0` | Evaluator outcome, including uncertainty/abstention |
| `wsp.contrast/1.0` | Authored paired comparison definition |
| `wsp.contrast-result/1.0` | Executed comparison, outcomes, and evaluations |
| `wsp.trace-comparison/1.0` | Structural trace comparison |
| `wsp.case/1.0` | Persistent investigation case |
| `wsp.case-export/1.1` | Self-contained case bundle with privacy metadata |
| `wsp.export-policy/1.0` | Export privacy policy and its declared limitations |
| `wsp.redaction/1.0` | Record of what a redacted export removed |
| `wsp.replay-plan/1.0` | Resolved replay plan with checks and declared tolerances |
| `wsp.replay/1.0` | Executed replay with four independent status levels |
| `wsp.alignment/1.0` | Token positions in prompt/generation/full index spaces |
| `wsp.span-resolution/1.0` | Behavioral span mapped onto token positions |
| `wsp.capture-points/1.0` | Addressable modules with architectural roles |
| `wsp.trace-capture/1.0` | Trace-linked activation measurements |
| `wsp.activation-vector/1.0` | Retained activation vector artifact metadata |
| `wsp.capture-comparison/1.0` | Numerical comparison of two captures |
| `wsp.investigation-evidence/1.0` | Measurement chain attached to a case |
| `wsp.activation-intervention/1.0` | Versioned, machine-readable activation intervention |
| `wsp.intervention-result/1.0` | One measured execution condition |
| `wsp.counterfactual-comparison/1.0` | Paired baseline/identity/intervention measurement |
| `wsp.continuation-comparison/1.0` | Paired greedy continuation comparison |

### Replay, alignment, and measurement

`prepare_replay` turns a stored trace into a `ReplayPlan`: the recorded token
sequence is resolved (token ids verbatim, or re-derived from text when the trace
recorded an encoding), model/tokenizer/template identities are recorded, declared
tolerances are fixed, and every check gets a status. `verify_replay` executes the
plan through the model's own forward pass under `torch.inference_mode()`,
compares recorded token log-probabilities with replay values, and reports
`numerical_agreement` separately from `generation_reproducibility` (which is
always `not-claimed` for teacher forcing). A parameter fingerprint is compared
before and after the pass; a change raises instead of returning numbers.

`build_alignment` maps recorded tokens into prompt, generation, and full-sequence
index spaces with character/byte ranges where they are trustworthy, and records
`consuming`/`predicting` semantics so a caller cannot confuse "the state at this
token" with "the state that predicts it". `resolve_span` maps detector evidence
onto those positions and reports ambiguity or an unresolved mapping explicitly.

`capture_trace_activations` runs the same replay path with a positioned capture
attached, so a measurement inherits replay verification and the immutability
check. Two backends implement the same capture protocol: plain PyTorch forward
hooks (always available) and an optional NNsight/nnterp adapter that normalises
layer paths and refuses unsupported architectures rather than guessing. Both
build measurements with the same shared builder, so their numbers are directly
comparable, and both record their backend identity.

`compare_captures` validates model, module, shape, mode, and alignment
compatibility, then computes metrics from the shared tensor-difference
implementation. `attach_investigation_evidence` packages the resulting chain —
findings, replays, alignments, spans, captures, comparisons, and unverified
hypotheses — as one artifact that a failure case can reference and export.

### Causal intervention and continuation

An intervention specification (`wsp.activation-intervention/1.0`) is data, not
code. It names the model identity, the replay/execution identity, the
instrumentation backend, one module and tensor role, the token positions and
position semantics, the selected feature slice, the operation (`zero`, `mean`,
`patch`, `identity`), the replacement source, and explicit resource limits.
`mean` and `patch` sources carry provenance: a donor capture (module, role,
positions, model fingerprints) or an explicitly supplied mean with
`token_alignment` and source captures/traces. Nothing is inferred silently.

Execution uses the generic PyTorch forward-hook backend. A `forward_pre_hook`
intervenes on module inputs and a `forward_hook` intervenes on module outputs;
the hook resolves the declared slice, clones the tensor, writes only the selected
region, and rebuilds the original tuple/mapping structure. Hooks are removed in
`finally`; the model's training flag is restored and a parameter fingerprint is
compared before and after every pass, so a run that mutates weights raises
instead of returning numbers. The optional nnterp backend remains
capture-only in V1.3 and is rejected for interventions rather than silently
substituted.

`run_paired_intervention` executes the untouched baseline, an identity
intervention, and the requested intervention under identical weights, tokens,
masks, dtype, backend, and inference settings, with an optional fourth
alternative control. The identity control is mandatory by default: bitwise
equality reports `exact`; a tolerance comparison can report `within_tolerance`;
anything else marks the experiment `invalid` and is never auto-corrected. The
comparison artifact reports targeted slice before/after statistics, propagation
into downstream module rows, selected logits, token log-probability deltas, and
effect sizes, and states plainly that an observed effect is not proof of a causal
circuit.

`run_counterfactual_continuation` is a separate execution mode, not an implicit
extension of teacher-forced replay: greedy, uncached, full-prefix decoding with
the intervention applied during the first prefix forward only. It records prefix
ids, decoding configuration, seeds, per-condition generated ids, termination
reasons, longest common prefix, first divergence, and identity-control
agreement. Baseline and identity must match under deterministic conditions or
the experiment is marked invalid.

### Temporal scheduling, plans, and staged execution

V1.4 adds three layers on top of the V1.3 intervention core, without a second
execution framework.

`wsp.interventions.schedule` defines the time axis. A schedule is a versioned
record (`wsp.intervention-schedule/1.0`) with three kinds
(`first-prefix-forward`, `generation-steps`, `generation-range`), inclusive or
half-open bounds, a bounded stride, and an event cap. `trigger` reserves the
schema for future event triggers; only `forward` executes. The module also owns
`TIMING_CONVENTIONS`, the single indexing table shared by replay, capture,
intervention, and generation, and `resolve_schedule`, which partitions resolved
steps into reached and not-reached sets at a known execution length.

`wsp.interventions.plan` defines `wsp.intervention-plan/1.0`. A plan wraps full
`ActivationIntervention` specifications, so the numerical operations stay in one
place, and adds a schedule, a position mode (`absolute` or `last`), a per-step
module call index, and an explicit order. Validation resolves schedules,
estimates cost, resolves target modules against a real module tree, and rejects
composition conflicts: same module, same step, overlapping rows or features
unless both sides declare `allow_overlap`. Ordering is never derived from layer
names. `identity_control_plan` derives the matching identity control, which keeps
every hook site and schedule and changes no number.

`wsp.interventions.plan_engine` is the executor. `PlanSession` groups
interventions by `(module, tensor_role)`, installs exactly one hook per group,
resets per-step call counters at `begin_step`, and applies same-module
interventions sequentially in ascending order against the tensor produced by the
previous firing. Each firing appends a bounded `InterventionEventRecord`;
rejections are recorded and re-raised, so a rejected event cannot be mistaken for
a successful run. After every execution the engine recomputes the parameter
fingerprint, a mutable-state fingerprint (sampled buffers plus training flags),
and the installed hook count, and raises if any changed. `event_log()` compares
the fired keys with the plan's expected keys to separate *missing* events from
*not reached* ones. `run_temporal_intervention` executes baseline, identity, and
plan conditions in one mode and builds a `wsp.temporal-intervention/1.0` record.

`wsp.interventions.composition` runs baseline, identity, A, B, and A+B through
the same path and reports the descriptive interaction contrast on one chosen
metric, plus pairwise logit/token differences and stepwise logit diffs. It
refuses plans recorded in different execution modes or against different prompts.

`wsp.execution.staged` separates staged execution from the uncached reference.
A `StagedDecoder` protocol exposes `capabilities`, `prefill`, `step`, `clone`,
`state_hash`, and `state_bytes`. `TinyStagedDecoder` drives the fixture's
explicit key/value cache; `HFStagedDecoder` drives a local checkpoint's own
`use_cache=True` path with an attention mask and `cache_position` spanning the
cached prefix. `classify_hf_architecture` labels a model
`full-attention-transformer`, `hybrid-recurrent`, or `unknown`, and the last two
are rejected before execution. `wsp.interventions.staged` requires a
`wsp.staged-verification/1.0` comparison against uncached execution before a
staged intervention can be `valid`, supports `recompute` and verified `clone`
isolation, and records prefill/decode timing and state size.

### Workspace layout

`wsp init` creates:

```text
.wsp/
├── workspace.json
├── artifacts/<2-hex-prefix>/<artifact-id>.<ext>
│   └── <artifact-id>.<ext>.meta.json
├── patches/p_<16-hex>.json
├── patchsets/ps_<16-hex>.json
├── runs/run_<timestamp>_<8-hex>/run.json
├── models/<fingerprint-prefix>/manifest.json
├── experiments/e_<16-hex>.json
├── traces/tr_<16-hex>.json
└── cases/case_<16-hex>/case.json
```

Artifacts are addressed by SHA-256 of their bytes. The ID is stable for the
same content; a second write of the same bytes is idempotent. `ArtifactRef.path`
is relative to the workspace root. Reads validate the size and checksum before
returning data, and path joins are constrained to the workspace.

Generated data is ignored by Git through `.gitignore` (`.wsp/`, `*.safetensors`,
`examples/out/`).

## Resource budgets

`Budget` is a pydantic model with conservative defaults. Every entry point
accepts overrides. Unknown budget keys are rejected rather than ignored.

| Field | Default | Used by |
| --- | ---: | --- |
| `max_tensor_numel` | 20,000,000 | exact fingerprints, patch targets, model loading |
| `max_slice_elements` | 5,000,000 | slice reads and comparisons |
| `max_preview_elements` | 256 | tensor previews |
| `max_analysis_elements` | 20,000,000 | materialized analysis input |
| `max_sample_elements` | 200,000 | deterministic samples |
| `max_svd_elements` | 4,000,000 | SVD/rank |
| `max_histogram_bins` | 256 | histograms |
| `max_manifest_tensors` | 10,000 | manifests and tensor listings |
| `max_compare_tensors` | 2,000 | checkpoint comparisons |
| `max_activation_elements_per_tensor` | 500,000 | retained activation values |
| `max_activation_elements_total` | 2,000,000 | retained activation values |
| `max_forward_output_elements` | 5,000,000 | retained output values |
| `max_model_load_elements` | 50,000,000 | loading a checkpoint into a module |
| `max_patch_payload_elements` | 1,000,000 | patch payloads |
| `max_patchset_tensors` | 256 | patches per set |
| `max_trace_tokens` | 20,000 | stored/generated token records |
| `max_trace_input_tokens` | 8,192 | local-HF prompt length |
| `max_trace_chars` | 2,000,000 | trace text and token text |
| `max_trace_bytes` | 33,554,432 | one trace artifact |
| `max_scan_tokens` | 200,000 | detector analysis units |
| `max_scan_findings` | 200 | findings per scan |
| `max_contrast_variants` | 16 | variations per contrast |
| `max_case_traces` | 64 | trace references in a case |
| `max_import_traces` | 64 | traces per imported document |
| `max_intervention_elements` | 500,000 | intervened slice elements |
| `max_intervention_positions` | 128 | selected positions per intervention |
| `max_intervention_applications` | 16 | applications per execution |
| `max_intervention_logit_elements` | 2,000,000 | selected logit rows per comparison |
| `max_counterfactual_modules` | 16 | downstream modules measured |
| `max_counterfactual_rows` | 256 | logit positions compared |
| `max_continuation_tokens` | 64 | generated tokens per continuation |
| `max_continuation_sequence_tokens` | 4,096 | full-prefix sequence length |
| `max_output_bytes` | 2,000,000 | serialized tool envelopes |
| `max_artifact_bytes` | 268,435,456 | one stored artifact |

Limits fail explicitly with `resource_limit`; they do not silently degrade. The
one exception is mathematically appropriate sampling, which is always labeled
in the result.

## Determinism and precision

- Checkpoint structure and tensor fingerprints are exact decoded-value hashes.
- Statistics and patch arithmetic use float64; storage dtypes are preserved.
- BF16 values are losslessly widened to float32 for numerical analysis.
- Patch results are cast back to storage dtype. Any cast that changes values is
  reported in `precision.cast_changed_values`.
- IDs for patches and patch sets are deterministic functions of their content,
  not timestamps. Timestamps describe creation, not identity.
- CPU execution is the supported deterministic baseline. GPU results may differ
  across drivers and software stacks; do not assume bitwise reproducibility.
- Run records store operation parameters, source/model identity, device, dtype,
  seed, budget, dependency versions, artifacts, and warnings. They do not
  promise bitwise replay on different hardware.

## Trust boundaries

A SafeTensors file is data, not executable code. V1 never loads pickle
checkpoints and never executes Python bundled with a model.

A Python adapter runs in the user’s interpreter and is **not a sandbox**. Loading
a user-supplied model factory or module is a trusted-code operation. The local
Hugging Face adapter refuses remote downloads by default and sets
`trust_remote_code=False`.

## Design constraints

- Keep checkpoint inspection separate from model loading.
- Keep patch representation separate from patch execution.
- Keep execution separate from experiment bookkeeping.
- Keep tool schemas independent of any LLM provider.
- Keep observations, findings, evaluations, and interpretations separate.
- Make missing measurements explicit; never fabricate token ids, log-probabilities,
  timings, or termination reasons.
- Keep third-party formats behind optional adapters; the canonical trace schema
  is native and versioned.
- Do not add abstraction layers without an immediate use.
- Do not infer concepts from measurements; report measurements only.
