# Roadmap

This roadmap separates what the workbench actually contains from hypotheses
about future work. Future items are not commitments and should not be
implemented merely because they sound useful.

## Completed in V1

### Foundation and inspection

- Installable `wsp` package and CLI.
- Versioned pydantic schemas and JSON envelopes.
- SafeTensors header-only inspection with tensor metadata.
- Structure, file, and per-tensor fingerprints.
- Model manifests and atomic manifest export.
- Checkpoint comparison with missing/extra/dtype/shape/value detection.
- Live `torch.nn.Module` inspection adapter.
- SafeTensors-only automatic loading; pickle rejected.

### Tensor analysis

- Mean/std/min/max, finite and non-finite counts.
- Quantiles and histograms.
- Vector and matrix norms.
- Cosine, dot product, projection, correlation-style comparisons.
- Matrix rank and singular-value summaries.
- Difference metrics and bounded deterministic sampling.
- Explicit sampled/approximate labels.

### Activation instrumentation

- Forward hooks on named modules and the root.
- Input/output capture with bounded retention.
- Summary-only capture mode.
- Token-position selection for 2-D/3-D tensors.
- Deterministic cleanup, including error paths.
- Activation comparison and summary aggregation.
- Device/dtype reporting.

### Controlled weight editing

- Immutable `WeightPatch` format.
- Replace, add, scale, rank-one, and projection operations.
- Shape/dtype/index/finiteness/resource validation.
- Exact source tensor fingerprints.
- Preview without committing.
- Patch sets with disjoint targets.
- Streaming export to a derived SafeTensors checkpoint.
- Rollback by verified source selection.
- Precision and casting reports.
- Deterministic patch and patch-set IDs.

### Execution and experiments

- Inference-mode CPU-baseline forward execution.
- Optional local Hugging Face causal-LM adapter.
- Output comparisons and activation comparisons.
- Offline built-in models for demos/tests.
- Workspace with content-addressed artifacts.
- Run records with environment, parameters, replay metadata, and warnings.
- Experiment grouping.
- Patch, patch-set, manifest, and artifact persistence.

### Interfaces and tests

- Structured operation registry with JSON Schemas.
- Python API and CLI sharing one service layer.
- CLI JSON output with machine-readable failures.
- Offline CPU test suite covering the 15 V1 acceptance behaviors.
- Offline examples for inspection, activation capture, patching, comparison,
  and rollback.

## Completed in V1.1

### Behavioral Observatory

- Versioned `wsp.trace/1.0` schema with token-level observations, termination
  reasons, timing, provenance, privacy policy, and explicit unavailable fields.
- Immutable trace artifacts plus a local trace index.
- Deterministic scripted generation backend and optional local Hugging Face
  generation backend with token ids/log-probabilities when available.
- Detector registry and five deterministic detectors: repetition, termination,
  structured output, reconsideration markers, and repeated tool actions.
- Sparse findings with evidence ranges, magnitude, parameters, and declared
  limitations.
- Evaluator abstraction with pass/fail/inconclusive/abstain/error outcomes,
  reference answers, JSON contracts, and human judgements.
- Contrastive tests with explicit invariants, exact authored inputs, stored
  traces, and result artifacts.
- Structural trace comparison.
- Persistent failure-case registry with review states, annotations, hypotheses,
  reproduction metadata, and references to existing activation/patch artifacts.
- Self-contained, checksum-verified case export and reload.
- INIF 0.1 and Inspect AI log adapters without adding them as core dependencies.
- CLI and structured operations for the entire workflow, sharing the V1 service
  layer.
- Offline CPU tests and an end-to-end example with synthetic traces.

## Completed in V1.2

### Replay and alignment

- Teacher-forced replay plans (`wsp.replay-plan/1.0`) that resolve the recorded
  token sequence explicitly, audit replayable traces, and declare tolerances.
- Replay results (`wsp.replay/1.0`) that report four independent statuses:
  token-sequence reconstruction, forward execution, numerical agreement, and —
  always `not-claimed` — generation reproducibility.
- Model, tokenizer, chat-template, and parameter-identity verification, plus a
  parameter fingerprint taken before and after every forward pass.
- Token alignment (`wsp.alignment/1.0`) with prompt/generation/full index
  spaces, character and byte ranges where trustworthy, special-token flags, and
  explicit `consuming`/`predicting` semantics.
- Span resolution (`wsp.span-resolution/1.0`) from detector findings, token
  ranges, and text spans, reporting ambiguity and unresolved mappings instead of
  guessing.

### Activation investigation

- Capture points with architectural classification, in two instrumentation
  backends: plain PyTorch hooks (default) and an optional isolated NNsight/nnterp
  adapter that records its own backend identity.
- Targeted capture (`wsp.trace-capture/1.0`) in `summary` and `selected` modes,
  bounded by element, position, module, and artifact budgets, with hook cleanup
  and inference mode guaranteed.
- Capture comparison (`wsp.capture-comparison/1.0`) with reviewed or identical
  position alignment, shared numerical metrics, and compatibility validation.
- Investigation evidence (`wsp.investigation-evidence/1.0`) linking findings,
  replays, alignments, captures, comparisons, and unverified hypotheses to a
  failure case.
- Case export/import of the new artifact types in both private and redacted
  modes, with rewritten references, recomputed ids, and explicit omission
  records.
- An offline end-to-end walkthrough, a synthetic hybrid transformer fixture, and
  an optional real-checkpoint test procedure.

## Completed in V1.3

### Controlled causal interventions

- Versioned intervention specifications (`wsp.activation-intervention/1.0`)
  covering model and execution identity, backend, module, tensor role, token
  positions and semantics, feature slice, operation, replacement provenance,
  expected shape/dtype, and resource limits.
- Zero ablation, mean ablation with explicit provenance, donor activation
  patching with compatibility validation, and an identity negative control.
- Reversible execution-time hooks that modify only the selected region, clone
  before writing, preserve structured module output, restore hook state in
  `finally`, and verify parameter immutability with a before/after fingerprint.
- Structured validation and preview without executing the intervention, plus a
  human-readable scope and cost estimate.

### Paired measurement and continuations

- Paired baseline/identity/intervention measurement
  (`wsp.counterfactual-comparison/1.0`), with an optional alternative control,
  shared inputs and settings, targeted slice before/after statistics, downstream
  module differences, selected logits, token log-probability deltas, and explicit
  effect sizes.
- Mandatory identity-control agreement reporting: `exact`, `within_tolerance`, or
  a failure that marks the experiment invalid.
- Bounded, deterministic, uncached full-prefix greedy continuations
  (`wsp.continuation-comparison/1.0`) with recorded prefix ids, generation
  configuration, seeds, per-condition outputs, termination reasons, longest
  common prefix, and first divergence.

### Evidence, privacy, and interfaces

- Intervention and continuation references integrated into
  `wsp.investigation-evidence/1.0` and `wsp.case/1.0` without breaking older
  records.
- Case export/import of all new artifact types in private and redacted modes.
- Privacy hardening: every removed string of any length is checked by exact
  equality inside decoded JSON, explicit sensitive markers are checked as
  substrings regardless of length, Unicode normalization and bounded
  base64-decoding are scanned, non-structural nested keys are redacted, and
  untransformable artifacts fail or are omitted under explicit policy.
- Nine structured operations and matching CLI commands sharing the existing
  service layer.
- Offline end-to-end demonstration, tiny-transformer integration tests, short/
  Unicode privacy tests, and optional real-checkpoint intervention tests.

## Completed in V1.4

### Temporal scheduling

- A versioned schedule record (`wsp.intervention-schedule/1.0`) with
  first-prefix-forward, explicit generation steps, and inclusive or half-open
  ranges with a bounded stride. Executable predicates are never accepted;
  `trigger` reserves the schema for future event triggers.
- One documented indexing convention shared by replay, capture, intervention,
  and generation: generation step `k` consumes the prefix plus `k` generated
  tokens and predicts generated token `k`, so `k = 0` is the first-prefix
  forward. Plans persist the convention with the record.
- Two position modes: `absolute` rows before the prompt length, and `last` for
  the newest row of the current forward. `position_semantics` stays descriptive
  and never shifts indices.
- Explicit handling of uncached recomputation: a one-time intervention is not
  persistent, a sustained schedule is an explicit repeat, and every firing is
  logged so the difference is visible rather than inferred.

### Intervention plans and composition

- `wsp.intervention-plan/1.0`: a portable plan of one to thirty-two scheduled
  interventions, each wrapping a full V1.3 specification, plus position mode,
  per-step module call index, explicit order, limits, provenance, and timing
  conventions.
- Deterministic ordering: distinct modules fire in actual model invocation
  order, same-module interventions fire in ascending explicit order, and
  conflicts (same module, overlapping step and region) are rejected unless both
  sides declare `allow_overlap`.
- Identity plans derived from a plan keep every hook site and schedule and change
  no number. Validation resolves modules against a real module tree and estimates
  forward calls and token positions before execution.
- `wsp.intervention-event-log/1.0` and `wsp.temporal-intervention/1.0`, with
  applied, identity, rejected, missing, and not-reached events, bounded slice
  statistics, replacement digests, and parameter/mutable-state/hook verification.
- `wsp.compositional-comparison/1.0`: baseline, identity, A, B, and A+B measured
  in one execution mode, with pairwise differences and a descriptive interaction
  contrast. Discrete metrics are labelled as such and never treated as
  continuous.

### Verified staged generation

- A staged decoder abstraction (prefill, step, clone, state hash/size) over the
  fixture's explicit key/value cache and over a local checkpoint's own cached
  forward path.
- `wsp.staged-verification/1.0`: an unmodified staged run must agree with the
  uncached reference on tokens, per-step logits, greedy choices, termination, and
  identity control before a staged intervention can be `valid`.
- Architecture classification with explicit rejection of hybrid-recurrent and
  unknown architectures. Qwen3.5-style Gated DeltaNet models are rejected rather
  than approximated.
- Verified `clone` isolation and default `recompute` isolation, plus prefill,
  decode, total, reference, and state-size measurements.
- `wsp.staged-continuation/1.0` records, including the verification block and the
  performance measurements.

### Evidence, privacy, and interfaces

- Six new artifact kinds integrated into `wsp.investigation-evidence/1.0`, case
  references, export/import, and the existing redaction policy, with fail-closed
  handling of malformed or untransformable payloads.
- Ten structured operations and matching CLI commands sharing the existing
  service layer.
- Offline 24-step demonstration, unit tests for schedules/plans/execution/
  composition/staged/privacy/tools, tiny-transformer integration tests, and
  optional real-checkpoint tests covering scheduled uncached interventions,
  multi-site plans, and staged support reporting.

## Deliberately deferred

These are **not** in V1.3:

- Autonomous research agents or LLM provider integration.
- Automatic weight-edit discovery, gradient optimization, or fine-tuning.
- Automated neuron/circuit interpretation or significance claims.
- Model merging, circuit grafting, or cross-model patching.
- Benchmark suites or claims of improved model intelligence.
- Training specialized interpretability models.
- GUI/dashboard, distributed inference, or cloud infrastructure.
- GGUF or direct quantized/float8 weight editing.
- Cached autoregressive generation tracing.
- Pooled raw-activation aggregation.
- Multi-patch edits to a single tensor inside one patch set.
- Autonomous behavioral investigators or automatic causal localization.
- LLM-based evaluators or hosted evaluation providers.
- Automatic weight-patch discovery from behavioral findings.
- A large defect taxonomy or benchmark leaderboard.
- Generation-time activation recording over long generations.
- Automatic span selection or automatic mechanism naming.
- nnterp-backed interventions, cached/stateful generation, and sampling
  continuations.
- Event-triggered (token-event) intervention scheduling.
- Verified staged generation for hybrid/recurrent architectures such as Qwen3.5.
- Sampling, beam search, or production decoding in either continuation mode.
- Automatic interaction interpretation beyond a descriptive non-additivity
  measurement.

## Possible future work (hypotheses, not commitments)

Ordered roughly by how directly they extend existing primitives.

1. **MCP adapter.** Expose the existing registry as MCP tools. The schemas are
   already provider-independent; this should be a thin transport layer, not a
   second implementation.
2. **Composable patch algebra.** Validate and combine multiple operations on one
   tensor before application, with explicit ordering and a single resulting
   `WeightPatch`.
3. **More precise quantized support.** Read/dequantize, operate in a documented
   higher-precision space, requantize, and report quantization error. This needs
   a careful format-specific design and should not be rushed.
4. **Chunked tensor algebra.** Stream large tensor statistics/differences
   without materializing the whole tensor, raising the patchable tensor size
   beyond `max_tensor_numel`.
5. **Generation-time instrumentation.** A documented adapter contract for
   capturing selected hidden states during cached decoding without pretending
   module hooks are equivalent to a full forward pass.
6. **Replay runner.** Re-execute a saved run record against the same local
   artifacts and report environment drift explicitly.
7. **Model-family adapters.** Small, explicit adapters for common local
   architectures and tokenizer preparation, kept outside the core.
8. **Activation dataset aggregation.** Pool raw values under explicit memory
   budgets, in addition to the current summary-statistics aggregation.
9. **Patch diff artifacts.** Export human-readable and machine-readable
   before/after summaries and a compact sparse delta for review.
10. **Deterministic execution profiles.** Record and compare backend settings
    (thread counts, deterministic algorithms) without promising cross-hardware
    bitwise identity.
11. **Inspect AI execution adapter.** Run the same controlled comparison through
    Inspect and import its logs, keeping the native service interfaces and case
    store authoritative.
12. **Generation-time activation capture.** Use a maintained instrumentation
    backend for selected decode steps and reference those captures from traces.
    The reviewed NNsight/nnterp pair is documented in
    `docs/prior-art-and-integration.md`.
13. **Case review workflow.** Add reviewer assignment, case linking, and a small
    local review queue without turning the project into a dashboard.

## Research directions (outside the tool core)

If the instruments prove trustworthy, later work may use them to ask questions
such as:

- Which weight directions are causally involved in a behavior?
- How local or distributed is a particular capability?
- Do edits interact across layers or tasks?
- Can a measured weight-space edit be reproduced across model families?

Those are research programs. V1 deliberately stops at instruments and
measurements. See the WSP research write-up and released logs referenced in the
project README for context and attribution; no upstream code or experiment
assets are bundled here.

## Quality gates for future changes

- New operations must have JSON Schemas and bounded outputs.
- New file formats must be versioned and safe to parse.
- New numerical behavior must report precision and sampling.
- New execution paths must preserve deterministic CPU tests or explicitly mark
  themselves environment-dependent.
- No feature should make source checkpoint mutation possible.
- No agent-only behavior. Python, CLI, and future tool transports must call the
  same service.

## Completed in V1.5

### Deterministic events

- `wsp.behavioral-event/1.0` with token occurrence, repeated sequence, first
  repetition, detector finding, manual, and text-pattern events.
- Absolute token positions, prompt offsets, generation steps, consumed
  positions, earliest online steps, alignment status, and content checksums.
- Text-boundary ambiguity reported instead of fabricated activation positions.

### Triggered interventions

- `wsp.event-trigger/1.0` with online and retrospective modes, delay, bounded
  window, firing mode, maximum count, and position mode.
- Strict causal ordering: an event observed after token `k` cannot fire before
  step `k + 1 + delay`.
- Separate observed, scheduled, fired, not-reached, skipped, and rejected states.
- `wsp.event-intervention/1.0` records the event life cycle, applications,
  conditions, identity control, stepwise logits, and validity.

### Fixed-context counterfactuals

- `wsp.fixed-context-replay/1.0` and `wsp.fixed-context-comparison/1.0`.
- Exact prefix reconstruction, shared token digest, mandatory identity control,
  selected log-probabilities and logits, optional downstream measurements.
- Optional bounded continuation kept separate, with divergence reporting.

### Accessibility

- `wsp.investigation-recipe/1.0` and `wsp.recipe-run/1.0`.
- Three built-in recipes with preview, prerequisites, cost estimates, artifacts,
  privacy notes, and structured partial-failure records.
- `wsp.investigation-dossier/1.0` separating observations, measurements,
  controls, hypotheses, limitations, and next operations.
- Discovery metadata for every operation, nine groups, and deterministic
  progressive disclosure through `discover_capabilities`.

### Evidence and safety

- New artifact kinds registered with the single privacy exporter.
- Event, comparison, recipe-run, and dossier references added to evidence and
  failure cases.
- New budgets for event lookback, trigger counts, fixed-context positions,
  recipe steps, passes, and artifacts.

## Candidate next work

- Evaluate the high-level recipes with actual general-purpose language models.
- Add a controlled comparison of recipe use versus low-level tool composition.
- Explore explicitly labelled semantic event detectors, without making a model
  judge mandatory.
- Extend fixed-context measurement to selected attention/MLP subspaces.
- Consider verified cached execution for fixed-context comparisons once a
  backend guarantees identical hidden state.
