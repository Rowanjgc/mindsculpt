# Controlled activation interventions

V1.3 adds a versioned, machine-readable way to modify an activation *during a
single execution* and measure what changes. The instrument is deliberately
separate from checkpoint patching: no weights are edited, no module state is
left changed, and every experiment is recorded as evidence.

Schema: `wsp.activation-intervention/1.0`
Artifact kind: `activation_intervention`

## What an intervention specification records

`ActivationIntervention` is a strict, validated record:

| Field | Meaning |
| --- | --- |
| `intervention_id` | `iv_<hex16>` stable id. |
| `model` | `ModelRef` for the checkpoint the execution uses. |
| `execution` | `ExecutionIdentity`: backend, backend version, device, dtype, execution mode, sequence length, inference mode, attention-mask and position-id conventions, and the trace/replay/capture/alignment ids that anchor the experiment. |
| `target` | `module`, optional `capture_point` (must equal `module` when present), tensor role (`input` or `output`), position semantics (`consuming` or `predicting`), and a `TensorSelector`. |
| `operation` | `identity`, `zero`, `mean`, or `patch`. |
| `mean` / `donor` | Replacement source for mean ablation or donor patching. |
| `apply_on_calls` | Forward-call indices during which the intervention applies. Defaults to `[0]`. |
| `allow_dtype_cast` | Explicit authorization for a recorded precision conversion. |
| `limits` | Per-experiment element, position, application, logits, and measured-module budgets. |
| `provenance` | Free-form but audited metadata about how the spec was produced. |
| `checksums` | `spec_sha256`, computed without `created_at` and without the checksum field itself. |

The selector is explicit about scope:

* `positions` are tensor positions, validated against `execution.sequence_length`.
* `axis` names the sequence axis. When omitted: rank 3 defaults to axis 1
  (`[batch, sequence, hidden]`), rank 2 to axis 0, and rank 1 requires
  `positions=[0]`.
* `feature_start`/`feature_end` select a contiguous feature range;
  `feature_indices` select explicit feature columns. They are mutually exclusive.
* `tensor_index` or `tensor_name` select one tensor from a structured module
  output. The engine never guesses: an ambiguous tuple or mapping is rejected.
* `expected_shape` and `expected_dtype` are optional assertions that must hold.

## Operations

### Identity

Re-applies the original activation values. This is the mandatory negative
control. The engine clones the tensor and writes the same values back, so the
execution path is identical to the intervention path. Identity is the only
operation that rejects a replacement source.

### Zero ablation

Replaces the selected region with zeros of the activation dtype. Only the
selected rows and feature columns are written; the rest of the tensor is a clone
of the original.

### Mean ablation

Replaces the selected region with a mean activation. The mean is never computed
implicitly from unrelated examples. `MeanActivation.provenance` must name:

* the aggregation method (`capture-mean` or `explicit`),
* the source capture artifact and/or source trace ids,
* the module, tensor role, positions, and position semantics,
* a non-empty `token_alignment` explanation of how mean rows correspond to the
  targeted tokens,
* the source model file digest and structure fingerprint.

Inline values must be finite. Values may also be stored in an artifact
(`activation_values`), in which case the target dtype is recorded and applied on
load.

### Activation patching

Replaces the selected region with an activation captured from a controlled donor
execution. `DonorActivation` records the donor trace, capture artifact, donor
execution identity, module, tensor role, positions, position semantics, and
condition (`healthy-control`, `alternative-input`, or `other`).

Compatibility is checked, not assumed:

* the donor module and tensor role must match the target,
* the donor value count must equal the feature width or
  `positions × feature width`,
* a single donor row may only be broadcast across several positions when
  `allow_position_broadcast` is set explicitly,
* dtype differences require `allow_dtype_cast` and are recorded as
  `"source->target"` in the application record.

## Validation and preview

`validate_activation_intervention` and `preview_activation_intervention` load the
model and perform a dry run (`apply=False`) to resolve the real tensor shape,
sequence axis, row indices, feature width, and estimated element count. They
return:

* `valid`,
* machine-readable `checks` (`pass`, `fail`, `warning`, `skipped`) with details,
* `errors` and `warnings`,
* a structured `scope` and human-readable lines,
* `estimated_elements` for the selected region.

Rejection is explicit for: unsupported backends, unknown modules, sequence or
position mismatches, ambiguous selectors, non-finite replacements, unexpected
shapes or dtypes, unavailable donor artifacts, unauthorized dtype casts, and
resource-limit violations.

`run_activation_intervention` refuses an invalid spec; it never silently repairs
one.

## Execution engine

`wsp.interventions.engine` installs one forward hook (or forward-pre hook for
`input` role) on the target module:

* `input` role uses `register_forward_pre_hook(..., with_kwargs=True)` and
  returns a rebuilt argument structure.
* `output` role uses `register_forward_hook` and returns the original structure
  with only the chosen tensor replaced.
* Tuples, lists, named tuples, and mappings are rebuilt, not mutated in place.
  Nested structures deeper than one level are rejected unless a supported path
  is explicit.
* Call indices are counted per module. Only `apply_on_calls` is modified.
* Every application clones the activation before writing, so shared states are
  not changed in place.
* Hooks are removed in `finally`; `close()` is idempotent.
* The model is placed in `eval()` mode and its previous training flag is
  restored.
* Execution runs under `torch.inference_mode()`.
* `parameter_fingerprint` is taken before and after the pass. A difference raises
  `ExecutionError`; parameters are never written by the intervention path.

Downstream measurements are optional. `measure_modules` captures selected
positions from named modules after the intervention, using the same
position-selection rules. Capture is bounded by the budget and by
`limits.max_measured_modules`.

## Paired counterfactual measurement

`run_paired_intervention` executes at least three conditions with identical
weights, token ids, attention mask, position configuration, dtype, backend, and
execution settings:

1. **baseline** – no hook application,
2. **identity control** – the same hook path with identity operation,
3. **intervention** – the requested operation.

An optional **alternative-control** intervention can be supplied as a fourth
condition.

The identity control is mandatory by default. Agreement is reported as:

* `exact` – bitwise equal logits,
* `within_tolerance` – `allclose` with the declared `atol`/`rtol`,
* `failed` – a real disagreement.

A failed identity control marks the comparison `invalid` and records the reason.
It is never auto-corrected or hidden. If the control is disabled, the comparison
is marked `limited`.

### Measurements

The comparison records:

* the targeted activation before and after (per-position statistics, max/mean
  absolute difference, relative L2, cosine),
* logits differences at selected positions (`TensorDiffStats`),
* selected next-token log-probabilities in baseline, identity, and intervention,
* downstream activation differences for measured modules,
* intervention magnitude and the modified element count,
* optional projection of the target delta onto an explicitly supplied direction,
* execution timing and resource counts.

Per-run records (`wsp.intervention-result/1.0`) preserve the same information for
each condition. `CounterfactualComparison`
(`wsp.counterfactual-comparison/1.0`) links the conditions, the identity control,
the effect summary, and the validity report.

### Validity is not causality

The fixed interpretation string on every comparison says that the observed
difference is an effect under the recorded conditions, not proof that the
targeted component explains a behaviour. The workbench does not emit causal
labels, circuits, or capability claims.

## Supported backends

* `torch_hooks` (mandatory): the generic PyTorch forward-hook backend. This is
  the supported path for synthetic models, `torch-module` models, and local
  Hugging Face checkpoints loaded through the replay backend.
* `nnterp` (optional): V1.2 capture support is retained, but **interventions are
  deferred** in V1.3. A spec that requests `nnterp` is rejected explicitly rather
  than silently executed with a different capture point.

## Recurrent and hybrid architectures

A module-level intervention can affect later recurrent state, KV caches, or
hybrid attention state. V1.3 interventions run in full-sequence teacher-forced
mode with `use_cache=False`. The comparison records the execution mode and does
not claim equivalence with cached or stateful generation. Continuations are a
separate, explicitly uncached mode (see
[counterfactual-continuations.md](counterfactual-continuations.md)).

## Structured tools

| Operation | Purpose |
| --- | --- |
| `create_activation_intervention` | Create and optionally save a validated spec. |
| `validate_activation_intervention` | Dry-run compatibility and scope checks. |
| `preview_activation_intervention` | Human-readable and structured scope estimate. |
| `run_activation_intervention` | Run one condition with measurements. |
| `run_paired_intervention` | Baseline, identity, intervention, optional alternative. |
| `compare_intervention_results` | Inspect a stored comparison and its results. |
| `attach_intervention_evidence` | Attach refs to a failure case. |

All tools return the standard `wsp.tool/1.0` envelope and respect the workspace
budget. See [tool-api.md](tool-api.md).

## Resource limits

Defaults in `Budget`:

* `max_intervention_elements = 500_000`
* `max_intervention_positions = 128`
* `max_intervention_applications = 16`
* `max_intervention_logit_elements = 2_000_000` (selected rows, not full logits)
* `max_counterfactual_modules = 16`
* `max_counterfactual_rows = 256`

Per-spec `InterventionLimits` can tighten these further. Full forward logits are
still bounded by `max_replay_logits_elements`.

## What this does not establish

* that a targeted component is necessary or sufficient for a behaviour,
* that a difference generalizes beyond the recorded prompt, dtype, device, and
  backend,
* that a mean or donor activation is semantically meaningful,
* that a downstream change is an improvement.

Those require additional controls, replication, and behavioural evaluation.

## Where V1.4 continues

V1.3 interventions fire inside one forward pass selected by module call index.
V1.4 keeps every V1.3 record readable and every V1.3 tool working, and adds a
time axis and a composition layer on top:

* [`docs/temporal-interventions.md`](temporal-interventions.md) — generation-step
  scheduling, position modes, event logs, and why an uncached one-time
  intervention does not persist.
* [`docs/intervention-plans.md`](intervention-plans.md) — multi-site plans,
  deterministic ordering, conflict rejection, identity plans, and
  individual-versus-combined comparisons.
* [`docs/staged-generation.md`](staged-generation.md) — verified cached
  execution, cache isolation, and the architectures that are rejected.

A V1.3 artifact is never reinterpreted: `apply_on_calls` still means a module
call index, and `execution.intervention_event = "first-prefix-forward"` still
means step 0. New experiments should use a plan, which records its schedule
explicitly.

## V1.5 event-triggered and fixed-context interventions

V1.5 adds two ways to reach the existing intervention operations.

- **Online event trigger** — the trigger observes generated tokens and applies
  the intervention at the earliest eligible subsequent forward. It uses the same
  reversible hook session, selection resolution, and model-state checks as V1.4.
- **Fixed-context comparison** — baseline, identity, and intervention consume one
  identical recorded sequence. The identity control remains mandatory; a failure
  invalidates attribution.

Neither path adds a new patching mechanism. A triggered or fixed-context effect
is still a numerical observation under recorded conditions, not a mechanism or a
repair. See [`event-driven-investigations.md`](event-driven-investigations.md)
and [`fixed-context-counterfactuals.md`](fixed-context-counterfactuals.md).
