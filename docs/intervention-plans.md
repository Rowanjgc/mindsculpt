# Intervention plans

An intervention plan is the portable, reproducible description of a
multi-site, scheduled experiment. It is a record, not a script: a plan can be
stored, diffed, checksummed, re-validated, and re-executed by a later caller
without reading Python.

Schema: `wsp.intervention-plan/1.0` (`wsp.interventions.plan.InterventionPlan`).

## What a plan contains

| Field | Meaning |
| --- | --- |
| `plan_id` | `pl_` + 16 hex, stable identity of this experiment |
| `created_at` | When the record was produced (excluded from the checksum) |
| `name` | Optional human label |
| `plan_kind` | `intervention` or `identity-control` |
| `model` | `ModelRef` of the checkpoint/structure the plan was built against |
| `execution_mode` | `teacher-forced`, `uncached-generation`, or `staged-generation` |
| `backend` | `torch_hooks` (the only numerical backend in this release) |
| `prompt_length`, `prompt_sha256`, `prompt_token_ids` | The prefix the plan was designed for |
| `continuation` | Greedy decoding settings, bounded |
| `interventions` | 1..32 scheduled interventions |
| `ordering_policy` | `explicit-order` |
| `limits` | `TemporalLimits` |
| `provenance` | Free-form investigator metadata |
| `timing_conventions` | The indexing table the plan was written against |
| `notes`, `limitations`, `metadata`, `checksums` | Provenance and integrity |

Each entry is a `PlanIntervention`:

- `plan_intervention_id` (`pi_` + 16 hex) — the id used in event records;
- `intervention` — a full `wsp.activation-intervention/1.0` specification, so
  `zero`, `mean`, `patch`, and `identity` are numerically defined in exactly one
  place;
- `schedule` — a `wsp.intervention-schedule/1.0` record;
- `position_mode` — `absolute` or `last` (see
  [`docs/temporal-interventions.md`](temporal-interventions.md));
- `on_step_call_indices` — which invocation of the module within a step to
  intervene on. `[0]` is the first call; this matters for modules invoked twice
  in one forward, such as a shared block;
- `order` — explicit ordering among interventions that target the same module;
- `label`.

## Composition that is supported

1. Several interventions at different modules during the same forward.
2. Several interventions at different token rows of the same module.
3. Interventions at different generation steps.
4. Bounded repeated interventions (a `generation-range` or an explicit step
   list).
5. Mixed operations, for example a zero ablation at one site and a donor patch
   at another.

## Ordering semantics

Ordering is deterministic and observable, and it is never inferred from layer
names. Architectures may share modules, invoke a module more than once, or
interleave execution; `layers.10` before `layers.11` is a naming convention, not
a promise.

- **Different modules** fire in the order the model actually invokes them. The
  plan does not reorder execution, and the event log records the actual order.
- **Same module** fires in ascending `order`. Each firing's replacement is
  computed against the tensor produced by the previous firing, and the before /
  after statistics of each event show that.
- `order` values must be unique within a plan. A duplicate is a validation
  error, not a silent tie-break.

### Collisions

Two interventions collide when they could modify the same element of the same
tensor on the same generation step. Validation rejects the plan when:

- they target the same module, row set, and feature region on an overlapping
  step (`duplicate-site` or `overlapping-region`);
- one uses `position_mode: last` and the other targets a row that may be the
  same row, unless their feature regions are provably disjoint;
- they resolve to the same module call index and overlapping rows.

Disjoint steps, disjoint rows, or disjoint feature regions do not collide.
`allow_overlap` (passed per scheduled intervention, stored as
`metadata.allow_overlap` on the specification) permits an overlap only when
**both** sides declare it. A one-sided declaration is still a conflict. When
overlap is allowed the explicit `order` values define the composition order, and
that order is recorded.

The workbench never merges two conflicting interventions into one region, and
never silently drops the loser.

## Identity plans

An identity plan is not an empty plan. `identity_control_plan(plan)` produces a
new plan with:

- the same modules, schedules, position modes, call indices, and order values;
- every specification converted to `operation: identity` with no mean or donor;
- new `pi_` ids and a new `plan_id`;
- `plan_kind: identity-control` and `metadata.derived_from_plan_id` /
  `derived_from_plan_sha256` / `control_kind: identity`.

Running it attaches every hook at every site and changes no number. If baseline
and identity disagree, the instrumentation itself perturbed the computation, and
attribution from the same plan is not licensed. That is a stronger control than
an empty plan, which would prove nothing about the hooks.

Both conditions exist for every temporal run: the engine derives the identity
plan automatically unless a caller supplies one, stores it as its own artifact,
and reports `identity_control.agreement`.

## Validation

`validate_plan(plan, root=None, budget=None)` returns a `PlanValidation` with
named checks:

| Check | What it establishes |
| --- | --- |
| `continuation_budget` | `max_new_tokens` fits the continuation budget |
| `generation_steps_budget` | `max_new_tokens` fits `max_generation_steps` |
| `scheduled_events_budget` | total scheduled events fit `max_scheduled_events` |
| `intervention_count_budget` | intervention count fits `max_temporal_interventions` |
| `sequence_budget` | prompt + continuation fits the sequence budget |
| `config_sequence_length` | the configured maximum length is not exceeded |
| `schedule_reachable` | at least one scheduled step is reachable (warning if not) |
| `composition_conflicts` | no unresolved collision |
| `staged_position_semantics` | staged plans only use `absolute` at step 0 |
| `plan` | the plan-level coherence checks |

When a model `root` is supplied, the target modules are resolved against the
real module tree and `resolved` reports each module's class. An unknown module
is a validation failure, not a runtime surprise.

Validation also estimates cost before anything runs:
`estimated_forward_calls` and `estimated_token_positions`. Uncached full-prefix
decoding evaluates `sum(P..P+steps-1)` token positions, so a long schedule is
visibly quadratic rather than invisibly expensive.

### Plan-level coherence rules

- `prompt_length >= 1`; `prompt_token_ids`, when present, must have that length.
- Every specification's `sequence_length` must equal the plan's `prompt_length`.
- `teacher-forced` plans require `teacher-forced` specifications;
  `uncached-generation` and `staged-generation` plans require
  `uncached-continuation` specifications. The tool layer retargets a
  specification to the requested mode and records that it did.
- Every specification must name the same model file and structure fingerprint as
  the plan.
- Absolute positions must be `< prompt_length`.
- `position_mode: last` requires the sentinel `positions: [0]`.
- An identity plan contains only `identity` operations with no mean or donor.

## Compositional comparisons

Given plans A and B, `run_compositional_comparison` measures, in one execution
mode:

- **baseline** — untouched;
- **identity** — the derived identity plan for the combined experiment;
- **A alone**, **B alone**, **A+B** — and optionally an investigator-supplied
  negative control.

Every condition shares the model, the prompt, the decoding configuration, the
backend, and the numerical precision. The comparison records which execution
mode was used and refuses to mix modes.

For a chosen scalar `metric` (`mean_generated_logprob`, `final_token_logprob`,
`logit_l2_vs_baseline`, or `generated_length`) the record reports the
descriptive interaction contrast

```
I_AB = (Y_AB - Y_0) - (Y_A - Y_0) - (Y_B - Y_0)
```

plus the individual deltas, the pairwise logit and token differences, and the
stepwise logit differences. When the primary metric is discrete
(`generated_length`), the contrast is marked `continuous: false` and a secondary
continuous contrast is attached so the two are never confused.

Non-additivity on a chosen scale is a measurement. It does not establish that A
and B form a functional circuit, that either component is necessary, or that the
interaction is unique to these two sites. The record says so in its own
`interpretation` and `metadata` fields.

Validity is `invalid` when the identity control fails, when any scheduled event
in A, B, or AB fails to fire, or when a condition errored. It is `limited` when
the identity control was skipped.

## Structured operations

| Operation | Purpose |
| --- | --- |
| `create_intervention_plan` | Build, validate, and store a plan |
| `validate_intervention_plan` | Re-check a plan, optionally against a model tree |
| `preview_intervention_plan` | Show every resolved firing without modifying anything |
| `run_compositional_comparison` | Baseline/identity/A/B/AB with interaction contrasts |
| `attach_temporal_evidence` | Attach plans, runs, logs, compositions, and staged records to a case |

```bash
wsp create-intervention-plan --json '{
  "name": "two-site",
  "prompt_token_ids": [1, 10, 11, 12],
  "execution_mode": "uncached-generation",
  "max_new_tokens": 4,
  "interventions": [
    {"intervention_artifact_id": "a_1111...", "order": 0, "label": "A",
     "schedule": {"kind": "generation-steps", "steps": [2]},
     "position_mode": "last"},
    {"intervention_artifact_id": "a_2222...", "order": 1, "label": "B",
     "schedule": {"kind": "generation-steps", "steps": [3]},
     "position_mode": "last"}
  ]
}'
wsp preview-intervention-plan --plan pl_3333... --source '{"kind":"synthetic"}'
wsp run-compositional-comparison --plan-a pl_4444... --plan-b pl_5555... --plan-ab pl_6666... \
    --source '{"kind":"synthetic"}' --metric mean_generated_logprob
wsp attach-temporal-evidence --title "two-site experiment" --case case_1a2b... \
    --plan pl_4444... --event-log el_7777... --run tr2_8888...
```

## Identity plans and empty plans are different records

An empty plan is rejected by the schema: `interventions` has a minimum length of
one. A plan with an `identity` operation is a valid control that still installs
hooks. The two must never be confused, and neither can substitute for the other.

## V1.5 event triggers and plans

The V1.4 plan engine remains the executor for precomputed schedules. V1.5 event
triggers are a declarative layer that decides *when* an intervention becomes
eligible, then applies the same `ActivationIntervention` operations. An online
trigger cannot be represented as a fixed plan before generation because its
step depends on observed tokens; the trigger record preserves that distinction.

Retrospective investigations use fixed-context replay rather than a fabricated
plan schedule. Combined plans, scheduling, and fixed-context comparisons all
share the same operation implementations and model-state protections.
