# Temporal interventions

V1.3 could intervene during one forward pass, addressed by a module call index.
That is enough to intervene at the prompt. It is not enough to intervene at
generation step 500, at two different steps, or at two layers at once.

V1.4 adds a *time axis* to activation interventions. The instruments are
deliberately conservative: a schedule is a declarative, versioned record, every
firing is logged, and an intervention that does not fire is reported as a
failure rather than quietly treated as a null result.

## The indexing convention

The same convention is used by replay, capture, intervention, and generation.
No component shifts an index because of how it is labelled.

Let `P` be the number of prefix (prompt) tokens supplied by the investigator.

| Term | Definition |
| --- | --- |
| Prompt token index | `0 .. P-1`, fixed for the whole experiment |
| Generation step `k` | 0-based. Step `k` consumes the prefix plus the `k` previously generated tokens and predicts *generated token* `k` |
| First-prefix-forward | Generation step `0`. It is the only event V1.3 could address |
| Generated token index | `0`-based, one per generation step. Generated token `j` has full-sequence index `P+j` |
| Full-sequence token index | `0 .. P+k-1` at step `k` |
| Forward invocation index | How many times a *module* has been invoked so far in the run. A module property, not a time axis |
| Activation consumption position | The sequence row of an activation tensor. In an uncached full-prefix forward over `N = P+k` tokens, row `p` is the state that consumed token index `p`. In a staged decode forward the input is length `1`, so row `0` is the state that consumed full-sequence index `P+k-1` |
| Next-token prediction position | The token index a row predicts. Row `p` predicts token `p+1`. `position_semantics` records intent only; `positions` always index rows |

`TIMING_CONVENTIONS` in `wsp.interventions.schedule` carries the same table in
machine-readable form, and every plan records it under `timing_conventions` so a
stored plan remains interpretable without this document.

### Position modes

A plan intervention resolves its selector rows at every firing:

- `absolute` — rows are literal token indices inside the current forward. They
  must be `< P`, because generated positions do not exist at every step. This is
  the default and reproduces V1.3 behaviour exactly.
- `last` — the selector carries the sentinel `positions=[0]` and resolves to the
  final row of the current forward: full index `P+k-1` for `k >= 1`, and `P-1`
  at step `0`. This is how an investigation targets "the token the model just
  produced" without knowing `P` in advance.

`absolute` in a staged plan is only accepted when the schedule resolves to
exactly `[0]`; a staged decode forward has one row and cannot host an arbitrary
earlier row. Plans that would break that rule are rejected by the
`staged_position_semantics` check instead of being silently reinterpreted.

## Uncached recomputation is not persistence

Uncached full-prefix decoding recomputes every earlier position on every step.
A hidden state modified at step `3` is **not** present in the step `4` forward
unless the schedule fires again. Therefore:

- an intervention fires only on its scheduled steps;
- the modified tensor is never written back to the model or to a cache;
- a one-time schedule and a sustained schedule are different experiments, not
  different spellings of the same one;
- a changed output at a later step is an effect *through the generated tokens*,
  not evidence that a modified activation persisted.

The engine records this distinction structurally: `TemporalInterventionRun`
metadata lists the one-time and the sustained interventions separately, and the
event log records the exact step of every firing.

## Schedule records

`wsp.intervention-schedule/1.0` supports three kinds.

```json
{"schema_version": "wsp.intervention-schedule/1.0",
 "kind": "generation-steps", "steps": [2, 5, 6]}
```

```json
{"schema_version": "wsp.intervention-schedule/1.0",
 "kind": "generation-range", "start": 4, "end": 8, "end_inclusive": true, "every": 2}
```

- `first-prefix-forward` resolves to `[0]` and is equivalent to
  `generation-steps` with `[0]`.
- `generation-range` uses **inclusive** bounds by default (`4..8` is five steps);
  `end_inclusive: false` makes it half-open (`4,5,6,7`). The chosen convention is
  stored on the record and echoed in the human-readable summary, so the two are
  never confused.
- `every` is a bounded stride applied to a range. Explicit `steps` lists may not
  carry a stride.

`trigger` is reserved for future event-triggered scheduling (for example "the
first occurrence of a token-level event"). Only `forward` is executable in this
release, and any other trigger is rejected. Schedules are never Python
predicates: a plan must be storable, diffable, and re-executable by a later
caller.

### Rejections

A schedule is rejected when it is negative, out of range, duplicated, unsorted,
longer than `max_events`, at or beyond `max_steps`, mixes explicit steps with
range fields, or omits a required field. `ResourceLimitError`/`InputError` is
raised through the ordinary envelope; nothing is truncated silently.

## Resolved versus executed schedules

`resolve_schedule(schedule, steps_executed=...)` partitions the resolved steps
into:

- **fired/pending** — the step was reached by this execution;
- **not reached** — the step lies beyond the last executed step.

This distinction matters. A step `5` intervention in a 4-token continuation was
never *expected* to fire, while a step `1` intervention that did not fire is a
real failure. The event log and the run validity treat them differently:

| Situation | Event log field | Effect on validity |
| --- | --- | --- |
| Step reached, hook fired | `events` | none |
| Step reached, hook did not fire | `missing_events` | `invalid` |
| Step never reached | `not_reached_events` | `warning`, run stays `valid` if everything reached fired |
| Hook rejected the tensor | `rejected_events` | `invalid` |

`InterventionEventLog.all_fired` is true only when `missing_events` is empty.
"Generation completed" is never accepted as evidence that a plan executed.

## Event records

Each applied event records the plan and intervention identity, generation step,
module forward invocation, module path, tensor role, position mode and
semantics, resolved rows, full-sequence token indices, operation, tensor shape
and dtype, modified element count, dtype cast, replacement digest, bounded
before/after statistics, the execution result (`applied`, `identity`,
`rejected`, `skipped`), and provenance references.

Whole hidden-state tensors are never retained. Only `SliceStats` summaries and
a digest of the replacement slice are stored, and the number of retained
records is capped by `max_event_records`.

## Budgets

Temporal execution is bounded at three levels, and the smallest limit wins:

| Budget | Default | Enforced in |
| --- | --- | --- |
| `max_generation_steps` | 64 | plan validation and the engine loop |
| `max_scheduled_events` | 64 | plan validation |
| `max_temporal_interventions` | 8 | plan validation and the engine |
| `max_event_records` | 512 | the session's event buffer |
| `max_staged_state_bytes` | 2 GB | staged prefill |

Per-plan `TemporalLimits` add `max_interventions`, `max_events_per_step`,
`max_targeted_modules`, `max_total_scheduled_events`, `max_continuation_tokens`,
`max_observed_logit_rows`, `max_selected_elements`, and `max_donor_elements`.

Uncached full-prefix decoding costs `O(steps^2)` token-position evaluations. A
plan records an estimate (`estimated_forward_calls`,
`estimated_token_positions`) before it runs so an unbounded schedule cannot
quietly trigger massive repeated computation.

## Structured operations

```bash
# A plan names a prompt, a mode, and one or more scheduled interventions.
wsp create-intervention-plan --json '{
  "name": "loop onset",
  "prompt_token_ids": [1, 10, 11, 12],
  "execution_mode": "uncached-generation",
  "max_new_tokens": 5,
  "interventions": [
    {"intervention_artifact_id": "a_1111...", "order": 0,
     "schedule": {"kind": "generation-steps", "steps": [2]},
     "position_mode": "last"}
  ]
}'

wsp validate-intervention-plan pl_2222... --source '{"kind":"synthetic"}'
wsp preview-intervention-plan --plan pl_2222... --source '{"kind":"synthetic"}'
wsp run-temporal-intervention --plan pl_2222... --source '{"kind":"synthetic"}'
wsp inspect-intervention-events <event-log-artifact-id>
wsp inspect-intervention-events --run <temporal-run-artifact-id>
```

The same operations are available through `ToolService.call`. See
[`docs/tool-api.md`](tool-api.md) for input fields, and
[`docs/intervention-plans.md`](intervention-plans.md) for multi-site plans and
ordering.

## What a temporal run reports

`wsp.temporal-intervention/1.0` records the plan identity and checksum, the
execution mode and backend, the prompt digest, the decoding configuration, one
`ConditionOutcome` per condition (baseline, identity, plan), the identity
control, the continuation comparison, per-step logit differences, the event log
artifact ids, a validity report, and limitations.

`identity_control.agreement` is one of `exact`, `within_tolerance`, `failed`, or
`not-performed`. A failed identity control sets `validity.status = invalid` and
adds a reason; the difference is still reported, but it cannot be attributed to
the requested intervention.

## Limits of the claim

- A fired event proves the instrument fired where it said it would. It does not
  prove that the targeted component explains a behaviour.
- One step, one site, one token sequence is not a circuit.
- Greedy uncached generation is a controlled experiment, not a production
  decoder.
- Behaviour beyond `max_new_tokens` is not observed.
- Results are specific to the recorded model, dtype, device, backend, and token
  sequence.
