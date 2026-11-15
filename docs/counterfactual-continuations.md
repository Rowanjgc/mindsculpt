# Counterfactual continuations

A paired teacher-forced measurement answers "what changed in the numbers?".
A continuation experiment answers a narrower behavioural question: "given the
same prefix, does the intervention change which tokens the model generates?".

V1.3 implements continuations as a **separate mode**, not as an implicit
extension of replay.

Schema: `wsp.continuation-comparison/1.0`
Artifact kind: `continuation_comparison`

## Execution mode

V1.3 supports one explicit mode:

* decoding: greedy (`argmax`) only,
* execution: **uncached full-prefix**. Each step re-runs the whole prefix
  through the model and reads the final position,
* `use_cache=False`, `do_sample=False`, `temperature=0`,
* intervention event: `first-prefix-forward`.

There is no KV-cache manipulation, no stateful generation, and no claim that the
result matches a cached or batched inference path. The record carries
`execution_mode`, `intervention_event`, and metadata stating the limitation.

Sampling-based continuations are not implemented. A request with sampling
parameters is rejected rather than silently downgraded.

## Position semantics

The intervention is applied during the **first forward pass over the shared
prefix**, at the module and positions declared in the spec. `apply_on_calls=[0]`
is the only supported event for continuations. This is deliberately distinct
from:

* intervening while processing a selected continuation token (not implemented),
* intervening on every generation step (not implemented),
* intervening inside a cached decode step (not implemented).

The spec must have `execution.execution_mode == "uncached-continuation"`.
Specs created from a teacher-forced capture are retargeted explicitly by the
tool handler, and the retargeting is visible in `execution.sequence_length`.
Inline specs must declare the prefix length themselves.

## Prefix requirements

The continuation record preserves:

* prefix token ids and their SHA-256,
* prefix length,
* generation configuration (`max_new_tokens`, `max_sequence_length`, EOS id,
  seed, decoding strategy, cache and sampling flags),
* backend and backend version,
* the intervention id and spec digest,
* per-condition generated token ids, decoded text when a tokenizer is
  available, per-step log-probabilities, step count, termination reason, and
  intervention events.

The prefix is reconstructed by the caller. It is not re-derived from text unless
the caller explicitly supplies token ids from a trace.

## Conditions and validity

Three conditions run from the same prefix with the same settings:

1. baseline (no hook),
2. identity control (the same hook path, identity operation),
3. intervention.

Baseline and identity must generate the same tokens under deterministic greedy
decoding. Agreement is `exact` or `failed`; there is no tolerance band for token
ids. A failure marks the experiment `invalid` with a diagnostic. Disabling the
identity control marks it `limited`.

Validity is also `invalid` when a condition raises an error or when no
intervention event was recorded.

## Comparison

`ContinuationComparison` records:

* `longest_common_prefix`,
* `first_divergence` (equal to the longest common prefix when the sequences
  differ, `null` when one sequence is a prefix of the other or they match),
* `length_delta`,
* `termination_changed`,
* `token_agreement_rate` (longest common prefix divided by the longer length),
* generated tokens per condition.

A shorter continuation is **not** labelled better. A different token is **not**
labelled a repair. The fixed interpretation string states that the result is a
behavioural difference under the tested conditions.

## Termination reasons

* `eos` – the configured end-of-sequence token was generated,
* `max_new_tokens` – the configured new-token budget was reached,
* `max_sequence_length` – the prefix plus generated tokens reached the declared
  sequence limit,
* `error` – the condition failed; the error text is preserved on the condition.

## Resource limits

Defaults in `Budget`:

* `max_continuation_tokens = 64`
* `max_continuation_sequence_tokens = 4_096`

The per-request `max_new_tokens` and `max_sequence_length` can only tighten these
limits, never exceed them. Each generated step is a full forward pass; the mode
trades throughput for reproducibility and is not a production inference engine.

## Supported models

Any model reachable through the replay backend with the `torch_hooks`
instrumentation path is supported. `nnterp` intervention support is deferred; a
continuation request that selects it is rejected. Architectures whose forward
path cannot be called with a plain token-id tensor, or whose module call does not
expose a tensor at the requested point, are rejected explicitly rather than
processed approximately.

## Tools

| Operation | Purpose |
| --- | --- |
| `run_counterfactual_continuation` | Run the three conditions and save the record. |
| `inspect_counterfactual_comparison` | Read either comparison kind with integrity checks. |

The CLI mirrors both operations (`run-counterfactual-continuation`,
`inspect-counterfactual-comparison`). Every command accepts `--json` for the full
request payload.

## Reproducibility

Deterministic greedy decoding on CPU is reproducible for a fixed model, dtype,
backend, and token sequence. The record distinguishes:

* **deterministic token agreement** (baseline vs identity, exact token ids),
* **numerical agreement** (tolerance-based comparisons in paired measurements,
  not used to declare continuation equality).

GPU kernels, non-deterministic reductions, or a different backend version can
change results. The provenance fields record what actually ran; they do not
promise portability of the numbers.

## What a continuation does not show

* that the intervention improved the model,
* that the changed token is the "correct" one,
* that the effect holds under sampling or cached decoding,
* that the targeted component explains the original behaviour.

Continuations are behavioural evidence under controlled conditions. Causal
interpretation requires replication, additional controls, and evaluation outside
this milestone.

## Where V1.4 continues

This document describes the V1.3 continuation mode: greedy, uncached,
full-prefix, with the intervention applied only during the first prefix forward.
That mode is unchanged and remains the reference implementation.

V1.4 generalizes the same decoding path to arbitrary generation steps (see
[`docs/temporal-interventions.md`](temporal-interventions.md)), adds multi-site
plans and composition ([`docs/intervention-plans.md`](intervention-plans.md)),
and adds *optional* verified cached decoding
([`docs/staged-generation.md`](staged-generation.md)). Staged results are only
reported as such after an unmodified staged run agrees with this uncached
reference on tokens, per-step logits, termination, and identity control.

A claim that a modified activation "persists" across steps is still false in
uncached decoding. A sustained schedule is an explicit repeat, not persistence.
