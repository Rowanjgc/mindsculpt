# Activation investigation

This document covers the V1.2 measurement layer: targeted activation capture,
controlled comparison, and persistent investigation evidence. It is the layer
that turns "the model did something interesting here" into stored numbers with
provenance.

The workbench deliberately separates five things. Do not collapse them.

1. **Observations** — traces, detector findings, imported logs.
2. **Numerical measurements** — replay verification, captures, comparisons.
3. **Hypotheses** — human-authored statements, always `unverified`.
4. **Causal interventions** — V1.3, separate specifications and paired
   measurements (`docs/causal-interventions.md`).
5. **Weight edits** — V1 patches, separate artifacts.

V1.2 ships the infrastructure for 1–3. It never generates 3 automatically and
never claims 4. V1.3 adds 4 without merging it into measurement or weight
editing.

## The investigation chain

```text
trace ──> detector finding ──> selected span ──> replay ──> alignment
                                                    │
                                                    v
                                          activation capture
                                                    │
                                                    v
                                          controlled comparison
                                                    │
                                                    v
                                     investigation evidence ──> case export
```

Every stage produces a stored artifact with a stable id, and every later stage
records the ids it depends on, so a case can be re-verified without re-running
anything:

| Stage | Artifact kind | Id form |
| --- | --- | --- |
| Trace | `generation_trace` | `tr_…` |
| Replay plan | `replay_plan` | `rp_…` |
| Replay result | `replay_result` | `ry_…` |
| Token alignment | `token_alignment` | `al_…` |
| Span resolution | `span_resolution` | `sp_…` |
| Capture record | `trace_capture` | `tc_…` |
| Retained vector | `activation_vector` | `a_…` (content-addressed) |
| Comparison | `capture_comparison` | `cmp_…` |
| Evidence | `investigation_evidence` | `inv_…` |

Artifact ids are `a_` plus the first 32 hex characters of the SHA-256 of the
stored bytes, so identical content is stored once and a checksum mismatch is
always detectable.

## Capture points

`list_capture_points` enumerates addressable modules and classifies them
(`attention`, `linear-attention`, `mlp`, `norm`, `embedding`, `head`, `block`,
`recurrent`, `aggregate`, `other`). Classification is by name and class, so a
hybrid stack (Gated DeltaNet + full attention) appears without the capture code
knowing the architecture.

```bash
wsp capture-points --model '{"kind":"local_hf","path":"./ckpt"}' --kind block
wsp capture-points --model '{"kind":"local_hf","path":"./ckpt"}' --instrumentation nnterp
```

Two instrumentation backends exist:

| Backend | Path style | Availability |
| --- | --- | --- |
| `torch_hooks` (default) | The checkpoint's own module names (`model.layers.11`, `blocks.1`) | Always |
| `nnterp` (optional) | Normalised names (`layers.11`, `layers.11.self_attn`, `embed_tokens`) | Requires the optional `instrumentation` extra |

The backend is explicit in every request and recorded in every capture record
(`backend`, `backend_version`). There is no silent substitution: asking for
`nnterp` without the dependencies raises `UnsupportedError` with the install
hint, and asking for it with a synthetic or non-Hugging-Face source raises
`InputError`.

## Capture modes

### `summary` (default)

Computes bounded statistics per measured tensor slice and retains no values:

- count, finite/nan/inf counts, zero count;
- mean, standard deviation, min, max, absolute max;
- L2 norm;
- the top eight dimensions by absolute value.

This is the right default: a trace with 2 000 generated tokens across 24 layers
would otherwise produce gigabytes of hidden states.

### `selected`

Retains explicitly requested activation vectors, each stored as a checksummed
`.npy` artifact with shape, dtype, device, module, position, source trace, replay
id, sampling stride, and schema version in its metadata.

A vector larger than `budget.max_activation_elements_per_tensor` (500 000
elements by default) is stride-sampled, and the record says so
(`sampled: true`, `stride: N`). Nothing is silently down-cast.

## Resource limits

| Limit | Default | Behaviour |
| --- | --- | --- |
| `budget.max_capture_positions` | 1024 | Refuses more requested token positions |
| `budget.max_capture_modules` | 32 | Refuses more requested modules |
| `budget.max_activation_elements_per_tensor` | 500 000 | Stride-samples beyond it |
| `budget.max_activation_elements_total` | 2 000 000 | `ResourceLimitError` when exceeded |
| `budget.max_capture_artifacts` | 64 | Bounds capture-point discovery |

`CaptureResources` in every record reports positions, modules, measurements,
elements measured, elements retained, artifact count, estimated retained bytes,
and the limits that were in force.

## Hook lifecycle

The torch backend installs forward hooks on the selected modules, records the
requested positions of each call, and removes every handle in a `finally`
block:

- hooks are removed on success *and* on exceptions;
- outputs are read, never written, so instrumentation cannot change the forward
  pass;
- retained tensors are detached and copied to NumPy before the pass continues;
- the forward runs under `torch.inference_mode()`;
- the module's previous training flag is restored;
- a parameter fingerprint is compared before and after, and a change raises
  `ExecutionError` rather than returning numbers.

Captures run through `execute_replay`, the same path as verification. A capture
is therefore never a separately trusted code path: it inherits replay identity
checks, the immutability check, and the verification status.

## Position semantics

The capture request must say which state it wants:

- `consuming` — the state at the token position;
- `predicting` — the state that predicts the token, i.e. position `i - 1`.

Token index 0 has no predicting position; it is dropped with a warning rather
than silently mapped to position 0. Both the requested tokens and the actual
positions are stored, so a reader can always reconstruct the mapping.

## Comparing captures

`compare_activation_captures` validates compatibility before computing anything:

- same model (file checksum, structure fingerprint, parameter count) unless
  `allow_different_models` is explicitly set;
- same module set, same position semantics, same capture mode, same sequence
  length — mismatches become compatibility flags and warnings;
- identical tensor shapes for elementwise metrics.

Position alignment is explicit, and two of the four strategies require human
review:

| Strategy | Requires review | Meaning |
| --- | --- | --- |
| `identical_positions` | no | Compare positions present in both captures |
| `matching_prefix` | no | Compare the shared prefix |
| `paired_positions` | yes | Investigator supplies explicit left/right pairs |
| `semantic_event` | yes | Investigator asserts two positions represent the same event |

The workbench never invents a pairing. Unreviewed explicit alignments are
rejected by the schema.

Metrics are computed from the shared tensor-difference implementation, not a
second numerical stack:

| Metric | Definition |
| --- | --- |
| `l2` | Euclidian norm of `left - right` |
| `rel_l2` | `l2 / ||right||` (`null` when the right norm is zero) |
| `cosine` | Cosine similarity |
| `dot` | `left · right` |
| `max_abs`, `mean_abs`, `rms` | Difference magnitude statistics |
| `projection` | `(left - right) · direction / ||direction||`, with an explicit direction |

A projection without a direction is refused; a direction whose length does not
match the tensor produces `null` with a note instead of a fabricated number.

Summary captures can be compared, but only through their aggregate statistics:
`values_compared` is `false`, elementwise metrics are absent, and
`unavailable["elementwise_metrics"]` explains why.

Every comparison carries:

- `unavailable["causal_interpretation"]`: a comparison reports numerical
  relationships, not mechanisms;
- limitations stating that a large difference is not evidence of a causal role
  and that a small difference is not evidence against one;
- the exact `atol`/`rtol` used.

## Investigation evidence

`attach_investigation_evidence` packages the measurement chain into one record
(`wsp.investigation-evidence/1.0`) and optionally links it to a failure case.

It holds:

- `findings` — detector references with their evidence and span artifact;
- `replays` — plan/result ids plus the agreement level;
- `alignments`, `spans` — artifact links;
- `captures` — capture ids, modes, backends, positions, replay ids;
- `comparisons` — comparison ids, metrics, alignment strategy;
- `hypotheses` — human statements with `status: "unverified"`;
- `limitations`, `warnings`, `privacy`, `checksums`.

Validated invariants (enforced by the model, not by convention):

- at least one finding reference or one measurement artifact is required;
- every comparison's capture ids must be present in `captures`;
- hypothesis capture/comparison references must exist;
- a replay recorded as `evidence_level: verified` must have
  `numerical_agreement: verified`.

```bash
wsp attach-evidence --title "repetition finding" --case case_1a2b... \\
    --observation tr_1f2e... --control tr_3c4d... \\
    --capture a_1111... --capture a_2222... \\
    --comparison a_3333... \\
    --hypothesis "the trailing stop token changes the next-token distribution"

wsp inspect-evidence a_4444...
```

`inspect_investigation_evidence` re-reads every referenced artifact and reports:

- `checksum_matches` — recomputed evidence checksum against the stored one
  (`null` when a redacted export dropped the checksum, so "not applicable" is
  never confused with "tampered");
- `missing_artifacts` and `kind_mismatches`;
- capture and comparison digests.

## Failures are first-class

If a replay cannot be verified, the capture still exists — as *exploratory*
evidence, with a reason:

- `numerical_agreement: mismatch` → `evidence_level: exploratory`, and the
  `numerical_agreement` check has status `fail`;
- `numerical_agreement: unavailable` (no recorded log-probabilities) →
  `exploratory`, with `unavailable_reason`;
- `forward_executed: false` → no capture at all, `InputError`.

The workbench never upgrades an exploratory capture to verified because the
numbers "look fine".

## Privacy

Activations, log-probabilities, and token ids are derived from the source
prompt. A redacted case export therefore:

- keeps capture **summary statistics** (they are the measurement) but removes
  retained vectors, their checksums, and their artifact references;
- drops `activation_vector` artifacts and records each omission in
  `artifact_map` (`old_id: null`) plus a limitation;
- drops links to replay, alignment, and span artifacts, because those artifacts
  embed recorded token text;
- redacts evidence titles, summaries, notes, hypothesis statements, detector
  values, and capture metadata;
- deletes the evidence checksum and reports `checksum_matches: null` on read.

A private export keeps everything, including vectors and replay artifacts, and
round-trips byte-for-byte. See `docs/failure-cases.md` for the policy itself.

## What V1.2 does not do

- No automatic mechanism discovery, circuit localisation, or neuron labelling.
  A dimension is a dimension; a label belongs in a hypothesis.
- No generation-time activation recording. Measurement happens on a
  teacher-forced replay of recorded tokens.
- No GPU-specific paths. The baseline is CPU.

V1.3 adds the controlled-intervention layer on top of these captures. See
`docs/causal-interventions.md` for intervention semantics and
`docs/counterfactual-continuations.md` for bounded generation-time comparisons.

## Where V1.4 continues

Captures remain the way to *observe* internal state. V1.4 adds the ability to
*intervene* at a chosen generation step, at several sites in one forward, and
repeatedly over a bounded range:

* [`docs/temporal-interventions.md`](temporal-interventions.md)
* [`docs/intervention-plans.md`](intervention-plans.md)
* [`docs/staged-generation.md`](staged-generation.md)

Capture artifacts, alignment artifacts, and trace ids are still referenced by
intervention execution identities, so temporal evidence remains linked to the
observation it came from.
