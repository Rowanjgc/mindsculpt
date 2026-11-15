# Fixed-context counterfactuals

A generated continuation can diverge after an intervention. Once two token
histories differ, later measurements are no longer made under identical
conditions. Fixed-context replay removes that confound: every condition
consumes exactly the same recorded token identifiers, in the same order, with
the same attention mask and position convention, and differs only in the
selected activation.

## Two artifacts

`wsp.fixed-context-replay/1.0` records the reconstructed prefix:

- source trace and trace artifact;
- selected replay step and consumed position;
- exact input and generated token identifiers;
- token sequence digest;
- model and tokenizer references;
- replay checks, warnings, and limitations.

Preparation reuses V1.2 `prepare_plan`. Token identifiers are taken from the
trace. Re-deriving from stored text is allowed only when the trace records the
tokenizer and no chat template was used; the replay marks the reconstruction
`re-derived-from-text`. A trace without exact token identities fails rather
than inventing a prefix.

`wsp.fixed-context-comparison/1.0` records the experiment:

- baseline, identity, and requested intervention conditions;
- the shared token digest and full sequence length;
- selected logit positions, token log-probabilities, ranks, and top-1 tokens;
- downstream activation differences and effect sizes;
- identity-control status and a validity report;
- optional continuation, kept separate;
- provenance, warnings, limitations, and checksums.

## Mandatory controls

Every comparison runs:

1. **Baseline** — no hooks, untouched forward.
2. **Identity control** — the same hook site and selection, with the original
   activation restored. This verifies that the instrumentation itself does not
   change the computation.
3. **Requested intervention** — zero, mean, donor patch, or identity.

The identity control must pass. A failure sets the comparison's validity to
`invalid` and adds an explicit reason: the observed difference cannot be
attributed solely to the requested intervention. The tool does not average over
the failure or hide it.

The target must be the selected replay position. Targeting a different position
is rejected, because that would not test the forward the investigator selected.

## Measurement

The comparison reports:

- selected next-token log-probabilities under baseline, identity, and
  intervention;
- per-position deltas;
- top-1 token ids;
- logit difference statistics over the selected rows;
- optional downstream activation statistics for explicitly named modules and
  positions;
- resource counts and execution time.

The numeric implementation is the existing V1.3 paired-intervention engine.
V1.5 adds prefix selection, fixed-context validation, event linkage, and the
comparison schema; it does not duplicate the arithmetic.

## Optional continuation

A bounded greedy continuation may be requested. It is stored in a separate
field. The intervention is applied only on the first continuation forward; the
record says so. Once tokens diverge, later continuation states are conditioned
on different histories and are not fixed-context measurements. The record
reports the common prefix, first divergence, length delta, termination change,
and token agreement rate.

## What a successful comparison establishes

It establishes that, under the recorded model, dtype, backend, token sequence,
selected positions, and operation, changing the selected activation changed a
measured downstream quantity.

It does not establish:

- the semantic meaning of the activation;
- the complete mechanism behind a behavior;
- that the intervention repairs the model;
- that an equivalent permanent weight edit exists;
- that the same effect generalises to another prompt or model.

These limits are carried in the artifact and in dossiers generated from it.
