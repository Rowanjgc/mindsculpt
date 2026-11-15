# Staged generation and verification

Staged generation exists to make long temporal experiments practical: prefill a
prefix once, then advance one token at a time instead of recomputing the whole
sequence on every step.

It is strictly optional, and it is never trusted by default. A staged run is
**not** accepted as evidence until an unmodified staged execution has been
compared against the full-prefix uncached reference on the same prompt, and the
comparison is recorded in the artifact.

Correctness takes priority over speed. When an architecture cannot be verified,
the workbench rejects staged execution and says so.

## Four different execution modes

These are separate concepts, and the records keep them separate:

1. **Full-prefix uncached decoding** — every step re-runs the model over the
   whole prefix plus generated tokens. This is the V1.3 continuation mode and the
   reference for staged comparison.
2. **Staged prefix processing** — a prefix is processed once to produce a
   generation state (`prefill`).
3. **Cached autoregressive decoding** — one token is fed per step, with the
   architecture's own cache or recurrent state (`step`).
4. **Architecture-specific recurrent/stateful generation** — architectures whose
   generation state is more than a key/value cache.

Modes 2 and 3 are what V1.4 implements, and only for architectures whose state
is verifiable. Producing the same final text is *not* evidence that two modes
executed the same computation; the verification below compares internals.

## The abstraction

`wsp.execution.staged` defines a `StagedDecoder` protocol:

| Member | Purpose |
| --- | --- |
| `capabilities()` | Supported or not, backend, architecture class, state kind, position/attention handling, and a reason |
| `prefill(token_ids)` | Build a generation state from an explicit prefix; return the logits for the last prefix position |
| `step(state, token_id)` | Advance one token; return logits for that token |
| `clone(state)` | A deep copy that shares no mutable storage |
| `state_hash(state)`, `state_bytes(state)`, `state_length(state)` | Identity, size, and length of the state |

Two implementations exist:

- `TinyStagedDecoder` for the offline fixture `TinyCausalLM`, which has an
  explicit `TinyKVCache` with `clone()` and a tensor-bytes `state_hash()`.
- `HFStagedDecoder` for local Hugging Face checkpoints that use the model's own
  `use_cache=True` path. It does not implement attention kernels or a new
  inference engine.

`build_staged_decoder(model)` classifies the architecture and returns a decoder
or raises `UnsupportedError` before anything executes.

## Architecture classes

`classify_hf_architecture` reports one of:

| Class | Basis | Staged support |
| --- | --- | --- |
| `full-attention-transformer` | `model_type` known and every declared layer is `full_attention` (or no per-layer types are declared and the model type is not hybrid) | Supported, after verification |
| `hybrid-recurrent` | any layer type other than `full_attention`, or a `qwen3_5`/`qwen3_5_moe`/`mamba`/`mamba2`/`zamba`/`jamba` model type | **Rejected** with `UnsupportedError` |
| `unknown` | encoder–decoder, or a configuration that cannot be classified confidently | **Rejected** with `UnsupportedError` |

### Qwen3.5 and hybrid state

Qwen3.5 contains Gated DeltaNet components. Its generation state is not only a
key/value cache, so cloning or manipulating KV tensors cannot produce a valid
independent continuation. V1.4 therefore **explicitly rejects** staged decoding
for hybrid-recurrent architectures:

```
this architecture carries recurrent or hybrid generation state in addition to
key/value attention state; staged decoding is not verified for it in this release
```

Nothing is silently approximated. Scheduled uncached interventions remain
available for those models, and that is the recommended path until a
state-complete verification exists.

## Verification

`verify_staged_generation(plan, model, prompt, ...)` must be run before a staged
intervention can be interpreted. It executes the uncached reference and an
unmodified staged baseline and compares:

- generated token sequences, step by step;
- per-step logits at the predicting row (`allclose`, `max_abs`, `mean_abs`);
- greedy token choices;
- termination decisions;
- the identity-control agreement under staged execution;
- the prefill state identity (`state_hash`) and size.

The result is a `wsp.staged-verification/1.0` record with
`agreement ∈ {exact, within_tolerance, failed}`, explicit `atol`/`rtol`
(default `1e-5`), and named checks (`token_sequence`, `stepwise_logits`,
`termination`, `identity_control`, `cache_isolation`).

Matching generated text alone is never accepted. A record whose tokens match but
whose logits do not is reported as `failed`; a record that claims agreement while
a token sequence mismatched is rejected by the schema.

### Tolerances

`exact` means bitwise-equal logits. `within_tolerance` means `allclose` at the
declared tolerances. On the offline fixture the observed maximum absolute logit
difference is about `3e-8`. On the real checkpoint the tolerance path is the
expected one, because cached and uncached matrix multiplications are not
guaranteed to reduce in the same order.

## Cache and state isolation

Counterfactual conditions must not share mutable cache or recurrent state. Two
isolation modes are recorded:

- `recompute` — each branch prefills and decodes independently. This is the
  default and the safest.
- `clone` — one prefill is deep-copied per branch. `assert_clone_isolation`
  verifies that the source state's hash is unchanged after a branch ran, and
  that the clone's storage is not the original's. A shallow copy is detected and
  raises `ExecutionError`.

`isolation_verified` is recorded on the verification. `isolation: "shared"` is
not a usable configuration; the schema only permits an unverified verification
record to state it.

## Performance measurements

Staged runs record engineering measurements, not capability benchmarks:

- prefill time, per-step decode time, total staged time;
- the uncached reference time for the same continuation, and a speedup ratio;
- peak process RSS where measurable;
- state size in bytes (`cache_state_bytes`).

Uncached full-prefix cost grows quadratically in the number of steps, so the
ratio improves with continuation length. These numbers are reported with the
platform they were measured on; they are not promises about other hardware.

## What has actually been verified

- **Offline fixture (`TinyCausalLM`)** — `exact`-or-`within_tolerance`
  agreement over multiple steps, per-step logits, tokens, termination, and
  identity control. Interrupting isolation is tested by deliberately passing a
  shallow copy, which `assert_clone_isolation` rejects.
- **Real checkpoint (`Qwen3-0.6B`, full attention)** — optional tier-3 tests
  verify staged agreement, multi-step intervention events, and validity through
  the same code path. This checkpoint is not Qwen3.5.
- **Qwen3.5-0.8B** — not present in the validation environment. Its hybrid
  architecture is rejected by classification, which is tested with a stand-in
  configuration. No staged-support claim is made for it.

A real defect was found by exactly this verification and is worth recording: an
early `HFStagedDecoder` fed a length-1 attention mask on cached decode steps, so
the architecture derived position 0 for the new token. Step 0 matched and step 1
diverged by 13.99 in absolute logits. The fix builds the mask (and
`cache_position` where the architecture accepts it) over the full cached
sequence. The staged and uncached token streams now agree. Had verification been
skipped, the staged engine would have produced confident nonsense.

## Limitations

- Staged support is limited to architectures that pass classification and
  verification. Everything else is rejected.
- The HF decoder uses the model's own forward path; it does not support
  architectures that require a custom generation loop.
- `logits_to_keep`-style truncation is not used; full-vocabulary logits are
  compared.
- Only greedy decoding is supported in either mode.
- Staged verification is per prompt, per model, per dtype, and per device. A
  verification recorded for one prompt does not license another.
- The staged record stores a verification block; a staged intervention whose
  verification is not `exact`/`within_tolerance` cannot have `validity: valid`.
