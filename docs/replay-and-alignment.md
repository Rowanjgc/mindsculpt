# Replay and token alignment

V1.1 records generation traces. A trace is an *observation of behavior*: it does
not by itself establish that the workbench can reconstruct the execution that
produced it. V1.2 adds an explicit replay mechanism with its own verification
semantics, and an alignment layer that maps behavioral spans onto forward-pass
positions.

The most important thing in this document is that **four different guarantees
are tracked separately**. Conflating them is the easiest way to publish a
misleading measurement.

| Question | Field | Meaning |
| --- | --- | --- |
| Did we recover the exact token sequence that was presented to the model? | `token_sequence_reconstructed` | Recorded token ids were used as-is, or re-derived from text with a recorded encoding |
| Did a forward pass actually run? | `forward_executed` | A bounded forward pass completed on the loaded model |
| Do the replay's numbers agree with the recorded ones? | `numerical_agreement` | `verified`, `partial`, `mismatch`, or `unavailable` |
| Would sampling reproduce the original generation? | `generation_reproducibility` | Always `not-claimed` for teacher-forced replay |

A replay can succeed as a forward execution while numerical agreement is
`unavailable` (the trace recorded no log-probabilities) or `mismatch` (different
weights, precision, or backend). It is still reported honestly as an
*exploratory* capture, never as verified evidence.

## Teacher-forced replay

`prepare_replay` and `verify_replay` implement teacher forcing: the recorded
prompt *and* the recorded generated tokens are replayed as a known sequence, and
the model is asked what it would have predicted at each step. The model is never
asked to generate, so:

- a successful replay says nothing about whether sampling would produce the same
  continuation;
- a successful replay does say whether the recorded per-token log-probabilities
  are reproducible on this machine with this checkpoint;
- attention masks, position ids, and the inference mode are fixed and recorded,
  because a different mask or position scheme changes the numbers.

The plan is explicit about its source:

| `sequence.source` | Meaning |
| --- | --- |
| `recorded-token-ids` | The trace stored prompt token ids; they are used verbatim and checked against `input.tokens_sha256` |
| `re-derived-from-text` | The trace stored text only; the text was re-encoded with an explicit `encoder` |

A chat-template trace without stored token ids is refused. Re-rendering a chat
template can change whitespace, special tokens, or the assistant prefix, and
silently guessing would make every subsequent measurement unattributable.

```bash
wsp replay-plan tr_1f2e...            # write a replay plan artifact
wsp replay tr_1f2e... --model '{"kind":"local_hf","path":"./ckpt"}'
wsp replay --plan a_9c31... --model '{"kind":"local_hf","path":"./ckpt"}'
```

```python
from wsp.tools.service import ToolService

service = ToolService(raise_errors=True)
plan = service.call("prepare_replay", {"trace_id": trace_id})["data"]
result = service.call(
    "verify_replay",
    {"plan_artifact_id": plan["plan_artifact_id"], "model": {"kind": "synthetic"}},
)["data"]
print(result["numerical_agreement"], result["generation_reproducibility"])
```

### Verification checks

`prepare_replay` records what it could and could not verify, one check per row
with status `pass`, `warning`, `fail`, `unavailable`, or `skipped`:

- `input_token_ids`, `input_tokens_checksum`, `generated_token_ids`,
  `recorded_logprobs`, `recorded_output_checksum`, `sequence_length`;
- `model_identity`, `tokenizer_identity`, `chat_template`, `special_tokens`;
- `attention_mask`, `position_ids`, `generation_reproducibility`.

`verify_replay` then compares the loaded model against the plan and fails with
`CompatibilityError` listing the failing checks when identities disagree. The
checks that can only fail at execution time are:

- `model_file_checksum`, `model_structure_fingerprint`, `model_parameter_count`;
- `tokenizer_fingerprint`, `tokenizer_vocab_size`, `chat_template`;
- `token_ids_in_vocabulary`, `parameters_unchanged`, `numerical_agreement`.

Token ids are validated against the **embedding matrix**, not against
`config.vocab_size`. Multimodal and hybrid checkpoints routinely pad the
embedding beyond the configured text vocabulary, and the tokenizer emits the
padded ids; using the config value rejects valid input. This was found while
testing a real Qwen3.5 checkpoint (config vocabulary 248 045, embedding rows
248 320).

### Numerical agreement

Agreement is computed over the recorded tokens' log-probabilities:

- logits at position `p` predict the token at `p + 1`, so token `i` of the
  generation is compared at forward position `prompt_length - 1 + i`;
- agreement per token is `|replay - recorded| <= logprob_atol + logprob_rtol * |recorded|`;
- `verified` requires the within-tolerance fraction to reach
  `verified_min_fraction` (default 1.0) **and** top-1 agreement to reach
  `top1_min_agreement` (default 0.95);
- between 0.5 and 1.0 within tolerance is `partial`; below that is `mismatch`.

The tolerances are part of the plan and are reported in the result, so a
measurement is never "verified" against an undeclared standard.

### Resource limits

`ReplayExecutionConfig` and `Budget` bound the work:

| Field | Default | Effect |
| --- | --- | --- |
| `execution.max_sequence_length` | 2048 | Refuses longer replays |
| `budget.max_replay_sequence_tokens` | 4096 | Hard ceiling under the config |
| `budget.max_replay_logits_elements` | 200 000 000 | Refuses forwards whose logits would not fit the budget |
| `budget.max_replay_agreement_tokens` | 512 | Caps stored per-token agreement rows |
| `budget.max_replay_sequence_tokens` | 4096 | Also used by alignment and capture |

Every replay stores only bounded output: the per-token agreement table is
truncated with `per_token_truncated: true` rather than silently shortened.

Replay runs under `torch.inference_mode()` with the module in eval mode, and the
previous training flag is restored in a `finally` block. A parameter fingerprint
is taken before and after the forward pass; a mismatch raises `ExecutionError`
instead of returning numbers, because a mutated model invalidates the
measurement.

### Hybrid architectures

Replay uses the model's own forward pass — no custom kernels. For recurrent or
hybrid stacks (for example Gated DeltaNet layers interleaved with full
attention), the workbench:

- passes the full sequence in one call, so the layer's supported state handling
  is used rather than an approximation;
- records a `hybrid_architecture_note` limitation on the plan and result;
- reports `capabilities["hybrid_layers"]` in the model identity.

Stateful layers may compute different intermediates than a cached autoregressive
run over a shorter window. The replay records the limitation rather than
assuming equivalence.

## Token alignment

A behavioral detector reports character or word spans in text. Activations live
at token positions in a forward pass. `align_trace_tokens` bridges them.

```bash
wsp align tr_1f2e... --model '{"kind":"local_hf","path":"./ckpt"}'
wsp resolve-span tr_1f2e... --kind token_range --role generation --start 0 --end 4
```

The alignment record (`wsp.alignment/1.0`) contains one `AlignmentPosition` per
full-sequence position:

| Field | Meaning |
| --- | --- |
| `full_index` | Position in the replayed sequence |
| `role` / `role_index` | `prompt` or `generation`, plus the index inside that segment |
| `token_id`, `text` | The token, when the trace recorded it |
| `char_start` / `char_end` | Character range in the recorded text, when offsets are trustworthy |
| `byte_start` / `byte_end` | UTF-8 byte range derived from the character range |
| `special` | `true`/`false` when a special-token set was supplied, else `null` |
| `offset_source` | `tokenizer`, `derived`, or `absent` |
| `consumes_token` | Whether this position consumes the token it holds |
| `predicts_full_index` | Position whose logits predict this token (`full_index + 1`) |
| `predicted_by_full_index` | Position whose state predicts this token |

Prompt-relative, generation-relative, and full-sequence indices are **never**
interchangeable. `TokenAlignment` exposes `to_full_index(role, index)` and
`to_role_index(full_index)` so a caller cannot mix them by accident.

### Prediction position semantics

For a causal language model, the activation at position `p` predicts the token at
`p + 1`. The state at the *last* prompt position is the one that produces the
first generated token, and the state at the last generated token predicts
nothing at all.

The alignment says this explicitly:

- `prediction_semantics`: `"position-p-predicts-token-p+1"`;
- `consuming_positions_for(indices)`: the positions that consume those tokens;
- `predicting_positions_for(indices)`: `i - 1` for each token index, de-duplicated.

Index 0 has no predicting position, and a request that selects it is reported as
ambiguous instead of being mapped to a neighbour.

Capture requests therefore state their semantics:

```json
{ "position_semantics": "consuming" }   // state at the token position
{ "position_semantics": "predicting" }  // state that predicts the token (i - 1)
```

## Span resolution

`resolve_finding_span` turns a `Finding` evidence reference, a token range, or a
text span into a `SpanResolution` (`wsp.span-resolution/1.0`):

| Status | Meaning |
| --- | --- |
| `resolved` | One unambiguous set of token positions |
| `ambiguous` | Positions found, but the mapping is not exact; the ambiguity is described |
| `unresolved` | No defensible mapping, with a reason |

Ambiguity is reported, never smoothed away:

- a text span that starts or ends inside a multi-character token adds
  "token at <role> index N extends beyond the requested span" and reports the
  coverage fraction;
- a span with no character offsets (for example after a text mismatch or a
  Unicode token the tokenizer cannot encode round-trip) is `unresolved`;
- selecting index 0 adds "the first token of the sequence has no predicting
  position";
- V1.1 `word_range` evidence stores character offsets, not word indices; the
  resolution says so and interprets them as characters.

### Offsets

Character offsets are used only when they are consistent with the recorded text:

- a fast tokenizer's offsets are accepted only if re-encoding its pieces
  reproduces the recorded token ids exactly;
- otherwise offsets are derived by concatenating the recorded decoded pieces and
  requiring exact equality with the recorded text;
- if either check fails, every character range is `null`, the segment's
  `text_status` stays `available`, and `unavailable["prompt_char_ranges"|"generation_char_ranges"]`
  explains why.

Byte ranges are computed from the character ranges with a codepoint-to-`byte`
prefix table, so non-ASCII text keeps correct byte offsets.

### Context, not truncation

A span inside the generation does not license a local window. By default a span
resolution reports `context_mode: "full-sequence"`. Requesting a window returns
`context_mode: "explicit-window"` **and** a warning: a window that does not start
at position zero does not preserve the causal prefix, so activations measured
inside it are a different-context approximation. The workbench never silently
turns a local window into a claimed equivalent measurement.

## What replay cannot tell you

- Whether the model would generate the same tokens again (sampling,
  temperature, backend kernels, and caching all matter).
- Whether a small numerical difference matters. `verified` means the recorded
  log-probabilities were reproduced within the declared tolerance, nothing more.
- Whether the original execution used the same attention implementation, dtype,
  or device. Those are recorded when known, and a `bf16`/`fp16` or `cuda`
  setting adds an explicit limitation.
- Whether a measurement is causal. Replay and alignment locate *states*; they do
  not attribute effects.

Replay is deliberately `use_cache=False` and teacher-forced. Causal
interventions are a separate layer built on these artifacts; see
`docs/causal-interventions.md`. Controlled continuations use an explicit
uncached, full-prefix mode and never claim cached-generation compatibility; see
`docs/counterfactual-continuations.md`.

## V1.5 fixed-context replay

`prepare_fixed_context_replay` wraps the V1.2 replay plan. It requires exact
recorded input and generated token identifiers, builds the full sequence, checks
the sequence digest and length budget, and records the selected replay step and
consumed position. It does not reconstruct exact prefixes from decoded text
unless the trace recorded the necessary tokenizer information.

`run_fixed_context_comparison` reuses the V1.3 paired-intervention engine over
that exact sequence. Baseline, identity, and intervention all consume the same
token identifiers. The selected intervention must target the consumed position
of the selected replay step.

Alignment status matters: an event marked `ambiguous` or `unavailable` cannot be
used for fixed-context targeting.
