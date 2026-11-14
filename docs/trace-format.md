# Generation trace format

Schema: `wsp.trace/1.0`
Canonical storage: content-addressed artifact, `kind="generation_trace"`
Index: `wsp.trace-index/1.0` at `traces/<trace_id>.json`

A trace is an observation of one generation. It records what was measured and
what was not. It never fills a missing measurement with a guess.

## Identity and time

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Always `wsp.trace/1.0` |
| `trace_id` | string | `tr_` + 16 lowercase hex characters |
| `created_at` | ISO-8601 UTC | Observation time |
| `checksums` | object | `output_sha256`, `tokens_sha256` when measurable |

The canonical artifact is immutable. Its SHA-256 is the `ArtifactRef.sha256`.
The trace index is a small mutable pointer; deleting an index entry does not
delete the artifact.

## Model, tokenizer, backend, input

| Field | Notes |
| --- | --- |
| `model` | `name`, `revision`, optional local `checkpoint_path`, `structure_fingerprint`, `file_sha256`, `parameter_count`, metadata |
| `tokenizer` | `name`, `revision`, `kind`, `vocab_size`, `fingerprint`; optional |
| `input` | Exact `text`, optional `system`, chat `messages`, recorded `token_ids` with `tokens_sha256`, `token_encoding` (`chat_template`, `text`, `unknown`), `add_special_tokens`, optional artifact reference and SHA-256 |
| `prompt` | Optional template name/hash, rendered prompt, and `add_generation_prompt` |
| `backend` | Backend name/version, device, dtype, determinism hint, runtime versions |
| `params` | `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop_sequences`, sampling flag, log-probability options, extra settings |
| `seed` | Integer seed, or null when unknown |

Inputs are preserved exactly when privacy policy permits. A test that claims to
be controlled must copy its exact inputs into its result; contrast results do
this.

## Tokens and text

`tokens` is an ordered list. The list index **is** the token position; there is
no separate index field.

| Token field | Meaning |
| --- | --- |
| `token_id` | Tokenizer id, or null when unavailable |
| `text` | Token text, or null when unavailable/redacted |
| `logprob` | Log-probability, or null when the backend did not expose scores |
| `top_logprobs` | Optional alternatives with ids, text, and log-probabilities |

`text` is the decoded output. Where both tokens and text exist, they are kept
independently because returning a backend's decoded text is more faithful than
reconstructing it. Detectors prefer token text when every token has it, and fall
back to whitespace-delimited words otherwise. The measurement records which unit
was used.

`annotations` are named half-open token ranges:

```json
{"name": "loop", "start": 3, "end": 11, "metadata": {}}
```

They are valid only when the trace has token records. Importers that cannot
express an annotation in token coordinates preserve it in metadata and emit a
warning rather than attaching it to the wrong positions.

`tool_calls` records available tool actions:

```json
{
  "index": 0,
  "name": "search",
  "arguments": {"query": "a"},
  "arguments_sha256": "sha256:...",
  "status": "ok",
  "result_summary": null
}
```

`index` is the call sequence number, not a token position. Arguments may contain
sensitive data and are removed by export redaction.

## Termination

`termination.reason` is one of:

| Reason | Meaning |
| --- | --- |
| `eos` | The backend reported an end-of-sequence token |
| `stop_sequence` | A configured stop string appeared (`stop_sequence` names it) |
| `tool_handoff` | The backend ended the turn for a tool call |
| `length_limit` | A configured token or context limit was reached |
| `error` | Execution failed during generation |
| `unknown` | The backend did not provide a recognized reason |

`detail` is backend-specific text and must not be interpreted as a cause.
Inspect logs often report `stop_reason="stop"`, which cannot distinguish a
natural EOS from a configured stop sequence; the importer records that ambiguity
in `warnings`.

`generated_tokens` equals the token-record count when records exist. It is null
when the backend reported only text.

## Timing and provenance

`timing` has `started_at`, `completed_at`, `duration_ms`, and
`tokens_per_second` when measurable. Imported logs often provide only a total
duration or nothing.

`provenance` distinguishes `generated` from `imported`:

| Field | Meaning |
| --- | --- |
| `origin` | `generated` or `imported` |
| `source_system` | e.g. `wsp.behavior.scripted`, `inif`, `inspect_ai` |
| `source_version` | Upstream format or package version |
| `source_ref` | Local source path or label |
| `source_sha256` | Whole-source-file checksum |
| `importer` | Adapter version, e.g. `wsp.inif/1.0` |
| `imported_at` | Import time |
| `created_by` | Optional human/process label |

## Unavailable data

`unavailable` is a map from field name to a human-readable reason, for example:

```json
{
  "tokens": "scripted fixture produced text without token records",
  "token_logprobs": "no token records were produced",
  "termination": "INIF does not encode a termination reason"
}
```

Fields can also simply be null. Null means "no value"; `unavailable` explains
why. No importer invents token ids, log-probabilities, timings, or termination
reasons. `warnings` carries non-fatal issues that do not map to one field.

## Privacy controls

`privacy` is copied from the collection request:

```json
{
  "store_input_text": true,
  "store_output_text": true,
  "classification": "unspecified",
  "note": null,
  "redacted": false
}
```

- `store_input_text=false` removes input text and chat messages, clears the
  rendered prompt, and adds an `unavailable` explanation.
- `store_output_text=false` removes decoded text and every token's text, but
  keeps token ids and positions.
- `redacted=true` records that stripping happened, either at collection time or
  during export.
- `classification` is a caller-supplied label. It is not a security control.
- Export redaction removes model names, local paths, file names, tool-call
  arguments, token ids, and prompt rendering, and rewrites any linked artifacts.
  See [failure-cases.md](failure-cases.md#redaction).
- The workbench does not encrypt, upload, or automatically expire traces. A
  workspace is a local directory; retention is managed by the user.

## Storage and verification

```text
.wsp/
├── artifacts/<prefix>/<artifact-id>.json          # immutable trace payload
│   └── <artifact-id>.json.meta.json               # ArtifactRef
└── traces/<trace_id>.json                         # index and checksum
```

`TraceStore.save` verifies token/character/byte budgets, writes the canonical
payload through `ArtifactStore`, then writes the index. The index record contains
the artifact id and checksum, model name/revision, backend, token count, text
length, termination reason, input checksum, and provenance origin.

`TraceStore.load` verifies the artifact checksum before parsing it. A reused
trace id with different content is rejected. Identical bytes are idempotent.

## Interchange

### WSP JSON

`export_trace(format="wsp")` writes the canonical trace object. `import_trace`
re-validates it and stores it as a new immutable artifact. This is the lossless
path.

### INIF 0.1

The adapter targets INIF format `0.1` and is dependency-free. It imports:

- metadata model name/revision/HuggingFace id and generation config,
- deduplicated sequences, expanded to real token records,
- token ids, texts, and conventional `logprob` extras,
- role-based text segments and assistant token ranges,
- token annotations, target/reference fields, scores, and sample metadata.

Preserved without silent loss:

- sample fields not mapped to first-class trace fields go to
  `metadata.inif_unmapped_sample_fields`;
- non-conventional token extras go to `metadata.inif_token_extras`;
- raw texts, spans, scores, references, and metadata remain under `metadata.inif_*`;
- the INIF version and adapter version are recorded in provenance.

Known mapping limits:

- INIF has no termination reason; imported traces use `unknown` and explain why.
- INIF does not identify a tokenizer; the field is unavailable.
- Indexed `.inif` archives are not supported; convert them to `.inif.json`
  first. Unsupported archives return a structured `unsupported` error.
- Export omits token arrays when any token id is missing, because INIF uses
  `id=None` to mean a sequence reference. The omission is reported in the
  document's export warnings; text and case evidence remain.
- INIF is an evolving format. The native `wsp.trace/1.0` schema remains
  canonical.

### Inspect AI logs

The adapter reads Inspect AI JSON logs directly and `.eval` archives through the
standard-library zip reader. It is dependency-free and does not import Inspect.

Mapped fields:

- `eval.model`, model args, generate config, packages, task/eval/run identifiers,
- sample input, target(s), messages, completion text, token usage, scores,
  timings, stop reason, and sample metadata,
- token text/log-probabilities when the log includes a logprob stream.

Known mapping limits:

- Inspect logs do not carry tokenizer ids or token ids in the portable payload;
  token-level annotations are unavailable. Text-level detectors still run.
- Inspect `stop_reason="stop"` is mapped to `eos` with an ambiguity warning.
- `max_tokens` and `model_length` map to `length_limit`; `tool_calls` maps to
  `tool_handoff`; execution errors map to `error`.
- Inspect 0.3.x `.eval` files use ZIP zstd compression. Python 3.14+ stdlib can
  read them; older runtimes should use `log_format="json"` or a converter.
  Unsupported runtimes return a structured explanation.

## Future activation references

`activation_captures` is a list of `ArtifactRef` records. V1.1 does not capture
activations during cached generation, but traces and cases can reference future
captures without a schema change. A future capture should preserve the same
artifact and checksum discipline and state which forward pass or decode step it
observed.

## Versioning policy

- Additive fields may appear in a compatible minor revision, but the current
  canonical version is explicitly `wsp.trace/1.0`.
- Removing or changing the meaning of a field requires a new schema version.
- Importers validate what they understand and preserve unknown metadata.
- The workbench never silently upgrades or rewrites a stored trace. A new
  transformation produces a new record with provenance.
