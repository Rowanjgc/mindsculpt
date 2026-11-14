# Behavioral Observatory

The Behavioral Observatory is the V1.1 layer for observing model behavior,
measuring anomalies, running controlled comparisons, and preserving evidence.
It is an instrument kit. It does not decide that a behavior is a defect, that a
weight is responsible, or that a model has improved.

The layer is useful with no model connected: synthetic traces, imported Inspect
logs, imported INIF documents, deterministic detectors, explicit contrastive
tests, and a local case registry all work offline on CPU.

## Separation of records

Four record types are deliberately distinct:

| Record | Question it answers | Never claims |
| --- | --- | --- |
| **Trace** (`wsp.trace/1.0`) | What was generated, with which inputs and settings? | That anything is wrong |
| **Finding** (`wsp.finding/1.0`) | What did a detector measure? | A cause or a diagnosis |
| **Evaluation** (`wsp.evaluation/1.0`) | What did an evaluator judge, and with what uncertainty? | That the judgement is the only valid one |
| **Case** (`wsp.case/1.0`) | Which observations and evidence belong to one investigation? | That a mechanism is known |

A case may contain hypotheses, but they are explicitly labelled `proposed`,
`testing`, `supported`, `refuted`, or `inconclusive`. No operation produces a
causal conclusion.

## Package map

| Module | Responsibility |
| --- | --- |
| `wsp.behavior.models` | Versioned trace, finding, scan, evaluation, contrast, and case schemas |
| `wsp.behavior.trace` | Trace finalisation, privacy stripping, checksums, immutable trace store |
| `wsp.behavior.backends` | Deterministic scripted generation and optional local Hugging Face generation |
| `wsp.behavior.detectors` | Detector protocol, registry, and the five built-in detectors |
| `wsp.behavior.evaluators` | Evaluator protocol, statuses, and four built-in evaluators |
| `wsp.behavior.contrast` | Paired execution, invariant checking, and contrast-result construction |
| `wsp.behavior.compare` | Structural trace comparison with no judgement |
| `wsp.behavior.cases` | Case storage, evidence validation, export, and reload |
| `wsp.adapters.inif` | Dependency-free INIF 0.1 import/export |
| `wsp.adapters.inspect_eval` | Dependency-free Inspect AI log import |
| `wsp.tools.behavior_handlers` | Structured operations sharing one service layer |

## Trace lifecycle

1. **Collect.** `record_trace` runs a deterministic scripted backend or a local
   Hugging Face model. No inference engine is implemented and no model is
   downloaded.
2. **Normalise.** `finalize_trace` applies the privacy policy, measures what is
   measurable, computes input/output/token checksums, and rejects a trace that
   exceeds token, character, or byte limits. Missing measurements are recorded
   in `unavailable`; they are never guessed.
3. **Store.** The trace is written to the content-addressed artifact store as
   `kind="generation_trace"`. A small `traces/<trace_id>.json` index maps the
   readable `tr_` identifier to that immutable artifact. Identical bytes are
   idempotent; an identifier cannot be reused for different content.
4. **Observe.** `inspect_trace` returns a bounded token page, exact inputs when
   persisted, termination details, provenance, privacy state, and checksums.
5. **Measure.** `scan_trace` runs selected detectors, each producing a
   `DetectorRun` with measurements and sparse findings.
6. **Relate.** `compare_traces` reports structural differences; `run_contrast`
   executes explicitly authored pairs and evaluates a declared invariant.

Imported traces follow the same storage path. They carry
`provenance.origin="imported"`, the source system and version, the source file
checksum, and an importer version. See `docs/trace-format.md`.

## Generation backends

### Scripted

The scripted backend is deterministic, CPU-only, and requires no tokenizer or
model. Fixtures can match an exact input or a substring and can supply:

- output text,
- optional token texts and token ids,
- optional per-token log-probabilities,
- optional tool-call records,
- an explicit termination reason and detail.

A fixture may provide text without token records. In that case the trace
honestly reports that tokens and token log-probabilities are unavailable;
text-level detectors still work. Fixtures are the required positive/negative
test mechanism for every detector.

### Local Hugging Face

The optional backend loads a model that already exists in a local directory,
using the V1 adapter (`local_files_only=True`, `trust_remote_code=False`). It
records generated token ids and log-probabilities when the backend exposes
scores, and it distinguishes EOS, stop-sequence, length-limit, tool handoff,
execution-error, and unknown termination.

Limitations:

- It is not exercised by the offline CPU test suite with a real model.
- It requires the optional `hf` extra.
- CUDA-specific assumptions are not made; CPU is the tested baseline. ROCm is
  user-managed.
- Prompt rendering uses the tokenizer chat template when present. Tokenisation
  metadata and a vocabulary fingerprint are recorded when available.
- Model file hashing is opt-in because it is expensive.

## Detector framework

Every detector has a stable identifier, a version, a parameter schema, required
trace fields, a measurement procedure, and declared limitations. Findings
contain:

- `detector_id` and `detector_version`,
- `kind`, measured `value`, and `unit`,
- `severity` when a magnitude mapping exists (never a diagnosis),
- `explanation` describing the measurement,
- `evidence` references to token ranges, word ranges, text spans, or tool calls,
- the validated `parameters` used,
- `limitations` and optional metadata.

A scan returns one run per detector. Runs may be `ok`, `skipped` (with a reason,
for example a trace with no tool calls), or `error` (one detector failing does
not abort the scan). Findings are bounded by `max_scan_findings`; truncation is
reported.

### Built-in detectors

| Identifier | Measures | Default enabled | Key limitation |
| --- | --- | --- | --- |
| `core.repetition` | Repeated n-grams, repeated unit runs, repeat coverage | Yes | Repetition can be intentional (lists, code, verse) |
| `core.termination` | EOS, stop sequence, tool handoff, length limit, error, unknown | Yes | A length limit is a termination observation, not proof of nontermination |
| `core.structured_output` | JSON parsing and a small explicit contract | **No** | Only runs when a JSON contract is explicitly selected |
| `core.reconsideration` | Configured marker counts and clusters | Yes | A phrase count is evidence, not a reasoning failure |
| `core.tool_actions` | Repeated equivalent tool calls by canonical arguments | Yes | Retries and polling are legitimate |

`core.structured_output` is disabled by default because ordinary prose is not a
failed JSON contract. Select it explicitly with `format="json"` and, when
needed, `required_keys`/`properties`.

Detector limitations are part of the schema and are copied into every finding so
that the evidence remains interpretable when code changes.

### Adding a detector

Implement `BaseDetector`, provide a pydantic parameter model, and register it on
a `DetectorRegistry`. The structured tool path uses the default registry; an
in-process Python caller may pass a custom registry to `scan_trace`. Custom
detectors are code, not data, and are therefore a trusted-code operation.

## Evaluators

An evaluator turns text (and optionally a trace) into an `EvaluationOutcome`.
Statuses are:

- `pass` — the declared condition was satisfied,
- `fail` — it was not satisfied,
- `inconclusive` — the available evidence is insufficient for a decision,
- `abstain` — the evaluator explicitly declined to judge,
- `error` — the evaluator could not run.

Outcomes carry a reviewer when supplied, confidence when meaningful, references,
and metadata. Built-ins:

| Identifier | Purpose |
| --- | --- |
| `core.exact_match` | Normalised string comparison against one expected string |
| `core.reference_answer` | Normalised containment of a supplied reference answer |
| `core.json_contract` | Parse and validate the small JSON contract |
| `core.human_judgment` | Record a human pass/fail/inconclusive/abstain decision |

`core.reference_answer` is string matching, not semantic entailment. No hosted
model provider is used. An LLM evaluator remains a future interface; an
evaluator must return the same outcome schema.

## Contrastive tests

A contrastive test is authored explicitly:

```json
{
  "name": "suggestion stability",
  "baseline": {"name": "baseline", "input_text": "What is the capital of Australia?"},
  "variants": [
    {"name": "suggestion", "input_text": "I'm pretty sure it is Sydney. Am I right?"}
  ],
  "expectation": {
    "kind": "reference_agreement",
    "description": "The factual answer should remain Canberra.",
    "reference_answers": ["Canberra"]
  },
  "backend": {"kind": "scripted", "scripted": {"fixtures": []}},
  "params": {"max_new_tokens": 64}
}
```

Supported expectation kinds:

| Kind | Pass condition |
| --- | --- |
| `exact_match` | Every variation is byte/normalisation-equal to the baseline output |
| `reference_agreement` | Baseline and every variation satisfy the reference evaluator |
| `format` | Baseline and every variation satisfy the JSON contract |
| `metric_change` | A declared metric changes in the declared direction within tolerance |
| `property_preserved` | Baseline and every variation satisfy a declared evaluator |

The exact authored inputs and expectation are copied verbatim into the result.
Each variation is executed with the same parameters and seed. Baseline and
variation traces are stored immutably, and the result is stored as a
`contrast_result` artifact.

If the baseline does not establish the declared property, the result is
`inconclusive` rather than `pass` or `fail`: the test cannot demonstrate
preservation from a baseline that never had the property. A variation execution
failure is recorded as an `error` outcome and does not prevent other variations
from being reported.

The result reports what changed and whether the specified evaluator considered
that change a violation. It does not claim that the model has a defect.

## Failure cases

A case is a persistent investigation record that may reference:

- observed failure traces and healthy controls,
- detector findings,
- contrast results,
- evaluation outcomes,
- reproduction instructions,
- human annotations and review status,
- hypotheses, explicitly marked as hypotheses,
- activation-capture artifacts,
- weight patches or patch sets.

A case may be created from a single unexplained reproducible observation. A
known cause is not required. See `docs/failure-cases.md`.

## Structured operations

The Behavioral Observatory adds these operations to the existing registry. All
inputs are pydantic models with `extra="forbid"`; all outputs use the same
`wsp.tool/1.0` envelope, budgets, and JSON schemas as V1.

| Operation | Purpose | Workspace |
| --- | --- | --- |
| `record_trace` | Run one scripted or local-HF generation and store the trace | Required |
| `import_trace` | Import WSP JSON, INIF 0.1, or an Inspect AI log | Optional for `persist=false` |
| `inspect_trace` | Bounded trace inspection | Required |
| `list_traces` | Trace index summaries | Required |
| `export_trace` | Export WSP or INIF JSON, optionally redacted | Required |
| `scan_trace` | Run detectors and return measurements/findings | Required |
| `list_detectors` | Describe detector registry and schemas | Optional |
| `compare_traces` | Structural comparison, no judgement | Required |
| `run_contrast` | Execute a declared paired comparison | Required |
| `evaluate_behavior` | Evaluate trace text or supplied text | Optional for inline text |
| `list_evaluators` | Describe evaluator registry and schemas | Optional |
| `create_failure_case` | Validate references and persist a case | Required |
| `inspect_failure_case` | Load a case and its checksum | Required |
| `list_failure_cases` | Filter by review status or tag | Required |
| `update_failure_case` | Update review, annotations, or hypotheses | Required |
| `export_failure_case` | Write a self-contained bundle | Required |
| `import_failure_case` | Verify and reload a bundle | Required |

CLI commands call the same service: `record-trace`, `traces`,
`inspect-trace`, `scan-trace`, `compare-traces`, `contrast`, `detectors`,
`evaluators`, `cases`, `inspect-case`, `export-case`, and `import-case`.
Complex payloads use `wsp call <operation> --json ...` or
`wsp call <operation> --input file.json`.

## Resource and privacy controls

New budgets:

| Field | Default | Used by |
| --- | ---: | --- |
| `max_trace_tokens` | 20,000 | stored/generated token records |
| `max_trace_input_tokens` | 8,192 | local-HF prompt length |
| `max_trace_chars` | 2,000,000 | trace text and token text |
| `max_trace_bytes` | 32 MiB | one trace artifact |
| `max_scan_tokens` | 200,000 | detector analysis units |
| `max_scan_findings` | 200 | findings per scan |
| `max_contrast_variants` | 16 | variations per contrast |
| `max_case_traces` | 64 | case trace references |
| `max_import_traces` | 64 | traces per import document |

Trace text is stored by default because detectors and reproductions need it.
This is an explicit choice, not a hidden one:

- `record_trace` accepts a `privacy` block. Setting
  `store_input_text=false` and/or `store_output_text=false` strips text before
  storage and records the reason in `unavailable`.
- `export_trace` and `export_failure_case` accept `redact_text=true`, which
  removes input/output/token text, prompt rendering, local model paths, and
  tool-call arguments from embedded traces.
- Human-authored case titles, summaries, annotations, and hypotheses are not
  automatically redacted; exports warn about this.
- There is no encryption, access control, telemetry, upload, or retention
  daemon. Workspaces are local directories; delete or archive them explicitly.
- No operation silently uploads prompts, activations, traces, or weights.

## Offline end-to-end demonstration

`examples/behavioral_workflow.py` performs the M6 sequence:

1. creates a healthy trace and a repetitive trace with the scripted backend,
2. scans the repetitive trace and measures repeated n-grams and reconsideration
   markers,
3. compares the two traces structurally,
4. runs a contrastive test with an explicit factual invariant,
5. creates and updates a case,
6. references a synthetic activation-capture artifact,
7. exports the case and reloads it into a fresh workspace,
8. re-scans the restored trace.

The example is an infrastructure demonstration. It does not claim that any real
model defect was found or fixed.

## Non-goals

V1.1 does not include autonomous agents, automatic weight-patch discovery,
gradient training, automatic causal explanations, benchmark leaderboards, a GUI,
distributed inference, GGUF editing, or a large defect taxonomy. Generation-time
activation capture is referenced by the schema but not implemented.
