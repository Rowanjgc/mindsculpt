# Structured tool API

Every structured operation is implemented once in `wsp.tools.service.ToolService`.
The Python API and the `wsp` CLI both call that service. The schemas are
provider-independent; they do not mention an LLM, a prompt format, or a remote
service.

## Envelope

Successful response:

```json
{
  "schema_version": "wsp.tool/1.0",
  "ok": true,
  "operation": "inspect_model",
  "data": { "...": "operation-specific" },
  "run_id": "run_20261114T192022Z_ab12cd34",
  "artifacts": [],
  "warnings": []
}
```

Failed response:

```json
{
  "schema_version": "wsp.tool/1.0",
  "ok": false,
  "operation": "inspect_model",
  "data": null,
  "error": {
    "code": "invalid_input",
    "message": "only .safetensors checkpoints are supported",
    "details": { "path": "model.ckpt" }
  },
  "warnings": []
}
```

`run_id` is present only when a workspace is active. `artifacts` contains
`ArtifactRef` records created by the operation. `warnings` never contain hidden
fallbacks; they identify sampling, truncation, precision effects, or missing
optional output.

### Error codes

| Code | Meaning |
| --- | --- |
| `invalid_input` | Schema validation failed, an index/dtype/shape was invalid, or a required precondition was not met |
| `not_found` | A path, tensor, patch, run, or artifact does not exist |
| `unsupported` | The request is outside V1 scope (pickle loading, float8 edits, GGUF, etc.) |
| `resource_limit` | A configured element/byte/patch-count limit would be exceeded |
| `incompatible` | Checkpoint structure, tensor fingerprint, shape, or dtype does not match the patch/run |
| `safety_violation` | The request would overwrite a source, escape a workspace path, or initialize a workspace over unrelated files |
| `artifact_error` | Stored data is corrupt, ambiguous, or has an invalid format |
| `execution_error` | A model forward pass failed |
| `internal_error` | An unexpected implementation error |

Python callers can raise typed `WspError` exceptions instead of parsing the
envelope:

```python
from wsp.core.budget import Budget
from wsp.tools.service import ToolService

service = ToolService(raise_errors=True)
response = service.call("inspect_model", {"path": "model.safetensors"})
```

## Calling from Python

```python
from wsp.core.budget import Budget
from wsp.tools.service import ToolService

service = ToolService(budget=Budget(max_preview_elements=32))
print(service.operations())
print(service.describe("inspect_tensor"))
result = service.call("inspect_model", {"path": "examples/out/tiny_mlp.safetensors"})
```

`service.call()` returns the envelope as a plain JSON-compatible dictionary.
Pass a pydantic input model instead of a dict if preferred. `raise_errors=True`
raises `WspError` for the same validation and domain failures.

## Calling from the CLI

```bash
wsp tools --operation analyze_tensor
wsp inspect-model model.safetensors
wsp list-tensors model.safetensors --contains fc
wsp inspect-tensor model.safetensors fc1.bias --preview-elements 8
wsp call analyze_tensor --json '{"path":"model.safetensors","tensor":"fc1.weight","operations":["stats","norms"]}'
wsp call create_patch --input examples/create_patch.json
```

`wsp call` accepts inline JSON (`--json/-j`), a JSON file (`--input/-i`), and
repeatable dotted parameters (`--param/-p key=value`). All successful and
failed responses are JSON. A failed operation exits with status 1.

Global options before the command:

- `--workspace/-w PATH`: workspace root.
- `--budget JSON`: budget overrides, for example
  `--budget '{"max_preview_elements":8}'`.

Without `--workspace`, the service uses `$WSP_WORKSPACE` or `./.wsp` only when
that directory already contains `workspace.json`. Commands that require a
workspace return `invalid_input` otherwise.

## Behavioral Observatory operations

V1.1 added operations for traces, detectors, contrasts, evaluations, and cases.
They use the same envelope, error codes, budget overrides, and run records as V1.
Full schemas are available from `wsp tools --operation <name>`.

| Operation | Purpose |
| --- | --- |
| `record_trace` | Run one scripted or local-HF generation and store the trace |
| `import_trace` | Import WSP JSON, INIF 0.1, or an Inspect AI log |
| `inspect_trace` | Bounded token page and explicit provenance/privacy/unavailable fields |
| `list_traces` | Trace index summaries |
| `export_trace` | Export WSP or INIF JSON under an explicit privacy policy |
| `scan_trace` | Run deterministic detectors over one trace |
| `list_detectors` | Detector registry, parameter schemas, limitations |
| `compare_traces` | Structural comparison without judgement |
| `run_contrast` | Execute a paired comparison with a declared invariant |
| `evaluate_behavior` | Evaluate trace text or supplied text |
| `list_evaluators` | Evaluator registry and parameter schemas |
| `create_failure_case` | Validate references and persist a case |
| `inspect_failure_case` | Load a case and checksum |
| `list_failure_cases` | Filter cases by review status or tag |
| `update_failure_case` | Update review state, annotations, hypotheses |
| `export_failure_case` | Write a self-contained case bundle under an explicit privacy policy |
| `import_failure_case` | Verify and reload a bundle |

A scripted trace can be recorded directly from the CLI:

```bash
wsp init
wsp record-trace "What is the capital of Australia?" \
  --output "Canberra" --model scripted/fixture
wsp scan-trace tr_0123456789abcdef --detector core.repetition
wsp compare-traces tr_0123456789abcdef tr_fedcba9876543210
wsp detectors
wsp cases
```

Contrasts and full case payloads are multi-field inputs; use `wsp contrast
--input contrast.json` or `wsp call run_contrast --json '...'`. The service
validates the exact same schema either way.

`record_trace`, `scan_trace`, `run_contrast`, and all case operations require a
workspace. `import_trace` can validate without persisting with
`{"persist": false}`. `evaluate_behavior` can evaluate an inline string without
a workspace, but persisting the resulting artifact requires one.

Inputs that are complex JSON documents use the behavior models in
`wsp.behavior.models` and `wsp.tools.behavior_schemas`. Every operation rejects
unknown fields.

## Replay operations

Replay resolves a recorded trace into an executable plan, then executes it. The
two operations are separate on purpose: a plan is auditable before anything runs,
and a plan artifact can be verified later without the trace.

| Operation | Purpose |
| --- | --- |
| `prepare_replay` | Resolve the recorded token sequence and audit replayability |
| `verify_replay` | Execute the teacher-forced forward pass and compare numbers |

```bash
wsp replay-plan tr_1f2e... --max-sequence-length 512
wsp replay --plan a_9c31... --model '{"kind":"local_hf","path":"./ckpt"}'
wsp replay tr_1f2e... --model '{"kind":"synthetic"}'
```

`prepare_replay` accepts `{trace_id, execution, tolerances, save, notes}`.
`verify_replay` accepts `{trace_id | plan_artifact_id, model, execution,
tolerances, include_per_token, save}` — exactly one of the first two.

## Alignment operations

| Operation | Purpose |
| --- | --- |
| `align_trace_tokens` | Map recorded tokens onto prompt/generation/full indices |
| `resolve_finding_span` | Map detector evidence or a text span onto token positions |

```bash
wsp align tr_1f2e... --model '{"kind":"local_hf","path":"./ckpt"}'
wsp resolve-span tr_1f2e... --kind token_range --role generation --start 0 --end 4
wsp resolve-span tr_1f2e... --kind finding_evidence --detail '{"kind":"token_range","start":0,"end":4}'
```

`align_trace_tokens` accepts `{trace_id, model?, special_token_ids?,
include_positions, position_offset, position_limit, save}`. `model` is optional;
without it the alignment cannot supply tokenizer offsets or a special-token set,
and says so. `resolve_finding_span` accepts `{trace_id, alignment_artifact_id?,
model?, kind, start, end, role, name, detail, window, save}`.

## Activation investigation operations

| Operation | Purpose |
| --- | --- |
| `list_capture_points` | Addressable modules with architectural roles |
| `capture_trace_activations` | Replay a trace and measure selected states |
| `inspect_activation_capture` | Read a stored capture and its integrity |
| `compare_activation_captures` | Numerically compare two captures |
| `attach_investigation_evidence` | Record measurements as evidence, optionally on a case |
| `inspect_investigation_evidence` | Read evidence and re-verify its references |

```bash
wsp capture-points --model '{"kind":"local_hf","path":"./ckpt"}' --kind block
wsp capture tr_1f2e... --module blocks.1 --position 3 --mode selected --values
wsp capture tr_1f2e... --module layers.11 --span a_abcd... --semantics predicting \
    --instrumentation nnterp --model '{"kind":"local_hf","path":"./ckpt"}'
wsp compare-captures a_1111... a_2222... --metric l2 --metric cosine --module blocks.1
wsp attach-evidence --title "repetition finding" --case case_1a2b... \
    --observation tr_1f2e... --capture a_1111... --comparison a_3333...
wsp inspect-evidence a_4444...
```

`capture_trace_activations` takes exactly one position source: `positions`,
`span_artifact_id`, or `role`/`role_start`/`role_end`. It also takes `mode`
(`summary` or `selected`), `position_semantics` (`consuming` or `predicting`),
`modules`, `instrumentation` (`torch_hooks` or `nnterp`), `include_inputs`,
`first_call_only`, `execution`, and `tolerances`.

`compare_activation_captures` takes two capture artifacts, optional `metrics`,
`modules`, `direction` (required for `projection`), an `alignment` object, and
`allow_different_models` (default `false`). Explicit alignments must set
`reviewed: true`.

## Causal intervention operations

| Operation | Purpose |
| --- | --- |
| `create_activation_intervention` | Build and validate a versioned intervention spec |
| `validate_activation_intervention` | Check compatibility without running the intervention |
| `preview_activation_intervention` | Human-readable scope, shape, and cost estimate |
| `run_activation_intervention` | Execute one condition and record a result |
| `run_paired_intervention` | Baseline, identity control, intervention (+ optional control) |
| `compare_intervention_results` | Re-read a stored paired comparison |
| `run_counterfactual_continuation` | Bounded greedy continuation comparison |
| `inspect_counterfactual_comparison` | Read a paired comparison or continuation record |
| `attach_intervention_evidence` | Attach interventions and continuations to a case |

```bash
wsp create-activation-intervention --name "zero block 1 token 3" \
    --capture a_1111... --module blocks.1 --position 3 --operation zero
wsp validate-activation-intervention a_2222... --model '{"kind":"local_hf","path":"./ckpt"}'
wsp preview-activation-intervention a_2222... --model '{"kind":"local_hf","path":"./ckpt"}'
wsp run-paired-intervention a_2222... --model '{"kind":"local_hf","path":"./ckpt"}' \
    --trace tr_1f2e... --measure-module blocks.2 --logit-position 3
wsp run-counterfactual-continuation a_2222... --model '{"kind":"local_hf","path":"./ckpt"}' \
    --token-id 1 --token-id 42 --token-id 7 --max-new-tokens 4
wsp attach-intervention-evidence --title "ablation effect" --case case_1a2b... \
    --intervention a_2222... --comparison a_3333... --continuation a_5555...
```

Creating from a `trace_capture` derives the model reference, backend, device,
dtype, sequence length, replay/alignment ids, and the target module from the
capture; callers still supply positions, role, semantics, and operation.
`mean` accepts `mean_values` or `mean_artifact_id`, or `mean_source_capture_artifact_id`
plus `mean_positions` and `mean_token_alignment`. `patch` accepts `donor_values`,
`donor_values_artifact_id`, or `donor_capture_artifact_id` plus `donor_positions`.

Paired runs require an execution `source`, and `token_ids` or a `trace_id`.
`measure_modules`, `measure_positions`, `measure_tensor_names`, `logit_positions`,
`alternative_intervention`, `direction`, `identity`, `identity_atol`, and
`identity_rtol` are optional. Continuations add `max_new_tokens`,
`max_sequence_length`, `eos_token_id`, `seed`, and `stop_on_eos`; the spec is
retargeted to `uncached-continuation` when it was created for teacher-forced
replay.

All nine operations are also available through `ToolService.call`, with the
same inputs and the same `wsp.tool/1.0` envelope. Failures are explicit:
incompatible models, shapes, dtypes, capture points, sequence lengths, budgets,
or an unreadable replacement raise a typed error rather than falling back to a
different operation.

## Privacy in exports

`export_trace` and `export_failure_case` take an explicit policy:

```json
{
  "mode": "redacted",
  "sensitive_markers": ["CANARY-1234"],
  "keep_token_ids": false,
  "on_untransformable_artifact": "reject"
}
```

`mode: "private"` (the default) keeps every artifact byte-identical.
`mode: "redacted"` removes text at every serialization depth, rewrites artifact
references and ids, drops retained activation vectors, and refuses the export if
it cannot prove the result is clean. See `docs/failure-cases.md`.

## Common input types

### Selection

```json
{"slices": [{"axis": 0, "start": 0, "stop": 4, "step": 1}]}
```

- An empty `slices` list (or omitted selection) means the full tensor.
- Axes not mentioned are selected in full.
- Indices are non-negative and half-open. Negative indices and negative steps
  are rejected.
- Duplicate axes, out-of-range axes, `start > stop`, and out-of-bounds stops
  are rejected before reading data.

### Built-in model specification

```json
{
  "kind": "mlp",
  "seed": 0,
  "input_dim": 8,
  "hidden_dim": 16,
  "output_dim": 4,
  "vocab_size": 32,
  "model_dim": 16,
  "num_heads": 4,
  "num_layers": 2,
  "max_seq_len": 16,
  "checkpoint_path": null
}
```

`kind` is `mlp`, `linear`, or `transformer`. This exists so the tool API and CLI
can run deterministic forward passes fully offline. For `transformer`,
`model_dim` must be divisible by `num_heads`. The `mlp` model takes `x`; the
`linear` model takes `x`; the `transformer` takes `input_ids` (2-D integer
tokens).

### Capture configuration

```json
{
  "module_names": ["<root>"],
  "record_inputs": false,
  "token_position": null,
  "include_values": true,
  "max_elements_per_tensor": null,
  "max_elements_total": null
}
```

`<root>` means the top-level module. If `include_values` is false, only summary
statistics are retained and the per-tensor/total value budgets are not charged.
`token_position` accepts `"last"` or an integer. For a 2-D tensor the sequence
axis is 0; for a 3-D tensor it is axis 1. Retained values are stored flat and
the summary reports that flat shape.

## Operations

The authoritative JSON Schema for each input is available from
`wsp tools --operation NAME` or `ToolService.describe(NAME)`. All input models
forbid unknown fields.

### `inspect_model`

Input: `path`, `include_tensors` (default true), `compute_file_sha256` (default
false), `save_manifest` (default false).

Output `data`: `manifest` (`wsp.manifest/1.0`) and, when saved,
`saved_manifest_path`. The manifest contains identity, source, tensor metadata,
totals, metadata, and truncation state.

Limits: `max_manifest_tensors`, `max_output_bytes`. The whole-file hash is
optional because it reads the entire file.

Errors: `not_found`, `invalid_input` for non-SafeTensors paths, `unsupported`
for pickle-based formats, `safety_violation` on unsafe paths.

### `list_tensors`

Input: `path`, `name_contains`, `offset` (default 0), `limit` (default 100).

Output `data`: `identity`, `total`, `offset`, `returned`, `truncated`, and
`tensors` (name, dtype, shape, numel, nbytes, offset).

Limits: `limit <= max_manifest_tensors`; `max_output_bytes`.

### `inspect_tensor`

Input: `path`, `tensor`, `preview_elements` (default 32), `include_fingerprint`
(default false).

Output `data`: `identity`, `tensor`, `resolved`, `stats`, `preview`, `sampled`,
`fingerprint` (or null). If the full tensor exceeds `max_analysis_elements`, a
deterministic sample bounded by `max_sample_elements` is used and
`sampled=true`. The fingerprint is only computed when the full tensor fits
`max_tensor_numel`; otherwise a warning is emitted and it is null.

Limits: `max_analysis_elements`, `max_sample_elements`, `max_preview_elements`,
`max_tensor_numel`, `max_output_bytes`.

### `read_tensor_slice`

Input: `path`, `tensor`, `selection`, `preview_elements` (default 256).

Output `data`: `identity`, `tensor`, `resolved`, `stats`, `preview`.

Limits: selected region must fit `max_slice_elements`; preview must fit
`max_preview_elements`. Missing tensors and invalid selections are
`not_found`/`invalid_input`.

### `compare_tensors`

Input: `left_path`, `right_path`, `tensor`, optional `right_tensor`, `selection`,
`atol` (1e-6), `rtol` (1e-5).

Output `data`: `left_identity`, `right_identity`, `left_tensor`,
`right_tensor`, `left_fingerprint`, `right_fingerprint`, and `diff` with
`max_abs`, `mean_abs`, `rms`, `relative_frobenius`, `cosine`, `allclose`,
non-finite counts, and whether the comparison was sampled.

Limits: each selected region and fingerprint fits `max_slice_elements`.

### `analyze_tensor`

Input: `path`, `tensor`, `selection`, `operations` (default `["stats"]`),
`quantiles`, `bins` (default 32), `top_k` (default 10).

Operations: `stats`, `quantiles`, `histogram`, `norms`, `svd`, `rank`.
`stats`, `quantiles`, and `histogram` are always included. Quantiles default to
`[0, .01, .05, .25, .5, .75, .95, .99, 1]`. Unknown operation names are
`invalid_input`.

Output `data`: `identity`, `tensor`, `resolved`, `sampled`, and `analysis`
(`stats`, `operations`, `requested`, `sampled_operations`, `notes`).

Limits: materialized input `max_analysis_elements`; deterministic sample
`max_sample_elements`; SVD/rank require a full finite 2-D tensor within
`max_svd_elements`; `bins <= max_histogram_bins`. SVD/rank on sampled input
raise `resource_limit` rather than returning a misleading approximation.

### `compare_checkpoints`

Input: `left_path`, `right_path`, optional `name_contains`, `max_tensors`,
`atol`, `rtol`.

Output `data.comparison` with left/right identity, compatibility, missing and
extra tensor names, per-tensor statuses (`equivalent`, `different`,
`sampled_different`, `dtype_mismatch`, `shape_mismatch`, `error`), and counts.
Large tensors are compared with deterministic samples; such entries are labeled.
`identical` is true only when every shared tensor was compared in full and
matched.

Limits: `max_compare_tensors` and `max_slice_elements` per tensor.

### `create_patch`

Input: `path`, `target`, `selection`, `params`, optional
`expected_fingerprint`, `notes`, `save` (default true),
`externalize_payloads` (default true).

`params` is a discriminated union:

```json
{"kind": "replace", "values": {"values": [0.0], "shape": [1], "dtype": "float64"}}
{"kind": "add", "delta": {"values": [0.125], "shape": [1], "dtype": "float64"}}
{"kind": "scale", "factor": 0.5}
{"kind": "rank_one", "u": {"values": [1.0], "shape": [1]}, "v": {"values": [1.0], "shape": [1]}, "scale": 1.0}
{"kind": "projection", "direction": {"values": [1.0], "shape": [1]}, "axis": 1, "mode": "remove", "strength": 1.0}
```

A payload is exactly one of inline `values` or a workspace `artifact_id`, plus
optional shape/dtype/numel/sha256. Large inline payloads (more than 4,096
elements or 64 KiB) are externalized to a `patch_payload` artifact when a
workspace is active.

Output `data`: `patch` (`wsp.patch/1.0`) and, if saved,
`saved_patch_path`. The patch contains the expected source structure, target
tensor identity, selection, operation parameters, before/after statistics,
effect metrics, precision report, and provenance.

Limits: selected region <= `max_tensor_numel` and payload <=
`max_patch_payload_elements`. The full target tensor must also fit
`max_tensor_numel` so an exact fingerprint can be recorded.

Errors: `not_found` for the tensor, `invalid_input` for bad shapes, non-finite
payloads, zero projection direction, invalid indices, or arithmetic on
non-floating tensors; `incompatible` if `expected_fingerprint` does not match.

### `preview_patch`

Input: `path`, and exactly one of `patch_id` (workspace) or an inline `patch`
object.

Output `data`: `patch_id` and `preview` (`wsp.preview/1.0`) with compatibility,
checks, before/after summaries, effect, precision, and warnings. Preview
revalidates structure, target dtype/shape, and the target’s exact fingerprint,
then recomputes the operation without writing anything.

Errors: any compatibility failure is returned as `incompatible`; a source file
hash change with matching structure/fingerprint is a warning, not a hard
failure.

### `apply_patch`

Input: `path`, `output_path`, exactly one of `patch_ids` (workspace) or
`patches`, `overwrite` (default false), `save_patchset` (default true).

Output `data.patchset`: compact patch-set record with ID, status, source,
patch IDs, output path, and output identity. `saved_patchset_path` is included
when persisted.

Limits: `max_patchset_tensors`; each target must fit `max_tensor_numel`.
All patches must share the source structure and have disjoint targets.
Application refuses `output_path == path`, refuses an existing output unless
`overwrite`, and writes the derived checkpoint atomically.

### `list_patches`

Input: optional `source_fingerprint`, `limit` (default 100). Requires a
workspace. Output: `total`, `returned`, `truncated`, `patches` (compact patch
records). Pagination is not implemented; the limit is a bound, not a cursor.

### `revert_patch`

Input: `path`, `patchset_id`. Requires a workspace. Output: `patchset_id`,
`source` identity, `restored_by_source: true`, an explanatory note, and the
discarded output path.

Reversion verifies that `path` is the patch set’s source and returns its
identity. It does not delete the derived checkpoint and does not need to,
because the source was never modified.

### `export_checkpoint`

Input: `path`, `output_path`, `overwrite` (default false). Output:
`output_path`, `identity`. Copies the checkpoint atomically; refuses
source/destination equality and existing outputs unless overwrite is true.

### `list_artifacts`

Input: optional `kind`, `limit` (default 100). Requires a workspace. Output:
`total`, `returned`, `truncated`, `artifacts` (`ArtifactRef` records with
workspace-relative `path`, format, SHA-256, size, and metadata).

### `workspace_status`

Input: none (budget only). Requires a workspace. Output: workspace status with
config, root, and counts of runs, patches, patch sets, manifests, artifacts, and
experiments.

## Execution operations

These use built-in deterministic models so the tool layer stays fully offline.
They are not a general model-serving interface.

### `run_forward`

Input: `model` (built-in model spec), `inputs` (object of arrays), `input_args`
(array), `device`, `dtype`, `seed`, optional `capture`, `include_output_values`
(default false), `output_preview_elements` (default 256), `save_artifacts`
(default true).

Output `data`: `forward` (`wsp.forward/1.0`), `output_artifacts` (key to
artifact ID), and `capture_artifact` (artifact ID or null). Saved output arrays
are flattened.

The `forward` record includes model class, parameter count, device, dtype,
seed, inference-mode/training-restoration flags, input/output specs, activation
summaries, and warnings. GPU nondeterminism is warned about; it is not hidden.

Limits: `max_forward_output_elements`, `max_activation_elements_per_tensor`,
`max_activation_elements_total`, `max_artifact_bytes`, and
`max_model_load_elements` when `checkpoint_path` is supplied.

Python note: the low-level `run_forward()` accepts tensors/arrays, not raw
Python lists. The structured tool layer prepares built-in inputs for you.

### `capture_activations`

Input: same as `run_forward`, but `capture` is required and `save_artifact`
(default true) replaces `save_artifacts`.

Output `data`: `forward`, `capture_id`, `summaries` (`ActivationSummary`
records, values included), and `capture_artifact`. Summary records identify
module, kind (`input`/`output`), call index, original dtype, device, shape,
statistics, retention/sampling state, token position, and notes.

### `compare_activations`

Input: `left_capture_artifact`, `right_capture_artifact`, `atol`, `rtol`.
Requires a workspace. Output `data.comparison` with per-key entries, shape/dtype
flags, retained-value diff when values were retained, stats-only comparison
when they were not, missing keys, and identical/different counts.

### `compare_forward`

Input: `left_run_id`, `right_run_id` (workspace run IDs such as
`run_20261114T192022Z_ab12cd34`, not internal `fw_...` IDs), `atol`, `rtol`,
`compare_activations` (default true). Requires a workspace.

Output `data.comparison` (`wsp.forward-comparison/1.0`) with `outputs_identical`,
per-output diff entries, missing keys, and an optional activation comparison.
If the runs contain capture artifacts, the comparison is attached.

## Output size and resource behavior

Every successful and failed envelope is checked against `max_output_bytes`.
Requests that would materialize too much data either fail with `resource_limit`
or use a deterministic sample that is explicitly labeled. Nothing silently
falls back to a different algorithm.

The tool service never downloads models, never loads pickle checkpoints, never
executes code from a checkpoint, and never writes to a source checkpoint.
