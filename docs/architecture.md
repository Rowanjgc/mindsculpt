# Architecture

WSP Workbench is a small set of composable layers. Each layer has one job and
can be used without the layers above it. The Python API, structured tool
service, and CLI are three entry points onto the same implementation.

```text
CLI (wsp) ───────────────┐
Structured tool service ─┼─> workspace / run records
Python API ──────────────┘        │
                                   ├─ inspection ──> SafeTensors / torch module
                                   ├─ tensors
                                   ├─ activations ─> execution ─> PyTorch model
                                   └─ patches ─────> streaming SafeTensors export
```

There is no central “workbench” object that owns all behavior. The service is a
thin validation and dispatch layer; the numerical and file-format logic lives
in the domain modules.

## Package map

| Module | Responsibility |
| --- | --- |
| `wsp.core` | Errors, JSON/atomic writes, hashing, dtype handling, selections, budgets, shared result models |
| `wsp.inspection` | Header-only SafeTensors access, live `torch.nn.Module` metadata, manifests, checkpoint comparison |
| `wsp.tensors` | Preview, statistics, difference metrics, quantiles/histograms/norms/SVD/rank |
| `wsp.activations` | Bounded forward hooks, activation summaries, comparison, summary aggregation |
| `wsp.patches` | Immutable patch model, operations, validation, patch sets, streamed export |
| `wsp.execution` | Inference-mode forward passes, model loading, output comparison, offline demo models |
| `wsp.workspace` | Local content-addressed artifacts, run records, experiments, patch storage |
| `wsp.tools` | Pydantic tool schemas, operation registry, shared `ToolService` |
| `wsp.cli` | Typer commands; every command prints the service’s JSON envelope |
| `wsp.adapters` | Optional model-family adapters; V1 includes a local-only Hugging Face causal-LM adapter |

## Data flow

### Inspection

`open_checkpoint()` accepts a local `.safetensors` path and returns a
`CheckpointSource`. `SafeTensorsSource` reads the SafeTensors header and exposes
`TensorInfo` records (name, dtype, shape, element count, byte size, offset).
Tensor payloads are not read unless a tensor operation or bounded slice asks
for them. The manifest records a structural fingerprint over names, dtypes, and
shapes, so two copied checkpoints with different file names can still be
recognized as the same structure.

The fingerprint and the optional whole-file SHA-256 answer different questions:

- `structure_fingerprint`: do these checkpoints contain the same tensor layout?
- `file_sha256`: are these exact files byte-for-byte identical?
- `tensor_fingerprint`: are the decoded numerical values of one tensor identical?

Patch compatibility uses the structure fingerprint plus the target tensor’s
exact value fingerprint.

### Tensor analysis

Selection is resolved against the real tensor shape before any read. Invalid
axes, duplicate axes, out-of-range bounds, and non-positive steps fail before
I/O. Statistics are computed in float64 on the decoded values. If an operation
must sample, the result carries `sampled=True` and the stride; it is never
presented as a full result.

### Activation capture

`ActivationCapture` installs forward hooks on named modules (or the root module)
for the lifetime of one forward pass. Hooks are removed in `finally`, including
when the model raises. Captured tensors are detached, moved to CPU, widened from
BF16 to float32 for analysis, and reduced to a summary plus, when requested, a
deterministic bounded sample of retained values. Summary-only captures do not
consume the retained-value budget.

Module inputs, module outputs, hidden states, and attention tensors are distinct
kinds in the result. A residual-stream state is only reported as such when an
adapter explicitly labels it; the generic hook layer does not guess.

### Forward execution

`run_forward()` puts the model in evaluation mode under `torch.inference_mode()`,
optionally seeds torch/numpy, optionally installs a capture, and restores the
previous training mode. Outputs are summarized and may be sampled under the
output budget. `compare_forward()` compares flattened output arrays and
activation summaries from two executions.

Model-specific input preparation belongs in adapters. The generic executor
accepts tensors or arrays; the structured tool layer builds inputs for the
built-in offline models.

### Patching

Patch creation is a pure read operation on the source:

1. Resolve and validate the target tensor and selection.
2. Read the full target tensor (bounded by `max_tensor_numel`) and compute its
   exact fingerprint.
3. Resolve payloads, validate shapes and finiteness, and apply the operation in
   float64.
4. Cast the result back to the tensor’s storage dtype and report any precision
   effect.
5. Record before/after statistics, effect metrics, expected source identity,
   provenance, and a deterministic patch ID.

Preview repeats validation against the current source without mutating it. A
patch set is a collection of patches with disjoint targets. Application
materializes only targeted tensors, writes all untouched tensors by streaming
copies from the source, and atomically renames the completed output. The source
file is never opened for writing.

Rollback is not an undo log: the source checkpoint was never changed, so
reversion selects and verifies the immutable source identity.

## Storage formats

All schemas are versioned and JSON-compatible.

| Schema | Meaning |
| --- | --- |
| `wsp.identity/1.0` | Checkpoint identity and fingerprints |
| `wsp.manifest/1.0` | Model manifest with tensor metadata and totals |
| `wsp.patch/1.0` | One immutable weight patch |
| `wsp.patchset/1.0` | Compatible patches applied to one derived checkpoint |
| `wsp.preview/1.0` | Revalidated patch preview |
| `wsp.activation/1.0` | One activation summary |
| `wsp.activation-capture/1.0` | Serialized capture artifact payload |
| `wsp.activation-comparison/1.0` | Activation comparison |
| `wsp.forward/1.0` | Forward-pass result |
| `wsp.forward-comparison/1.0` | Forward comparison |
| `wsp.tool/1.0` | Structured tool response envelope |
| `wsp.artifact/1.0` | Content-addressed artifact record |
| `wsp.run/1.0` | Reproduction/run record |
| `wsp.workspace/1.0` | Workspace configuration |
| `wsp.experiment/1.0` | Named experiment grouping runs |

### Workspace layout

`wsp init` creates:

```text
.wsp/
├── workspace.json
├── artifacts/<2-hex-prefix>/<artifact-id>.<ext>
│   └── <artifact-id>.<ext>.meta.json
├── patches/p_<16-hex>.json
├── patchsets/ps_<16-hex>.json
├── runs/run_<timestamp>_<8-hex>/run.json
├── models/<fingerprint-prefix>/manifest.json
└── experiments/e_<16-hex>.json
```

Artifacts are addressed by SHA-256 of their bytes. The ID is stable for the
same content; a second write of the same bytes is idempotent. `ArtifactRef.path`
is relative to the workspace root. Reads validate the size and checksum before
returning data, and path joins are constrained to the workspace.

Generated data is ignored by Git through `.gitignore` (`.wsp/`, `*.safetensors`,
`examples/out/`).

## Resource budgets

`Budget` is a pydantic model with conservative defaults. Every entry point
accepts overrides. Unknown budget keys are rejected rather than ignored.

| Field | Default | Used by |
| --- | ---: | --- |
| `max_tensor_numel` | 20,000,000 | exact fingerprints, patch targets, model loading |
| `max_slice_elements` | 5,000,000 | slice reads and comparisons |
| `max_preview_elements` | 256 | tensor previews |
| `max_analysis_elements` | 20,000,000 | materialized analysis input |
| `max_sample_elements` | 200,000 | deterministic samples |
| `max_svd_elements` | 4,000,000 | SVD/rank |
| `max_histogram_bins` | 256 | histograms |
| `max_manifest_tensors` | 10,000 | manifests and tensor listings |
| `max_compare_tensors` | 2,000 | checkpoint comparisons |
| `max_activation_elements_per_tensor` | 500,000 | retained activation values |
| `max_activation_elements_total` | 2,000,000 | retained activation values |
| `max_forward_output_elements` | 5,000,000 | retained output values |
| `max_model_load_elements` | 50,000,000 | loading a checkpoint into a module |
| `max_patch_payload_elements` | 1,000,000 | patch payloads |
| `max_patchset_tensors` | 256 | patches per set |
| `max_output_bytes` | 2,000,000 | serialized tool envelopes |
| `max_artifact_bytes` | 268,435,456 | one stored artifact |

Limits fail explicitly with `resource_limit`; they do not silently degrade. The
one exception is mathematically appropriate sampling, which is always labeled
in the result.

## Determinism and precision

- Checkpoint structure and tensor fingerprints are exact decoded-value hashes.
- Statistics and patch arithmetic use float64; storage dtypes are preserved.
- BF16 values are losslessly widened to float32 for numerical analysis.
- Patch results are cast back to storage dtype. Any cast that changes values is
  reported in `precision.cast_changed_values`.
- IDs for patches and patch sets are deterministic functions of their content,
  not timestamps. Timestamps describe creation, not identity.
- CPU execution is the supported deterministic baseline. GPU results may differ
  across drivers and software stacks; do not assume bitwise reproducibility.
- Run records store operation parameters, source/model identity, device, dtype,
  seed, budget, dependency versions, artifacts, and warnings. They do not
  promise bitwise replay on different hardware.

## Trust boundaries

A SafeTensors file is data, not executable code. V1 never loads pickle
checkpoints and never executes Python bundled with a model.

A Python adapter runs in the user’s interpreter and is **not a sandbox**. Loading
a user-supplied model factory or module is a trusted-code operation. The local
Hugging Face adapter refuses remote downloads by default and sets
`trust_remote_code=False`.

## Design constraints

- Keep checkpoint inspection separate from model loading.
- Keep patch representation separate from patch execution.
- Keep execution separate from experiment bookkeeping.
- Keep tool schemas independent of any LLM provider.
- Do not add abstraction layers without an immediate use.
- Do not infer concepts from measurements; report measurements only.
