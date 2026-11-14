# WSP Workbench

An open-source, model-agnostic **Weight-Space Programming Workbench**: bounded
instruments for observing, inspecting, measuring, patching, executing, and
reproducing work on neural-network weights and behavior.

V1.1 is deliberately **tools first**. It is not an autonomous agent, a training
loop, a benchmark platform, or an interpreter for neurons. It provides reliable
primitives that a person or a future tool-using agent can compose, with the same
operations exposed through Python, a CLI, and JSON tool schemas.

> **Status:** V1.1 complete. CPU-only, offline tests and examples. No LLM, cloud
> service, API key, or network access is required.

## What V1 includes

- **Checkpoint inspection** — SafeTensors metadata, tensor enumeration, bounded
  reads, manifests, fingerprints, and checkpoint comparison without loading a
  whole model.
- **Tensor analysis** — statistics, quantiles/histograms, norms, cosine/dot/
  projection comparisons, matrix rank and singular-value summaries, all with
  explicit sampling labels.
- **Activation instrumentation** — bounded forward hooks, input/output capture,
  summary-only capture, token-position selection, comparison, and cleanup on
  error paths.
- **Controlled weight editing** — immutable, versioned patches; replace/add/
  scale/rank-one/projection operations; preview; disjoint patch sets; streaming
  export to a derived checkpoint; rollback by immutable source.
- **Experiments and artifacts** — content-addressed artifacts, run records with
  environment/replay metadata, manifests, patch storage, and experiment
  grouping.
- **Structured interfaces** — versioned JSON schemas and a shared service layer
  behind both the Python API and the `wsp` CLI.

## What V1.1 adds: the Behavioral Observatory

- **Generation traces** — a versioned `wsp.trace/1.0` record of tokens, text,
  log-probabilities (when available), exact inputs, settings, termination,
  timing, provenance, checksums, privacy state, and explicitly unavailable
  fields.
- **Collection** — a deterministic scripted backend and an optional local
  Hugging Face backend; imports for WSP JSON, INIF 0.1, and Inspect AI logs.
- **Detectors** — repetition, generation termination, structured output,
  reconsideration markers, and repeated tool actions, with sparse findings,
  evidence ranges, and declared limitations.
- **Contrastive tests** — explicitly authored baseline/variation pairs with a
  declared invariant, stored traces, and evaluator outcomes.
- **Evaluation** — deterministic, reference-answer, JSON-contract, and human
  judgement evaluators that distinguish pass, fail, inconclusive, abstention,
  and error.
- **Failure cases** — a local registry linking observations, controls, findings,
  contrasts, evaluations, annotations, hypotheses, activation captures, and
  patch references; self-contained export and checksum-verified reload.

## Install

Requirements: Python 3.11+ and PyTorch. CPU-only installation is the supported
baseline.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Optional local Hugging Face support:

```bash
pip install -e ".[hf]"
```

ROCm is not required for installation or tests. GPU execution is optional and
not promised to be bitwise reproducible across stacks.

## Quick start

The examples are fully offline and create their own tiny deterministic model.

```bash
python examples/make_tiny_model.py
python examples/inspect_and_patch.py
```

`examples/inspect_and_patch.py` performs the complete workflow:

1. opens the checkpoint without loading the model;
2. prints a manifest and tensor statistics;
3. captures `fc1` activations in a baseline forward pass;
4. creates and previews an `add 0.125` patch on `fc1.bias`;
5. applies it to a derived checkpoint;
6. reloads the derived checkpoint, verifies the expected parameter values, and
   compares outputs and activations;
7. reverts by selecting the untouched source and verifies identical output;
8. stores the manifest, patch, patch set, and run metadata in a workspace.

The source checkpoint hash is checked before and after. It never changes.

### Behavioral workflow (offline)

```bash
python examples/behavioral_workflow.py
```

The example records synthetic healthy and repetitive traces, detects a
repetition anomaly, compares the traces, runs a controlled comparison with an
explicit factual invariant, stores a case, exports it, reloads it in a fresh
workspace, and re-scans the restored trace. It is an infrastructure
demonstration, not a scientific result.

Minimal CLI version:

```bash
wsp init
wsp record-trace "What is the capital of Australia?" --output "Canberra"
wsp traces
wsp scan-trace tr_... --detector core.repetition
wsp detectors
```

### CLI walkthrough

```bash
# A local workspace (defaults to ./.wsp) for manifests, patches, and run records.
wsp init

wsp inspect-model examples/out/tiny_mlp.safetensors --file-sha256
wsp list-tensors examples/out/tiny_mlp.safetensors --contains fc
wsp inspect-tensor examples/out/tiny_mlp.safetensors fc1.bias --preview-elements 8
wsp call analyze_tensor \
  --json '{"path":"examples/out/tiny_mlp.safetensors","tensor":"fc1.weight","operations":["stats","norms"]}'
```

Create a patch using the checked-in input example, then preview and apply it:

```bash
wsp --workspace examples/out/workspace call create_patch \
  --input examples/create_patch.json

# Use the patch_id printed by the previous command.
wsp --workspace examples/out/workspace call preview_patch \
  --json '{"path":"examples/out/tiny_mlp.safetensors","patch_id":"p_..."}'

wsp --workspace examples/out/workspace call apply_patch \
  --json '{"path":"examples/out/tiny_mlp.safetensors","output_path":"examples/out/cli_patched.safetensors","patch_ids":["p_..."]}'
```

Every command prints JSON. A failed operation prints a JSON error and exits
with status 1. Use `wsp tools` for the operation registry and
`wsp tools --operation NAME` for the exact input schema.

### Python API

```python
from wsp.core.budget import Budget
from wsp.inspection.factory import open_checkpoint
from wsp.inspection.manifest import build_manifest
from wsp.tensors.stats import basic_stats

with open_checkpoint("examples/out/tiny_mlp.safetensors", compute_file_sha256=True) as source:
    manifest = build_manifest(source, include_tensors=True)
    info = source.tensor_info("fc1.bias")
    values = source.read_full("fc1.bias", max_elements=Budget().max_tensor_numel)

print(manifest.identity.structure_fingerprint)
print(info.shape, info.dtype)
print(basic_stats(values, storage_dtype=info.dtype))
```

The structured service is also available directly:

```python
from wsp.tools.service import ToolService

service = ToolService(raise_errors=True)
result = service.call("inspect_model", {"path": "examples/out/tiny_mlp.safetensors"})
```

## Safety properties

- Source checkpoints are never modified in place.
- Patch application writes a new checkpoint atomically and refuses to overwrite
  the source or an existing destination unless explicitly allowed.
- Checkpoints are data. Pickle-based loading is rejected; bundled code is never
  executed. A Python model adapter is **not** a sandbox.
- File paths are constrained to explicit destinations and workspaces.
- Measurements are separated from interpretations; sampled/approximate results
  are labeled and never presented as full results.
- CPU is the deterministic baseline. GPU nondeterminism is surfaced, not hidden.
- No telemetry, no automatic downloads, no required cloud services, no API keys.
- Generated data is excluded from Git by default.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — layers, data flow, storage
  formats, budgets, precision, trust boundaries.
- [`docs/behavioral-observatory.md`](docs/behavioral-observatory.md) — trace
  lifecycle, detectors, evaluators, contrasts, cases, and limits.
- [`docs/trace-format.md`](docs/trace-format.md) — the versioned trace schema,
  provenance, privacy, and INIF/Inspect interoperability.
- [`docs/failure-cases.md`](docs/failure-cases.md) — investigation cases,
  review states, hypotheses, and export/reload bundles.
- [`docs/prior-art-and-integration.md`](docs/prior-art-and-integration.md) —
  evaluated upstream projects, tested versions, and integration decisions.
- [`docs/tool-api.md`](docs/tool-api.md) — every structured operation, schemas,
  errors, limits, and examples.
- [`docs/patch-format.md`](docs/patch-format.md) — the versioned patch format,
  operations, compatibility checks, casting, export, and rollback.
- [`docs/limitations.md`](docs/limitations.md) — supported formats,
  architecture/execution assumptions, safety boundaries, and known gaps.
- [`docs/roadmap.md`](docs/roadmap.md) — completed V1 capabilities and
  deliberately deferred/future work.
- [`examples/README.md`](examples/README.md) — offline examples.

## Development

```bash
python -m pytest -q
```

The test suite is CPU-only and offline. It includes unit tests for core,
patches, activations/execution, workspace/tools, and CLI behavior, plus an
end-to-end integration test covering inspection, patching, export, reload,
comparison, rollback, provenance, and failure safety.

## Research context and attribution

The project was informed by the public Weight-Space Programming research and its
released experiment logs:

- <https://rikiyat.github.io/blog/weight-space-programming/>
- <https://github.com/RikiyaT/lm_understand-logs>

The workbench is an independent, general-purpose implementation of engineering
primitives suggested by that work (bounded model access, explicit tensor
metadata, reversible edits, measurement provenance, execution budgets). No
upstream code or experiment assets are bundled here. Upstream materials have
their own licenses, including mixed content in the logs repository; consult
those sources for their terms. See `docs/limitations.md` for the scope boundary.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
