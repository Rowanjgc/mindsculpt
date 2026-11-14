# Prior art and integration review

Status: V1.1 feasibility review, completed 2026-11-14.
Scope: projects named in the V1.1 handoff, plus direct evaluation of whether each can be
adopted without compromising the V1 constraints (offline, CPU-capable, local-first,
immutable artifacts, optional dependencies).

This document records what was inspected, what was actually executed, and what was
decided. It is deliberately conservative: a project is only described as "tested" when
code from it ran in an isolated environment.

## 1. Method

- Repository metadata (stars, license, last push) was read from the GitHub API and
  release/version metadata from PyPI on 2026-11-14.
- Source distributions/wheels were unpacked and read for API and schema details,
  without installing heavy dependency trees into the project environment.
- Two integrations were exercised end to end in throwaway environments:
  - Inspect AI 0.3.289 with its offline `mockllm/model` provider, writing both
    `.eval` and JSON logs.
  - NNsight 0.8.2 + nnterp 2.0.0rc1 on a locally constructed tiny Qwen3.5 hybrid
    model (Gated DeltaNet + full attention), CPU only.
- The project virtual environment was not modified for experiments. The Inspect
  experiment used `/tmp/venv-inspect` (Python 3.13). The nnterp experiment used
  `pip install --target /tmp/il3 --no-deps ...` against the project environment's
  Python 3.14.7/torch 2.12.0 installation.
- One install attempt (`nnsight` + `nnterp` into a fresh venv with system site
  packages) failed with `OSError: [Errno 122] Disk quota exceeded` because pip tried
  to download a CUDA PyTorch build. This is recorded as an environment limitation, not
  a project finding; the `--no-deps`/`--target` approach avoided it.

## 2. Versions evaluated

| Project | Upstream | Version / revision evaluated | License | PyPI `requires-python` |
|---|---|---|---|---|
| Inspect AI | `UKGovernmentBEIS/inspect_ai` | 0.3.289 | MIT | `>=3.11` |
| Inspect Scout | `meridianlabs-ai/inspect_scout` (renamed/moved from the AISI org) | 0.6.2 | MIT | `>=3.11` |
| LangTest | `PacificAI/langtest` (moved from `JohnSnowLabs`) | 2.8.0 | Apache-2.0 | `>=3.12,<3.15` |
| INIF | `ndif-team/inif` | package 0.0.1, format `0.1` | MIT | `>=3.10` |
| NNsight | `ndif-team/nnsight` | 0.8.2 | MIT | `>=3.10` |
| nnterp | `ndif-team/nnterp` | 1.3.0 (PyPI), 2.0.0rc1 (tested) | MIT | `>=3.10` |
| TransformerLens | `TransformerLensOrg/TransformerLens` | 4.4.0 | MIT | `>=3.10,<4.0` |
| pyvene | `stanfordnlp/pyvene` | 0.1.8 | Apache-2.0 | `>=3.9` |
| LM Evaluation Harness | `EleutherAI/lm-evaluation-harness` | repository HEAD, no PyPI release | MIT | n/a |
| EasyEdit | `zjunlp/EasyEdit` | repository HEAD | MIT | n/a |
| SAELens | `decoderesearch/SAELens` | 6.55.5 | MIT | `>=3.10,<4.0` |

All candidate licenses are permissive and compatible with Apache-2.0 redistribution.
No upstream code is copied into this repository. The adapters described below are
independent implementations of documented JSON shapes.

## 3. Environment baseline relevant to all decisions

- Development target: Fedora Linux, Ryzen 5 7600X, Radeon RX 7800 XT (16 GB), 32 GB RAM.
- Project venv: Python 3.14.7, torch 2.12.0+cu130, transformers 5.10.2 (optional extra),
  numpy 2.5.4, pydantic 2.12.5, typer 0.24.1.
- `torch.cuda.is_available()` is **False** and `torch.version.hip` is `None`: the
  installed PyTorch is a CUDA build on a machine whose GPU is AMD. ROCm is installed at
  the driver level, but GPU acceleration requires a user-managed ROCm PyTorch build.
  Therefore CPU-only is the tested path for V1.1, and no integration may assume CUDA.
- Disk pressure made large optional dependency trees impractical to install. This
  reinforces the decision to keep integrations optional and data-level where possible.

## 4. Inspect AI

### What it provides

- Task/dataset execution, solvers, scorers, model providers, display, and structured
  evaluation logs.
- Log formats: `.eval` (a zip archive) and `log_format="json"` (a single JSON file).
- Offline operation with the built-in `mockllm/model` provider.

### What was tested

A throwaway Python 3.13 venv installed `inspect-ai==0.3.289` (83 packages). An eval ran
with `mockllm/model` custom outputs and `match()` scoring, once with the default `.eval`
log and once with `log_format="json"`; both succeeded with no network access.

Findings that matter for integration:

1. `.eval` logs written by 0.3.289 use ZIP compression method 93 (`ZIP_ZSTANDARD`).
   Python 3.14's stdlib `zipfile` supports this; Python 3.11–3.13 stdlib does not.
   The third-party `zstandard` package provides a fallback. JSON logs avoid the issue
   entirely.
2. The log payload is ordinary JSON with a stable, documented-by-shape layout:
   `header.json` carries `eval` metadata (model, model_args, generate config, packages,
   task, dataset), and each sample carries `input`, `target`, `messages`, `output`,
   `scores`, timings, and token usage.
3. `sample.output.completion`, `sample.output.usage`, `sample.output.choices[*].stop_reason`,
   and `sample.scores` are sufficient to construct a behavioral trace without Inspect
   installed.
4. Inspect is a **runtime and evaluation framework**, not a trace format. Adopting it as
   our execution backend would make an 83-package dependency part of the core path and
   would not by itself give us local Hugging Face generation with token-level traces.

### Decision: ADAPT (data-level), do not adopt as a dependency

Implement a dependency-free Inspect log importer that reads JSON logs and, when the
runtime supports it, `.eval` archives. The importer is an optional adapter invoked
explicitly; core code never imports Inspect. Users who want Inspect's model backends
run Inspect themselves and import the resulting log.

Why: real interoperability is achievable at the artifact level; making Inspect a
runtime dependency would double the dependency surface and would not remove the need
for our own trace schema, detectors, and evidence store.

## 5. Inspect Scout

### What it provides

- Scanners over evaluation transcripts, producing sparse findings.
- `@scanner` decorator, `llm_scanner`, `grep_scanner`, and custom scanners.
- Results as per-scanner tables (`scan_results_df`) and a separate `scans/` store.

### Findings

- Scout depends on `inspect-ai`, `duckdb` (1.4–2.x), `libcst`, FastAPI, and more. Its
  transcript model is Inspect-shaped (`messages` + `events`), not token-trace-shaped.
- The `Result` model is a good vocabulary for findings: `value`, `answer`,
  `explanation`, `references`, `label`, `type`, plus identifiers. Sparse results are
  normal.
- Custom scanners are clean to implement, but only inside Scout's transcript model.

### Decision: REFERENCE + DEFER adapter

Adopt the finding vocabulary (a sparse result with a measured value, explanation,
evidence references, and optional label/type) in our native detector registry. Do not
add Scout or duckdb. A future adapter can convert a WSP trace into a Scout-compatible
transcript; V1.1 does not need it, and the conversion would lose token-level detail.

## 6. LangTest

### What it provides

Perturbation generators and robustness tests, including sycophancy, consistency, and
instruction-following categories.

### Findings

- Version 2.8.0 requires Python `>=3.12,<3.15` and depends unconditionally on
  `openai>=1.61`, `torch>=2.8`, pandas, and more; several test families assume hosted
  model access. This conflicts with the offline/no-hosted-provider constraint for core
  operation.
- Its perturbation ideas are useful, but its execution and dependency model are heavier
  than the value delivered in V1.1.

### Decision: REFERENCE, defer

Author perturbations explicitly in contrastive tests. Revisit selected generators only
if they can be used without the framework or hosted clients.

## 7. INIF

### What it provides

A JSON interchange format for tokenized generation traces:

- `metadata` with model info, package versions, sources, and creation time,
- `sequences` for deduplicated token patterns,
- `samples` with tokens (`token`, optional `id`, arbitrary extras), texts with role-based
  names and token ranges, annotations (token ranges), spans (position lists), scores,
  target/references/error fields,
- an optional indexed `.inif` zip-like archive and an HTML viewer.

The README states the API and format are in early development and may change without
deprecation. The PyPI package is 0.0.1 and hard-depends on `transformers`.

### Findings

- The format is a close match for the trace data we need and is explicitly intended to
  bridge evaluation traces and interpretability tools.
- It is not a stable foundation for canonical storage. It has no compatibility
  guarantees, a very small user base, and a hard `transformers` dependency.
- The plain JSON shape is simple enough to read/write without the package.

### Decision: ADAPT behind a versioned adapter; do NOT make it canonical

Canonical storage is `wsp.trace/1.0`. A dependency-free adapter imports/exports INIF
format `0.1` JSON documents, preserves unknown token extras and sample metadata, and
records the targeted INIF version. Unsupported INIF structures are preserved in trace
metadata and reported in warnings rather than dropped silently. Indexed `.inif`
archives are out of scope; users can convert them to JSON with INIF itself.

## 8. NNsight and nnterp

### What they provide

- NNsight: tracing and intervention over PyTorch models, including generation.
- nnterp: a normalized accessor layer over nnsight across ~98 model families, with
  explicit support for hybrid/recurrent architectures (Gated DeltaNet, Mamba, etc.).

### What was tested

A tiny local Qwen3.5-text hybrid model was constructed offline (4 layers: three
`linear_attention` Gated DeltaNet layers, one `full_attention`), with a synthetic
tokenizer. `StandardizedTransformer` loaded it and a trace captured layer 0 output,
layer 3 attention output, and logits (shapes `(1,4,32)`, `(1,4,32)`, `(1,4,64)`).

Important findings:

1. **nnterp 1.3.0 does not import with nnsight 0.8.2.** It imports
   `nnsight.intervention.tracing.globals`, but tracing moved to top-level
   `nnsight/tracing` in 0.8.x. Its metadata only requires `nnsight>=0.6`, so pip can
   resolve an incompatible pair. The tested working pair is nnsight 0.8.2 +
   nnterp 2.0.0rc1 (a pre-release).
2. nnterp exposes Qwen3.5 hybrid elements (`linear_attn` with `decays`, `betas`,
   `state`, `attention_head_outputs`, etc.), not generic `queries/keys/values`.
3. Accessor availability depends on the execution path. On the tiny model, the
   Gated DeltaNet kernel fell back to a pure-torch path, and requesting
   `linear_attn.attention_queries` raised `SourceNotAvailable` with a precise list of
   the ops that actually ran. This is good diagnostic behavior, but it proves that
   generic hooks do not map correctly to every architecture and that adapters must
   treat unavailable fields as unavailable.
4. Access order is significant in nnterp 2.0 (`OutOfOrderError` if a later module is
   requested before an earlier one), and source introspection requires a real file
   (not stdin) for the trace block.

### Decision: DEFER the dependency; document the tested path

Do not add nnsight/nnterp to core or default extras. V1.1 has no requirement for
activation-level investigation, and pinning a pre-release pair would make the core
fragile. The evidence format reserves activation-capture references, and the review
records the exact working versions and constraints for a future optional adapter.

### V1.2 outcome: implemented as an optional adapter

The A/B tests above were re-run during V1.2 and the adapter was implemented in
`wsp.investigation.nnsight_backend`:

- it lives behind the same capture protocol as the plain PyTorch hook backend, so
  the shared `MeasurementBuilder` produces comparable numbers from both;
- it is installed through the optional `instrumentation` extra
  (`nnsight==0.8.2`, `nnterp==2.0.0rc1`) and is never imported by the CPU-only
  core;
- it loads the checkpoint with `attn_implementation="eager"` because nnterp only
  exposes attention outputs under eager attention on a pre-built module;
- it records `backend: "nnterp"` plus both versions in every capture record, and
  refuses unsupported architectures instead of guessing at a module layout;
- it was validated against the tiny locally constructed Qwen3.5 hybrid
  checkpoint (Gated DeltaNet interleaved with full attention) and against a real
  local Qwen3.5-0.8B checkpoint.

Cross-backend agreement was measured on the synthetic hybrid fixture: the same
position of `model.layers.1` (hooks) and `layers.1` (nnterp) agree to within
`1e-6`. The generic hook backend remains the default and the only mandatory one.

## 9. Secondary references

| Project | Relevant capability | Decision |
|---|---|---|
| TransformerLens | Hooked transformer internals, activation cache | Reference only; transformer-specific and heavy |
| pyvene | Declarative interventions, activation patching | Reference only; future causal adapter candidate |
| LM Evaluation Harness | Broad task evaluation | Deferred; Inspect/import path covers V1.1 needs |
| EasyEdit | Knowledge/weight editing methods | Deferred; patch engine already covers bounded edits |
| SAELens | Sparse autoencoders/features | Deferred; out of V1.1 scope |

These are not installed, and no core interface depends on them.

## 10. Integration decisions summary

| Candidate | Decision | Reason |
|---|---|---|
| Inspect AI | Adopt log interoperability (importer), not a runtime dependency | Real data-level interop without 83-package coupling |
| Inspect Scout | Reference finding vocabulary; defer adapter | Transcript model mismatch and duckdb dependency |
| LangTest | Reference perturbation ideas; defer | Hosted-provider and Python-version constraints |
| INIF | Adapter for format 0.1; canonical schema stays native | Useful interchange, unstable upstream |
| NNsight/nnterp | Adopt as an optional instrumentation backend | Pre-release pairing is acceptable when pinned and optional; adds normalised layer access |
| TransformerLens/pyvene/EasyEdit/SAELens/lm-eval | Reference/defer | Overlap or out of scope |

## 11. Consequences for the V1.1 architecture

- Every external integration is an **adapter at the data boundary**, not a core
  dependency. Core operations must work with no optional packages installed.
- Trace identity, checksums, privacy controls, and bounded sizes remain ours; imported
  traces carry provenance and importer versions.
- Detector findings use our own versioned schema but borrow Scout's sparse-result
  discipline: a finding exists only when something was measured, and it carries the
  measurement, evidence, parameters, and limitations.
- Missing data is represented as absent or explicitly unavailable; importers never
  invent token IDs, log-probabilities, or timing values.
- External failures are surfaced as structured errors or warnings from the adapter and
  do not disable unrelated operations.

## 12. Limitations of this review

- Inspect was exercised only with `mockllm`, not with a real provider or a local model;
  token-level log-probabilities were therefore not verified end to end.
- The `.eval` importer was verified against Python 3.14's stdlib `zipfile`; on older
  Pythons it requires the optional `zstandard` package.
- No nnterp run used a real pretrained checkpoint, GPU, or ROCm. The tiny hybrid model
  validates wiring and failure modes, not numerical fidelity of large models.
- LangTest and Scout were reviewed from source/wheels, not executed.
- All version numbers are those observed on 2026-11-14 and should be re-checked before a
  future dependency is pinned.
