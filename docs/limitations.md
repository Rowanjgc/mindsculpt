# Limitations and safety boundaries

V1.2 is an instrument kit, not a complete research harness. The following limits
are intentional and should be visible in results rather than worked around
silently.

## Checkpoint support

- **SafeTensors only** for automatic checkpoint loading and inspection.
  `.safetensors` is the only extension accepted by `open_checkpoint()` and
  `export_checkpoint()`.
- Pickle-based formats (`.pt`, `.pth`, `.bin`) are rejected rather than loaded
  with `torch.load`. This avoids executing arbitrary pickle payloads.
- PyTorch modules can be inspected through `TorchModuleSource` when the caller
  already has a live module. That is a trusted-code operation.
- GGUF and other quantized formats are unsupported. Quantized tensors cannot be
  edited in V1.
- Float8 tensors can be enumerated, but their numerical values are not decoded
  for analysis or editing; the operation raises `unsupported`.
- BF16 values are losslessly widened to float32 for analysis. Any result that
  depends on original storage precision reports it.
- Sparse tensors are not specially handled. They should be treated as
  unsupported unless a safe concrete representation is available.

## Model loading and architecture assumptions

- Loading a checkpoint into a module requires a compatible model class and
  state dict. V1 does not infer architecture from tensor names.
- Hugging Face Transformers support is optional and local-only. The adapter
  requires a local directory, uses `local_files_only=True`, and sets
  `trust_remote_code=False`. It does not download from the Hub.
- No model-specific preprocessing is embedded in the core. Causal-LM tokenizer
  preparation belongs to the adapter; generic models receive tensors/arrays.
- The built-in `mlp`, `linear`, and `transformer` models exist for offline
  demonstrations and tests, not as a general model zoo.
- Device placement is explicit. CPU is the mandatory baseline. GPU support is
  optional and depends on the installed torch build; ROCm is not required for
  installation or tests.
- GPU execution is not promised to be bitwise reproducible across hardware,
  drivers, kernels, or library versions.

## Activation capture

- Hooks capture module inputs and outputs by name. They do not understand
  semantic roles. A residual-stream state is only labeled as such when an
  adapter says so.
- The generic layer reports `kind="input"` or `kind="output"`; attention
  internals are not separated unless a module boundary exposes them.
- Hooks are reliable for ordinary forward passes and bounded teacher-forced
  language-model passes. Hooks during cached autoregressive generation are
  **not** a supported tracing mechanism in V1: cached generation can skip or
  reorder module calls, reuse key/value caches, and execute model-specific
  paths. Do not treat generation-time captures as equivalent to a full forward
  pass.
- Retained activation values are stored flat. `retained_shape` reflects the
  flat sample, not the original selected tensor shape; `shape` retains the
  original tensor shape.
- `token_position` is supported for 2-D (sequence axis 0) and 3-D (axis 1)
  tensors. Other ranks are rejected.
- Captures are bounded by `max_activation_elements_per_tensor` and
  `max_activation_elements_total`. Summary-only captures do not consume the
  value budget, but they still run hooks and compute statistics.
- In multi-call modules, entries are keyed by `(module, kind, call_index)`.
  Hooks are removed after the pass even when the model raises.

## Tensor analysis

- Statistics and difference metrics use float64. Inputs are decoded according
  to their storage dtype; BF16 is widened. Do not interpret float64 output as
  proof that the model computes in float64.
- NaN and infinity are counted, and distribution statistics use finite values
  only. If there are no finite values, scalar statistics are null.
- SVD/rank require a finite 2-D tensor and the full selected tensor within
  `max_svd_elements`. They do not run on sampled data.
- Norms require finite input; otherwise they are reported as unavailable rather
  than silently ignoring non-finite entries.
- Large operations use deterministic stride sampling where mathematically
  appropriate. Every sampled result is labeled with `sampled=true` and the
  stride. Sampling is not a substitute for a full answer when the operation
  requires one; such requests fail with `resource_limit`.
- The tool layer enforces `max_output_bytes` on the serialized envelope. A very
  large request may fail even if its underlying computation would succeed.

## Patching

- A patch targets one tensor and one rectangular selection. A patch set can
  contain multiple patches, but their targets must be disjoint. Multiple edits
  to one tensor must be composed into one patch or applied in separate sets.
- Exact source identity requires reading the full target tensor. Therefore a
  target tensor larger than `max_tensor_numel` (20M elements by default) cannot
  be patched in V1, even if only a small slice is edited.
- Patch payloads are bounded by `max_patch_payload_elements` (1M by default).
- Arithmetic is supported only on floating-point tensors. Integer and boolean
  tensors support exact `replace` only.
- Float8, quantized, and GGUF tensors cannot be edited.
- Patch math is float64, then cast to storage dtype. Casting can change values;
  this is reported, not corrected.
- There is no automatic patch optimizer, no gradient-based editing, no search
  for which weights to change, and no semantic interpretation of a patch.
- Patch application exports a new SafeTensors file. It does not mutate the
  source and does not produce a diff file or an undo log.
- The patch set identity includes the output path when supplied. Applying the
  same patches to different paths creates different patch-set IDs by design.

## Workspace and privacy

- Workspaces are local directories. There is no telemetry, no upload, no cloud
  service, and no API key requirement.
- Run records contain operation parameters, identities, device/dtype/seed,
  budget, dependency versions, artifact references, timings, and warnings.
  They do not intentionally store complete prompts or complete activation
  tensors. Artifacts are stored only when an operation requests them.
- Activation capture artifacts contain retained summaries and sampled values
  when `include_values` is true. Treat them as potentially sensitive model
  data. Summary-only capture is available when values are not needed.
- Workspace paths are checked against traversal, and artifacts are verified by
  size and SHA-256 on read. This is not encryption or access control; anyone
  with filesystem access can read the workspace.
- Generated data is excluded from Git by default. Back up or delete workspaces
  deliberately.

## Security model

- A SafeTensors file is treated as data, not code.
- A Python adapter or user-supplied model factory executes in the user’s
  interpreter. **It is not a sandbox.** Only load code you trust.
- The local Hugging Face adapter does not enable remote code execution, but the
  Transformers library and the model implementation are still ordinary Python
  dependencies with their own behavior.
- Path checks prevent accidental workspace escapes in the workbench’s own file
  operations; they are not a container boundary.
- Resource budgets protect against accidental large allocations, not against a
  malicious process with the same privileges.

## Scientific scope

- The workbench reports measurements. It does not decide that a direction,
  neuron, or layer “means” something because a number changed.
- There is no significance testing, causal claim generation, or automatic
  circuit interpretation in V1.2.
- There is no benchmark platform, leaderboard, or claim of improved model
  capability.
- A patch that produces a measurable difference is not evidence that the edit
  is useful, safe, or semantically meaningful.

## Behavioral observation

- V1.1 does not capture activations during cached autoregressive generation. A
  trace can reference future activation-capture artifacts, but no generation-time
  hook adapter is implemented.
- The scripted backend is deterministic but synthetic. It validates detector and
  evidence plumbing; it says nothing about a real model.
- The local Hugging Face backend is optional and not exercised by the offline
  test suite with a real checkpoint. Token log-probabilities depend on the
  backend returning generation scores; when it does not, the field is explicitly
  unavailable.
- Detectors measure text/token patterns. They do not know whether a repeated
  phrase is deliberate, quoted, formatted, or pathological. Severity is a
  magnitude mapping, not a diagnosis.
- The structured-output detector supports a deliberately small JSON contract:
  required top-level keys and top-level primitive types. It is not a general
  JSON Schema validator and is disabled by default because ordinary prose is not
  a failed JSON contract.
- Reconsideration-marker detection is literal and language-specific unless the
  caller supplies markers. It undercounts paraphrases and can double-count
  overlapping markers.
- Tool-call grouping uses canonical arguments. Semantically equivalent calls
  with different surface arguments are not grouped, and legitimate retries can
  be reported as repeated actions.
- Contrastive tests can only establish that authored pairs behaved as declared.
  Changing a prompt does not prove that a semantic variable was isolated, and
  the workbench does not assume paraphrases are equivalent.
- A baseline that does not satisfy the declared property makes a contrast
  inconclusive. A `fail` result means the declared relationship was violated,
  not that the model is defective.
- The reference-answer evaluator is string containment after normalization, not
  semantic entailment. Human judgements are subjective and carry reviewer
  metadata, but no automatic inter-rater reliability.
- No LLM evaluator is included, and no hosted provider is required or contacted.
- Failure-case export is self-contained for traces, contrast results, activation
  artifacts, and patches, but the combined bundle is not split automatically and
  can be large. Redacted exports are structural and scan-verified, but they
  deliberately reduce replay capability and refuse linked artifacts that cannot
  be transformed.
- Traces store input/output text by default. Use `privacy.store_*` at collection
  or an export policy at export time. This is not encryption or access control.
- Redaction protects content carried in fields the workbench knows about. A
  caller who embeds a secret inside a machine-identifier field (for example a
  dict key that looks like an identifier) must declare it through
  `sensitive_markers`; the export is then refused rather than silently shipping
  it. Values shorter than 8 characters are removed but are not searched for.
- Cases link evidence; they do not prove causation. Hypotheses remain
  hypotheses.

## Replay and alignment

- Teacher-forced replay does not regenerate tokens. A replay result always
  reports `generation_reproducibility: not-claimed`, and no V1.2 code path
  upgrades that to a claim.
- `numerical_agreement: verified` means the recorded token log-probabilities were
  reproduced within the declared tolerance on this machine with this checkpoint.
  It is not a statement about sampling, caching, or another device.
- Replay requires recorded token ids, or recorded text plus an explicit encoder
  and encoding. A chat-template trace without stored token ids is refused rather
  than re-rendered.
- Replay uses the model's own forward pass. Hybrid and recurrent stacks
  (Gated DeltaNet, Mamba-style mixers) may compute different intermediates than
  cached autoregressive decoding; the plan and result carry a limitation saying
  so instead of asserting equivalence.
- CPU replay is not bitwise identical to GPU replay, and bf16/fp16 execution adds
  a precision limitation to every result where it applies.
- Character offsets are only reported when they can be validated against the
  recorded text. Non-ASCII text the tokenizer cannot encode round-trip (for
  example an unknown-token fallback) yields `null` offsets, an `unavailable`
  entry, and a warning.
- A text span that begins or ends inside a token is reported as ambiguous with a
  coverage fraction. The workbench does not silently pick the nearest boundary.
- A local window is never treated as equivalent to the full context: requesting
  one sets `context_mode: "explicit-window"` and warns that the causal prefix is
  not preserved.

## Activation investigation

- Captures happen on a teacher-forced replay, not on the original generation.
  Live generation-time capture with KV caches is not implemented.
- Summary mode retains statistics, not values. Elementwise comparison of two
  summary captures is refused with an explicit `unavailable` reason.
- `selected` mode stride-samples vectors larger than the per-tensor element
  limit and records `sampled`/`stride`; it does not silently downcast.
- The plain hook backend addresses the checkpoint's own module names. The nnterp
  backend addresses normalised names. A path from one backend is not valid in
  the other, and the backends are never substituted silently.
- The nnterp adapter requires `attn_implementation="eager"` for attention-output
  capture, refuses architectures nnterp does not cover, and pins its tested
  versions. It is validated against a locally constructed tiny hybrid Qwen3.5
  checkpoint, not against every hybrid architecture.
- A large activation difference is not a mechanism, and a small one is not
  evidence against one. Comparisons carry this limitation and never label
  dimensions, neurons, or directions with concepts.
- Explicit position pairings (`paired_positions`, `semantic_event`) must be
  marked `reviewed`. The workbench cannot decide that two contexts represent the
  same event.
- `sensitive_markers` is the only mechanism for content hidden inside
  identifier-shaped fields. Undeclared identifiers that look structural are
  removed but not searched for.

## Known V1.2 gaps

- No MCP server is included. The JSON schemas and service layer are designed so
  a thin adapter can be added without changing core logic.
- No general model registry or remote checkpoint download.
- No activation capture during generation with KV caches.
- No pooled activation aggregation over raw values; aggregation is over
  summary statistics (`mean_of_summaries`) and states this.
- No automatic GPU memory planning or multi-device execution.
- No patch composition on a single tensor inside one set.
- No bitwise replay guarantee across machines.
- No activation patching, ablation, or counterfactual continuation. V1.2
  measures states; it does not intervene on them.
- No automatic span selection, mechanism discovery, or dimension labelling.
- No generation-time activation recording, and no pooled aggregation of raw
  activations across positions beyond the documented summary statistics.
- The optional nnterp backend has been exercised on one synthetic hybrid
  architecture and on one real Qwen3.5 hybrid checkpoint; other families are
  untested and may be refused.
