# Roadmap

This roadmap separates what V1 actually contains from hypotheses about future
work. Future items are not commitments and should not be implemented merely
because they sound useful.

## Completed in V1

### Foundation and inspection

- Installable `wsp` package and CLI.
- Versioned pydantic schemas and JSON envelopes.
- SafeTensors header-only inspection with tensor metadata.
- Structure, file, and per-tensor fingerprints.
- Model manifests and atomic manifest export.
- Checkpoint comparison with missing/extra/dtype/shape/value detection.
- Live `torch.nn.Module` inspection adapter.
- SafeTensors-only automatic loading; pickle rejected.

### Tensor analysis

- Mean/std/min/max, finite and non-finite counts.
- Quantiles and histograms.
- Vector and matrix norms.
- Cosine, dot product, projection, correlation-style comparisons.
- Matrix rank and singular-value summaries.
- Difference metrics and bounded deterministic sampling.
- Explicit sampled/approximate labels.

### Activation instrumentation

- Forward hooks on named modules and the root.
- Input/output capture with bounded retention.
- Summary-only capture mode.
- Token-position selection for 2-D/3-D tensors.
- Deterministic cleanup, including error paths.
- Activation comparison and summary aggregation.
- Device/dtype reporting.

### Controlled weight editing

- Immutable `WeightPatch` format.
- Replace, add, scale, rank-one, and projection operations.
- Shape/dtype/index/finiteness/resource validation.
- Exact source tensor fingerprints.
- Preview without committing.
- Patch sets with disjoint targets.
- Streaming export to a derived SafeTensors checkpoint.
- Rollback by verified source selection.
- Precision and casting reports.
- Deterministic patch and patch-set IDs.

### Execution and experiments

- Inference-mode CPU-baseline forward execution.
- Optional local Hugging Face causal-LM adapter.
- Output comparisons and activation comparisons.
- Offline built-in models for demos/tests.
- Workspace with content-addressed artifacts.
- Run records with environment, parameters, replay metadata, and warnings.
- Experiment grouping.
- Patch, patch-set, manifest, and artifact persistence.

### Interfaces and tests

- Structured operation registry with JSON Schemas.
- Python API and CLI sharing one service layer.
- CLI JSON output with machine-readable failures.
- Offline CPU test suite covering the 15 V1 acceptance behaviors.
- Offline examples for inspection, activation capture, patching, comparison,
  and rollback.

## Deliberately deferred

These are **not** in V1:

- Autonomous research agents or LLM provider integration.
- Automatic weight-edit discovery, gradient optimization, or fine-tuning.
- Automated neuron/circuit interpretation or significance claims.
- Model merging, circuit grafting, or cross-model patching.
- Benchmark suites or claims of improved model intelligence.
- Training specialized interpretability models.
- GUI/dashboard, distributed inference, or cloud infrastructure.
- GGUF or direct quantized/float8 weight editing.
- Cached autoregressive generation tracing.
- Pooled raw-activation aggregation.
- Multi-patch edits to a single tensor inside one patch set.

## Possible future work (hypotheses, not commitments)

Ordered roughly by how directly they extend existing primitives.

1. **MCP adapter.** Expose the existing registry as MCP tools. The schemas are
   already provider-independent; this should be a thin transport layer, not a
   second implementation.
2. **Composable patch algebra.** Validate and combine multiple operations on one
   tensor before application, with explicit ordering and a single resulting
   `WeightPatch`.
3. **More precise quantized support.** Read/dequantize, operate in a documented
   higher-precision space, requantize, and report quantization error. This needs
   a careful format-specific design and should not be rushed.
4. **Chunked tensor algebra.** Stream large tensor statistics/differences
   without materializing the whole tensor, raising the patchable tensor size
   beyond `max_tensor_numel`.
5. **Generation-time instrumentation.** A documented adapter contract for
   capturing selected hidden states during cached decoding without pretending
   module hooks are equivalent to a full forward pass.
6. **Replay runner.** Re-execute a saved run record against the same local
   artifacts and report environment drift explicitly.
7. **Model-family adapters.** Small, explicit adapters for common local
   architectures and tokenizer preparation, kept outside the core.
8. **Activation dataset aggregation.** Pool raw values under explicit memory
   budgets, in addition to the current summary-statistics aggregation.
9. **Patch diff artifacts.** Export human-readable and machine-readable
   before/after summaries and a compact sparse delta for review.
10. **Deterministic execution profiles.** Record and compare backend settings
    (thread counts, deterministic algorithms) without promising cross-hardware
    bitwise identity.

## Research directions (outside the tool core)

If the instruments prove trustworthy, later work may use them to ask questions
such as:

- Which weight directions are causally involved in a behavior?
- How local or distributed is a particular capability?
- Do edits interact across layers or tasks?
- Can a measured weight-space edit be reproduced across model families?

Those are research programs. V1 deliberately stops at instruments and
measurements. See the WSP research write-up and released logs referenced in the
project README for context and attribution; no upstream code or experiment
assets are bundled here.

## Quality gates for future changes

- New operations must have JSON Schemas and bounded outputs.
- New file formats must be versioned and safe to parse.
- New numerical behavior must report precision and sampling.
- New execution paths must preserve deterministic CPU tests or explicitly mark
  themselves environment-dependent.
- No feature should make source checkpoint mutation possible.
- No agent-only behavior. Python, CLI, and future tool transports must call the
  same service.
