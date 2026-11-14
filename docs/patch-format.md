# Patch format (`wsp.patch/1.0`)

A `WeightPatch` is an immutable, self-describing description of one edit to one
tensor. It is not a function pointer and not a Python pickled object. Creating,
previewing, or storing a patch never modifies a checkpoint.

## Top-level fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Always `wsp.patch/1.0` for this format |
| `patch_id` | Deterministic `p_<16 hex>` identifier |
| `created_at` | Creation timestamp (not part of identity) |
| `source` | `CheckpointIdentity` of the source checkpoint |
| `target` | Tensor name |
| `target_info` | `TensorInfo`: name, dtype, shape, numel, nbytes, offset |
| `selection` | Resolved rectangular selection and selected shape |
| `operation` | `replace`, `add`, `scale`, `rank_one`, or `projection` |
| `params` | Operation-specific parameters |
| `expected` | Exact target fingerprint, dtype, and shape at creation time |
| `parameter_count` | Number of selected elements |
| `estimated_memory_bytes` | Conservative estimate for creation/preview working memory |
| `before` / `after` | `StatsSummary` for the selected region |
| `effect` | Change counts, max/mean absolute change, relative Frobenius change, cosine, introduced non-finite flag |
| `precision` | Compute/storage dtypes and casting report |
| `provenance` | Creator, tool, dependency versions, notes, optional request ID |

Example excerpt:

```json
{
  "schema_version": "wsp.patch/1.0",
  "patch_id": "p_1c71f988491ebc01",
  "source": {
    "schema_version": "wsp.identity/1.0",
    "format": "safetensors",
    "structure_fingerprint": "sha256:6bce22d902cdc46d98da0ed46e3ac0b332400d4092e7c2da18c056575a5ffa00",
    "file_sha256": "sha256:183143c71040db4e080d4973528c148e857606f8f1e2a6747e7363d74dab258d"
  },
  "target": "fc1.bias",
  "operation": "add",
  "selection": {
    "source_shape": [16],
    "selected_shape": [16],
    "numel": 16,
    "dims": [{"axis": 0, "start": 0, "stop": 16, "step": 1, "length": 16}]
  },
  "expected": {
    "fingerprint": "sha256:b0d2130e6e4c25499c63dfce7c7e694ed34a5667bacab2066f58393127350949",
    "dtype": "float32",
    "shape": [16]
  },
  "params": {
    "kind": "add",
    "delta": {"values": [0.125], "shape": [16], "dtype": "float64", "numel": 16, "sha256": "sha256:..."}
  }
}
```

## Payloads

`PayloadRef` contains exactly one source:

- inline `values`: a JSON-compatible nested real-number array; or
- `artifact_id`: a workspace artifact ID containing a `.npy` array.

It may also carry `shape`, `dtype`, `numel`, and `sha256`. Shape is validated
exactly against the resolved selection. Payloads must be finite real numbers and
are bounded by `max_patch_payload_elements`.

Large inline payloads created through the tool service are externalized to a
workspace artifact by default (more than 4,096 elements or 64 KiB). The patch
then references the artifact ID. A patch with an external payload is not
self-contained; the artifact must remain available in the workspace.

## Selection

`Selection` is resolved against the target shape before the patch is created.
The stored `selection` contains the source shape, selected shape, numel, and one
`dims` entry per axis. Axes not mentioned are full. Indices are non-negative and
half-open; negative indices/steps are deliberately rejected.

## Operations

All arithmetic is performed in float64 on the decoded selected values. `X` is
the selected region as a 2-D matrix where applicable.

| Kind | Parameters | Mathematical effect |
| --- | --- | --- |
| `replace` | `values` | `X = values` |
| `add` | `delta` | `X = X + delta` |
| `scale` | `factor` | `X = factor * X` |
| `rank_one` | `u`, `v`, `scale` | `X = X + scale * outer(u, v)` |
| `projection` | `direction`, `axis`, `mode`, `strength` | Remove or add a component along a unit direction |

Projection requires a 2-D selected region. The direction is normalized to unit
length `d`. For `axis=1`, each row is treated as a vector and the update is
`outer(X @ d, d)`. For `axis=0`, each column is treated as a vector and the
update is `outer(d, d @ X)`. `mode="remove"` subtracts `strength` times the
update; `mode="add"` adds it. A zero-norm direction is rejected.

Rank-one and projection updates operate on the selected region, not on the full
tensor if the selection is partial. A partial rank-one update is not a
mathematical rank-one update of the full matrix.

## Precision and casting

- Compute dtype is float64.
- The result is cast back to the target tensor’s storage dtype.
- BF16 is converted through a torch round-trip; float16/32/64 use NumPy casts.
- Integer and boolean tensors support `replace` only, and only exact
  representable values. Arithmetic on non-floating tensors is rejected.
- Float8 and quantized tensors (including GGUF) are not editable in V1.
- If casting changes any value, `precision.cast_changed_values` is true and a
  note is recorded.
- If casting produces a non-finite value where the computed value was finite,
  the patch is rejected.

Precision is reported, not hidden. A patch that is exact in float64 may differ
from its computed result when stored as BF16.

## Compatibility checks

A patch is compatible only when all of the following hold:

1. The source structure fingerprint matches the current checkpoint.
2. The target tensor exists and has the recorded dtype and shape.
3. The exact target tensor fingerprint matches `expected.fingerprint`.
4. The selection is valid for the target shape.
5. The recomputed operation does not introduce non-finite values.

The target fingerprint covers the **full** tensor, so patch creation needs the
full target tensor to fit `max_tensor_numel` (20,000,000 elements by default).
This is deliberate: a patch must be tied to exact source values, not just a
shape.

Preview additionally warns when the source file hash differs but the structure
and target fingerprint still match (for example, a byte-identical tensor in a
re-saved file), or when stored before/after summaries differ from recomputation.
Such warnings do not silently invalidate a mathematically compatible patch, but
they are visible.

## Patch identity

`patch_id` is a deterministic hash of:

- source structure fingerprint,
- target name,
- canonical selection,
- normalized operation parameters and payload identity,
- expected target fingerprint.

`created_at`, provenance notes, and request IDs are not part of identity.
Recreating the same mathematical patch produces the same ID. Workspace saves
treat an existing ID as idempotent when the semantic content matches, ignoring
timestamps, per-run request IDs, and provenance notes/dependencies. The first
saved provenance is retained. Reapplying an equivalent patch set to the same
destination refreshes the journaled output checksum (the run record preserves
each individual application); a SafeTensors header may serialize its metadata
in a different order, so file bytes are not a mathematical identity. Any other
content difference under the same ID is an `artifact_error`.

## Patch sets (`wsp.patchset/1.0`)

A patch set groups patches that are applied together:

| Field | Meaning |
| --- | --- |
| `schema_version` | `wsp.patchset/1.0` |
| `patchset_id` | Deterministic `ps_<16 hex>` |
| `created_at` | Creation timestamp |
| `source` | Source checkpoint identity |
| `patch_ids` | Ordered patch IDs |
| `patches` | Embedded `WeightPatch` records |
| `output_path` | Derived checkpoint path (when applied) |
| `output_identity` | Identity of the exported checkpoint (when applied) |
| `status` | `pending` or `applied` |

Rules:

- All patches must have the same source structure fingerprint as the set.
- Targets must be disjoint. Multiple patches on the same tensor are rejected;
  apply them in separate sets or compose the operation before creating a patch.
- At most `max_patchset_tensors` patches.
- `patchset_id` includes the output path when one is supplied, so applying the
  same patches to different destinations produces distinct IDs.
- Applying the same set to the same destination writes the same bytes on CPU
  and is idempotent at the workspace-record level.

## Application and export

Application reads only the targeted tensors into memory. The SafeTensors header
is parsed, data offsets are validated for overlap and length, and untouched
tensors are copied in 8 MiB streaming chunks. The new header preserves source
metadata. The output is written to a temporary file and atomically renamed.

The exported checkpoint is reopened and its structure fingerprint is checked
against the source before the result is returned. The source path is refused as
a destination, and the source file is never opened for writing.

`overwrite=false` is the default. An existing destination causes
`safety_violation`.

## Rollback

There is no mutation to undo. `revert_patchset()` verifies that the supplied
path is the patch set’s source structure and returns the immutable source
identity. The derived checkpoint can be deleted by the user; the source remains
the authoritative baseline.

## Out of scope for V1

- Multiple operations composed against one tensor in a single patch set.
- Automatic patch optimization, search, or gradient-based editing.
- Quantized/GGUF tensor editing.
- In-place source modification.
- Bitwise-identical GPU replay.
