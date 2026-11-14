# Failure cases

Schema: `wsp.case/1.0`
Storage: `cases/<case_id>/case.json`
Export: `wsp.case-export/1.0`

A failure case is a durable investigation record, not a verdict. It may begin as
a single unexplained reproducible observation. It does not require a known
cause, a confirmed detector hit, or a weight edit.

## Case fields

| Field | Meaning |
| --- | --- |
| `case_id` | `case_` + 16 lowercase hex characters |
| `title`, `summary` | Human-authored description |
| `created_at`, `updated_at`, `revision` | Lifecycle metadata |
| `review` | Review status, reviewers, notes, last review time |
| `model` | Model identity; inferred from the first referenced trace when omitted |
| `inference` | Generation settings; inferred from the first trace when omitted |
| `observations` | Trace ids showing the suspected failure |
| `controls` | Trace ids for healthy or comparison observations |
| `findings` | Detector findings copied into the case as evidence |
| `contrasts` | References to stored `contrast_result` artifacts |
| `evaluations` | Stored evaluation outcomes |
| `reproduction` | Backend, command, notes, environment, parameters, seed |
| `annotations` | Human notes with author and kind |
| `hypotheses` | Explicitly labelled candidate explanations |
| `activation_captures` | `ArtifactRef` records for V1 hook activation evidence |
| `investigations` | `InvestigationLink` records pointing at investigation-evidence artifacts |
| `patches` | Patch or patch-set references |
| `tags` | Free-form labels for filtering |
| `metadata` | Additional structured information |
| `warnings` | Non-fatal construction warnings |

A case requires at least one observation trace, one control trace, or one
finding. Observation and control trace ids must be disjoint and unique.

## Review status

| Status | Meaning |
| --- | --- |
| `open` | Newly recorded |
| `triage` | Under review |
| `confirmed` | Reviewers accept that the observation is reproducible and relevant |
| `rejected` | Reviewers do not accept the case |
| `archived` | Kept for reference, not active |

`update_failure_case` changes review status, appends a reviewer, replaces review
notes, adds an annotation, adds a hypothesis, and/or changes hypothesis statuses.
Every update increments `revision` and refreshes `updated_at`.

## Hypotheses

A hypothesis is always marked with one of:

- `proposed` — not yet tested,
- `testing` — an investigation is in progress,
- `supported` — evidence supports it, but no causal proof is implied,
- `refuted` — evidence contradicts it,
- `inconclusive` — evidence is insufficient.

The workbench does not move hypotheses automatically. Nothing in V1.2 claims
that a weight, direction, layer, or activation caused the observed behavior.

## Reference validation

On creation and save, the case store verifies:

- every observation and control trace exists and its artifact checksum matches,
- every contrast artifact exists and has kind `contrast_result`,
- every activation-capture artifact exists and verifies,
- every investigation-evidence artifact exists, has kind
  `investigation_evidence`, carries the linked `evidence_id`, belongs to the
  same case when it records a `case_id`, and whose referenced capture artifacts
  exist with kind `trace_capture` and comparison artifacts with kind
  `capture_comparison`,
- every patch or patch-set identifier exists in the workspace.

A case may reference a trace whose activation data is absent; that is normal.
Activation captures are optional evidence, not a requirement.

## Export bundle

`export_failure_case` writes a self-contained JSON document and takes an explicit
privacy policy:

```json
{
  "schema_version": "wsp.case-export/1.1",
  "exported_at": "2026-11-14T20:00:00Z",
  "case_sha256": "sha256:...",
  "case": { "...": "wsp.case/1.0" },
  "privacy": {
    "mode": "private",
    "policy": { "...": "wsp.export-policy/1.0" },
    "limitations": [],
    "removed_fields": 0,
    "removed_strings": 0,
    "checksums_dropped": [],
    "replay_capability": "full",
    "leak_scan": "clean"
  },
  "traces": [
    {
      "trace_id": "tr_...",
      "artifact_id": "a_...",
      "artifact_sha256": "sha256:...",
      "redacted": false,
      "trace": { "...": "wsp.trace/1.1" }
    }
  ],
  "contrast_results": [
    {"artifact_id": "a_...", "kind": "contrast_result", "format": "json",
     "sha256": "sha256:...", "nbytes": 1234, "data_base64": "..."}
  ],
  "activation_captures": [
    {"artifact_id": "a_...", "kind": "activation_capture", "format": "json",
     "sha256": "sha256:...", "nbytes": 1234, "data_base64": "..."}
  ],
  "patches": [
    {"kind": "patch", "patch_id": "p_...", "payload": {}}
  ],
  "payload_artifacts": [
    {"artifact_id": "a_...", "kind": "patch_payload", "format": "npy",
     "sha256": "sha256:...", "nbytes": 1234, "data_base64": "..."}
  ],
  "payload_artifacts": [],
  "investigation_artifacts": [],
  "artifact_map": {},
  "references": {
    "traces": [],
    "contrast_artifact_ids": [],
    "activation_capture_artifact_ids": [],
    "patches": [],
    "investigation_artifact_ids": []
  },
  "warnings": []
}
```

The bundle contains enough data to restore the case in a fresh workspace:
embedded traces are stored as immutable trace artifacts, contrast results and
activation artifacts are recreated from their payloads (content addressing gives
them the same identifiers), patch payload artifacts are restored, and
patches/patch sets are restored through the existing workspace store.

Investigation evidence is exported as a *graph*, not as one file. A private
export walks the evidence record and embeds everything it reaches — capture
records, retained activation vectors, replay plans and results, token alignments,
span resolutions, and comparisons — so the restored case can reproduce the same
measurements. A redacted export keeps only the capture and comparison artifacts,
because replay results and alignments embed recorded token text.

`import_failure_case` verifies:

- the export schema version (legacy `wsp.case-export/1.0` bundles still import),
- `case_sha256` against the canonical case payload,
- each embedded trace id and its content checksum,
- contrast and activation artifact identifiers/checksums against the restored
  content,
- patch identifiers against their payloads,
- `artifact_map` targets against the artifacts actually restored.

A tampered payload fails with `artifact_error`. An existing trace or case with
the same identifier but different content is rejected unless `overwrite=true` is
explicitly supplied for the case; traces are immutable and are never overwritten.

## Redaction

There is exactly one redaction implementation,
`wsp.workspace.privacy`, shared by every export path (traces, cases, and later
activation measurements). A redacted export is rebuilt rather than scrubbed
field by field: every string that is not on an explicit structural allowlist is
replaced, and every replaced string is remembered.

`export_failure_case(policy={"mode": "redacted"})` removes:

- trace input text, chat messages, rendered prompts, and input checksums,
- decoded output text, token text, token log-probabilities, and top alternatives,
- token ids (they are decodable text with the right tokenizer) unless
  `keep_token_ids=true` is explicitly requested,
- model names, local checkpoint paths, and file names,
- tool-call arguments, results, and their digests,
- case titles and summaries, human annotations, hypothesis statements and
  rationales, tags, review notes, and reproduction commands,
- detector evidence names/details, finding explanations and free-form values,
- contrast test names, variant inputs/outputs, expectations, and observed values,
- activation-capture payloads (reduced to metadata and summary statistics),
- investigation-evidence titles, summaries, notes, hypothesis statements, and
  detector values,
- evidence links to replay, alignment, and span artifacts (those artifacts embed
  recorded token text and are not part of a redacted export),
- retained activation vectors and their checksums,
- provenance notes, environment metadata, warnings, and serialized errors.

Redacted exports rewrite nested artifacts rather than copying them: contrast
results, activation captures, patch payloads, and patch provenance are
transformed, their bytes are re-checksummed, and their artifact identifiers are
recomputed. `artifact_map` records every old identifier and its replacement (or
`null` when the artifact was omitted), and the case's own references are rewritten
to the new identifiers so the bundle stays internally consistent.

After serialization the whole document is scanned for every removed string and
for any caller-supplied `sensitive_markers`. A single hit is a hard error: the
export is refused and no file is written. The scan is a backstop for the
transformer, not a substitute for it.

Limitations are recorded in the bundle (`privacy.limitations` and
`case.redaction`), and the export says that replay capability is reduced. When
token ids are dropped, the export is no longer reproducible evidence and is
marked `replay_capability: "unavailable"`. With `keep_token_ids=true` it is
`"reduced"`: positions can be replayed only against the same tokenizer and model.

Retained activation vectors are the one numeric kind that is dropped by design
rather than refused: a vector's content *is* the measurement, so there is no way
to remove the source text from it. The export drops the artifact, rewrites the
capture measurement to `retained: false` with `value_artifact_id: null`, records
`artifact_map[old_id] = null`, and adds a limitation. That is a deliberate,
recorded reduction — not a silent omission.

A linked artifact of any other kind that cannot be transformed safely (for
example an opaque binary blob) makes the redacted export **fail**. Setting
`on_untransformable_artifact: "omit"` accepts a recorded gap instead: the artifact
is dropped, `artifact_map` maps it to `null`, and a limitation is added. A linked
contrast result that does not match its own schema is refused, because the
transformer cannot prove it covered every field.

Redacted traces cannot be checksum-verified against the original artifact;
import accepts them as redacted evidence and reports the reduced capability.
Legacy `wsp.case-export/1.0` bundles produced by earlier releases may still
contain nested contrast inputs. They are imported for compatibility but are not
produced by the current exporter.

## Reload and portability

- `include_traces=false` produces a smaller bundle that references traces rather
  than embedding them. Such a bundle is reloadable only in a workspace that
  already contains those traces; otherwise the import fails rather than
  fabricating evidence.
- Activation arrays are base64-encoded bytes, not pickles. The same artifact
  format and checksum rules apply.
- No export operation uploads anything. The output path is chosen by the caller.
- Exported bundles can be large when activation captures are included. The
  existing artifact-size budget bounds one component, but the combined bundle is
  only checked against `max_artifact_bytes` at write time; very large
  investigations should be split or exported without embedded traces.

## What a case is not

- It is not a diagnosis.
- It is not a benchmark result.
- It is not proof that a weight patch caused an improvement.
- It is not a substitute for a controlled experiment.
- It does not become true because a detector produced a finding.
