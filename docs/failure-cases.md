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
| `activation_captures` | `ArtifactRef` records for activation evidence |
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

The workbench does not move hypotheses automatically. Nothing in V1.1 claims
that a weight, direction, layer, or activation caused the observed behavior.

## Reference validation

On creation and save, the case store verifies:

- every observation and control trace exists and its artifact checksum matches,
- every contrast artifact exists and has kind `contrast_result`,
- every activation-capture artifact exists and verifies,
- every patch or patch-set identifier exists in the workspace.

A case may reference a trace whose activation data is absent; that is normal.
Activation captures are optional evidence, not a requirement.

## Export bundle

`export_failure_case` writes a self-contained JSON document:

```json
{
  "schema_version": "wsp.case-export/1.0",
  "exported_at": "2026-11-14T20:00:00Z",
  "case_sha256": "sha256:...",
  "case": { "...": "wsp.case/1.0" },
  "traces": [
    {
      "trace_id": "tr_...",
      "artifact_id": "a_...",
      "artifact_sha256": "sha256:...",
      "redacted": false,
      "trace": { "...": "wsp.trace/1.0" }
    }
  ],
  "contrast_results": [
    {"artifact_id": "a_...", "sha256": "sha256:...", "nbytes": 1234, "payload": {}}
  ],
  "activation_captures": [
    {
      "artifact_id": "a_...",
      "kind": "activation_capture",
      "format": "json",
      "sha256": "sha256:...",
      "nbytes": 1234,
      "data_base64": "..."
    }
  ],
  "patches": [
    {"kind": "patch", "patch_id": "p_...", "payload": {}}
  ],
  "payload_artifacts": [
    {
      "artifact_id": "a_...",
      "kind": "patch_payload",
      "format": "npy",
      "sha256": "sha256:...",
      "nbytes": 1234,
      "data_base64": "..."
    }
  ],
  "references": {
    "traces": [],
    "contrast_artifact_ids": [],
    "activation_capture_artifact_ids": [],
    "patches": []
  },
  "warnings": []
}
```

The bundle contains enough data to restore the case in a fresh workspace:
embedded traces are stored as immutable trace artifacts, contrast results and
activation artifacts are recreated from their payloads (content addressing gives
them the same identifiers), patch payload artifacts are restored, and
patches/patch sets are restored through the existing workspace store.

`import_failure_case` verifies:

- the export schema version,
- `case_sha256` against the canonical case payload,
- each embedded trace id and (unless redacted) its content checksum,
- contrast and activation artifact identifiers/checksums against the restored
  content,
- patch identifiers against their payloads.

A tampered payload fails with `artifact_error`. An existing trace or case with
the same identifier but different content is rejected unless `overwrite=true` is
explicitly supplied for the case; traces are immutable and are never overwritten.

## Redaction

`export_failure_case(redact_text=true)` removes:

- trace input text and chat messages,
- decoded output text and token text,
- rendered prompts,
- local model/source paths,
- tool-call arguments.

It keeps token ids, positions, termination, provenance, checksums, and reference
identifiers. The bundle records `redacted=true` per trace and warns that
human-authored titles, summaries, annotations, and hypotheses are **not**
automatically redacted. Redacted traces cannot be checksum-verified against the
original artifact; import accepts them as redacted evidence and reports it.

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
