"""Persistent failure-case registry and self-contained export bundles.

Export bundles have two modes, and the difference is a property of the archive
rather than a promise about the code path:

``private``
    The bundle embeds the exact traces, contrast results, measurements, and
    notes that were recorded.  Nested artifacts are embedded byte-for-byte, so
    their content identifiers survive a round trip unchanged.

``redacted``
    Every free-form string is replaced through
    :mod:`wsp.behavior.redaction`, linked artifacts are transformed (and get new
    content identifiers), all references are rewritten to point at the
    transformed artifacts, and the finished archive is scanned for every string
    the policy removed.  A surviving string aborts the export.

An artifact with no safe transformation aborts a redacted export unless the
caller explicitly accepts omission, in which case the case says so.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Iterable

from wsp.behavior.ids import new_case_id
from wsp.behavior.models import (
    CaseSummary,
    FailureCase,
    GenerationTrace,
    InvestigationLink,
)
from wsp.behavior.redaction import (
    ArtifactRewrite,
    RedactedCase,
    collect_artifact_ids,
    dropped_trace_checksums,
    redact_case,
    redact_trace_payload,
    rewrite_artifact_ids,
)
from wsp.behavior.trace import TraceStore
from wsp.core.budget import Budget, enforce_bytes
from wsp.core.errors import (
    ArtifactError,
    InputError,
    NotFoundError,
    ResourceLimitError,
    SafetyError,
)
from wsp.core.hashing import sha256_bytes
from wsp.core.json import atomic_write_bytes, canonical_bytes, dumps, read_json, to_jsonable
from wsp.core.time import utc_now_iso
from wsp.patches.spec import PatchSet, WeightPatch
from wsp.workspace.privacy import (
    ExportPolicy,
    RemovedText,
    base64_payload,
    content_id,
    enforce_no_leaks,
    safe_display_name,
    scrub_provenance,
    transform_artifact,
)
from wsp.workspace.store import Workspace

CASE_ID_PATTERN = re.compile(r"^case_[0-9a-f]{16}$")
EXPORT_SCHEMA = "wsp.case-export/1.1"
LEGACY_EXPORT_SCHEMA = "wsp.case-export/1.0"
SUPPORTED_EXPORT_SCHEMAS = (EXPORT_SCHEMA, LEGACY_EXPORT_SCHEMA)


def new_case_identifier() -> str:
    return new_case_id()


def case_bytes(case: FailureCase) -> bytes:
    return canonical_bytes(case.model_dump(mode="json"))


def case_sha256(case: FailureCase) -> str:
    return sha256_bytes(case_bytes(case))


class CaseStore:
    def __init__(self, workspace: Workspace, *, traces: TraceStore | None = None) -> None:
        self.workspace = workspace
        self.budget = workspace.budget
        self.traces = traces or TraceStore(workspace)
        self.directory = workspace.root / "cases"
        self.directory.mkdir(parents=True, exist_ok=True)

    def _directory_for(self, case_id: str) -> Path:
        if not case_id or "/" in case_id or "\\" in case_id or case_id.startswith("."):
            raise InputError("invalid case identifier", details={"case_id": case_id})
        path = (self.directory / case_id).resolve()
        if self.directory.resolve() not in path.parents:
            raise SafetyError("case path escapes the workspace")
        return path

    def path_for(self, case_id: str) -> Path:
        return self._directory_for(case_id) / "case.json"

    def validate_references(self, case: FailureCase) -> None:
        for trace_id in case.observations + case.controls:
            record = self.traces.get_record(trace_id)
            self.workspace.artifacts.verify(record.artifact)
        for reference in case.contrasts:
            if reference.artifact_id is not None:
                ref, _path = self.workspace.artifacts.get(reference.artifact_id)
                if ref.kind != "contrast_result":
                    raise ArtifactError(
                        "contrast reference does not point at a contrast result",
                        details={"artifact_id": reference.artifact_id, "kind": ref.kind},
                    )
                payload = self.workspace.artifacts.read_json(reference.artifact_id)
                if isinstance(payload, dict) and payload.get("result_id") not in (
                    None,
                    reference.result_id,
                ):
                    raise ArtifactError(
                        "contrast result identifier does not match its artifact",
                        details={
                            "artifact_id": reference.artifact_id,
                            "expected": reference.result_id,
                        },
                    )
        for artifact in case.activation_captures:
            self.workspace.artifacts.verify(artifact)
        for link in case.investigations:
            self._validate_investigation(link, case)
        for patch_reference in case.patches:
            if patch_reference.patch_id is not None:
                self.workspace.load_patch(patch_reference.patch_id)
            if patch_reference.patchset_id is not None:
                self.workspace.load_patchset(patch_reference.patchset_id)

    def _validate_investigation(self, link: InvestigationLink, case: FailureCase) -> None:
        """An investigation link must resolve to a consistent evidence record."""

        ref, _path = self.workspace.artifacts.get(link.artifact_id)
        if ref.kind != "investigation_evidence":
            raise ArtifactError(
                "investigation link does not point at an investigation evidence record",
                details={"artifact_id": link.artifact_id, "kind": ref.kind},
            )
        payload = self.workspace.artifacts.read_json(link.artifact_id)
        if not isinstance(payload, dict):
            raise ArtifactError(
                "investigation evidence is not a JSON object",
                details={"artifact_id": link.artifact_id},
            )
        if payload.get("evidence_id") != link.evidence_id:
            raise ArtifactError(
                "investigation evidence identifier does not match its artifact",
                details={
                    "artifact_id": link.artifact_id,
                    "expected": link.evidence_id,
                    "found": payload.get("evidence_id"),
                },
            )
        recorded_case = payload.get("case_id")
        if recorded_case is not None and recorded_case != case.case_id:
            raise ArtifactError(
                "investigation evidence belongs to a different case",
                details={"case_id": case.case_id, "recorded": recorded_case},
            )
        for capture in payload.get("captures") or []:
            artifact_id = capture.get("artifact_id")
            if artifact_id is None:
                continue
            capture_ref, _path = self.workspace.artifacts.get(artifact_id)
            if capture_ref.kind != "trace_capture":
                raise ArtifactError(
                    "investigation references an artifact that is not a trace capture",
                    details={"artifact_id": artifact_id, "kind": capture_ref.kind},
                )
        for comparison in payload.get("comparisons") or []:
            artifact_id = comparison.get("artifact_id")
            if artifact_id is None:
                continue
            comparison_ref, _path = self.workspace.artifacts.get(artifact_id)
            if comparison_ref.kind != "capture_comparison":
                raise ArtifactError(
                    "investigation references an artifact that is not a capture comparison",
                    details={"artifact_id": artifact_id, "kind": comparison_ref.kind},
                )
        for intervention in payload.get("interventions") or []:
            artifact_id = intervention.get("artifact_id")
            if artifact_id is None:
                continue
            intervention_ref, _path = self.workspace.artifacts.get(artifact_id)
            if intervention_ref.kind != "activation_intervention":
                raise ArtifactError(
                    "investigation references an artifact that is not an activation intervention",
                    details={"artifact_id": artifact_id, "kind": intervention_ref.kind},
                )
        for continuation in payload.get("continuations") or []:
            artifact_id = continuation.get("artifact_id")
            if artifact_id is None:
                continue
            continuation_ref, _path = self.workspace.artifacts.get(artifact_id)
            if continuation_ref.kind != "continuation_comparison":
                raise ArtifactError(
                    "investigation references an artifact that is not a continuation comparison",
                    details={"artifact_id": artifact_id, "kind": continuation_ref.kind},
                )

    def save(self, case: FailureCase) -> tuple[Path, str]:
        if not CASE_ID_PATTERN.match(case.case_id):
            raise ArtifactError("invalid case identifier", details={"case_id": case.case_id})
        self.validate_references(case)
        payload = case_bytes(case)
        enforce_bytes(len(payload), self.budget.max_artifact_bytes, what="failure case")
        directory = self._directory_for(case.case_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "case.json"
        existing_sha = None
        if path.exists():
            existing_sha = sha256_bytes(path.read_bytes())
            if existing_sha == sha256_bytes(payload):
                return path, existing_sha
        atomic_write_bytes(path, payload, overwrite=True)
        return path, sha256_bytes(payload)

    def load(self, case_id: str) -> FailureCase:
        path = self._directory_for(case_id) / "case.json"
        if not path.exists():
            raise NotFoundError(f"failure case {case_id} was not found")
        try:
            case = FailureCase.model_validate(read_json(path))
        except Exception as exc:
            raise ArtifactError(f"malformed failure case {case_id}: {exc}") from exc
        if case.case_id != case_id:
            raise ArtifactError("case filename does not match its identifier")
        return case

    def list(
        self,
        *,
        status: str | None = None,
        tag: str | None = None,
        limit: int = 100,
    ) -> list[CaseSummary]:
        summaries: list[CaseSummary] = []
        for path in sorted(self.directory.glob("case_*/case.json")):
            case = self.load(path.parent.name)
            if status is not None and case.review.status != status:
                continue
            if tag is not None and tag not in case.tags:
                continue
            summaries.append(
                CaseSummary(
                    case_id=case.case_id,
                    title=case.title,
                    status=case.review.status,
                    created_at=case.created_at,
                    updated_at=case.updated_at,
                    revision=case.revision,
                    observation_count=len(case.observations),
                    control_count=len(case.controls),
                    finding_count=len(case.findings),
                    hypothesis_count=len(case.hypotheses),
                    tags=case.tags,
                    model_name=case.model.name,
                )
            )
        summaries.sort(key=lambda item: item.updated_at, reverse=True)
        return summaries[:limit]


# --------------------------------------------------------------------------- #
# trace redaction (standalone export path)
# --------------------------------------------------------------------------- #


def redact_trace(trace: GenerationTrace) -> GenerationTrace:
    """Return a redacted trace for a standalone trace export.

    Kept for the ``export_trace`` tool, which writes one trace and not a case.
    The transformation is the same one the case exporter uses; there is only one
    redaction implementation.
    """

    policy = ExportPolicy(mode="redacted")
    removed = RemovedText()
    payload = redact_trace_payload(
        trace.model_dump(mode="json"),
        policy,
        removed,
        keep_artifact_refs=False,
    )
    return GenerationTrace.model_validate(payload)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def _artifact_entry(
    rewrite: ArtifactRewrite,
    *,
    removed: RemovedText,
) -> dict[str, Any] | None:
    if rewrite.ref is None or rewrite.data is None:
        return None
    entry: dict[str, Any] = {
        "artifact_id": write_id(rewrite.data),
        "kind": rewrite.kind,
        "format": rewrite.format,
        "sha256": rewrite.ref["sha256"],
        "nbytes": rewrite.ref["nbytes"],
        "data_base64": base64_payload(rewrite.data),
    }
    if rewrite.old_id != entry["artifact_id"]:
        entry["old_artifact_id"] = rewrite.old_id
    if rewrite.format == "json":
        try:
            entry["payload"] = json.loads(rewrite.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
            raise SafetyError("a transformed JSON artifact is not valid JSON") from exc
    return entry


def write_id(data: bytes) -> str:
    return content_id(data)


def _patch_payload_scrub(
    payload: dict[str, Any],
    policy: ExportPolicy,
    removed: RemovedText,
) -> dict[str, Any]:
    """Scrub a weight-patch payload without changing its identifier.

    ``patch_id`` is derived from the source fingerprint, target, selection, and
    operation parameters, not from provenance, so removing author notes is safe.
    """

    if not policy.redacted:
        return payload
    payload = dict(payload)
    payload["provenance"] = scrub_provenance(payload.get("provenance"), removed)
    source = payload.get("source")
    if isinstance(source, dict):
        source = dict(source)
        source["display_name"] = safe_display_name(source.get("display_name"), removed)
        source["metadata"] = removed.scrub_soft(source.get("metadata"))
        payload["source"] = source
    # ``target`` (a checkpoint tensor name) is structural and must survive:
    # the patch cannot be reapplied without it.
    return payload


def _patch_set_payload_scrub(
    payload: dict[str, Any],
    policy: ExportPolicy,
    removed: RemovedText,
) -> dict[str, Any]:
    if not policy.redacted:
        return payload
    payload = dict(payload)
    removed.take(payload.get("name"))
    payload["name"] = None
    removed.take(payload.get("notes"))
    payload["notes"] = None
    removed.take(payload.get("created_by"))
    payload["created_by"] = None
    if isinstance(payload.get("metadata"), dict):
        payload["metadata"] = removed.scrub_soft(payload["metadata"])
    return payload


def _investigation_artifact_ids(
    artifacts: Any,
    case_payload: dict[str, Any],
    *,
    policy: ExportPolicy,
    budget: Budget,
) -> set[str]:
    """Collect the artifacts reachable from a case's investigation evidence.

    In private mode the whole referenced graph is included, so the exported case
    is complete.  In redacted mode only capture, comparison, intervention, and
    continuation artifacts are followed: replay results, token alignments, and
    span resolutions embed recorded token text and are deliberately excluded
    from a redacted export.
    """

    found: set[str] = set()
    limit = int(getattr(budget, "max_case_artifacts", 2048))
    pending: list[str] = []
    for link in case_payload.get("investigations") or []:
        if not isinstance(link, dict):
            continue
        evidence_id = link.get("artifact_id")
        if isinstance(evidence_id, str) and evidence_id.startswith("a_"):
            pending.append(evidence_id)
            found.add(evidence_id)

    if not policy.redacted:
        # Full fidelity: follow the whole referenced graph (capture records
        # reference their retained vectors, alignments, and replay results).
        while pending:
            artifact_id = pending.pop()
            try:
                payload = artifacts.read_json(artifact_id)
            except Exception:
                continue
            for nested in collect_artifact_ids(payload):
                if nested not in found:
                    found.add(nested)
                    pending.append(nested)
            if len(found) > limit:
                raise ResourceLimitError(
                    "the investigation evidence graph is too large to export in one case",
                    details={"artifacts": len(found), "limit": limit},
                )
        return found

    for artifact_id in list(found):
        try:
            evidence = artifacts.read_json(artifact_id)
        except Exception:
            continue
        if not isinstance(evidence, dict):
            continue
        for key in (
            "captures",
            "comparisons",
            "interventions",
            "continuations",
            "plans",
            "event_logs",
            "temporal_runs",
            "compositions",
            "staged_continuations",
        ):
            for reference in evidence.get(key) or []:
                if not isinstance(reference, dict):
                    continue
                nested = reference.get("artifact_id")
                if isinstance(nested, str) and nested.startswith("a_"):
                    found.add(nested)

    # Retained activation vectors cannot be redacted, so they are dropped.  Add
    # them anyway so the export records the omission explicitly instead of
    # silently losing the reference.
    for artifact_id in list(found):
        try:
            record = artifacts.read_json(artifact_id)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        for measurement in record.get("measurements") or []:
            if not isinstance(measurement, dict):
                continue
            vector_id = measurement.get("value_artifact_id")
            if isinstance(vector_id, str) and vector_id.startswith("a_"):
                found.add(vector_id)
    return found


def build_export(
    case: FailureCase,
    *,
    traces: TraceStore,
    artifacts: Any = None,
    policy: ExportPolicy | None = None,
    include_traces: bool = True,
    redact_text: bool | None = None,
    budget: Budget | None = None,
) -> dict[str, Any]:
    budget = budget or traces.budget
    artifacts = artifacts if artifacts is not None else traces.workspace.artifacts
    if policy is None:
        policy = ExportPolicy(mode="redacted" if redact_text else "private")
    elif redact_text is not None and bool(redact_text) != policy.redacted:
        raise InputError(
            "redact_text and policy.mode disagree",
            details={"redact_text": redact_text, "mode": policy.mode},
        )
    include_traces = include_traces and policy.include_traces

    warnings: list[str] = []
    removed = RemovedText()
    limitations: list[str] = []
    checksums_dropped: list[str] = []

    # Walk the *original* case payload: after redaction the link ids have already
    # been rewritten to ids that only exist inside this export.
    investigation_ids = _investigation_artifact_ids(
        artifacts,
        case.model_dump(mode="json"),
        policy=policy,
        budget=budget,
    )

    if policy.redacted:
        redacted: RedactedCase = redact_case(
            case.model_dump(mode="json"),
            policy=policy,
            artifacts=artifacts,
            extra_artifact_ids=investigation_ids,
        )
        case_payload = redacted.payload
        removed = redacted.removed
        limitations = list(redacted.limitations)
        rewrites: dict[str, ArtifactRewrite] = {
            rewrite.old_id: rewrite for rewrite in redacted.artifacts
        }
    else:
        case_payload = case.model_dump(mode="json")
        rewrites = {}

    embedded_traces: list[dict[str, Any]] = []
    if include_traces:
        for trace_id in case.observations + case.controls:
            record = traces.get_record(trace_id)
            trace = traces.load(trace_id)
            payload = trace.model_dump(mode="json")
            if policy.redacted:
                checksums_dropped.extend(
                    f"trace:{trace_id}:{name}" for name in dropped_trace_checksums(payload)
                )
                payload = redact_trace_payload(
                    payload,
                    policy,
                    removed,
                    keep_artifact_refs=False,
                )
                data = canonical_bytes(payload)
                artifact_id = content_id(data)
                artifact_sha = sha256_bytes(data)
            else:
                artifact_id = record.artifact.artifact_id
                artifact_sha = record.artifact.sha256
            embedded_traces.append(
                {
                    "trace_id": trace_id,
                    "artifact_id": artifact_id,
                    "artifact_sha256": artifact_sha,
                    "redacted": policy.redacted,
                    "trace": payload,
                }
            )
    else:
        warnings.append("trace payloads were not included")

    patch_payloads: list[dict[str, Any]] = []
    patch_artifact_ids: set[str] = set()
    for reference in case.patches:
        if reference.patch_id is not None:
            patch = traces.workspace.load_patch(reference.patch_id)
            payload = patch.model_dump(mode="json")
            patch_artifact_ids.update(collect_artifact_ids(payload))
            patch_payloads.append(
                {
                    "kind": "patch",
                    "patch_id": patch.patch_id,
                    "payload": _patch_payload_scrub(payload, policy, removed),
                    "redacted": policy.redacted,
                }
            )
        elif reference.patchset_id is not None:
            patchset = traces.workspace.load_patchset(reference.patchset_id)
            payload = patchset.model_dump(mode="json")
            patch_artifact_ids.update(collect_artifact_ids(payload))
            patch_payloads.append(
                {
                    "kind": "patchset",
                    "patchset_id": patchset.patchset_id,
                    "payload": _patch_set_payload_scrub(payload, policy, removed),
                    "redacted": policy.redacted,
                }
            )

    needed_ids = (
        collect_artifact_ids(case.model_dump(mode="json"))
        | patch_artifact_ids
        | investigation_ids
    )
    missing = sorted(needed_ids - set(rewrites))
    if missing:
        if policy.redacted:
            rewrites.update(
                rewrite_artifact_ids(
                    missing,
                    policy=policy,
                    artifacts=artifacts,
                    removed=removed,
                    limitations=limitations,
                )
            )
        else:
            for artifact_id in missing:
                ref, path = artifacts.get(artifact_id)
                data = path.read_bytes()
                rewrites[artifact_id] = ArtifactRewrite(
                    old_id=artifact_id,
                    ref={
                        "artifact_id": ref.artifact_id,
                        "kind": ref.kind,
                        "format": ref.format,
                        "sha256": ref.sha256,
                        "nbytes": ref.nbytes,
                        "path": ref.path,
                    },
                    kind=ref.kind,
                    format=ref.format,
                    data=data,
                )

    entries: dict[str, dict[str, Any]] = {}
    for old_id, rewrite in rewrites.items():
        entry = _artifact_entry(rewrite, removed=removed)
        if entry is not None:
            entries[old_id] = entry

    def entries_for(artifact_ids: Iterable[str]) -> list[dict[str, Any]]:
        return [entries[item] for item in artifact_ids if item in entries]

    contrast_ids = [
        reference.artifact_id
        for reference in case.contrasts
        if reference.artifact_id is not None
    ]
    activation_ids = [artifact.artifact_id for artifact in case.activation_captures]
    contrast_entries = entries_for(contrast_ids)
    activation_entries = entries_for(activation_ids)

    if policy.redacted:
        case_payload["redaction"]["checksums_dropped"] = sorted(set(checksums_dropped))
        limitations.extend(
            f"linked artifact {item.old_id} was omitted" for item in rewrites.values() if item.omitted
        )

    exported_case_sha = sha256_bytes(canonical_bytes(case_payload))

    artifact_map = (
        {
            old_id: (rewrite.ref["artifact_id"] if rewrite.ref else None)
            for old_id, rewrite in sorted(rewrites.items())
        }
        if policy.redacted
        else {}
    )

    reference_traces = [
        item["trace_id"] for item in embedded_traces
    ] or (case.observations + case.controls)

    bundle: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA,
        "exported_at": utc_now_iso(),
        "case_sha256": exported_case_sha,
        "case": case_payload,
        "privacy": {
            "mode": policy.mode,
            "policy": _policy_payload(policy),
            "limitations": limitations,
            "removed_fields": len(removed),
            "removed_strings": len(removed.values),
            "checksums_dropped": sorted(set(checksums_dropped)),
            "replay_capability": (
                ("reduced" if policy.keep_token_ids else "unavailable")
                if policy.redacted
                else "full"
            ),
            "leak_scan": "clean",
        },
        "traces": embedded_traces,
        "contrast_results": contrast_entries,
        "activation_captures": activation_entries,
        "patches": patch_payloads,
        "payload_artifacts": entries_for(sorted(patch_artifact_ids)),
        "investigation_artifacts": entries_for(sorted(investigation_ids)),
        "artifact_map": artifact_map,
        "references": {
            "traces": reference_traces,
            "contrast_artifact_ids": [item["artifact_id"] for item in contrast_entries],
            "activation_capture_artifact_ids": [
                item["artifact_id"] for item in activation_entries
            ],
            "patches": list(case_payload.get("patches") or []),
            "investigation_artifact_ids": [
                item["artifact_id"] for item in entries_for(sorted(investigation_ids))
            ],
        },
        "warnings": warnings,
    }
    if policy.redacted:
        bundle["warnings"] = warnings + [
            "this is a redacted export: text, notes, and measurement payloads were removed",
            "replay capability is reduced; see privacy.limitations",
        ]

    encoded = dumps(to_jsonable(bundle))
    if policy.redacted:
        # A private export intentionally retains the original text, including any
        # string the caller declared sensitive.  Only the redacted mode makes a
        # no-leak guarantee, so only that mode is scanned.
        enforce_no_leaks(
            encoded,
            list(removed.values),
            markers=list(policy.sensitive_markers),
            what=f"{policy.mode} case export",
        )
    enforce_bytes(len(encoded), budget.max_artifact_bytes, what="case export")
    return bundle


def _policy_payload(policy: ExportPolicy) -> dict[str, Any]:
    payload = policy.model_dump(mode="json")
    payload.pop("sensitive_markers", None)
    return payload


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #


def _decode_entry(entry: dict[str, Any], what: str) -> tuple[str, bytes, str]:
    if isinstance(entry.get("data_base64"), str):
        try:
            data = base64.b64decode(entry["data_base64"], validate=True)
        except Exception as exc:
            raise InputError(f"{what} is not valid base64: {exc}") from exc
        return str(entry.get("kind", what)), data, str(entry.get("format", "bin"))
    if "payload" in entry:
        payload = entry["payload"]
        data = canonical_bytes(payload)
        return str(entry.get("kind", what)), data, "json"
    raise InputError(f"case export contains a malformed {what}")


def _import_artifacts(
    payload: dict[str, Any],
    *,
    traces: TraceStore,
    warnings: list[str],
    check_ids: bool,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for section, default_kind in (
        ("contrast_results", "contrast_result"),
        ("activation_captures", "activation_capture"),
        ("payload_artifacts", "patch_payload"),
        ("investigation_artifacts", "investigation_evidence"),
    ):
        for entry in payload.get(section, []) or []:
            if not isinstance(entry, dict):
                raise InputError(f"case export contains a malformed {section} entry")
            kind, data, format_name = _decode_entry(entry, default_kind)
            ref = traces.workspace.artifacts.put_bytes(kind, data, format=format_name)
            recorded = entry.get("artifact_id")
            if recorded not in (None, ref.artifact_id):
                if not check_ids:
                    warnings.append(
                        f"{section} entry was re-identified on import "
                        f"({recorded} -> {ref.artifact_id})"
                    )
                else:
                    raise ArtifactError(
                        f"{section} content hash does not match its exported identifier",
                        details={"recorded": recorded, "actual": ref.artifact_id},
                    )
            if entry.get("sha256") not in (None, ref.sha256):
                raise ArtifactError(f"{section} checksum does not match the export")
            old_id = entry.get("old_artifact_id")
            if isinstance(old_id, str):
                mapping[old_id] = ref.artifact_id
    return mapping


def import_export(
    path: str | Path,
    *,
    traces: TraceStore,
    cases: CaseStore,
    overwrite: bool = False,
    budget: Budget | None = None,
) -> tuple[FailureCase, list[str]]:
    source = Path(path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise NotFoundError(f"case export was not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"could not read case export: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") not in SUPPORTED_EXPORT_SCHEMAS:
        raise InputError("file is not a supported case export")
    schema = payload["schema_version"]
    modern = schema == EXPORT_SCHEMA
    case_payload = payload.get("case")
    recorded_sha = payload.get("case_sha256")
    if not isinstance(case_payload, dict) or not isinstance(recorded_sha, str):
        raise InputError("case export is missing its case payload or checksum")
    if sha256_bytes(canonical_bytes(case_payload)) != recorded_sha:
        raise ArtifactError("case payload checksum does not match the export")
    try:
        case = FailureCase.model_validate(case_payload)
    except Exception as exc:
        raise InputError(f"exported case is not valid: {exc}") from exc
    warnings: list[str] = []

    for entry in payload.get("traces", []) or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("trace"), dict):
            raise InputError("case export contains a malformed trace entry")
        try:
            trace = GenerationTrace.model_validate(entry["trace"])
        except Exception as exc:
            raise InputError(f"exported trace is not valid: {exc}") from exc
        if trace.trace_id != entry.get("trace_id"):
            raise ArtifactError("exported trace identifier does not match its entry")
        expected = entry.get("artifact_sha256")
        actual = sha256_bytes(canonical_bytes(trace.model_dump(mode="json")))
        if modern and expected and expected != actual:
            raise ArtifactError(
                "exported trace checksum does not match its payload",
                details={"trace_id": trace.trace_id},
            )
        if not modern and not entry.get("redacted") and expected and expected != actual:
            raise ArtifactError(
                "exported trace checksum does not match its payload",
                details={"trace_id": trace.trace_id},
            )
        if entry.get("redacted"):
            warnings.append(
                f"trace {trace.trace_id} was redacted; replay capability is reduced"
            )
        existing = None
        try:
            existing = traces.get_record(trace.trace_id)
        except NotFoundError:
            existing = None
        if existing is not None:
            current = traces.load(trace.trace_id)
            if canonical_bytes(current.model_dump(mode="json")) != canonical_bytes(
                trace.model_dump(mode="json")
            ):
                if entry.get("redacted"):
                    warnings.append(
                        f"workspace already holds trace {trace.trace_id}; "
                        "the redacted copy was not imported and the existing trace is kept"
                    )
                    continue
                raise ArtifactError(
                    "workspace already contains a different trace with this identifier",
                    details={"trace_id": trace.trace_id},
                )
            continue
        traces.save(trace)

    _import_artifacts(
        payload,
        traces=traces,
        warnings=warnings,
        check_ids=modern,
    )

    reference_map = payload.get("artifact_map")
    if isinstance(reference_map, dict):
        for old_id, new_id in reference_map.items():
            if new_id is None:
                continue
            try:
                traces.workspace.artifacts.get(new_id)
            except NotFoundError as exc:
                raise ArtifactError(
                    "exported artifact map points at content that is not in the archive",
                    details={"artifact_id": new_id},
                ) from exc

    for entry in payload.get("patches", []) or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
            raise InputError("case export contains a malformed patch reference")
        if entry.get("kind") == "patch":
            patch = WeightPatch.model_validate(entry["payload"])
            if entry.get("patch_id") not in (None, patch.patch_id):
                raise ArtifactError("patch identifier does not match its payload")
            try:
                traces.workspace.save_patch(patch)
            except ArtifactError:
                warnings.append(
                    f"workspace already holds patch {patch.patch_id} with different metadata; "
                    "the existing record is kept"
                )
        elif entry.get("kind") == "patchset":
            patchset = PatchSet.model_validate(entry["payload"])
            if entry.get("patchset_id") not in (None, patchset.patchset_id):
                raise ArtifactError("patch set identifier does not match its payload")
            try:
                traces.workspace.save_patchset(patchset)
            except ArtifactError:
                warnings.append(
                    f"workspace already holds patch set {patchset.patchset_id} with "
                    "different metadata; the existing record is kept"
                )
        else:
            raise InputError("unknown patch entry kind in case export")

    existing_case = None
    try:
        existing_case = cases.load(case.case_id)
    except NotFoundError:
        existing_case = None
    if existing_case is not None:
        if case_bytes(existing_case) != case_bytes(case):
            if not overwrite:
                raise ArtifactError(
                    "workspace already contains a different case with this identifier",
                    details={"case_id": case.case_id, "hint": "set overwrite=true"},
                )
            cases.save(case)
            warnings.append(f"case {case.case_id} was overwritten")
        else:
            warnings.append(f"case {case.case_id} was already present and identical")
    else:
        cases.save(case)
    if budget is not None:
        enforce_bytes(len(dumps(to_jsonable(payload))), budget.max_artifact_bytes, what="case import")
    return case, warnings
