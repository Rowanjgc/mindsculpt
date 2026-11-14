"""Persistent failure-case registry and self-contained export bundles."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from wsp.behavior.ids import new_case_id
from wsp.behavior.models import (
    CaseSummary,
    FailureCase,
    GenerationTrace,
)
from wsp.behavior.trace import TraceStore
from wsp.core.budget import Budget, enforce_bytes
from wsp.core.errors import ArtifactError, InputError, NotFoundError, SafetyError
from wsp.core.hashing import sha256_bytes
from wsp.core.json import atomic_write_bytes, canonical_bytes, dumps, read_json, to_jsonable
from wsp.core.time import utc_now_iso
from wsp.patches.spec import PatchSet, WeightPatch
from wsp.workspace.store import Workspace

CASE_ID_PATTERN = re.compile(r"^case_[0-9a-f]{16}$")
EXPORT_SCHEMA = "wsp.case-export/1.0"


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
        for artifact in case.activation_captures:
            self.workspace.artifacts.verify(artifact)
        for patch_reference in case.patches:
            if patch_reference.patch_id is not None:
                self.workspace.load_patch(patch_reference.patch_id)
            if patch_reference.patchset_id is not None:
                self.workspace.load_patchset(patch_reference.patchset_id)

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


def redact_trace(trace: GenerationTrace) -> GenerationTrace:
    stripped = trace.model_copy(deep=True)
    stripped.input.text = None
    stripped.input.messages = []
    stripped.input.sha256 = None
    stripped.text = None
    if stripped.prompt is not None:
        stripped.prompt.rendered = None
    for token in stripped.tokens:
        token.text = None
    for call in stripped.tool_calls:
        call.arguments = None
    stripped.model.checkpoint_path = None
    stripped.provenance.source_ref = None
    stripped.privacy.store_input_text = False
    stripped.privacy.store_output_text = False
    stripped.privacy.redacted = True
    stripped.unavailable.setdefault("input_text", "removed by export redaction")
    stripped.unavailable.setdefault("output_text", "removed by export redaction")
    return stripped


def _collect_artifact_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        artifact_id = value.get("artifact_id")
        if isinstance(artifact_id, str) and artifact_id.startswith("a_"):
            found.add(artifact_id)
        for item in value.values():
            found.update(_collect_artifact_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_collect_artifact_ids(item))
    return found


def build_export(
    case: FailureCase,
    *,
    traces: TraceStore,
    artifacts: Any,
    include_traces: bool = True,
    redact_text: bool = False,
    budget: Budget | None = None,
) -> dict[str, Any]:
    budget = budget or traces.budget
    warnings: list[str] = []
    embedded_traces: list[dict[str, Any]] = []
    if include_traces:
        for trace_id in case.observations + case.controls:
            record = traces.get_record(trace_id)
            trace = traces.load(trace_id)
            payload = trace.model_dump(mode="json")
            if redact_text:
                payload = redact_trace(trace).model_dump(mode="json")
            embedded_traces.append(
                {
                    "trace_id": trace_id,
                    "artifact_id": record.artifact.artifact_id,
                    "artifact_sha256": record.artifact.sha256,
                    "redacted": redact_text,
                    "trace": payload,
                }
            )
    else:
        warnings.append("trace payloads were not included")
    if redact_text:
        warnings.append(
            "case titles, summaries, annotations, and hypotheses are human-authored and were not redacted"
        )
        warnings.append(
            "case model/reproduction metadata is reduced, but embedded contrast results "
            "retain their exact inputs and outputs; review them before sharing"
        )

    case_payload = case.model_dump(mode="json")
    if redact_text:
        if isinstance(case_payload.get("model"), dict):
            case_payload["model"]["checkpoint_path"] = None
        if isinstance(case_payload.get("reproduction"), dict):
            case_payload["reproduction"]["environment"] = {}
    exported_case_sha = sha256_bytes(canonical_bytes(case_payload))

    contrasts: list[dict[str, Any]] = []
    for reference in case.contrasts:
        if reference.artifact_id is None:
            continue
        ref, _path = artifacts.get(reference.artifact_id)
        contrasts.append(
            {
                "artifact_id": ref.artifact_id,
                "sha256": ref.sha256,
                "nbytes": ref.nbytes,
                "payload": artifacts.read_json(ref.artifact_id),
            }
        )

    activation_payloads: list[dict[str, Any]] = []
    for artifact in case.activation_captures:
        ref, path = artifacts.get(artifact.artifact_id)
        data = path.read_bytes()
        activation_payloads.append(
            {
                "artifact_id": ref.artifact_id,
                "kind": ref.kind,
                "format": ref.format,
                "sha256": ref.sha256,
                "nbytes": ref.nbytes,
                "data_base64": base64.b64encode(data).decode("ascii"),
            }
        )
    patch_payloads: list[dict[str, Any]] = []
    patch_artifact_ids: set[str] = set()
    for reference in case.patches:
        if reference.patch_id is not None:
            patch = traces.workspace.load_patch(reference.patch_id)
            payload = patch.model_dump(mode="json")
            patch_artifact_ids.update(_collect_artifact_ids(payload))
            patch_payloads.append(
                {"kind": "patch", "patch_id": patch.patch_id, "payload": payload}
            )
        elif reference.patchset_id is not None:
            patchset = traces.workspace.load_patchset(reference.patchset_id)
            payload = patchset.model_dump(mode="json")
            patch_artifact_ids.update(_collect_artifact_ids(payload))
            patch_payloads.append(
                {
                    "kind": "patchset",
                    "patchset_id": patchset.patchset_id,
                    "payload": payload,
                }
            )
    patch_artifacts: list[dict[str, Any]] = []
    for artifact_id in sorted(patch_artifact_ids):
        ref, path = artifacts.get(artifact_id)
        data = path.read_bytes()
        patch_artifacts.append(
            {
                "artifact_id": ref.artifact_id,
                "kind": ref.kind,
                "format": ref.format,
                "sha256": ref.sha256,
                "nbytes": ref.nbytes,
                "data_base64": base64.b64encode(data).decode("ascii"),
            }
        )

    bundle = {
        "schema_version": EXPORT_SCHEMA,
        "exported_at": utc_now_iso(),
        "case_sha256": exported_case_sha,
        "case": case_payload,
        "traces": embedded_traces,
        "contrast_results": contrasts,
        "activation_captures": activation_payloads,
        "patches": patch_payloads,
        "payload_artifacts": patch_artifacts,
        "references": {
            "traces": case.observations + case.controls,
            "contrast_artifact_ids": [item["artifact_id"] for item in contrasts],
            "activation_capture_artifact_ids": [
                artifact.artifact_id for artifact in case.activation_captures
            ],
            "patches": [reference.model_dump(mode="json") for reference in case.patches],
        },
        "warnings": warnings,
    }
    encoded = dumps(to_jsonable(bundle))
    enforce_bytes(len(encoded), budget.max_artifact_bytes, what="case export")
    return bundle


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
    if not isinstance(payload, dict) or payload.get("schema_version") != EXPORT_SCHEMA:
        raise InputError("file is not a supported case export")
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
        if not entry.get("redacted"):
            expected = entry.get("artifact_sha256")
            actual = sha256_bytes(canonical_bytes(trace.model_dump(mode="json")))
            if expected and expected != actual:
                raise ArtifactError(
                    "exported trace checksum does not match its payload",
                    details={"trace_id": trace.trace_id},
                )
        else:
            warnings.append(f"trace {trace.trace_id} was redacted in the export")
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
                raise ArtifactError(
                    "workspace already contains a different trace with this identifier",
                    details={"trace_id": trace.trace_id},
                )
            continue
        traces.save(trace)
    for entry in payload.get("contrast_results", []) or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("payload"), (dict, list)):
            raise InputError("case export contains a malformed contrast result")
        ref = traces.workspace.artifacts.put_json("contrast_result", entry["payload"])
        if entry.get("artifact_id") not in (None, ref.artifact_id):
            raise ArtifactError("contrast result content hash does not match its exported identifier")
        if entry.get("sha256") not in (None, ref.sha256):
            raise ArtifactError("contrast result checksum does not match the export")
    for entry in payload.get("activation_captures", []) or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("data_base64"), str):
            raise InputError("case export contains a malformed activation capture")
        try:
            data = base64.b64decode(entry["data_base64"], validate=True)
        except Exception as exc:
            raise InputError(f"activation capture is not valid base64: {exc}") from exc
        ref = traces.workspace.artifacts.put_bytes(
            str(entry.get("kind", "activation_capture")),
            data,
            format=str(entry.get("format", "json")),
        )
        if entry.get("artifact_id") not in (None, ref.artifact_id):
            raise ArtifactError("activation capture content hash does not match its exported identifier")
        if entry.get("sha256") not in (None, ref.sha256):
            raise ArtifactError("activation capture checksum does not match the export")
    for entry in payload.get("payload_artifacts", []) or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("data_base64"), str):
            raise InputError("case export contains a malformed patch payload artifact")
        try:
            data = base64.b64decode(entry["data_base64"], validate=True)
        except Exception as exc:
            raise InputError(f"patch payload artifact is not valid base64: {exc}") from exc
        ref = traces.workspace.artifacts.put_bytes(
            str(entry.get("kind", "patch_payload")),
            data,
            format=str(entry.get("format", "npy")),
        )
        if entry.get("artifact_id") not in (None, ref.artifact_id):
            raise ArtifactError("patch payload content hash does not match its exported identifier")
        if entry.get("sha256") not in (None, ref.sha256):
            raise ArtifactError("patch payload checksum does not match the export")
    for entry in payload.get("patches", []) or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
            raise InputError("case export contains a malformed patch reference")
        if entry.get("kind") == "patch":
            patch = WeightPatch.model_validate(entry["payload"])
            if entry.get("patch_id") not in (None, patch.patch_id):
                raise ArtifactError("patch identifier does not match its payload")
            traces.workspace.save_patch(patch)
        elif entry.get("kind") == "patchset":
            patchset = PatchSet.model_validate(entry["payload"])
            if entry.get("patchset_id") not in (None, patchset.patchset_id):
                raise ArtifactError("patch set identifier does not match its payload")
            traces.workspace.save_patchset(patchset)
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
