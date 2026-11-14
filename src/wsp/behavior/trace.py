"""Trace construction helpers and immutable trace storage."""

from __future__ import annotations

from pathlib import Path

from wsp.behavior.models import (
    GenerationTrace,
    TraceRecord,
    TraceSummary,
)
from wsp.core.budget import Budget, enforce_bytes
from wsp.core.errors import ArtifactError, InputError, NotFoundError, ResourceLimitError, SafetyError
from wsp.core.hashing import sha256_bytes
from wsp.core.json import atomic_write_json, canonical_bytes, read_json
from wsp.workspace.artifacts import ArtifactStore
from wsp.workspace.store import Workspace

TRACE_ID_PATTERN = r"^tr_[0-9a-f]{16}$"
_INDEX_SCHEMA = "wsp.trace-index/1.0"


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return sha256_bytes(value.encode("utf-8"))


def _measure_trace_bytes(trace: GenerationTrace) -> int:
    return len(canonical_bytes(trace.model_dump(mode="json")))


def enforce_trace_limits(trace: GenerationTrace, budget: Budget) -> None:
    if len(trace.tokens) > budget.max_trace_tokens:
        raise ResourceLimitError(
            "trace contains more generated tokens than the configured limit",
            details={
                "tokens": len(trace.tokens),
                "limit": budget.max_trace_tokens,
                "hint": "raise budget.max_trace_tokens or record a shorter generation",
            },
        )
    characters = len(trace.text or "")
    characters += sum(len(token.text or "") for token in trace.tokens)
    characters += len(trace.input.text or "")
    if characters > budget.max_trace_chars:
        raise ResourceLimitError(
            "trace text exceeds the configured character limit",
            details={"characters": characters, "limit": budget.max_trace_chars},
        )
    nbytes = _measure_trace_bytes(trace)
    if nbytes > budget.max_trace_bytes:
        raise ResourceLimitError(
            "trace exceeds the configured byte limit",
            details={"bytes": nbytes, "limit": budget.max_trace_bytes},
        )


def apply_privacy(trace: GenerationTrace) -> GenerationTrace:
    """Strip text that the caller asked not to persist.

    Token text can reveal output content, so it is stripped together with the
    output text.  The number of tokens and their identities/ranges remain,
    which is enough for structural detectors.
    """

    privacy = trace.privacy
    if not privacy.store_input_text:
        if trace.input.text is not None or trace.input.messages:
            privacy.redacted = True
        trace.input.text = None
        trace.input.messages = []
        if trace.prompt is not None:
            trace.prompt.rendered = None
        trace.unavailable.setdefault(
            "input_text", "not persisted (privacy.store_input_text=false)"
        )
    if not privacy.store_output_text:
        if trace.text is not None or any(token.text for token in trace.tokens):
            privacy.redacted = True
        trace.text = None
        for token in trace.tokens:
            token.text = None
        trace.unavailable.setdefault(
            "output_text", "not persisted (privacy.store_output_text=false)"
        )
    return trace


def finalize_trace(trace: GenerationTrace, budget: Budget) -> GenerationTrace:
    """Apply privacy, measure what is measurable, and enforce bounds.

    This is intentionally the only place where checksums are filled in.  A
    backend cannot accidentally claim a checksum for content it did not
    produce.
    """

    apply_privacy(trace)
    if trace.tokens and trace.termination.generated_tokens is None:
        trace.termination.generated_tokens = len(trace.tokens)
    trace.input.sha256 = (
        trace.input.sha256
        if trace.input.artifact_id is not None
        else sha256_text(trace.input.text)
    )
    if trace.input.token_ids:
        trace.input.tokens_sha256 = sha256_bytes(
            canonical_bytes([int(value) for value in trace.input.token_ids])
        )
    else:
        trace.input.tokens_sha256 = None
    if trace.text is not None:
        trace.checksums["output_sha256"] = sha256_text(trace.text) or ""
    else:
        trace.checksums.pop("output_sha256", None)
    trace.checksums["tokens_sha256"] = sha256_bytes(
        canonical_bytes([token.model_dump(mode="json") for token in trace.tokens])
    )
    enforce_trace_limits(trace, budget)
    return trace


class TraceStore:
    """Immutable traces plus a small mutable index.

    The trace itself lives in the content-addressed artifact store, so it is
    checksum-verified and cannot be silently rewritten.  The index is a
    per-workspace pointer that maps a readable ``tr_`` identifier to that
    artifact.
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.budget = workspace.budget
        self.artifacts: ArtifactStore = workspace.artifacts
        self.directory = workspace.root / "traces"
        self.directory.mkdir(parents=True, exist_ok=True)

    def _index_path(self, trace_id: str) -> Path:
        if not trace_id or "/" in trace_id or "\\" in trace_id or trace_id.startswith("."):
            raise InputError("invalid trace identifier", details={"trace_id": trace_id})
        path = (self.directory / f"{trace_id}.json").resolve()
        if self.directory.resolve() not in path.parents:
            raise SafetyError("trace path escapes the workspace")
        return path

    def save(self, trace: GenerationTrace) -> TraceRecord:
        enforce_trace_limits(trace, self.budget)
        index_path = self._index_path(trace.trace_id)
        existing = read_json(index_path) if index_path.exists() else None
        if existing is not None:
            record = TraceRecord.model_validate(existing)
            if record.artifact.sha256 != sha256_bytes(
                canonical_bytes(trace.model_dump(mode="json"))
            ):
                raise ArtifactError(
                    "trace identifier is already used by a different trace",
                    details={"trace_id": trace.trace_id},
                )
            return record
        payload = trace.model_dump(mode="json")
        ref = self.artifacts.put_json(
            "generation_trace",
            payload,
            metadata={
                "trace_id": trace.trace_id,
                "model_name": trace.model.name,
                "backend": trace.backend.name,
                "termination_reason": trace.termination.reason,
            },
        )
        enforce_bytes(ref.nbytes, self.budget.max_trace_bytes, what="trace artifact")
        record = TraceRecord(
            trace_id=trace.trace_id,
            artifact=ref,
            created_at=trace.created_at,
            model_name=trace.model.name,
            model_revision=trace.model.revision,
            backend=trace.backend.name,
            token_count=len(trace.tokens),
            text_chars=len(trace.text or ""),
            termination_reason=trace.termination.reason,
            input_sha256=trace.input.sha256,
            provenance_origin=trace.provenance.origin,
        )
        atomic_write_json(index_path, record.model_dump(mode="json"), overwrite=False)
        return record

    def get_record(self, trace_id: str) -> TraceRecord:
        path = self._index_path(trace_id)
        if not path.exists():
            raise NotFoundError(f"trace {trace_id} was not found")
        try:
            record = TraceRecord.model_validate(read_json(path))
        except Exception as exc:
            raise ArtifactError(f"malformed trace index {trace_id}: {exc}") from exc
        if record.trace_id != trace_id:
            raise ArtifactError("trace index filename does not match its identifier")
        return record

    def load(self, trace_id: str) -> GenerationTrace:
        record = self.get_record(trace_id)
        try:
            payload = self.artifacts.read_json(record.artifact.artifact_id)
            trace = GenerationTrace.model_validate(payload)
        except Exception as exc:
            raise ArtifactError(f"malformed trace artifact for {trace_id}: {exc}") from exc
        if trace.trace_id != trace_id:
            raise ArtifactError("trace artifact identifier does not match its index")
        return trace

    def list_records(self, *, limit: int | None = None) -> list[TraceRecord]:
        records: list[TraceRecord] = []
        for path in self.directory.glob("tr_*.json"):
            try:
                records.append(TraceRecord.model_validate(read_json(path)))
            except Exception as exc:
                raise ArtifactError(f"malformed trace index {path.name}: {exc}") from exc
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records if limit is None else records[:limit]

    def list(self, *, limit: int = 100) -> list[TraceSummary]:
        return [summarize(record) for record in self.list_records(limit=limit)]


def summarize(record: TraceRecord) -> TraceSummary:
    return TraceSummary(
        trace_id=record.trace_id,
        created_at=record.created_at,
        model_name=record.model_name,
        model_revision=record.model_revision,
        backend=record.backend,
        token_count=record.token_count,
        text_chars=record.text_chars,
        termination_reason=record.termination_reason,
        provenance_origin=record.provenance_origin,
        artifact_id=record.artifact.artifact_id,
        artifact_sha256=record.artifact.sha256,
    )


def trace_summary_from_trace(trace: GenerationTrace, record: TraceRecord) -> TraceSummary:
    summary = summarize(record)
    summary.model_revision = trace.model.revision
    summary.provenance_origin = trace.provenance.origin
    return summary
