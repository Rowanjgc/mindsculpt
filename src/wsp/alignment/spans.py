"""Resolve behavioral spans onto token positions.

Detector evidence in V1.1 uses three shapes, and this module handles them
without redefining what they mean:

``token_range``
    generation-relative token indices, ``end`` exclusive;
``text_span``
    character offsets into the recorded output text;
``word_range``
    character offsets into the recorded output text, emitted by the repetition
    detector for word units.  Despite the name, the stored values are character
    offsets, so they are interpreted that way and the interpretation is reported.

Nothing here guesses: when text is missing, offsets are unavailable, or a span
does not line up with token boundaries, the result says so.
"""

from __future__ import annotations

from typing import Any, Literal

from wsp.alignment.ids import new_span_id
from wsp.alignment.models import (
    PREDICTION_NOTE,
    AlignmentRole,
    ContextMode,
    SpanResolution,
    TokenAlignment,
)
from wsp.core.errors import InputError, ResourceLimitError
from wsp.core.time import utc_now_iso

SpanKind = Literal["token_range", "text_span", "word_range", "finding_evidence"]


def _tokens_in_char_range(
    alignment: TokenAlignment, role: AlignmentRole, start: int, end: int
) -> tuple[list[int], list[str], float]:
    """Return ``(full_indices, ambiguity_notes, coverage_fraction)``."""

    positions = alignment.for_role(role)
    selected: list[int] = []
    notes: list[str] = []
    covered = 0
    for position in positions:
        if position.char_start is None or position.char_end is None:
            continue
        token_start, token_end = position.char_start, position.char_end
        overlap = min(token_end, end) - max(token_start, start)
        if overlap <= 0:
            continue
        selected.append(position.full_index)
        covered += overlap
        if token_start < start or token_end > end:
            notes.append(
                f"token at {role} index {position.role_index} extends beyond the requested span"
            )
    span_length = max(0, end - start)
    coverage = (covered / span_length) if span_length else 0.0
    return selected, notes, coverage


def _byte_range_of(alignment: TokenAlignment, full_indices: list[int]) -> dict[str, int] | None:
    """Byte offsets are only meaningful inside one role's text."""

    roles = {alignment.position(index).role for index in full_indices if 0 <= index < alignment.full_length}
    if len(roles) != 1:
        return None
    starts = [
        alignment.position(index).byte_start
        for index in full_indices
        if 0 <= index < alignment.full_length
    ]
    ends = [
        alignment.position(index).byte_end
        for index in full_indices
        if 0 <= index < alignment.full_length
    ]
    known_starts = [value for value in starts if value is not None]
    known_ends = [value for value in ends if value is not None]
    if not known_starts or not known_ends:
        return None
    return {"start": min(known_starts), "end": max(known_ends)}


def resolve_span(
    alignment: TokenAlignment,
    *,
    kind: SpanKind,
    start: int | None = None,
    end: int | None = None,
    role: AlignmentRole | None = None,
    prompt_text: str | None = None,
    generation_text: str | None = None,
    name: str | None = None,
    window: dict[str, int] | None = None,
    detail: dict[str, Any] | None = None,
    budget: Any | None = None,
) -> SpanResolution:
    """Map a requested span onto full-sequence token indices."""

    if window is not None:
        window_start = int(window.get("start_full_index", 0))
        window_end = int(window.get("end_full_index", alignment.full_length))
        if window_start < 0 or window_end > alignment.full_length or window_end <= window_start:
            raise InputError(
                "the requested context window is outside the aligned sequence",
                details={
                    "start_full_index": window_start,
                    "end_full_index": window_end,
                    "full_length": alignment.full_length,
                },
            )

    effective_role: AlignmentRole = role or "generation"
    warnings: list[str] = []
    ambiguity: list[str] = []
    candidates: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    requested_kind: SpanKind = kind

    if kind == "word_range":
        ambiguity.append(
            "evidence kind 'word_range' stores character offsets in V1.1 detector output, "
            "so it was interpreted as character offsets into the output text"
        )

    if kind == "finding_evidence":
        if detail is None or "kind" not in detail:
            raise InputError(
                "finding_evidence requires the original evidence object in 'detail'",
            )
        inner = str(detail["kind"])
        if inner not in {"token_range", "text_span", "word_range"}:
            raise InputError(
                "this evidence kind cannot be mapped to token positions",
                details={"kind": inner},
            )
        metadata["evidence_kind"] = inner
        metadata["evidence_detail"] = detail.get("detail")
        if detail.get("name"):
            name = name or str(detail["name"])
        start = detail.get("start")
        end = detail.get("end")
        kind = inner  # type: ignore[assignment]
        effective_role = role or "generation"
        if inner == "word_range":
            ambiguity.append(
                "evidence kind 'word_range' stores character offsets in V1.1 detector output, "
                "so it was interpreted as character offsets into the output text"
            )

    if start is None or end is None:
        raise InputError("span resolution requires start and end")
    start = int(start)
    end = int(end)
    if start < 0:
        raise InputError("span start must not be negative", details={"start": start})
    if end <= start:
        return SpanResolution(
            resolution_id=new_span_id(),
            created_at=utc_now_iso(),
            trace_id=alignment.trace_id,
            alignment_id=alignment.alignment_id,
            requested_kind=requested_kind,  # type: ignore[arg-type]
            requested_role=role,
            requested_start=start,
            requested_end=end,
            requested_name=name,
            status="unresolved",
            ambiguity=["the requested span is empty"],
            warnings=warnings,
            metadata=metadata,
        )

    full_indices: list[int] = []
    char_range: dict[str, int] | None = None
    byte_range: dict[str, int] | None = None
    selected_text: str | None = None
    coverage = 1.0

    if kind == "token_range":
        segment_length = (
            alignment.prompt_length if effective_role == "prompt" else alignment.generated_length
        )
        if end > segment_length:
            raise InputError(
                "the requested token range is outside the aligned sequence segment",
                details={
                    "role": effective_role,
                    "start": start,
                    "end": end,
                    "segment_length": segment_length,
                },
            )
        for role_index in range(start, end):
            full_indices.append(alignment.to_full_index(effective_role, role_index))
        limit = getattr(budget, "max_alignment_span_tokens", None)
        if limit is not None and len(full_indices) > limit:
            raise ResourceLimitError(
                "the requested span is larger than the alignment span limit",
                details={"tokens": len(full_indices), "limit": limit},
            )
    else:
        text = prompt_text if effective_role == "prompt" else generation_text
        if text is None:
            return SpanResolution(
                resolution_id=new_span_id(),
                created_at=utc_now_iso(),
                trace_id=alignment.trace_id,
                alignment_id=alignment.alignment_id,
                requested_kind=requested_kind,  # type: ignore[arg-type]
                requested_role=effective_role,
                requested_start=start,
                requested_end=end,
                requested_name=name,
                status="unresolved",
                ambiguity=[
                    f"the recorded {effective_role} text is unavailable, so character offsets cannot be mapped"
                ],
                warnings=warnings,
                metadata=metadata,
            )
        if end > len(text):
            raise InputError(
                "the requested character span is outside the recorded text",
                details={"end": end, "text_chars": len(text), "role": effective_role},
            )
        char_range = {"start": start, "end": end}
        byte_range = {
            "start": len(text[:start].encode("utf-8")),
            "end": len(text[:end].encode("utf-8")),
        }
        selected_text = text[start:end]
        offsets_available = all(
            position.char_start is not None for position in alignment.for_role(effective_role)
        )
        if not offsets_available:
            return SpanResolution(
                resolution_id=new_span_id(),
                created_at=utc_now_iso(),
                trace_id=alignment.trace_id,
                alignment_id=alignment.alignment_id,
                requested_kind=requested_kind,  # type: ignore[arg-type]
                requested_role=effective_role,
                requested_start=start,
                requested_end=end,
                requested_name=name,
                status="unresolved",
                char_range=char_range,
                selected_text=selected_text,
                ambiguity=[
                    "character offsets are unavailable for this segment, so the span cannot be mapped to tokens"
                ],
                warnings=warnings,
                metadata=metadata,
            )
        full_indices, boundary_notes, coverage = _tokens_in_char_range(
            alignment, effective_role, start, end
        )
        ambiguity.extend(boundary_notes)
        if not full_indices:
            return SpanResolution(
                resolution_id=new_span_id(),
                created_at=utc_now_iso(),
                trace_id=alignment.trace_id,
                alignment_id=alignment.alignment_id,
                requested_kind=requested_kind,  # type: ignore[arg-type]
                requested_role=effective_role,
                requested_start=start,
                requested_end=end,
                requested_name=name,
                status="unresolved",
                char_range=char_range,
                selected_text=selected_text,
                ambiguity=["no token overlaps the requested character span"],
                warnings=warnings,
                metadata=metadata,
            )
        if coverage < 0.999:
            ambiguity.append(
                f"token offsets cover {coverage:.3f} of the requested character span"
            )

    if window is not None:
        full_indices = [
            index
            for index in full_indices
            if int(window.get("start_full_index", 0)) <= index < int(window.get("end_full_index", alignment.full_length))
        ]
        if not full_indices:
            ambiguity.append("no part of the resolved span falls inside the requested window")

    consuming = alignment.consuming_positions_for(full_indices)
    predicting = alignment.predicting_positions_for(full_indices)
    if predicting and predicting[0] < 0:  # pragma: no cover - defensive
        predicting = [index for index in predicting if index >= 0]
    if 0 in full_indices:
        ambiguity.append(
            "the first token of the sequence has no predicting position: no earlier state can predict it"
        )
    context_mode: ContextMode = "full-sequence" if window is None else "explicit-window"
    if window is not None and int(window.get("start_full_index", 0)) > 0:
        warnings.append(
            "a window that does not start at position zero does not preserve the causal prefix; "
            "activations measured inside it are a different-context approximation"
        )

    roles: list[AlignmentRole] = []
    role_ranges: list[dict[str, Any]] = []
    by_role: dict[str, list[int]] = {}
    for full_index in full_indices:
        role_name, role_index = alignment.to_role_index(full_index)
        by_role.setdefault(role_name, []).append(role_index)
        if role_name not in roles:
            roles.append(role_name)
    for role_name, indices in by_role.items():
        role_ranges.append(
            {
                "role": role_name,
                "first_index": min(indices),
                "last_index": max(indices),
                "token_count": len(indices),
            }
        )

    status: Literal["resolved", "ambiguous", "unresolved"] = "resolved"
    if ambiguity:
        status = "ambiguous"

    return SpanResolution(
        resolution_id=new_span_id(),
        created_at=utc_now_iso(),
        trace_id=alignment.trace_id,
        alignment_id=alignment.alignment_id,
        requested_kind=requested_kind,  # type: ignore[arg-type]
        requested_role=effective_role,
        requested_start=start,
        requested_end=end,
        requested_name=name,
        status=status,
        roles=roles,
        role_ranges=role_ranges,
        full_token_indices=full_indices,
        consuming_positions=consuming,
        predicting_positions=predicting,
        char_range=char_range,
        byte_range=byte_range if byte_range is not None else _byte_range_of(alignment, full_indices),
        selected_text=selected_text,
        context_mode=context_mode,
        context_window=window,
        context_note=PREDICTION_NOTE,
        ambiguity=ambiguity,
        candidates=candidates,
        warnings=warnings,
        metadata=metadata,
    )


__all__ = ["SpanKind", "resolve_span"]
