"""Deterministic behavioral events.

An event records something that was observed in a generation stream: a token
occurred, an n-gram repeated, a detector produced a finding, or an investigator
marked a position.  Events are measurements, not diagnoses.  Nothing in this
module attaches a mechanism, a fault label, or a repair claim to an event.

All token positions are absolute indices into the full sequence (prompt tokens
first, then generated tokens).  The generation step that produced an absolute
position ``p`` is ``p - prompt_length``; an event detected at that position
cannot affect that forward, so its earliest online intervention is normally
``p - prompt_length + 1``.
"""

from __future__ import annotations

import secrets
from typing import Any, Literal, Sequence

from pydantic import Field, model_validator

from wsp.behavior.models import GenerationTrace
from wsp.core.budget import Budget
from wsp.core.errors import InputError, ResourceLimitError
from wsp.core.hashing import sha256_bytes
from wsp.core.time import utc_now_iso
from wsp.investigation.models import StrictModel

SCHEMA_EVENT = "wsp.behavioral-event/1.0"

EventType = Literal[
    "token_occurrence",
    "repeated_token_sequence",
    "first_repetition",
    "detector_finding",
    "manual",
    "text_pattern",
]
ObservationMode = Literal["online", "reconstructed"]
AlignmentStatus = Literal["exact", "ambiguous", "unavailable"]

EVENT_LIMITATIONS = [
    "An event is an observation. It is not evidence of a defect, a reasoning failure, or an intended mechanism.",
    "Repeated tokens can be intentional (lists, tables, code, verse, or emphasis).",
    "An exact token position does not by itself identify the computation that caused the event.",
    "Events reconstructed from text without token identities cannot support exact activation targeting.",
]


def _new(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def new_event_id() -> str:
    return _new("ev")


def new_trigger_id() -> str:
    return _new("et")


def new_event_run_id() -> str:
    return _new("ei")


class TokenSpan(StrictModel):
    """Half-open absolute token span ``[start, end)``."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "TokenSpan":
        if self.end <= self.start:
            raise ValueError("a token span end must be greater than its start")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start


class EventTriggerDefinition(StrictModel):
    """The declarative condition that produced an event.

    A definition never contains an executable predicate.  It is a validated,
    serialisable record so a later caller can re-evaluate exactly the same
    condition against the same or another token stream.
    """

    kind: EventType
    token_id: int | None = Field(default=None, ge=0)
    token_ids: list[int] = Field(default_factory=list, max_length=256)
    ngram_size: int | None = Field(default=None, ge=2, le=256)
    min_occurrences: int | None = Field(default=None, ge=2, le=1000)
    consecutive: bool = False
    lookback: int | None = Field(default=None, ge=2, le=100_000)
    text_pattern: str | None = Field(default=None, min_length=1, max_length=512)
    detector_id: str | None = None
    finding_index: int | None = Field(default=None, ge=0)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> "EventTriggerDefinition":
        if self.kind == "token_occurrence":
            supplied = int(self.token_id is not None) + int(bool(self.token_ids))
            if supplied != 1:
                raise ValueError(
                    "token_occurrence requires exactly one of token_id or token_ids"
                )
            if any(value < 0 for value in self.token_ids):
                raise ValueError("token identifiers must be non-negative")
        if self.kind in ("repeated_token_sequence", "first_repetition"):
            if self.ngram_size is None or self.min_occurrences is None:
                raise ValueError(
                    f"{self.kind} requires ngram_size and min_occurrences"
                )
            if self.consecutive and self.lookback is not None:
                raise ValueError(
                    "consecutive repetition measures adjacent occurrences; a lookback "
                    "window is only meaningful for nonconsecutive repetition"
                )
        if self.kind == "text_pattern" and not self.text_pattern:
            raise ValueError("text_pattern requires text_pattern")
        if self.kind == "detector_finding":
            if not self.detector_id or self.finding_index is None:
                raise ValueError(
                    "detector_finding requires detector_id and finding_index"
                )
        return self


class EventAlignment(StrictModel):
    """How an event maps to the exact token stream and to a forward step."""

    status: AlignmentStatus
    prompt_length: int = Field(default=0, ge=0)
    full_length: int = Field(default=0, ge=0)
    token_position: int | None = Field(default=None, ge=0)
    consumed_position: int | None = Field(default=None, ge=0)
    generation_step: int | None = Field(default=None, ge=0)
    earliest_online_step: int | None = Field(default=None, ge=0)
    detail: str | None = None

    @model_validator(mode="after")
    def _status(self) -> "EventAlignment":
        if self.status == "exact" and self.token_position is None:
            raise ValueError("an exact alignment requires a token position")
        return self


class ArtifactPointer(StrictModel):
    artifact_id: str
    kind: str | None = None
    sha256: str | None = None
    note: str | None = None


class BehavioralEvent(StrictModel):
    """One observed event, independent of any scientific interpretation."""

    schema_version: Literal[SCHEMA_EVENT] = SCHEMA_EVENT
    event_id: str = Field(pattern=r"^ev_[0-9a-f]{16}$")
    created_at: str = Field(default_factory=utc_now_iso)
    event_type: EventType
    trace_id: str | None = None
    trace_artifact_id: str | None = None
    observation_mode: ObservationMode
    generation_step: int | None = Field(default=None, ge=0)
    token_position: int | None = Field(default=None, ge=0)
    token_span: TokenSpan | None = None
    token_ids: list[int] = Field(default_factory=list)
    prompt_length: int = Field(default=0, ge=0)
    full_length: int = Field(default=0, ge=0)
    trigger: EventTriggerDefinition
    alignment: EventAlignment
    evidence: list[ArtifactPointer] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=lambda: list(EVENT_LIMITATIONS))
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _position(self) -> "BehavioralEvent":
        if self.token_span is not None:
            if self.full_length and self.token_span.end > self.full_length:
                raise ValueError("the token span exceeds the recorded full sequence")
            if self.token_position is None:
                self.token_position = self.token_span.end - 1
        if self.token_position is not None and self.full_length:
            if self.token_position >= self.full_length:
                raise ValueError("token_position exceeds the recorded full sequence")
        return self


def event_sha256(event: BehavioralEvent) -> str:
    """Stable content hash, excluding the hash field itself."""

    payload = event.model_dump(mode="json")
    payload.get("checksums", {}).pop("event_sha256", None)
    canonical = __import__("json").dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + sha256_bytes(canonical.encode("utf-8")).split(":", 1)[1]


def finalize_event(event: BehavioralEvent) -> BehavioralEvent:
    event.checksums["event_sha256"] = event_sha256(event)
    return event


def _alignment(
    position: int | None,
    *,
    prompt_length: int,
    full_length: int,
    status: AlignmentStatus,
    detail: str | None = None,
) -> EventAlignment:
    if position is None or status != "exact":
        return EventAlignment(
            status=status,
            prompt_length=prompt_length,
            full_length=full_length,
            detail=detail,
        )
    generation_step = max(0, position - prompt_length)
    # The forward that predicts the token at ``position`` consumes the preceding
    # position.  ``consumed_position`` names that forward unambiguously.
    return EventAlignment(
        status="exact",
        prompt_length=prompt_length,
        full_length=full_length,
        token_position=position,
        consumed_position=max(0, position - 1),
        generation_step=generation_step,
        earliest_online_step=generation_step + 1,
        detail=detail,
    )


def _event(
    *,
    event_type: EventType,
    trigger: EventTriggerDefinition,
    tokens: Sequence[int],
    prompt_length: int,
    position: int | None,
    span: TokenSpan | None = None,
    alignment_status: AlignmentStatus = "exact",
    alignment_detail: str | None = None,
    observation_mode: ObservationMode = "reconstructed",
    trace_id: str | None = None,
    trace_artifact_id: str | None = None,
    evidence: Sequence[ArtifactPointer] = (),
    metadata: dict[str, Any] | None = None,
) -> BehavioralEvent:
    full = [int(value) for value in tokens]
    event = BehavioralEvent(
        event_id=new_event_id(),
        event_type=event_type,
        trace_id=trace_id,
        trace_artifact_id=trace_artifact_id,
        observation_mode=observation_mode,
        generation_step=None
        if position is None or alignment_status != "exact"
        else max(0, position - prompt_length),
        token_position=position if alignment_status == "exact" else position,
        token_span=span,
        token_ids=full[max(0, (span.start if span else (position or 0))) : (span.end if span else (position + 1 if position is not None else 0))],
        prompt_length=prompt_length,
        full_length=len(full),
        trigger=trigger,
        alignment=_alignment(
            position,
            prompt_length=prompt_length,
            full_length=len(full),
            status=alignment_status,
            detail=alignment_detail,
        ),
        evidence=list(evidence),
        metadata=metadata or {},
    )
    return finalize_event(event)


# --------------------------------------------------------------------------- #
# deterministic detectors
# --------------------------------------------------------------------------- #


def _generated(tokens: Sequence[int], prompt_length: int) -> list[int]:
    return [int(value) for value in tokens[prompt_length:]]


def token_occurrences(
    tokens: Sequence[int],
    *,
    prompt_length: int,
    token_id: int | None = None,
    token_sequence: Sequence[int] | None = None,
    limit: int = 32,
) -> list[int]:
    """Return absolute positions at which a token or token sequence ends."""

    full = [int(value) for value in tokens]
    if prompt_length < 0 or prompt_length > len(full):
        raise InputError("prompt_length lies outside the token sequence")
    needle = [int(token_id)] if token_id is not None else [int(v) for v in token_sequence or []]
    if not needle:
        raise InputError("a token occurrence needs a token id or a non-empty sequence")
    if any(value < 0 for value in needle):
        raise InputError("token identifiers must be non-negative")
    found: list[int] = []
    for start in range(prompt_length, len(full) - len(needle) + 1):
        if full[start : start + len(needle)] == needle:
            found.append(start + len(needle) - 1)
            if len(found) >= limit:
                break
    return found


def repeated_sequence_positions(
    tokens: Sequence[int],
    *,
    prompt_length: int,
    ngram_size: int,
    min_occurrences: int,
    consecutive: bool = False,
    lookback: int | None = None,
    limit: int = 32,
) -> list[tuple[int, int, int, TokenSpan]]:
    """Find generated positions where an n-gram repetition condition becomes true.

    Returns ``(end_position, occurrences, span_start, span)`` for each newly
    satisfied repetition.  A position is reported once, at the moment the
    configured occurrence count is first reached.  ``consecutive`` counts only
    adjacent copies of the same n-gram; otherwise any occurrence within the
    bounded lookback counts.
    """

    if ngram_size < 2:
        raise InputError("ngram_size must be at least 2")
    if min_occurrences < 2:
        raise InputError("min_occurrences must be at least 2")
    full = [int(value) for value in tokens]
    generated = _generated(full, prompt_length)
    if ngram_size > len(generated):
        return []
    window = max(1, int(lookback or max(ngram_size * min_occurrences * 8, ngram_size * 4)))
    found: list[tuple[int, int, int, TokenSpan]] = []
    seen_ends: set[int] = set()
    for end in range(ngram_size - 1, len(generated)):
        start = end - ngram_size + 1
        key = tuple(generated[start : end + 1])
        search_from = max(0, start - window)
        starts = [
            index
            for index in range(search_from, end + 1)
            if tuple(generated[index : index + ngram_size]) == key
        ]
        if consecutive:
            # Keep the longest suffix whose copies are adjacent.
            chain = [starts[-1]]
            for index in reversed(starts[:-1]):
                if index + ngram_size == chain[0]:
                    chain.insert(0, index)
                else:
                    break
            count = len(chain)
            first = chain[0]
        else:
            count = len(starts)
            first = starts[0]
        if count < min_occurrences or end in seen_ends:
            continue
        seen_ends.add(end)
        absolute_end = prompt_length + end
        absolute_start = prompt_length + first
        found.append(
            (
                absolute_end,
                count,
                absolute_start,
                TokenSpan(start=absolute_start, end=absolute_end + 1),
            )
        )
        if len(found) >= limit:
            break
    return found


def earliest_repetition(
    tokens: Sequence[int],
    *,
    prompt_length: int,
    ngram_size: int,
    min_occurrences: int,
    consecutive: bool = False,
    lookback: int | None = None,
) -> tuple[int, int, TokenSpan] | None:
    found = repeated_sequence_positions(
        tokens,
        prompt_length=prompt_length,
        ngram_size=ngram_size,
        min_occurrences=min_occurrences,
        consecutive=consecutive,
        lookback=lookback,
        limit=1,
    )
    if not found:
        return None
    end, count, _start, span = found[0]
    return end, count, span


def detect_events(
    tokens: Sequence[int],
    *,
    definition: EventTriggerDefinition,
    prompt_length: int,
    observation_mode: ObservationMode = "reconstructed",
    trace_id: str | None = None,
    trace_artifact_id: str | None = None,
    budget: Budget | None = None,
    max_events: int = 16,
    start_after: int | None = None,
) -> list[BehavioralEvent]:
    """Evaluate one declarative trigger definition against a token stream."""

    budget = budget or Budget()
    full = [int(value) for value in tokens]
    if prompt_length < 0 or prompt_length > len(full):
        raise InputError("prompt_length lies outside the token sequence")
    if max_events < 1:
        raise InputError("max_events must be positive")
    if len(full) > budget.max_event_sequence_tokens:
        raise ResourceLimitError(
            "the token sequence exceeds the event-detection budget",
            details={"tokens": len(full), "limit": budget.max_event_sequence_tokens},
        )
    lookback = definition.lookback
    if lookback is not None and lookback > budget.max_event_lookback:
        raise ResourceLimitError(
            "the event lookback exceeds the configured limit",
            details={"lookback": lookback, "limit": budget.max_event_lookback},
        )

    floor = prompt_length if start_after is None else max(prompt_length, start_after + 1)
    events: list[BehavioralEvent] = []

    def append(**kwargs: Any) -> None:
        if len(events) >= max_events:
            return
        events.append(_event(**kwargs))

    if definition.kind == "token_occurrence":
        positions = token_occurrences(
            full,
            prompt_length=prompt_length,
            token_id=definition.token_id,
            token_sequence=definition.token_ids or None,
            limit=max_events * 4,
        )
        for position in positions:
            if position < floor:
                continue
            append(
                event_type="token_occurrence",
                trigger=definition,
                tokens=full,
                prompt_length=prompt_length,
                position=position,
                observation_mode=observation_mode,
                trace_id=trace_id,
                trace_artifact_id=trace_artifact_id,
            )
            if len(events) >= max_events:
                break

    elif definition.kind in ("repeated_token_sequence", "first_repetition"):
        assert definition.ngram_size is not None and definition.min_occurrences is not None
        found = repeated_sequence_positions(
            full,
            prompt_length=prompt_length,
            ngram_size=definition.ngram_size,
            min_occurrences=definition.min_occurrences,
            consecutive=definition.consecutive,
            lookback=definition.lookback,
            limit=max_events * 4,
        )
        for end, count, _start, span in found:
            if end < floor:
                continue
            if definition.kind == "first_repetition" and events:
                break
            append(
                event_type=definition.kind,
                trigger=definition,
                tokens=full,
                prompt_length=prompt_length,
                position=end,
                span=span,
                observation_mode=observation_mode,
                trace_id=trace_id,
                trace_artifact_id=trace_artifact_id,
                metadata={"occurrences": count, "ngram_size": definition.ngram_size},
            )
            if len(events) >= max_events:
                break

    elif definition.kind == "manual":
        start = definition.parameters.get("start")
        end = definition.parameters.get("end")
        position = definition.parameters.get("position")
        if position is not None:
            position = int(position)
            if position < prompt_length or position >= len(full):
                raise InputError(
                    "manual event positions must reference generated tokens",
                    details={"position": position, "prompt_length": prompt_length},
                )
            append(
                event_type="manual",
                trigger=definition,
                tokens=full,
                prompt_length=prompt_length,
                position=position,
                observation_mode=observation_mode,
                trace_id=trace_id,
                trace_artifact_id=trace_artifact_id,
            )
        elif start is not None and end is not None:
            span = TokenSpan(start=int(start), end=int(end))
            if span.start < prompt_length or span.end > len(full):
                raise InputError("manual event spans must reference generated tokens")
            append(
                event_type="manual",
                trigger=definition,
                tokens=full,
                prompt_length=prompt_length,
                position=span.end - 1,
                span=span,
                observation_mode=observation_mode,
                trace_id=trace_id,
                trace_artifact_id=trace_artifact_id,
            )
        else:
            raise InputError("a manual event requires position or start/end")

    elif definition.kind == "text_pattern":
        raise InputError(
            "text-pattern events require decoded token text; use detect_text_pattern "
            "with a trace so token boundaries can be checked"
        )

    elif definition.kind == "detector_finding":
        raise InputError(
            "detector findings are resolved against a recorded trace; use resolve_finding_event"
        )

    return events


def detect_text_pattern(
    trace: GenerationTrace,
    pattern: str,
    *,
    prompt_length: int,
    trace_artifact_id: str | None = None,
    max_events: int = 8,
) -> list[BehavioralEvent]:
    """Find a literal text pattern and report token-boundary ambiguity honestly.

    The search runs over the recorded generated token texts.  A hit is exact only
    when both pattern boundaries coincide with token boundaries.  Otherwise the
    event records the enclosing token span but marks its alignment ambiguous, so
    it cannot silently be used for activation targeting.
    """

    if not pattern:
        raise InputError("a text pattern must not be empty")
    tokens = [token for token in trace.tokens]
    if not tokens:
        raise InputError("the trace does not contain generated token records")
    text = "".join(token.text or "" for token in tokens)
    if len(text) > 2_000_000:
        raise ResourceLimitError("the decoded token text exceeds the event-detection limit")
    prompt_ids = [int(value) for value in (trace.input.token_ids or [])]
    if len(prompt_ids) != prompt_length:
        prompt_ids = [0] * prompt_length
    generated_ids = [int(token.token_id or 0) for token in tokens]
    definition = EventTriggerDefinition(kind="text_pattern", text_pattern=pattern)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        piece = token.text or ""
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    events: list[BehavioralEvent] = []
    start = text.find(pattern)
    while start >= 0 and len(events) < max_events:
        end = start + len(pattern)
        token_start = next(
            (index for index, (a, b) in enumerate(offsets) if a <= start < b), None
        )
        token_end = next(
            (index for index, (a, b) in enumerate(offsets) if a < end <= b), None
        )
        exact = (
            token_start is not None
            and token_end is not None
            and offsets[token_start][0] == start
            and offsets[token_end][1] == end
        )
        if token_start is None or token_end is None:
            events.append(
                _event(
                    event_type="text_pattern",
                    trigger=definition,
                    tokens=prompt_ids + generated_ids,
                    prompt_length=prompt_length,
                    position=None,
                    alignment_status="unavailable",
                    alignment_detail="the pattern does not map to any recorded token boundary",
                    trace_id=trace.trace_id,
                    trace_artifact_id=trace_artifact_id,
                    metadata={"text_match_start": start, "text_match_end": end},
                )
            )
        else:
            span = TokenSpan(
                start=prompt_length + token_start, end=prompt_length + token_end + 1
            )
            events.append(
                _event(
                    event_type="text_pattern",
                    trigger=definition,
                    tokens=prompt_ids + generated_ids,
                    prompt_length=prompt_length,
                    position=span.end - 1,
                    span=span,
                    alignment_status="exact" if exact else "ambiguous",
                    alignment_detail=None
                    if exact
                    else "the pattern begins or ends inside a token; the enclosing span is approximate",
                    trace_id=trace.trace_id,
                    trace_artifact_id=trace_artifact_id,
                    metadata={"text_match_start": start, "text_match_end": end},
                )
            )
        start = text.find(pattern, start + 1)

    if len(text) > 2_000_000:
        raise ResourceLimitError("the decoded token text exceeds the event-detection limit")
    return events


def resolve_finding_event(
    trace: GenerationTrace,
    finding: Any,
    *,
    finding_index: int = 0,
    trace_artifact_id: str | None = None,
) -> BehavioralEvent:
    """Resolve a recorded detector finding to an exact generated-token position.

    The repetition detector and the other built-in detectors report token
    indices *within the generated tokens*.  The event records absolute
    positions and states the offset it applied, so a reader can check it.
    """

    token_ids = [int(token.token_id) for token in trace.tokens]
    prompt_length = len(trace.input.token_ids or [])
    full = [int(value) for value in (trace.input.token_ids or [])] + token_ids
    evidence = [item for item in (finding.evidence or []) if item.kind in ("token_range", "word_range")]
    token_evidence = [item for item in evidence if item.kind == "token_range"]
    definition = EventTriggerDefinition(
        kind="detector_finding",
        detector_id=finding.detector_id,
        finding_index=finding_index,
        parameters={"kind": finding.kind, "detector_version": finding.detector_version},
    )
    if not token_evidence:
        return _event(
            event_type="detector_finding",
            trigger=definition,
            tokens=full,
            prompt_length=prompt_length,
            position=None,
            alignment_status="ambiguous",
            alignment_detail=(
                "the finding has no token-range evidence; only word or text spans were "
                "recorded, so no exact activation position can be derived"
            ),
            trace_id=trace.trace_id,
            trace_artifact_id=trace_artifact_id,
            metadata={"finding_kind": finding.kind},
        )
    first = token_evidence[0]
    start = first.start
    end = first.end
    if start is None or end is None:
        return _event(
            event_type="detector_finding",
            trigger=definition,
            tokens=full,
            prompt_length=prompt_length,
            position=None,
            alignment_status="ambiguous",
            alignment_detail="the finding's token-range evidence has no start/end",
            trace_id=trace.trace_id,
            trace_artifact_id=trace_artifact_id,
        )
    if end <= start or end > len(token_ids):
        return _event(
            event_type="detector_finding",
            trigger=definition,
            tokens=full,
            prompt_length=prompt_length,
            position=None,
            alignment_status="ambiguous",
            alignment_detail=(
                "the finding's token range is outside the recorded generated tokens"
            ),
            trace_id=trace.trace_id,
            trace_artifact_id=trace_artifact_id,
        )
    span = TokenSpan(start=prompt_length + start, end=prompt_length + end)
    detail = (
        f"resolved detector token_range [{start}, {end}) to absolute "
        f"[{span.start}, {span.end}); offset = prompt_length = {prompt_length}"
    )
    return _event(
        event_type="detector_finding",
        trigger=definition,
        tokens=full,
        prompt_length=prompt_length,
        position=span.end - 1,
        span=span,
        alignment_detail=detail,
        trace_id=trace.trace_id,
        trace_artifact_id=trace_artifact_id,
        metadata={
            "finding_kind": finding.kind,
            "detector_offset": prompt_length,
            "generated_span": [start, end],
        },
    )


def manual_event(
    tokens: Sequence[int],
    *,
    prompt_length: int,
    position: int | None = None,
    start: int | None = None,
    end: int | None = None,
    trace_id: str | None = None,
    trace_artifact_id: str | None = None,
    note: str | None = None,
) -> BehavioralEvent:
    """Let a human or external model mark an explicit token position or span."""

    parameters: dict[str, Any] = {"note": note} if note else {}
    if position is not None:
        parameters["position"] = int(position)
    else:
        parameters["start"] = start
        parameters["end"] = end
    definition = EventTriggerDefinition(kind="manual", parameters=parameters)
    events = detect_events(
        tokens,
        definition=definition,
        prompt_length=prompt_length,
        observation_mode="reconstructed",
        trace_id=trace_id,
        trace_artifact_id=trace_artifact_id,
        max_events=1,
    )
    if not events:
        raise InputError("the manual event did not resolve to a generated position")
    return events[0]


__all__ = [
    "SCHEMA_EVENT",
    "EVENT_LIMITATIONS",
    "AlignmentStatus",
    "ArtifactPointer",
    "BehavioralEvent",
    "EventAlignment",
    "EventTriggerDefinition",
    "EventType",
    "TokenSpan",
    "detect_events",
    "detect_text_pattern",
    "earliest_repetition",
    "event_sha256",
    "finalize_event",
    "manual_event",
    "new_event_id",
    "new_event_run_id",
    "new_trigger_id",
    "repeated_sequence_positions",
    "resolve_finding_event",
    "token_occurrences",
]
