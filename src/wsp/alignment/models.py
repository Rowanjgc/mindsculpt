"""Token-to-activation alignment.

A trace records *what* a model produced.  Alignment records *where* those
tokens sit in the sequence a forward pass would see, which is what an
investigator needs in order to ask for an activation at a position.

Three index spaces exist and are never interchangeable:

``full``
    position in the token sequence presented to the model, prompt first;
``prompt``
    position relative to the recorded input tokens;
``generation``
    position relative to the first recorded output token.

For an ordinary autoregressive model the activation at position ``p`` predicts
the token at ``p + 1``.  The state that *consumes* a token and the state that
*predicts* the following token are therefore different positions, and this
module records both explicitly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_ALIGNMENT = "wsp.alignment/1.0"
SCHEMA_SPAN = "wsp.span-resolution/1.0"

AlignmentRole = Literal["prompt", "generation"]
OffsetSource = Literal["tokenizer", "derived", "absent"]
TextStatus = Literal["available", "missing", "redacted"]
SpanStatus = Literal["resolved", "ambiguous", "unresolved"]
ContextMode = Literal["full-sequence", "explicit-window"]

PREDICTION_SEMANTICS = "position-p-predicts-token-p+1"

PREDICTION_NOTE = (
    "In a causal model the activation at a position predicts the token that "
    "follows it. The state that consumes token p is not the state that predicts "
    "token p; prediction positions therefore precede the predicted tokens."
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AlignmentPosition(StrictModel):
    """One token position in the replayed sequence."""

    full_index: int = Field(ge=0)
    role: AlignmentRole
    role_index: int = Field(ge=0)
    token_id: int | None = Field(default=None, ge=0)
    text: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    byte_start: int | None = Field(default=None, ge=0)
    byte_end: int | None = Field(default=None, ge=0)
    special: bool | None = None
    offset_source: OffsetSource = "absent"
    #: This position consumes ``token_id``.
    consumes_token: bool = True
    #: Position whose logits predict the token that follows this one.
    predicts_full_index: int | None = Field(default=None, ge=0)
    #: Position whose logits predict the token at this position.
    predicted_by_full_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _ranges(self) -> "AlignmentPosition":
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be provided together")
        if self.char_start is not None and self.char_end is not None:
            if self.char_end < self.char_start:
                raise ValueError("char_end must not precede char_start")
        if (self.byte_start is None) != (self.byte_end is None):
            raise ValueError("byte_start and byte_end must be provided together")
        if self.byte_start is not None and self.byte_end is not None:
            if self.byte_end < self.byte_start:
                raise ValueError("byte_end must not precede byte_start")
        if self.role == "prompt" and self.predicted_by_full_index is not None:
            if self.predicted_by_full_index >= self.full_index:
                raise ValueError("a position cannot be predicted by itself or a later position")
        return self


class AlignmentSegment(StrictModel):
    role: AlignmentRole
    start_full_index: int = Field(ge=0)
    end_full_index: int = Field(ge=0)
    token_count: int = Field(ge=0)
    text_status: TextStatus
    text_chars: int = Field(default=0, ge=0)
    offsets_available: bool = False


class TokenAlignment(StrictModel):
    """A versioned map between trace tokens and forward-sequence positions."""

    schema_version: str = SCHEMA_ALIGNMENT
    alignment_id: str = Field(pattern=r"^al_[0-9a-f]{16}$")
    created_at: str
    trace_id: str = Field(pattern=r"^tr_[0-9a-f]{16}$")
    trace_artifact_sha256: str | None = None
    model_name: str | None = None
    tokenizer_name: str | None = None
    tokenizer_fingerprint: str | None = None
    sequence_sha256: str | None = None
    prompt_length: int = Field(ge=0)
    generated_length: int = Field(ge=0)
    full_length: int = Field(ge=0)
    prediction_semantics: str = PREDICTION_SEMANTICS
    positions: list[AlignmentPosition] = Field(default_factory=list)
    segments: list[AlignmentSegment] = Field(default_factory=list)
    special_positions: list[int] = Field(default_factory=list)
    prompt_text_status: TextStatus = "available"
    generation_text_status: TextStatus = "available"
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unavailable: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> "TokenAlignment":
        if self.full_length != self.prompt_length + self.generated_length:
            raise ValueError("full_length must equal prompt_length + generated_length")
        if len(self.positions) != self.full_length:
            raise ValueError("positions must cover the full sequence")
        for index, position in enumerate(self.positions):
            if position.full_index != index:
                raise ValueError("positions must be ordered and contiguous from zero")
        return self

    # ------------------------------------------------------------------ queries

    def position(self, full_index: int) -> AlignmentPosition:
        if full_index < 0 or full_index >= len(self.positions):
            raise IndexError(f"full index {full_index} is outside the aligned sequence")
        return self.positions[full_index]

    def for_role(self, role: AlignmentRole) -> list[AlignmentPosition]:
        return [position for position in self.positions if position.role == role]

    def to_full_index(self, role: AlignmentRole, role_index: int) -> int:
        offset = 0 if role == "prompt" else self.prompt_length
        length = self.prompt_length if role == "prompt" else self.generated_length
        if role_index < 0 or role_index >= length:
            raise IndexError(f"{role} index {role_index} is outside the aligned sequence")
        return offset + role_index

    def to_role_index(self, full_index: int) -> tuple[AlignmentRole, int]:
        position = self.position(full_index)
        return position.role, position.role_index

    def predicting_positions_for(self, full_indices: list[int]) -> list[int]:
        """Positions whose logits predict the given tokens, in causal order."""

        selected: list[int] = []
        for full_index in full_indices:
            candidate = full_index - 1
            if candidate >= 0 and candidate not in selected:
                selected.append(candidate)
        return selected

    def consuming_positions_for(self, full_indices: list[int]) -> list[int]:
        return sorted({index for index in full_indices if 0 <= index < self.full_length})


class SpanResolution(StrictModel):
    """The result of mapping a behavioral span onto token positions."""

    schema_version: str = SCHEMA_SPAN
    resolution_id: str = Field(pattern=r"^sp_[0-9a-f]{16}$")
    created_at: str
    trace_id: str = Field(pattern=r"^tr_[0-9a-f]{16}$")
    alignment_id: str = Field(pattern=r"^al_[0-9a-f]{16}$")
    requested_kind: Literal["token_range", "text_span", "word_range", "finding_evidence"]
    requested_role: AlignmentRole | None = None
    requested_start: int | None = None
    requested_end: int | None = None
    requested_name: str | None = None
    status: SpanStatus = "unresolved"
    roles: list[AlignmentRole] = Field(default_factory=list)
    role_ranges: list[dict[str, Any]] = Field(default_factory=list)
    full_token_indices: list[int] = Field(default_factory=list)
    consuming_positions: list[int] = Field(default_factory=list)
    predicting_positions: list[int] = Field(default_factory=list)
    char_range: dict[str, int] | None = None
    byte_range: dict[str, int] | None = None
    selected_text: str | None = None
    context_mode: ContextMode = "full-sequence"
    context_window: dict[str, int] | None = None
    context_note: str = PREDICTION_NOTE
    ambiguity: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "PREDICTION_NOTE",
    "PREDICTION_SEMANTICS",
    "SCHEMA_ALIGNMENT",
    "SCHEMA_SPAN",
    "AlignmentPosition",
    "AlignmentRole",
    "AlignmentSegment",
    "ContextMode",
    "SpanResolution",
    "SpanStatus",
    "TextStatus",
    "TokenAlignment",
]
