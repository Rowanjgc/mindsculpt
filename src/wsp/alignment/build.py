"""Build a token alignment from a stored trace.

Alignment prefers recorded token identifiers.  Text is used only to derive
character and byte ranges, and when the derived text does not match the
recorded text exactly those ranges are reported as unavailable instead of being
guessed.  The same rule applies to special tokens: they are reported only when a
special-token set is supplied.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from wsp.alignment.ids import new_alignment_id
from wsp.alignment.models import (
    PREDICTION_NOTE,
    AlignmentPosition,
    AlignmentSegment,
    TokenAlignment,
)
from wsp.core.budget import Budget
from wsp.core.errors import InputError, ResourceLimitError
from wsp.core.time import utc_now_iso


def _decode(decoder: Any, token_id: int) -> str | None:
    if decoder is None:
        return None
    try:
        value = decoder([int(token_id)])
    except Exception:  # pragma: no cover - tokenizer specific
        return None
    return value if isinstance(value, str) else None


def _char_ranges(
    pieces: Sequence[str | None], text: str | None
) -> tuple[list[tuple[int, int] | None], str | None]:
    """Derive character ranges only when the token texts reproduce the text.

    Returns ``(ranges, reason)``.  ``reason`` explains why ranges are absent,
    which becomes an explicit warning rather than a silent gap.
    """

    if text is None:
        return [None] * len(pieces), "the recorded text is unavailable, so character ranges cannot be derived"
    if any(piece is None for piece in pieces):
        return [None] * len(pieces), "the tokenizer could not decode every token identifier"
    joined = "".join(piece for piece in pieces if piece is not None)
    if joined != text:
        return (
            [None] * len(pieces),
            "decoded token text does not reproduce the recorded text exactly, so character ranges are ambiguous",
        )
    ranges: list[tuple[int, int] | None] = []
    cursor = 0
    for piece in pieces:
        assert piece is not None
        ranges.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return ranges, None


def _byte_ranges(
    char_ranges: Sequence[tuple[int, int] | None], text: str | None
) -> list[tuple[int, int] | None]:
    if text is None:
        return [None] * len(char_ranges)
    encoded_prefix: list[int] = [0]
    for character in text:
        encoded_prefix.append(encoded_prefix[-1] + len(character.encode("utf-8")))
    result: list[tuple[int, int] | None] = []
    for value in char_ranges:
        if value is None:
            result.append(None)
        else:
            result.append((encoded_prefix[value[0]], encoded_prefix[value[1]]))
    return result


def _merge_ranges(
    given: Sequence[tuple[int, int]], pieces: Sequence[str | None], text: str | None
) -> tuple[list[tuple[int, int] | None], str | None]:
    """Use tokenizer offsets, validating them against the recorded text."""

    if len(given) != len(pieces):
        return _char_ranges(pieces, text)
    if text is None:
        return [None] * len(pieces), "the recorded text is unavailable, so tokenizer offsets cannot be checked"
    for start, end in given:
        if start < 0 or end > len(text) or end < start:
            return [None] * len(pieces), "tokenizer offsets fall outside the recorded text"
    return list(given), None


def build_alignment(
    trace: Any,
    *,
    budget: Budget | None = None,
    offsets: Sequence[tuple[int, int]] | None = None,
    generation_offsets: Sequence[tuple[int, int]] | None = None,
    tokenizer: Any = None,
    special_token_ids: Iterable[int] | None = None,
    trace_artifact_sha256: str | None = None,
    include_annotations: bool = True,
) -> TokenAlignment:
    """Map every recorded token onto a forward-sequence position.

    ``offsets``/``generation_offsets`` may be supplied by a tokenizer that
    reports character spans.  ``tokenizer`` is used only for decoding token text
    when the trace did not record it; no model is loaded here.
    """

    budget = budget or Budget()
    if not trace.input.token_ids:
        raise InputError(
            "this trace has no recorded input token identifiers, so alignment would have to guess",
            details={
                "hint": "record the trace with a backend that stores token ids, or import a trace that has them"
            },
        )
    generation_ids = [token.token_id for token in trace.tokens]
    if any(token_id is None for token_id in generation_ids):
        raise InputError(
            "some recorded output tokens have no token identifier, so alignment would be incomplete",
            details={"tokens": len(generation_ids)},
        )
    prompt_ids = [int(value) for value in trace.input.token_ids]
    generated_ids = [int(value) for value in generation_ids if value is not None]
    full_length = len(prompt_ids) + len(generated_ids)
    if full_length > budget.max_alignment_tokens:
        raise ResourceLimitError(
            "the trace is longer than the alignment token limit",
            details={"tokens": full_length, "limit": budget.max_alignment_tokens},
        )

    decoder = None
    if tokenizer is not None:
        decoder = tokenizer if callable(tokenizer) else getattr(tokenizer, "decode", None)

    prompt_pieces: list[str | None] = [None] * len(prompt_ids)
    if decoder is not None:
        for index, token_id in enumerate(prompt_ids):
            prompt_pieces[index] = _decode(decoder, token_id)

    generation_pieces: list[str | None] = [token.text for token in trace.tokens]
    if any(piece is None for piece in generation_pieces) and decoder is not None:
        generation_pieces = [
            piece if piece is not None else _decode(decoder, token_id)
            for piece, token_id in zip(generation_pieces, generated_ids)
        ]

    prompt_text = trace.input.text
    generation_text = trace.text
    warnings: list[str] = []
    unavailable: dict[str, str] = {}

    if prompt_text is None:
        prompt_ranges = [None] * len(prompt_ids)
        prompt_reason = "the trace does not store its input text"
    elif offsets is not None:
        prompt_ranges, prompt_reason = _merge_ranges(offsets, prompt_pieces, prompt_text)
    else:
        prompt_ranges, prompt_reason = _char_ranges(prompt_pieces, prompt_text)

    if generation_text is None:
        generation_ranges = [None] * len(generated_ids)
        generation_reason = "the trace does not store its output text"
    elif generation_offsets is not None:
        generation_ranges, generation_reason = _merge_ranges(
            generation_offsets, generation_pieces, generation_text
        )
    else:
        generation_ranges, generation_reason = _char_ranges(generation_pieces, generation_text)

    prompt_offset_source = "tokenizer" if offsets is not None else ("derived" if prompt_reason is None else "absent")
    generation_offset_source = (
        "tokenizer" if generation_offsets is not None else ("derived" if generation_reason is None else "absent")
    )
    if prompt_reason is not None:
        warnings.append(f"prompt offsets unavailable: {prompt_reason}")
        unavailable["prompt_char_ranges"] = prompt_reason
    if generation_reason is not None:
        warnings.append(f"generation offsets unavailable: {generation_reason}")
        unavailable["generation_char_ranges"] = generation_reason

    prompt_bytes = _byte_ranges(prompt_ranges, prompt_text)
    generation_bytes = _byte_ranges(generation_ranges, generation_text)

    specials = {int(value) for value in (special_token_ids or ())}
    special_known = special_token_ids is not None
    if not special_known:
        warnings.append(
            "no special-token set was supplied, so special positions are reported as unknown"
        )
        unavailable["special_tokens"] = "the caller did not supply a special-token set"

    positions: list[AlignmentPosition] = []
    for index, token_id in enumerate(prompt_ids):
        char_range = prompt_ranges[index]
        byte_range = prompt_bytes[index]
        positions.append(
            AlignmentPosition(
                full_index=index,
                role="prompt",
                role_index=index,
                token_id=token_id,
                text=prompt_pieces[index],
                char_start=char_range[0] if char_range else None,
                char_end=char_range[1] if char_range else None,
                byte_start=byte_range[0] if byte_range else None,
                byte_end=byte_range[1] if byte_range else None,
                special=(token_id in specials) if special_known else None,
                offset_source=prompt_offset_source,  # type: ignore[arg-type]
                consumes_token=True,
                predicts_full_index=index + 1 if index + 1 < full_length else None,
                predicted_by_full_index=index - 1 if index > 0 else None,
            )
        )
    for index, token_id in enumerate(generated_ids):
        full_index = len(prompt_ids) + index
        char_range = generation_ranges[index]
        byte_range = generation_bytes[index]
        positions.append(
            AlignmentPosition(
                full_index=full_index,
                role="generation",
                role_index=index,
                token_id=token_id,
                text=generation_pieces[index],
                char_start=char_range[0] if char_range else None,
                char_end=char_range[1] if char_range else None,
                byte_start=byte_range[0] if byte_range else None,
                byte_end=byte_range[1] if byte_range else None,
                special=(token_id in specials) if special_known else None,
                offset_source=generation_offset_source,  # type: ignore[arg-type]
                consumes_token=True,
                predicts_full_index=full_index + 1 if full_index + 1 < full_length else None,
                predicted_by_full_index=full_index - 1,
            )
        )

    segments = [
        AlignmentSegment(
            role="prompt",
            start_full_index=0,
            end_full_index=max(0, len(prompt_ids) - 1),
            token_count=len(prompt_ids),
            text_status=(
                "available" if prompt_text is not None else "redacted"
            ),
            text_chars=len(prompt_text or ""),
            offsets_available=prompt_reason is None,
        ),
        AlignmentSegment(
            role="generation",
            start_full_index=len(prompt_ids),
            end_full_index=max(len(prompt_ids), full_length - 1) if generated_ids else len(prompt_ids),
            token_count=len(generated_ids),
            text_status=(
                "available" if generation_text is not None else "redacted"
            ),
            text_chars=len(generation_text or ""),
            offsets_available=generation_reason is None,
        ),
    ]

    annotations: list[dict[str, Any]] = []
    if include_annotations:
        for annotation in trace.annotations:
            entry: dict[str, Any] = {
                "name": annotation.name,
                "start": annotation.start,
                "end": annotation.end,
                "resolution": "unresolved",
            }
            if prompt_text is not None:
                if annotation.start >= len(prompt_text) and generation_text is not None:
                    entry["resolution"] = "generation-relative"
            annotations.append(entry)

    return TokenAlignment(
        alignment_id=new_alignment_id(),
        created_at=utc_now_iso(),
        trace_id=trace.trace_id,
        trace_artifact_sha256=trace_artifact_sha256,
        model_name=trace.model.name,
        tokenizer_name=trace.tokenizer.name,
        tokenizer_fingerprint=trace.tokenizer.fingerprint,
        sequence_sha256=trace.checksums.get("tokens_sha256"),
        prompt_length=len(prompt_ids),
        generated_length=len(generated_ids),
        full_length=full_length,
        positions=positions,
        segments=segments,
        special_positions=[
            position.full_index for position in positions if position.special is True
        ],
        prompt_text_status="available" if prompt_text is not None else "redacted",
        generation_text_status="available" if generation_text is not None else "redacted",
        annotations=annotations,
        warnings=warnings,
        unavailable={**unavailable, "prediction_semantics": PREDICTION_NOTE},
        metadata={
            "prediction_semantics": PREDICTION_NOTE,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated_ids),
        },
    )


__all__ = ["build_alignment"]
