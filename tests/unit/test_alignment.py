"""Token alignment and span resolution.

The fixture tokenizer has multi-character tokens, a BOS prefix, and a
greedy longest-match rule, so these tests exercise the cases that matter:
role-relative versus full-sequence indices, special tokens, spans that cut
through a token, and text that cannot be mapped at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support.tiny_lm import build_runtime, make_trace
from wsp.alignment import build_alignment, resolve_span
from wsp.alignment.models import PREDICTION_SEMANTICS
from wsp.core.errors import ArtifactError, InputError
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace


def _aligned(prompt: str = "the cat sat on the mat", max_new_tokens: int = 6):
    runtime = build_runtime()
    trace = make_trace(runtime, prompt, max_new_tokens=max_new_tokens)
    alignment = build_alignment(
        trace, tokenizer=runtime.tokenizer, special_token_ids=[0, 1, 2, 3]
    )
    return runtime, trace, alignment


def test_alignment_covers_prompt_and_generation_with_distinct_index_spaces():
    _runtime, trace, alignment = _aligned()

    assert alignment.prompt_length == len(trace.input.token_ids or [])
    assert alignment.generated_length == len(trace.tokens)
    assert alignment.full_length == alignment.prompt_length + alignment.generated_length
    assert alignment.prediction_semantics == PREDICTION_SEMANTICS
    assert [position.full_index for position in alignment.positions] == list(
        range(alignment.full_length)
    )

    for position in alignment.for_role("prompt"):
        assert alignment.to_full_index("prompt", position.role_index) == position.full_index
    for position in alignment.for_role("generation"):
        assert alignment.to_full_index("generation", position.role_index) == position.full_index
        assert position.full_index != position.role_index  # not interchangeable
        assert alignment.to_role_index(position.full_index) == ("generation", position.role_index)


def test_prediction_positions_precede_the_predicted_tokens():
    _runtime, _trace, alignment = _aligned()

    first = alignment.position(0)
    second = alignment.position(1)
    assert first.predicted_by_full_index is None
    assert first.predicts_full_index == 1
    assert second.predicted_by_full_index == 0
    assert second.predicts_full_index == 2
    assert alignment.predicting_positions_for([3, 4]) == [2, 3]
    assert alignment.consuming_positions_for([4, 3, 3]) == [3, 4]


def test_special_tokens_are_flagged_only_when_a_special_set_is_supplied():
    _runtime, _trace, alignment = _aligned()
    assert alignment.special_positions[0] == 0
    assert alignment.position(0).special is True
    assert alignment.position(1).special is False

    runtime = build_runtime()
    trace = make_trace(runtime, "hello", max_new_tokens=2)
    without = build_alignment(trace, tokenizer=runtime.tokenizer)
    assert without.position(0).special is None
    assert "special_tokens" in without.unavailable
    assert any("special-token set" in warning for warning in without.warnings)


def test_character_and_byte_ranges_line_up_with_the_recorded_text():
    runtime, trace, alignment = _aligned(prompt="the cat sat")
    prompt_text = trace.input.text or ""
    for position in alignment.for_role("prompt"):
        if position.char_start is None:
            continue
        assert prompt_text[position.char_start : position.char_end] == position.text
    # ASCII text: byte offsets equal character offsets.
    assert alignment.position(1).byte_start == alignment.position(1).char_start

    generation_text = trace.text or ""
    for position in alignment.for_role("generation"):
        if position.char_start is None:
            continue
        assert generation_text[position.char_start : position.char_end] == position.text


def test_unmappable_text_reports_ambiguity_instead_of_inventing_offsets():
    runtime = build_runtime()
    trace = make_trace(runtime, "héllo", max_new_tokens=2)
    alignment = build_alignment(trace, tokenizer=runtime.tokenizer)

    # The fixture tokenizer has no accented character, so decoding cannot
    # reproduce the recorded prompt text exactly.
    assert all(position.char_start is None for position in alignment.for_role("prompt"))
    assert "prompt_char_ranges" in alignment.unavailable
    assert alignment.prompt_text_status == "available"


def test_missing_token_ids_are_refused():
    runtime = build_runtime()
    trace = make_trace(runtime, "hello", max_new_tokens=2)
    trace.input = trace.input.model_copy(update={"token_ids": None, "tokens_sha256": None})

    with pytest.raises(InputError) as excinfo:
        build_alignment(trace)
    assert "guess" in str(excinfo.value)


def test_redacted_text_keeps_token_alignment_but_loses_offsets():
    runtime = build_runtime()
    trace = make_trace(runtime, "hello there", max_new_tokens=3)
    from wsp.behavior.trace import finalize_trace
    from wsp.core.budget import Budget

    trace.text = None
    trace.privacy = trace.privacy.model_copy(update={"store_output_text": False})
    trace = finalize_trace(trace, Budget())

    alignment = build_alignment(trace, tokenizer=runtime.tokenizer)
    assert alignment.full_length == alignment.prompt_length + alignment.generated_length
    assert alignment.generation_text_status == "redacted"
    assert all(position.char_start is None for position in alignment.for_role("generation"))
    assert "generation_char_ranges" in alignment.unavailable


def test_token_range_span_maps_to_full_indices_and_predicting_positions():
    _runtime, trace, alignment = _aligned(max_new_tokens=6)
    resolution = resolve_span(
        alignment, kind="token_range", start=1, end=3, generation_text=trace.text
    )

    assert resolution.status == "resolved"
    assert resolution.full_token_indices == [
        alignment.to_full_index("generation", 1),
        alignment.to_full_index("generation", 2),
    ]
    assert resolution.predicting_positions == [index - 1 for index in resolution.full_token_indices]
    assert resolution.roles == ["generation"]
    assert resolution.context_mode == "full-sequence"


def test_first_token_of_the_sequence_has_no_predicting_position():
    _runtime, trace, alignment = _aligned()
    resolution = resolve_span(
        alignment, kind="token_range", start=0, end=1, role="prompt", generation_text=trace.text
    )
    assert resolution.status == "ambiguous"
    assert resolution.full_token_indices == [0]
    assert resolution.predicting_positions == []
    assert any("no predicting position" in note for note in resolution.ambiguity)


def test_text_span_that_splits_a_token_is_reported_as_ambiguous():
    _runtime, trace, alignment = _aligned(prompt="the cat sat")
    assert (trace.input.text or "").startswith("the")
    # Character 1..2 sits inside the multi-character token "the".
    resolution = resolve_span(
        alignment,
        kind="text_span",
        start=1,
        end=2,
        role="prompt",
        prompt_text=trace.input.text,
        generation_text=trace.text,
    )
    assert resolution.status == "ambiguous"
    assert resolution.selected_text == (trace.input.text or "")[1:2]
    assert any("extends beyond the requested span" in note for note in resolution.ambiguity)
    assert resolution.full_token_indices == [1]


def test_text_span_without_offsets_is_unresolved():
    runtime = build_runtime()
    trace = make_trace(runtime, "héllo", max_new_tokens=2)
    alignment = build_alignment(trace, tokenizer=runtime.tokenizer)
    resolution = resolve_span(
        alignment,
        kind="text_span",
        start=0,
        end=2,
        role="prompt",
        prompt_text=trace.input.text,
        generation_text=trace.text,
    )
    assert resolution.status == "unresolved"
    assert "character offsets are unavailable" in resolution.ambiguity[0]


def test_window_is_marked_as_a_different_context_approximation():
    _runtime, trace, alignment = _aligned()
    resolution = resolve_span(
        alignment,
        kind="token_range",
        start=0,
        end=2,
        generation_text=trace.text,
        window={"start_full_index": 4, "end_full_index": 12},
    )
    assert resolution.context_mode == "explicit-window"
    assert any("causal prefix" in warning for warning in resolution.warnings)


def test_word_range_is_interpreted_as_character_offsets():
    _runtime, trace, alignment = _aligned()
    resolution = resolve_span(
        alignment, kind="word_range", start=0, end=1, generation_text=trace.text
    )
    assert any("character offsets" in note for note in resolution.ambiguity)
    assert resolution.status in {"resolved", "ambiguous"}


def test_detector_evidence_resolves_through_the_service(service: ToolService, workspace: Workspace):
    recorded = service.call(
        "record_trace",
        {
            "input_text": "the cat sat on the mat",
            "params": {"max_new_tokens": 12},
            "backend": {"kind": "synthetic", "synthetic": {"model": {}}},
        },
    )
    trace_id = recorded["data"]["trace"]["trace_id"]
    scanned = service.call("scan_trace", {"trace_id": trace_id})
    assert scanned["ok"], scanned
    findings = scanned["data"]["findings"]
    assert findings, "the fixture is expected to produce a repetition finding"
    evidence = findings[0]["evidence"][0]
    assert evidence["kind"] in {"token_range", "word_range", "text_span"}

    aligned = service.call(
        "align_trace_tokens", {"trace_id": trace_id, "model": {"kind": "synthetic"}}
    )
    assert aligned["ok"], aligned
    alignment = aligned["data"]
    assert alignment["alignment_artifact_id"]
    assert alignment["special_positions"] == [0]

    resolved = service.call(
        "resolve_finding_span",
        {
            "alignment_artifact_id": alignment["alignment_artifact_id"],
            "kind": "finding_evidence",
            "detail": evidence,
        },
    )
    assert resolved["ok"], resolved
    data = resolved["data"]
    assert data["schema_version"] == "wsp.span-resolution/1.0"
    assert data["status"] in {"resolved", "ambiguous"}
    assert data["full_token_indices"]
    assert data["predicting_positions"] == [index - 1 for index in data["full_token_indices"]]
    assert data["span_artifact_id"]


def test_alignment_artifact_identity_is_enforced(service: ToolService):
    recorded = service.call(
        "record_trace",
        {
            "input_text": "hello there",
            "backend": {"kind": "synthetic", "synthetic": {"model": {}}},
            "params": {"max_new_tokens": 2},
        },
    )
    trace_id = recorded["data"]["trace"]["trace_id"]
    replay = service.call(
        "verify_replay", {"trace_id": trace_id, "model": {"kind": "synthetic"}}
    )
    result_artifact_id = replay["data"]["result_artifact_id"]

    with pytest.raises(ArtifactError):
        service.call(
            "resolve_finding_span",
            {
                "alignment_artifact_id": result_artifact_id,
                "kind": "token_range",
                "start": 0,
                "end": 1,
            },
        )
