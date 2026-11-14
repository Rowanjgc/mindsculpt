from __future__ import annotations

import pytest

from wsp.core.errors import InputError, NotFoundError


def _scan(service, trace_id: str, detector: str, parameters: dict | None = None) -> dict:
    return service.call(
        "scan_trace",
        {
            "trace_id": trace_id,
            "detectors": [{"detector_id": detector, "parameters": parameters or {}}],
        },
    )["data"]


def _run(data: dict, detector: str) -> dict:
    return next(run for run in data["scan"]["runs"] if run["detector_id"] == detector)


def test_repetition_detects_repeated_ngram(service, make_trace):
    tokens = ["alpha", "beta", "gamma", "delta"] * 4 + ["end"]
    trace_id = make_trace(
        " ".join(tokens),
        tokens=tokens,
        token_ids=list(range(len(tokens))),
    )
    data = _scan(
        service,
        trace_id,
        "core.repetition",
        {"ngram_size": 4, "min_occurrences": 2},
    )
    run = _run(data, "core.repetition")
    assert run["status"] == "ok"
    assert run["measurements"]["unit"] == "tokens"
    assert run["measurements"]["max_occurrences"] == 4
    finding = next(item for item in run["findings"] if item["kind"] == "repeated_ngram")
    assert finding["value"]["count"] == 4
    assert finding["value"]["sequence"] == ["alpha", "beta", "gamma", "delta"]
    assert finding["severity"] == "medium"
    assert "not diagnose" in finding["explanation"]
    assert finding["evidence"][0]["kind"] == "token_range"
    assert "intentional" in finding["limitations"][0]


def test_repetition_negative_case(service, make_trace):
    tokens = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    trace_id = make_trace(" ".join(tokens), tokens=tokens, token_ids=list(range(8)))
    data = _scan(service, trace_id, "core.repetition", {"ngram_size": 3})
    run = _run(data, "core.repetition")
    assert run["findings"] == []
    assert run["measurements"]["max_occurrences"] == 0


def test_repetition_detects_single_unit_run(service, make_trace):
    tokens = ["loop"] * 10
    trace_id = make_trace(" ".join(tokens), tokens=tokens, token_ids=[7] * 10)
    data = _scan(
        service,
        trace_id,
        "core.repetition",
        {"ngram_size": 8, "run_length_threshold": 5},
    )
    run = _run(data, "core.repetition")
    finding = next(item for item in run["findings"] if item["kind"] == "repeated_token_run")
    assert finding["value"]["length"] == 10
    assert run["measurements"]["longest_run"] == 10


def test_termination_distinguishes_reasons(service, make_trace):
    eos = make_trace("done", termination="eos")
    length = make_trace("cut", termination="length_limit")
    unknown = make_trace("?", termination="unknown")
    error = make_trace("boom", termination="error", termination_detail="backend failed")
    assert _run(_scan(service, eos, "core.termination"), "core.termination")["findings"] == []
    length_finding = _run(_scan(service, length, "core.termination"), "core.termination")[
        "findings"
    ][0]
    assert length_finding["kind"] == "length_limit_termination"
    assert "does not show why" in length_finding["explanation"]
    assert length_finding["severity"] == "medium"
    unknown_finding = _run(_scan(service, unknown, "core.termination"), "core.termination")[
        "findings"
    ][0]
    assert unknown_finding["kind"] == "unknown_termination"
    assert unknown_finding["severity"] == "info"
    error_finding = _run(_scan(service, error, "core.termination"), "core.termination")[
        "findings"
    ][0]
    assert error_finding["kind"] == "generation_error"
    assert error_finding["severity"] == "high"


def test_structured_output_validation(service, make_trace):
    valid = make_trace('{"name": "Ada", "count": 2}')
    invalid = make_trace("{not json}")
    missing = make_trace('{"name": "Ada"}')
    wrong_type = make_trace('{"name": 3, "count": 2}')
    fenced = make_trace('```json\n{"name": "Ada", "count": 2}\n```')
    params = {
        "format": "json",
        "contract": {
            "required_keys": ["name", "count"],
            "properties": {"name": "string", "count": "integer"},
        },
    }

    valid_run = _run(_scan(service, valid, "core.structured_output", params), "core.structured_output")
    assert valid_run["measurements"]["valid"] is True
    assert valid_run["findings"] == []

    invalid_run = _run(
        _scan(service, invalid, "core.structured_output", params), "core.structured_output"
    )
    assert invalid_run["findings"][0]["kind"] == "invalid_json"

    missing_run = _run(
        _scan(service, missing, "core.structured_output", params), "core.structured_output"
    )
    assert missing_run["findings"][0]["kind"] == "missing_required_key"
    assert missing_run["findings"][0]["value"] == "count"

    type_run = _run(
        _scan(service, wrong_type, "core.structured_output", params),
        "core.structured_output",
    )
    assert type_run["findings"][0]["kind"] == "type_mismatch"
    assert type_run["findings"][0]["value"]["observed"] == "int"

    fenced_run = _run(
        _scan(service, fenced, "core.structured_output", params), "core.structured_output"
    )
    assert fenced_run["measurements"]["used_code_fence"] is True
    assert fenced_run["findings"] == []


def test_reconsideration_counts_markers_without_diagnosing(service, make_trace):
    text = (
        "The answer might be Sydney. but wait, Canberra is the capital. "
        "but wait, let me reconsider. but wait, that's not right."
    )
    trace_id = make_trace(text)
    data = _scan(
        service,
        trace_id,
        "core.reconsideration",
        {"markers": ["but wait"], "min_markers": 3},
    )
    run = _run(data, "core.reconsideration")
    assert run["measurements"]["total_markers"] == 3
    finding = next(item for item in run["findings"] if item["kind"] == "reconsideration_markers")
    assert finding["value"]["count"] == 3
    assert "not a diagnosis" in finding["explanation"]
    assert finding["evidence"][0]["kind"] == "text_span"


def test_reconsideration_negative_case(service, make_trace):
    trace_id = make_trace("A calm, direct answer. but wait, one clarification.")
    data = _scan(
        service,
        trace_id,
        "core.reconsideration",
        {"markers": ["but wait"], "min_markers": 3},
    )
    run = _run(data, "core.reconsideration")
    assert run["measurements"]["total_markers"] == 1
    assert run["findings"] == []


def test_repeated_tool_calls_are_measured_with_retry_ambiguity(service, make_trace):
    calls = [
        {"index": 0, "name": "search", "arguments": {"query": "a"}, "status": "error"},
        {"index": 1, "name": "search", "arguments": {"query": "a"}, "status": "ok"},
        {"index": 2, "name": "search", "arguments": {"query": "b"}, "status": "ok"},
    ]
    trace_id = make_trace("tool trace", tool_calls=calls)
    data = _scan(service, trace_id, "core.tool_actions", {"min_repeats": 2})
    run = _run(data, "core.tool_actions")
    finding = run["findings"][0]
    assert finding["kind"] == "repeated_tool_call"
    assert finding["value"]["count"] == 2
    assert finding["metadata"]["retry_like"] is True
    assert "legitimate retries" in finding["limitations"][0]
    assert finding["severity"] == "low"

    single = make_trace("single", tool_calls=[calls[0]])
    single_data = _scan(service, single, "core.tool_actions", {"min_repeats": 2})
    assert _run(single_data, "core.tool_actions")["findings"] == []


def test_default_scan_excludes_undeclared_structured_output(service, make_trace):
    trace_id = make_trace("This is ordinary prose, not JSON.")
    data = service.call("scan_trace", {"trace_id": trace_id})["data"]
    ids = [run["detector_id"] for run in data["scan"]["runs"]]
    assert "core.structured_output" not in ids
    assert "core.repetition" in ids
    assert "core.termination" in ids


def test_unknown_detector_and_bad_parameters(service, make_trace):
    trace_id = make_trace("text")
    with pytest.raises(NotFoundError):
        service.call(
            "scan_trace",
            {"trace_id": trace_id, "detectors": [{"detector_id": "core.missing"}]},
        )
    with pytest.raises(InputError):
        service.call(
            "scan_trace",
            {
                "trace_id": trace_id,
                "detectors": [
                    {"detector_id": "core.repetition", "parameters": {"unknown": 1}}
                ],
            },
        )


def test_detector_registry_is_described(service):
    data = service.call("list_detectors", {})["data"]
    ids = {item["detector_id"] for item in data["detectors"]}
    assert {
        "core.repetition",
        "core.termination",
        "core.structured_output",
        "core.reconsideration",
        "core.tool_actions",
    } <= ids
    repetition = next(
        item for item in data["detectors"] if item["detector_id"] == "core.repetition"
    )
    assert repetition["version"] == "1.0.0"
    assert "parameters_schema" in repetition
    assert repetition["limitations"]
