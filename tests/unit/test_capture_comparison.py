"""Activation comparison and investigation evidence."""

from __future__ import annotations

import json
import pytest

from wsp.core.errors import ArtifactError, CompatibilityError, InputError
from wsp.investigation.compare import AlignmentSpec, CaptureComparison, compare_captures
from wsp.investigation.models import TraceCaptureRecord
from wsp.tools.service import ToolService



def _details(error: BaseException) -> str:
    """Serialise an error's structured details for assertions."""

    payload = getattr(error, "details", None)
    return json.dumps(payload, default=str) if payload else str(error)


def _record(service: ToolService, text: str, *, max_new_tokens: int = 6) -> str:
    envelope = service.call(
        "record_trace",
        {
            "input_text": text,
            "params": {"max_new_tokens": max_new_tokens},
            "backend": {"kind": "synthetic", "synthetic": {"model": {}}},
        },
    )
    assert envelope["ok"], envelope
    return envelope["data"]["trace"]["trace_id"]


def _capture(service: ToolService, trace_id: str, *, positions=(1, 2), modules=("blocks.1",)):
    envelope = service.call(
        "capture_trace_activations",
        {
            "trace_id": trace_id,
            "model": {"kind": "synthetic"},
            "modules": list(modules),
            "positions": list(positions),
            "mode": "selected",
        },
    )
    assert envelope["ok"], envelope
    return envelope["data"]


def _compare(service: ToolService, left: str, right: str, **overrides):
    payload = {
        "left_capture_artifact_id": left,
        "right_capture_artifact_id": right,
    }
    payload.update(overrides)
    return service.call("compare_activation_captures", payload)


def test_identical_captures_compare_as_identical(service):
    trace_id = _record(service, "the cat sat on the mat")
    left = _capture(service, trace_id)
    right = _capture(service, trace_id)
    assert left["capture_id"] != right["capture_id"]

    envelope = _compare(service, left["capture_artifact_id"], right["capture_artifact_id"])
    assert envelope["ok"], envelope
    data = envelope["data"]

    assert data["schema_version"] == "wsp.capture-comparison/1.0"
    assert data["counts"]["compared"] == len(data["entries"]) == 2
    for entry in data["entries"]:
        metrics = {item["metric"]: item["value"] for item in entry["metrics"]}
        assert metrics["l2"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["cosine"] == pytest.approx(1.0, abs=1e-9)
        assert metrics["max_abs"] == pytest.approx(0.0, abs=1e-12)
    assert len(data["limitations"]) >= 3
    assert any("causal" in limitation for limitation in data["limitations"])
    assert data["unavailable"]["causal_interpretation"]


def test_different_inputs_produce_a_nonzero_difference(service):
    left_trace = _record(service, "the cat sat on the mat")
    right_trace = _record(service, "the dog sat on the mat")
    # Positions 1 and 2 hold tokens ("the", " ") that both prompts share, so a
    # causal model must produce identical states there.  The divergence starts
    # at the first differing token.
    left = _capture(service, left_trace, positions=(3, 4))
    right = _capture(service, right_trace, positions=(3, 4))

    envelope = _compare(service, left["capture_artifact_id"], right["capture_artifact_id"])
    assert envelope["ok"], envelope
    entries = envelope["data"]["entries"]
    metrics = {item["metric"]: item["value"] for item in entries[0]["metrics"]}

    assert metrics["l2"] > 1e-6
    assert metrics["cosine"] < 1.0
    assert envelope["data"]["aggregates"][0]["pairs"] >= 1


def test_comparison_reports_incompatible_modules_without_inventing_numbers(service):
    left_trace = _record(service, "the cat sat on the mat")
    right_trace = _record(service, "the cat sat on the mat")
    left = _capture(service, left_trace, modules=("blocks.1",))
    right = _capture(service, right_trace, modules=("blocks.2",))

    envelope = _compare(service, left["capture_artifact_id"], right["capture_artifact_id"])
    assert envelope["ok"], envelope
    data = envelope["data"]

    assert data["entries"] == []
    assert data["counts"]["entries"] == 0
    assert any("no position pair" in warning for warning in data["warnings"])
    assert data["compatibility"]["same_module_set"] is False


def test_summary_captures_compare_summary_statistics_only(service):
    trace_id = _record(service, "the cat sat on the mat")
    summary = service.call(
        "capture_trace_activations",
        {
            "trace_id": trace_id,
            "model": {"kind": "synthetic"},
            "modules": ["blocks.1"],
            "positions": [1],
        },
    )
    assert summary["ok"], summary
    selected = _capture(service, trace_id, positions=(1,))

    envelope = _compare(
        service,
        summary["data"]["capture_artifact_id"],
        selected["capture_artifact_id"],
    )
    assert envelope["ok"], envelope
    data = envelope["data"]

    assert data["counts"]["without_values"] == 1
    assert data["entries"][0]["values_compared"] is False
    assert data["entries"][0]["metrics"] == []
    assert data["unavailable"]["elementwise_metrics"]
    assert any("different modes" in warning for warning in data["warnings"])


def test_paired_alignment_requires_review_and_pairs_positions(service):
    left_trace = _record(service, "the cat sat on the mat")
    right_trace = _record(service, "the cat sat on the mat")
    left = _capture(service, left_trace)
    right = _capture(service, right_trace, positions=(2, 3))

    with pytest.raises(InputError) as error:
        _compare(
            service,
            left["capture_artifact_id"],
            right["capture_artifact_id"],
            alignment={
                "strategy": "paired_positions",
                "pairs": [{"left_position": 1, "right_position": 2}],
            },
        )
    assert "reviewed" in _details(error.value)

    envelope = _compare(
        service,
        left["capture_artifact_id"],
        right["capture_artifact_id"],
        alignment={
            "strategy": "paired_positions",
            "reviewed": True,
            "pairs": [{"left_position": 1, "right_position": 2, "note": "same event"}],
        },
    )
    assert envelope["ok"], envelope
    entry = envelope["data"]["entries"][0]
    assert entry["left_position"] == 1
    assert entry["right_position"] == 2
    assert envelope["data"]["alignment"]["reviewed"] is True


def test_identical_positions_uses_only_shared_positions(service):
    trace_id = _record(service, "the cat sat on the mat")
    left = _capture(service, trace_id, positions=(1, 2))
    right = _capture(service, trace_id, positions=(2, 3))

    envelope = _compare(service, left["capture_artifact_id"], right["capture_artifact_id"])
    assert envelope["ok"], envelope
    data = envelope["data"]

    assert [entry["left_position"] for entry in data["entries"]] == [2]
    assert data["counts"]["pairs"] == 1


def test_projection_requires_a_direction_and_reports_a_scalar(service):
    trace_id = _record(service, "the cat sat on the mat")
    left = _capture(service, trace_id)
    right = _capture(service, trace_id)

    with pytest.raises(InputError) as error:
        _compare(
            service,
            left["capture_artifact_id"],
            right["capture_artifact_id"],
            metrics=["projection"],
        )
    assert "direction" in _details(error.value)

    direction = [1.0] * 32
    envelope = _compare(
        service,
        left["capture_artifact_id"],
        right["capture_artifact_id"],
        metrics=["l2", "projection"],
        direction=direction,
    )
    assert envelope["ok"], envelope
    entry = envelope["data"]["entries"][0]
    projection = [item for item in entry["metrics"] if item["metric"] == "projection"][0]
    assert projection["value"] == pytest.approx(0.0, abs=1e-9)


def test_wrong_sized_direction_is_reported_not_invented(service):
    trace_id = _record(service, "the cat sat on the mat")
    left = _capture(service, trace_id)
    right = _capture(service, trace_id)

    envelope = _compare(
        service,
        left["capture_artifact_id"],
        right["capture_artifact_id"],
        metrics=["projection"],
        direction=[1.0, 0.0],
    )
    assert envelope["ok"], envelope
    projection = envelope["data"]["entries"][0]["metrics"][0]
    assert projection["value"] is None
    assert "32" in projection["note"] or "elements" in projection["note"]


def test_comparison_of_a_capture_with_itself_is_refused(service):
    trace_id = _record(service, "the cat sat on the mat")
    capture = _capture(service, trace_id)
    with pytest.raises(InputError) as error:
        _compare(service, capture["capture_artifact_id"], capture["capture_artifact_id"])
    assert "different captures" in _details(error.value)


def test_comparison_rejects_non_capture_artifacts(service):
    trace_id = _record(service, "the cat sat on the mat")
    capture = _capture(service, trace_id)
    with pytest.raises(ArtifactError):
        _compare(service, trace_id, capture["capture_artifact_id"])


def test_different_models_are_refused_unless_requested(synthetic_left_right):
    left, right = synthetic_left_right
    with pytest.raises(CompatibilityError) as error:
        compare_captures(
            left,
            right,
            left_vectors={},
            right_vectors={},
        )
    assert "different checkpoints" in str(error.value)


@pytest.fixture
def synthetic_left_right():
    """Two capture records that claim different checkpoints."""

    def record(name: str, checksum: str) -> TraceCaptureRecord:
        return TraceCaptureRecord(
            capture_id="tc_0123456789abcdef",
            created_at="2026-01-01T00:00:00Z",
            trace_id="tr_0123456789abcdef",
            model_name=name,
            model_file_sha256=checksum,
            positions=[1],
            sequence_length=4,
        )

    return record("a", "sha256:" + "a" * 64), record("b", "sha256:" + "b" * 64)


def test_evidence_attaches_to_a_case(service):
    left_trace = _record(service, "the cat sat on the mat")
    right_trace = _record(service, "the dog sat on the mat")
    left = _capture(service, left_trace)
    right = _capture(service, right_trace)
    comparison = _compare(service, left["capture_artifact_id"], right["capture_artifact_id"])
    assert comparison["ok"], comparison

    case = service.call(
        "create_failure_case",
        {
            "title": "loop versus stop",
            "observations": [left_trace],
            "controls": [right_trace],
        },
    )
    assert case["ok"], case
    case_id = case["data"]["case"]["case_id"]

    envelope = service.call(
        "attach_investigation_evidence",
        {
            "case_id": case_id,
            "title": "internal comparison for loop versus stop",
            "summary": "same prompt, one terminates",
            "observation_trace_id": left_trace,
            "control_trace_ids": [right_trace],
            "capture_artifact_ids": [
                left["capture_artifact_id"],
                right["capture_artifact_id"],
            ],
            "comparison_artifact_ids": [comparison["data"]["comparison_artifact_id"]],
            "hypotheses": [
                {
                    "statement": "The stop token changes the next-token distribution.",
                    "related_capture_ids": [left["capture_id"]],
                }
            ],
            "notes": ["measurement only; no causal claim"],
        },
    )
    assert envelope["ok"], envelope
    data = envelope["data"]

    assert data["schema_version"] == "wsp.investigation-evidence/1.0"
    assert data["attached_to_case"] is True
    assert len(data["captures"]) == 2
    assert len(data["comparisons"]) == 1
    assert data["hypotheses"][0]["status"] == "unverified"
    assert data["checksums"]["evidence_sha256"].startswith("sha256:")
    assert any("causal" in limitation for limitation in data["limitations"])

    stored = service.call(
        "inspect_investigation_evidence", {"artifact_id": data["evidence_artifact_id"]}
    )
    assert stored["ok"], stored
    assert stored["data"]["integrity"]["checksum_matches"] is True
    assert stored["data"]["integrity"]["missing_artifacts"] == []
    assert len(stored["data"]["capture_records"]) == 2
    assert stored["data"]["comparison_records"][0]["counts"]["compared"] >= 1

    case_data = service.call("inspect_failure_case", {"case_id": case_id})["data"]
    links = case_data["case"]["investigations"]
    assert len(links) == 1
    assert links[0]["evidence_id"] == data["evidence_id"]
    assert links[0]["capture_ids"] == [item["capture_id"] for item in data["captures"]]


def test_evidence_requires_measurements(service):
    trace_id = _record(service, "the cat sat on the mat")
    with pytest.raises(InputError) as error:
        service.call(
            "attach_investigation_evidence",
            {"title": "empty", "observation_trace_id": trace_id},
        )
    assert "at least one" in _details(error.value)


def test_evidence_rejects_verified_replay_without_agreement(service):
    trace_id = _record(service, "the cat sat on the mat")
    capture = _capture(service, trace_id)
    replay = service.call(
        "verify_replay",
        {"trace_id": trace_id, "model": {"kind": "synthetic"}},
    )
    assert replay["ok"], replay

    envelope = service.call(
        "attach_investigation_evidence",
        {
            "title": "replay evidence",
            "observation_trace_id": trace_id,
            "replay_artifact_ids": [replay["data"]["result_artifact_id"]],
            "capture_artifact_ids": [capture["capture_artifact_id"]],
        },
    )
    assert envelope["ok"], envelope
    data = envelope["data"]
    assert data["replays"][0]["numerical_agreement"] == "verified"
    assert data["replays"][0]["evidence_level"] == "verified"


def test_evidence_records_a_mismatched_replay_honestly(service):
    trace_id = _record(service, "the cat sat on the mat")
    capture = _capture(service, trace_id)
    replay = service.call(
        "verify_replay",
        {
            "trace_id": trace_id,
            "model": {"kind": "synthetic", "synthetic": {"model": {"seed": 99}}},
            "tolerances": {"logprob_atol": 1e-6, "logprob_rtol": 0.0},
        },
    )
    assert replay["ok"], replay

    envelope = service.call(
        "attach_investigation_evidence",
        {
            "title": "honest mismatch",
            "observation_trace_id": trace_id,
            "replay_artifact_ids": [replay["data"]["result_artifact_id"]],
            "capture_artifact_ids": [capture["capture_artifact_id"]],
        },
    )
    assert envelope["ok"], envelope
    data = envelope["data"]
    assert data["replays"][0]["numerical_agreement"] in ("mismatch", "partial", "unavailable")
    assert data["replays"][0]["evidence_level"] == "exploratory"


def test_evidence_rejects_unknown_artifact_kind(service):
    trace_id = _record(service, "the cat sat on the mat")
    capture = _capture(service, trace_id)
    import pytest as _pytest

    with _pytest.raises(ArtifactError):
        service.call(
            "attach_investigation_evidence",
            {
                "title": "wrong kind",
                "observation_trace_id": trace_id,
                "capture_artifact_ids": [capture["capture_artifact_id"]],
                "comparison_artifact_ids": [capture["capture_artifact_id"]],
            },
        )
