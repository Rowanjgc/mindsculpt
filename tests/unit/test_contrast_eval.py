from __future__ import annotations

import pytest

from wsp.core.errors import InputError, NotFoundError


def _backend(fixtures: list[dict]) -> dict:
    return {"kind": "scripted", "scripted": {"fixtures": fixtures}}


def _fixture(input_text: str, output_text: str, *, tokens: list[str] | None = None) -> dict:
    payload = {
        "input_equals": input_text,
        "output_text": output_text,
    }
    if tokens is not None:
        payload["output_tokens"] = [
            {"text": token, "token_id": index + 1} for index, token in enumerate(tokens)
        ]
    return payload


def _contrast(
    *,
    baseline_text: str = "Canberra",
    variant_text: str = "Canberra",
    expectation: dict | None = None,
    baseline_tokens: list[str] | None = None,
    variant_tokens: list[str] | None = None,
) -> dict:
    return {
        "name": "controlled comparison",
        "baseline": {"name": "baseline", "input_text": "baseline prompt"},
        "variants": [{"name": "variant", "input_text": "variant prompt"}],
        "expectation": expectation
        or {
            "kind": "exact_match",
            "description": "The answer should not change under the perturbation.",
        },
        "backend": _backend(
            [
                _fixture("baseline prompt", baseline_text, tokens=baseline_tokens),
                _fixture("variant prompt", variant_text, tokens=variant_tokens),
            ]
        ),
        "params": {"max_new_tokens": 64},
    }


def test_exact_match_contrast_pass(service, workspace):
    data = service.call("run_contrast", _contrast())["data"]
    assert data["overall_status"] == "pass"
    assert data["baseline"]["role"] == "baseline"
    assert data["outcomes"][0]["status"] == "pass"
    assert data["test"]["baseline"]["input_text"] == "baseline prompt"
    assert data["test"]["variants"][0]["input_text"] == "variant prompt"
    assert data["test"]["expectation"]["description"].startswith("The answer")
    assert len(data["trace_ids"]) == 2
    assert workspace.artifacts.verify(data["artifact_id"])


def test_exact_match_contrast_failure_is_not_a_diagnosis(service):
    data = service.call(
        "run_contrast", _contrast(variant_text="Sydney")
    )["data"]
    assert data["overall_status"] == "fail"
    assert data["outcomes"][0]["status"] == "fail"
    assert "violated the declared invariant" in data["summary"]


def test_reference_agreement_uses_reference_answers(service):
    expectation = {
        "kind": "reference_agreement",
        "description": "The factual answer should remain Canberra.",
        "reference_answers": ["Canberra"],
    }
    passed = service.call(
        "run_contrast", _contrast(expectation=expectation)
    )["data"]
    assert passed["overall_status"] == "pass"
    assert passed["outcomes"][0]["evaluation"]["status"] == "pass"

    failed = service.call(
        "run_contrast",
        _contrast(variant_text="Sydney", expectation=expectation),
    )["data"]
    assert failed["overall_status"] == "fail"
    assert failed["outcomes"][0]["evaluation"]["evaluator_id"] == "core.reference_answer"


def test_reference_agreement_with_failed_baseline_is_inconclusive(service):
    expectation = {
        "kind": "reference_agreement",
        "description": "The factual answer should remain Canberra.",
        "reference_answers": ["Canberra"],
    }
    data = service.call(
        "run_contrast",
        _contrast(
            baseline_text="Sydney",
            variant_text="Sydney",
            expectation=expectation,
        ),
    )["data"]
    assert data["overall_status"] == "inconclusive"
    assert data["outcomes"][0]["status"] == "inconclusive"
    assert "baseline did not establish" in data["outcomes"][0]["explanation"]


def test_format_contrast_preserves_required_contract(service):
    expectation = {
        "kind": "format",
        "description": "Output must remain a JSON object with a name.",
        "contract": {"required_keys": ["name"]},
    }
    passed = service.call(
        "run_contrast", _contrast(baseline_text='{"name": "a"}', variant_text='{"name": "b"}', expectation=expectation)
    )["data"]
    assert passed["overall_status"] == "pass"
    failed = service.call(
        "run_contrast", _contrast(baseline_text='{"name": "a"}', variant_text="not json", expectation=expectation)
    )["data"]
    assert failed["overall_status"] == "fail"


def test_metric_change_contrast(service):
    expectation = {
        "kind": "metric_change",
        "description": "Generation length should increase.",
        "metric": {"name": "generated_tokens", "direction": "increase", "tolerance": 0},
    }
    data = service.call(
        "run_contrast",
        _contrast(
            baseline_tokens=["a", "b", "c"],
            variant_tokens=["a", "b", "c", "d", "e", "f"],
            expectation=expectation,
        ),
    )["data"]
    assert data["overall_status"] == "pass"
    evaluation = data["outcomes"][0]["evaluation"]
    assert evaluation["evaluator_id"] == "core.metric_change"
    assert evaluation["value"]["baseline"] == 3
    assert evaluation["value"]["variant"] == 6

    unchanged = dict(expectation)
    unchanged["metric"] = {"name": "generated_tokens", "direction": "unchanged", "tolerance": 0}
    failed = service.call(
        "run_contrast",
        _contrast(
            baseline_tokens=["a", "b", "c"],
            variant_tokens=["a", "b", "c", "d", "e", "f"],
            expectation=unchanged,
        ),
    )["data"]
    assert failed["overall_status"] == "fail"


def test_property_evaluator_abstention_is_inconclusive(service):
    expectation = {
        "kind": "property_preserved",
        "description": "A reviewer must confirm the property.",
        "evaluator": {
            "evaluator_id": "core.human_judgment",
            "parameters": {"judgment": "abstain", "reviewer": "reviewer-1"},
        },
    }
    data = service.call("run_contrast", _contrast(expectation=expectation))["data"]
    assert data["overall_status"] == "inconclusive"
    assert data["outcomes"][0]["status"] == "inconclusive"
    assert data["outcomes"][0]["evaluation"]["status"] == "abstain"
    assert data["outcomes"][0]["evaluation"]["reviewer"] == "reviewer-1"


def test_variant_execution_error_is_preserved(service):
    payload = _contrast()
    payload["backend"] = _backend([_fixture("baseline prompt", "Canberra")])
    data = service.call("run_contrast", payload)["data"]
    assert data["overall_status"] == "error"
    assert data["outcomes"][0]["status"] == "error"
    assert data["warnings"]
    assert data["baseline"]["trace_id"]


def test_evaluators_distinguish_failure_uncertainty_and_abstention(service, workspace):
    exact = service.call(
        "evaluate_behavior",
        {
            "text": "Canberra",
            "evaluator": {
                "evaluator_id": "core.exact_match",
                "parameters": {"expected": "Canberra"},
            },
        },
    )["data"]
    assert exact["status"] == "pass"
    assert workspace.artifacts.verify(exact["artifact_id"])

    abstained = service.call(
        "evaluate_behavior",
        {
            "evaluator": {
                "evaluator_id": "core.reference_answer",
                "parameters": {"answers": ["Canberra"]},
            }
        },
        raise_errors=False,
    )
    # The schema requires text or a trace, so this is an input error.
    assert abstained["ok"] is False

    uncertain = service.call(
        "evaluate_behavior",
        {
            "text": "Canberra",
            "evaluator": {
                "evaluator_id": "core.human_judgment",
                "parameters": {
                    "judgment": "inconclusive",
                    "confidence": 0.4,
                    "reviewer": "human-1",
                },
            },
        },
    )["data"]
    assert uncertain["status"] == "inconclusive"
    assert uncertain["confidence"] == 0.4
    assert uncertain["reviewer"] == "human-1"

    failed = service.call(
        "evaluate_behavior",
        {
            "text": "Sydney",
            "evaluator": {
                "evaluator_id": "core.reference_answer",
                "parameters": {"answers": ["Canberra"]},
            },
        },
    )["data"]
    assert failed["status"] == "fail"


def test_json_contract_evaluator(service):
    passed = service.call(
        "evaluate_behavior",
        {
            "text": '```json\n{"name": "Ada", "count": 2}\n```',
            "evaluator": {
                "evaluator_id": "core.json_contract",
                "parameters": {
                    "contract": {
                        "required_keys": ["name", "count"],
                        "properties": {"name": "string", "count": "integer"},
                    }
                },
            },
        },
    )["data"]
    assert passed["status"] == "pass"
    failed = service.call(
        "evaluate_behavior",
        {
            "text": '{"name": 3}',
            "evaluator": {
                "evaluator_id": "core.json_contract",
                "parameters": {"contract": {"required_keys": ["name", "count"]}},
            },
        },
    )["data"]
    assert failed["status"] == "fail"
    assert failed["value"]["errors"]


def test_evaluator_abstains_when_trace_text_was_redacted(service, make_trace):
    trace_id = make_trace("secret output", privacy={"store_output_text": False})
    outcome = service.call(
        "evaluate_behavior",
        {
            "trace_id": trace_id,
            "evaluator": {
                "evaluator_id": "core.reference_answer",
                "parameters": {"answers": ["secret"]},
            },
        },
    )["data"]
    assert outcome["status"] == "abstain"
    assert outcome["trace_id"] == trace_id


def test_unknown_evaluator_and_invalid_parameters(service):
    with pytest.raises(NotFoundError):
        service.call(
            "evaluate_behavior",
            {"text": "x", "evaluator": {"evaluator_id": "core.missing"}},
        )
    with pytest.raises(InputError):
        service.call(
            "evaluate_behavior",
            {
                "text": "x",
                "evaluator": {
                    "evaluator_id": "core.exact_match",
                    "parameters": {"unknown": 1},
                },
            },
        )
