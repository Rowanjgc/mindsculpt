"""Replay plan and teacher-forced execution semantics.

These tests use a real (tiny) causal transformer and a real tokenizer, both
constructed locally.  Replay must reproduce the recorded log-probabilities
exactly for this fixture because generation re-ran the full prefix at every
step; any discrepancy would therefore be a bug in position handling or in the
comparison, not floating-point luck.
"""

from __future__ import annotations

import pytest
import torch

from wsp.core.budget import Budget
from wsp.core.errors import CompatibilityError, ExecutionError, InputError, ResourceLimitError
from wsp.core.json import dumps
from wsp.replay import (
    ReplayExecutionConfig,
    ReplayTolerances,
    TorchReplayModel,
    execute_replay,
    plan_checksum,
    prepare_plan,
)
from support.tiny_lm import CharTokenizer, TinyCausalLM, TinyRuntime, make_trace


def _rewrite(plan, trace):
    return prepare_plan(trace)


def test_plan_records_the_exact_sequence(tiny_lm):
    trace = make_trace(tiny_lm, "the cat sat on the mat", max_new_tokens=6)
    plan = prepare_plan(trace)

    assert plan.schema_version == "wsp.replay-plan/1.0"
    assert plan.semantics == "teacher-forced"
    assert plan.sequence.source == "recorded-token-ids"
    assert plan.sequence.input_token_ids == trace.input.token_ids
    assert plan.sequence.generated_token_ids == [token.token_id for token in trace.tokens]
    assert plan.sequence.full_length == plan.sequence.prompt_length + plan.sequence.generated_length
    statuses = {check.name: check.status for check in plan.checks}
    assert statuses["generated_token_ids"] == "pass"
    assert statuses["recorded_logprobs"] == "pass"
    assert statuses["generation_reproducibility"] == "unavailable"


def test_teacher_forced_replay_reproduces_recorded_logprobs(tiny_lm):
    trace = make_trace(tiny_lm, "hello there", max_new_tokens=8)
    plan = prepare_plan(trace)
    execution = execute_replay(plan, tiny_lm.replay_model())
    result = execution.result

    assert result.schema_version == "wsp.replay/1.0"
    assert result.token_sequence_reconstructed is True
    assert result.forward_executed is True
    assert result.numerical_agreement == "verified"
    assert result.evidence_level == "verified"
    assert result.exploratory_reason is None
    assert result.agreement is not None
    assert result.agreement.tokens_compared == len(trace.tokens)
    assert result.agreement.top1_agreement_rate == 1.0
    assert result.agreement.max_abs_diff is not None
    assert result.agreement.max_abs_diff < 1e-6
    assert result.forward is not None
    assert result.forward.sequence_length == plan.sequence.full_length
    assert result.forward.logits_shape == [1, plan.sequence.full_length, tiny_lm.tokenizer.vocab_size]


def test_teacher_forcing_never_claims_generation_reproducibility(tiny_lm):
    trace = make_trace(tiny_lm, "and the loop", max_new_tokens=6)
    result = execute_replay(prepare_plan(trace), tiny_lm.replay_model()).result

    assert result.generation_reproducibility == "not-claimed"
    check = next(check for check in result.checks if check.name == "generation_reproducibility")
    assert check.status == "unavailable"
    assert any("does not regenerate" in limitation for limitation in result.limitations)


def test_prediction_position_differs_from_token_position(tiny_lm):
    trace = make_trace(tiny_lm, "a dog", max_new_tokens=4)
    plan = prepare_plan(trace)
    result = execute_replay(plan, tiny_lm.replay_model()).result

    assert result.agreement is not None
    prompt_length = plan.sequence.prompt_length
    for index, row in enumerate(result.agreement.per_token):
        # The state that predicts generated token ``index`` is the position
        # immediately before it, never the token's own position.
        assert row.predicting_position == prompt_length - 1 + index
        assert row.generated_index == index
        assert plan.sequence.full_token_ids[row.predicting_position] != row.token_id or index == 0


def test_redacted_or_absent_token_ids_require_explicit_reconstruction(tiny_lm):
    trace = make_trace(tiny_lm, "hello", max_new_tokens=4)
    stripped = trace.model_copy(
        update={
            "input": trace.input.model_copy(
                update={"token_ids": None, "tokens_sha256": None}
            )
        }
    )
    with pytest.raises(InputError) as excinfo:
        prepare_plan(stripped)
    assert "token identifiers" in str(excinfo.value)

    rebuilt = prepare_plan(
        stripped,
        encoder=lambda text, add_special_tokens=True: tiny_lm.tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ),
    )
    assert rebuilt.sequence.source == "re-derived-from-text"
    assert rebuilt.sequence.warnings
    result = execute_replay(rebuilt, tiny_lm.replay_model()).result
    assert result.numerical_agreement == "verified"


def test_chat_template_prompt_is_never_guessed(tiny_lm):
    trace = make_trace(tiny_lm, "hello", max_new_tokens=2)
    chat = trace.model_copy(
        update={
            "input": trace.input.model_copy(
                update={
                    "token_ids": None,
                    "tokens_sha256": None,
                    "token_encoding": "chat_template",
                }
            )
        }
    )
    with pytest.raises(InputError) as excinfo:
        prepare_plan(chat, encoder=lambda text, add=True: [1, 2, 3])
    assert "chat template" in str(excinfo.value)


def test_incompatible_tokenizer_is_rejected(tiny_lm):
    trace = make_trace(tiny_lm, "hello", max_new_tokens=3)
    plan = prepare_plan(trace)
    other_tokenizer = CharTokenizer(extra_tokens=["xyzzy"])
    other_model = TinyCausalLM(other_tokenizer.vocab_size, seed=7)
    other = TorchReplayModel(other_model, name="other", tokenizer=other_tokenizer)

    with pytest.raises(CompatibilityError) as excinfo:
        execute_replay(plan, other)
    detail = excinfo.value.details
    names = {entry["name"] for entry in detail["checks"]}
    assert "tokenizer_fingerprint" in names


def test_sequence_length_limit_is_enforced(tiny_lm):
    trace = make_trace(tiny_lm, "the cat sat on the mat", max_new_tokens=12)
    with pytest.raises(ResourceLimitError) as excinfo:
        prepare_plan(trace, execution=ReplayExecutionConfig(max_sequence_length=8))
    assert excinfo.value.details["limit"] == 8


def test_logits_element_budget_is_enforced(tiny_lm):
    trace = make_trace(tiny_lm, "hello", max_new_tokens=4)
    plan = prepare_plan(trace)
    budget = Budget(max_replay_logits_elements=16)
    with pytest.raises(ResourceLimitError) as excinfo:
        execute_replay(plan, tiny_lm.replay_model(), budget=budget)
    assert "logits" in str(excinfo.value)


def test_parameter_mutation_is_detected(tiny_lm):
    class Mutating(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            logits = self.inner(input_ids)
            with torch.no_grad():
                self.inner.blocks[0].attn.qkv.weight[0, 0] += 0.5
            return logits

    trace = make_trace(tiny_lm, "hello", max_new_tokens=2)
    # The wrapper is a different module tree, so drop the recorded structure
    # fingerprint: this test isolates mutation detection, not identity checking.
    trace.model = trace.model.model_copy(update={"structure_fingerprint": None})
    plan = prepare_plan(trace)
    model = TorchReplayModel(Mutating(tiny_lm.model), tokenizer=tiny_lm.tokenizer)
    with pytest.raises(ExecutionError) as excinfo:
        execute_replay(plan, model)
    assert "changed during replay" in str(excinfo.value)


def test_replay_does_not_mutate_parameters(tiny_lm):
    trace = make_trace(tiny_lm, "hello", max_new_tokens=4)
    before = {name: value.detach().clone() for name, value in tiny_lm.model.state_dict().items()}
    result = execute_replay(prepare_plan(trace), tiny_lm.replay_model()).result
    after = tiny_lm.model.state_dict()

    assert set(before) == set(after)
    for name, value in before.items():
        assert torch.equal(value, after[name]), name
    check = next(check for check in result.checks if check.name == "parameters_unchanged")
    assert check.status == "pass"


def test_capture_hooks_are_closed_when_forward_fails(tiny_lm):
    class Exploding(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("deliberate failure")

    class DummyCapture:
        def __init__(self) -> None:
            self.installed = 0
            self.closed = 0

        def install(self) -> None:
            self.installed += 1

        def close(self) -> None:
            self.closed += 1

    trace = make_trace(tiny_lm, "hello", max_new_tokens=2)
    trace.model = trace.model.model_copy(update={"structure_fingerprint": None})
    plan = prepare_plan(trace)
    capture = DummyCapture()
    with pytest.raises(ExecutionError):
        execute_replay(
            plan,
            TorchReplayModel(Exploding(), tokenizer=tiny_lm.tokenizer),
            capture=capture,
        )
    assert capture.installed == 1
    assert capture.closed == 1


def test_absent_logprobs_are_reported_as_unavailable(tiny_lm):
    trace = make_trace(tiny_lm, "hello", max_new_tokens=4)
    bare = trace.model_copy(
        update={
            "tokens": [
                token.model_copy(update={"logprob": None}) for token in trace.tokens
            ]
        }
    )
    result = execute_replay(prepare_plan(bare), tiny_lm.replay_model()).result

    assert result.forward_executed is True
    assert result.numerical_agreement == "unavailable"
    assert result.evidence_level == "exploratory"
    assert result.exploratory_reason is not None
    assert result.agreement is not None
    assert result.agreement.unavailable_reason is not None


def test_falsified_logprobs_report_a_mismatch(tiny_lm):
    trace = make_trace(tiny_lm, "hello", max_new_tokens=8)
    shifted = trace.model_copy(
        update={
            "tokens": [
                token.model_copy(
                    update={"logprob": (token.logprob or 0.0) + 5.0}
                )
                for token in trace.tokens
            ]
        }
    )
    result = execute_replay(prepare_plan(shifted), tiny_lm.replay_model()).result

    assert result.numerical_agreement == "mismatch"
    assert result.evidence_level == "exploratory"
    assert result.agreement is not None
    assert result.agreement.within_tolerance_fraction == 0.0
    check = next(check for check in result.checks if check.name == "numerical_agreement")
    assert check.status == "fail"


def test_partial_agreement_is_reported_as_partial(tiny_lm):
    trace = make_trace(tiny_lm, "hello", max_new_tokens=8)
    tokens = []
    for index, token in enumerate(trace.tokens):
        if index % 2 == 0:
            tokens.append(token.model_copy(update={"logprob": (token.logprob or 0.0) + 5.0}))
        else:
            tokens.append(token)
    result = execute_replay(
        prepare_plan(trace.model_copy(update={"tokens": tokens})),
        tiny_lm.replay_model(),
    ).result

    assert result.numerical_agreement == "partial"
    assert 0.0 < (result.agreement.within_tolerance_fraction or 0.0) < 1.0


def test_plan_checksum_is_deterministic(tiny_lm):
    trace = make_trace(tiny_lm, "hello", max_new_tokens=3)
    plan = prepare_plan(trace)
    assert plan_checksum(plan) == plan_checksum(plan)
    assert plan_checksum(plan).startswith("sha256:")


def test_replay_result_is_strict_json(tiny_lm):
    trace = make_trace(tiny_lm, "hello", max_new_tokens=3)
    result = execute_replay(prepare_plan(trace), tiny_lm.replay_model()).result
    payload = dumps(result.model_dump(mode="json"))
    assert "NaN" not in payload
    assert '"generation_reproducibility": "not-claimed"' in payload


def test_tolerances_are_configurable_and_reported(tiny_lm):
    trace = make_trace(tiny_lm, "hello", max_new_tokens=4)
    shifted = trace.model_copy(
        update={
            "tokens": [
                token.model_copy(update={"logprob": (token.logprob or 0.0) + 0.01})
                for token in trace.tokens
            ]
        }
    )
    tight = execute_replay(
        prepare_plan(shifted, tolerances=ReplayTolerances(logprob_atol=1e-6, logprob_rtol=0.0)),
        tiny_lm.replay_model(),
    ).result
    assert tight.numerical_agreement == "mismatch"

    loose = execute_replay(
        prepare_plan(shifted, tolerances=ReplayTolerances(logprob_atol=0.5, logprob_rtol=0.0)),
        tiny_lm.replay_model(),
    ).result
    assert loose.numerical_agreement == "verified"
    assert loose.agreement is not None
    assert loose.agreement.tolerance.logprob_atol == 0.5
