"""Tier 1/2: staged/cached generation verification and cache isolation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from support.temporal import one_plan
from wsp.core.errors import ExecutionError, UnsupportedError
from wsp.execution.staged import (
    HYBRID_ARCHITECTURE_NOTE,
    TinyStagedDecoder,
    assert_clone_isolation,
    build_staged_decoder,
    classify_hf_architecture,
)
from wsp.interventions.plan_models import (
    StagedVerification,
    staged_verification_sha256,
)
from wsp.interventions.staged import (
    run_staged_intervention,
    verify_staged_generation,
)


def _prompt(runtime) -> list[int]:
    return runtime.tokenizer.encode("abc", add_special_tokens=True)


def _staged_plan(runtime, *, steps=(0, 2), max_new_tokens: int = 4):
    return one_plan(
        "blocks.1",
        prompt_length=len(_prompt(runtime)),
        steps=list(steps),
        execution_mode="staged-generation",
        max_new_tokens=max_new_tokens,
    )


def test_tiny_decoder_is_offered_as_supported(tiny_lm):
    decoder = build_staged_decoder(tiny_lm.replay_model())
    capabilities = decoder.capabilities()
    assert isinstance(decoder, TinyStagedDecoder)
    assert capabilities.supported is True
    assert capabilities.architecture_class == "full-attention-transformer"
    assert "key/value" in capabilities.state_kind


def test_staged_generation_agrees_with_the_uncached_reference(tiny_lm):
    plan = _staged_plan(tiny_lm)
    verification, timings = verify_staged_generation(
        plan, tiny_lm.replay_model(), _prompt(tiny_lm)
    )
    assert verification.agreement in ("exact", "within_tolerance")
    assert verification.token_sequence_match is True
    assert verification.termination_match is True
    assert verification.identity_control_agreement == "exact"
    assert verification.max_abs_logits is not None
    assert verification.max_abs_logits <= verification.atol + verification.rtol * 8
    assert verification.steps == plan.continuation.max_new_tokens
    assert verification.state_identity_sha256
    assert verification.state_bytes and verification.state_bytes > 0
    assert verification.position_ids_mode
    assert verification.attention_mask_mode
    assert timings
    assert verification.checksums["staged_verification_sha256"] == (
        staged_verification_sha256(verification)
    )


def test_staged_verification_checks_are_reported(tiny_lm):
    plan = _staged_plan(tiny_lm)
    verification, _ = verify_staged_generation(
        plan, tiny_lm.replay_model(), _prompt(tiny_lm)
    )
    checks = {check.name: check.status for check in verification.checks}
    assert checks["token_sequence"] == "pass"
    assert checks["stepwise_logits"] == "pass"
    assert checks["termination"] == "pass"
    assert checks["identity_control"] == "pass"


def test_staged_stepwise_detail_records_tokens_and_agreement(tiny_lm):
    plan = _staged_plan(tiny_lm)
    verification, _ = verify_staged_generation(
        plan, tiny_lm.replay_model(), _prompt(tiny_lm)
    )
    assert [step.step for step in verification.steps_detail] == [0, 1, 2, 3]
    for step in verification.steps_detail:
        assert step.token_matches is True
        assert step.allclose is True
        assert step.max_abs_logits is not None and step.max_abs_logits >= 0.0


def test_staged_intervention_fires_at_prefill_and_later_steps(tiny_lm):
    plan = _staged_plan(tiny_lm)
    outcome = run_staged_intervention(plan, tiny_lm.replay_model(), _prompt(tiny_lm))
    record = outcome.record
    assert record.validity.status == "valid"
    assert record.verification.agreement in ("exact", "within_tolerance")
    log = outcome.event_logs["plan"]
    assert log.all_fired
    assert [event.generation_step for event in log.events] == [0, 2]
    prefill, decode = log.events
    # The prefill forward carries the whole prefix; the decode forward carries one row.
    assert prefill.tensor_shape[1] == 4
    assert prefill.rows == [3]
    assert decode.tensor_shape[1] == 1
    assert decode.rows == [0]
    # ... and that single row is the full-sequence index prompt_length + step - 1.
    assert decode.token_indices == [4 + 2 - 1]


def test_staged_run_reports_engineering_performance(tiny_lm):
    plan = _staged_plan(tiny_lm)
    record = run_staged_intervention(
        plan, tiny_lm.replay_model(), _prompt(tiny_lm)
    ).record
    performance = record.performance
    assert performance.prefill_ms is not None and performance.prefill_ms >= 0
    assert performance.decode_ms_total is not None
    assert performance.total_ms is not None
    assert performance.uncached_reference_ms is not None
    assert performance.cache_state_bytes and performance.cache_state_bytes > 0
    assert "not model capability benchmarks" in (performance.note or "")


def test_clone_isolation_accepts_a_deep_copy(tiny_lm):
    decoder = TinyStagedDecoder(tiny_lm.model)
    logits, state = decoder.prefill(_prompt(tiny_lm))
    before = decoder.state_hash(state)
    branch = decoder.clone(state)
    decoder.step(branch, 11)
    assert_clone_isolation(decoder, state, before, action="test branch")
    assert decoder.state_hash(state) == before


def test_clone_isolation_detects_shared_mutable_storage(tiny_lm):
    decoder = TinyStagedDecoder(tiny_lm.model)
    _, state = decoder.prefill(_prompt(tiny_lm))
    before = decoder.state_hash(state)

    # A deliberately wrong "clone": the tensors are the same objects.
    shallow = type(state).__new__(type(state))
    shallow.keys = state.keys
    shallow.values = state.values
    shallow.token_ids = list(state.token_ids)

    decoder.step(shallow, 12)
    with pytest.raises(ExecutionError):
        assert_clone_isolation(decoder, state, before, action="test branch")


def test_staged_clone_isolation_is_recorded_on_the_run(tiny_lm):
    plan = _staged_plan(tiny_lm)
    record = run_staged_intervention(
        plan,
        tiny_lm.replay_model(),
        _prompt(tiny_lm),
        isolation="clone",
    ).record
    assert record.verification.isolation == "clone"
    assert record.verification.isolation_verified is True
    assert record.validity.status == "valid"


class _StandIn(torch.nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):  # pragma: no cover - never executed
        raise AssertionError("a rejected architecture must never be executed")


def test_hybrid_architectures_are_rejected_before_execution():
    config = SimpleNamespace(
        model_type="qwen3_5",
        layer_types=["full_attention", "linear_attention"],
    )
    module = _StandIn(SimpleNamespace(text_config=config))
    architecture, reason = classify_hf_architecture(module)
    assert architecture == "hybrid-recurrent"
    assert "linear_attention" in (reason or "")

    with pytest.raises(UnsupportedError) as excinfo:
        build_staged_decoder(module)
    assert HYBRID_ARCHITECTURE_NOTE in str(excinfo.value)


def test_hybrid_model_type_alone_is_rejected():
    config = SimpleNamespace(model_type="qwen3_5_moe")
    module = _StandIn(config)
    architecture, reason = classify_hf_architecture(module)
    assert architecture == "hybrid-recurrent"
    assert "qwen3_5_moe" in (reason or "")


def test_encoder_decoder_config_is_unknown_and_unsupported():
    config = SimpleNamespace(model_type="t5", is_encoder_decoder=True)
    module = _StandIn(config)
    architecture, reason = classify_hf_architecture(module)
    assert architecture == "unknown"
    with pytest.raises(UnsupportedError):
        build_staged_decoder(module)


def test_plain_attention_config_is_classified_as_supported():
    config = SimpleNamespace(model_type="llama", layer_types=["full_attention"] * 2)
    module = _StandIn(config)
    architecture, reason = classify_hf_architecture(module)
    assert architecture == "full-attention-transformer"
    assert reason is None


def test_verification_record_refuses_to_claim_agreement_with_divergent_tokens():
    with pytest.raises(ValidationError):
        StagedVerification(
            verification_id="sv_" + "0" * 16,
            created_at="2026-01-01T00:00:00Z",
            execution_backend="tiny",
            execution_mode="staged-prefill",
            architecture_class="full-attention-transformer",
            prompt_length=4,
            steps=4,
            agreement="within_tolerance",
            token_sequence_match=False,
            isolation="recompute",
        )


def test_verification_record_refuses_shared_isolation_as_verified():
    with pytest.raises(ValidationError):
        StagedVerification(
            verification_id="sv_" + "0" * 16,
            created_at="2026-01-01T00:00:00Z",
            execution_backend="tiny",
            execution_mode="staged-prefill",
            architecture_class="full-attention-transformer",
            prompt_length=4,
            steps=4,
            agreement="exact",
            token_sequence_match=True,
            isolation="shared",
        )


def test_tiny_cached_and_uncached_logits_match_stepwise(tiny_lm):
    # Direct check of the fixture's cached path, independent of the plan layer.
    decoder = TinyStagedDecoder(tiny_lm.model)
    prompt = _prompt(tiny_lm)
    logits, state = decoder.prefill(prompt)
    cached = [logits[0, -1]]
    tokens = [int(torch.argmax(logits[0, -1]).item())]
    for _ in range(3):
        logits, state = decoder.step(state, tokens[-1])
        cached.append(logits[0, -1])
        tokens.append(int(torch.argmax(logits[0, -1]).item()))

    sequence = list(prompt)
    uncached = []
    with torch.inference_mode():
        for _ in range(4):
            full = tiny_lm.model(torch.tensor([sequence]))
            uncached.append(full[0, -1])
            token = int(torch.argmax(full[0, -1]).item())
            sequence.append(token)
    for left, right in zip(cached, uncached):
        assert torch.allclose(left.to(torch.float64), right.to(torch.float64), atol=1e-5, rtol=1e-5)


def test_staged_plan_with_absolute_positions_beyond_step_zero_is_rejected(tiny_lm):
    from wsp.interventions.plan import validate_plan
    from support.temporal import make_plan, make_plan_intervention, make_spec

    spec = make_spec("blocks.1", prompt_length=4, positions=[1], label="abs")
    plan = make_plan(
        [make_plan_intervention(spec, order=0, schedule=[2], position_mode="absolute")],
        prompt_length=4,
        execution_mode="staged-generation",
    )
    validation = validate_plan(plan, root=tiny_lm.model)
    assert validation.valid is False
    assert any("staged" in item for item in validation.errors)
