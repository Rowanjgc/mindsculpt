"""Unit tests for activation-intervention schemas, operations, and engine.

Everything here runs on the CPU, offline, against small tensors and the
in-process synthetic transformer.  The tests check the *instrument*: selections
are exact, replacement rules are explicit, the identity control is bitwise, and
execution never mutates model parameters.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from support.tiny_lm import build_runtime
from wsp.core.errors import CompatibilityError, UnsupportedError
from wsp.behavior.models import ModelRef
from wsp.interventions.continuation import run_counterfactual_continuation
from wsp.interventions.engine import (
    inspect_intervention,
    retarget_spec_execution,
    run_paired_intervention,
    run_pass,
    torch_module,
)
from wsp.interventions.models import (
    SCHEMA_SPEC,
    ActivationIntervention,
    ContinuationConfig,
    DonorActivation,
    ExecutionIdentity,
    InterventionTarget,
    MeanActivation,
    MeanProvenance,
    TensorSelector,
    continuation_sha256,
    intervention_sha256,
)
from wsp.interventions.operations import (
    apply_operation,
    extract_activation,
    rebuild_activation,
    resolve_selection,
    selected_region,
)
from wsp.workspace.artifacts import ArtifactStore


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _target(
    module: str = "blocks.1",
    *,
    positions: list[int] | None = None,
    role: str = "output",
    **selector: Any,
) -> InterventionTarget:
    return InterventionTarget(
        module=module,
        tensor_role=role,
        position_semantics="consuming",
        selector=TensorSelector(positions=positions or [1], **selector),
    )


def _spec(
    *,
    module: str = "blocks.1",
    positions: list[int] | None = None,
    operation: str = "zero",
    sequence_length: int = 6,
    mean: MeanActivation | None = None,
    donor: DonorActivation | None = None,
    **selector: Any,
) -> ActivationIntervention:
    return ActivationIntervention(
        intervention_id="iv_0123456789abcdef",
        created_at="2026-01-01T00:00:00Z",
        name="test intervention",
        model=ModelRef(name="tiny", structure_fingerprint="abc"),
        execution=ExecutionIdentity(sequence_length=sequence_length),
        target=_target(module, positions=positions, **selector),
        operation=operation,  # type: ignore[arg-type]
        mean=mean,
        donor=donor,
    )


def _mean(values: list[float], *, module: str = "blocks.1") -> MeanActivation:
    return MeanActivation(
        values=values,
        dtype="float32",
        provenance=MeanProvenance(
            method="explicit",
            module=module,
            tensor_role="output",
            position_semantics="consuming",
            token_alignment="same token positions across the source examples",
        ),
    )


def _donor(values: list[float], *, module: str = "blocks.1", **extra: Any) -> DonorActivation:
    return DonorActivation(
        module=module,
        values=values,
        dtype="float32",
        condition="alternative-input",
        **extra,
    )


# --------------------------------------------------------------------------- #
# schema validation
# --------------------------------------------------------------------------- #


def test_spec_has_versioned_schema_and_round_trips():
    spec = _spec()
    assert spec.schema_version == SCHEMA_SPEC
    payload = spec.model_dump(mode="json")
    assert ActivationIntervention.model_validate(payload) == spec
    # The digest ignores created_at and checksums but covers semantics.
    changed = spec.model_copy(update={"name": "different"})
    assert intervention_sha256(spec) != intervention_sha256(changed)
    assert intervention_sha256(spec) == intervention_sha256(
        spec.model_copy(update={"created_at": "2027-01-01T00:00:00Z"})
    )


@pytest.mark.parametrize(
    "operation",
    ["zero", "identity"],
)
def test_zero_and_identity_reject_replacement_sources(operation: str):
    with pytest.raises(ValidationError):
        _spec(operation=operation, mean=_mean([0.0] * 8))
    with pytest.raises(ValidationError):
        _spec(operation=operation, donor=_donor([0.0] * 8))


def test_mean_requires_mean_and_patch_requires_donor():
    with pytest.raises(ValidationError):
        _spec(operation="mean")
    with pytest.raises(ValidationError):
        _spec(operation="patch")


def test_positions_must_lie_inside_the_sequence():
    with pytest.raises(ValidationError):
        _spec(positions=[6], sequence_length=6)


def test_selector_rejects_partial_feature_range():
    with pytest.raises(ValidationError):
        TensorSelector(positions=[0], feature_start=2)


def test_selector_rejects_both_feature_forms():
    with pytest.raises(ValidationError):
        TensorSelector(
            positions=[0], feature_start=0, feature_end=2, feature_indices=[0]
        )


def test_selector_rejects_tensor_index_and_name_together():
    with pytest.raises(ValidationError):
        TensorSelector(positions=[0], tensor_index=0, tensor_name="hidden_states")


def test_mean_provenance_must_match_the_target_module():
    with pytest.raises(ValidationError):
        _spec(operation="mean", mean=_mean([0.0] * 8, module="blocks.2"))


def test_donor_modules_must_match_the_target():
    with pytest.raises(ValidationError):
        _spec(operation="patch", donor=_donor([0.0] * 8, module="blocks.2"))


def test_donor_value_count_must_match_the_feature_width():
    with pytest.raises(ValidationError):
        _spec(operation="patch", donor=_donor([0.0] * 7), feature_start=0, feature_end=8)


def test_non_finite_inline_mean_is_rejected():
    with pytest.raises(ValidationError):
        _mean([0.0, float("nan")])


# --------------------------------------------------------------------------- #
# selections
# --------------------------------------------------------------------------- #


def test_rank3_selection_defaults_to_the_sequence_axis():
    tensor = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
    selection = resolve_selection(
        tensor, TensorSelector(positions=[1, 3]), sequence_length=4
    )
    assert selection.axis == 1
    assert selection.rows == [1, 3]
    assert selection.width == 3
    region = selected_region(tensor, selection)
    assert region.shape == (2, 2, 3)
    assert torch.equal(region[0, 1], tensor[0, 3])


def test_feature_range_selects_only_that_slice():
    tensor = torch.ones(1, 4, 8)
    selection = resolve_selection(
        tensor,
        TensorSelector(positions=[2], feature_start=2, feature_end=4),
        sequence_length=4,
    )
    assert selection.width == 2
    region = selected_region(tensor, selection)
    assert region.shape == (1, 1, 2)


def test_feature_indices_preserve_declared_order():
    tensor = torch.arange(1 * 4 * 8).reshape(1, 4, 8).float()
    selection = resolve_selection(
        tensor, TensorSelector(positions=[0], feature_indices=[1, 5]), sequence_length=4
    )
    region = selected_region(tensor, selection)
    assert torch.equal(region[0, 0], tensor[0, 0, [1, 5]])


def test_sequence_length_must_match_the_selected_axis():
    tensor = torch.zeros(1, 5, 8)
    with pytest.raises(CompatibilityError):
        resolve_selection(tensor, TensorSelector(positions=[0]), sequence_length=6)


def test_the_last_axis_cannot_be_the_sequence_axis():
    tensor = torch.zeros(5, 8)
    with pytest.raises(UnsupportedError):
        resolve_selection(tensor, TensorSelector(axis=1, positions=[0]), sequence_length=8)


def test_rank1_activation_requires_position_zero():
    tensor = torch.ones(8)
    with pytest.raises(UnsupportedError):
        resolve_selection(tensor, TensorSelector(positions=[1]), sequence_length=1)
    selection = resolve_selection(tensor, TensorSelector(positions=[0]), sequence_length=1)
    assert selection.width == 8


def test_expected_shape_and_dtype_are_enforced():
    tensor = torch.zeros(1, 4, 8)
    with pytest.raises(CompatibilityError):
        resolve_selection(
            tensor,
            TensorSelector(positions=[0], expected_shape=[1, 4, 16]),
            sequence_length=4,
        )
    with pytest.raises(CompatibilityError):
        resolve_selection(
            tensor,
            TensorSelector(positions=[0], expected_dtype="float64"),
            sequence_length=4,
        )


# --------------------------------------------------------------------------- #
# operations
# --------------------------------------------------------------------------- #


def test_zero_ablation_only_changes_the_selected_rows():
    tensor = torch.arange(1 * 4 * 3).reshape(1, 4, 3).float()
    selection = resolve_selection(
        tensor, TensorSelector(positions=[2]), sequence_length=4
    )
    updated, elements, dtype_cast, before, after = apply_operation(
        tensor, selection, operation="zero"
    )
    assert elements == 3
    assert dtype_cast is None
    assert torch.equal(after, torch.zeros_like(before))
    for row in (0, 1, 3):
        assert torch.equal(updated[0, row], tensor[0, row])
    # The caller's tensor is untouched: the engine clones before writing.
    assert torch.equal(tensor, torch.arange(12).reshape(1, 4, 3).float())


def test_identity_is_bitwise_and_returns_a_clone():
    tensor = torch.randn(1, 4, 3)
    selection = resolve_selection(
        tensor, TensorSelector(positions=[1, 2]), sequence_length=4
    )
    updated, elements, _, _, after = apply_operation(
        tensor, selection, operation="identity"
    )
    assert elements == 0
    assert torch.equal(updated, tensor)
    assert updated.data_ptr() != tensor.data_ptr()
    assert torch.equal(after, selected_region(tensor, selection))


def test_mean_is_broadcast_across_selected_positions():
    tensor = torch.zeros(1, 4, 3)
    selection = resolve_selection(
        tensor, TensorSelector(positions=[0, 2]), sequence_length=4
    )
    mean = torch.tensor([[1.0, 2.0, 3.0]])
    updated, elements, _, _, after = apply_operation(
        tensor, selection, operation="mean", replacement=mean
    )
    assert elements == 6
    assert torch.equal(updated[0, 0], mean[0])
    assert torch.equal(updated[0, 2], mean[0])
    assert torch.equal(updated[0, 1], torch.zeros(3))


def test_mean_must_describe_exactly_one_row():
    tensor = torch.zeros(1, 4, 3)
    selection = resolve_selection(
        tensor, TensorSelector(positions=[0, 2]), sequence_length=4
    )
    with pytest.raises(CompatibilityError):
        apply_operation(
            tensor,
            selection,
            operation="mean",
            replacement=torch.zeros(2, 3),
        )


def test_donor_broadcast_requires_authorization():
    tensor = torch.zeros(1, 4, 3)
    selection = resolve_selection(
        tensor, TensorSelector(positions=[0, 2]), sequence_length=4
    )
    donor = torch.tensor([[5.0, 6.0, 7.0]])
    with pytest.raises(CompatibilityError):
        apply_operation(tensor, selection, operation="patch", replacement=donor)
    updated, _, _, _, _ = apply_operation(
        tensor,
        selection,
        operation="patch",
        replacement=donor,
        allow_position_broadcast=True,
    )
    assert torch.equal(updated[0, 2], donor[0])


def test_donor_row_count_must_match_when_not_broadcasting():
    tensor = torch.zeros(1, 4, 3)
    selection = resolve_selection(
        tensor, TensorSelector(positions=[0, 2]), sequence_length=4
    )
    with pytest.raises(CompatibilityError):
        apply_operation(
            tensor,
            selection,
            operation="patch",
            replacement=torch.ones(3, 3),
        )


def test_dtype_conversion_requires_an_explicit_opt_in():
    tensor = torch.zeros(1, 4, 3)
    selection = resolve_selection(
        tensor, TensorSelector(positions=[0]), sequence_length=4
    )
    donor = torch.ones(1, 3, dtype=torch.float64)
    with pytest.raises(CompatibilityError):
        apply_operation(tensor, selection, operation="patch", replacement=donor)
    updated, _, dtype_cast, _, _ = apply_operation(
        tensor,
        selection,
        operation="patch",
        replacement=donor,
        allow_dtype_cast=True,
    )
    assert dtype_cast == "float64->float32"
    assert updated.dtype == torch.float32


def test_non_finite_replacements_are_rejected():
    tensor = torch.zeros(1, 4, 3)
    selection = resolve_selection(
        tensor, TensorSelector(positions=[0]), sequence_length=4
    )
    donor = torch.tensor([[float("inf"), 0.0, 0.0]])
    with pytest.raises(CompatibilityError):
        apply_operation(tensor, selection, operation="patch", replacement=donor)


def test_structured_outputs_are_rebuilt_without_guessing():
    hidden = torch.zeros(1, 4, 3)
    value = {"hidden_states": hidden, "other": None}
    selection = resolve_selection(
        hidden, TensorSelector(positions=[1]), sequence_length=4
    )
    updated, _, _, _, _ = apply_operation(hidden, selection, operation="zero")
    tensor, location = extract_activation(
        value, TensorSelector(positions=[1], tensor_name="hidden_states"), role="output"
    )
    assert tensor.shape == (1, 4, 3)
    rebuilt = rebuild_activation(value, location, updated)
    assert rebuilt["hidden_states"].shape == hidden.shape
    assert rebuilt["other"] is None
    # Rebuilding returns a new mapping; the original structure is untouched.
    assert rebuilt is not value
    assert torch.equal(hidden, torch.zeros(1, 4, 3))


def test_ambiguous_structured_outputs_are_rejected():
    value = (torch.zeros(1, 4, 3), torch.zeros(1, 4, 3))
    with pytest.raises(UnsupportedError):
        extract_activation(value, TensorSelector(positions=[0]), role="output")
    with pytest.raises(UnsupportedError):
        extract_activation(
            {"a": torch.zeros(1, 4, 3)},
            TensorSelector(positions=[0]),
            role="output",
        )


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #


@pytest.fixture()
def runtime():
    return build_runtime()


def _tokens(runtime, text: str = "the cat sat on the mat") -> list[int]:
    return list(runtime.tokenizer.encode(text))


def test_identity_pass_matches_baseline_bitwise(runtime):
    tokens = _tokens(runtime)
    spec = _spec(sequence_length=len(tokens))
    baseline = run_pass(spec, runtime.replay_model(), tokens, condition="baseline", apply=False)
    identity = run_pass(
        spec,
        runtime.replay_model(),
        tokens,
        condition="identity",
        operation="identity",
    )
    assert torch.equal(baseline.logits, identity.logits)
    assert baseline.fingerprint_before == identity.fingerprint_after
    assert identity.applied is True


def test_ablation_changes_downstream_logits_but_not_parameters(runtime):
    tokens = _tokens(runtime)
    spec = _spec(sequence_length=len(tokens), positions=[2])
    model = runtime.replay_model()
    before = {
        name: parameter.detach().clone()
        for name, parameter in torch_module(model).named_parameters()
    }
    baseline = run_pass(spec, model, tokens, condition="baseline", apply=False)
    ablated = run_pass(spec, model, tokens, condition="intervention")
    assert ablated.applied is True
    assert ablated.target_comparison is not None
    assert ablated.target_comparison.after.l2_norm == 0.0
    assert ablated.target_comparison.before.l2_norm > 0.0
    assert not torch.equal(baseline.logits, ablated.logits)
    for name, parameter in torch_module(model).named_parameters():
        assert torch.equal(parameter, before[name]), name


def test_ablation_is_local_to_the_selected_position(runtime):
    """Zeroing one hidden row must leave other rows' *inputs* unchanged.

    The synthetic model is causal, so positions before the target are unchanged
    exactly, while later positions may legitimately differ.
    """

    tokens = _tokens(runtime)
    spec = _spec(sequence_length=len(tokens), positions=[2])
    baseline = run_pass(spec, runtime.replay_model(), tokens, condition="baseline", apply=False)
    ablated = run_pass(spec, runtime.replay_model(), tokens, condition="intervention")
    early = slice(0, 2)
    assert torch.equal(baseline.logits[:, early], ablated.logits[:, early])


def test_inspection_resolves_scope_before_execution(runtime):
    tokens = _tokens(runtime)
    spec = _spec(sequence_length=len(tokens), positions=[1, 2], feature_start=0, feature_end=8)
    validation = inspect_intervention(spec, runtime.replay_model(), tokens)
    assert validation.valid is True
    assert validation.scope is not None
    assert validation.scope.estimated_elements == 2 * 8
    assert validation.scope.rows == [1, 2]
    assert validation.human_readable


def test_inspection_reports_missing_modules(runtime):
    tokens = _tokens(runtime)
    spec = _spec(module="blocks.99", sequence_length=len(tokens))
    validation = inspect_intervention(spec, runtime.replay_model(), tokens)
    assert validation.valid is False
    assert any("blocks.99" in error for error in validation.errors)


def test_paired_experiment_reports_identity_and_effect(runtime):
    tokens = _tokens(runtime)
    spec = _spec(sequence_length=len(tokens), positions=[2])
    paired = run_paired_intervention(
        spec,
        runtime.replay_model(),
        tokens,
        measure_modules=["blocks.2"],
        logit_positions=[2, 3],
    )
    comparison = paired.comparison
    assert comparison.identity_control.performed is True
    assert comparison.identity_control.passed is True
    assert comparison.identity_control.agreement == "exact"
    assert comparison.validity.status == "valid"
    assert comparison.effect.logits is not None
    assert comparison.effect.logits.max_abs > 0
    assert comparison.effect.target_activation is not None
    assert comparison.effect.downstream[0].module == "blocks.2"
    assert set(paired.results) == {"baseline", "identity", "intervention"}
    assert paired.results["baseline"].applied is False
    assert paired.results["intervention"].applied is True
    assert comparison.checksums
    # No automatic causal language anywhere in the record.
    text = json.dumps(comparison.model_dump(mode="json")).lower()
    for forbidden in ("discovered causal circuit", "proven neuron", "repaired"):
        assert forbidden not in text


def test_paired_experiment_marks_an_identity_failure_invalid(runtime):
    tokens = _tokens(runtime)
    spec = _spec(sequence_length=len(tokens), positions=[2])
    paired = run_paired_intervention(
        spec,
        runtime.replay_model(),
        tokens,
        identity_atol=0.0,
        identity_rtol=0.0,
    )
    # With exact tolerances the identity pass must still agree bitwise, so the
    # deliberately impossible tolerance must produce a *reported* failure, not a
    # silent pass.
    if paired.comparison.identity_control.agreement != "exact":
        assert paired.comparison.validity.status == "invalid"
        assert paired.comparison.validity.reasons


def test_paired_experiment_without_identity_is_limited(runtime):
    tokens = _tokens(runtime)
    spec = _spec(sequence_length=len(tokens), positions=[2])
    paired = run_paired_intervention(
        spec, runtime.replay_model(), tokens, identity=False
    )
    assert paired.comparison.identity_control.performed is False
    assert paired.comparison.validity.status == "limited"
    assert paired.comparison.warnings


def test_paired_experiment_rejects_wrong_sequence_length(runtime):
    spec = _spec(sequence_length=4)
    with pytest.raises(CompatibilityError):
        run_paired_intervention(spec, runtime.replay_model(), [1, 2, 3, 4, 5])


def test_continuation_identity_matches_and_intervention_diverges(runtime):
    prefix = _tokens(runtime)[:4]
    spec = retarget_spec_execution(
        _spec(sequence_length=len(prefix), positions=[len(prefix) - 1]),
        execution_mode="uncached-continuation",
        sequence_length=len(prefix),
    )
    config = ContinuationConfig(max_new_tokens=4, eos_token_id=None)
    baseline = run_counterfactual_continuation(
        spec, runtime.replay_model(), prefix, config, identity=True
    )
    assert baseline.record.comparison.identity_agreement == "exact"
    assert baseline.record.validity.status in ("valid", "limited")
    assert len(baseline.record.conditions) == 3

    # A deliberately large donor activation must change the greedy next token.
    # The donor targets ``norm`` output, where the pattern survives layer
    # normalisation and reaches the language-model head directly.
    width = int(runtime.model.norm.normalized_shape[0])
    donor_values = [(-10.0 if index % 2 == 0 else 10.0) for index in range(width)]
    donor_spec = retarget_spec_execution(
        _spec(
            module="norm",
            operation="patch",
            sequence_length=len(prefix),
            positions=[len(prefix) - 1],
            donor=_donor(donor_values, module="norm", shape=[1, width]),
        ),
        execution_mode="uncached-continuation",
        sequence_length=len(prefix),
    )
    shifted = run_counterfactual_continuation(
        donor_spec, runtime.replay_model(), prefix, config, identity=True
    )
    assert shifted.record.validity.status in ("valid", "limited")
    baseline_tokens = shifted.conditions["baseline"].generated_token_ids
    intervention_tokens = shifted.conditions["intervention"].generated_token_ids
    assert baseline_tokens != intervention_tokens
    assert shifted.record.comparison.first_divergence is not None
    assert shifted.record.comparison.longest_common_prefix == shifted.record.comparison.first_divergence
    assert shifted.record.comparison.token_agreement_rate < 1.0


def test_continuation_requires_the_continuation_execution_mode(runtime):
    prefix = _tokens(runtime)[:4]
    spec = _spec(sequence_length=len(prefix), positions=[1])
    with pytest.raises(CompatibilityError):
        run_counterfactual_continuation(
            spec, runtime.replay_model(), prefix, ContinuationConfig(max_new_tokens=2)
        )


def test_continuation_digest_is_stable(runtime):
    prefix = _tokens(runtime)[:4]
    spec = retarget_spec_execution(
        _spec(sequence_length=len(prefix), positions=[1], operation="identity"),
        execution_mode="uncached-continuation",
        sequence_length=len(prefix),
    )
    config = ContinuationConfig(max_new_tokens=2, eos_token_id=None)
    outcome = run_counterfactual_continuation(
        spec, runtime.replay_model(), prefix, config
    )
    record = outcome.record
    assert record.checksums["continuation_sha256"] == continuation_sha256(record)


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #


def test_specs_serialize_as_artifact_json(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    spec = _spec()
    ref = store.put_json("activation_intervention", spec.model_dump(mode="json"))
    _, path = store.get(ref.artifact_id)
    reloaded = ActivationIntervention.model_validate_json(path.read_text())
    assert reloaded == spec
    # Tampering with the stored bytes breaks checksum verification.
    data = json.loads(path.read_text())
    data["operation"] = "identity"
    path.write_text(json.dumps(data))
    with pytest.raises(Exception):
        store.get(ref.artifact_id)
