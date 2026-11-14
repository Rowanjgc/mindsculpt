"""Tool schemas for activation interventions and counterfactual measurement."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from wsp.replay.sources import ReplayModelSource
from wsp.tools.schemas import ToolInput

Operation = Literal["zero", "mean", "patch", "identity"]
TensorRole = Literal["input", "output"]
PositionSemantics = Literal["consuming", "predicting"]
DonorCondition = Literal["healthy-control", "alternative-input", "other"]
Backend = Literal["torch_hooks", "nnterp"]


class CreateActivationInterventionInput(ToolInput):
    """Build a specification from an existing capture, or from explicit identity."""

    name: str | None = None
    capture_artifact_id: str | None = None
    trace_id: str | None = None
    module: str = Field(min_length=1)
    capture_point: str | None = None
    tensor_role: TensorRole = "output"
    position_semantics: PositionSemantics = "consuming"
    positions: list[int] = Field(min_length=1, max_length=4096)
    axis: int | None = None
    feature_start: int | None = Field(default=None, ge=0)
    feature_end: int | None = Field(default=None, ge=1)
    feature_indices: list[int] = Field(default_factory=list)
    tensor_name: str | None = None
    tensor_index: int | None = Field(default=None, ge=0)
    expected_shape: list[int] | None = None
    expected_dtype: str | None = None

    operation: Operation
    mean_source_capture_artifact_id: str | None = None
    mean_positions: list[int] = Field(default_factory=list)
    mean_token_alignment: str | None = None
    mean_method: Literal["capture-mean", "explicit"] | None = None
    mean_source_trace_ids: list[str] = Field(default_factory=list)
    mean_values: list[float] | None = None
    mean_artifact_id: str | None = None
    mean_notes: str | None = None

    donor_capture_artifact_id: str | None = None
    donor_trace_id: str | None = None
    donor_positions: list[int] = Field(default_factory=list)
    donor_condition: DonorCondition = "healthy-control"
    donor_values: list[float] | None = None
    donor_values_artifact_id: str | None = None
    donor_shape: list[int] | None = None
    donor_dtype: str | None = None
    donor_allow_position_broadcast: bool = False
    donor_note: str | None = None

    model: dict[str, Any] | None = None
    device: str = "cpu"
    dtype: str | None = None
    backend: Backend = "torch_hooks"
    backend_version: str | None = None
    sequence_length: int | None = Field(default=None, ge=1)

    apply_on_calls: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=16)
    allow_dtype_cast: bool = False
    max_elements: int = Field(default=500_000, ge=1)
    max_positions: int = Field(default=128, ge=1)
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    save: bool = True

    @model_validator(mode="after")
    def _coherent(self) -> "CreateActivationInterventionInput":
        if self.capture_artifact_id is None and (self.model is None or self.sequence_length is None):
            raise ValueError(
                "either capture_artifact_id or an explicit model and sequence_length must be given"
            )
        if self.feature_start is not None and self.feature_end is None:
            raise ValueError("feature_end is required when feature_start is set")
        if self.feature_start is None and self.feature_end is not None:
            raise ValueError("feature_start is required when feature_end is set")
        if self.feature_indices and self.feature_start is not None:
            raise ValueError("feature_indices and a feature range are mutually exclusive")
        if self.tensor_name is not None and self.tensor_index is not None:
            raise ValueError("tensor_name and tensor_index are mutually exclusive")
        if self.operation == "mean":
            sources = [
                self.mean_values is not None,
                self.mean_artifact_id is not None,
                self.mean_source_capture_artifact_id is not None,
            ]
            if sum(sources) != 1:
                raise ValueError(
                    "a mean ablation requires exactly one of mean_values, mean_artifact_id, "
                    "or mean_source_capture_artifact_id"
                )
            if self.mean_source_capture_artifact_id is not None and not self.mean_positions:
                raise ValueError("mean_positions are required for a capture-derived mean")
            if self.mean_source_capture_artifact_id is not None and not self.mean_token_alignment:
                raise ValueError(
                    "mean_token_alignment is required: a mean must record how the source "
                    "positions align with the target positions"
                )
        elif any(
            value
            for value in (
                self.mean_values,
                self.mean_artifact_id,
                self.mean_source_capture_artifact_id,
            )
        ):
            raise ValueError(f"operation '{self.operation}' does not take a mean source")
        if self.operation == "patch":
            sources = [
                self.donor_values is not None,
                self.donor_values_artifact_id is not None,
                self.donor_capture_artifact_id is not None,
            ]
            if sum(sources) != 1:
                raise ValueError(
                    "an activation patch requires exactly one of donor_values, "
                    "donor_values_artifact_id, or donor_capture_artifact_id"
                )
            if self.donor_capture_artifact_id is not None and not self.donor_positions:
                raise ValueError("donor_positions are required for a capture-derived donor")
        elif any(
            value
            for value in (
                self.donor_values,
                self.donor_values_artifact_id,
                self.donor_capture_artifact_id,
            )
        ):
            raise ValueError(f"operation '{self.operation}' does not take a donor source")
        return self


class _InterventionReference(ToolInput):
    intervention: dict[str, Any] | None = None
    intervention_artifact_id: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "_InterventionReference":
        if (self.intervention is None) == (self.intervention_artifact_id is None):
            raise ValueError(
                "supply exactly one of intervention (an inline specification) or "
                "intervention_artifact_id"
            )
        return self


class _ExecutionRequest(_InterventionReference):
    source: ReplayModelSource
    token_ids: list[int] | None = Field(default=None, min_length=1, max_length=4096)
    trace_id: str | None = None
    save: bool = True
    case_id: str | None = None

    @model_validator(mode="after")
    def _tokens(self) -> "_ExecutionRequest":
        if not self.token_ids and self.trace_id is None:
            raise ValueError("supply token_ids or a trace_id")
        return self


class PreviewActivationInterventionInput(_ExecutionRequest):
    pass


class ValidateActivationInterventionInput(_ExecutionRequest):
    pass


class RunActivationInterventionInput(_ExecutionRequest):
    measure_modules: list[str] = Field(default_factory=list)
    measure_positions: list[int] = Field(default_factory=list)
    measure_tensor_names: dict[str, str] = Field(default_factory=dict)
    logit_positions: list[int] = Field(default_factory=list)


class RunPairedInterventionInput(RunActivationInterventionInput):
    alternative_intervention: dict[str, Any] | None = None
    direction: list[float] | None = None
    identity: bool = True
    identity_atol: float = Field(default=1e-6, ge=0)
    identity_rtol: float = Field(default=1e-5, ge=0)


class RunCounterfactualContinuationInput(_ExecutionRequest):
    max_new_tokens: int = Field(default=8, ge=1, le=4096)
    max_sequence_length: int = Field(default=2048, ge=2)
    eos_token_id: int | None = Field(default=None, ge=0)
    seed: int | None = None
    stop_on_eos: bool = True
    identity: bool = True


class CompareInterventionResultsInput(ToolInput):
    comparison_artifact_id: str
    include_result_records: bool = True
    max_entries: int = Field(default=50, ge=1, le=500)


class InspectCounterfactualComparisonInput(ToolInput):
    artifact_id: str
    include_records: bool = True


class AttachInterventionEvidenceInput(ToolInput):
    case_id: str | None = None
    title: str = Field(min_length=1)
    summary: str | None = None
    observation_trace_id: str | None = None
    control_trace_ids: list[str] = Field(default_factory=list)
    intervention_artifact_ids: list[str] = Field(default_factory=list)
    comparison_artifact_ids: list[str] = Field(default_factory=list)
    continuation_artifact_ids: list[str] = Field(default_factory=list)
    capture_artifact_ids: list[str] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    save: bool = True

    @model_validator(mode="after")
    def _has_evidence(self) -> "AttachInterventionEvidenceInput":
        if not (
            self.intervention_artifact_ids
            or self.comparison_artifact_ids
            or self.continuation_artifact_ids
        ):
            raise ValueError(
                "at least one intervention, comparison, or continuation artifact is required"
            )
        return self


__all__ = [
    "AttachInterventionEvidenceInput",
    "CompareInterventionResultsInput",
    "CreateActivationInterventionInput",
    "InspectCounterfactualComparisonInput",
    "PreviewActivationInterventionInput",
    "RunActivationInterventionInput",
    "RunCounterfactualContinuationInput",
    "RunPairedInterventionInput",
    "ValidateActivationInterventionInput",
]
