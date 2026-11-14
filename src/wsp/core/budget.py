"""Configurable resource budgets.

Defaults target a CPU-only laptop and a 16 GB VRAM GPU.  They are deliberately
conservative; callers may raise them explicitly for a specific operation.
Every limit is enforced before a large allocation or a large response.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from wsp.core.errors import InputError, ResourceLimitError


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Tensor/selection work
    max_tensor_numel: int = Field(default=20_000_000, gt=0)
    max_slice_elements: int = Field(default=5_000_000, gt=0)
    max_preview_elements: int = Field(default=256, gt=0)
    max_analysis_elements: int = Field(default=20_000_000, gt=0)
    max_sample_elements: int = Field(default=200_000, gt=0)
    max_svd_elements: int = Field(default=4_000_000, gt=0)
    max_histogram_bins: int = Field(default=256, gt=0)

    # Checkpoint/manifest/compare work
    max_manifest_tensors: int = Field(default=10_000, gt=0)
    max_compare_tensors: int = Field(default=2_000, gt=0)

    # Activation capture
    max_activation_elements_per_tensor: int = Field(default=500_000, gt=0)
    max_activation_elements_total: int = Field(default=2_000_000, gt=0)
    max_forward_output_elements: int = Field(default=5_000_000, gt=0)
    max_model_load_elements: int = Field(default=50_000_000, gt=0)

    # Patch work
    max_patch_payload_elements: int = Field(default=1_000_000, gt=0)
    max_patchset_tensors: int = Field(default=256, gt=0)

    # Behavioral observation
    max_trace_tokens: int = Field(default=20_000, gt=0)
    max_trace_input_tokens: int = Field(default=8_192, gt=0)
    max_trace_chars: int = Field(default=2_000_000, gt=0)
    max_trace_bytes: int = Field(default=33_554_432, gt=0)
    max_scan_tokens: int = Field(default=200_000, gt=0)
    max_scan_findings: int = Field(default=200, gt=0)
    max_contrast_variants: int = Field(default=16, gt=0)
    max_case_traces: int = Field(default=64, gt=0)
    max_import_traces: int = Field(default=64, gt=0)

    # Replay and internal investigation
    max_replay_sequence_tokens: int = Field(default=4_096, gt=0)
    max_replay_logits_elements: int = Field(default=200_000_000, gt=0)
    max_replay_agreement_tokens: int = Field(default=512, gt=0)
    max_alignment_tokens: int = Field(default=20_000, gt=0)
    max_alignment_span_tokens: int = Field(default=4_096, gt=0)
    max_capture_positions: int = Field(default=1_024, gt=0)
    max_capture_modules: int = Field(default=32, gt=0)
    max_capture_artifacts: int = Field(default=64, gt=0)
    max_comparison_elements: int = Field(default=20_000_000, gt=0)
    max_activation_scan_captures: int = Field(default=64, gt=0)
    max_case_artifacts: int = Field(default=2048, gt=0)

    # Structured output
    max_output_bytes: int = Field(default=2_000_000, gt=0)
    max_artifact_bytes: int = Field(default=268_435_456, gt=0)

    def merge(self, overrides: dict[str, Any] | None) -> "Budget":
        if not overrides:
            return self
        unknown = set(overrides) - set(type(self).model_fields)
        if unknown:
            raise InputError(
                "unknown budget field(s): " + ", ".join(sorted(unknown)),
                details={"allowed": sorted(type(self).model_fields)},
            )
        merged = self.model_dump()
        merged.update(overrides)
        return Budget.model_validate(merged)


DEFAULT_BUDGET = Budget()


def enforce_numel(
    numel: int,
    limit: int,
    *,
    what: str,
    details: dict[str, Any] | None = None,
) -> None:
    if numel > limit:
        payload = {"numel": numel, "limit": limit, "what": what}
        if details:
            payload.update(details)
        raise ResourceLimitError(
            f"{what} has {numel:,} elements, exceeding the configured limit of {limit:,}",
            details=payload,
        )


def enforce_bytes(nbytes: int, limit_bytes: int, *, what: str) -> None:
    if nbytes > limit_bytes:
        raise ResourceLimitError(
            f"{what} is {nbytes:,} bytes, exceeding the configured limit of {limit_bytes:,} bytes",
            details={"bytes": nbytes, "limit_bytes": limit_bytes, "what": what},
        )
