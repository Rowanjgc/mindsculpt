"""Aggregate activation statistics across supplied examples.

Aggregation is explicitly over summary statistics (``mean_of_summaries``), not
over raw pooled activations.  Callers that need pooled statistics must capture
and concatenate the raw values themselves.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wsp.activations.capture import ActivationSummary


class AggregatedActivation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: str
    kind: str
    examples: int
    shapes: list[list[int]]
    mean_of_means: float | None = None
    mean_of_stds: float | None = None
    min_of_mins: float | None = None
    max_of_maxs: float | None = None
    mean_of_abs_max: float | None = None
    total_count: int = 0
    total_non_finite: int = 0
    note: str = "statistics are aggregated across examples, not pooled over raw values"


class ActivationAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.activation-aggregate/1.0"
    method: str = "mean_of_summaries"
    capture_ids: list[str] = Field(default_factory=list)
    groups: list[AggregatedActivation] = Field(default_factory=list)


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(present)) if present else None


def _min(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.min(present)) if present else None


def _max(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.max(present)) if present else None


def aggregate_activations(
    summaries: Iterable[ActivationSummary],
) -> ActivationAggregate:
    items = list(summaries)
    groups: dict[tuple[str, str], list[ActivationSummary]] = {}
    for item in items:
        groups.setdefault((item.module, item.kind), []).append(item)
    aggregated: list[AggregatedActivation] = []
    for (module, kind), group in sorted(groups.items()):
        aggregated.append(
            AggregatedActivation(
                module=module,
                kind=kind,
                examples=len(group),
                shapes=[item.shape for item in group],
                mean_of_means=_mean([item.stats.mean for item in group]),
                mean_of_stds=_mean([item.stats.std for item in group]),
                min_of_mins=_min([item.stats.min for item in group]),
                max_of_maxs=_max([item.stats.max for item in group]),
                mean_of_abs_max=_mean([item.stats.abs_max for item in group]),
                total_count=sum(item.stats.count for item in group),
                total_non_finite=sum(
                    item.stats.nan_count
                    + item.stats.posinf_count
                    + item.stats.neginf_count
                    for item in group
                ),
            )
        )
    return ActivationAggregate(
        capture_ids=sorted({item.capture_id for item in items}),
        groups=aggregated,
    )
