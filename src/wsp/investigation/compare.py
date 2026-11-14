"""Compare two trace-linked activation captures.

This is measurement infrastructure: it reports numerical relationships between
retained activation vectors and nothing else.  A large difference is not a
mechanism and a small one does not rule a mechanism out, so the record carries
no labels of that kind.

Two properties are treated as load-bearing:

* **Compatibility is checked, not assumed.**  Model identity, module identity,
  tensor shape, and the position alignment are validated before any number is
  computed, and incompatibilities are reported rather than silently averaged
  away.
* **Position alignment is explicit.**  Positions in two executions are only
  paired when the caller says so, by strategy or by explicit pairs.  Semantic
  or event-based pairings must be marked as reviewed, because the workbench
  cannot know that two contexts represent the same event.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from wsp.core.budget import Budget
from wsp.core.errors import CompatibilityError, InputError, ResourceLimitError
from wsp.core.models import StatsSummary, TensorDiffStats
from wsp.core.time import utc_now_iso
from wsp.alignment.ids import new_comparison_id
from wsp.investigation.models import Measurement, StrictModel, TraceCaptureRecord
from wsp.tensors.diff import compare_arrays

SCHEMA_COMPARISON = "wsp.capture-comparison/1.0"

AlignmentStrategy = Literal[
    "identical_positions",
    "paired_positions",
    "matching_prefix",
    "semantic_event",
]
MetricName = Literal[
    "l2",
    "rel_l2",
    "cosine",
    "dot",
    "max_abs",
    "mean_abs",
    "rms",
    "projection",
]

#: Strategies that require the investigator to confirm the pairing.  The
#: workbench can pair equal token indices mechanically; it cannot decide that
#: "the token where the loop starts" in one run is the same event as in another.
REVIEW_REQUIRED: frozenset[str] = frozenset({"paired_positions", "semantic_event"})

DEFAULT_METRICS: tuple[str, ...] = ("l2", "rel_l2", "cosine", "max_abs")

COMPARISON_LIMITATIONS = [
    "this record reports numerical relationships between measurements; it does not identify a mechanism",
    "a large activation difference is not evidence of a causal role, and a small one is not evidence against it",
    "vectors come from teacher-forced replays of recorded tokens, not from fresh generations",
]


class PositionPair(StrictModel):
    """One explicit pairing between a left and a right forward position."""

    left_position: int = Field(ge=0)
    right_position: int = Field(ge=0)
    left_token_index: int | None = Field(default=None, ge=0)
    right_token_index: int | None = Field(default=None, ge=0)
    note: str | None = None


class AlignmentSpec(StrictModel):
    """How token positions in the two captures correspond."""

    strategy: AlignmentStrategy = "identical_positions"
    pairs: list[PositionPair] = Field(default_factory=list)
    reviewed: bool = False
    note: str | None = None

    @model_validator(mode="after")
    def _coherent(self) -> "AlignmentSpec":
        if self.strategy in REVIEW_REQUIRED and not self.pairs:
            raise ValueError(f"strategy '{self.strategy}' requires explicit position pairs")
        if self.strategy == "identical_positions" and self.pairs:
            raise ValueError("identical_positions derives pairs from equal positions; do not pass pairs")
        if self.strategy in ("paired_positions", "semantic_event") and not self.reviewed:
            raise ValueError(
                f"strategy '{self.strategy}' must be marked reviewed; the workbench cannot "
                "decide that two contexts represent the same event"
            )
        if len({pair.left_position for pair in self.pairs}) != len(self.pairs):
            raise ValueError("each left position may appear in at most one pair")
        return self


class MetricValue(StrictModel):
    metric: MetricName
    value: float | None = None
    note: str | None = None


class ComparisonEntry(StrictModel):
    """One module/tensor compared at one aligned position pair."""

    module: str
    tensor_role: str
    tensor_name: str
    call_index: int
    left_position: int
    right_position: int
    left_token_index: int | None = None
    right_token_index: int | None = None
    left_shape: list[int] = Field(default_factory=list)
    right_shape: list[int] = Field(default_factory=list)
    left_dtype: str | None = None
    right_dtype: str | None = None
    compatible: bool = True
    values_compared: bool = False
    metrics: list[MetricValue] = Field(default_factory=list)
    diff: TensorDiffStats | None = None
    left_norm: float | None = None
    right_norm: float | None = None
    warnings: list[str] = Field(default_factory=list)


class AggregateComparison(StrictModel):
    """Summary across positions for one module and tensor role."""

    module: str
    tensor_role: str
    pairs: int = 0
    mean_l2: float | None = None
    max_l2: float | None = None
    mean_cosine: float | None = None
    min_cosine: float | None = None


class ComparisonCompatibility(StrictModel):
    same_model: bool = True
    same_module_set: bool = True
    same_position_semantics: bool = True
    same_mode: bool = True
    same_sequence_length: bool = True
    checked_model_identity: bool = True
    notes: list[str] = Field(default_factory=list)


class CaptureComparison(StrictModel):
    schema_version: str = SCHEMA_COMPARISON
    comparison_id: str = Field(pattern=r"^cmp_[0-9a-f]{16}$")
    created_at: str
    left_capture_id: str
    right_capture_id: str
    left_capture_artifact_id: str | None = None
    right_capture_artifact_id: str | None = None
    left_trace_id: str
    right_trace_id: str
    modules: list[str] = Field(default_factory=list)
    alignment: AlignmentSpec = Field(default_factory=AlignmentSpec)
    metrics_requested: list[str] = Field(default_factory=list)
    direction: list[float] | None = None
    direction_note: str | None = None
    entries: list[ComparisonEntry] = Field(default_factory=list)
    aggregates: list[AggregateComparison] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    compatibility: ComparisonCompatibility = Field(default_factory=ComparisonCompatibility)
    elements_compared: int = 0
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    unavailable: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _vector_key(measurement: Measurement) -> tuple[str, str, str, int, int]:
    return (
        measurement.module,
        measurement.tensor_role,
        measurement.tensor_name,
        measurement.call_index,
        measurement.position,
    )


def _norm(vector: np.ndarray) -> float | None:
    value = float(np.linalg.norm(vector))
    return value if math.isfinite(value) else None


def _scalar(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _position_map(record: TraceCaptureRecord) -> dict[tuple[str, str, str, int], dict[int, Measurement]]:
    grouped: dict[tuple[str, str, str, int], dict[int, Measurement]] = {}
    for measurement in record.measurements:
        key = (
            measurement.module,
            measurement.tensor_role,
            measurement.tensor_name,
            measurement.call_index,
        )
        grouped.setdefault(key, {})[measurement.position] = measurement
    return grouped


def _model_identity(record: TraceCaptureRecord) -> tuple[str | None, str | None, str | None]:
    return (record.model_name, record.model_file_sha256, record.model_structure_fingerprint)


def _resolve_metrics(metrics: Iterable[str] | None) -> list[str]:
    requested = list(dict.fromkeys(metrics or DEFAULT_METRICS))
    allowed = set(MetricName.__args__)  # type: ignore[attr-defined]
    for metric in requested:
        if metric not in allowed:
            raise InputError(
                f"unknown comparison metric '{metric}'",
                details={"allowed": sorted(allowed)},
            )
    return requested


def derive_pairs(
    left: TraceCaptureRecord,
    right: TraceCaptureRecord,
    alignment: AlignmentSpec,
) -> tuple[list[PositionPair], list[str]]:
    """Resolve an alignment strategy into concrete position pairs."""

    warnings: list[str] = []
    if alignment.strategy == "identical_positions":
        common = sorted(set(left.positions) & set(right.positions))
        only_left = sorted(set(left.positions) - set(right.positions))
        only_right = sorted(set(right.positions) - set(left.positions))
        if only_left:
            warnings.append(
                f"{len(only_left)} left position(s) have no counterpart in the right capture and were not compared"
            )
        if only_right:
            warnings.append(
                f"{len(only_right)} right position(s) have no counterpart in the left capture and were not compared"
            )
        return [PositionPair(left_position=index, right_position=index) for index in common], warnings

    if alignment.strategy == "matching_prefix":
        length = min(left.sequence_length, right.sequence_length)
        if left.sequence_length != right.sequence_length:
            warnings.append(
                f"the captures cover {left.sequence_length} and {right.sequence_length} positions; "
                f"only the shared prefix of {length} was compared"
            )
        usable = [
            pair
            for pair in (
                PositionPair(left_position=index, right_position=index)
                for index in range(length)
            )
            if pair.left_position in set(left.positions) and pair.right_position in set(right.positions)
        ]
        if not usable:
            warnings.append(
                "no requested position falls inside the shared prefix, so nothing was compared"
            )
        return usable, warnings

    # Explicit pairs: keep only those that both captures actually measured.
    kept: list[PositionPair] = []
    for pair in alignment.pairs:
        if pair.left_position not in set(left.positions) or pair.right_position not in set(right.positions):
            warnings.append(
                f"pair {pair.left_position}->{pair.right_position} was not measured in both captures and was skipped"
            )
            continue
        kept.append(pair)
    return kept, warnings


def compare_captures(
    left: TraceCaptureRecord,
    right: TraceCaptureRecord,
    *,
    left_vectors: Mapping[tuple[str, str, str, int, int], np.ndarray],
    right_vectors: Mapping[tuple[str, str, str, int, int], np.ndarray],
    alignment: AlignmentSpec | None = None,
    metrics: Iterable[str] | None = None,
    direction: Sequence[float] | None = None,
    modules: Iterable[str] | None = None,
    allow_different_models: bool = False,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    budget: Budget | None = None,
    left_artifact_id: str | None = None,
    right_artifact_id: str | None = None,
) -> CaptureComparison:
    """Compare two captures and return a self-describing measurement record."""

    settings = budget or Budget()
    spec = alignment or AlignmentSpec()
    requested = _resolve_metrics(metrics)
    warnings: list[str] = []
    unavailable: dict[str, str] = {}

    left_identity = _model_identity(left)
    right_identity = _model_identity(right)
    same_model = left_identity == right_identity or (
        left_identity[1] is None or right_identity[1] is None
    )
    if not same_model:
        message = "the captures come from different checkpoints, so elementwise metrics are not comparable"
        if not allow_different_models:
            raise CompatibilityError(
                message,
                details={
                    "left_model": left_identity[0],
                    "right_model": right_identity[0],
                    "hint": "pass allow_different_models=true to record the difference as a measurement warning",
                },
            )
        warnings.append(message + " (recorded as warnings at the caller's request)")

    filter_set = set(modules) if modules else None
    pairs, pair_warnings = derive_pairs(left, right, spec)
    warnings.extend(pair_warnings)

    left_map = _position_map(left)
    right_map = _position_map(right)
    keys = sorted(set(left_map) | set(right_map))
    if filter_set is not None:
        keys = [key for key in keys if key[0] in filter_set]

    entries: list[ComparisonEntry] = []
    elements_compared = 0
    elements_limit = int(settings.max_comparison_elements)
    compared_keys: list[tuple[str, str]] = []

    for key in keys:
        module, role, tensor_name, call_index = key
        left_positions = left_map.get(key, {})
        right_positions = right_map.get(key, {})
        for pair in pairs:
            left_measurement = left_positions.get(pair.left_position)
            right_measurement = right_positions.get(pair.right_position)
            if left_measurement is None and right_measurement is None:
                continue
            if left_measurement is None or right_measurement is None:
                warnings.append(
                    f"module '{module}' at position {pair.left_position}->{pair.right_position} "
                    "was measured in only one capture"
                )
                continue

            left_shape = list(left_measurement.slice_shape or left_measurement.shape)
            right_shape = list(right_measurement.slice_shape or right_measurement.shape)
            entry = ComparisonEntry(
                module=module,
                tensor_role=role,
                tensor_name=tensor_name,
                call_index=call_index,
                left_position=pair.left_position,
                right_position=pair.right_position,
                left_token_index=pair.left_token_index,
                right_token_index=pair.right_token_index,
                left_shape=left_shape,
                right_shape=right_shape,
                left_dtype=left_measurement.dtype,
                right_dtype=right_measurement.dtype,
                left_norm=left_measurement.norm,
                right_norm=right_measurement.norm,
            )
            if left_shape != right_shape:
                entry.compatible = False
                entry.warnings.append(
                    "the two measurements have different shapes, so only their summary statistics are comparable"
                )
                entries.append(entry)
                continue
            if left_measurement.dtype != right_measurement.dtype:
                entry.warnings.append(
                    f"source dtypes differ ({left_measurement.dtype} vs {right_measurement.dtype}); "
                    "vectors were compared after conversion to float64"
                )

            left_vector = left_vectors.get(_vector_key(left_measurement))
            right_vector = right_vectors.get(_vector_key(right_measurement))
            if left_vector is None or right_vector is None:
                entry.warnings.append(
                    "at least one capture did not retain values for this position; "
                    "only summary statistics are available"
                )
                unavailable.setdefault(
                    "elementwise_metrics",
                    "one or both captures were stored in summary mode or their vectors are missing",
                )
                entries.append(entry)
                continue

            left_array = np.asarray(left_vector, dtype=np.float64).reshape(-1)
            right_array = np.asarray(right_vector, dtype=np.float64).reshape(-1)
            if left_array.size != right_array.size:
                entry.compatible = False
                entry.warnings.append(
                    f"retained vectors have {left_array.size} and {right_array.size} elements"
                )
                entries.append(entry)
                continue
            elements_compared += int(left_array.size)
            if elements_compared > elements_limit:
                raise ResourceLimitError(
                    "the comparison exceeds the configured element budget",
                    details={"limit": elements_limit, "elements": elements_compared},
                )

            entry.values_compared = True
            diff = compare_arrays(left_array, right_array, atol=atol, rtol=rtol)
            entry.diff = diff
            entry.metrics = _metrics_for(
                left_array,
                right_array,
                requested=requested,
                direction=direction,
                diff=diff,
            )
            entries.append(entry)
            compared_keys.append((module, role))

    aggregate = _aggregate(entries)
    counts = {
        "pairs": len(pairs),
        "entries": len(entries),
        "compared": sum(1 for entry in entries if entry.values_compared),
        "incompatible": sum(1 for entry in entries if not entry.compatible),
        "without_values": sum(1 for entry in entries if entry.compatible and not entry.values_compared),
    }
    if not entries:
        warnings.append("no position pair was measured in both captures for the requested modules")
        unavailable.setdefault("entries", "no comparable measurement was found")

    compatibility = ComparisonCompatibility(
        same_model=same_model,
        same_module_set=set(left.modules) == set(right.modules),
        same_position_semantics=left.position_semantics == right.position_semantics,
        same_mode=left.mode == right.mode,
        same_sequence_length=left.sequence_length == right.sequence_length,
        checked_model_identity=True,
    )
    if not compatibility.same_position_semantics:
        warnings.append(
            "one capture measures consuming positions and the other predicting positions; "
            "the pairing is only meaningful if both semantics were requested deliberately"
        )
    if not compatibility.same_mode:
        warnings.append(
            "the captures used different modes; a summary capture retains no vectors, so elementwise metrics are one-sided"
        )
    if direction is not None and "projection" not in requested:
        warnings.append("a projection direction was supplied but 'projection' was not requested")

    unavailable.setdefault(
        "causal_interpretation",
        "this record contains no causal claim; differences are measurements only",
    )

    return CaptureComparison(
        comparison_id=new_comparison_id(),
        created_at=utc_now_iso(),
        left_capture_id=left.capture_id,
        right_capture_id=right.capture_id,
        left_capture_artifact_id=left_artifact_id,
        right_capture_artifact_id=right_artifact_id,
        left_trace_id=left.trace_id,
        right_trace_id=right.trace_id,
        modules=sorted({entry.module for entry in entries}),
        alignment=spec,
        metrics_requested=requested,
        direction=[float(value) for value in direction] if direction is not None else None,
        direction_note=(
            "projection is the component of the left-minus-right difference along the supplied direction"
            if direction is not None
            else None
        ),
        entries=entries,
        aggregates=aggregate,
        counts=counts,
        compatibility=compatibility,
        elements_compared=elements_compared,
        warnings=warnings,
        limitations=list(COMPARISON_LIMITATIONS),
        unavailable=unavailable,
        metadata={
            "left_backend": left.backend,
            "right_backend": right.backend,
            "left_semantics": left.position_semantics,
            "right_semantics": right.position_semantics,
            "atol": atol,
            "rtol": rtol,
            "compared_module_keys": sorted({f"{module}:{role}" for module, role in compared_keys}),
        },
    )


def _metrics_for(
    left: np.ndarray,
    right: np.ndarray,
    *,
    requested: Sequence[str],
    direction: Sequence[float] | None,
    diff: TensorDiffStats,
) -> list[MetricValue]:
    """Compute the requested metrics from the shared tensor-diff result."""

    values: list[MetricValue] = []
    delta = left - right
    for metric in requested:
        if metric == "l2":
            values.append(MetricValue(metric="l2", value=_scalar(float(np.linalg.norm(delta)))))
        elif metric == "rel_l2":
            denominator = float(np.linalg.norm(right))
            values.append(
                MetricValue(
                    metric="rel_l2",
                    value=_scalar(float(np.linalg.norm(delta)) / denominator) if denominator > 0 else None,
                    note=None if denominator > 0 else "the right vector has zero norm",
                )
            )
        elif metric == "cosine":
            values.append(MetricValue(metric="cosine", value=diff.cosine))
        elif metric == "dot":
            values.append(MetricValue(metric="dot", value=_scalar(float(np.dot(left, right)))))
        elif metric == "max_abs":
            values.append(MetricValue(metric="max_abs", value=diff.max_abs))
        elif metric == "mean_abs":
            values.append(MetricValue(metric="mean_abs", value=diff.mean_abs))
        elif metric == "rms":
            values.append(MetricValue(metric="rms", value=diff.rms))
        elif metric == "projection":
            if direction is None:
                values.append(
                    MetricValue(
                        metric="projection",
                        value=None,
                        note="no direction was supplied",
                    )
                )
                continue
            unit = np.asarray(list(direction), dtype=np.float64).reshape(-1)
            if unit.size != delta.size:
                values.append(
                    MetricValue(
                        metric="projection",
                        value=None,
                        note=f"the direction has {unit.size} elements but the vectors have {delta.size}",
                    )
                )
                continue
            norm = float(np.linalg.norm(unit))
            if norm == 0:
                values.append(
                    MetricValue(metric="projection", value=None, note="the direction has zero norm")
                )
                continue
            values.append(
                MetricValue(metric="projection", value=_scalar(float(np.dot(delta, unit)) / norm))
            )
        else:  # pragma: no cover - guarded by _resolve_metrics
            raise InputError(f"unknown comparison metric '{metric}'")
    return values


def _aggregate(entries: list[ComparisonEntry]) -> list[AggregateComparison]:
    grouped: dict[tuple[str, str], list[ComparisonEntry]] = {}
    for entry in entries:
        if not entry.values_compared:
            continue
        grouped.setdefault((entry.module, entry.tensor_role), []).append(entry)

    aggregates: list[AggregateComparison] = []
    for (module, role), items in sorted(grouped.items()):
        l2_values = [value.value for item in items for value in item.metrics if value.metric == "l2" and value.value is not None]
        cosine_values = [
            value.value for item in items for value in item.metrics if value.metric == "cosine" and value.value is not None
        ]
        aggregates.append(
            AggregateComparison(
                module=module,
                tensor_role=role,
                pairs=len(items),
                mean_l2=float(np.mean(l2_values)) if l2_values else None,
                max_l2=float(np.max(l2_values)) if l2_values else None,
                mean_cosine=float(np.mean(cosine_values)) if cosine_values else None,
                min_cosine=float(np.min(cosine_values)) if cosine_values else None,
            )
        )
    return aggregates


def summary_statistics_delta(left: StatsSummary | None, right: StatsSummary | None) -> dict[str, float | None]:
    """Differences between two activation summaries, when no values were retained."""

    if left is None or right is None:
        return {}
    deltas: dict[str, float | None] = {}
    for field in ("mean", "std", "min", "max", "abs_max"):
        a = getattr(left, field)
        b = getattr(right, field)
        deltas[field] = None if a is None or b is None else float(a - b)
    return deltas


def _validate_entry_shapes(entry: ComparisonEntry) -> None:  # pragma: no cover - reserved
    if not entry.compatible and entry.values_compared:
        raise CompatibilityError("an incompatible entry cannot carry compared values")


__all__ = [
    "SCHEMA_COMPARISON",
    "AlignmentSpec",
    "AlignmentStrategy",
    "AggregateComparison",
    "CaptureComparison",
    "ComparisonCompatibility",
    "ComparisonEntry",
    "COMPARISON_LIMITATIONS",
    "DEFAULT_METRICS",
    "MetricName",
    "MetricValue",
    "PositionPair",
    "REVIEW_REQUIRED",
    "compare_captures",
    "derive_pairs",
    "summary_statistics_delta",
]
