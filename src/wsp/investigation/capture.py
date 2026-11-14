"""Run a trace-linked activation capture on a verified replay.

The capture is executed through the same :func:`wsp.replay.engine.execute_replay`
path as verification, so a capture inherits replay verification, parameter
immutability checks, and hook cleanup; it is never a separately trusted code
path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from wsp.alignment.ids import new_capture_id
from wsp.alignment.models import TokenAlignment
from wsp.core.budget import Budget
from wsp.core.errors import InputError
from wsp.core.hashing import sha256_bytes
from wsp.core.json import canonical_bytes
from wsp.core.models import ArtifactRef
from wsp.core.time import utc_now_iso
from wsp.investigation.collector import PositionedActivationCapture
from wsp.investigation.models import (
    CaptureMode,
    CaptureResources,
    Measurement,
    ModuleSummary,
    PositionSemantics,
    TraceCaptureRecord,
)
from wsp.investigation.points import torch_module_of, validate_modules
from wsp.replay.engine import execute_replay
from wsp.replay.models import ReplayPlan, ReplayResult
from wsp.workspace.artifacts import ArtifactStore

VECTOR_ARTIFACT_KIND = "activation_vector"

CAPTURE_LIMITATIONS = [
    "this record contains numerical measurements only; a large or small difference is not evidence of a mechanism",
    "activations were measured on a teacher-forced replay of recorded tokens, not on a fresh generation",
]


@dataclass
class CaptureOutcome:
    record: TraceCaptureRecord
    result: ReplayResult
    collector: PositionedActivationCapture
    artifacts: list[ArtifactRef] = field(default_factory=list)


def position_targets(
    requested_tokens: Iterable[int],
    semantics: PositionSemantics,
    *,
    sequence_length: int,
) -> tuple[list[int], list[int], list[str]]:
    """Map token indices of interest to forward positions.

    ``consuming`` measures the position of the token itself.  ``predicting``
    measures the preceding position, because that is the state a causal model
    uses to predict the token.  The mapping is explicit and always reported.
    """

    if semantics not in ("consuming", "predicting"):
        raise InputError("position semantics must be 'consuming' or 'predicting'")
    requested = sorted({int(index) for index in requested_tokens})
    for index in requested:
        if not 0 <= index < sequence_length:
            raise InputError(
                f"token index {index} is outside the replayed sequence",
                details={"sequence_length": sequence_length},
            )
    warnings: list[str] = []
    positions: list[int] = []
    dropped: list[int] = []
    for index in requested:
        if semantics == "consuming":
            positions.append(index)
            continue
        if index == 0:
            dropped.append(index)
            warnings.append(
                "token index 0 has no predicting position; the first token of a sequence is not predicted by any earlier state"
            )
            continue
        positions.append(index - 1)
    return sorted(set(positions)), dropped, warnings


def _unavailable(mode: CaptureMode) -> dict[str, str]:
    unavailable = {
        "sampling_reproducibility": "teacher-forced replay does not regenerate tokens",
    }
    if mode == "summary":
        unavailable["elementwise_comparison"] = (
            "summary captures do not retain values; request mode='selected' to compare elementwise"
        )
    return unavailable


def _module_summaries(measurements: list[Measurement]) -> list[ModuleSummary]:
    grouped: dict[tuple[str, str], list[Measurement]] = {}
    for measurement in measurements:
        grouped.setdefault((measurement.module, measurement.tensor_role), []).append(measurement)
    summaries: list[ModuleSummary] = []
    for (module, role), items in grouped.items():
        means = [item.stats.mean for item in items if item.stats is not None and item.stats.mean is not None]
        norms = [item.norm for item in items if item.norm is not None]
        abs_max = [
            item.stats.abs_max for item in items if item.stats is not None and item.stats.abs_max is not None
        ]
        summaries.append(
            ModuleSummary(
                module=module,
                class_name=items[0].class_name,
                tensor_role=role,  # type: ignore[arg-type]
                positions=sorted({item.position for item in items}),
                measurements=len(items),
                mean_of_means=float(sum(means) / len(means)) if means else None,
                mean_norm=float(sum(norms) / len(norms)) if norms else None,
                max_abs=max(abs_max) if abs_max else None,
            )
        )
    return sorted(summaries, key=lambda item: (item.module, item.tensor_role))


def capture_checksum(record: TraceCaptureRecord) -> str:
    payload = record.model_dump(mode="json")
    payload.pop("checksums", None)
    payload.pop("created_at", None)
    return sha256_bytes(canonical_bytes(payload))


def run_capture(
    plan: ReplayPlan,
    model: object,
    *,
    alignment: TokenAlignment | None,
    alignment_artifact_id: str | None = None,
    span_artifact_id: str | None = None,
    requested_tokens: Iterable[int],
    positions: Iterable[int],
    modules: Iterable[str],
    mode: CaptureMode = "summary",
    semantics: PositionSemantics = "consuming",
    include_inputs: bool = False,
    first_call_only: bool = True,
    store: ArtifactStore | None = None,
    budget: Budget | None = None,
    warnings: Iterable[str] = (),
) -> CaptureOutcome:
    settings = budget or Budget()
    sequence_length = len(plan.sequence.full_token_ids)
    module = torch_module_of(model)
    paths = validate_modules(module, modules, budget=settings)

    collector = PositionedActivationCapture(
        module,
        paths,
        positions,
        sequence_length=sequence_length,
        mode=mode,
        record_inputs=include_inputs,
        first_call_only=first_call_only,
        budget=settings,
    )
    execution = execute_replay(plan, model, budget=settings, capture=collector)
    result = execution.result
    if result.forward is None or not result.forward_executed:
        raise InputError("replay did not execute, so no activations could be captured")

    measurements = collector.ordered_measurements()
    artifacts: list[ArtifactRef] = []
    if mode == "selected" and store is not None:
        for measurement in measurements:
            key = (measurement.module, measurement.call_index, measurement.tensor_name, measurement.position)
            vector = collector.vectors.get(key)
            if vector is None:
                continue
            ref = store.put_array(
                VECTOR_ARTIFACT_KIND,
                vector,
                metadata={
                    "capture": "activation_vector",
                    "module": measurement.module,
                    "position": measurement.position,
                    "token_position": measurement.position,
                    "call_index": measurement.call_index,
                    "tensor_name": measurement.tensor_name,
                    "dtype": measurement.dtype,
                    "source_trace": plan.trace_id,
                    "replay_id": result.replay_id,
                    "sampled": measurement.sampled,
                    "schema_version": "wsp.activation-vector/1.0",
                },
            )
            measurement.value_artifact_id = ref.artifact_id
            measurement.value_sha256 = ref.sha256
            artifacts.append(ref)

    total_retained = sum(item.retained_numel for item in measurements)
    record = TraceCaptureRecord(
        capture_id=new_capture_id(),
        created_at=utc_now_iso(),
        trace_id=plan.trace_id,
        trace_artifact_sha256=plan.trace_artifact_sha256,
        plan_id=plan.plan_id,
        replay_id=result.replay_id,
        alignment_id=alignment.alignment_id if alignment is not None else None,
        alignment_artifact_id=alignment_artifact_id,
        span_artifact_id=span_artifact_id,
        model_name=result.identity.model_name,
        model_file_sha256=result.identity.file_sha256,
        model_structure_fingerprint=result.identity.structure_fingerprint,
        model_parameter_count=result.identity.parameter_count,
        backend=result.identity.backend or "torch-forward-hooks",
        backend_version=result.identity.backend_version,
        device=result.identity.device,
        dtype=result.identity.dtype,
        mode=mode,
        position_semantics=semantics,
        requested_tokens=sorted({int(index) for index in requested_tokens}),
        positions=list(collector.positions),
        modules=paths,
        include_inputs=bool(include_inputs),
        first_call_only=bool(first_call_only),
        sequence_length=sequence_length,
        measurements=measurements,
        modules_summary=_module_summaries(measurements),
        resources=CaptureResources(
            positions=len(collector.positions),
            modules=len(paths),
            measurements=len(measurements),
            elements_measured=collector.elements_measured,
            elements_retained=int(total_retained),
            artifacts=len(artifacts),
            estimated_retained_bytes=int(sum(ref.nbytes for ref in artifacts)),
            element_limit_per_tensor=collector.max_elements_per_tensor,
            element_limit_total=collector.max_elements_total,
        ),
        replay_verification={
            "replay_id": result.replay_id,
            "numerical_agreement": result.numerical_agreement,
            "evidence_level": result.evidence_level,
            "forward_executed": result.forward_executed,
            "token_sequence_reconstructed": result.token_sequence_reconstructed,
            "generation_reproducibility": result.generation_reproducibility,
            "agreement": result.agreement.model_dump(mode="json") if result.agreement is not None else None,
        },
        warnings=list(warnings) + list(collector.notes) + list(result.warnings),
        limitations=list(CAPTURE_LIMITATIONS) + list(result.limitations),
        unavailable=_unavailable(mode),
        privacy={
            "mode": "private",
            "contains_activation_values": mode == "selected",
            "note": "activation values may encode information about the source input; treat this artifact as sensitive",
        },
        metadata={
            "sequence_sha256": plan.sequence.sequence_sha256,
            "sequence_source": plan.sequence.source,
            "capture_positions_note": (
                "positions are full-sequence forward positions; requested_tokens are the token "
                "indices whose states were requested"
            ),
        },
    )
    record.checksums["measurement_sha256"] = capture_checksum(record)
    record.checksums["vectors_sha256"] = sha256_bytes(
        canonical_bytes([[ref.artifact_id, ref.sha256] for ref in artifacts])
    )
    return CaptureOutcome(record=record, result=result, collector=collector, artifacts=artifacts)
