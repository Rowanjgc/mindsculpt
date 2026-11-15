"""Agent-readable investigation dossiers.

A dossier is deliberately smaller than the artifacts it references.  It
separates what was observed, what was measured, which controls passed, what an
investigator proposed, and what cannot be concluded.  It never invents a
mechanism or a repair claim.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Sequence

from pydantic import Field

from wsp.core.hashing import sha256_bytes
from wsp.core.time import utc_now_iso
from wsp.investigation.models import StrictModel

SCHEMA_DOSSIER = "wsp.investigation-dossier/1.0"

DOSSIER_LIMITATIONS = [
    "A dossier summarises recorded evidence; it does not add scientific conclusions.",
    "A measured effect is not a mechanism, a repair, or a validated weight edit.",
    "Controls establish internal validity for the recorded conditions only.",
    "Token repetition or any other event is not by itself evidence of a failure.",
]


def _new(prefix: str) -> str:
    return f"{prefix}_{__import__('secrets').token_hex(8)}"


class DossierObservation(StrictModel):
    kind: str
    summary: str
    artifact_id: str | None = None
    trace_id: str | None = None
    event_ids: list[str] = Field(default_factory=list)
    token_position: int | None = Field(default=None, ge=0)
    generation_step: int | None = Field(default=None, ge=0)
    alignment_status: str | None = None


class DossierMeasurement(StrictModel):
    name: str
    condition: str | None = None
    position: int | None = Field(default=None, ge=0)
    token_id: int | None = Field(default=None, ge=0)
    value: float | None = None
    unit: str | None = None
    artifact_id: str | None = None


class DossierControl(StrictModel):
    name: str
    status: Literal["pass", "fail", "skipped", "warning"]
    detail: str | None = None
    artifact_id: str | None = None


class DossierHypothesis(StrictModel):
    hypothesis_id: str | None = None
    statement: str
    rationale: str | None = None
    status: Literal["proposed", "testing", "supported", "refuted", "inconclusive"] = "proposed"
    proposed_by: str | None = None
    artifact_id: str | None = None


class DossierNextOperation(StrictModel):
    operation: str
    reason: str
    inputs_hint: dict[str, Any] = Field(default_factory=dict)


class InvestigationDossier(StrictModel):
    schema_version: Literal[SCHEMA_DOSSIER] = SCHEMA_DOSSIER
    dossier_id: str = Field(pattern=r"^dos_[0-9a-f]{16}$")
    created_at: str = Field(default_factory=utc_now_iso)
    title: str
    case_id: str | None = None
    trace_id: str | None = None
    event_ids: list[str] = Field(default_factory=list)
    observations: list[DossierObservation] = Field(default_factory=list)
    measurements: list[DossierMeasurement] = Field(default_factory=list)
    controls: list[DossierControl] = Field(default_factory=list)
    hypotheses: list[DossierHypothesis] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: list(DOSSIER_LIMITATIONS))
    next_operations: list[DossierNextOperation] = Field(default_factory=list)
    artifact_refs: list[dict[str, Any]] = Field(default_factory=list)
    summary: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)


def dossier_sha256(dossier: InvestigationDossier) -> str:
    payload = dossier.model_dump(mode="json")
    payload.get("checksums", {}).pop("dossier_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + sha256_bytes(canonical.encode("utf-8")).split(":", 1)[1]


def _measurements_from_comparison(comparison: Any) -> list[DossierMeasurement]:
    items: list[DossierMeasurement] = []
    rows = getattr(comparison, "measurements", None)
    if rows is None:
        effect = getattr(comparison, "effect", None)
        rows = []
        for delta in getattr(effect, "token_logprobs", []) or []:
            for name, value, condition in (
                ("baseline_logprob", delta.baseline, "baseline"),
                ("identity_logprob", delta.identity, "identity"),
                ("intervention_logprob", delta.intervention, "intervention"),
                ("intervention_delta", delta.intervention_delta, "intervention"),
            ):
                if value is None:
                    continue
                items.append(
                    DossierMeasurement(
                        name=name,
                        condition=condition,
                        position=delta.position,
                        token_id=delta.token_id,
                        value=float(value),
                        unit="log-probability",
                    )
                )
        for item in getattr(effect, "downstream", []) or []:
            items.append(
                DossierMeasurement(
                    name="downstream_activation_max_abs",
                    condition="intervention",
                    value=float(item.diff.max_abs),
                    unit="absolute activation",
                )
            )
        return items
    for measurement in rows:
        for name, value, unit in (
            ("baseline_logprob", measurement.baseline_logprob, "log-probability"),
            ("identity_logprob", measurement.identity_logprob, "log-probability"),
            ("intervention_logprob", measurement.intervention_logprob, "log-probability"),
            ("intervention_delta", measurement.intervention_delta, "log-probability"),
        ):
            if value is None:
                continue
            items.append(
                DossierMeasurement(
                    name=name,
                    condition=name.split("_", 1)[0] if name != "intervention_delta" else "intervention",
                    position=measurement.position,
                    token_id=measurement.token_id,
                    value=float(value),
                    unit=unit,
                )
            )
    for item in getattr(comparison, "downstream", []) or []:
        items.append(
            DossierMeasurement(
                name="downstream_activation_max_abs",
                condition="intervention",
                position=None,
                value=float(item.diff.max_abs),
                unit="absolute activation",
            )
        )
    return items


def _controls_from_comparison(comparison: Any, artifact_id: str | None) -> list[DossierControl]:
    controls: list[DossierControl] = []
    identity = comparison.identity_control
    controls.append(
        DossierControl(
            name="identity_control",
            status=(
                "pass"
                if identity.passed
                else "fail"
                if identity.performed
                else "skipped"
            ),
            detail=identity.detail,
            artifact_id=artifact_id,
        )
    )
    for check in comparison.validity.checks:
        controls.append(
            DossierControl(
                name=check.name,
                status=check.status if check.status in ("pass", "fail", "skipped") else "warning",
                detail=check.detail,
                artifact_id=artifact_id,
            )
        )
    seen = {control.name for control in controls}
    for check in comparison.metadata.get("checks", []) if isinstance(comparison.metadata, dict) else []:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name"))
        if name in seen:
            continue
        status = check.get("status")
        controls.append(
            DossierControl(
                name=name,
                status=status if status in ("pass", "fail", "skipped") else "warning",
                detail=check.get("detail"),
                artifact_id=artifact_id,
            )
        )
        seen.add(name)
    return controls


def _next_operations(
    *,
    has_event: bool,
    comparison: Any | None,
    identity_failed: bool,
) -> list[DossierNextOperation]:
    operations: list[DossierNextOperation] = []
    if not has_event:
        operations.extend(
            [
                DossierNextOperation(
                    operation="detect_behavioral_events",
                    reason="no exact event was supplied; detect a deterministic event first",
                    inputs_hint={"trace_id": "<trace>"},
                ),
                DossierNextOperation(
                    operation="discover_capabilities",
                    reason="list observation and investigation capabilities for this workspace",
                    inputs_hint={"query": "behavioral event"},
                ),
            ]
        )
        return operations
    if comparison is None:
        operations.extend(
            [
                DossierNextOperation(
                    operation="prepare_fixed_context_replay",
                    reason="reconstruct the exact prefix for the selected event",
                    inputs_hint={"trace_id": "<trace>", "event_id": "<event>"},
                ),
                DossierNextOperation(
                    operation="create_activation_intervention",
                    reason="choose a module, position, and operation before testing a hypothesis",
                    inputs_hint={},
                ),
            ]
        )
        return operations
    if identity_failed:
        operations.append(
            DossierNextOperation(
                operation="run_fixed_context_comparison",
                reason=(
                    "the identity control failed; repeat the comparison before attributing "
                    "any difference to the intervention"
                ),
                inputs_hint={"replay_artifact_id": "<replay>", "identity": True},
            )
        )
        return operations
    operations.extend(
        [
            DossierNextOperation(
                operation="run_fixed_context_comparison",
                reason="test a different module, position, operation, or feature region",
                inputs_hint={"replay_artifact_id": "<replay>"},
            ),
            DossierNextOperation(
                operation="attach_event_evidence",
                reason="attach the event, replay, and comparison to a failure case",
                inputs_hint={"case_id": "<case>"},
            ),
            DossierNextOperation(
                operation="list_investigation_recipes",
                reason="use a higher-level recipe for a multi-step follow-up",
                inputs_hint={},
            ),
        ]
    )
    return operations


def build_dossier(
    *,
    title: str,
    case_id: str | None = None,
    trace_id: str | None = None,
    events: Sequence[Any] = (),
    comparison: Any | None = None,
    comparison_artifact_id: str | None = None,
    condition_artifact_ids: Sequence[str] = (),
    hypotheses: Sequence[Any] = (),
    extra_limitations: Sequence[str] = (),
    artifact_refs: Sequence[dict[str, Any]] = (),
    summary: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> InvestigationDossier:
    event_items = list(events)
    observations: list[DossierObservation] = []
    for event in event_items:
        observations.append(
            DossierObservation(
                kind=event.event_type,
                summary=(
                    f"{event.event_type} observed with alignment "
                    f"'{event.alignment.status}'"
                ),
                trace_id=event.trace_id,
                event_ids=[event.event_id],
                token_position=event.token_position,
                generation_step=event.generation_step,
                alignment_status=event.alignment.status,
            )
        )
    if comparison is not None:
        observations.append(
            DossierObservation(
                kind="fixed_context_comparison",
                summary=(
                    "baseline, identity, and intervention ran on identical token identifiers"
                ),
                artifact_id=comparison_artifact_id,
                trace_id=getattr(comparison, "trace_id", None),
                event_ids=list(getattr(comparison, "event_ids", []) or []),
                token_position=getattr(comparison, "consumed_position", None),
                alignment_status="exact",
            )
        )
    measurements = (
        _measurements_from_comparison(comparison) if comparison is not None else []
    )
    controls = (
        _controls_from_comparison(comparison, comparison_artifact_id)
        if comparison is not None
        else []
    )
    identity_failed = bool(
        comparison is not None
        and comparison.identity_control.performed
        and not comparison.identity_control.passed
    )
    limitations = list(DOSSIER_LIMITATIONS)
    if comparison is not None:
        for item in comparison.limitations:
            if item not in limitations:
                limitations.append(item)
        continuation = getattr(comparison, "continuation", None)
        if continuation is not None:
            limitations.append(
                "The optional continuation diverges from the baseline after "
                f"{continuation.diff.longest_common_prefix} generated token(s); "
                "later states are not fixed-context measurements."
            )
    for item in extra_limitations:
        if item not in limitations:
            limitations.append(item)
    if identity_failed:
        limitations.insert(
            0,
            "The identity control failed. Do not attribute the measured difference solely "
            "to the requested intervention.",
        )
    hypothesis_items: list[DossierHypothesis] = []
    for item in hypotheses:
        if isinstance(item, DossierHypothesis):
            hypothesis_items.append(item)
        elif isinstance(item, dict):
            hypothesis_items.append(DossierHypothesis.model_validate(item))
    dossier = InvestigationDossier(
        dossier_id=_new("dos"),
        title=title,
        case_id=case_id,
        trace_id=trace_id,
        event_ids=[event.event_id for event in event_items],
        observations=observations,
        measurements=measurements,
        controls=controls,
        hypotheses=hypothesis_items,
        limitations=limitations,
        next_operations=_next_operations(
            has_event=bool(event_items),
            comparison=comparison,
            identity_failed=identity_failed,
        ),
        artifact_refs=[
            *[dict(item) for item in artifact_refs],
            *(
                [{"artifact_id": comparison_artifact_id, "kind": "fixed_context_comparison"}]
                if comparison_artifact_id
                else []
            ),
            *[
                {"artifact_id": artifact_id, "kind": None}
                for artifact_id in condition_artifact_ids
            ],
        ],
        summary=summary,
        provenance=provenance or {"created_by": "wsp.dossier"},
        metadata={
            "comparison_id": getattr(comparison, "comparison_id", None),
            "identity_status": (
                None
                if comparison is None
                else comparison.identity_control.agreement
            ),
            "validity_status": None if comparison is None else comparison.validity.status,
        },
    )
    dossier.checksums["dossier_sha256"] = dossier_sha256(dossier)
    return dossier


__all__ = [
    "SCHEMA_DOSSIER",
    "DOSSIER_LIMITATIONS",
    "DossierControl",
    "DossierHypothesis",
    "DossierMeasurement",
    "DossierNextOperation",
    "DossierObservation",
    "InvestigationDossier",
    "build_dossier",
    "dossier_sha256",
]
