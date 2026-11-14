"""Investigation evidence: the durable link from behaviour to measurement.

A case already records behaviour: traces, findings, contrasts.  This record adds
the internal-measurement layer and keeps the layers visibly separate, because
mixing them is how a numerical difference quietly becomes a claimed mechanism.

The record is a *reference* document.  It stores identifiers and the small
amount of context needed to read them (which capture, which backend, which
alignment), and never copies activation tensors.  Everything it points at stays
content-addressed by the artifact store, so the case export can re-derive and
verify it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from wsp.alignment.ids import new_evidence_id
from wsp.core.hashing import sha256_bytes
from wsp.core.json import canonical_bytes
from wsp.core.time import utc_now_iso
from wsp.investigation.models import StrictModel

SCHEMA_EVIDENCE = "wsp.investigation-evidence/1.0"

EvidenceStage = Literal["observation", "measurement", "hypothesis", "intervention", "weight-edit"]

EVIDENCE_LIMITATIONS = [
    "measurements in this record are numerical observations under a specific replay configuration; they are not causal claims",
    "hypotheses are human-authored and explicitly unverified until an intervention tests them",
    "weight edits, if referenced, are separate artifacts and are not implied by any measurement here",
]


class ArtifactLink(StrictModel):
    """A pointer to a stored record, kept by identifier rather than by value."""

    artifact_id: str
    record_id: str | None = None
    trace_id: str | None = None
    schema_version: str | None = None
    note: str | None = None


class FindingRef(StrictModel):
    """A behavioural finding selected for investigation."""

    trace_id: str
    detector_id: str
    detector_version: str | None = None
    kind: str
    value: float | None = None
    unit: str | None = None
    severity: str | None = None
    span_artifact_id: str | None = None
    note: str | None = None


class ReplayRef(StrictModel):
    trace_id: str
    plan_id: str | None = None
    plan_artifact_id: str | None = None
    replay_id: str | None = None
    result_artifact_id: str | None = None
    numerical_agreement: str | None = None
    evidence_level: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _agreement_matches_evidence(self) -> "ReplayRef":
        if self.evidence_level == "verified" and self.numerical_agreement != "verified":
            raise ValueError(
                "a verified replay requires numerical_agreement='verified'; "
                "token reconstruction alone is not numerical agreement"
            )
        return self


class CaptureRef(StrictModel):
    capture_id: str
    artifact_id: str
    trace_id: str
    mode: str
    backend: str
    position_semantics: str
    modules: list[str] = Field(default_factory=list)
    positions: list[int] = Field(default_factory=list)
    replay_id: str | None = None
    note: str | None = None


class ComparisonRef(StrictModel):
    comparison_id: str
    artifact_id: str
    left_capture_id: str
    right_capture_id: str
    metrics: list[str] = Field(default_factory=list)
    alignment_strategy: str | None = None
    note: str | None = None


class HypothesisNote(StrictModel):
    """A candidate explanation.  Always unverified until an intervention runs."""

    hypothesis_id: str = Field(pattern=r"^hy_[0-9a-f]{16}$")
    statement: str = Field(min_length=1)
    status: Literal["unverified"] = "unverified"
    proposed_by: str | None = None
    rationale: str | None = None
    related_capture_ids: list[str] = Field(default_factory=list)
    related_comparison_ids: list[str] = Field(default_factory=list)


class InvestigationEvidence(StrictModel):
    schema_version: str = SCHEMA_EVIDENCE
    evidence_id: str = Field(pattern=r"^inv_[0-9a-f]{16}$")
    created_at: str
    case_id: str | None = None
    title: str = Field(min_length=1)
    summary: str | None = None
    observation_trace_id: str | None = None
    control_trace_ids: list[str] = Field(default_factory=list)
    findings: list[FindingRef] = Field(default_factory=list)
    replays: list[ReplayRef] = Field(default_factory=list)
    alignments: list[ArtifactLink] = Field(default_factory=list)
    spans: list[ArtifactLink] = Field(default_factory=list)
    captures: list[CaptureRef] = Field(default_factory=list)
    comparisons: list[ComparisonRef] = Field(default_factory=list)
    hypotheses: list[HypothesisNote] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    stages: list[EvidenceStage] = Field(
        default_factory=lambda: ["observation", "measurement", "hypothesis"]
    )
    limitations: list[str] = Field(default_factory=lambda: list(EVIDENCE_LIMITATIONS))
    warnings: list[str] = Field(default_factory=list)
    privacy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> "InvestigationEvidence":
        if not (self.findings or self.replays or self.captures or self.comparisons):
            raise ValueError(
                "an investigation record needs at least one finding, replay, capture, or comparison"
            )
        comparison_ids = [item.comparison_id for item in self.comparisons]
        if len(set(comparison_ids)) != len(comparison_ids):
            raise ValueError("comparison ids must be unique")
        capture_ids = [item.capture_id for item in self.captures]
        if len(set(capture_ids)) != len(capture_ids):
            raise ValueError("capture ids must be unique")
        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError("hypothesis ids must be unique")
        known_captures = set(capture_ids)
        for comparison in self.comparisons:
            for capture_id in (comparison.left_capture_id, comparison.right_capture_id):
                if known_captures and capture_id not in known_captures:
                    raise ValueError(
                        f"comparison '{comparison.comparison_id}' references capture "
                        f"'{capture_id}', which is not part of this investigation record"
                    )
        for hypothesis in self.hypotheses:
            for capture_id in hypothesis.related_capture_ids:
                if known_captures and capture_id not in known_captures:
                    raise ValueError(
                        f"hypothesis '{hypothesis.hypothesis_id}' references unknown capture '{capture_id}'"
                    )
        return self

    @property
    def capture_ids(self) -> list[str]:
        return [item.capture_id for item in self.captures]

    @property
    def comparison_ids(self) -> list[str]:
        return [item.comparison_id for item in self.comparisons]


def evidence_checksum(record: InvestigationEvidence) -> str:
    """Checksum over the record with its own checksums and timestamp removed."""

    payload = record.model_dump(mode="json")
    payload.pop("checksums", None)
    payload.pop("created_at", None)
    return sha256_bytes(canonical_bytes(payload))


def new_hypothesis_id() -> str:
    """Evidence-local hypothesis ids use the same shape as case hypotheses."""

    from wsp.behavior.ids import new_hypothesis_id as _behavior_hypothesis_id

    return _behavior_hypothesis_id()


def build_evidence(
    *,
    title: str,
    case_id: str | None = None,
    summary: str | None = None,
    observation_trace_id: str | None = None,
    control_trace_ids: list[str] | None = None,
    findings: list[FindingRef] | None = None,
    replays: list[ReplayRef] | None = None,
    alignments: list[ArtifactLink] | None = None,
    spans: list[ArtifactLink] | None = None,
    captures: list[CaptureRef] | None = None,
    comparisons: list[ComparisonRef] | None = None,
    hypotheses: list[HypothesisNote] | None = None,
    notes: list[str] | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> InvestigationEvidence:
    """Assemble and seal a record, stamping its checksum."""

    record = InvestigationEvidence(
        evidence_id=new_evidence_id(),
        created_at=utc_now_iso(),
        case_id=case_id,
        title=title,
        summary=summary,
        observation_trace_id=observation_trace_id,
        control_trace_ids=list(control_trace_ids or []),
        findings=list(findings or []),
        replays=list(replays or []),
        alignments=list(alignments or []),
        spans=list(spans or []),
        captures=list(captures or []),
        comparisons=list(comparisons or []),
        hypotheses=list(hypotheses or []),
        notes=list(notes or []),
        warnings=list(warnings or []),
        privacy={
            "contains_activation_values": False,
            "contains_source_text": True,
            "note": (
                "this record references measurements and traces; the referenced artifacts carry "
                "their own privacy metadata and must be exported under the same policy"
            ),
        },
        metadata=dict(metadata or {}),
    )
    record.checksums["evidence_sha256"] = evidence_checksum(record)
    return record


__all__ = [
    "SCHEMA_EVIDENCE",
    "EVIDENCE_LIMITATIONS",
    "ArtifactLink",
    "CaptureRef",
    "ComparisonRef",
    "EvidenceStage",
    "FindingRef",
    "HypothesisNote",
    "InvestigationEvidence",
    "ReplayRef",
    "build_evidence",
    "evidence_checksum",
    "new_hypothesis_id",
]
