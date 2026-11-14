"""Field-aware export redaction for traces, cases, and linked artifacts.

The rule this module implements is deliberately blunt: a redacted export keeps
*structure* — identifiers, enumerations, timestamps, numbers, relations — and
removes every free-form string, because free-form strings are the only fields
that can carry prompt, output, or human-authored content.  Anything that does
not fit that rule must be handled explicitly here or the export is refused.

Every removal is recorded in a :class:`~wsp.workspace.privacy.RemovedText`
collector.  The exporter then scans the finished archive for those strings, so
the policy's claim is checked against the artifact instead of being asserted
about the code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wsp.core.errors import SafetyError
from wsp.core.time import utc_now_iso
from wsp.workspace.privacy import (
    is_dropped_kind,
    REDACTED,
    ExportPolicy,
    RemovedText,
    artifact_ref_payload,
    redact_evaluation,
    redact_generation_params,
    safe_display_name,
    scrub_model_ref,
    scrub_provenance,
    transform_artifact,
)

_OMIT = object()


@dataclass
class ArtifactRewrite:
    """One linked artifact after the policy was applied."""

    old_id: str
    ref: dict[str, Any] | None
    kind: str
    format: str
    data: bytes | None

    @property
    def omitted(self) -> bool:
        return self.ref is None


@dataclass
class RedactedCase:
    """A redacted case payload plus everything needed to rebuild its artifacts."""

    payload: dict[str, Any]
    artifacts: list[ArtifactRewrite] = field(default_factory=list)
    removed: RemovedText = field(default_factory=RemovedText)
    limitations: list[str] = field(default_factory=list)
    checksums_dropped: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _drop_key(container: dict[str, Any], key: str, removed: RemovedText) -> None:
    """Remove a free-form field, recording its value as a scan needle."""

    if key in container:
        removed.take(container.get(key))
        container[key] = None


def _scrub_strings(values: Any, removed: RemovedText) -> list[str]:
    return removed.scrub_string_list(values)


def _policy_payload(policy: ExportPolicy) -> dict[str, Any]:
    """The policy as recorded in an export.

    ``sensitive_markers`` are caller-supplied secrets; they are never written.
    """

    payload = policy.model_dump(mode="json")
    payload.pop("sensitive_markers", None)
    return payload


# --------------------------------------------------------------------------- #
# trace redaction
# --------------------------------------------------------------------------- #


def redact_trace_payload(
    payload: dict[str, Any],
    policy: ExportPolicy,
    removed: RemovedText,
    *,
    keep_artifact_refs: bool = False,
) -> dict[str, Any]:
    """Return a redacted ``GenerationTrace`` payload.

    ``keep_artifact_refs`` is set by the case exporter, which rewrites every
    reference to the redacted artifact it embeds.  A standalone trace export
    cannot embed those artifacts, so it must drop the references rather than
    leave them dangling.
    """

    if not policy.redacted:
        return dict(payload)

    payload = dict(payload)
    keep_tokens = policy.keep_token_ids

    source = _mapping(payload.get("input"))
    _drop_key(source, "text", removed)
    _drop_key(source, "system", removed)
    for message in _sequence(source.get("messages")):
        if isinstance(message, dict):
            _drop_key(message, "content", removed)
    source["messages"] = []
    _drop_key(source, "sha256", removed)
    if not keep_tokens:
        _drop_key(source, "tokens_sha256", removed)
        source["token_ids"] = None
        source["token_encoding"] = None
    source["artifact_id"] = None
    payload["input"] = source

    prompt = payload.get("prompt")
    if isinstance(prompt, dict):
        prompt = _mapping(prompt)
        _drop_key(prompt, "rendered", removed)
        prompt["template_name"] = safe_display_name(prompt.get("template_name"), removed)
        payload["prompt"] = prompt

    payload["model"] = scrub_model_ref(payload.get("model"), removed)

    backend = payload.get("backend")
    if isinstance(backend, dict):
        backend = _mapping(backend)
        backend["runtime"] = removed.scrub_soft(backend.get("runtime"))
        payload["backend"] = backend

    params = payload.get("params")
    if isinstance(params, dict):
        payload["params"] = redact_generation_params(params, removed)

    tokens: list[dict[str, Any]] = []
    for token in _sequence(payload.get("tokens")):
        if not isinstance(token, dict):
            continue
        token = _mapping(token)
        _drop_key(token, "text", removed)
        if not keep_tokens:
            token["token_id"] = None
        removed.drop(token.get("top_logprobs"))
        token["top_logprobs"] = []
        tokens.append(token)
    payload["tokens"] = tokens

    _drop_key(payload, "text", removed)

    termination = payload.get("termination")
    if isinstance(termination, dict):
        termination = _mapping(termination)
        _drop_key(termination, "stop_sequence", removed)
        _drop_key(termination, "detail", removed)
        payload["termination"] = termination

    tool_calls: list[dict[str, Any]] = []
    for call in _sequence(payload.get("tool_calls")):
        if not isinstance(call, dict):
            continue
        call = _mapping(call)
        _drop_key(call, "arguments", removed)
        _drop_key(call, "result_summary", removed)
        _drop_key(call, "arguments_sha256", removed)
        call["status"] = call.get("status") or "unknown"
        tool_calls.append(call)
    payload["tool_calls"] = tool_calls

    annotations: list[dict[str, Any]] = []
    for annotation in _sequence(payload.get("annotations")):
        if not isinstance(annotation, dict):
            continue
        annotation = _mapping(annotation)
        _drop_key(annotation, "name", removed)
        annotation["name"] = REDACTED
        annotation["metadata"] = removed.scrub_soft(annotation.get("metadata"))
        annotations.append(annotation)
    payload["annotations"] = annotations

    payload["expected_answers"] = _scrub_strings(payload.get("expected_answers"), removed)

    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        payload["provenance"] = scrub_provenance(provenance, removed)

    privacy = _mapping(payload.get("privacy"))
    privacy["store_input_text"] = False
    privacy["store_output_text"] = False
    privacy["redacted"] = True
    _drop_key(privacy, "note", removed)
    payload["privacy"] = privacy

    payload["checksums"] = {}
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["warnings"] = _scrub_strings(payload.get("warnings"), removed)

    if not keep_artifact_refs:
        payload["activation_captures"] = []

    unavailable = _mapping(payload.get("unavailable"))
    unavailable["input_text"] = "removed by export redaction"
    unavailable["output_text"] = "removed by export redaction"
    unavailable["tool_calls"] = "arguments and results removed by export redaction"
    if not keep_tokens:
        unavailable["token_ids"] = (
            "token identifiers removed by export redaction; this trace cannot be replayed"
        )
    unavailable["top_logprobs"] = "token alternatives removed by export redaction"
    payload["unavailable"] = unavailable
    return payload


def redact_trace(
    payload: dict[str, Any],
    policy: ExportPolicy,
    removed: RemovedText,
) -> dict[str, Any]:
    return redact_trace_payload(payload, policy, removed)


def dropped_trace_checksums(payload: dict[str, Any]) -> list[str]:
    checksums = payload.get("checksums")
    if not isinstance(checksums, dict):
        return []
    return sorted(str(key) for key in checksums)


# --------------------------------------------------------------------------- #
# artifact references
# --------------------------------------------------------------------------- #


def collect_artifact_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            # Both the schema's ``artifact_id`` field and the qualified forms
            # used by measurement records (``value_artifact_id``,
            # ``plan_artifact_id``, ...) are references to stored artifacts.
            if isinstance(item, str) and (key == "artifact_id" or key.endswith("_artifact_id")):
                if item.startswith("a_"):
                    found.add(item)
            found.update(collect_artifact_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(collect_artifact_ids(item))
    return found


#: Transform order for artifacts that reference other artifacts.
_REFERENCE_RANK = {
    "activation_vector": 0,
    "activation_values": 0,
    "trace_capture": 1,
    "capture_comparison": 2,
    "investigation_evidence": 3,
}


def rewrite_artifact_ids(
    artifact_ids: Any,
    *,
    policy: ExportPolicy,
    artifacts: Any,
    removed: RemovedText,
    limitations: list[str],
) -> dict[str, ArtifactRewrite]:
    """Transform each linked artifact once and describe where it moved."""

    # References form a small DAG (captures -> comparisons -> evidence).  A
    # transformed record must point at the transformed ids of the records it
    # references, so transform referents first and remap as we go.
    ranked: list[tuple[int, str, Any, Any]] = []
    for artifact_id in sorted(artifact_ids):
        ref, path = artifacts.get(artifact_id)
        ranked.append((_REFERENCE_RANK.get(ref.kind, 0), artifact_id, ref, path))

    rewrites: dict[str, ArtifactRewrite] = {}
    for _rank, artifact_id, ref, path in sorted(ranked, key=lambda item: (item[0], item[1])):
        if policy.redacted and is_dropped_kind(ref.kind):
            # A retained activation vector cannot be redacted: its content is the
            # measurement.  Drop it and say so, rather than exporting it or
            # failing every redacted export that contains a selected capture.
            rewrites[artifact_id] = ArtifactRewrite(
                old_id=artifact_id, ref=None, kind=ref.kind, format=ref.format, data=None
            )
            limitations.append(
                f"retained numeric artifact {artifact_id} ({ref.kind}) was dropped: "
                "activation values cannot be redacted, so this export carries its "
                "summary statistics only"
            )
            continue
        data = path.read_bytes()
        remap = {
            old_id: (rewrite.ref["artifact_id"] if rewrite.ref else None)
            for old_id, rewrite in rewrites.items()
        }
        try:
            kind, format_name, new_data = transform_artifact(
                kind=ref.kind,
                format_name=ref.format,
                data=data,
                policy=policy,
                removed=removed,
                remap=remap,
            )
        except SafetyError:
            if policy.on_untransformable_artifact == "omit":
                rewrites[artifact_id] = ArtifactRewrite(
                    old_id=artifact_id, ref=None, kind=ref.kind, format=ref.format, data=None
                )
                limitations.append(
                    f"linked artifact {artifact_id} ({ref.kind}) could not be redacted and "
                    "was omitted; the exported case no longer references it"
                )
                continue
            raise
        new_ref = artifact_ref_payload(
            kind,
            new_data,
            format_name=format_name,
            created_at=ref.created_at,
            schema_version=ref.schema_version,
            metadata=removed.scrub_soft(ref.metadata),
        )
        rewrites[artifact_id] = ArtifactRewrite(
            old_id=artifact_id, ref=new_ref, kind=kind, format=format_name, data=new_data
        )
    return rewrites


def _apply_rewrites(value: Any, rewrites: dict[str, ArtifactRewrite]) -> Any:
    if isinstance(value, list):
        rewritten = [_apply_rewrites(item, rewrites) for item in value]
        return [item for item in rewritten if item is not _OMIT]
    if not isinstance(value, dict):
        return value
    artifact_id = value.get("artifact_id")
    looks_like_ref = (
        isinstance(artifact_id, str)
        and "sha256" in value
        and "nbytes" in value
        and ("path" in value or "kind" in value)
    )
    if looks_like_ref and isinstance(artifact_id, str):
        rewrite = rewrites.get(artifact_id)
        if rewrite is not None:
            if rewrite.ref is None:
                return _OMIT
            return dict(rewrite.ref)
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, str) and (key == "artifact_id" or key.endswith("_artifact_id")):
            rewrite = rewrites.get(item)
            if rewrite is not None:
                result[key] = rewrite.ref["artifact_id"] if rewrite.ref else None
                continue
        rewritten = _apply_rewrites(item, rewrites)
        result[key] = None if rewritten is _OMIT else rewritten
    return result


# --------------------------------------------------------------------------- #
# case redaction
# --------------------------------------------------------------------------- #


def _redact_evidence(value: Any, removed: RemovedText) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for reference in _sequence(value):
        if not isinstance(reference, dict):
            continue
        reference = _mapping(reference)
        _drop_key(reference, "name", removed)
        reference["name"] = REDACTED
        if "detail" in reference:
            reference["detail"] = removed.scrub_soft(reference.get("detail"))
        evidence.append(reference)
    return evidence


def _redact_finding(value: dict[str, Any], removed: RemovedText) -> dict[str, Any]:
    finding = _mapping(value)
    _drop_key(finding, "explanation", removed)
    finding["explanation"] = REDACTED
    finding["parameters"] = removed.scrub_soft(finding.get("parameters"))
    finding["metadata"] = removed.scrub_soft(finding.get("metadata"))
    finding["limitations"] = _scrub_strings(finding.get("limitations"), removed)
    finding["value"] = removed.scrub_deep(finding.get("value"))
    finding["evidence"] = _redact_evidence(finding.get("evidence"), removed)
    return finding


def redact_case_payload(
    payload: dict[str, Any],
    policy: ExportPolicy,
    removed: RemovedText,
) -> dict[str, Any]:
    """Redact a ``FailureCase`` payload in place-free form.

    The result is structural: it must still validate as a ``FailureCase`` and
    still satisfy the case invariants (observations, controls, findings).
    """

    if not policy.redacted:
        return dict(payload)

    case = dict(payload)
    _drop_key(case, "title", removed)
    case["title"] = f"redacted case {payload.get('case_id', '')}".strip()
    _drop_key(case, "summary", removed)
    case["created_at"] = payload.get("created_at") or utc_now_iso()
    case["updated_at"] = payload.get("updated_at") or case["created_at"]

    case["model"] = scrub_model_ref(case.get("model"), removed)

    inference = case.get("inference")
    if isinstance(inference, dict):
        case["inference"] = redact_generation_params(inference, removed)

    review = _mapping(case.get("review"))
    review["reviewers"] = _scrub_strings(review.get("reviewers"), removed)
    _drop_key(review, "notes", removed)
    case["review"] = review

    case["findings"] = [
        _redact_finding(finding, removed)
        for finding in _sequence(case.get("findings"))
        if isinstance(finding, dict)
    ]

    contrasts: list[dict[str, Any]] = []
    for reference in _sequence(case.get("contrasts")):
        if not isinstance(reference, dict):
            continue
        reference = _mapping(reference)
        _drop_key(reference, "test_name", removed)
        contrasts.append(reference)
    case["contrasts"] = contrasts

    case["evaluations"] = [
        redact_evaluation(evaluation, removed)
        for evaluation in _sequence(case.get("evaluations"))
        if isinstance(evaluation, dict)
    ]

    reproduction = _mapping(case.get("reproduction"))
    _drop_key(reproduction, "command", removed)
    _drop_key(reproduction, "notes", removed)
    reproduction["environment"] = removed.scrub_soft(reproduction.get("environment"))
    params = reproduction.get("params")
    if isinstance(params, dict):
        reproduction["params"] = redact_generation_params(params, removed)
    case["reproduction"] = reproduction

    annotations: list[dict[str, Any]] = []
    for annotation in _sequence(case.get("annotations")):
        if not isinstance(annotation, dict):
            continue
        annotation = _mapping(annotation)
        _drop_key(annotation, "text", removed)
        annotation["text"] = REDACTED
        _drop_key(annotation, "author", removed)
        annotations.append(annotation)
    case["annotations"] = annotations

    hypotheses: list[dict[str, Any]] = []
    for hypothesis in _sequence(case.get("hypotheses")):
        if not isinstance(hypothesis, dict):
            continue
        hypothesis = _mapping(hypothesis)
        _drop_key(hypothesis, "statement", removed)
        hypothesis["statement"] = REDACTED
        _drop_key(hypothesis, "rationale", removed)
        hypotheses.append(hypothesis)
    case["hypotheses"] = hypotheses

    patches: list[dict[str, Any]] = []
    for patch in _sequence(case.get("patches")):
        if not isinstance(patch, dict):
            continue
        patch = _mapping(patch)
        _drop_key(patch, "note", removed)
        patches.append(patch)
    case["patches"] = patches

    investigations: list[dict[str, Any]] = []
    for link in _sequence(case.get("investigations")):
        if not isinstance(link, dict):
            continue
        link = _mapping(link)
        _drop_key(link, "title", removed)
        link["title"] = f"redacted evidence {link.get('evidence_id', '')}".strip()
        _drop_key(link, "note", removed)
        investigations.append(link)
    case["investigations"] = investigations

    case["tags"] = removed.scrub_soft_list(case.get("tags"))
    case["metadata"] = removed.scrub_soft(case.get("metadata"))
    case["warnings"] = _scrub_strings(case.get("warnings"), removed)
    case["activation_captures"] = _sequence(case.get("activation_captures"))
    return case


def redact_case(
    payload: dict[str, Any],
    *,
    policy: ExportPolicy,
    artifacts: Any,
    extra_artifact_ids: Iterable[str] = (),
) -> RedactedCase:
    """Redact a case payload and every artifact it references.

    ``extra_artifact_ids`` lets a caller include artifacts reachable through
    evidence records that are not part of the case payload itself, so that every
    transformed record is produced in one pass with a consistent id mapping.
    """

    if not policy.redacted:
        return RedactedCase(payload=dict(payload), removed=RemovedText(), limitations=[])

    removed = RemovedText()
    limitations: list[str] = []
    case = redact_case_payload(payload, policy, removed)
    rewrites = rewrite_artifact_ids(
        collect_artifact_ids(case) | {str(item) for item in extra_artifact_ids},
        policy=policy,
        artifacts=artifacts,
        removed=removed,
        limitations=limitations,
    )
    case = _apply_rewrites(case, rewrites)

    checksums_dropped: list[str] = []
    token_ids_retained = policy.keep_token_ids
    artifact_map = {
        old: (rewrite.ref["artifact_id"] if rewrite.ref else None)
        for old, rewrite in rewrites.items()
    }
    limitations.extend(policy.limitations())
    if any(rewrite.omitted for rewrite in rewrites.values()):
        limitations.append("some linked artifacts were omitted; see artifact_map")

    case["redaction"] = {
        "schema_version": "wsp.redaction/1.0",
        "mode": "redacted",
        "applied_at": utc_now_iso(),
        "policy": _policy_payload(policy),
        "removed_fields": len(removed),
        "removed_strings": len(removed.values),
        "checksums_dropped": checksums_dropped,
        "token_ids_retained": token_ids_retained,
        "replay_capability": "reduced" if token_ids_retained else "unavailable",
        "artifact_map": artifact_map,
        "limitations": limitations,
        "leak_scan": "clean",
        "notes": [
            "this case is a redacted export; it is not a complete record of the original",
            "records removed here are not recoverable from this archive",
        ],
    }
    return RedactedCase(
        payload=case,
        artifacts=list(rewrites.values()),
        removed=removed,
        limitations=limitations,
        checksums_dropped=checksums_dropped,
    )
