"""One export privacy policy for every WSP export path.

Redaction here is *structural*.  Instead of dumping a record and hoping a
scrubber recognises every field that could contain text, a redacted record is
rebuilt: every string that is not on an explicit structural allowlist is
removed, and every removed string is remembered.  After the export has been
serialised it is scanned for all of them.  A hit is a hard error, never a
warning, so a policy gap can fail an export but cannot leak data.

Three guarantees are deliberately separated:

``private``
    Full fidelity.  The export is a byte-for-byte scientific record of what was
    measured and may contain the exact prompts, outputs, annotations, and
    measurements.

``redacted``
    No known sensitive string survives anywhere in the serialised export,
    including nested records, metadata, filenames, and error text.  Replay
    capability is deliberately reduced and the export says so.

``rejected``
    An export is refused when a linked artifact cannot be transformed without
    either leaking data or silently discarding scientific content.

The scan is a backstop for the transformer, not a substitute for it.  Very
short values (fewer than ``MIN_SCAN_CHARS`` characters) are removed but not
searched for, because searching for them would match unrelated structural text.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wsp.core.errors import SafetyError
from wsp.core.json import canonical_bytes, to_jsonable

REDACTED = "[redacted]"
REDACTED_IMAGE = REDACTED
MIN_SCAN_CHARS = 8

ExportMode = Literal["private", "redacted"]

#: Longest needle echoed back in an error message.  Leak reports never contain
#: the sensitive value itself; they contain a hash prefix and a length.
_REPORT_PREFIX = 12


class ExportPolicy(BaseModel):
    """Explicit export privacy policy.

    ``mode`` is the only required decision.  The remaining fields exist so a
    caller can state exactly which reductions it accepts rather than relying on
    a default that might change.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "wsp.export-policy/1.0"
    mode: ExportMode = "private"
    #: Whether trace payloads are embedded at all.
    include_traces: bool = True
    #: Whether raw measurement payloads (retained activation values, forward
    #: outputs) are embedded.  Redacted exports always reduce these to
    #: metadata, whatever this flag says.
    include_measurements: bool = True
    #: Redacted traces keep token identifiers only when this is explicitly set.
    #: Token identifiers are decodable text with the right tokenizer, so the
    #: default drops them and marks replay capability as unavailable.
    keep_token_ids: bool = False
    #: Human-authored prose (titles, notes, hypotheses) is removed in redacted
    #: mode.  This flag exists so that a caller cannot accidentally believe the
    #: opposite; it cannot be turned off.
    redact_human_text: Literal[True] = True
    #: Extra literal strings the caller knows to be sensitive.  They are scanned
    #: for in every export mode.
    sensitive_markers: list[str] = Field(default_factory=list)
    #: What to do when a linked artifact has no safe transformation.
    on_untransformable_artifact: Literal["reject", "omit"] = "reject"

    @model_validator(mode="after")
    def _redacted_consistency(self) -> "ExportPolicy":
        if self.mode == "private" and self.keep_token_ids:
            # Harmless, but it signals a misunderstanding of the mode.
            return self
        return self

    @property
    def redacted(self) -> bool:
        return self.mode == "redacted"

    def limitations(self) -> list[str]:
        """Everything a reader must know about what this export cannot show."""

        if not self.redacted:
            return []
        limitations = [
            "prompt and output text were removed by redaction",
            "human-authored titles, notes, annotations, and hypotheses were removed",
            "tool-call arguments and results were removed",
            "paths, environment metadata, and serialised error text were removed",
        ]
        if not self.keep_token_ids:
            limitations.append(
                "token identifiers were removed, so this export cannot reconstruct "
                "the model input; replay capability is unavailable"
            )
        limitations.append(
            "measurement payloads are reduced to metadata and summary statistics; "
            "retained activation values are not included"
        )
        return limitations


# --------------------------------------------------------------------------- #
# removed-text collection
# --------------------------------------------------------------------------- #

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:/-]+$")


def looks_structural(value: str) -> bool:
    """True when a string looks like a machine identifier rather than content."""

    return len(value) <= 64 and bool(_IDENTIFIER.match(value))


class RemovedText:
    """Collect every string a transformation removes.

    The collected values become the needles for the post-export leak scan.  This
    is why the scan is meaningful: it checks that nothing the policy removed
    reappears anywhere else in the archive, including in fields the policy did
    not think about.
    """

    def __init__(self, *, min_scan_chars: int = MIN_SCAN_CHARS) -> None:
        self.values: list[str] = []
        self.field_count = 0
        self.min_scan_chars = min_scan_chars

    def drop(self, value: Any) -> None:
        """Record a value that is being removed, without returning anything."""

        if isinstance(value, str):
            self.field_count += 1
            if len(value) >= self.min_scan_chars:
                self.values.append(value)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                self.drop(item)
        elif isinstance(value, dict):
            for item in value.values():
                self.drop(item)

    def drop_text(self, value: str | None) -> None:
        if value is not None:
            self.field_count += 1
            if len(value) >= self.min_scan_chars:
                self.values.append(value)

    def scrub(self, value: Any) -> Any:
        """Return a structurally equivalent value with every string removed."""

        if isinstance(value, str):
            self.drop_text(value)
            return REDACTED
        if isinstance(value, dict):
            return self.scrub_mapping(value)
        if isinstance(value, (list, tuple)):
            return [self.scrub(item) for item in value]
        return value

    def scrub_mapping(self, value: dict[str, Any] | None) -> dict[str, Any]:
        if not value:
            return {}
        return {str(key): self.scrub(item) for key, item in value.items()}

    def scrub_deep(self, value: Any) -> Any:
        """Remove every string, including mapping *keys*.

        Detector values and evaluation payloads store content in both positions:
        a repetition marker map is ``{"but wait": 8}``.  Keys that look like
        machine identifiers survive so the structure stays readable.
        """

        if isinstance(value, str):
            self.drop_text(value)
            return REDACTED
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                new_key = key_text if looks_structural(key_text) else REDACTED
                if new_key in result and new_key != key_text:
                    new_key = f"{new_key}#{len(result)}"
                result[new_key] = self.scrub_deep(item)
            return result
        if isinstance(value, (list, tuple)):
            return [self.scrub_deep(item) for item in value]
        return value

    def scrub_soft(self, value: Any) -> Any:
        """Like :meth:`scrub`, but does not treat identifiers as needles.

        Detector parameters and metadata maps are full of enum-like machine
        values (``"zscore"``, ``"repeat_span"``).  Those are removed anyway, but
        recording them as needles would make the leak scan fire on the enum
        that legitimately remains elsewhere.  Values that look like free text
        are still recorded.
        """

        if isinstance(value, str):
            self.field_count += 1
            if not looks_structural(value) and len(value) >= self.min_scan_chars:
                self.values.append(value)
            return REDACTED
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                new_key = key_text if looks_structural(key_text) else REDACTED
                if new_key in result and new_key != key_text:
                    new_key = f"{new_key}#{len(result)}"
                result[new_key] = self.scrub_soft(item)
            return result
        if isinstance(value, (list, tuple)):
            return [self.scrub_soft(item) for item in value]
        return value

    def scrub_list(self, values: Iterable[Any] | None) -> list[Any]:
        if not values:
            return []
        return [self.scrub(item) for item in values]

    def scrub_string_list(self, values: Iterable[str] | None) -> list[str]:
        if not values:
            return []
        return [str(self.scrub(item)) for item in values]

    def take(self, value: str | None) -> None:
        """Remove a free-text field entirely (stored as ``None``)."""

        self.drop_text(value)

    def take_soft(self, value: str | None) -> None:
        """Remove a field whose value may be a machine identifier.

        Names, tags, tensor names, and version strings are removed, but they are
        not recorded as needles: the same identifier legitimately survives in a
        structural position and would otherwise look like a leak.
        """

        if value is None:
            return
        self.field_count += 1
        if not looks_structural(value) and len(value) >= self.min_scan_chars:
            self.values.append(value)

    def scrub_soft_list(self, values: Iterable[str] | None) -> list[str]:
        if not values:
            return []
        result: list[str] = []
        for item in values:
            self.take_soft(item)
            result.append(REDACTED)
        return result

    def __len__(self) -> int:
        return self.field_count


# --------------------------------------------------------------------------- #
# scanning
# --------------------------------------------------------------------------- #


def _candidates(needle: str) -> list[str]:
    variants = {needle}
    try:
        variants.add(json.dumps(needle)[1:-1])
    except (TypeError, ValueError):  # pragma: no cover - strings always encode
        pass
    return [variant for variant in variants if variant]


def scan_for_leaks(text: str, needles: Iterable[str]) -> list[dict[str, Any]]:
    """Return a report for every needle that still occurs in ``text``.

    The report deliberately omits the needle.  A privacy failure must not
    reproduce the secret in the error it raises.
    """

    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for needle in needles:
        if not isinstance(needle, str) or len(needle) < MIN_SCAN_CHARS:
            continue
        if needle in seen:
            continue
        offset = -1
        for candidate in _candidates(needle):
            position = text.find(candidate)
            if position >= 0:
                offset = position
                break
        if offset < 0:
            continue
        seen.add(needle)
        findings.append(
            {
                "length": len(needle),
                "sha256_prefix": hashlib.sha256(needle.encode("utf-8")).hexdigest()[
                    :_REPORT_PREFIX
                ],
                "offset": offset,
            }
        )
    return findings


def enforce_no_leaks(
    text: str,
    needles: Sequence[str],
    *,
    markers: Sequence[str] = (),
    what: str = "export",
) -> None:
    findings = scan_for_leaks(text, list(needles) + list(markers))
    if findings:
        raise SafetyError(
            f"{what} still contains text that the privacy policy removed",
            details={
                "leaked_values": len(findings),
                "reports": findings[:8],
                "hint": (
                    "the redaction policy did not cover the field that carries this text; "
                    "refusing to write an export that claims to be redacted"
                ),
            },
        )


# --------------------------------------------------------------------------- #
# content-addressed artifacts
# --------------------------------------------------------------------------- #

_FORMAT_SUFFIX = {
    "json": ".json",
    "npy": ".npy",
    "txt": ".txt",
    "bin": ".bin",
    "safetensors": ".safetensors",
}


def content_id(data: bytes) -> str:
    """The artifact identifier the workspace will assign to ``data``."""

    return "a_" + hashlib.sha256(data).hexdigest()[:32]


def artifact_path(artifact_id: str, format_name: str) -> str:
    suffix = _FORMAT_SUFFIX.get(format_name, "." + format_name)
    return f"artifacts/{artifact_id[2:4]}/{artifact_id}{suffix}"


def artifact_ref_payload(
    kind: str,
    data: bytes,
    *,
    format_name: str,
    created_at: str,
    schema_version: str | None = "wsp.artifact/1.0",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A complete ``ArtifactRef`` for content that is not written yet.

    Exports embed transformed payloads, so the reference must be computed from
    the transformed bytes rather than read from the source workspace.
    """

    artifact_id = content_id(data)
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "path": artifact_path(artifact_id, format_name),
        "format": format_name,
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "nbytes": len(data),
        "created_at": created_at,
        "schema_version": schema_version,
        "metadata": metadata or {},
    }


# --------------------------------------------------------------------------- #
# artifact transformation registry
# --------------------------------------------------------------------------- #

#: A transformer receives the decoded JSON payload (JSON artifacts) or the raw
#: bytes (everything else) and returns ``(new_payload, new_format)``.
ArtifactTransformer = Callable[[Any, ExportPolicy, RemovedText], tuple[Any, str]]


def _transform_json_payload(
    payload: Any,
    policy: ExportPolicy,
    removed: RemovedText,
    handler: Callable[[dict[str, Any], ExportPolicy, RemovedText], dict[str, Any]],
) -> tuple[Any, str]:
    if not isinstance(payload, dict):
        raise SafetyError("artifact payload is not a JSON object; refusing to redact it")
    return handler(dict(payload), policy, removed), "json"


def _redact_contrast_result(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """Reduce a contrast result to structure, statuses, and numeric measurements."""

    test = payload.get("test")
    if isinstance(test, dict):
        test = dict(test)
        removed.take(test.get("name"))
        test["name"] = REDACTED
        removed.take(test.get("description"))
        test["description"] = None
        test["metadata"] = removed.scrub_soft(test.get("metadata"))
        test["tags"] = removed.scrub_string_list(test.get("tags"))
        for key in ("baseline",):
            variant = test.get(key)
            if isinstance(variant, dict):
                variant = dict(variant)
                removed.take(variant.get("input_text"))
                variant["input_text"] = REDACTED
                removed.take(variant.get("system"))
                variant["system"] = None
                removed.take(variant.get("notes"))
                variant["notes"] = None
                # Variant names are author-chosen labels; keep them only when
                # they are simple identifiers.
                name = variant.get("name")
                if not isinstance(name, str) or not name.isidentifier():
                    removed.take(name)
                    variant["name"] = "redacted"
                test[key] = variant
        variants = test.get("variants")
        if isinstance(variants, list):
            test["variants"] = [
                _redact_variant(variant, removed) for variant in variants
            ]
        expectation = test.get("expectation")
        if isinstance(expectation, dict):
            expectation = dict(expectation)
            removed.take(expectation.get("description"))
            expectation["description"] = REDACTED
            expectation["reference_answers"] = removed.scrub_string_list(
                expectation.get("reference_answers")
            )
            contract = expectation.get("contract")
            if isinstance(contract, dict):
                contract = dict(contract)
                removed.scrub_string_list(contract.get("required_keys"))
                contract["required_keys"] = []
                expectation["contract"] = contract
            test["expectation"] = expectation
        test["backend"] = _redact_backend_spec(test.get("backend"), policy, removed)
        params = test.get("params")
        if isinstance(params, dict):
            test["params"] = _redact_generation_params(params, removed)
        test["metadata"] = removed.scrub_soft(test.get("metadata"))
        test["seed"] = test.get("seed")
        payload["test"] = test

    payload["baseline"] = _redact_variant(payload.get("baseline"), removed)
    outcomes = payload.get("outcomes")
    if isinstance(outcomes, list):
        payload["outcomes"] = [_redact_variant(item, removed) for item in outcomes]
    removed.take(payload.get("summary"))
    payload["summary"] = REDACTED
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["trace_ids"] = list(payload.get("trace_ids") or [])
    return payload


def _redact_variant(variant: Any, removed: RemovedText) -> dict[str, Any]:
    if not isinstance(variant, dict):
        return {"name": REDACTED, "status": "unknown"}
    variant = dict(variant)
    removed.take(variant.get("input_text"))
    variant["input_text"] = REDACTED
    removed.take(variant.get("output_text"))
    variant["output_text"] = None
    removed.take(variant.get("output_sha256"))
    variant["output_sha256"] = None
    removed.take(variant.get("explanation"))
    variant["explanation"] = None
    variant["observed"] = removed.scrub_deep(variant.get("observed") or {})
    evaluation = variant.get("evaluation")
    if isinstance(evaluation, dict):
        variant["evaluation"] = _redact_evaluation(evaluation, removed)
    name = variant.get("name")
    if not isinstance(name, str) or not name.isidentifier():
        removed.take(name)
        variant["name"] = "redacted"
    variant["trace_id"] = variant.get("trace_id")
    return variant


def _redact_evaluation(evaluation: dict[str, Any], removed: RemovedText) -> dict[str, Any]:
    evaluation = dict(evaluation)
    removed.take(evaluation.get("explanation"))
    evaluation["explanation"] = None
    removed.take(evaluation.get("reviewer"))
    evaluation["reviewer"] = None
    evaluation["metadata"] = removed.scrub_soft(evaluation.get("metadata"))
    if "value" in evaluation:
        evaluation["value"] = removed.scrub_deep(evaluation.get("value"))
    return evaluation


def _redact_backend_spec(value: Any, policy: ExportPolicy, removed: RemovedText) -> Any:
    if not isinstance(value, dict):
        return value
    value = dict(value)
    huggingface = value.get("huggingface")
    if isinstance(huggingface, dict):
        huggingface = dict(huggingface)
        removed.take_soft(huggingface.get("model_path"))
        huggingface["model_path"] = REDACTED
        removed.take_soft(huggingface.get("tokenizer_path"))
        huggingface["tokenizer_path"] = None
        removed.take_soft(huggingface.get("model_name"))
        huggingface["model_name"] = None
        value["huggingface"] = huggingface
    scripted = value.get("scripted")
    if isinstance(scripted, dict):
        value["scripted"] = _redact_scripted_config(scripted, policy, removed)
    return value


def _redact_scripted_config(
    config: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    config = dict(config)

    def fixture(item: Any) -> Any:
        if not isinstance(item, dict):
            return None
        item = dict(item)
        removed.take(item.get("input_equals"))
        item["input_equals"] = None
        removed.take(item.get("input_contains"))
        item["input_contains"] = None
        removed.take(item.get("output_text"))
        item["output_text"] = REDACTED
        removed.take(item.get("termination_detail"))
        item["termination_detail"] = None
        tokens = item.get("output_tokens")
        if isinstance(tokens, list):
            reduced_tokens: list[dict[str, Any]] = []
            for token in tokens:
                if not isinstance(token, dict):
                    reduced_tokens.append({"text": REDACTED})
                    continue
                removed.take(token.get("text"))
                entry: dict[str, Any] = {"text": REDACTED}
                if policy.keep_token_ids and token.get("token_id") is not None:
                    entry["token_id"] = token["token_id"]
                reduced_tokens.append(entry)
            item["output_tokens"] = reduced_tokens
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            item["tool_calls"] = [
                {
                    "index": call.get("index", 0),
                    "name": REDACTED,
                    "status": call.get("status", "unknown"),
                }
                for call in tool_calls
                if isinstance(call, dict)
            ]
            for call in tool_calls:
                if isinstance(call, dict):
                    removed.take(call.get("name"))
                    removed.drop(call.get("arguments"))
                    removed.take(call.get("result_summary"))
        return item

    fixtures = config.get("fixtures")
    if isinstance(fixtures, list):
        config["fixtures"] = [fixture(item) for item in fixtures]
    if config.get("default_fixture") is not None:
        config["default_fixture"] = fixture(config["default_fixture"])
    removed.take_soft(config.get("model_name"))
    config["model_name"] = "redacted/model"
    removed.take_soft(config.get("tokenizer_name"))
    config["tokenizer_name"] = "redacted/tokenizer"
    return config


def _redact_generation_params(params: dict[str, Any], removed: RemovedText) -> dict[str, Any]:
    params = dict(params)
    params["stop_sequences"] = removed.scrub_string_list(params.get("stop_sequences"))
    params["extra"] = removed.scrub_soft(params.get("extra"))
    return params


def _redact_activation_capture(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """V1 ``wsp.activation-capture/1.0``: keep statistics, drop values.

    Summary records carry a fixed set of machine fields plus free-form ``notes``.
    Known machine fields are kept so the measurement stays readable; every other
    field is scrubbed, so an unexpected field cannot carry text out of the
    export.
    """

    _SUMMARY_MACHINE_FIELDS = {
        "capture_id",
        "module",
        "kind",
        "call_index",
        "original_dtype",
        "device",
        "shape",
        "numel",
        "stats",
        "retained",
        "retained_shape",
        "retained_numel",
        "sampled",
        "stride",
        "token_position",
        "artifact_id",
    }

    payload = dict(payload)
    summaries = payload.get("summaries")
    if isinstance(summaries, list):
        reduced: list[Any] = []
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            summary = dict(summary)
            removed.drop(summary.get("values"))
            summary["values"] = None
            if summary.get("retained"):
                summary["retained"] = False
                notes = list(summary.get("notes") or [])
                notes.append("retained values were removed by export redaction")
                summary["notes"] = notes
            summary["notes"] = removed.scrub_string_list(summary.get("notes"))
            for key in list(summary):
                if key in _SUMMARY_MACHINE_FIELDS or key == "notes":
                    continue
                summary[key] = removed.scrub_soft(summary[key])
            reduced.append(summary)
        payload["summaries"] = reduced
    for key in list(payload):
        if key not in {"capture_id", "run_id", "summaries", "redacted"}:
            payload[key] = removed.scrub_soft(payload[key])
    payload["redacted"] = True
    return payload


_TRACE_CAPTURE_KEEP_FIELDS = frozenset(
    {
        "schema_version",
        "capture_id",
        "created_at",
        "trace_id",
        "trace_artifact_sha256",
        "plan_id",
        "replay_id",
        "alignment_id",
        "model_file_sha256",
        "model_structure_fingerprint",
        "model_parameter_count",
        "backend",
        "backend_version",
        "device",
        "dtype",
        "mode",
        "position_semantics",
        "position_semantics_note",
        "requested_tokens",
        "positions",
        "modules",
        "include_inputs",
        "first_call_only",
        "sequence_length",
        "measurements",
        "measurements_truncated",
        "modules_summary",
        "resources",
        "replay_verification",
        "limitations",
    }
)


def _redact_trace_capture(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.trace-capture/1.0``: keep measurements, drop retained vectors.

    A capture record is a measurement, so its statistics and module identity are
    scientific content that a redacted export should keep.  Retained vectors are
    raw activations derived from the source prompt; they are dropped, and the
    record says so instead of silently reporting values it no longer carries.
    Links to replay and alignment artifacts are dropped too, because those
    artifacts are not part of a redacted export.
    """

    payload = dict(payload)
    payload["model_name"] = safe_display_name(payload.get("model_name"), removed)
    for key in ("alignment_artifact_id", "span_artifact_id", "replay_artifact_id"):
        if key in payload:
            payload[key] = None

    measurements: list[Any] = []
    for measurement in payload.get("measurements") or []:
        if not isinstance(measurement, dict):
            continue
        item = dict(measurement)
        removed.drop(item.get("values"))
        item["values"] = None
        removed.take(item.get("value_sha256"))
        item["value_sha256"] = None
        if item.get("value_artifact_id") is not None:
            item["value_artifact_id"] = None
            item["retained"] = False
            item["retained_numel"] = 0
            item["retained_shape"] = None
            notes = list(item.get("notes") or [])
            notes.append("retained values were removed by export redaction")
            item["notes"] = notes
        item["notes"] = removed.scrub_string_list(item.get("notes"))
        measurements.append(item)
    payload["measurements"] = measurements

    summaries: list[Any] = []
    for summary in payload.get("modules_summary") or []:
        if not isinstance(summary, dict):
            continue
        item = dict(summary)
        item["notes"] = removed.scrub_string_list(item.get("notes"))
        summaries.append(item)
    payload["modules_summary"] = summaries

    resources = payload.get("resources")
    if isinstance(resources, dict):
        resources = dict(resources)
        resources["elements_retained"] = 0
        resources["estimated_retained_bytes"] = 0
        resources["artifacts"] = 0
        payload["resources"] = resources

    verification = payload.get("replay_verification")
    if isinstance(verification, dict):
        verification = dict(verification)
        agreement = verification.get("agreement")
        if isinstance(agreement, dict):
            agreement = dict(agreement)
            removed.drop(agreement.get("per_token"))
            agreement["per_token"] = []
            agreement["per_token_truncated"] = True
            verification["agreement"] = agreement
        payload["replay_verification"] = verification

    privacy = payload.get("privacy")
    if isinstance(privacy, dict):
        privacy = dict(privacy)
        privacy["contains_activation_values"] = False
        privacy["note"] = removed.scrub_soft(privacy.get("note"))
        payload["privacy"] = privacy

    unavailable = payload.get("unavailable")
    unavailable = dict(unavailable) if isinstance(unavailable, dict) else {}
    unavailable["retained_values"] = (
        "retained activation vectors were removed by export redaction"
    )
    unavailable["cross_references"] = (
        "links to replay and alignment artifacts were removed because those artifacts "
        "are not part of a redacted export"
    )
    payload["unavailable"] = unavailable

    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["checksums"] = {}
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    for key in list(payload):
        if key in _TRACE_CAPTURE_KEEP_FIELDS or key in {
            "model_name",
            "warnings",
            "checksums",
            "metadata",
            "privacy",
            "unavailable",
        }:
            continue
        payload[key] = removed.scrub_soft(payload[key])
    return payload


def _redact_capture_comparison(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.capture-comparison/1.0``: keep metrics, drop supplied directions."""

    payload = dict(payload)
    removed.drop(payload.get("direction"))
    payload["direction"] = None
    payload["direction_note"] = removed.scrub_soft(payload.get("direction_note"))

    alignment = payload.get("alignment")
    if isinstance(alignment, dict):
        alignment = dict(alignment)
        pairs: list[Any] = []
        for pair in alignment.get("pairs") or []:
            if not isinstance(pair, dict):
                continue
            item = dict(pair)
            item["note"] = removed.scrub_soft(item.get("note"))
            pairs.append(item)
        alignment["pairs"] = pairs
        alignment["note"] = removed.scrub_soft(alignment.get("note"))
        payload["alignment"] = alignment

    entries: list[Any] = []
    for entry in payload.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        metrics: list[Any] = []
        for metric in item.get("metrics") or []:
            if not isinstance(metric, dict):
                continue
            value = dict(metric)
            value["note"] = removed.scrub_soft(value.get("note"))
            metrics.append(value)
        item["metrics"] = metrics
        item["warnings"] = removed.scrub_string_list(item.get("warnings"))
        entries.append(item)
    payload["entries"] = entries

    compatibility = payload.get("compatibility")
    if isinstance(compatibility, dict):
        compatibility = dict(compatibility)
        compatibility["notes"] = removed.scrub_string_list(compatibility.get("notes"))
        payload["compatibility"] = compatibility

    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    return payload


def _redact_investigation_evidence(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.investigation-evidence/1.0``: keep the measurement graph only."""

    payload = dict(payload)
    removed.drop(payload.get("title"))
    payload["title"] = REDACTED
    removed.drop(payload.get("summary"))
    payload["summary"] = None
    payload["notes"] = removed.scrub_string_list(payload.get("notes"))

    findings: list[Any] = []
    for finding in payload.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        item = dict(finding)
        item["value"] = removed.scrub_deep(item.get("value"))
        item["note"] = removed.scrub_soft(item.get("note"))
        findings.append(item)
    payload["findings"] = findings

    replays: list[Any] = []
    for replay in payload.get("replays") or []:
        if not isinstance(replay, dict):
            continue
        item = dict(replay)
        item["plan_artifact_id"] = None
        item["result_artifact_id"] = None
        item["note"] = removed.scrub_soft(item.get("note"))
        replays.append(item)
    payload["replays"] = replays

    # Alignments and span resolutions carry decoded prompt and generation text,
    # so a redacted export keeps their identifiers (if any) and drops the links.
    payload["alignments"] = []
    payload["spans"] = []

    captures: list[Any] = []
    for capture in payload.get("captures") or []:
        if not isinstance(capture, dict):
            continue
        item = dict(capture)
        item["note"] = removed.scrub_soft(item.get("note"))
        captures.append(item)
    payload["captures"] = captures

    comparisons: list[Any] = []
    for comparison in payload.get("comparisons") or []:
        if not isinstance(comparison, dict):
            continue
        item = dict(comparison)
        item["note"] = removed.scrub_soft(item.get("note"))
        comparisons.append(item)
    payload["comparisons"] = comparisons

    hypotheses: list[Any] = []
    for hypothesis in payload.get("hypotheses") or []:
        if not isinstance(hypothesis, dict):
            continue
        item = dict(hypothesis)
        removed.drop(item.get("statement"))
        item["statement"] = REDACTED
        removed.drop(item.get("rationale"))
        item["rationale"] = None
        item["proposed_by"] = None
        hypotheses.append(item)
    payload["hypotheses"] = hypotheses

    privacy = payload.get("privacy")
    if isinstance(privacy, dict):
        privacy = dict(privacy)
        privacy["contains_source_text"] = False
        privacy["note"] = removed.scrub_soft(privacy.get("note"))
        payload["privacy"] = privacy

    limitations = list(payload.get("limitations") or [])
    limitations.append(
        "this redacted evidence record keeps numerical measurement references only; "
        "token alignments, span resolutions, replay artifacts, and retained activation "
        "vectors were removed"
    )
    payload["limitations"] = limitations
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["checksums"] = {}
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    return payload


def _redact_artifact_json_generic(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """Fallback for unknown JSON artifacts: keep keys, remove all strings.

    This keeps the reference graph intact (identifiers and numbers survive) and
    records everything it dropped, so the leak scan can prove the result is
    clean.  Artifacts whose numeric content is itself sensitive must opt out
    through :data:`ARTIFACT_TRANSFORMERS`.
    """

    payload = dict(payload)
    return removed.scrub_soft(payload)


def _redact_npy(payload: bytes, policy: ExportPolicy, removed: RemovedText) -> tuple[Any, str]:
    raise SafetyError(
        "a linked binary measurement artifact cannot be transformed safely",
        details={
            "hint": (
                "redacted exports keep numeric measurement metadata but not raw arrays; "
                "detach the artifact from the case, export it privately, or set "
                "on_untransformable_artifact=omit to accept reduced completeness"
            )
        },
    )


#: Kinds that are safe to keep as raw numeric arrays.  A patch payload is a
#: weight delta; it is derived from a checkpoint, not from user text.
RAW_NUMERIC_KINDS = frozenset({"patch_payload"})

#: Numeric kinds that cannot be redacted at all because their content *is* the
#: sensitive measurement.  A retained activation vector is derived from the
#: source prompt; there is no way to remove the text from it.  A redacted export
#: therefore drops these artifacts and records that it did.
DERIVED_NUMERIC_KINDS = frozenset({"activation_vector", "activation_values"})


def is_dropped_kind(kind: str) -> bool:
    """Whether a redacted export must drop artifacts of this kind."""

    return kind in DERIVED_NUMERIC_KINDS

JSON_TRANSFORMERS: dict[str, Callable[..., dict[str, Any]]] = {
    "contrast_result": _redact_contrast_result,
    "activation_capture": _redact_activation_capture,
    "trace_capture": _redact_trace_capture,
    "capture_comparison": _redact_capture_comparison,
    "investigation_evidence": _redact_investigation_evidence,
}


def _artifact_model(kind: str) -> Any:
    """Return the strict model an artifact kind is written with, if known."""

    if kind == "contrast_result":
        from wsp.behavior.models import ContrastResult

        return ContrastResult
    if kind == "trace_capture":
        from wsp.investigation.models import TraceCaptureRecord

        return TraceCaptureRecord
    if kind == "capture_comparison":
        from wsp.investigation.compare import CaptureComparison

        return CaptureComparison
    if kind == "investigation_evidence":
        from wsp.investigation.evidence import InvestigationEvidence

        return InvestigationEvidence
    return None


def _validate_artifact_payload(kind: str, payload: dict[str, Any]) -> None:
    """Refuse to transform an artifact that is not shaped like its schema.

    The transformers walk known fields.  A payload with unexpected fields would
    therefore keep whatever those fields contain.  WSP writes these artifacts
    with strict models, so a non-conforming payload means the file was produced
    or edited outside the workbench, and redaction cannot be guaranteed.
    """

    model = _artifact_model(kind)
    if model is None:
        return
    try:
        model.model_validate(payload)
    except Exception as exc:
        raise SafetyError(
            "a linked contrast result does not match its schema, so redaction "
            "cannot be guaranteed",
            details={
                "kind": kind,
                "reason": str(exc)[:200],
                "hint": "re-export the artifact with WSP or export the case privately",
            },
        ) from exc


def substitute_artifact_references(value: Any, mapping: dict[str, str | None]) -> Any:
    """Replace artifact references with the ids they have in the export.

    An artifact's own bytes determine its exported id, so a transformed record
    that references another transformed record cannot carry the original id.
    Callers transform referenced artifacts first and pass the accumulated
    mapping here.
    """

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, str) and (key == "artifact_id" or key.endswith("_artifact_id")):
                result[key] = mapping.get(item, item)
                continue
            result[key] = substitute_artifact_references(item, mapping)
        return result
    if isinstance(value, list):
        return [substitute_artifact_references(item, mapping) for item in value]
    return value


def transform_artifact(
    *,
    kind: str,
    format_name: str,
    data: bytes,
    policy: ExportPolicy,
    removed: RemovedText,
    remap: dict[str, str | None] | None = None,
) -> tuple[str, str, bytes]:
    """Return ``(kind, format, bytes)`` for one linked artifact.

    In private mode the artifact is returned unchanged.  In redacted mode every
    artifact must have an explicit transformation, except raw numeric arrays of
    an allowlisted kind.
    """

    if not policy.redacted:
        return kind, format_name, data
    if format_name == "json":
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetyError(
                "a linked JSON artifact could not be parsed for redaction",
                details={"kind": kind, "reason": str(exc)[:120]},
            ) from exc
        if isinstance(payload, dict):
            _validate_artifact_payload(kind, payload)
        transformer = JSON_TRANSFORMERS.get(kind, _redact_artifact_json_generic)
        new_payload, new_format = _transform_json_payload(payload, policy, removed, transformer)
        if remap:
            new_payload = substitute_artifact_references(new_payload, remap)
        return kind, new_format, canonical_bytes(to_jsonable(new_payload))
    if kind in RAW_NUMERIC_KINDS:
        return kind, format_name, data
    raise SafetyError(
        "a linked artifact has no safe redaction transformation",
        details={
            "kind": kind,
            "format": format_name,
            "allowed_raw_kinds": sorted(RAW_NUMERIC_KINDS),
            "hint": (
                "refusing to include an untransformable artifact in a redacted export; "
                "set on_untransformable_artifact=omit to export without it"
            ),
        },
    )


# --------------------------------------------------------------------------- #
# generic structural scrubbing for text fields
# --------------------------------------------------------------------------- #


def safe_display_name(value: str | None, removed: RemovedText) -> str | None:
    """Remove a file name entirely.

    A file name can encode content (``prompt-about-X.safetensors``) and is not
    needed to identify a checkpoint: the structure fingerprint and file hash are
    the identity that matters.  Both directory and basename are recorded as
    needles and the field is replaced.
    """

    if value is None:
        return None
    removed.drop_text(value)
    return REDACTED


def scrub_provenance(provenance: Any, removed: RemovedText) -> dict[str, Any] | None:
    """Remove author-supplied provenance text without breaking either schema.

    Two provenance records exist: the trace ``Provenance`` (which has
    ``source_ref`` and ``importer``) and the patch ``PatchProvenance`` (whose
    ``created_by`` is required).  Only keys that are present are touched, and
    required string fields are replaced rather than nulled.
    """

    if not isinstance(provenance, dict):
        return None
    provenance = dict(provenance)
    if "notes" in provenance:
        removed.take(provenance.get("notes"))
        provenance["notes"] = None
    if "created_by" in provenance:
        removed.take_soft(provenance.get("created_by"))
        provenance["created_by"] = "redacted"
    if "source_ref" in provenance:
        removed.take_soft(provenance.get("source_ref"))
        provenance["source_ref"] = None
    if "importer" in provenance:
        removed.take_soft(provenance.get("importer"))
        provenance["importer"] = "wsp"
    if "request_id" in provenance:
        removed.take_soft(provenance.get("request_id"))
        provenance["request_id"] = None
    if "dependencies" in provenance:
        provenance["dependencies"] = removed.scrub_soft(provenance.get("dependencies"))
    return provenance


def scrub_model_ref(model: Any, removed: RemovedText) -> dict[str, Any] | None:
    if not isinstance(model, dict):
        return None
    model = dict(model)
    removed.take_soft(model.get("checkpoint_path"))
    model["checkpoint_path"] = None
    model["name"] = safe_display_name(model.get("name"), removed)
    removed.scrub_mapping(model.get("metadata"))
    model["metadata"] = {}
    return model


def scrub_tokenizer_ref(tokenizer: Any, removed: RemovedText) -> dict[str, Any] | None:
    """Remove a tokenizer label: its fingerprint is the identity that matters.

    ``name`` is ``name_or_path`` for a locally loaded tokenizer, so it can carry
    a directory path that is itself sensitive.
    """

    if not isinstance(tokenizer, dict):
        return None
    tokenizer = dict(tokenizer)
    tokenizer["name"] = safe_display_name(tokenizer.get("name"), removed)
    return tokenizer


def base64_payload(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


#: Public aliases for the per-section scrubbers used by case redaction.
redact_evaluation = _redact_evaluation
redact_generation_params = _redact_generation_params
redact_backend_spec = _redact_backend_spec


__all__ = [
    "REDACTED",
    "ExportPolicy",
    "RemovedText",
    "RAW_NUMERIC_KINDS",
    "artifact_path",
    "artifact_ref_payload",
    "base64_payload",
    "content_id",
    "enforce_no_leaks",
    "safe_display_name",
    "scan_for_leaks",
    "scrub_model_ref",
    "scrub_provenance",
    "transform_artifact",
]
