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

The scan is a backstop for the transformer, not a substitute for it.  The exact
guarantee is:

* every string the transformer removed (at any length) is checked against every
  JSON string value and mapping key in the serialised export; needles of
  ``MIN_SCAN_CHARS`` characters or more are additionally searched as substrings
  and in NFC/NFKC normalised form;
* strings the caller explicitly supplied as ``sensitive_markers`` are searched
  as substrings at any length, in the raw text, in normalised text, and in
  bounded base64-decoded payloads;
* base64-looking payloads are decoded on a best-effort basis (up to 1 MiB) and
  rescanned, so a nested encoded record cannot carry a known marker through;
* text that is not valid JSON is scanned with delimiter-anchored patterns for
  short needles rather than unanchored substrings.

What this cannot guarantee: a value hidden behind an encoding the scanner does
not recognise (compression, encryption, a bespoke alphabet), a marker that was
never declared and never removed, or a semantic paraphrase of sensitive text.
Those cases require the artifact to be transformed structurally or rejected;
they are not solved by scanning.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import unicodedata
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
        limitations.append(
            "short removed identifiers are checked by exact equality inside decoded "
            "JSON values and keys; only explicitly supplied sensitive markers are "
            "searched as substrings at every length"
        )
        return limitations


# --------------------------------------------------------------------------- #
# removed-text collection
# --------------------------------------------------------------------------- #

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:/-]+$")


def looks_structural(value: str) -> bool:
    """True when a string looks like a machine identifier rather than content."""

    return len(value) <= 64 and bool(_IDENTIFIER.match(value))


def _scrub_mapping_key(mapping: dict[str, Any], key: str, removed: RemovedText) -> str:
    """Rename a non-structural key in place so its text cannot leak.

    Structural keys (``layer.0``, ``float32``) are kept so a redacted record still
    has readable shape.  Anything else is recorded and replaced, with a suffix to
    keep sibling keys distinct.
    """

    value = mapping.pop(key)
    if looks_structural(key):
        mapping[key] = value
        return key
    removed.drop_text(key)
    safe = REDACTED
    if safe in mapping:
        suffix = 2
        while f"{safe}#{suffix}" in mapping:
            suffix += 1
        safe = f"{safe}#{suffix}"
    mapping[safe] = value
    return safe


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
            self._remember(value)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                self.drop(item)
        elif isinstance(value, dict):
            for item in value.values():
                self.drop(item)

    def _remember(self, value: str | None) -> None:
        """Record one removed string, whatever its length.

        Short strings are recorded too.  The scanner treats them differently
        (exact equality inside decoded JSON strings and keys instead of a
        substring search) rather than dropping them, because a four-character
        identifier is just as sensitive as a long sentence.
        """

        if not isinstance(value, str) or value == "":
            return
        self.values.append(value)

    def drop_text(self, value: str | None) -> None:
        if value is not None:
            self.field_count += 1
            self._remember(value)

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
        # Mapping *keys* can carry content just as values can.  Structural keys
        # (``module``, ``position``) survive so the record stays readable; every
        # other key is replaced and recorded as removed text.
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if looks_structural(key_text):
                new_key = key_text
            else:
                self._remember(key_text)
                new_key = REDACTED
            while new_key in result:
                new_key = f"{new_key}#{len(result)}"
            result[new_key] = self.scrub(item)
        return result

    def scrub_deep(self, value: Any) -> Any:
        """Remove every string, including mapping *keys*.

        Detector values and evaluation payloads store content in both positions:
        a repetition marker map is ``{"but wait": 8}``.  Keys that look like
        machine identifiers survive so the structure stays readable; every key
        that is replaced is recorded as removed text.
        """

        if isinstance(value, str):
            self.drop_text(value)
            return REDACTED
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                if looks_structural(key_text):
                    new_key = key_text
                else:
                    self._remember(key_text)
                    new_key = REDACTED
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
        are still recorded, at any length.
        """

        if isinstance(value, str):
            self.field_count += 1
            if not looks_structural(value):
                self._remember(value)
            return REDACTED
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                if looks_structural(key_text):
                    new_key = key_text
                else:
                    self._remember(key_text)
                    new_key = REDACTED
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
        structural position and would otherwise look like a leak.  Free-text
        values of any length are recorded.
        """

        if value is None:
            return
        self.field_count += 1
        if not looks_structural(value):
            self._remember(value)

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


def _normalized_forms(value: str) -> list[str]:
    forms = [value]
    for form in ("NFC", "NFKC"):
        try:
            normalized = unicodedata.normalize(form, value)
        except (TypeError, ValueError):  # pragma: no cover - strings always normalise
            continue
        if normalized not in forms:
            forms.append(normalized)
    return forms


def _candidates(needle: str) -> list[str]:
    variants: set[str] = set()
    for form in _normalized_forms(needle):
        variants.add(form)
        try:
            variants.add(json.dumps(form)[1:-1])
        except (TypeError, ValueError):  # pragma: no cover - strings always encode
            pass
    return [variant for variant in variants if variant]


def _json_strings(text: str) -> list[str] | None:
    """Every string value and mapping key in a JSON document, or ``None``.

    Short needles cannot be searched as substrings without matching unrelated
    structural text, so they are compared for equality against this list.
    """

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    strings: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                strings.append(str(key))
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(payload)
    return strings


_BASE64_TOKEN = re.compile(r"[A-Za-z0-9+/_=-]{24,}")
_MAX_ENCODED_BYTES = 1_048_576


def _encoded_texts(text: str) -> list[str]:
    """Best-effort decoding of base64-looking strings, bounded in size.

    This covers the common ``data_base64`` payload without pretending to handle
    every possible encoding.  The guarantees documented in the module docstring
    are explicit about that boundary.
    """

    decoded: list[str] = []
    budget = _MAX_ENCODED_BYTES
    for match in _BASE64_TOKEN.finditer(text):
        if budget <= 0:
            break
        token = match.group(0)
        padded = token + "=" * (-len(token) % 4)
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                raw = decoder(padded, validate=False)
            except (ValueError, TypeError):
                continue
            if not raw or len(raw) > budget:
                continue
            candidate = raw.decode("utf-8", errors="ignore")
            if candidate:
                decoded.append(candidate)
                budget -= len(raw)
            break
    return decoded


def _substring_hit(text: str, needle: str) -> int:
    for candidate in _candidates(needle):
        position = text.find(candidate)
        if position >= 0:
            return position
    for form in ("NFC", "NFKC"):
        try:
            normalized_text = unicodedata.normalize(form, text)
            normalized_needle = unicodedata.normalize(form, needle)
        except (TypeError, ValueError):  # pragma: no cover
            continue
        position = normalized_text.find(normalized_needle)
        if position >= 0:
            return position
    return -1


def _exact_hit(strings: Sequence[str], needle: str) -> bool:
    targets = set(_normalized_forms(needle))
    for value in strings:
        if any(form in targets for form in _normalized_forms(value)):
            return True
    return False


def _short_needle_pattern(needle: str) -> re.Pattern[str] | None:
    escaped = re.escape(needle)
    try:
        return re.compile(rf"(?<![A-Za-z0-9_.-]){escaped}(?![A-Za-z0-9_.-])")
    except re.error:  # pragma: no cover - escape makes this safe
        return None


def scan_for_leaks(
    text: str,
    needles: Iterable[str],
    markers: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Return a report for every needle that still survives in ``text``.

    Two classes are distinguished:

    * explicitly supplied ``markers`` are searched as substrings at any length,
      because the caller declared them sensitive;
    * automatically collected needles shorter than ``MIN_SCAN_CHARS`` are
      compared by exact equality against every JSON string value and mapping
      key, and against decoded base64 payloads.  They are never searched as
      unanchored substrings, which would match unrelated structural text.

    The report deliberately omits the needle.  A privacy failure must not
    reproduce the secret in the error it raises.
    """

    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    decoded_strings = _json_strings(text)
    encoded_texts: list[str] | None = None

    def encoded() -> list[str]:
        nonlocal encoded_texts
        if encoded_texts is None:
            encoded_texts = _encoded_texts(text)
        return encoded_texts

    def report(value: str, kind: str, offset: int) -> None:
        key = (kind, value)
        if key in seen:
            return
        seen.add(key)
        findings.append(
            {
                "length": len(value),
                "sha256_prefix": hashlib.sha256(value.encode("utf-8")).hexdigest()[
                    :_REPORT_PREFIX
                ],
                "offset": offset,
                "match": kind,
            }
        )

    for marker in markers:
        if not isinstance(marker, str) or marker == "":
            continue
        offset = _substring_hit(text, marker)
        if offset >= 0:
            report(marker, "marker-substring", offset)
            continue
        for candidate_text in encoded():
            offset = _substring_hit(candidate_text, marker)
            if offset >= 0:
                report(marker, "marker-encoded", offset)
                break

    for needle in needles:
        if not isinstance(needle, str) or needle == "":
            continue
        if len(needle) >= MIN_SCAN_CHARS:
            offset = _substring_hit(text, needle)
            if offset >= 0:
                report(needle, "substring", offset)
                continue
            for candidate_text in encoded():
                offset = _substring_hit(candidate_text, needle)
                if offset >= 0:
                    report(needle, "encoded-substring", offset)
                    break
            continue

        # Short automatically collected needle: exact equality only, and only
        # for free text.  A three-character machine identifier such as a token
        # text ("eos") or an enum ("cpu") is removed structurally but would
        # otherwise match a legitimate enum elsewhere in the archive; a caller
        # that considers such a value sensitive must supply it as a marker.
        if looks_structural(needle):
            continue
        if decoded_strings is not None:
            if _exact_hit(decoded_strings, needle):
                report(needle, "exact", -1)
                continue
        else:
            pattern = _short_needle_pattern(needle)
            if pattern is not None and pattern.search(text):
                report(needle, "delimited", -1)
                continue
        for candidate_text in encoded():
            candidate_strings = _json_strings(candidate_text)
            if candidate_strings is not None and _exact_hit(candidate_strings, needle):
                report(needle, "encoded-exact", -1)
                break
            if len(needle) >= MIN_SCAN_CHARS and needle in candidate_text:
                report(needle, "encoded-substring", -1)
                break

    return findings


def enforce_no_leaks(
    text: str,
    needles: Sequence[str],
    *,
    markers: Sequence[str] = (),
    what: str = "export",
) -> None:
    findings = scan_for_leaks(text, list(needles), markers=list(markers))
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
                safe_key = _scrub_mapping_key(summary, key, removed)
                summary[safe_key] = removed.scrub_soft(summary[safe_key])
            reduced.append(summary)
        payload["summaries"] = reduced
    for key in list(payload):
        if key not in {"capture_id", "run_id", "summaries", "redacted"}:
            safe_key = _scrub_mapping_key(payload, key, removed)
            payload[safe_key] = removed.scrub_soft(payload[safe_key])
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

    # Temporal references carry identifiers, counts, and enum labels only.  The
    # one free-form field (a plan note) is scrubbed like every other string.
    plans: list[Any] = []
    for plan in payload.get("plans") or []:
        if not isinstance(plan, dict):
            continue
        item = dict(plan)
        item["note"] = removed.scrub_soft(item.get("note"))
        plans.append(item)
    payload["plans"] = plans

    events: list[Any] = []
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        item = dict(event)
        item["note"] = removed.scrub_soft(item.get("note"))
        item["alignment_status"] = removed.scrub_soft(item.get("alignment_status"))
        events.append(item)
    payload["events"] = events

    fixed: list[Any] = []
    for comparison in payload.get("fixed_context_comparisons") or []:
        if not isinstance(comparison, dict):
            continue
        item = dict(comparison)
        item["note"] = removed.scrub_soft(item.get("note"))
        item["validity_status"] = removed.scrub_soft(item.get("validity_status"))
        item["identity_agreement"] = removed.scrub_soft(item.get("identity_agreement"))
        fixed.append(item)
    payload["fixed_context_comparisons"] = fixed

    recipe_runs: list[Any] = []
    for run in payload.get("recipe_runs") or []:
        if not isinstance(run, dict):
            continue
        item = dict(run)
        item["recipe_id"] = removed.scrub_soft(item.get("recipe_id"))
        item["status"] = removed.scrub_soft(item.get("status"))
        item["note"] = removed.scrub_soft(item.get("note"))
        recipe_runs.append(item)
    payload["recipe_runs"] = recipe_runs

    dossiers: list[Any] = []
    for dossier in payload.get("dossiers") or []:
        if not isinstance(dossier, dict):
            continue
        item = dict(dossier)
        removed.drop(item.get("title"))
        item["title"] = REDACTED
        item["note"] = removed.scrub_soft(item.get("note"))
        dossiers.append(item)
    payload["dossiers"] = dossiers

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


def _scrub_validation_text(payload: dict[str, Any], removed: RemovedText) -> None:
    """Remove free-form validation text, which can quote prompts or errors."""

    checks: list[Any] = []
    for check in payload.get("checks") or []:
        if not isinstance(check, dict):
            continue
        item = dict(check)
        item["detail"] = removed.scrub_soft(item.get("detail"))
        checks.append(item)
    if checks:
        payload["checks"] = checks
    validity = payload.get("validity")
    if isinstance(validity, dict):
        validity = dict(validity)
        validity["reasons"] = removed.scrub_string_list(validity.get("reasons"))
        inner: list[Any] = []
        for check in validity.get("checks") or []:
            if not isinstance(check, dict):
                continue
            item = dict(check)
            item["detail"] = removed.scrub_soft(item.get("detail"))
            inner.append(item)
        validity["checks"] = inner
        payload["validity"] = validity


def _redact_intervention_artifact(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.activation-intervention/1.0``: keep the structural target only.

    A redacted export keeps the machine-readable scope (module, tensor role,
    position semantics, operation, positions, shape expectations) so the
    experiment remains interpretable, and removes every human string, inline
    mean or donor activation, and link to artifacts that are not part of a
    redacted package.
    """

    payload = dict(payload)
    removed.drop(payload.get("name"))
    payload["name"] = REDACTED

    model = payload.get("model")
    if isinstance(model, dict):
        payload["model"] = scrub_model_ref(model, removed)

    execution = payload.get("execution")
    if isinstance(execution, dict):
        execution = dict(execution)
        execution["replay_artifact_id"] = None
        execution["alignment_artifact_id"] = None
        execution["capture_artifact_id"] = execution.get("capture_artifact_id")
        payload["execution"] = execution

    target = payload.get("target")
    if isinstance(target, dict):
        target = dict(target)
        target["capture_point"] = target.get("capture_point")
        payload["target"] = target

    mean = payload.get("mean")
    if isinstance(mean, dict):
        mean = dict(mean)
        # Inline values are the measurement itself; an artifact reference is
        # dropped because retained numeric vectors never enter a redacted pack.
        removed.drop(mean.get("values"))
        mean["values"] = None
        mean["artifact_id"] = None
        provenance = mean.get("provenance")
        if isinstance(provenance, dict):
            provenance = dict(provenance)
            removed.drop(provenance.get("token_alignment"))
            provenance["token_alignment"] = REDACTED
            provenance["source_capture_artifact_id"] = provenance.get(
                "source_capture_artifact_id"
            )
            provenance["notes"] = removed.scrub_soft(provenance.get("notes"))
            mean["provenance"] = provenance
        payload["mean"] = mean

    donor = payload.get("donor")
    if isinstance(donor, dict):
        donor = dict(donor)
        removed.drop(donor.get("values"))
        donor["values"] = None
        donor["values_artifact_id"] = None
        donor["note"] = removed.scrub_soft(donor.get("note"))
        donor["capture_artifact_id"] = donor.get("capture_artifact_id")
        payload["donor"] = donor

    payload["provenance"] = removed.scrub_soft(payload.get("provenance"))
    payload["notes"] = removed.scrub_soft(payload.get("notes"))
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["checksums"] = {}
    return payload


def _redact_intervention_result(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.intervention-result/1.0``: keep aggregate numerical measurements."""

    payload = dict(payload)
    if isinstance(payload.get("model"), dict):
        payload["model"] = scrub_model_ref(payload["model"], removed)
    # Per-token records embed prompt token identifiers, so they are omitted from
    # a redacted package.  Aggregate diff statistics remain.
    payload["token_logprobs"] = []
    _scrub_validation_text(payload, removed)
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["checksums"] = {}
    return payload


def _redact_counterfactual_comparison(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.counterfactual-comparison/1.0``: keep validity and effect statistics."""

    payload = dict(payload)
    if isinstance(payload.get("model"), dict):
        payload["model"] = scrub_model_ref(payload["model"], removed)
    payload["replay_artifact_id"] = None
    conditions: list[Any] = []
    for condition in payload.get("conditions") or []:
        if not isinstance(condition, dict):
            continue
        item = dict(condition)
        item["artifact_id"] = None
        item["top1_token_ids"] = {}
        conditions.append(item)
    payload["conditions"] = conditions
    identity = payload.get("identity_control")
    if isinstance(identity, dict):
        identity = dict(identity)
        identity["detail"] = removed.scrub_soft(identity.get("detail"))
        payload["identity_control"] = identity
    _scrub_validation_text(payload, removed)
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["checksums"] = {}
    return payload


def _redact_continuation_comparison(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.continuation-comparison/1.0``: keep divergence structure, not text."""

    payload = dict(payload)
    if isinstance(payload.get("model"), dict):
        payload["model"] = scrub_model_ref(payload["model"], removed)
    payload["replay_artifact_id"] = None
    removed.drop(payload.get("prefix_sha256"))
    payload["prefix_sha256"] = None
    removed.drop(payload.get("prefix_token_ids"))
    payload["prefix_token_ids"] = []

    conditions: list[Any] = []
    for condition in payload.get("conditions") or []:
        if not isinstance(condition, dict):
            continue
        item = dict(condition)
        removed.drop(item.get("generated_token_ids"))
        item["generated_token_ids"] = []
        removed.drop(item.get("generated_text"))
        item["generated_text"] = None
        item["logprobs"] = []
        item["intervention_events"] = []
        item["errors"] = removed.scrub_string_list(item.get("errors"))
        conditions.append(item)
    payload["conditions"] = conditions
    _scrub_validation_text(payload, removed)
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["checksums"] = {}
    return payload


def _redact_intervention_plan(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.intervention-plan/1.0``: keep the schedule and target structure."""

    payload = dict(payload)
    removed.drop(payload.get("name"))
    payload["name"] = REDACTED
    if isinstance(payload.get("model"), dict):
        payload["model"] = scrub_model_ref(payload["model"], removed)
    removed.drop(payload.get("prompt_token_ids"))
    payload["prompt_token_ids"] = []
    removed.drop(payload.get("prompt_sha256"))
    payload["prompt_sha256"] = None
    payload["provenance"] = removed.scrub_soft(payload.get("provenance"))
    payload["notes"] = removed.scrub_soft(payload.get("notes"))
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    nested: list[Any] = []
    for item in payload.get("interventions") or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        entry["label"] = removed.scrub_soft(entry.get("label"))
        spec = entry.get("intervention")
        if isinstance(spec, dict):
            entry["intervention"] = _redact_intervention_artifact(
                spec, policy, removed
            )
        nested.append(entry)
    payload["interventions"] = nested
    payload["checksums"] = {}
    return payload


def _redact_event_log(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.intervention-event-log/1.0``: keep firing structure and statistics."""

    payload = dict(payload)
    payload["events"] = [
        _redact_event_entry(item, removed)
        for item in payload.get("events") or []
        if isinstance(item, dict)
    ]
    payload["rejected_events"] = [
        _redact_event_entry(item, removed)
        for item in payload.get("rejected_events") or []
        if isinstance(item, dict)
    ]
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["checksums"] = {}
    return payload


def _redact_event_entry(item: dict[str, Any], removed: RemovedText) -> dict[str, Any]:
    entry = dict(item)
    entry["label"] = removed.scrub_soft(entry.get("label"))
    entry["rejection_reason"] = removed.scrub_soft(entry.get("rejection_reason"))
    entry["provenance"] = removed.scrub_soft(entry.get("provenance"))
    # Row and token *indices* are positions, not token identifiers, so they
    # remain interpretable; no generated text or token id is present here.
    return entry


def _redact_condition_sequences(item: dict[str, Any], removed: RemovedText) -> dict[str, Any]:
    entry = dict(item)
    removed.drop(entry.get("generated_token_ids"))
    entry["generated_token_ids"] = []
    removed.drop(entry.get("generated_text"))
    entry["generated_text"] = None
    entry["logprobs"] = []
    entry["top1_token_ids"] = []
    entry["errors"] = removed.scrub_string_list(entry.get("errors"))
    return entry


def _redact_temporal_run(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.temporal-intervention/1.0``: keep counts, validity, and logit diffs."""

    payload = dict(payload)
    if isinstance(payload.get("model"), dict):
        payload["model"] = scrub_model_ref(payload["model"], removed)
    removed.drop(payload.get("prompt_sha256"))
    payload["prompt_sha256"] = None
    payload["conditions"] = [
        _redact_condition_sequences(item, removed)
        for item in payload.get("conditions") or []
        if isinstance(item, dict)
    ]
    identity = payload.get("identity_control")
    if isinstance(identity, dict):
        identity = dict(identity)
        identity["detail"] = removed.scrub_soft(identity.get("detail"))
        payload["identity_control"] = identity
    _scrub_validation_text(payload, removed)
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["checksums"] = {}
    return payload


def _redact_composition(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.compositional-comparison/1.0``: keep numeric contrasts only."""

    payload = dict(payload)
    if isinstance(payload.get("model"), dict):
        payload["model"] = scrub_model_ref(payload["model"], removed)
    removed.drop(payload.get("prompt_sha256"))
    payload["prompt_sha256"] = None
    payload["conditions"] = [
        _redact_condition_sequences(item, removed)
        for item in payload.get("conditions") or []
        if isinstance(item, dict)
    ]
    for effect in payload.get("pairwise") or []:
        if not isinstance(effect, dict):
            continue
        for step in effect.get("stepwise") or []:
            if isinstance(step, dict):
                step["token_uncached"] = None
                step["token_staged"] = None
    identity = payload.get("identity_control")
    if isinstance(identity, dict):
        identity = dict(identity)
        identity["detail"] = removed.scrub_soft(identity.get("detail"))
        payload["identity_control"] = identity
    _scrub_validation_text(payload, removed)
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["checksums"] = {}
    return payload


def _redact_staged_verification(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.staged-verification/1.0``: keep agreement structure, not token ids."""

    payload = dict(payload)
    if isinstance(payload.get("model"), dict):
        payload["model"] = scrub_model_ref(payload["model"], removed)
    removed.drop(payload.get("prompt_sha256"))
    payload["prompt_sha256"] = None
    for step in payload.get("steps_detail") or []:
        if isinstance(step, dict):
            step["token_uncached"] = None
            step["token_staged"] = None
    _scrub_validation_text(payload, removed)
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["checksums"] = {}
    return payload


def _redact_staged_continuation(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.staged-continuation/1.0``: keep verified structure and timings."""

    payload = dict(payload)
    if isinstance(payload.get("model"), dict):
        payload["model"] = scrub_model_ref(payload["model"], removed)
    removed.drop(payload.get("prompt_sha256"))
    payload["prompt_sha256"] = None
    payload["conditions"] = [
        _redact_condition_sequences(item, removed)
        for item in payload.get("conditions") or []
        if isinstance(item, dict)
    ]
    verification = payload.get("verification")
    if isinstance(verification, dict):
        payload["verification"] = _redact_staged_verification(verification, policy, removed)
    for step in payload.get("stepwise_logits") or []:
        if isinstance(step, dict):
            step["token_uncached"] = None
            step["token_staged"] = None
    identity = payload.get("identity_control")
    if isinstance(identity, dict):
        identity = dict(identity)
        identity["detail"] = removed.scrub_soft(identity.get("detail"))
        payload["identity_control"] = identity
    _scrub_validation_text(payload, removed)
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["checksums"] = {}
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

def _redact_behavioral_event(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.behavioral-event/1.0``: keep the measured token position only."""

    payload = dict(payload)
    removed.drop(payload.get("token_ids"))
    payload["token_ids"] = []
    trigger = payload.get("trigger")
    if isinstance(trigger, dict):
        trigger = dict(trigger)
        removed.drop(trigger.get("text_pattern"))
        trigger["text_pattern"] = None
        trigger["parameters"] = removed.scrub_soft(trigger.get("parameters"))
        payload["trigger"] = trigger
    payload["provenance"] = removed.scrub_soft(payload.get("provenance"))
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["limitations"] = removed.scrub_string_list(payload.get("limitations"))
    payload["checksums"] = {}
    return payload


def _redact_event_trigger(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.event-trigger/1.0``: keep mode, timing, and bounds."""

    payload = dict(payload)
    removed.drop(payload.get("name"))
    payload["name"] = None
    removed.drop(payload.get("event_id"))
    payload["event_id"] = None
    removed.drop(payload.get("notes"))
    payload["notes"] = None
    definition = payload.get("definition")
    if isinstance(definition, dict):
        definition = dict(definition)
        removed.drop(definition.get("text_pattern"))
        definition["text_pattern"] = None
        definition["parameters"] = removed.scrub_soft(definition.get("parameters"))
        payload["definition"] = definition
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    return payload


def _redact_event_intervention(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.event-intervention/1.0``: keep timing, controls, and validity."""

    payload = dict(payload)
    if isinstance(payload.get("model"), dict):
        payload["model"] = scrub_model_ref(payload["model"], removed)
    removed.drop(payload.get("prompt_token_ids_sha256"))
    payload["prompt_token_ids_sha256"] = None
    trigger = payload.get("trigger")
    if isinstance(trigger, dict):
        payload["trigger"] = _redact_event_trigger(trigger, policy, removed)
    conditions = []
    for item in payload.get("conditions") or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        removed.drop(entry.get("generated_token_ids"))
        entry["generated_token_ids"] = []
        entry["logprobs"] = []
        entry["top1_token_ids"] = []
        entry["errors"] = removed.scrub_string_list(entry.get("errors"))
        conditions.append(entry)
    payload["conditions"] = conditions
    events = []
    for item in payload.get("events") or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        event = entry.get("event")
        if isinstance(event, dict):
            entry["event"] = _redact_behavioral_event(event, policy, removed)
        entry["rejection_reason"] = removed.scrub_soft(entry.get("rejection_reason"))
        entry["notes"] = removed.scrub_string_list(entry.get("notes"))
        events.append(entry)
    payload["events"] = events
    identity = payload.get("identity_control")
    if isinstance(identity, dict):
        identity = dict(identity)
        identity["detail"] = removed.scrub_soft(identity.get("detail"))
        payload["identity_control"] = identity
    stepwise = []
    for item in payload.get("stepwise") or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        removed.drop(entry.get("left_token_id"))
        removed.drop(entry.get("right_token_id"))
        entry["left_token_id"] = None
        entry["right_token_id"] = None
        stepwise.append(entry)
    payload["stepwise"] = stepwise
    _scrub_validation_text(payload, removed)
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["checksums"] = {}
    return payload


def _redact_fixed_replay(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.fixed-context-replay/1.0``: keep positions and checks, drop tokens."""

    payload = dict(payload)
    if isinstance(payload.get("model"), dict):
        payload["model"] = scrub_model_ref(payload["model"], removed)
    if isinstance(payload.get("tokenizer"), dict):
        payload["tokenizer"] = scrub_tokenizer_ref(payload["tokenizer"], removed)
    sequence = payload.get("sequence")
    if isinstance(sequence, dict):
        sequence = dict(sequence)
        removed.drop(sequence.get("input_token_ids"))
        removed.drop(sequence.get("generated_token_ids"))
        removed.drop(sequence.get("full_token_ids"))
        sequence["input_token_ids"] = []
        sequence["generated_token_ids"] = []
        sequence["full_token_ids"] = []
        sequence["prompt_length"] = 0
        sequence["generated_length"] = 0
        sequence["full_length"] = 0
        removed.drop(sequence.get("sequence_sha256"))
        sequence["sequence_sha256"] = None
        sequence["warnings"] = removed.scrub_string_list(sequence.get("warnings"))
        payload["sequence"] = sequence
    removed.drop(payload.get("replay_step"))
    removed.drop(payload.get("consumed_position"))
    payload["replay_step"] = None
    payload["consumed_position"] = None
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["provenance"] = removed.scrub_soft(payload.get("provenance"))
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["checksums"] = {}
    return payload


def _redact_fixed_comparison(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.fixed-context-comparison/1.0``: keep numerical effects and controls."""

    payload = dict(payload)
    removed.drop(payload.get("token_ids_sha256"))
    payload["token_ids_sha256"] = ""
    if isinstance(payload.get("model"), dict):
        payload["model"] = scrub_model_ref(payload["model"], removed)
    measurements = []
    for item in payload.get("measurements") or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        removed.drop(entry.get("token_id"))
        entry["token_id"] = None
        removed.drop(entry.get("baseline_top1_token_id"))
        removed.drop(entry.get("intervention_top1_token_id"))
        entry["baseline_top1_token_id"] = None
        entry["intervention_top1_token_id"] = None
        measurements.append(entry)
    payload["measurements"] = measurements
    continuation = payload.get("continuation")
    if isinstance(continuation, dict):
        continuation = dict(continuation)
        removed.drop(continuation.get("baseline_token_ids"))
        removed.drop(continuation.get("intervention_token_ids"))
        continuation["baseline_token_ids"] = []
        continuation["intervention_token_ids"] = []
        continuation["config"] = removed.scrub_soft(continuation.get("config"))
        payload["continuation"] = continuation
    identity = payload.get("identity_control")
    if isinstance(identity, dict):
        identity = dict(identity)
        identity["detail"] = removed.scrub_soft(identity.get("detail"))
        payload["identity_control"] = identity
    _scrub_validation_text(payload, removed)
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["limitations"] = removed.scrub_string_list(payload.get("limitations"))
    payload["provenance"] = removed.scrub_soft(payload.get("provenance"))
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["checksums"] = {}
    return payload


def _redact_dossier(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.investigation-dossier/1.0``: keep structure, drop narrative text."""

    payload = dict(payload)
    removed.drop(payload.get("title"))
    payload["title"] = REDACTED
    removed.drop(payload.get("summary"))
    payload["summary"] = None
    observations = []
    for item in payload.get("observations") or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        entry["summary"] = removed.scrub_soft(entry.get("summary"))
        observations.append(entry)
    payload["observations"] = observations
    measurements = []
    for item in payload.get("measurements") or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        removed.drop(entry.get("token_id"))
        entry["token_id"] = None
        measurements.append(entry)
    payload["measurements"] = measurements
    controls = []
    for item in payload.get("controls") or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        entry["detail"] = removed.scrub_soft(entry.get("detail"))
        controls.append(entry)
    payload["controls"] = controls
    hypotheses = []
    for item in payload.get("hypotheses") or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        removed.drop(entry.get("statement"))
        removed.drop(entry.get("rationale"))
        entry["statement"] = REDACTED
        entry["rationale"] = None
        entry["proposed_by"] = None
        hypotheses.append(entry)
    payload["hypotheses"] = hypotheses
    payload["limitations"] = removed.scrub_string_list(payload.get("limitations"))
    payload["provenance"] = removed.scrub_soft(payload.get("provenance"))
    payload["metadata"] = removed.scrub_soft(payload.get("metadata"))
    payload["checksums"] = {}
    return payload


def _redact_recipe_run(
    payload: dict[str, Any], policy: ExportPolicy, removed: RemovedText
) -> dict[str, Any]:
    """``wsp.recipe-run/1.0``: keep step status and artifact references."""

    payload = dict(payload)
    removed.drop(payload.get("inputs_sha256"))
    payload["inputs_sha256"] = None
    steps = []
    for item in payload.get("steps") or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        entry["detail"] = removed.scrub_soft(entry.get("detail"))
        entry["error"] = removed.scrub_soft(entry.get("error"))
        steps.append(entry)
    payload["steps"] = steps
    payload["validity"] = removed.scrub_soft(payload.get("validity"))
    payload["warnings"] = removed.scrub_string_list(payload.get("warnings"))
    payload["limitations"] = removed.scrub_string_list(payload.get("limitations"))
    payload["provenance"] = removed.scrub_soft(payload.get("provenance"))
    payload["checksums"] = {}
    return payload


JSON_TRANSFORMERS: dict[str, Callable[..., dict[str, Any]]] = {
    "contrast_result": _redact_contrast_result,
    "activation_capture": _redact_activation_capture,
    "trace_capture": _redact_trace_capture,
    "capture_comparison": _redact_capture_comparison,
    "investigation_evidence": _redact_investigation_evidence,
    "activation_intervention": _redact_intervention_artifact,
    "intervention_result": _redact_intervention_result,
    "counterfactual_comparison": _redact_counterfactual_comparison,
    "continuation_comparison": _redact_continuation_comparison,
    "intervention_plan": _redact_intervention_plan,
    "intervention_event_log": _redact_event_log,
    "temporal_intervention": _redact_temporal_run,
    "compositional_comparison": _redact_composition,
    "staged_verification": _redact_staged_verification,
    "staged_continuation": _redact_staged_continuation,
    "behavioral_event": _redact_behavioral_event,
    "event_trigger": _redact_event_trigger,
    "event_intervention": _redact_event_intervention,
    "fixed_context_replay": _redact_fixed_replay,
    "fixed_context_comparison": _redact_fixed_comparison,
    "investigation_dossier": _redact_dossier,
    "recipe_run": _redact_recipe_run,
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
    if kind == "activation_intervention":
        from wsp.interventions.models import ActivationIntervention

        return ActivationIntervention
    if kind == "intervention_result":
        from wsp.interventions.models import InterventionResult

        return InterventionResult
    if kind == "counterfactual_comparison":
        from wsp.interventions.models import CounterfactualComparison

        return CounterfactualComparison
    if kind == "continuation_comparison":
        from wsp.interventions.models import ContinuationComparison

        return ContinuationComparison
    if kind == "intervention_plan":
        from wsp.interventions.plan import InterventionPlan

        return InterventionPlan
    if kind == "intervention_event_log":
        from wsp.interventions.plan_models import InterventionEventLog

        return InterventionEventLog
    if kind == "temporal_intervention":
        from wsp.interventions.plan_models import TemporalInterventionRun

        return TemporalInterventionRun
    if kind == "compositional_comparison":
        from wsp.interventions.plan_models import CompositionalComparison

        return CompositionalComparison
    if kind == "staged_verification":
        from wsp.interventions.plan_models import StagedVerification

        return StagedVerification
    if kind == "staged_continuation":
        from wsp.interventions.plan_models import StagedContinuation

        return StagedContinuation
    if kind == "behavioral_event":
        from wsp.behavior.events import BehavioralEvent

        return BehavioralEvent
    if kind == "event_trigger":
        from wsp.interventions.event_trigger import EventTrigger

        return EventTrigger
    if kind == "event_intervention":
        from wsp.interventions.event_trigger import EventInterventionRun

        return EventInterventionRun
    if kind == "fixed_context_replay":
        from wsp.interventions.fixed_context import FixedContextReplay

        return FixedContextReplay
    if kind == "fixed_context_comparison":
        from wsp.interventions.fixed_context import FixedContextComparison

        return FixedContextComparison
    if kind == "investigation_dossier":
        from wsp.investigation.dossier import InvestigationDossier

        return InvestigationDossier
    if kind == "recipe_run":
        from wsp.recipes.models import RecipeRun

        return RecipeRun
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
            f"a linked {kind} artifact does not match its schema, so redaction "
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
    if "model_class" in model:
        removed.drop(model.get("model_class"))
        model["model_class"] = REDACTED
    if "config_architectures" in model:
        removed.drop(model.get("config_architectures"))
        model["config_architectures"] = []
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
