"""Dependency-free adapter for INIF format 0.1 documents.

INIF is an evolving interchange format.  This module deliberately implements
only the documented JSON shape, works without the ``inif`` package, and records
the targeted format version.  Unknown sample fields and per-token extras are
preserved in the trace metadata instead of being silently discarded.
"""

from __future__ import annotations

from typing import Any

from wsp import __version__
from wsp.behavior.ids import new_trace_id
from wsp.behavior.models import (
    BackendRef,
    ChatMessage,
    GenerationParams,
    GenerationTrace,
    InputRef,
    ModelRef,
    Provenance,
    SpanAnnotation,
    TerminationInfo,
    TimingInfo,
    TokenRecord,
)
from wsp.behavior.trace import finalize_trace
from wsp.core.budget import Budget
from wsp.core.errors import InputError, ResourceLimitError
from wsp.core.time import utc_now_iso

INIF_VERSION = "0.1"
ADAPTER_VERSION = "wsp.inif/1.0"

_KNOWN_SAMPLE_FIELDS = {
    "id",
    "tokens",
    "texts",
    "annotations",
    "spans",
    "scores",
    "target",
    "references",
    "choices",
    "interaction_type",
    "error",
    "sample_hash",
    "metadata",
    "total_time",
    "input_tokens",
    "output_tokens",
}


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _text_role(name: str) -> str:
    if name.startswith("assistant"):
        return "assistant"
    if name.startswith("system"):
        return "system"
    if name.startswith("tool"):
        return "tool"
    return "user"


def _token_extras(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key not in {"token", "id"}
    }


def _expand_tokens(
    tokens: Any,
    sequences: dict[str, list[dict[str, Any]]],
    warnings: list[str],
) -> tuple[list[str], list[int | None], list[dict[str, Any]], list[tuple[int, int]]]:
    texts: list[str] = []
    ids: list[int | None] = []
    extras: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    if not isinstance(tokens, list):
        return texts, ids, extras, spans
    pending = list(tokens)
    guard = 0
    while pending:
        guard += 1
        if guard > 1_000_000:  # defensive against cyclic sequence references
            warnings.append("token expansion exceeded the safety limit")
            break
        item = pending.pop(0)
        if not isinstance(item, dict):
            warnings.append("a token entry was not an object and was skipped")
            continue
        token_id = item.get("id")
        token_text = item.get("token")
        if token_id is None and isinstance(token_text, str) and token_text in sequences:
            start = len(texts)
            for sequence_item in sequences[token_text]:
                texts.append(
                    sequence_item.get("token")
                    if isinstance(sequence_item.get("token"), str)
                    else ""
                )
                ids.append(_as_int(sequence_item.get("id")))
                extras.append(_token_extras(sequence_item))
            spans.append((start, len(texts)))
            continue
        start = len(texts)
        texts.append(token_text if isinstance(token_text, str) else "")
        ids.append(_as_int(token_id))
        extras.append(_token_extras(item))
        spans.append((start, len(texts)))
    return texts, ids, extras, spans


def import_inif_document(
    payload: Any,
    *,
    budget: Budget,
    source_ref: str | None = None,
    source_sha256: str | None = None,
) -> tuple[list[GenerationTrace], list[str]]:
    if not isinstance(payload, dict):
        raise InputError("INIF document must be a JSON object")
    metadata = payload.get("metadata")
    samples = payload.get("samples")
    if not isinstance(metadata, dict) or not isinstance(samples, list):
        raise InputError("INIF document must contain metadata and samples")
    inif_version = metadata.get("inif_version")
    if inif_version is not None and str(inif_version) != INIF_VERSION:
        warnings_version = (
            f"INIF format version {inif_version!r} differs from the supported version {INIF_VERSION!r}"
        )
    else:
        warnings_version = None
    warnings: list[str] = []
    if warnings_version:
        warnings.append(warnings_version)
    if len(samples) > budget.max_import_traces:
        raise ResourceLimitError(
            "INIF document has more samples than the configured import limit",
            details={"samples": len(samples), "limit": budget.max_import_traces},
        )
    sequences: dict[str, list[dict[str, Any]]] = {}
    for sequence in payload.get("sequences", []) or []:
        if isinstance(sequence, dict) and isinstance(sequence.get("id"), str):
            items = sequence.get("tokens")
            if isinstance(items, list):
                sequences[sequence["id"]] = [item for item in items if isinstance(item, dict)]

    model_info = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
    generation_config = model_info.get("generation_config")
    params = _params_from_inif(generation_config if isinstance(generation_config, dict) else {})
    traces: list[GenerationTrace] = []
    for sample in samples:
        if not isinstance(sample, dict):
            warnings.append("a sample was not an object and was skipped")
            continue
        sample_id = str(sample.get("id", len(traces)))
        texts = sample.get("texts") if isinstance(sample.get("texts"), list) else []
        input_parts: list[str] = []
        output_parts: list[str] = []
        messages: list[ChatMessage] = []
        assistant_start: int | None = None
        assistant_end: int | None = None
        for text in texts:
            if not isinstance(text, dict):
                continue
            name = str(text.get("name", "text"))
            value = text.get("value")
            if not isinstance(value, str):
                continue
            role = _text_role(name)
            if role == "assistant":
                output_parts.append(value)
                start = _as_int(text.get("start"))
                end = _as_int(text.get("end"))
                if start is not None and end is not None:
                    assistant_start = start if assistant_start is None else min(assistant_start, start)
                    assistant_end = end if assistant_end is None else max(assistant_end, end)
            else:
                input_parts.append(value)
                if role in {"system", "user", "tool"}:
                    messages.append(ChatMessage(role=role, content=value))
        token_texts, token_ids, token_extras, token_spans = _expand_tokens(
            sample.get("tokens", []), sequences, warnings
        )

        def expanded_range(start: int, end: int) -> tuple[int, int] | None:
            if not token_spans:
                return None
            start = max(0, min(start, len(token_spans)))
            end = max(0, min(end, len(token_spans)))
            if end <= start:
                return None
            return token_spans[start][0], token_spans[end - 1][1]

        offset = 0
        if assistant_start is not None and assistant_end is not None:
            expanded = expanded_range(assistant_start, assistant_end)
            if expanded is not None:
                offset, end = expanded
                token_texts = token_texts[offset:end]
                token_ids = token_ids[offset:end]
                token_extras = token_extras[offset:end]
        elif output_parts and token_texts:
            warnings.append(
                f"sample {sample_id}: assistant token range unavailable; token records may include prompt tokens"
            )
        tokens: list[TokenRecord] = []
        extras_by_index: dict[str, Any] = {}
        for index, (text, token_id, extra) in enumerate(zip(token_texts, token_ids, token_extras)):
            logprob = None
            candidate = extra.pop("logprob", None)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                logprob = float(candidate)
            tokens.append(TokenRecord(token_id=token_id, text=text, logprob=logprob))
            if extra:
                extras_by_index[str(index)] = extra

        annotations: list[SpanAnnotation] = []
        raw_annotations = sample.get("annotations")
        if isinstance(raw_annotations, list):
            for annotation in raw_annotations:
                if not isinstance(annotation, dict):
                    continue
                name = annotation.get("name")
                ranges = annotation.get("ranges")
                if not isinstance(name, str) or not isinstance(ranges, list):
                    continue
                for pair in ranges:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    start, end = _as_int(pair[0]), _as_int(pair[1])
                    if start is None or end is None:
                        continue
                    mapped = expanded_range(start, end)
                    if mapped is None:
                        warnings.append(
                            f"sample {sample_id}: annotation {name!r} fell outside the generated token range"
                        )
                        continue
                    start, end = mapped
                    start -= offset
                    end -= offset
                    if start < 0 or end > len(tokens) or end <= start:
                        warnings.append(
                            f"sample {sample_id}: annotation {name!r} fell outside the generated token range"
                        )
                        continue
                    annotations.append(
                        SpanAnnotation(
                            name=name,
                            start=start,
                            end=end,
                            metadata=annotation.get("metadata")
                            if isinstance(annotation.get("metadata"), dict)
                            else {},
                        )
                    )

        total_time = sample.get("total_time")
        duration_ms = float(total_time) * 1000.0 if isinstance(total_time, (int, float)) else None
        termination = TerminationInfo(reason="unknown")
        error = sample.get("error")
        if error:
            termination = TerminationInfo(
                reason="error",
                detail=error if isinstance(error, str) else str(error),
            )
        unavailable = {
            "termination": "INIF does not encode a termination reason",
            "tokenizer": "INIF does not identify the tokenizer",
        }
        if not tokens:
            unavailable["tokens"] = "INIF sample did not provide token records"
        if tokens and all(token.logprob is None for token in tokens):
            unavailable["token_logprobs"] = "INIF token extras did not include log-probabilities"

        target = sample.get("target")
        expected: list[str] = []
        if isinstance(target, str):
            expected = [target]
        elif isinstance(target, list):
            expected = [str(item) for item in target if isinstance(item, str)]
        metadata_extra: dict[str, Any] = {
            "inif_sample_id": sample_id,
            "inif_scores": sample.get("scores", []),
            "inif_spans": sample.get("spans", []),
            "inif_texts": texts,
            "inif_references": sample.get("references", []),
            "inif_unmapped_sample_fields": {
                key: value for key, value in sample.items() if key not in _KNOWN_SAMPLE_FIELDS
            },
            "inif_document": {
                "packages": metadata.get("packages", {}),
                "sources": metadata.get("sources", []),
                "source_eval": metadata.get("source_eval"),
                "created_at": metadata.get("created_at"),
                "total_time": metadata.get("total_time"),
                "extra": metadata.get("extra", {}),
                "unmapped_fields": {
                    key: value
                    for key, value in payload.items()
                    if key not in {"metadata", "sequences", "samples"}
                },
            },
        }
        if extras_by_index:
            metadata_extra["inif_token_extras"] = extras_by_index
        if sample.get("metadata") is not None:
            metadata_extra["inif_sample_metadata"] = sample.get("metadata")
        if sample.get("sample_hash") is not None:
            metadata_extra["inif_sample_hash"] = sample.get("sample_hash")
        if sample.get("input_tokens") is not None:
            metadata_extra["inif_input_tokens"] = sample.get("input_tokens")
        if sample.get("output_tokens") is not None:
            metadata_extra["inif_output_tokens"] = sample.get("output_tokens")

        trace = GenerationTrace(
            trace_id=new_trace_id(),
            created_at=utc_now_iso(),
            model=ModelRef(
                name=model_info.get("name") if isinstance(model_info.get("name"), str) else None,
                revision=model_info.get("revision")
                if isinstance(model_info.get("revision"), str)
                else None,
                metadata={
                    "huggingface_id": model_info.get("huggingface_id"),
                    "loading_config": model_info.get("loading_config"),
                },
            ),
            tokenizer=None,
            input=InputRef(
                text="\n".join(input_parts) if input_parts else None,
                messages=messages,
                sha256=None,
            ),
            backend=BackendRef(
                name="inif-import",
                version=str(inif_version) if inif_version is not None else None,
                deterministic=None,
                runtime={"inif_version": inif_version, "adapter": ADAPTER_VERSION},
            ),
            params=params,
            seed=None,
            tokens=tokens,
            text="".join(output_parts) if output_parts else None,
            termination=termination,
            timing=TimingInfo(
                started_at=None,
                completed_at=None,
                duration_ms=duration_ms,
                tokens_per_second=None,
            ),
            annotations=annotations,
            expected_answers=expected,
            provenance=Provenance(
                origin="imported",
                source_system="inif",
                source_version=str(inif_version) if inif_version is not None else None,
                source_ref=source_ref,
                source_sha256=source_sha256,
                importer=ADAPTER_VERSION,
                imported_at=utc_now_iso(),
            ),
            unavailable=unavailable,
            warnings=list(warnings),
            metadata=metadata_extra,
        )
        traces.append(finalize_trace(trace, budget))
    if not traces:
        warnings.append("INIF document contained no importable samples")
    return traces, warnings


def _params_from_inif(config: dict[str, Any]) -> GenerationParams:
    mapping = {
        "max_new_tokens": config.get("max_tokens") or config.get("max_new_tokens"),
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "top_k": config.get("top_k"),
        "repetition_penalty": config.get("repetition_penalty"),
    }
    kwargs: dict[str, Any] = {}
    for key, value in mapping.items():
        if value is not None:
            kwargs[key] = value
    stops = config.get("stop_seqs") or config.get("stop_sequences")
    if isinstance(stops, list) and all(isinstance(item, str) for item in stops):
        kwargs["stop_sequences"] = stops
    try:
        return GenerationParams(**kwargs)
    except Exception:
        return GenerationParams()


def export_inif(trace: GenerationTrace) -> dict[str, Any]:
    warnings: list[str] = []
    tokens: list[dict[str, Any]] = []
    can_export_tokens = bool(trace.tokens) and all(
        token.token_id is not None for token in trace.tokens
    )
    if can_export_tokens:
        for token in trace.tokens:
            item: dict[str, Any] = {"token": token.text or "", "id": token.token_id}
            if token.logprob is not None:
                item["logprob"] = token.logprob
            tokens.append(item)
    elif trace.tokens:
        warnings.append("token ids are unavailable; INIF tokens were omitted")

    texts: list[dict[str, Any]] = []
    input_text = trace.input.text
    if trace.input.system:
        texts.append({"name": "system_0", "value": trace.input.system})
    if input_text is not None:
        texts.append({"name": "user_0", "value": input_text})
    if trace.text is not None:
        assistant: dict[str, Any] = {"name": "assistant_0", "value": trace.text}
        if can_export_tokens:
            assistant["start"] = 0
            assistant["end"] = len(trace.tokens)
        texts.append(assistant)

    annotations: list[dict[str, Any]] = []
    if can_export_tokens:
        for annotation in trace.annotations:
            annotations.append(
                {
                    "name": annotation.name,
                    "ranges": [[annotation.start, annotation.end]],
                    "metadata": annotation.metadata,
                }
            )
    elif trace.annotations:
        warnings.append("token annotations were omitted because token ids are unavailable")

    generation_config: dict[str, Any] = {
        "max_tokens": trace.params.max_new_tokens,
        "temperature": trace.params.temperature,
        "top_p": trace.params.top_p,
        "top_k": trace.params.top_k,
        "stop_seqs": trace.params.stop_sequences,
    }
    sample = {
        "id": trace.trace_id,
        "tokens": tokens,
        "texts": texts,
        "annotations": annotations,
        "spans": [],
        "scores": [],
        "target": trace.expected_answers[0] if trace.expected_answers else None,
        "references": [],
        "metadata": {
            "wsp_trace_id": trace.trace_id,
            "wsp_backend": trace.backend.name,
            "wsp_provenance_origin": trace.provenance.origin,
            "wsp_termination": trace.termination.model_dump(mode="json"),
            "wsp_unavailable": trace.unavailable,
            "wsp_warnings": warnings,
        },
        "total_time": (
            trace.timing.duration_ms / 1000.0
            if trace.timing.duration_ms is not None
            else None
        ),
        "output_tokens": len(trace.tokens) if can_export_tokens else None,
        "error": trace.termination.detail if trace.termination.reason == "error" else None,
    }
    return {
        "metadata": {
            "model": {
                "name": trace.model.name,
                "revision": trace.model.revision,
                "huggingface_id": trace.model.metadata.get("huggingface_id"),
                "generation_config": generation_config,
                "loading_config": {},
            },
            "inif_version": INIF_VERSION,
            "packages": {"wsp-workbench": __version__},
            "sources": ["wsp-workbench"],
            "created_at": trace.created_at,
            "total_time": sample["total_time"],
            "extra": {
                "wsp_trace_id": trace.trace_id,
                "wsp_schema_version": trace.schema_version,
                "wsp_export_warnings": warnings,
            },
        },
        "sequences": [],
        "samples": [sample],
    }
