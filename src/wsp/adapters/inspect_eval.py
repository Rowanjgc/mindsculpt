"""Dependency-free importer for Inspect AI evaluation logs.

Inspect is not imported.  The adapter reads the documented log payload: JSON
logs directly, and ``.eval`` archives through the standard-library zip reader.
Inspect 0.3.x writes ``.eval`` members with ZIP zstd compression, which requires
Python 3.14+ (or an external converter); older runtimes are told to export JSON
logs instead.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from wsp.behavior.ids import new_trace_id
from wsp.behavior.models import (
    BackendRef,
    ChatMessage,
    GenerationParams,
    GenerationTrace,
    InputRef,
    ModelRef,
    Provenance,
    TerminationInfo,
    TimingInfo,
    TokenRecord,
)
from wsp.behavior.trace import finalize_trace
from wsp.core.budget import Budget
from wsp.core.errors import InputError, NotFoundError, ResourceLimitError, UnsupportedError
from wsp.core.hashing import sha256_file
from wsp.core.time import utc_now_iso

ADAPTER_VERSION = "wsp.inspect/1.0"


def _content_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "".join(parts) if parts else None
    if isinstance(value, dict):
        return _content_text(value.get("content"))
    return None


def load_inspect_log(
    path: str | Path,
    *,
    max_member_bytes: int = 64 * 1024 * 1024,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise NotFoundError(f"Inspect log was not found: {source}")
    if source.suffix == ".json":
        if source.stat().st_size > max_member_bytes:
            raise ResourceLimitError(
                "Inspect JSON log exceeds the configured import size limit",
                details={"bytes": source.stat().st_size, "limit": max_member_bytes},
            )
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InputError(f"could not read Inspect JSON log: {exc}") from exc
        if not isinstance(payload, dict):
            raise InputError("Inspect JSON log must contain an object")
        header = {
            "eval": payload.get("eval", {}),
            "version": payload.get("version"),
            "status": payload.get("status"),
        }
        samples = payload.get("samples") or []
        if not isinstance(samples, list):
            raise InputError("Inspect JSON log has a malformed samples section")
        return header, [sample for sample in samples if isinstance(sample, dict)]
    if source.suffix == ".eval":
        try:
            archive = zipfile.ZipFile(source)
        except (OSError, zipfile.BadZipFile) as exc:
            raise InputError(f"could not read Inspect .eval archive: {exc}") from exc
        try:
            try:
                header_info = archive.getinfo("header.json")
                if header_info.file_size > max_member_bytes:
                    raise ResourceLimitError(
                        "Inspect .eval header exceeds the configured import size limit",
                        details={"bytes": header_info.file_size, "limit": max_member_bytes},
                    )
                header = json.loads(archive.read("header.json"))
            except NotImplementedError as exc:
                raise UnsupportedError(
                    "this Python runtime cannot read the zstd-compressed .eval archive; "
                    "write the log with log_format='json' or use Python 3.14+",
                    details={"path": str(source)},
                ) from exc
            except KeyError as exc:
                raise InputError("Inspect .eval archive has no header.json") from exc
            names = sorted(
                name
                for name in archive.namelist()
                if name.startswith("samples/") and name.endswith(".json")
            )
            samples: list[dict[str, Any]] = []
            for name in names:
                info = archive.getinfo(name)
                if info.file_size > max_member_bytes:
                    raise ResourceLimitError(
                        "an Inspect .eval sample exceeds the configured import size limit",
                        details={"member": name, "bytes": info.file_size, "limit": max_member_bytes},
                    )
                try:
                    sample = json.loads(archive.read(name))
                except NotImplementedError as exc:
                    raise UnsupportedError(
                        "this Python runtime cannot read zstd-compressed .eval samples; "
                        "write the log with log_format='json' or use Python 3.14+",
                        details={"member": name},
                    ) from exc
                if isinstance(sample, dict):
                    samples.append(sample)
        finally:
            archive.close()
        if not isinstance(header, dict):
            raise InputError("Inspect .eval header is malformed")
        return header, samples
    raise InputError(
        "unsupported Inspect log extension (expected .json or .eval)",
        details={"path": str(source)},
    )


def _params_from_config(config: dict[str, Any]) -> GenerationParams:
    kwargs: dict[str, Any] = {}
    mapping = {
        "max_new_tokens": config.get("max_tokens"),
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "top_k": config.get("top_k"),
        "seed": config.get("seed"),
    }
    for key, value in mapping.items():
        if value is not None and key != "seed":
            kwargs[key] = value
    stops = config.get("stop_seqs") or config.get("stop_sequences")
    if isinstance(stops, list) and all(isinstance(item, str) for item in stops):
        kwargs["stop_sequences"] = stops
    extras = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_seqs",
            "stop_sequences",
            "seed",
        }
        and value is not None
    }
    try:
        return GenerationParams(**kwargs, extra=extras)
    except Exception:
        return GenerationParams(extra=extras)


def _tokens_from_output(output: dict[str, Any]) -> list[TokenRecord]:
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    choice = choices[0]
    if not isinstance(choice, dict):
        return []
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else logprobs
    if not isinstance(content, list):
        return []
    tokens: list[TokenRecord] = []
    for item in content:
        if isinstance(item, str):
            tokens.append(TokenRecord(text=item))
        elif isinstance(item, dict):
            token_text = item.get("token")
            logprob = item.get("logprob")
            tokens.append(
                TokenRecord(
                    token_id=None,
                    text=token_text if isinstance(token_text, str) else None,
                    logprob=(
                        float(logprob)
                        if isinstance(logprob, (int, float)) and not isinstance(logprob, bool)
                        else None
                    ),
                )
            )
    return tokens


def _targets(sample: dict[str, Any]) -> list[str]:
    target = sample.get("target")
    if isinstance(target, str):
        return [target]
    if isinstance(target, list):
        return [str(item) for item in target if isinstance(item, str)]
    return []


def _termination(sample: dict[str, Any], output: dict[str, Any]) -> tuple[TerminationInfo, list[str]]:
    warnings: list[str] = []
    error = sample.get("error")
    if error:
        return (
            TerminationInfo(
                reason="error",
                detail=error if isinstance(error, str) else json.dumps(error, ensure_ascii=False),
            ),
            warnings,
        )
    stop_reason = output.get("stop_reason")
    choices = output.get("choices")
    if stop_reason is None and isinstance(choices, list) and choices and isinstance(choices[0], dict):
        stop_reason = choices[0].get("stop_reason")
    mapping = {
        "stop": "eos",
        "max_tokens": "length_limit",
        "model_length": "length_limit",
        "tool_calls": "tool_handoff",
        "content_filter": "unknown",
        "length": "length_limit",
    }
    reason = mapping.get(stop_reason, "unknown")
    if stop_reason == "stop":
        warnings.append(
            "Inspect stop_reason='stop' does not distinguish natural EOS from a configured stop sequence"
        )
    return TerminationInfo(reason=reason, detail=str(stop_reason) if stop_reason else None), warnings


def import_inspect_log(
    path: str | Path,
    *,
    budget: Budget,
    sample_ids: list[str] | None = None,
) -> tuple[list[GenerationTrace], list[str]]:
    source = Path(path).expanduser().resolve()
    header, samples = load_inspect_log(source, max_member_bytes=budget.max_trace_bytes)
    if len(samples) > budget.max_import_traces:
        raise ResourceLimitError(
            "Inspect log contains more samples than the configured import limit",
            details={"samples": len(samples), "limit": budget.max_import_traces},
        )
    if sample_ids:
        wanted = set(sample_ids)
        samples = [sample for sample in samples if str(sample.get("id")) in wanted]
        if not samples:
            raise InputError(
                "none of the requested sample ids were present in the log",
                details={"sample_ids": sorted(wanted)},
            )
    eval_info = header.get("eval") if isinstance(header.get("eval"), dict) else {}
    config = eval_info.get("model_generate_config")
    params = _params_from_config(config if isinstance(config, dict) else {})
    seed = config.get("seed") if isinstance(config, dict) else None
    if not isinstance(seed, int) or isinstance(seed, bool):
        seed = None
    source_sha = sha256_file(source)
    warnings: list[str] = []
    traces: list[GenerationTrace] = []
    for sample in samples:
        sample_id = str(sample.get("id", len(traces)))
        output = sample.get("output") if isinstance(sample.get("output"), dict) else {}
        text = _content_text(output.get("completion"))
        if text is None:
            choices = output.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if isinstance(message, dict):
                    text = _content_text(message.get("content"))
        messages_raw = sample.get("messages")
        input_messages: list[ChatMessage] = []
        if isinstance(messages_raw, list):
            for message in messages_raw:
                if not isinstance(message, dict):
                    continue
                if message.get("source") == "generate":
                    continue
                role = message.get("role")
                content = _content_text(message.get("content"))
                if role in {"system", "user", "tool"} and content is not None:
                    input_messages.append(ChatMessage(role=role, content=content))
        input_text = _content_text(sample.get("input"))
        if input_text is None:
            input_text = "\n".join(message.content for message in input_messages) or None
        tokens = _tokens_from_output(output)
        termination, termination_warnings = _termination(sample, output)
        warnings.extend(termination_warnings)
        unavailable = {
            "token_ids": "Inspect logs do not include tokenizer ids in the portable log payload",
            "tokenizer": "Inspect logs do not include tokenizer identity",
        }
        if not tokens:
            unavailable["tokens"] = "Inspect log did not include a token log-probability stream"
            unavailable["token_logprobs"] = "Inspect log did not include token log-probabilities"
        total_time = sample.get("total_time")
        duration_ms = (
            float(total_time) * 1000.0
            if isinstance(total_time, (int, float)) and not isinstance(total_time, bool)
            else None
        )
        usage = output.get("usage") if isinstance(output.get("usage"), dict) else {}
        if usage.get("output_tokens") is not None:
            unavailable.pop("generated_token_count", None)
        metadata: dict[str, Any] = {
            "inspect_sample_id": sample_id,
            "inspect_sample_uuid": sample.get("uuid"),
            "inspect_epoch": sample.get("epoch"),
            "inspect_scores": sample.get("scores"),
            "inspect_usage": usage,
            "inspect_stop_reason": output.get("stop_reason"),
            "inspect_sample_metadata": sample.get("metadata"),
            "inspect_working_time": sample.get("working_time"),
            "inspect_turn_count": sample.get("turn_count"),
            "inspect_task": eval_info.get("task"),
            "inspect_task_id": eval_info.get("task_id"),
            "inspect_eval_id": eval_info.get("eval_id"),
            "inspect_run_id": eval_info.get("run_id"),
            "inspect_model_args": eval_info.get("model_args"),
            "inspect_packages": eval_info.get("packages"),
        }
        trace = GenerationTrace(
            trace_id=new_trace_id(),
            created_at=(
                sample.get("completed_at")
                if isinstance(sample.get("completed_at"), str)
                else utc_now_iso()
            ),
            model=ModelRef(
                name=eval_info.get("model") if isinstance(eval_info.get("model"), str) else None,
                revision=None,
                metadata={
                    "model_args": eval_info.get("model_args"),
                    "provider_model_generate_config": config,
                },
            ),
            tokenizer=None,
            input=InputRef(
                text=input_text,
                messages=input_messages,
                sha256=None,
            ),
            backend=BackendRef(
                name="inspect-import",
                version=str(header.get("version")) if header.get("version") else None,
                deterministic=None,
                runtime={
                    "inspect_version": header.get("version"),
                    "packages": eval_info.get("packages"),
                    "adapter": ADAPTER_VERSION,
                },
            ),
            params=params,
            seed=seed,
            tokens=tokens,
            text=text,
            termination=termination,
            timing=TimingInfo(
                started_at=sample.get("started_at") if isinstance(sample.get("started_at"), str) else None,
                completed_at=(
                    sample.get("completed_at")
                    if isinstance(sample.get("completed_at"), str)
                    else None
                ),
                duration_ms=duration_ms,
                tokens_per_second=None,
            ),
            expected_answers=_targets(sample),
            provenance=Provenance(
                origin="imported",
                source_system="inspect_ai",
                source_version=str(header.get("version")) if header.get("version") else None,
                source_ref=str(source),
                source_sha256=source_sha,
                importer=ADAPTER_VERSION,
                imported_at=utc_now_iso(),
            ),
            unavailable=unavailable,
            warnings=list(warnings),
            metadata=metadata,
        )
        traces.append(finalize_trace(trace, budget))
    return traces, warnings
