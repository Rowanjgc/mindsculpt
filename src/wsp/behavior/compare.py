"""Deterministic trace comparison without judgements."""

from __future__ import annotations

from typing import Any

from wsp.behavior.models import GenerationTrace
from wsp.behavior.trace import sha256_text


def _same_optional(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return left == right


def compare_traces(left: GenerationTrace, right: GenerationTrace) -> dict[str, Any]:
    warnings: list[str] = []
    left_tokens = left.tokens
    right_tokens = right.tokens
    left_ids_available = bool(left_tokens) and all(token.token_id is not None for token in left_tokens)
    right_ids_available = bool(right_tokens) and all(
        token.token_id is not None for token in right_tokens
    )
    use_ids = left_ids_available and right_ids_available
    left_keys = [
        token.token_id if use_ids else token.text for token in left_tokens
    ]
    right_keys = [
        token.token_id if use_ids else token.text for token in right_tokens
    ]
    common = 0
    for left_key, right_key in zip(left_keys, right_keys):
        if left_key != right_key or left_key is None or right_key is None:
            break
        common += 1
    tokens_identical = bool(left_keys) and left_keys == right_keys
    if not use_ids:
        warnings.append(
            "token identity comparison used token text because token ids were unavailable on at least one trace"
        )
    left_input_sha = left.input.sha256 or sha256_text(left.input.text)
    right_input_sha = right.input.sha256 or sha256_text(right.input.text)
    output_identical = left.text is not None and right.text is not None and left.text == right.text
    if left.text is None or right.text is None:
        warnings.append("output text is unavailable on at least one trace")
    return {
        "schema_version": "wsp.trace-comparison/1.0",
        "left_trace_id": left.trace_id,
        "right_trace_id": right.trace_id,
        "model": {
            "same": _same_optional(
                (left.model.name, left.model.revision, left.model.structure_fingerprint),
                (right.model.name, right.model.revision, right.model.structure_fingerprint),
            ),
            "left": left.model.name,
            "right": right.model.name,
        },
        "input": {
            "same_sha256": (
                left_input_sha == right_input_sha
                if left_input_sha is not None and right_input_sha is not None
                else None
            ),
            "left_sha256": left_input_sha,
            "right_sha256": right_input_sha,
        },
        "params": {
            "same": left.params.model_dump(mode="json") == right.params.model_dump(mode="json"),
            "left": left.params.model_dump(mode="json"),
            "right": right.params.model_dump(mode="json"),
        },
        "seed": {
            "same": left.seed == right.seed,
            "left": left.seed,
            "right": right.seed,
        },
        "output": {
            "identical": output_identical if (left.text is not None and right.text is not None) else None,
            "left_sha256": sha256_text(left.text),
            "right_sha256": sha256_text(right.text),
            "left_chars": len(left.text or ""),
            "right_chars": len(right.text or ""),
            "similarity_note": "exact byte comparison; no semantic similarity is computed",
        },
        "tokens": {
            "left_count": len(left_tokens),
            "right_count": len(right_tokens),
            "identity_basis": "token_id" if use_ids else "token_text",
            "identical": tokens_identical,
            "common_prefix": common,
            "first_divergence": (
                None
                if tokens_identical
                else common
                if common < max(len(left_keys), len(right_keys))
                else None
            ),
            "logprobs_available": all(token.logprob is not None for token in left_tokens)
            and all(token.logprob is not None for token in right_tokens),
        },
        "termination": {
            "changed": left.termination.reason != right.termination.reason,
            "left": left.termination.model_dump(mode="json"),
            "right": right.termination.model_dump(mode="json"),
        },
        "annotations": {
            "left": len(left.annotations),
            "right": len(right.annotations),
            "left_names": sorted({annotation.name for annotation in left.annotations}),
            "right_names": sorted({annotation.name for annotation in right.annotations}),
        },
        "timing": {
            "left_duration_ms": left.timing.duration_ms,
            "right_duration_ms": right.timing.duration_ms,
            "delta_ms": (
                right.timing.duration_ms - left.timing.duration_ms
                if left.timing.duration_ms is not None and right.timing.duration_ms is not None
                else None
            ),
        },
        "warnings": warnings,
    }
