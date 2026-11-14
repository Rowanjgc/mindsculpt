"""Tool handlers for token alignment and span resolution."""

from __future__ import annotations

from typing import Any

from wsp.alignment.build import build_alignment
from wsp.alignment.models import SCHEMA_ALIGNMENT, TokenAlignment
from wsp.alignment.spans import resolve_span
from wsp.behavior.trace import TraceStore
from wsp.core.budget import Budget
from wsp.core.errors import ArtifactError, InputError
from wsp.replay.sources import load_replay_source
from wsp.tools.alignment_schemas import AlignTraceTokensInput, ResolveFindingSpanInput

ALIGNMENT_ARTIFACT_KIND = "token_alignment"
SPAN_ARTIFACT_KIND = "span_resolution"


def _result(
    data: dict[str, Any],
    artifacts: list[Any] | None = None,
    warnings: list[str] | None = None,
):
    from wsp.tools.service import ToolResult

    return ToolResult(data=data, artifacts=list(artifacts or []), warnings=list(warnings or []))


def _require_workspace(service: Any) -> Any:
    if service.workspace is None:
        raise InputError(
            "this operation requires a workspace; run 'wsp init' or pass --workspace",
            details={"hint": "the workspace stores traces and alignment evidence"},
        )
    return service.workspace


def _tokenizer_context(
    model_source: Any,
    *,
    prompt_text: str | None,
    prompt_ids: list[int],
    generation_text: str | None,
    generation_ids: list[int],
) -> tuple[
    list[tuple[int, int]] | None,
    list[tuple[int, int]] | None,
    set[int] | None,
    Any,
    list[str],
]:
    """Ask a model's tokenizer for offsets and special ids, validating them.

    Offsets are only used when they describe exactly the recorded identifiers;
    otherwise they are dropped with a warning, because a misaligned offset table
    is worse than an explicit gap.
    """

    if model_source is None:
        return None, None, None, None, []
    model = load_replay_source(model_source)
    support = model.tokenizer_support()
    warnings: list[str] = []
    prompt_offsets = None
    generation_offsets = None
    if support.encode_with_offsets is not None and prompt_text is not None:
        try:
            ids, offsets = support.encode_with_offsets(prompt_text)
        except Exception as exc:  # pragma: no cover - tokenizer specific
            warnings.append(f"tokenizer offsets could not be produced: {exc}")
        else:
            if list(ids) == list(prompt_ids):
                prompt_offsets = offsets
            else:
                warnings.append(
                    "tokenizer offsets were not used: encoding the prompt text did not reproduce the "
                    "recorded token identifiers"
                )
    if support.encode_with_offsets is not None and generation_text is not None:
        try:
            ids, offsets = support.encode_with_offsets(generation_text)
        except Exception as exc:  # pragma: no cover - tokenizer specific
            warnings.append(f"output tokenizer offsets could not be produced: {exc}")
        else:
            if list(ids) == list(generation_ids):
                generation_offsets = offsets
            else:
                warnings.append(
                    "output tokenizer offsets were not used: encoding the output text did not reproduce "
                    "the recorded token identifiers"
                )
    return (
        prompt_offsets,
        generation_offsets,
        set(support.special_ids) or None,
        support.decode,
        warnings,
    )


def _alignment_for(
    workspace: Any,
    request: Any,
    budget: Budget,
) -> tuple[TokenAlignment, list[Any], list[str]]:
    traces = TraceStore(workspace)
    alignment_artifact_id = getattr(request, "alignment_artifact_id", None)
    if alignment_artifact_id is not None:
        ref, _path = workspace.artifacts.get(alignment_artifact_id)
        if ref.kind != ALIGNMENT_ARTIFACT_KIND:
            raise ArtifactError(
                "the referenced artifact is not a token alignment",
                details={"artifact_id": alignment_artifact_id, "kind": ref.kind},
            )
        payload = workspace.artifacts.read_json(alignment_artifact_id)
        alignment = TokenAlignment.model_validate(payload)
        if alignment.schema_version != SCHEMA_ALIGNMENT:
            raise ArtifactError(
                "unsupported alignment schema",
                details={"schema_version": alignment.schema_version},
            )
        if request.trace_id is not None and alignment.trace_id != request.trace_id:
            raise InputError(
                "the stored alignment belongs to a different trace",
                details={"alignment_trace": alignment.trace_id, "requested": request.trace_id},
            )
        return alignment, [ref], []

    assert request.trace_id is not None
    trace = traces.load(request.trace_id)
    record = traces.get_record(request.trace_id)
    generation_ids = [token.token_id for token in trace.tokens if token.token_id is not None]
    prompt_ids = [int(value) for value in (trace.input.token_ids or [])]
    (
        prompt_offsets,
        generation_offsets,
        special_from_model,
        decoder,
        warnings,
    ) = _tokenizer_context(
        request.model,
        prompt_text=trace.input.text,
        prompt_ids=prompt_ids,
        generation_text=trace.text,
        generation_ids=generation_ids,
    )
    special_ids = (
        set(request.special_token_ids)
        if request.special_token_ids is not None
        else special_from_model
    )
    alignment = build_alignment(
        trace,
        budget=budget,
        offsets=prompt_offsets,
        generation_offsets=generation_offsets,
        tokenizer=decoder,
        special_token_ids=special_ids,
        trace_artifact_sha256=record.artifact.sha256,
    )
    warnings.extend(alignment.warnings)
    artifacts: list[Any] = []
    if request.save:
        ref = workspace.artifacts.put_json(
            ALIGNMENT_ARTIFACT_KIND,
            alignment.model_dump(mode="json"),
            metadata={
                "schema_version": alignment.schema_version,
                "alignment_id": alignment.alignment_id,
                "trace_id": alignment.trace_id,
                "prediction_semantics": alignment.prediction_semantics,
            },
        )
        artifacts.append(ref)
    return alignment, artifacts, warnings


def align_trace_tokens(service: Any, request: AlignTraceTokensInput, budget: Budget, run: Any):
    """Map every recorded trace token onto a model-forward sequence position."""

    workspace = _require_workspace(service)
    alignment, artifacts, warnings = _alignment_for(workspace, request, budget)
    positions = alignment.positions
    page = positions[request.position_offset : request.position_offset + request.position_limit]
    data: dict[str, Any] = {
        "alignment_id": alignment.alignment_id,
        "alignment_artifact_id": artifacts[0].artifact_id if artifacts else None,
        "trace_id": alignment.trace_id,
        "schema_version": alignment.schema_version,
        "prediction_semantics": alignment.prediction_semantics,
        "prompt_length": alignment.prompt_length,
        "generated_length": alignment.generated_length,
        "full_length": alignment.full_length,
        "prompt_text_status": alignment.prompt_text_status,
        "generation_text_status": alignment.generation_text_status,
        "segments": [segment.model_dump(mode="json") for segment in alignment.segments],
        "special_positions": alignment.special_positions,
        "warnings": alignment.warnings,
        "unavailable": alignment.unavailable,
        "position_offset": request.position_offset,
        "position_total": len(positions),
        "positions": (
            [position.model_dump(mode="json") for position in page]
            if request.include_positions
            else None
        ),
    }
    return _result(data, artifacts, warnings)


def resolve_finding_span(service: Any, request: ResolveFindingSpanInput, budget: Budget, run: Any):
    """Resolve a behavioral span (including detector evidence) onto token positions."""

    workspace = _require_workspace(service)
    alignment, alignment_artifacts, warnings = _alignment_for(workspace, request, budget)
    traces = TraceStore(workspace)
    trace = traces.load(alignment.trace_id)
    resolution = resolve_span(
        alignment,
        kind=request.kind,
        start=request.start,
        end=request.end,
        role=request.role,
        prompt_text=trace.input.text,
        generation_text=trace.text,
        name=request.name,
        window=request.window,
        detail=request.detail,
        budget=budget,
    )
    artifacts = list(alignment_artifacts)
    if request.save:
        ref = workspace.artifacts.put_json(
            SPAN_ARTIFACT_KIND,
            resolution.model_dump(mode="json"),
            metadata={
                "schema_version": resolution.schema_version,
                "resolution_id": resolution.resolution_id,
                "trace_id": resolution.trace_id,
                "alignment_id": resolution.alignment_id,
                "status": resolution.status,
            },
        )
        artifacts.append(ref)
    data = resolution.model_dump(mode="json")
    data["alignment_artifact_id"] = (
        alignment_artifacts[0].artifact_id if alignment_artifacts else request.alignment_artifact_id
    )
    data["span_artifact_id"] = artifacts[-1].artifact_id if artifacts else None
    return _result(data, artifacts, warnings + resolution.warnings)


def register_alignment_tools(service: Any) -> None:
    from wsp.tools.service import ToolDefinition

    def bind(function):
        def handler(request, budget, run):
            return function(service, request, budget, run)

        return handler

    definitions = [
        (
            "align_trace_tokens",
            "Map every recorded trace token onto prompt-relative, generation-relative, and "
            "full-sequence positions, including which position predicts which token.",
            AlignTraceTokensInput,
            align_trace_tokens,
            "Token alignment with per-position token ids, text, character/byte ranges, special "
            "flags, and explicit ambiguity warnings.",
            True,
        ),
        (
            "resolve_finding_span",
            "Resolve a behavioral span or detector evidence item onto consuming and predicting "
            "token positions, or report that the mapping is ambiguous.",
            ResolveFindingSpanInput,
            resolve_finding_span,
            "Span resolution with full-sequence indices, predicting positions, context requirements, "
            "and ambiguity notes.",
            True,
        ),
    ]
    for name, description, schema, function, output, requires_workspace in definitions:
        service.register_tool(
            ToolDefinition(
                name=name,
                description=description,
                input_model=schema,
                handler=bind(function),
                output_description=output,
                requires_workspace=requires_workspace,
            )
        )


__all__ = ["align_trace_tokens", "register_alignment_tools", "resolve_finding_span"]
