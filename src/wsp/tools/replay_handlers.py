"""Tool handlers for verified execution and teacher-forced replay."""

from __future__ import annotations

from typing import Any

from wsp.behavior.trace import TraceStore
from wsp.core.budget import Budget
from wsp.core.errors import ArtifactError, InputError
from wsp.replay.engine import execute_replay
from wsp.replay.models import SCHEMA_PLAN, ReplayPlan
from wsp.replay.plan import plan_checksum, prepare_plan
from wsp.replay.sources import load_replay_source
from wsp.tools.replay_schemas import PrepareReplayInput, VerifyReplayInput

PLAN_ARTIFACT_KIND = "replay_plan"
RESULT_ARTIFACT_KIND = "replay_result"


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
            details={"hint": "the workspace stores traces, plans, and replay evidence"},
        )
    return service.workspace


def _load_plan(workspace: Any, artifact_id: str) -> tuple[ReplayPlan, Any]:
    ref, _path = workspace.artifacts.get(artifact_id)
    if ref.kind != PLAN_ARTIFACT_KIND:
        raise ArtifactError(
            "the referenced artifact is not a replay plan",
            details={"artifact_id": artifact_id, "kind": ref.kind},
        )
    payload = workspace.artifacts.read_json(artifact_id)
    plan = ReplayPlan.model_validate(payload)
    if plan.schema_version != SCHEMA_PLAN:
        raise ArtifactError(
            "unsupported replay plan schema",
            details={"schema_version": plan.schema_version, "supported": [SCHEMA_PLAN]},
        )
    return plan, ref


def prepare_replay(service: Any, request: PrepareReplayInput, budget: Budget, run: Any):
    """Turn a stored trace into an explicit, versioned replay plan."""

    workspace = _require_workspace(service)
    traces = TraceStore(workspace)
    trace = traces.load(request.trace_id)
    record = traces.get_record(request.trace_id)
    plan = prepare_plan(
        trace,
        trace_artifact_id=record.artifact.artifact_id,
        trace_artifact_sha256=record.artifact.sha256,
        execution=request.execution,
        tolerances=request.tolerances,
        budget=budget,
        notes=request.notes,
    )
    warnings = list(plan.warnings)
    artifact_ref = None
    if request.save:
        artifact_ref = workspace.artifacts.put_json(
            PLAN_ARTIFACT_KIND,
            plan.model_dump(mode="json"),
            metadata={
                "schema_version": plan.schema_version,
                "trace_id": plan.trace_id,
                "plan_id": plan.plan_id,
                "semantics": plan.semantics,
                "run_id": getattr(run, "run_id", None),
            },
        )
        warnings.extend(
            [
                "verify_replay needs a model source; this plan is not a measurement yet",
            ]
        )
    statuses = {check.name: check.status for check in plan.checks}
    data: dict[str, Any] = {
        "plan_id": plan.plan_id,
        "schema_version": plan.schema_version,
        "trace_id": plan.trace_id,
        "semantics": plan.semantics,
        "plan_checksum": plan_checksum(plan),
        "sequence": {
            "source": plan.sequence.source,
            "prompt_length": plan.sequence.prompt_length,
            "generated_length": plan.sequence.generated_length,
            "full_length": plan.sequence.full_length,
            "sequence_sha256": plan.sequence.sequence_sha256,
            "token_encoding": plan.sequence.token_encoding,
        },
        "checks": [check.model_dump(mode="json") for check in plan.checks],
        "check_statuses": statuses,
        "failed_checks": [check.name for check in plan.failed_checks()],
        "limitations": plan.limitations,
        "plan_artifact_id": artifact_ref.artifact_id if artifact_ref else None,
        "plan": plan.model_dump(mode="json") if not request.save else None,
    }
    return _result(data, [artifact_ref] if artifact_ref else [], warnings)


def verify_replay(service: Any, request: VerifyReplayInput, budget: Budget, run: Any):
    """Execute one teacher-forced replay and report what actually agreed."""

    workspace = _require_workspace(service)
    traces = TraceStore(workspace)
    plan_artifact_id = request.plan_artifact_id
    plan_ref = None
    if plan_artifact_id is not None:
        plan, plan_ref = _load_plan(workspace, plan_artifact_id)
        if request.execution is not None or request.tolerances is not None:
            updates: dict[str, Any] = {}
            if request.execution is not None:
                updates["execution"] = request.execution
            if request.tolerances is not None:
                updates["tolerances"] = request.tolerances
            plan = plan.model_copy(update=updates)
        trace_id = plan.trace_id
    else:
        assert request.trace_id is not None
        trace = traces.load(request.trace_id)
        record = traces.get_record(request.trace_id)
        plan = prepare_plan(
            trace,
            trace_artifact_id=record.artifact.artifact_id,
            trace_artifact_sha256=record.artifact.sha256,
            execution=request.execution,
            tolerances=request.tolerances,
            budget=budget,
        )
        trace_id = request.trace_id
        if request.save:
            plan_ref = workspace.artifacts.put_json(
                PLAN_ARTIFACT_KIND,
                plan.model_dump(mode="json"),
                metadata={
                    "schema_version": plan.schema_version,
                    "trace_id": plan.trace_id,
                    "plan_id": plan.plan_id,
                    "semantics": plan.semantics,
                    "run_id": getattr(run, "run_id", None),
                },
            )

    model = load_replay_source(request.model)
    execution = execute_replay(plan, model, budget=budget)
    result = execution.result
    if not request.include_per_token and result.agreement is not None:
        result = result.model_copy(
            update={
                "agreement": result.agreement.model_copy(
                    update={"per_token": [], "per_token_truncated": True}
                )
            }
        )

    result_ref = None
    if request.save:
        payload = result.model_dump(mode="json")
        result_ref = workspace.artifacts.put_json(
            RESULT_ARTIFACT_KIND,
            payload,
            metadata={
                "schema_version": result.schema_version,
                "replay_id": result.replay_id,
                "trace_id": result.trace_id,
                "numerical_agreement": result.numerical_agreement,
                "evidence_level": result.evidence_level,
                "run_id": getattr(run, "run_id", None),
            },
        )

    data: dict[str, Any] = {
        "replay_id": result.replay_id,
        "schema_version": result.schema_version,
        "trace_id": trace_id,
        "plan_id": plan.plan_id,
        "plan_artifact_id": plan_ref.artifact_id if plan_ref else None,
        "result_artifact_id": result_ref.artifact_id if result_ref else None,
        "trace_artifact_sha256": result.trace_artifact_sha256,
        "semantics": result.semantics,
        "token_sequence_reconstructed": result.token_sequence_reconstructed,
        "forward_executed": result.forward_executed,
        "numerical_agreement": result.numerical_agreement,
        "generation_reproducibility": result.generation_reproducibility,
        "evidence_level": result.evidence_level,
        "exploratory_reason": result.exploratory_reason,
        "agreement": {
            "level": result.agreement.level,
            "tokens_compared": result.agreement.tokens_compared,
            "tokens_within_tolerance": result.agreement.tokens_within_tolerance,
            "within_tolerance_fraction": result.agreement.within_tolerance_fraction,
            "top1_agreement_rate": result.agreement.top1_agreement_rate,
            "max_abs_diff": result.agreement.max_abs_diff,
            "mean_abs_diff": result.agreement.mean_abs_diff,
            "correlation": result.agreement.correlation,
            "unavailable_reason": result.agreement.unavailable_reason,
            "tolerance": result.agreement.tolerance.model_dump(mode="json"),
            "per_token": [entry.model_dump(mode="json") for entry in result.agreement.per_token],
            "per_token_truncated": result.agreement.per_token_truncated,
        },
        "forward": result.forward.model_dump(mode="json") if result.forward else None,
        "checks": [check.model_dump(mode="json") for check in result.checks],
        "identity": result.identity.model_dump(mode="json"),
        "special_token_ids": result.special_token_ids,
        "special_positions": result.special_positions,
        "limitations": result.limitations,
        "warnings": result.warnings,
        "metadata": result.metadata,
        "result": result.model_dump(mode="json") if not request.save else None,
    }
    artifacts = [ref for ref in (plan_ref, result_ref) if ref is not None]
    return _result(data, artifacts, list(result.warnings))


def register_replay_tools(service: Any) -> None:
    from wsp.tools.service import ToolDefinition

    def bind(function):
        def handler(request, budget, run):
            return function(service, request, budget, run)

        return handler

    definitions = [
        (
            "prepare_replay",
            "Resolve a stored trace into an explicit teacher-forced replay plan with identity and "
            "metadata checks. Preparing a plan is not a measurement.",
            PrepareReplayInput,
            prepare_replay,
            "Replay plan, checksum, per-check statuses, and the plan artifact identifier.",
            True,
        ),
        (
            "verify_replay",
            "Execute a teacher-forced replay of recorded tokens and report token-sequence "
            "reconstruction, forward execution, and measured numerical agreement separately.",
            VerifyReplayInput,
            verify_replay,
            "Replay result with separate reconstruction/execution/agreement statuses and per-token "
            "log-probability differences.",
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


__all__ = ["register_replay_tools", "prepare_replay", "verify_replay"]
