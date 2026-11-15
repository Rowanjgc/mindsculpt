"""Temporal plan, event, composition, and staged operations through the service.

Every command below goes through the same shared service layer the CLI uses, so
the schemas, envelopes, identifiers, and errors are the ones a caller sees.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from wsp.cli.main import app
from wsp.core.errors import CompatibilityError, InputError
from wsp.core.json import canonical_bytes
from wsp.interventions.plan import plan_sha256
from wsp.interventions.plan import InterventionPlan
from wsp.interventions.plan_models import (
    CompositionalComparison,
    InterventionEventLog,
    StagedContinuation,
    StagedVerification,
    TemporalInterventionRun,
)
from wsp.tools.service import ToolService

SOURCE = {
    "kind": "synthetic",
    "synthetic": {
        "name": "tiny",
        "model": {"seed": 7, "model_dim": 32, "num_layers": 3, "max_seq_len": 96},
    },
}
TOKENS = [1, 10, 11, 12]

TEMPORAL_OPERATIONS = {
    "create_intervention_plan",
    "validate_intervention_plan",
    "preview_intervention_plan",
    "run_temporal_intervention",
    "inspect_intervention_events",
    "run_compositional_comparison",
    "prepare_staged_generation",
    "verify_staged_generation",
    "run_staged_intervention",
    "attach_temporal_evidence",
}


def _record(model, data: dict):
    """Validate the versioned part of an envelope that also carries summaries."""

    return model.model_validate(
        {key: value for key, value in data.items() if key in model.model_fields}
    )


def _spec(module: str, *, label: str = "spec", positions=(0,)) -> dict:
    from support.temporal import make_spec

    return make_spec(
        module, prompt_length=len(TOKENS), positions=positions, label=label
    ).model_dump(mode="json")


def _create_plan(
    service: ToolService,
    modules: list[str] | None = None,
    *,
    steps: list[int] | None = None,
    execution_mode: str = "uncached-generation",
    name: str = "temporal plan",
    max_new_tokens: int = 4,
    save: bool = True,
) -> dict:
    modules = modules or ["blocks.1"]
    envelope = service.call(
        "create_intervention_plan",
        {
            "name": name,
            "prompt_token_ids": TOKENS,
            "execution_mode": execution_mode,
            "max_new_tokens": max_new_tokens,
            "interventions": [
                {
                    "intervention": _spec(module, label=f"site {index}"),
                    "order": index,
                    "schedule": steps if steps is not None else [1, 2],
                    "position_mode": "last",
                }
                for index, module in enumerate(modules)
            ],
            "save": save,
        },
    )
    assert envelope["ok"], envelope
    return envelope["data"]


def _run(service: ToolService, plan_artifact_id: str, *, token_ids=None, save=True) -> dict:
    envelope = service.call(
        "run_temporal_intervention",
        {
            "plan_artifact_id": plan_artifact_id,
            "source": SOURCE,
            "token_ids": token_ids or TOKENS,
            "save": save,
        },
    )
    assert envelope["ok"], envelope
    return envelope["data"]


def test_temporal_operations_are_registered_with_schemas(service):
    operations = service.operations()
    assert TEMPORAL_OPERATIONS <= set(operations)
    described = {
        entry["name"]: entry for entry in service.describe()["operations"]
    }
    for name in sorted(TEMPORAL_OPERATIONS):
        entry = described[name]
        assert entry["description"]
        assert entry["input_schema"]["properties"]
        assert entry["requires_workspace"] is True


def test_create_plan_validates_saves_and_records_a_checksum(service):
    data = _create_plan(service)
    assert data["valid"] is True
    assert data["artifact_id"]

    plan = _record(InterventionPlan, data)
    assert plan.schema_version == "wsp.intervention-plan/1.0"
    assert plan.plan_kind == "intervention"
    assert len(plan.interventions) == 1
    assert plan.interventions[0].schedule.resolve() == [1, 2]
    assert plan.interventions[0].position_mode == "last"
    assert plan.prompt_length == len(TOKENS)

    stored = service.workspace.artifacts.read_json(data["artifact_id"])
    assert stored["checksums"]["plan_sha256"] == plan_sha256(plan)
    assert canonical_bytes(stored) == canonical_bytes(plan.model_dump(mode="json"))


def test_invalid_plan_is_reported_and_not_saved(service):
    envelope = service.call(
        "create_intervention_plan",
        {
            "prompt_token_ids": TOKENS,
            "interventions": [
                {
                    "intervention": _spec("blocks.1"),
                    "order": 0,
                    "schedule": [1],
                    "position_mode": "last",
                },
                {
                    "intervention": _spec("blocks.1"),
                    "order": 1,
                    "schedule": [1],
                    "position_mode": "last",
                },
            ],
        },
    )
    assert envelope["ok"], envelope
    data = envelope["data"]
    assert data["valid"] is False
    assert data["artifact_id"] is None
    validation = data["validation"]
    assert any("overlap" in error for error in validation["errors"])
    checks = {item["name"]: item["status"] for item in validation["checks"]}
    assert checks["composition_conflicts"] == "fail"
    assert validation["conflicts"][0]["kind"] == "duplicate-site"


def test_validate_and_preview_report_not_reached_events(service):
    plan = _create_plan(service, steps=[1, 9], max_new_tokens=4)
    preview = service.call(
        "preview_intervention_plan",
        {"plan_artifact_id": plan["artifact_id"], "source": SOURCE, "token_ids": TOKENS},
    )
    assert preview["ok"], preview
    data = preview["data"]
    assert data["validation"]["valid"] is True
    assert data["scheduled_events"] == 2
    assert len(data["firings"]) == 1
    assert data["not_reached_events"] == 1
    assert any("not reached" in warning for warning in data["warnings"])

    validated = service.call(
        "validate_intervention_plan",
        {"plan_artifact_id": plan["artifact_id"], "source": SOURCE},
    )
    assert validated["ok"], validated
    checks = {item["name"]: item["status"] for item in validated["data"]["checks"]}
    assert checks["schedule_reachable"] == "warning"
    assert validated["data"]["valid"] is True
    assert validated["data"]["resolved"][0]["module_class"] == "TinyBlock"


def test_run_temporal_reports_identity_agreement_and_exact_firings(service):
    plan = _create_plan(service)
    data = _run(service, plan["artifact_id"])
    record = _record(TemporalInterventionRun, data)
    assert record.validity.status == "valid"
    assert record.identity_control.agreement == "exact"
    assert record.metadata["events_fired"] == 2
    assert record.metadata["scheduled_events"] == 2
    assert data["summary"]["events_missing"] == 0
    assert record.metadata["one_time_interventions"] == []
    assert record.comparison.first_divergence == 2
    assert [item.condition for item in record.conditions] == [
        "baseline",
        "identity",
        "plan",
    ]
    assert data["event_summaries"]["plan"]["fired"] == 2
    assert data["event_summaries"]["identity"]["fired"] == 2
    assert data["event_summaries"]["baseline"]["fired"] == 0
    assert data["summary"]["events_fired"] == 2

    log_ids = data["event_log_artifact_ids"]
    assert set(log_ids) == {"baseline", "identity", "plan"}
    log = InterventionEventLog.model_validate(
        service.workspace.artifacts.read_json(log_ids["plan"])
    )
    assert [(event.generation_step, event.module) for event in log.events] == [
        (1, "blocks.1"),
        (2, "blocks.1"),
    ]
    assert log.all_fired

    inspected = service.call(
        "inspect_intervention_events",
        {"run_artifact_id": data["artifact_id"], "include_events": True},
    )
    assert inspected["ok"], inspected
    assert inspected["data"]["total_fired"] == 4  # plan plus identity control
    assert inspected["data"]["total_missing"] == 0

    counts_only = service.call(
        "inspect_intervention_events",
        {"event_log_artifact_id": log_ids["plan"], "include_events": False},
    )
    assert counts_only["ok"], counts_only
    assert counts_only["data"]["logs"][0]["fired"] == 2
    assert "events" not in counts_only["data"]["logs"][0]


def test_run_rejects_a_plan_for_a_different_prompt_length(service):
    plan = _create_plan(service)
    with pytest.raises(CompatibilityError):
        service.call(
            "run_temporal_intervention",
            {
                "plan_artifact_id": plan["artifact_id"],
                "source": SOURCE,
                "token_ids": TOKENS + [13],
            },
        )

    tolerant = ToolService(workspace=service.workspace, raise_errors=False)
    envelope = tolerant.call(
        "run_temporal_intervention",
        {
            "plan_artifact_id": plan["artifact_id"],
            "source": SOURCE,
            "token_ids": TOKENS + [13],
        },
    )
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "incompatible"


def test_plan_budget_limits_are_enforced(service):
    plan = _create_plan(service, steps=[0, 1, 2, 3], max_new_tokens=4)

    envelope = service.call(
        "validate_intervention_plan",
        {
            "plan_artifact_id": plan["artifact_id"],
            "budget": {"max_scheduled_events": 2},
        },
    )
    assert envelope["ok"], envelope
    assert envelope["data"]["valid"] is False
    checks = {item["name"]: item["status"] for item in envelope["data"]["checks"]}
    assert checks["scheduled_events_budget"] == "fail"
    assert any("more events than the budget" in item for item in envelope["data"]["errors"])

    base = _record(InterventionPlan, plan)
    # The record itself refuses a per-intervention event budget it violates, so
    # a plan cannot be constructed that only fails later at execution time.
    with pytest.raises(InputError):
        service.call(
            "validate_intervention_plan",
            {
                "plan": base.model_copy(
                    update={
                        "limits": base.limits.model_copy(
                            update={"max_scheduled_events": 2}
                        )
                    }
                ).model_dump(mode="json"),
            },
        )

    truncated_var = service.call(
        "validate_intervention_plan",
        {
            "plan": base.model_copy(
                update={
                    "limits": base.limits.model_copy(update={"max_events_per_step": 1})
                }
            ).model_dump(mode="json"),
        },
    )
    assert truncated_var["ok"], truncated_var
    assert truncated_var["data"]["valid"] is True


def test_composition_reports_a_descriptive_interaction(service):
    plan_a = _create_plan(service, ["blocks.1"], steps=[1])
    plan_b = _create_plan(service, ["blocks.2"], steps=[1])
    plan_ab = _create_plan(service, ["blocks.1", "blocks.2"], steps=[1])

    envelope = service.call(
        "run_compositional_comparison",
        {
            "plan_a_artifact_id": plan_a["artifact_id"],
            "plan_b_artifact_id": plan_b["artifact_id"],
            "plan_ab_artifact_id": plan_ab["artifact_id"],
            "source": SOURCE,
            "token_ids": TOKENS,
        },
    )
    assert envelope["ok"], envelope
    data = envelope["data"]
    record = _record(CompositionalComparison, data)
    assert record.validity.status == "valid"
    assert record.identity_control.agreement == "exact"
    assert [item.condition for item in record.conditions] == [
        "baseline",
        "identity",
        "A",
        "B",
        "AB",
    ]
    contrast = record.interactions[0]
    expected = (contrast.y_ab - contrast.y_baseline) - (
        contrast.y_a - contrast.y_baseline
    ) - (contrast.y_b - contrast.y_baseline)
    assert contrast.interaction == pytest.approx(expected, abs=1e-12)
    assert "does not establish a functional circuit" in record.metadata["note"]
    assert data["summary"]["validity"] == "valid"


def test_staged_prepare_and_verify_report_support_explicitly(service):
    prepared = service.call(
        "prepare_staged_generation", {"source": SOURCE, "token_ids": TOKENS}
    )
    assert prepared["ok"], prepared
    assert prepared["data"]["supported"] is True
    assert prepared["data"]["architecture_class"] == "full-attention-transformer"
    assert prepared["data"]["state_kind"]
    assert prepared["data"]["state_bytes"] > 0
    assert any("does not verify" in warning for warning in prepared["warnings"])

    plan = _create_plan(service, steps=[0, 2], execution_mode="staged-generation")
    verified = service.call(
        "verify_staged_generation",
        {"plan_artifact_id": plan["artifact_id"], "source": SOURCE, "token_ids": TOKENS},
    )
    assert verified["ok"], verified
    record = _record(StagedVerification, verified["data"])
    assert record.agreement in ("exact", "within_tolerance")
    assert record.token_sequence_match is True
    assert record.termination_match is True
    assert record.max_abs_logits <= record.atol + record.rtol

    staged = service.call(
        "run_staged_intervention",
        {"plan_artifact_id": plan["artifact_id"], "source": SOURCE, "token_ids": TOKENS},
    )
    assert staged["ok"], staged
    run = _record(StagedContinuation, staged["data"])
    assert run.validity.status == "valid"
    assert run.verification.agreement in ("exact", "within_tolerance")
    assert run.performance.speedup_ratio is not None


def test_attach_temporal_evidence_requires_a_temporal_record(service):
    with pytest.raises(InputError):
        service.call("attach_temporal_evidence", {"title": "empty"})


def test_attach_temporal_evidence_records_references(service):
    plan = _create_plan(service)
    data = _run(service, plan["artifact_id"])

    trace = service.call(
        "record_trace",
        {
            "input_text": "the loop begins here",
            "params": {"max_new_tokens": 8},
            "backend": {"kind": "synthetic", "synthetic": {"model": {}}},
        },
    )
    assert trace["ok"], trace
    case = service.call(
        "create_failure_case",
        {
            "title": "temporal loop",
            "observations": [trace["data"]["trace"]["trace_id"]],
        },
    )
    assert case["ok"], case
    case_id = case["data"]["case"]["case_id"]

    evidence = service.call(
        "attach_temporal_evidence",
        {
            "case_id": case_id,
            "title": "scheduled ablation",
            "plan_artifact_ids": [plan["artifact_id"]],
            "run_artifact_ids": [data["artifact_id"]],
            "event_log_artifact_ids": [data["event_log_artifact_ids"]["plan"]],
            "hypotheses": [{"statement": "blocks.1 matters at step 2"}],
        },
    )
    assert evidence["ok"], evidence
    assert evidence["data"]["attached_to_case"] is True

    stored = service.call(
        "inspect_investigation_evidence",
        {"artifact_id": evidence["data"]["evidence_artifact_id"]},
    )
    assert stored["ok"], stored
    assert stored["data"]["integrity"]["missing_artifacts"] == []
    assert [item["plan_id"] for item in stored["data"]["plans"]]
    assert stored["data"]["temporal_runs"][0]["events_fired"] == 2
    assert stored["data"]["hypotheses"][0]["statement"] == "blocks.1 matters at step 2"


def test_cli_temporal_workflow(tmp_path):
    runner = CliRunner()
    root = tmp_path / "cli-temporal"
    initialized = runner.invoke(
        app, ["--workspace", str(root), "init"], catch_exceptions=False
    )
    assert initialized.exit_code == 0, initialized.stdout

    created = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "create-intervention-plan",
            "--name",
            "cli scheduled ablation",
            "--tokens",
            ",".join(str(item) for item in TOKENS),
            "--execution-mode",
            "uncached-generation",
            "--max-new-tokens",
            "4",
            "--json",
            json.dumps(
                {
                    "name": "cli scheduled ablation",
                    "prompt_token_ids": TOKENS,
                    "max_new_tokens": 4,
                    "interventions": [
                        {
                            "intervention": _spec("blocks.1"),
                            "order": 0,
                            "schedule": [1, 2],
                            "position_mode": "last",
                        }
                    ],
                }
            ),
        ],
        catch_exceptions=False,
    )
    assert created.exit_code == 0, created.stdout
    plan_id = json.loads(created.stdout)["data"]["artifact_id"]

    previewed = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "preview-intervention-plan",
            "--plan",
            plan_id,
            "--source",
            json.dumps(SOURCE),
            "--tokens",
            ",".join(str(item) for item in TOKENS),
        ],
        catch_exceptions=False,
    )
    assert previewed.exit_code == 0, previewed.stdout
    assert json.loads(previewed.stdout)["data"]["scheduled_events"] == 2

    ran = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "run-temporal-intervention",
            "--plan",
            plan_id,
            "--source",
            json.dumps(SOURCE),
            "--tokens",
            ",".join(str(item) for item in TOKENS),
        ],
        catch_exceptions=False,
    )
    assert ran.exit_code == 0, ran.stdout
    payload = json.loads(ran.stdout)["data"]
    assert payload["validity"]["status"] == "valid"
    assert payload["summary"]["events_fired"] == 2

    inspected = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "inspect-intervention-events",
            payload["artifact_id"],
            "--run",
        ],
        catch_exceptions=False,
    )
    assert inspected.exit_code == 0, inspected.stdout
    assert json.loads(inspected.stdout)["data"]["total_fired"] == 4

    prepared = runner.invoke(
        app,
        [
            "--workspace",
            str(root),
            "prepare-staged-generation",
            "--source",
            json.dumps(SOURCE),
            "--tokens",
            ",".join(str(item) for item in TOKENS),
        ],
        catch_exceptions=False,
    )
    assert prepared.exit_code == 0, prepared.stdout
    assert json.loads(prepared.stdout)["data"]["supported"] is True

    # API and CLI agree on the parsed record, not just on exit status.
    service = ToolService(workspace=service_workspace(root), raise_errors=True)
    api = service.call(
        "run_temporal_intervention",
        {"plan_artifact_id": plan_id, "source": SOURCE, "token_ids": TOKENS, "save": False},
    )
    assert api["ok"], api
    assert {
        key: value for key, value in api["data"]["summary"].items() if key != "run_id"
    } == {
        key: value for key, value in payload["summary"].items() if key != "run_id"
    }


def service_workspace(root):
    from wsp.workspace.store import Workspace

    return Workspace(root, name="cli")
