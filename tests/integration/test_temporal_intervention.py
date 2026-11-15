"""Tier 2: the complete V1.4 temporal-intervention workflow, offline.

The scenarios here mirror the acceptance criteria: scheduled firing at a named
generation step, repeated schedules, multi-site plans, individual-versus-combined
comparisons, event accounting, reload, state preservation, and verified staged
generation.  Nothing in this file claims a mechanistic discovery; it checks that
the instruments read what they say they read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from support.temporal import make_plan, make_plan_intervention, make_spec
from wsp.core.errors import ExecutionError
from wsp.execution.staged import assert_clone_isolation
from wsp.execution.synthetic_lm import TinyKVCache
from wsp.interventions.engine import parameter_fingerprint
from wsp.interventions.plan import (
    InterventionPlan,
    identity_control_plan,
    plan_sha256,
    validate_plan,
)
from wsp.interventions.plan_engine import (
    mutable_state_fingerprint,
    preview_plan,
    run_temporal_intervention,
)
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace

PROMPT = "abc"
TOKENS = [1, 10, 11, 12]
SOURCE = {
    "kind": "synthetic",
    "synthetic": {
        "name": "tiny-causal-lm",
        "model": {"seed": 7, "model_dim": 32, "num_layers": 3, "max_seq_len": 96},
    },
}


def _spec(module: str, **kwargs):
    return make_spec(module, prompt_length=len(TOKENS), **kwargs)


def _plan_from(payload: dict) -> InterventionPlan:
    return InterventionPlan.model_validate(
        {key: value for key, value in payload.items() if key in InterventionPlan.model_fields}
    )


@dataclass
class World:
    root: Path
    workspace: Workspace
    service: ToolService
    plans: dict[str, dict]
    runs: dict[str, dict]
    composition: dict
    staged_plan: dict
    staged_run: dict
    parameters_before: str
    parameters_after: str


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> World:
    from support.tiny_lm import build_runtime

    runtime = build_runtime()
    with torch.no_grad():
        parameters_before = parameter_fingerprint(runtime.model)
        state_before = mutable_state_fingerprint(runtime.model)

    root = tmp_path_factory.mktemp("temporal-integration")
    workspace = Workspace(root / "ws", create=True, name="temporal")
    service = ToolService(workspace=workspace, raise_errors=True)

    def item(module, *, steps, positions=(0,), operation="zero",
             position_mode="last", order=0, label=None, **kwargs):
        spec = _spec(
            module, positions=positions, operation=operation,
            label=label or module, **kwargs
        )
        return {
            "intervention": spec.model_dump(mode="json"),
            "order": order,
            "schedule": steps,
            "position_mode": position_mode,
            "label": label or module,
        }

    def plan(name, items, *, execution_mode="uncached-generation", max_new_tokens=4):
        data = service.call(
            "create_intervention_plan",
            {
                "name": name,
                "prompt_token_ids": TOKENS,
                "execution_mode": execution_mode,
                "max_new_tokens": max_new_tokens,
                "interventions": items,
            },
        )["data"]
        assert data["valid"], (name, data["validation"])
        assert data["artifact_id"], name
        return data

    plans = {
        "step0": plan("step0", [item("blocks.1", steps=[0])]),
        "later": plan("later", [item("blocks.1", steps=[2])]),
        "repeated": plan("repeated", [item("blocks.1", steps=[0, 1, 2])]),
        "multi": plan(
            "multi",
            [
                item("blocks.0", steps=[1], order=0, label="site-a"),
                item("blocks.2", steps=[1], order=1, label="site-b"),
            ],
        ),
        "A": plan("A", [item("blocks.1", steps=[1])]),
        "B": plan("B", [item("blocks.2", steps=[1])]),
        "AB": plan(
            "AB",
            [
                item("blocks.1", steps=[1], order=0, label="A"),
                item("blocks.2", steps=[1], order=1, label="B"),
            ],
        ),
        "predictable": plan(
            "predictable",
            [
                item(
                    "norm",
                    steps=[0],
                    positions=[len(TOKENS) - 1],
                    position_mode="absolute",
                    label="norm-output-zero",
                )
            ],
        ),
    }

    def run(plan: str | None = None, *, artifact_id: str | None = None,
            identity: bool = True):
        payload = {
            "source": SOURCE,
            "token_ids": TOKENS,
            "identity": identity,
            "save": True,
        }
        if artifact_id is not None:
            payload["plan_artifact_id"] = artifact_id
        else:
            payload["plan"] = plan
        return service.call("run_temporal_intervention", payload)["data"]

    runs = {"baseline": run(artifact_id=plans["step0"]["artifact_id"])}
    identity_payload = identity_control_plan(_plan_from(plans["step0"])).model_dump(mode="json")
    runs["identity-standalone"] = run(identity_payload)
    for name in ("step0", "later", "repeated", "multi", "predictable"):
        runs[name] = run(artifact_id=plans[name]["artifact_id"])

    composition = service.call(
        "run_compositional_comparison",
        {
            "plan_a_artifact_id": plans["A"]["artifact_id"],
            "plan_b_artifact_id": plans["B"]["artifact_id"],
            "plan_ab_artifact_id": plans["AB"]["artifact_id"],
            "source": SOURCE,
            "token_ids": TOKENS,
            "metric": "mean_generated_logprob",
            "save": True,
        },
    )["data"]

    staged_plan = plan(
        "staged",
        [item("blocks.1", steps=[1, 2])],
        execution_mode="staged-generation",
    )
    staged_run = service.call(
        "run_staged_intervention",
        {
            "plan_artifact_id": staged_plan["artifact_id"],
            "source": SOURCE,
            "token_ids": TOKENS,
            "save": True,
        },
    )["data"]

    with torch.no_grad():
        parameters_after = parameter_fingerprint(runtime.model)
        state_after = mutable_state_fingerprint(runtime.model)
    assert state_before == state_after, "an experiment mutated persistent model state"

    return World(
        root=root,
        workspace=workspace,
        service=service,
        plans=plans,
        runs=runs,
        composition=composition,
        staged_plan=staged_plan,
        staged_run=staged_run,
        parameters_before=parameters_before,
        parameters_after=parameters_after,
    )


# --------------------------------------------------------------- 1. baseline


def test_baseline_generation_is_plain_greedy_uncached(world: World):
    run = world.runs["baseline"]
    assert run["summary"]["validity"] == "valid"
    assert [item["condition"] for item in run["conditions"]] == [
        "baseline",
        "identity",
        "plan",
    ]
    generated = run["conditions"][0]["generated_token_ids"]
    assert len(generated) == 4
    assert run["conditions"][0]["events_applied"] == 0
    assert run["identity_control"]["performed"] is True
    assert run["identity_control"]["agreement"] == "exact"


# ------------------------------------------------------ 2. identity condition


def test_identity_plan_reproduces_the_schedule_and_changes_nothing(world: World):
    run = world.runs["identity-standalone"]
    assert run["summary"]["validity"] == "valid"
    assert run["plan_id"].startswith("pl_")
    assert run["plan_id"] != world.plans["step0"]["plan_id"]
    tokens = [item["generated_token_ids"] for item in run["conditions"]]
    baseline_tokens = world.runs["baseline"]["conditions"][0]["generated_token_ids"]
    assert tokens == [baseline_tokens, baseline_tokens, baseline_tokens]
    condition = run["conditions"][2]
    assert condition["events_identity"] == 1
    assert condition["events_applied"] == 0
    log = run["event_summaries"]["plan"]
    assert log["fired"] == 1
    assert log["rejected"] == 0
    assert log["by_operation"] == {"identity": 1}


def test_identity_plan_attaches_the_same_hook_sites(world: World):
    baseline_plan = _plan_from(world.plans["step0"])
    identity_plan = identity_control_plan(baseline_plan)
    assert identity_plan.module_paths() == baseline_plan.module_paths()
    assert [item.order for item in identity_plan.interventions] == [
        item.order for item in baseline_plan.interventions
    ]
    assert [item.schedule.resolve() for item in identity_plan.interventions] == [
        item.schedule.resolve() for item in baseline_plan.interventions
    ]
    assert identity_plan.plan_id != baseline_plan.plan_id


# ------------------------------------------------------- 3/4/5. step targets


def test_step_zero_intervention_fires_exactly_once(world: World):
    run = world.runs["step0"]
    assert run["summary"]["validity"] == "valid"
    assert run["identity_control"]["agreement"] == "exact"
    log = run["event_summaries"]["plan"]
    assert log["fired"] == 1
    assert log["missing"] == 0
    assert log["not_reached"] == 0
    assert log["by_step"] == {"0": 1}
    condition = run["conditions"][-1]
    assert condition["condition"] == "plan"
    assert condition["events_applied"] == 1
    assert condition["events_missing"] == 0


def test_later_step_intervention_fires_at_the_named_step(world: World):
    run = world.runs["later"]
    log = run["event_summaries"]["plan"]
    assert log["by_step"] == {"2": 1}
    assert len(run["metadata"]["one_time_interventions"]) == 1
    assert run["metadata"]["one_time_interventions"][0].startswith("pi_")
    assert run["metadata"]["sustained_interventions"] == []


def test_one_time_and_sustained_schedules_are_distinct(world: World):
    one_time = world.runs["step0"]
    sustained = world.runs["repeated"]
    assert len(one_time["metadata"]["one_time_interventions"]) == 1
    assert one_time["metadata"]["one_time_interventions"][0].startswith("pi_")
    assert sustained["metadata"]["one_time_interventions"] == []
    assert sustained["event_summaries"]["plan"]["by_step"] == {"0": 1, "1": 1, "2": 1}
    assert sustained["event_summaries"]["plan"]["fired"] == 3

    # The generated tokens may coincide even when the schedules differ: once a
    # changed token enters the prefix, later recomputations can reproduce the
    # same continuation.  Equality of text is therefore NOT evidence that the
    # first intervention persisted.  The event records are the evidence.
    assert one_time["event_summaries"]["plan"]["fired"] == 1
    assert sustained["event_summaries"]["plan"]["fired"] == 3
    assert (
        one_time["event_log_artifact_ids"]["plan"]
        != sustained["event_log_artifact_ids"]["plan"]
    )
    assert one_time["metadata"]["uncached_note"]


def test_event_log_records_that_a_one_time_intervention_is_not_persistent(world: World):
    run = world.runs["step0"]
    log = run["event_summaries"]["plan"]
    assert log["fired"] == 1
    assert log["steps_executed"] == 4
    # The later forwards recompute the prefix without the intervention, so the
    # event log must not claim any event for steps 1..3.
    assert set(log["by_step"]) == {"0"}


# ------------------------------------------------------------ 6. multi-site


def test_two_modules_fire_in_the_same_forward(world: World):
    run = world.runs["multi"]
    assert run["summary"]["validity"] == "valid"
    log = run["event_summaries"]["plan"]
    assert log["fired"] == 2
    assert log["by_step"] == {"1": 2}
    assert set(log["by_module"]) == {"blocks.0", "blocks.2"}
    assert run["conditions"][-1]["events_applied"] == 2


def test_multi_site_plan_preview_reports_both_sites(world: World):
    from support.tiny_lm import build_runtime

    runtime = build_runtime()
    plan = _plan_from(world.plans["multi"])
    preview = preview_plan(plan, runtime.replay_model(), TOKENS)
    assert len(preview.firings) == 2
    assert {firing.module for firing in preview.firings} == {"blocks.0", "blocks.2"}
    assert all(firing.generation_step == 1 for firing in preview.firings)


# --------------------------------------- 7. mathematically predictable effect


def test_zeroing_the_final_residual_leaves_only_the_documented_repeat_bias(world: World):
    """A hand-checkable prediction.

    Zeroing the normalised residual at the last prompt position makes every
    logit at that position zero, because the fixture head has no bias.  The only
    surviving term is the fixture's documented ``repeat_bias`` (0.25), which is
    added to the token consumed one position earlier.  The predicted token and
    its log-probability are therefore computable without running the model at
    all: ``logprob = 0.25 - log((V - 1) + exp(0.25))``.
    """

    import math

    from support.tiny_lm import build_runtime

    runtime = build_runtime()
    plan = _plan_from(world.plans["predictable"])
    outcome = run_temporal_intervention(
        plan, runtime.replay_model(), TOKENS, identity=False, store=None
    )
    row = outcome.record.conditions[-1]
    expected_token = TOKENS[-2]
    assert row.generated_token_ids[0] == expected_token

    vocabulary = runtime.tokenizer.vocab_size
    expected_logprob = 0.25 - math.log((vocabulary - 1) + math.exp(0.25))
    assert row.logprobs[0] == pytest.approx(expected_logprob, abs=1e-5)


def test_the_intervention_is_applied_at_the_prediction_position_only(world: World):
    from support.tiny_lm import build_runtime

    runtime = build_runtime()
    plan = _plan_from(world.plans["predictable"])
    outcome = run_temporal_intervention(
        plan, runtime.replay_model(), TOKENS, identity=False, store=None
    )
    events = outcome.event_logs["plan"].events
    assert len(events) == 1
    event = events[0]
    assert event.generation_step == 0
    assert event.rows == [len(TOKENS) - 1]
    assert event.tensor_shape == [1, len(TOKENS), 32]
    assert event.elements_modified == 32
    assert event.after.l2_norm == pytest.approx(0.0, abs=1e-6)
    assert event.before.l2_norm > 0.0


# ------------------------------------ 8/9. individual and combined conditions


def test_individual_and_combined_conditions_share_the_control(world: World):
    record = world.composition
    assert [item["condition"] for item in record["conditions"]] == [
        "baseline",
        "identity",
        "A",
        "B",
        "AB",
    ]
    assert record["validity"]["status"] == "valid"
    assert record["identity_control"]["agreement"] == "exact"
    tokens = {
        item["condition"]: item["generated_token_ids"] for item in record["conditions"]
    }
    assert tokens["baseline"] == tokens["identity"]
    logs = record["event_log_artifact_ids"]
    assert set(logs) >= {"baseline", "identity", "A", "B", "AB"}
    for name in ("A", "B", "AB"):
        log = world.workspace.artifacts.read_json(logs[name])
        assert log["missing_events"] == []
        assert len(log["events"]) == len(log["expected_events"])
        assert {event["generation_step"] for event in log["events"]} == {1}
    # A greedy argmax can survive a real logit change; the numerical difference,
    # not the token string, is the measurement.
    pairwise = {(item["condition"], item["reference"]): item for item in record["pairwise"]}
    for name in ("A", "B", "AB"):
        effect = pairwise[(name, "baseline")]
        assert effect["logits"]["max_abs"] > 0.0
        assert effect["logit_l2_vs_reference"] > 0.0


def test_interaction_contrast_matches_the_documented_formula(world: World):
    record = world.composition
    interaction = record["interactions"][0]
    assert interaction["metric"] == "mean_generated_logprob"
    y0 = interaction["y_baseline"]
    ya = interaction["y_a"]
    yb = interaction["y_b"]
    yab = interaction["y_ab"]
    expected = (yab - y0) - (ya - y0) - (yb - y0)
    assert interaction["interaction"] == pytest.approx(expected, abs=1e-9)
    assert interaction["continuous"] is True
    assert "does not establish a functional circuit" in record["metadata"]["note"]


# --------------------------------------------- 10/11. downstream measurement


def test_downstream_measurements_are_recorded(world: World):
    run = world.runs["step0"]
    record = run
    assert record["comparison"]["first_divergence"] == 0
    assert record["stepwise_logits"], "stepwise logits must be recorded"
    step = record["stepwise_logits"][0]
    assert step["step"] == 0
    assert step["token_matches"] is False
    assert step["max_abs"] > 0.0

    plan_condition = record["conditions"][-1]
    assert plan_condition["logprobs"]
    assert plan_condition["top1_token_ids"]
    assert plan_condition["termination_reason"] in {
        "eos",
        "max_new_tokens",
        "max_sequence_length",
    }


def test_event_records_carry_numerical_context(world: World):
    from support.tiny_lm import build_runtime

    runtime = build_runtime()
    plan = _plan_from(world.plans["later"])
    outcome = run_temporal_intervention(plan, runtime.replay_model(), TOKENS, store=None)
    log = outcome.event_logs["plan"]
    event = log.events[0]
    assert event.generation_step == 2
    assert event.forward_invocation == 0
    assert event.tensor_shape == [1, 6, 32]
    assert event.rows == [5]
    assert event.token_indices == [5]
    assert event.dtype == "float32"
    assert event.before.l2_norm > 0
    assert event.after.l2_norm == pytest.approx(0.0, abs=1e-6)
    expected_peak = max(abs(event.before.minimum), abs(event.before.maximum))
    assert event.max_abs == pytest.approx(expected_peak, rel=0.0, abs=1e-6)
    assert event.execution_result == "applied"
    assert event.replacement_sha256.startswith("sha256:")
    assert log.all_fired


def test_missing_events_are_reported_not_silently_skipped():
    from support.tiny_lm import build_runtime

    runtime = build_runtime()
    spec = make_spec("blocks.1", prompt_length=len(TOKENS), positions=(0,))
    plan = make_plan(
        [
            make_plan_intervention(
                spec,
                schedule=[0],
                position_mode="last",
                on_step_call_indices=[1],
            )
        ],
        prompt_length=len(TOKENS),
        max_new_tokens=2,
        seed="missing",
    )
    outcome = run_temporal_intervention(
        plan, runtime.replay_model(), TOKENS, store=None
    )
    assert outcome.record.validity.status == "invalid"
    assert outcome.event_logs["plan"].missing_events
    assert any(
        "did not fire" in reason for reason in outcome.record.validity.reasons
    )


# --------------------------------------------------- 12. artifact round trip


def test_plan_artifact_round_trip_preserves_the_checksum(world: World):
    data = world.service.call(
        "validate_intervention_plan",
        {"plan_artifact_id": world.plans["step0"]["artifact_id"]},
    )["data"]
    assert data["valid"] is True
    assert data["plan_id"] == world.plans["step0"]["plan_id"]

    stored = world.workspace.artifacts.read_json(world.plans["step0"]["artifact_id"])
    plan = InterventionPlan.model_validate(stored)
    assert plan.checksums["plan_sha256"] == plan_sha256(plan)
    assert stored["checksums"]["plan_sha256"] == plan_sha256(plan)


def test_tampered_plan_artifact_is_rejected(world: World):
    from wsp.core.errors import ArtifactError

    stored = world.workspace.artifacts.read_json(world.plans["step0"]["artifact_id"])
    tampered = dict(stored)
    tampered["name"] = "tampered"
    ref = world.workspace.artifacts.put_json(
        "intervention_plan", tampered, metadata={"tampered": True}
    )
    with pytest.raises(ArtifactError):
        world.service.call(
            "validate_intervention_plan", {"plan_artifact_id": ref.artifact_id}
        )


def test_plan_rebuild_is_deterministic(world: World):
    first = _plan_from(world.plans["step0"])
    second = InterventionPlan.model_validate(first.model_dump(mode="json"))
    assert plan_sha256(first) == plan_sha256(second)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_artifact_ids_are_content_addressed_and_resolvable(world: World):
    for name, run in world.runs.items():
        for key, artifact_id in run["event_log_artifact_ids"].items():
            ref, path = world.workspace.artifacts.get(artifact_id)
            assert ref.kind == "intervention_event_log", (name, key)
            assert path.exists()


# ------------------------------------------------------- 13. state protection


def test_parameters_and_mutable_state_are_unchanged(world: World):
    assert world.parameters_before == world.parameters_after
    from support.tiny_lm import build_runtime

    runtime = build_runtime()
    before = mutable_state_fingerprint(runtime.model)
    plan = _plan_from(world.plans["repeated"])
    run_temporal_intervention(plan, runtime.replay_model(), TOKENS, store=None)
    assert mutable_state_fingerprint(runtime.model) == before


def test_a_parameter_mutation_inside_a_hook_is_rejected():
    from support.tiny_lm import build_runtime

    runtime = build_runtime()
    model = runtime.model
    weight = model.blocks[1].mlp[0].weight
    saved = weight.detach().clone()

    plan = make_plan(
        [
            make_plan_intervention(
                make_spec("blocks.1", prompt_length=len(TOKENS), positions=(0,)),
                schedule=[0],
                position_mode="last",
            )
        ],
        prompt_length=len(TOKENS),
        max_new_tokens=2,
        seed="corrupt",
    )

    original = plan.interventions[0].intervention

    def _mutating_hook(module, args, output):
        with torch.no_grad():
            weight.add_(1.0)
        return output

    from wsp.interventions import plan_engine

    real_install = plan_engine.PlanSession.install

    def _install(self):
        real_install(self)
        handle = model.blocks[1].register_forward_hook(_mutating_hook)
        self._handles.append(handle)

    plan_engine.PlanSession.install = _install
    try:
        with pytest.raises(ExecutionError):
            run_temporal_intervention(
                plan, runtime.replay_model(), TOKENS, store=None
            )
    finally:
        plan_engine.PlanSession.install = real_install
        with torch.no_grad():
            weight.copy_(saved)
    assert parameter_fingerprint(model) == _fingerprint_of(runtime, saved)


def _fingerprint_of(runtime, saved) -> str:
    model = runtime.model
    weight = model.blocks[1].mlp[0].weight
    with torch.no_grad():
        original = weight.detach().clone()
        weight.copy_(saved)
        value = parameter_fingerprint(model)
        weight.copy_(original)
    return value


# --------------------------------------------- 14/15. staged generation paths


def test_staged_verification_agrees_with_the_uncached_reference(world: World):
    verification = world.staged_run["verification"]
    assert verification["agreement"] in {"exact", "within_tolerance"}
    assert verification["token_sequence_match"] is True
    assert verification["termination_match"] is True
    assert verification["identity_control_agreement"] in {"exact", "within_tolerance"}
    assert verification["max_abs_logits"] <= verification["atol"] + verification["rtol"] * max(
        1.0, verification["max_abs_logits"]
    )
    named = {check["name"]: check["status"] for check in verification["checks"]}
    assert named["token_sequence"] == "pass"
    assert named["stepwise_logits"] == "pass"
    assert named["termination"] == "pass"
    assert named["identity_control"] == "pass"


def test_staged_run_reports_scheduled_events_and_performance(world: World):
    run = world.staged_run
    assert run["validity"]["status"] == "valid"
    assert run["execution_mode"] == "staged-generation"
    logs = run["event_log_artifact_ids"]
    assert set(logs) >= {"baseline", "identity", "plan"}
    log = world.workspace.artifacts.read_json(logs["plan"])
    assert log["missing_events"] == []
    assert len(log["events"]) == 2
    assert sorted(event["generation_step"] for event in log["events"]) == [1, 2]
    performance = run["performance"]
    assert performance["prefill_ms"] is not None
    assert performance["total_ms"] is not None
    assert performance["cache_state_bytes"] > 0
    assert "not model capability benchmarks" in performance["note"]


def test_staged_and_uncached_agree_on_the_unmodified_baseline(world: World):
    staged_tokens = world.staged_run["conditions"][0]["generated_token_ids"]
    uncached_tokens = world.runs["baseline"]["conditions"][0]["generated_token_ids"]
    assert staged_tokens == uncached_tokens


def test_shared_cache_state_is_detected():
    from support.tiny_lm import build_runtime
    from wsp.execution.staged import build_staged_decoder

    runtime = build_runtime()
    decoder = build_staged_decoder(runtime.replay_model())
    _logits, state = decoder.prefill(TOKENS)
    before = decoder.state_hash(state)

    clone = decoder.clone(state)
    assert clone is not state
    decoder.step(clone, 11)
    assert clone.length == state.length + 1
    assert_clone_isolation(decoder, state, before, action="deep-clone branch")

    # A deliberate shallow copy shares the list objects and the tensor storage.
    # Continuing that branch must be caught rather than silently contaminating
    # the reference state.
    shallow = TinyKVCache(
        keys=state.keys, values=state.values, token_ids=state.token_ids
    )
    decoder.step(shallow, 11)
    with pytest.raises(ExecutionError) as excinfo:
        assert_clone_isolation(decoder, state, before, action="shallow branch")
    assert "shares mutable storage" in str(excinfo.value)


def test_unsupported_hybrid_architectures_fail_explicitly():
    from wsp.core.errors import UnsupportedError
    from wsp.execution.staged import build_staged_decoder

    class _StandIn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(
                    model_type="qwen3_5",
                    layer_types=["full_attention", "linear_attention"],
                )
            )

        def forward(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("an unsupported architecture must not be executed")

    with pytest.raises(UnsupportedError):
        build_staged_decoder(_StandIn())


from types import SimpleNamespace  # noqa: E402  (kept beside its only use)
