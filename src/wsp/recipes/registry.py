"""The built-in recipe registry.

The registry is data, not an execution engine.  Recipes describe their steps,
inputs, costs, and limitations so a caller can inspect them before running.
"""

from __future__ import annotations

from wsp.recipes.models import RecipeDefinition, RecipeInput, RecipeStep

REPETITION_NOTE = (
    "Repetition is a measured token pattern. The recipe does not assume that repetition "
    "is pathological or that it demonstrates a reasoning loop."
)

_COMMON_PRIVACY = [
    "Inputs may contain prompt text, generated text, token identifiers, model paths, and user notes.",
    "Use the workspace exporter with a redacted policy before sharing artifacts.",
]

INVESTIGATE_FINDING = RecipeDefinition(
    recipe_id="investigate_behavioral_finding",
    version="1.0.0",
    title="Investigate a behavioral finding",
    summary="Resolve a detector finding to an exact token event, prepare a replay, optionally capture activations, and produce a dossier.",
    purpose=(
        "Turn an observed behavioral finding into a reproducible internal-investigation "
        "case without writing custom PyTorch code. The recipe records observations and "
        "measurements; it does not claim a mechanism."
    ),
    inputs=[
        RecipeInput(name="trace_id", type="string", description="Recorded trace to investigate."),
        RecipeInput(name="model", type="object", description="Replay model source (synthetic or local_hf) for optional capture."),
        RecipeInput(name="detector_id", type="string", required=False, default="core.repetition", description="Detector used to produce findings."),
        RecipeInput(name="detector_parameters", type="object", required=False, default={}, description="Detector parameters."),
        RecipeInput(name="finding_index", type="integer", required=False, default=0, description="Index into the flat finding list."),
        RecipeInput(name="capture", type="object", required=False, description="Capture request: modules, positions, mode."),
        RecipeInput(name="case_id", type="string", required=False, description="Attach evidence to this failure case."),
        RecipeInput(name="title", type="string", required=False, description="Dossier title."),
    ],
    steps=[
        RecipeStep(index=0, name="inspect_trace", operation="inspect_trace", description="Read the trace and confirm it exists.", produces=["trace summary"]),
        RecipeStep(index=1, name="scan", operation="scan_trace", description="Run the requested detector and select the finding.", produces=["findings"]),
        RecipeStep(index=2, name="resolve_event", operation="detect_behavioral_events", description="Resolve the finding to an exact absolute token position.", produces=["behavioral_event", "alignment"]),
        RecipeStep(index=3, name="prepare_replay", operation="prepare_fixed_context_replay", description="Reconstruct the exact recorded prefix.", produces=["fixed_context_replay"]),
        RecipeStep(index=4, name="capture", operation="capture_trace_activations", required=False, description="Optionally capture selected activations at the event position.", produces=["trace_capture"], forward_passes=1, est_cost="medium"),
        RecipeStep(index=5, name="dossier", operation="create_investigation_dossier", description="Summarise observations, measurements, controls, and limitations.", produces=["investigation_dossier"]),
        RecipeStep(index=6, name="attach", operation="attach_event_evidence", required=False, description="Attach the evidence to a failure case when one was supplied.", produces=["investigation_evidence"]),
    ],
    prerequisites=[
        "A trace with exact input and generated token identifiers.",
        "A replay model source if activation capture is requested.",
        "A workspace for event, replay, capture, and dossier artifacts.",
    ],
    limitations=[
        "A resolved finding is an observation, not a defect or a mechanism.",
        "Capture requires a model that can replay the trace token-for-token.",
        "The recipe does not run an intervention; use the hypothesis recipe for that.",
        REPETITION_NOTE,
    ],
    privacy_implications=_COMMON_PRIVACY,
    expected_artifacts=["behavioral_event", "fixed_context_replay", "trace_capture (optional)", "investigation_dossier", "investigation_evidence (optional)"],
    followups=["test_activation_hypothesis", "inspect_investigation_dossier", "discover_capabilities"],
)

TEST_HYPOTHESIS = RecipeDefinition(
    recipe_id="test_activation_hypothesis",
    version="1.0.0",
    title="Test an activation hypothesis",
    summary="Run baseline, identity, and requested intervention on one fixed token context and report a controlled numerical comparison.",
    purpose=(
        "Hold the token history fixed while changing one selected activation, then measure "
        "the downstream numerical effect. This is a controlled counterfactual, not a repair."
    ),
    inputs=[
        RecipeInput(name="trace_id", type="string", description="Trace whose recorded prefix is replayed."),
        RecipeInput(name="model", type="object", description="Replay model source."),
        RecipeInput(name="intervention", type="object", required=False, description="Inline activation intervention specification."),
        RecipeInput(name="intervention_artifact_id", type="string", required=False, description="Stored intervention specification."),
        RecipeInput(name="replay_step", type="integer", required=False, description="Explicit generation step to replay; defaults to the event's step."),
        RecipeInput(name="target_position", type="integer", required=False, description="Absolute consumed-token position targeted by the intervention."),
        RecipeInput(name="logit_positions", type="array", required=False, description="Positions whose next-token distributions are measured."),
        RecipeInput(name="measure_modules", type="array", required=False, description="Downstream modules to measure."),
        RecipeInput(name="case_id", type="string", required=False, description="Attach evidence to this failure case."),
    ],
    steps=[
        RecipeStep(index=0, name="validate", operation="validate_activation_intervention", description="Validate the intervention and its target.", produces=["validation"]),
        RecipeStep(index=1, name="prepare_replay", operation="prepare_fixed_context_replay", description="Reconstruct the exact token prefix.", produces=["fixed_context_replay"]),
        RecipeStep(index=2, name="compare", operation="run_fixed_context_comparison", description="Run baseline, identity, and intervention conditions.", produces=["fixed_context_comparison"], forward_passes=3, est_cost="medium"),
        RecipeStep(index=3, name="dossier", operation="create_investigation_dossier", description="Record measurements, controls, and limitations.", produces=["investigation_dossier"]),
        RecipeStep(index=4, name="attach", operation="attach_event_evidence", required=False, description="Attach evidence to a case when one was supplied.", produces=["investigation_evidence"]),
    ],
    prerequisites=[
        "A trace with exact token identifiers.",
        "An intervention whose selector targets the selected replay position.",
        "A replay model source with the same structure as the trace's model.",
    ],
    limitations=[
        "The identity control is mandatory; a failure invalidates attribution.",
        "The comparison establishes a numerical effect under the recorded conditions only.",
        "It does not establish semantic meaning, a circuit, or an equivalent weight edit.",
    ],
    privacy_implications=_COMMON_PRIVACY
    + ["Intervention specifications may carry donor activation references."],
    expected_artifacts=["fixed_context_replay", "fixed_context_comparison", "investigation_dossier", "investigation_evidence (optional)"],
    followups=["run_fixed_context_comparison", "list_investigation_recipes", "create_patch"],
)

INVESTIGATE_REPETITION = RecipeDefinition(
    recipe_id="investigate_repetition_event",
    version="1.0.0",
    title="Investigate a repetition event",
    summary="Detect the earliest configured repetition, reconstruct its context, run controlled measurements, and produce a portable dossier.",
    purpose=(
        "Measure a repetition event deterministically and test a selected activation "
        "hypothesis under a fixed token context."
    ),
    inputs=[
        RecipeInput(name="trace_id", type="string", description="Trace containing the repetition."),
        RecipeInput(name="model", type="object", description="Replay model source."),
        RecipeInput(name="ngram_size", type="integer", required=False, default=2, description="Repeated n-gram length."),
        RecipeInput(name="min_occurrences", type="integer", required=False, default=2, description="Occurrences required to report repetition."),
        RecipeInput(name="consecutive", type="boolean", required=False, default=False, description="Count only adjacent copies."),
        RecipeInput(name="lookback", type="integer", required=False, description="Bounded lookback in generated tokens."),
        RecipeInput(name="intervention", type="object", required=False, description="Inline activation intervention."),
        RecipeInput(name="intervention_artifact_id", type="string", required=False, description="Stored intervention specification."),
        RecipeInput(name="case_id", type="string", required=False, description="Attach evidence to a case."),
    ],
    steps=[
        RecipeStep(index=0, name="detect_repetition", operation="detect_behavioral_events", description="Find the earliest configured repetition in the trace.", produces=["behavioral_event"]),
        RecipeStep(index=1, name="prepare_replay", operation="prepare_fixed_context_replay", description="Reconstruct the exact prefix around the event.", produces=["fixed_context_replay"]),
        RecipeStep(index=2, name="compare", operation="run_fixed_context_comparison", description="Run baseline, identity, and intervention conditions.", produces=["fixed_context_comparison"], forward_passes=3, est_cost="medium"),
        RecipeStep(index=3, name="dossier", operation="create_investigation_dossier", description="Produce a portable investigation dossier.", produces=["investigation_dossier"]),
        RecipeStep(index=4, name="attach", operation="attach_event_evidence", required=False, description="Attach evidence to a case when one was supplied.", produces=["investigation_evidence"]),
    ],
    prerequisites=[
        "A trace with exact generated token identifiers.",
        "An intervention specification and a compatible replay model.",
        "A workspace for events, replays, comparisons, and dossiers.",
    ],
    limitations=[
        REPETITION_NOTE,
        "The measured repetition is a token-level pattern, not a semantic judgment.",
        "A measured effect does not establish that repetition is a failure or that the intervention repairs it.",
    ],
    privacy_implications=_COMMON_PRIVACY,
    expected_artifacts=["behavioral_event", "fixed_context_replay", "fixed_context_comparison", "investigation_dossier", "investigation_evidence (optional)"],
    followups=["test_activation_hypothesis", "inspect_investigation_dossier", "discover_capabilities"],
)

RECIPES: dict[str, RecipeDefinition] = {
    item.recipe_id: item
    for item in (INVESTIGATE_FINDING, TEST_HYPOTHESIS, INVESTIGATE_REPETITION)
}


def list_recipes(group: str | None = None) -> list[RecipeDefinition]:
    items = sorted(RECIPES.values(), key=lambda item: item.recipe_id)
    if group:
        items = [item for item in items if item.group == group]
    return items


def get_recipe(recipe_id: str) -> RecipeDefinition:
    from wsp.core.errors import NotFoundError

    try:
        return RECIPES[recipe_id]
    except KeyError as exc:
        raise NotFoundError(
            f"unknown investigation recipe {recipe_id!r}",
            details={"available": sorted(RECIPES)},
        ) from exc


__all__ = ["RECIPES", "get_recipe", "list_recipes"]
