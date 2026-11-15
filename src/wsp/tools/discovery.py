"""Deterministic capability discovery for humans and language models.

Discovery is metadata lookup and keyword scoring, not hidden reasoning.  A
caller can ask a broad question, receive a small set of relevant operations and
recipes, then request the full schema for one operation.
"""

from __future__ import annotations

from typing import Any

from wsp.core.errors import InputError

GROUPS = [
    "inspect",
    "observe",
    "compare",
    "replay",
    "capture",
    "intervene",
    "evaluate",
    "investigate",
    "export",
]

GROUP_DESCRIPTIONS = {
    "inspect": "Open checkpoints, list tensors, and read bounded numerical summaries.",
    "observe": "Record and inspect traces, run detectors, and manage failure cases.",
    "compare": "Compare traces, captures, checkpoints, and numerical measurements.",
    "replay": "Reconstruct exact token prefixes and replay recorded generations.",
    "capture": "Measure selected internal activations at selected token positions.",
    "intervene": "Apply reversible activation interventions, schedules, and event triggers.",
    "evaluate": "Run evaluators and contrasts over generated behavior.",
    "investigate": "Compose evidence, dossiers, recipes, and case attachments.",
    "export": "Export and import artifacts and cases under a privacy policy.",
}

_INSPECT_PREFIXES = (
    "inspect_",
    "list_",
    "read_",
    "workspace_",
)
_OBSERVE_PREFIXES = ("record_", "import_trace", "traces")
_COMPARE_PREFIXES = ("compare_",)
_REPLAY_PREFIXES = ("replay", "prepare_replay", "align_", "resolve_span")
_CAPTURE_PREFIXES = ("capture_",)
_INTERVENE_PREFIXES = (
    "create_activation_intervention",
    "validate_activation_intervention",
    "preview_activation_intervention",
    "run_activation_intervention",
    "run_paired_intervention",
    "run_counterfactual",
    "inspect_counterfactual",
    "create_intervention_plan",
    "validate_intervention_plan",
    "preview_intervention_plan",
    "run_temporal_intervention",
    "inspect_intervention_events",
    "run_compositional_comparison",
    "prepare_staged_generation",
    "verify_staged_generation",
    "run_staged_intervention",
    "create_event_trigger",
    "run_event_triggered_intervention",
    "run_fixed_context_comparison",
)
_EVALUATE_PREFIXES = ("evaluate_", "run_contrast", "list_evaluators")
_INVESTIGATE_PREFIXES = (
    "attach_",
    "create_failure_case",
    "inspect_failure_case",
    "update_failure_case",
    "list_failure_cases",
    "create_investigation",
    "inspect_investigation",
    "list_investigation",
    "preview_investigation",
    "run_investigation",
    "inspect_recipe",
    "discover_capabilities",
    "detect_behavioral",
    "resolve_event",
    "prepare_fixed_context",
)
_EXPORT_PREFIXES = ("export_", "import_case", "list_artifacts")


_EXPLICIT_GROUPS = {
    "record_trace": "observe",
    "import_trace": "observe",
    "list_traces": "observe",
    "inspect_trace": "observe",
    "scan_trace": "observe",
    "compare_traces": "compare",
    "list_detectors": "observe",
    "list_evaluators": "evaluate",
    "evaluate_behavior": "evaluate",
    "run_contrast": "evaluate",
    "create_failure_case": "observe",
    "inspect_failure_case": "observe",
    "list_failure_cases": "observe",
    "update_failure_case": "observe",
    "export_failure_case": "export",
    "import_failure_case": "export",
    "record_contrast": "evaluate",
    "attach_investigation_evidence": "investigate",
    "inspect_investigation_evidence": "investigate",
    "compare_activation_captures": "compare",
    "list_capture_points": "capture",
    "inspect_activation_capture": "capture",
    "capture_trace_activations": "capture",
    "list_artifacts": "inspect",
    "workspace_status": "inspect",
    "inspect_model": "inspect",
    "list_tensors": "inspect",
    "inspect_tensor": "inspect",
    "read_tensor_slice": "inspect",
    "analyze_tensor": "inspect",
    "inspect_tensor": "inspect",
    "compare_tensors": "compare",
    "compare_checkpoints": "compare",
    "compare_forward": "compare",
    "run_forward": "observe",
    "list_patches": "inspect",
    "export_checkpoint": "export",
    "prepare_replay": "replay",
    "verify_replay": "replay",
    "replay_trace": "replay",
    "align_tokens": "replay",
    "resolve_span": "replay",
}


def group_for(name: str) -> str:
    explicit = _EXPLICIT_GROUPS.get(name)
    if explicit:
        return explicit
    for prefix in _INTERVENE_PREFIXES:
        if name == prefix or name.startswith(prefix):
            return "intervene"
    for prefix in _EVALUATE_PREFIXES:
        if name == prefix or name.startswith(prefix):
            return "evaluate"
    for prefix in _INVESTIGATE_PREFIXES:
        if name == prefix or name.startswith(prefix):
            return "investigate"
    for prefix in _EXPORT_PREFIXES:
        if name == prefix or name.startswith(prefix):
            return "export"
    for prefix in _CAPTURE_PREFIXES:
        if name.startswith(prefix):
            return "capture"
    for prefix in _REPLAY_PREFIXES:
        if name == prefix or name.startswith(prefix):
            return "replay"
    for prefix in _OBSERVE_PREFIXES:
        if name == prefix or name.startswith(prefix):
            return "observe"
    for prefix in _COMPARE_PREFIXES:
        if name.startswith(prefix):
            return "compare"
    for prefix in _INSPECT_PREFIXES:
        if name.startswith(prefix):
            return "inspect"
    return "investigate"


_CURATED: dict[str, dict[str, Any]] = {
    "discover_capabilities": {
        "purpose": "Discover relevant operations and recipes for a question without reading the whole registry.",
        "followups": ["list_investigation_recipes", "preview_investigation_recipe"],
        "examples": [{"query": "behavioral repetition", "detail": "compact"}],
    },
    "inspect_trace": {
        "purpose": "Read one recorded generation trace, its token identifiers, and its input.",
        "artifact_kinds": [],
        "privacy": ["The trace may contain prompt and generated text."],
    },
    "scan_trace": {
        "purpose": "Run behavioral detectors over a recorded trace and return measured findings.",
        "followups": ["detect_behavioral_events", "create_failure_case"],
        "limitations": ["A detector finding is a measurement, not a diagnosis."],
    },
    "detect_behavioral_events": {
        "purpose": "Turn a token pattern, detector finding, text match, or manual mark into an exact behavioral event.",
        "artifact_kinds": ["behavioral_event"],
        "followups": ["prepare_fixed_context_replay", "create_event_trigger"],
        "limitations": [
            "An event is an observation; it is not a defect or a mechanism.",
            "Repetition events are measured token patterns, not diagnoses.",
            "Ambiguous text alignment is reported instead of inventing a position.",
        ],
        "examples": [
            {
                "trace_id": "tr_...",
                "event_definition": {"kind": "first_repetition", "ngram_size": 2, "min_occurrences": 2},
            }
        ],
    },
    "create_event_trigger": {
        "purpose": "Declare when an observed event may schedule an intervention, with explicit delay and budgets.",
        "artifact_kinds": ["event_trigger"],
        "followups": ["run_event_triggered_intervention"],
        "limitations": ["Online triggers cannot act before the event token is generated."],
    },
    "run_event_triggered_intervention": {
        "purpose": "Run an online event-triggered intervention or a retrospective fixed-context investigation.",
        "artifact_kinds": ["event_intervention", "fixed_context_replay", "fixed_context_comparison"],
        "backends": ["torch_hooks"],
        "cost_class": "medium",
        "followups": ["inspect_investigation_dossier", "attach_event_evidence"],
        "limitations": [
            "An event with no subsequent forward is reported as not reached.",
            "Retrospective runs are labelled and are not online predictions.",
        ],
    },
    "prepare_fixed_context_replay": {
        "purpose": "Reconstruct the exact recorded token prefix for a fixed-context experiment.",
        "artifact_kinds": ["fixed_context_replay"],
        "prerequisites": ["A trace with recorded token identifiers."],
        "followups": ["run_fixed_context_comparison"],
        "examples": [{"trace_id": "tr_...", "event_artifact_ids": ["a_..."]}],
    },
    "run_fixed_context_comparison": {
        "purpose": "Run baseline, identity control, and one activation intervention on identical token identifiers.",
        "artifact_kinds": ["fixed_context_comparison"],
        "prerequisites": [
            "A fixed-context replay artifact.",
            "An activation intervention targeting the replay position.",
            "A replay model source matching the trace.",
        ],
        "followups": ["create_investigation_dossier", "attach_event_evidence"],
        "limitations": [
            "A numerical effect is not a mechanism, a repair, or a weight edit.",
            "A failed identity control invalidates attribution.",
        ],
        "examples": [
            {
                "replay_artifact_id": "a_...",
                "intervention_artifact_id": "a_...",
                "source": {"kind": "synthetic"},
            }
        ],
    },
    "capture_trace_activations": {
        "purpose": "Replay a trace and measure selected internal activation statistics at selected token positions.",
        "artifact_kinds": ["trace_capture"],
        "prerequisites": ["A trace with exact token identifiers.", "A replay model source."],
        "cost_class": "medium",
        "backends": ["torch_hooks", "nnterp"],
    },
    "run_paired_intervention": {
        "purpose": "Measure baseline, identity, and one activation intervention on a teacher-forced token sequence.",
        "artifact_kinds": ["counterfactual_comparison", "intervention_result"],
        "cost_class": "medium",
        "limitations": ["A paired difference is not a causal explanation of behavior."],
    },
    "run_temporal_intervention": {
        "purpose": "Execute a versioned multi-site intervention plan over scheduled generation steps.",
        "artifact_kinds": ["temporal_intervention", "intervention_event_log"],
        "cost_class": "medium",
        "followups": ["inspect_intervention_events", "run_compositional_comparison"],
    },
    "create_investigation_dossier": {
        "purpose": "Build a concise agent-readable dossier separating observations, measurements, controls, hypotheses, and limitations.",
        "artifact_kinds": ["investigation_dossier"],
        "followups": ["inspect_investigation_dossier", "attach_event_evidence"],
        "limitations": ["A dossier summarises evidence; it does not add conclusions."],
    },
    "attach_event_evidence": {
        "purpose": "Attach events, fixed-context comparisons, recipe runs, and dossiers to a failure case.",
        "artifact_kinds": ["investigation_evidence"],
        "followups": ["inspect_failure_case", "export_failure_case"],
    },
    "list_investigation_recipes": {
        "purpose": "List high-level investigation recipes with their inputs, steps, and limitations.",
        "followups": ["preview_investigation_recipe", "run_investigation_recipe"],
    },
    "preview_investigation_recipe": {
        "purpose": "Inspect a recipe's planned steps, prerequisites, costs, artifacts, and privacy implications without running it.",
        "followups": ["run_investigation_recipe"],
    },
    "run_investigation_recipe": {
        "purpose": "Run a bounded multi-step investigation composed from existing validated operations.",
        "artifact_kinds": ["recipe_run", "investigation_dossier", "behavioral_event", "fixed_context_replay", "fixed_context_comparison"],
        "cost_class": "medium",
        "followups": ["inspect_recipe_run", "inspect_investigation_dossier"],
        "limitations": ["A recipe failure preserves partial artifacts and is never reported as completed."],
    },
    "export_failure_case": {
        "purpose": "Export a failure case and its referenced artifacts under a private or redacted policy.",
        "artifact_kinds": ["case_export"],
        "followups": ["import_failure_case"],
        "limitations": ["Redacted exports transform or drop sensitive artifacts; unknown kinds fail closed."],
    },
}

_ARTIFACT_HINTS = {
    "trace": ("record_trace", "inspect_trace", "scan_trace", "compare_traces", "export_trace", "import_trace"),
    "capture": ("capture_", "inspect_activation_capture", "compare_activation_captures"),
    "replay": ("replay", "prepare_replay", "inspect_replay"),
    "intervention": ("intervention", "plan", "temporal", "composition", "staged"),
    "case": ("failure_case",),
    "evidence": ("evidence",),
}


def _artifact_kinds(name: str) -> list[str]:
    kinds: list[str] = []
    for kind, prefixes in _ARTIFACT_HINTS.items():
        if any(name.startswith(prefix) or prefix in name for prefix in prefixes):
            kinds.append(kind)
    return kinds


def metadata_for(definition: Any) -> dict[str, Any]:
    """Return complete discovery metadata for one registered tool."""

    name = definition.name
    curated = _CURATED.get(name, {})
    group = definition.group or group_for(name)
    description = definition.description.strip()
    purpose = definition.purpose or curated.get("purpose") or description
    prerequisites = list(definition.prerequisites) or list(
        curated.get("prerequisites", [])
    )
    if definition.requires_workspace and not prerequisites:
        prerequisites = ["An initialized workspace."]
    cost_class = definition.cost_class
    if cost_class == "low" and curated.get("cost_class"):
        cost_class = curated["cost_class"]
    if cost_class == "low" and any(
        token in name
        for token in ("capture", "replay", "intervention", "temporal", "staged", "recipe", "composition")
    ):
        cost_class = "medium"
    backends = list(definition.backends) or list(curated.get("backends", ["torch_hooks"]))
    if not definition.backends and not curated.get("backends"):
        backends = ["torch_hooks"] if any(
            token in name for token in ("capture", "intervention", "replay", "event", "fixed")
        ) else []
    side_effects = list(definition.side_effects) or (
        ["Writes a workspace artifact."] if definition.requires_workspace else []
    )
    privacy = list(definition.privacy) or list(curated.get("privacy", []))
    if not privacy and ("trace" in name or "case" in name or "dossier" in name or "evidence" in name):
        privacy = ["May reference prompt text, generated text, token ids, and local paths."]
    limitations = list(definition.limitations) or list(curated.get("limitations", []))
    if definition.requires_workspace and not limitations:
        limitations = ["Requires a workspace and bounded inputs."]
    return {
        "group": group,
        "purpose": purpose,
        "prerequisites": prerequisites,
        "backends": backends,
        "artifact_kinds": list(definition.artifact_kinds)
        or list(curated.get("artifact_kinds", []))
        or _artifact_kinds(name),
        "cost_class": cost_class,
        "side_effects": side_effects,
        "privacy": privacy,
        "limitations": limitations,
        "followups": list(definition.followups) or list(curated.get("followups", [])),
        "examples": [dict(item) for item in definition.examples]
        or [dict(item) for item in curated.get("examples", [])],
    }


def capability_overview(
    service: Any,
    *,
    query: str | None = None,
    group: str | None = None,
    detail: str = "compact",
    include_recipes: bool = True,
    max_operations: int = 12,
) -> dict[str, Any]:
    """Return a bounded, deterministic capability overview."""

    from wsp.recipes.registry import list_recipes

    registry = service.describe(detail="compact")
    entries = registry["operations"]
    if group:
        if group not in GROUPS:
            raise InputError(
                "unknown capability group",
                details={"group": group, "available": GROUPS},
            )
        entries = [item for item in entries if item["group"] == group]
    terms = [term for term in (query or "").lower().split() if term]
    if terms:
        scored: list[tuple[int, dict[str, Any]]] = []
        for entry in entries:
            haystack = " ".join(
                [
                    entry["name"],
                    entry["description"],
                    entry.get("purpose") or "",
                    entry["group"],
                    " ".join(entry.get("artifact_kinds") or []),
                    " ".join(entry.get("limitations") or []),
                ]
            ).lower()
            score = sum(3 if term in entry["name"].lower() else 1 for term in terms if term in haystack)
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1]["name"]))
        selected = [entry for _score, entry in scored[:max_operations]]
    elif group:
        selected = entries[:max_operations]
    else:
        # Progressive disclosure: show groups and a small, useful starting set.
        selected = [
            entry
            for entry in entries
            if entry["name"]
            in {
                "discover_capabilities",
                "list_traces",
                "scan_trace",
                "detect_behavioral_events",
                "prepare_fixed_context_replay",
                "run_fixed_context_comparison",
                "list_investigation_recipes",
                "run_investigation_recipe",
                "create_investigation_dossier",
                "export_failure_case",
            }
        ]
        if not selected:
            selected = entries[:max_operations]

    recipes = []
    if include_recipes:
        recipe_items = list_recipes()
        if terms:
            matched = [
                recipe
                for recipe in recipe_items
                if any(
                    term
                    in (
                        recipe.recipe_id
                        + " "
                        + recipe.title
                        + " "
                        + recipe.summary
                        + " "
                        + recipe.purpose
                    ).lower()
                    for term in terms
                )
            ]
            if matched:
                recipe_items = matched
        recipes = [
            {
                "recipe_id": recipe.recipe_id,
                "title": recipe.title,
                "summary": recipe.summary,
                "group": recipe.group,
                "steps": len(recipe.steps),
                "required_inputs": [item.name for item in recipe.inputs if item.required],
                "preview_operation": "preview_investigation_recipe",
                "run_operation": "run_investigation_recipe",
            }
            for recipe in recipe_items
        ]
    return {
        "schema_version": "wsp.capabilities/1.0",
        "query": query,
        "group": group,
        "detail": detail,
        "groups": [
            {
                "name": name,
                "description": GROUP_DESCRIPTIONS[name],
                "operations": sum(1 for item in registry["operations"] if item["group"] == name),
            }
            for name in GROUPS
        ],
        "operations": selected,
        "recipes": recipes,
        "next_operations": [
            {"operation": "preview_investigation_recipe", "reason": "inspect a recipe before running it"},
            {"operation": "inspect_trace", "reason": "read the recorded behavior"},
            {"operation": "detect_behavioral_events", "reason": "resolve an exact token position"},
        ],
        "total_operations": len(registry["operations"]),
        "note": (
            "This overview is deterministic metadata filtering. It is not an autonomous "
            "research plan, and it makes no claim about scientific conclusions."
        ),
    }


__all__ = [
    "GROUPS",
    "GROUP_DESCRIPTIONS",
    "capability_overview",
    "group_for",
    "metadata_for",
]
