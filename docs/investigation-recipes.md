# Investigation recipes

V1.5 exposes three high-level recipes. They are declarative, versioned,
previewable, resource-bounded, and implemented by composing existing validated
operations. They are not a workflow engine and they do not execute arbitrary
Python.

## Registry

`wsp.investigation-recipe/1.0` describes:

- recipe id and version;
- purpose and summary;
- typed inputs, required or optional;
- ordered steps, each naming a structured operation, whether it is required,
  its expected artifacts, and a rough cost class;
- prerequisites, limitations, privacy implications, expected artifacts, and
  follow-up recipes.

List them with `list_investigation_recipes`, inspect one with
`preview_investigation_recipe`, run it with `run_investigation_recipe`, and read
the result with `inspect_recipe_run`.

## Recipe A: `investigate_behavioral_finding`

Inputs include a trace, a detector and its parameters, a finding index, an
optional replay model, optional capture modules, and an optional case id.

Steps:

1. inspect the trace;
2. run the detector and select a finding;
3. resolve the finding to an exact behavioral event;
4. reconstruct the exact prefix;
5. optionally capture activations at the event position;
6. create a dossier;
7. attach evidence to a case when one was supplied.

The recipe does not claim a mechanism. It records what was observed and where.

## Recipe B: `test_activation_hypothesis`

Inputs include a trace, a replay model source, an intervention or intervention
artifact, an optional target position, measurement positions/modules, and an
optional case id.

Steps:

1. validate the intervention;
2. reconstruct the fixed-context replay at the selected position;
3. run baseline, identity, and intervention;
4. create a dossier;
5. attach evidence when a case was supplied.

If the identity control fails, the dossier and comparison say so, and the
validity is `invalid`.

## Recipe C: `investigate_repetition_event`

Inputs include a trace, repetition settings, a replay model, an intervention,
and a case id.

Steps:

1. detect the earliest configured repetition;
2. reconstruct the prefix around it;
3. run the fixed-context comparison;
4. create a dossier;
5. attach evidence when a case was supplied.

The recipe explicitly does not assume that repetition is pathological.

## Preview before execution

`preview_investigation_recipe` returns the required inputs, provided inputs,
missing inputs, missing prerequisites, planned steps and operations, estimated
forward passes and cost class, expected artifacts, privacy implications, and
whether the recipe is currently executable. Previewing performs no model work.

## Failure handling

A required step failure stops the recipe. The run record:

- keeps every artifact produced by completed steps;
- records the failing step, operation, error code, and message;
- reports status `failed` (never `completed`);
- omits the dossier id if no dossier was created;
- carries enough provenance to repeat the attempt safely.

Optional steps such as capture or case attachment may be skipped with a warning.
A failed recipe is a structured partial record, not a scientific report.
