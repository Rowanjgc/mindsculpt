# Event-driven investigations

An event is a measurement of something observable in a generation stream. It is
not a diagnosis. A repeated token, a completed n-gram, a detector finding, or a
position marked by an investigator all become the same kind of record:
`wsp.behavioral-event/1.0`.

The point of the schema is to make the step from "I saw something" to "I know
which forward to investigate" explicit, reproducible, and auditable.

## Timing vocabulary

- **Prompt length `P`** — the number of input token identifiers presented to the
  model.
- **Generation step `k`** — the forward that consumes the prompt plus `k`
  generated tokens and predicts generated token `k`.
- **Token position** — an absolute index into the full sequence (prompt first).
- **Consumed position** — the absolute position whose activation the forward
  consumes. For the forward that predicts generated token `k`, this is
  `P + k - 1`.
- **Earliest online step** — normally `k + 1`. A token observed after step `k`
  cannot affect the forward that already produced it.

These values are stored on every event. A reader does not need to re-derive
them from prose.

## Event types

| Type | Meaning |
| --- | --- |
| `token_occurrence` | A configured token id or exact token-id sequence was observed. |
| `repeated_token_sequence` | A configured n-gram repetition condition became true. |
| `first_repetition` | The earliest position at which that condition became true. |
| `detector_finding` | A recorded detector finding was resolved against its trace. |
| `manual` | A human or external model marked a generated position or span. |
| `text_pattern` | A literal text pattern was found, with token-boundary ambiguity reported. |

Text matching is deliberately conservative. If both ends of the match do not
coincide with token boundaries, the event keeps the enclosing span but reports
`alignment.status = "ambiguous"`. It is never silently converted into an exact
activation target.

Detector findings use the V1.2 token alignment convention: the detector reports
indices within the generated tokens, and the event adds `prompt_length` to
produce absolute positions. The applied offset is recorded in the event.

## Event fields

A `wsp.behavioral-event/1.0` record contains:

- an event id and creation time;
- source trace id and artifact reference;
- event type and the declarative trigger definition;
- observation mode (`online` or `reconstructed`);
- absolute token position or span;
- generation step and the alignment record (`exact`, `ambiguous`, or
  `unavailable`);
- prompt/full lengths;
- evidence pointers, provenance, limitations, metadata, and a content checksum.

The schema contains no `fault`, `loop`, `failure`, or `repair` field. Those are
investigator hypotheses, not observations.

## Online execution

An online trigger is a deterministic state machine:

1. Generate a token normally.
2. Evaluate the trigger against tokens that have actually been generated.
3. If an event is observed after step `k`, schedule its earliest eligible
   forward at `k + 1 + delay_steps`.
4. Apply the intervention only when that forward is reached.
5. If generation ends first, report the event as `not_reached` and set
   `no_subsequent_forward = true`.

States are separate: `observed`, `scheduled`, `fired`, `not_reached`, `skipped`,
and `rejected`. A run that observed events but fired none is reported as
`limited`, not as a successful intervention.

The online executor uses the same `ActivationIntervention` operations and the
same reversible hook machinery as the V1.4 plan engine. It does not introduce a
second patching path.

## Retrospective execution

A retrospective trigger names an event and an explicit replay step from an
existing trace. The step may precede the event, because the experiment is a
deliberate reconstruction. Every retrospective record carries
`mode = "retrospective"` and a note stating that it is not an online prediction.

Retrospective investigation normally uses fixed-context replay, described in
[`fixed-context-counterfactuals.md`](fixed-context-counterfactuals.md).

## Trigger record

`wsp.event-trigger/1.0` is declarative:

- mode: `online` or `retrospective`;
- event definition;
- delay after observation;
- bounded window of subsequent steps;
- `once` or `every_occurrence` with an explicit maximum trigger count;
- position mode (`absolute` or `last`);
- notes and metadata.

Validation rejects:

- a retrospective trigger with no event or replay step;
- a delay on a retrospective trigger;
- `once` with a maximum greater than one;
- unbounded repetition or trigger counts above the configured budget;
- unsupported modes and event kinds.

## Limitations

- The event detector measures token patterns. It does not judge whether the
  pattern is meaningful.
- An exact position identifies a forward, not a mechanism.
- An online trigger cannot use future tokens; no implementation may pretend
  otherwise.
- An event that has no subsequent forward cannot produce an online intervention.
- A numerical effect from a triggered intervention is not a repair.
