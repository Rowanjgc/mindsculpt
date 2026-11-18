# Changelog

All notable changes to WSP Workbench are documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions; each
milestone matches a documented capability release (see `docs/roadmap.md`).

The workbench is deliberately scoped: every entry below adds bounded
instruments. Non-goals and deferred ideas live in `docs/roadmap.md` and are
not changelog entries until implemented.

## [Unreleased]

### Added

- GitHub Actions CI: CPU-only offline test suite on Python 3.11–3.13.
- `CHANGELOG.md` and `CONTRIBUTING.md`.
- Git tags for the historical releases (`v0.1.0` … `v0.5.0`).
- **MCP stdio adapter** (`wsp mcp`, optional `mcp` extra): the shared
  `ToolService` registry served over the Model Context Protocol as a thin
  transport with verbatim `wsp.tool/1.0` envelope passthrough, structured
  error results, schemas generated from the same pydantic input models, and
  workspace/limitation notes in tool descriptions. See
  [`docs/mcp.md`](docs/mcp.md).
- First (scripted, non-autonomous) agent evaluation of the tool surface
  through MCP on a real hybrid checkpoint. See
  [`docs/agent-evaluation.md`](docs/agent-evaluation.md).

### Fixed

- `inspect_model` on a directory path now fails with an actionable
  `InputError` ("pass the `.safetensors` weights file inside it") instead of a
  misleading pickle-support message.
- `wsp mcp` now accepts `--workspace/-w` and `--budget` directly on the
  subcommand (overriding the global options), validates the workspace before
  starting (structured error envelope on stderr instead of a raw traceback),
  and gains `--auto-init` to create and initialize a missing workspace at
  startup.
- Workspace requirement notes and hints no longer assume a shell: MCP tool
  descriptions and runtime error hints now point at starting `wsp mcp` with
  an initialized workspace or restarting with `--auto-init`, and two hints
  that referenced a nonexistent `wsp workspace init` command were corrected
  to `wsp init`.
- Capture skip warnings now include the observed tensor shape, making the
  reported failure concrete.
- Directory-path checkpoint errors now enumerate the `.safetensors` files the
  directory actually contains (message + `details.safetensors_files`), found
  by an autonomous agent that kept guessing checkpoint filenames.

## [0.5.0] — V1.5: event-driven investigation

### Added

- `wsp.behavioral-event/1.0`: deterministic events from token occurrences,
  repeated sequences, earliest repetitions, detector findings, manual marks,
  and text patterns, with absolute positions, prompt offsets, alignment
  status, and content checksums.
- `wsp.event-trigger/1.0`: online and retrospective triggers with delay,
  bounded window, firing mode, maximum count, and strict causal ordering
  (`k + 1 + delay`). Distinct observed/scheduled/fired/not-reached/skipped/
  rejected states; `wsp.event-intervention/1.0` records the life cycle.
- `wsp.fixed-context-replay/1.0` and `wsp.fixed-context-comparison/1.0`:
  exact prefix reconstruction, shared token digest, mandatory identity
  control, selected log-probabilities and logits, optional bounded
  continuation with divergence reporting.
- `wsp.investigation-recipe/1.0` and `wsp.recipe-run/1.0`: three built-in
  recipes with preview, prerequisites, cost estimates, and structured
  partial-failure records.
- `wsp.investigation-dossier/1.0` and operation discovery metadata with
  deterministic progressive disclosure via `discover_capabilities`.
- New artifact kinds registered with the single privacy exporter; new
  budgets for event lookback, trigger counts, fixed-context positions,
  recipe steps, passes, and artifacts.

## [0.4.0] — V1.4: temporal and compositional interventions

### Added

- `wsp.intervention-schedule/1.0`: first-prefix-forward, explicit generation
  steps, inclusive/half-open bounded ranges with stride. Declarative only;
  executable predicates are never accepted.
- One documented indexing convention shared by replay, capture,
  intervention, and generation; persisted in every plan.
- `wsp.intervention-plan/1.0`: one to thirty-two scheduled interventions with
  deterministic ordering, collision detection (`allow_overlap`), identity
  plans, and pre-execution cost estimates.
- `wsp.intervention-event-log/1.0`, `wsp.temporal-intervention/1.0`, and
  `wsp.compositional-comparison/1.0` with a descriptive interaction contrast.
- Verified staged generation: a staged decoder abstraction over the fixture
  cache and a local checkpoint's cached forward path, with
  `wsp.staged-verification/1.0` equivalence checking and explicit rejection
  of hybrid-recurrent architectures.
- Six new artifact kinds integrated into evidence, cases, export/import, and
  the redaction policy, fail-closed.

## [0.3.0] — V1.3: controlled causal interventions

### Added

- `wsp.activation-intervention/1.0`: versioned intervention specifications
  covering module, capture point, tensor role, position semantics, slice,
  operation, replacement provenance, dtype policy, and resource limits.
- Zero ablation, mean ablation with mandatory provenance, donor activation
  patching, and a mandatory identity negative control. Hooks modify only the
  selected region, clone before writing, never edit parameters, and clean up
  on success and failure.
- `wsp.counterfactual-comparison/1.0`: paired baseline/identity/intervention
  measurement with effect sizes; a failed identity control invalidates the
  experiment rather than being hidden.
- `wsp.continuation-comparison/1.0`: bounded, deterministic, uncached greedy
  continuations with longest common prefix and first divergence.
- Privacy hardening in the exporter: exact-match scanning of removed strings,
  substring scanning of sensitive markers, Unicode normalization and bounded
  base64 scanning, non-structural nested key redaction, and fail-or-omit
  behavior for untransformable artifacts.
- Nine structured operations and CLI commands sharing the existing service
  layer; offline end-to-end demonstration and optional real-checkpoint
  tests.
- Note: V1.2 content (teacher-forced replay, token alignment, targeted
  capture, investigation evidence) was developed and documented during the
  0.2.x series and ships in this release's lineage.

## [0.2.0] — V1.1: the Behavioral Observatory

### Added

- Versioned `wsp.trace/1.0` schema with token-level observations, termination
  reasons, timing, provenance, privacy policy, and explicit unavailable
  fields; immutable trace artifacts plus a local trace index.
- Deterministic scripted generation backend and an optional local Hugging
  Face generation backend.
- Detector registry with five deterministic detectors (repetition,
  termination, structured output, reconsideration markers, repeated tool
  actions) producing sparse findings with evidence ranges.
- Evaluator abstraction with pass/fail/inconclusive/abstain/error outcomes;
  contrastive tests with declared invariants.
- Persistent failure-case registry with review states, annotations,
  hypotheses, and artifact references; checksum-verified case export and
  reload.
- INIF 0.1 and Inspect AI log adapters without adding them as dependencies.
- V1.2 internal investigation (verified replay, token alignment, targeted
  capture, capture comparison, investigation evidence, and the optional
  NNsight/nnterp backend) was built on this baseline and documented before
  the next version bump.

## [0.1.0] — V1: foundation

### Added

- SafeTensors header-only checkpoint inspection, manifests, fingerprints,
  and comparison without loading a whole model.
- Tensor statistics, quantiles/histograms, norms, comparisons, rank/SVD
  summaries, with explicit sampled/approximate labels.
- Forward-hook activation capture with bounded retention, summary mode,
  token-position selection, and deterministic cleanup.
- Immutable versioned weight patches (replace/add/scale/rank-one/projection)
  with preview, disjoint patch sets, streaming export, and rollback; source
  checkpoints are never modified in place.
- Workspace with content-addressed artifacts, run records, and experiment
  grouping.
- Structured operation registry with JSON Schemas behind one service layer
  shared by the Python API and the `wsp` CLI; offline CPU test suite.

[Unreleased]: https://github.com/Rowanjgc/mindsculpt/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Rowanjgc/mindsculpt/commits/v0.5.0
[0.4.0]: https://github.com/Rowanjgc/mindsculpt/commits/v0.4.0
[0.3.0]: https://github.com/Rowanjgc/mindsculpt/commits/v0.3.0
[0.2.0]: https://github.com/Rowanjgc/mindsculpt/commits/v0.2.0
[0.1.0]: https://github.com/Rowanjgc/mindsculpt/commits/v0.1.0
