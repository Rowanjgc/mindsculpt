# Contributing to WSP Workbench

Thank you for considering a contribution. This project is a **tools-first
instrument kit** for weight-space programming: bounded, honest primitives for
observing, measuring, patching, and reproducing work on neural-network
weights and behavior. It measures; it does not conclude.

## Ground rules

These are the project's standing invariants. A change that violates any of
them will not be merged, regardless of other merits.

1. **Source checkpoints are never modified in place.** Patching exports a
   new checkpoint atomically and refuses dangerous overwrites.
2. **Measurements are separated from interpretations.** Sampled or
   approximate results are labeled as such and never presented as full
   results. No automatic causal claims, mechanism names, or dimension
   labels.
3. **No agent-only behavior.** Every operation must be reachable through the
   structured service layer so Python, the `wsp` CLI, and future transports
   call the same code.
4. **No network, no telemetry.** The default suite and examples are CPU-only
   and fully offline. Nothing downloads automatically and nothing phones
   home.
5. **New structured operations ship with JSON Schemas and bounded outputs.**
   A very large request must fail cleanly against `max_output_bytes` /
   resource budgets, not truncate silently.
6. **New file formats must be versioned and safe to parse.** Pickle-based
   loading is rejected; bundled code is never executed.
7. **New numerical behavior must report precision and sampling** (storage
   dtype, float64 analysis space, stride when sampled).
8. **New execution paths must preserve deterministic CPU tests** or
   explicitly mark themselves environment-dependent via test markers.

## Development setup

Requirements: Python 3.11+ and PyTorch. CPU-only is the supported baseline.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras exist (`hf` for local Hugging Face support, `instrumentation`
for the pinned NNsight/nnterp backend) but are never required for core work.

## Running the tests

```bash
python -m pytest -q
```

The default suite is offline and CPU-only. It includes unit tests for core,
patches, activations/execution, workspace/tools, and CLI behavior, plus
end-to-end integration tests.

Test markers you may need:

- `slow` — tests that take more than a few seconds.
- `hf` — requires the optional `transformers` dependency.
- `gpu` — requires a CUDA-capable device.
- `real_model` — requires `WSP_REAL_MODEL` to point at a **local**
  pretrained checkpoint. These tier-3 tests never download weights, and a
  skipped run means real-checkpoint compatibility is unverified, not
  passing:

  ```bash
  WSP_REAL_MODEL=<local checkpoint directory> \
      .venv/bin/python -m pytest tests/integration/test_real_checkpoint.py -q
  ```

## What to work on

`docs/roadmap.md` separates completed capabilities from hypotheses. Items
under "Possible future work" are not commitments — please open an issue or
discussion before building one, and do not implement deferred items merely
because they sound useful. Bug fixes, documentation improvements, and
hardening of existing primitives are always in scope.

## Expectations for a pull request

- **Scope:** one coherent change. Prefer small PRs.
- **Tests:** new behavior needs offline CPU tests; bug fixes need a test
  that fails before and passes after.
- **Docs:** user-visible changes update the relevant `docs/*.md` page, and
  any new limitation is recorded in `docs/limitations.md` honestly.
  Undocumented limitations are treated as bugs.
- **Changelog:** add an entry under `Unreleased` in `CHANGELOG.md`.
- **Commits:** short imperative subject lines (see `git log` for house
  style).
- **CI:** the offline test suite must pass on Python 3.11–3.13; the tier-3
  real-model tests may be skipped locally.

## Reporting issues

Include the Python and torch versions, the operating system, and a minimal
reproducer. For numerical surprises, include the declared tolerances and the
relevant artifact or schema record. Security-relevant reports (path
traversal, unsafe loading, privacy-export failures) should be described
carefully — the privacy exporter is designed to refuse rather than guess,
and any case where it silently ships removed content is high priority.

## License

By contributing, you agree that your contributions are licensed under the
Apache-2.0 license of this repository.
