# Offline examples

These examples require no network access, no Hugging Face download, and no LLM.
They use deterministic tiny models defined in the workbench itself (a `TinyMLP`
for checkpoint mechanics and a small causal transformer for behavioral,
replay, and intervention work).

## 1. Create the tiny checkpoint

```bash
python examples/make_tiny_model.py
```

Writes `examples/out/tiny_mlp.safetensors` (4 tensors, 212 parameters). The
script refuses to overwrite an existing file unless `--force` is passed.

## 2. Inspect, capture, patch, compare, and roll back

```bash
python examples/inspect_and_patch.py
```

This demonstrates:

- header-only checkpoint inspection and a versioned manifest;
- bounded tensor statistics;
- a baseline activation capture on `fc1`;
- creation and preview of an immutable `add` patch;
- application to `examples/out/tiny_mlp.patched.safetensors`;
- exact verification of the patched `fc1.bias` values;
- output and activation comparison against the baseline;
- rollback by selecting the untouched source;
- a workspace under `examples/out/workspace` containing the manifest, patch,
  patch set, and run records.

The source checkpoint’s SHA-256 is checked before and after. It is never
modified. Re-running with `--force` is allowed; existing workspace records are
treated idempotently when their semantic content is unchanged.

## 3. Behavior traces, detectors, contrast, and cases

```bash
python examples/behavioral_workflow.py
```

This creates a fresh temporary workspace under `examples/out/` and performs the
V1.1 Behavioral Observatory workflow using synthetic scripted generations only:

- records a healthy trace and a repetitive trace;
- scans the repetitive trace for repeated n-grams and reconsideration markers;
- compares the two traces structurally;
- runs a contrastive test with an explicit factual invariant;
- creates a case referencing a synthetic activation-capture artifact;
- exports the case to a checksum-verified bundle;
- reloads it into a fresh workspace and re-scans the restored trace.

The script prints the finding kinds, contrast status, case id, and reload
results. It is an infrastructure demonstration, not a scientific claim about a
real model.

## 4. Replay, alignment, capture, and investigation evidence

```bash
python examples/investigation_walkthrough.py
```

This runs the V1.2 chain against a small causal transformer built in process:

- records a looping generation, a terminating control, and a third trace with a
  genuinely different input;
- runs the detectors and resolves the finding onto token positions;
- prepares and verifies a teacher-forced replay, reporting numerical agreement
  and the explicit `generation_reproducibility: not-claimed`;
- aligns tokens to prompt/generation/full positions;
- captures selected activations at a chosen module;
- compares an identical-context control (which must be zero) and a
  different-input pair (which must not be);
- attaches the measurements to a failure case as investigation evidence;
- exports the case privately, imports it into a fresh workspace, and re-verifies
  every reference and checksum;
- confirms the model parameters did not change;
- repeats the export in redacted mode and checks that the source prompt is gone.

Nothing is downloaded. Pass `--model <local checkpoint>` to run the same chain
against a real Hugging Face checkpoint instead of the synthetic model.

The script is an infrastructure demonstration. It does not claim to find a real
model defect or a causal mechanism.

## 5. Causal intervention and continuation

```bash
python examples/causal_investigation.py
```

This runs the V1.3 chain end-to-end against the tiny deterministic transformer,
offline:

- records an observation and a control trace;
- prepares and verifies a teacher-forced replay;
- captures selected activations and creates an identity intervention;
- verifies baseline and identity agreement, then creates a zero ablation;
- runs a paired measurement with downstream module differences;
- captures a donor activation and applies a patch;
- runs a bounded greedy continuation and reports the first divergence;
- attaches the experiment to a failure case and exports it privately and in
  redacted mode;
- reloads both bundles into fresh workspaces, verifies checksums and references,
  and confirms the model parameters never changed.

Pass `--workspace <path>` to keep the generated workspace. The script prints
`[1]`–`[19]` stage markers. It demonstrates infrastructure correctness, not a
scientific discovery.

## 6. Same patch through the CLI

```bash
wsp --workspace examples/out/workspace call create_patch \
  --input examples/create_patch.json
```

The response prints a `patch_id` such as `p_1c71f988491ebc01`. Use it in:

```bash
wsp --workspace examples/out/workspace call preview_patch \
  --json '{"path":"examples/out/tiny_mlp.safetensors","patch_id":"p_..."}'

wsp --workspace examples/out/workspace call apply_patch \
  --json '{"path":"examples/out/tiny_mlp.safetensors","output_path":"examples/out/cli_patched.safetensors","patch_ids":["p_..."]}'
```

## Cleanup

All generated files live under `examples/out/`, which is excluded from Git.
Delete that directory to start over:

```bash
rm -rf examples/out
```
