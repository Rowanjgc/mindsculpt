# Offline examples

These examples require no network access, no Hugging Face download, and no LLM.
They use a deterministic tiny `TinyMLP` defined in the workbench itself.

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

## 3. Same patch through the CLI

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
