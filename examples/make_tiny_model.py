"""Create a deterministic tiny MLP checkpoint for the offline examples.

The checkpoint is written under ``examples/out/`` by default.  The script
refuses to overwrite an existing file unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from safetensors.torch import save_file

from wsp.core.dtypes import torch_dtype_name
from wsp.execution.builtin_models import TinyMLP
from wsp.patches.safetensors_export import copy_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "tiny_mlp.safetensors",
        help="destination SafeTensors file",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true", help="overwrite an existing checkpoint")
    args = parser.parse_args()

    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.force:
        print(f"refusing to overwrite {destination}; pass --force to replace it", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    model = TinyMLP(input_dim=8, hidden_dim=16, output_dim=4)
    model.eval()
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    try:
        save_file(
            state,
            temporary,
            metadata={"format": "pt", "example": "wsp-tiny-mlp", "seed": str(args.seed)},
        )
        # Re-emit through the workbench exporter so the header metadata is
        # sorted and identical model weights always produce identical bytes.
        copy_checkpoint(temporary, destination, overwrite=destination.exists())
    finally:
        temporary.unlink(missing_ok=True)

    total = sum(tensor.numel() for tensor in state.values())
    print(f"wrote {destination}")
    print(f"dtype={torch_dtype_name(next(iter(state.values())))} tensors={len(state)} parameters={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
