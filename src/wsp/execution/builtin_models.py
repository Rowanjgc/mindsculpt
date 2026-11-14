"""Small deterministic models used by offline examples, tests, and the CLI.

These are ordinary ``torch.nn.Module`` objects.  They are not special research
models; they exist so every operation can be demonstrated without a network
download or a Hugging Face checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from wsp.core.budget import Budget
from wsp.core.errors import InputError
from wsp.execution.loaders import load_state_dict_into_module
from wsp.tools.schemas import BuiltinModelSpec


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TinyLinear(nn.Module):
    def __init__(self, input_dim: int = 8, output_dim: int = 4) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class TinyTransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ff1 = nn.Linear(model_dim, model_dim * 2)
        self.act = nn.GELU()
        self.ff2 = nn.Linear(model_dim * 2, model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _weights = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attended)
        return self.ln2(x + self.ff2(self.act(self.ff1(x))))


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32,
        model_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 16,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(model_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise InputError("transformer input_ids must be a 2-D [batch, sequence] tensor")
        if input_ids.shape[1] > self.max_seq_len:
            raise InputError(
                "input sequence is longer than the model's max_seq_len",
                details={"sequence": int(input_ids.shape[1]), "max_seq_len": self.max_seq_len},
            )
        hidden = self.embed(input_ids.long())
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.norm(hidden))


def build_builtin_model(
    spec: BuiltinModelSpec,
    *,
    budget: Budget | None = None,
) -> nn.Module:
    """Construct a deterministic built-in model, optionally loading local weights."""

    torch.manual_seed(int(spec.seed))
    np.random.seed(int(spec.seed))
    if spec.kind == "linear":
        model: nn.Module = TinyLinear(spec.input_dim, spec.output_dim)
    elif spec.kind == "mlp":
        model = TinyMLP(spec.input_dim, spec.hidden_dim, spec.output_dim)
    elif spec.kind == "transformer":
        model = TinyTransformer(
            vocab_size=spec.vocab_size,
            model_dim=spec.model_dim,
            num_heads=spec.num_heads,
            num_layers=spec.num_layers,
            max_seq_len=spec.max_seq_len,
        )
    else:  # pragma: no cover - pydantic restricts the literal
        raise InputError(f"unknown built-in model kind: {spec.kind}")
    if spec.checkpoint_path is not None:
        path = Path(spec.checkpoint_path).expanduser()
        if not path.is_file():
            raise InputError(
                f"checkpoint does not exist: {path}",
                details={"path": str(path)},
            )
        load_state_dict_into_module(
            model,
            path,
            budget=budget or Budget(),
            strict=True,
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _convert(value: Any, *, integer: bool) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        array = np.asarray(value)
        if integer or np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_:
            tensor = torch.as_tensor(array.astype(np.int64))
        else:
            tensor = torch.as_tensor(array.astype(np.float32))
    if integer and tensor.dtype != torch.int64:
        tensor = tensor.long()
    return tensor


def prepare_builtin_inputs(
    spec: BuiltinModelSpec,
    inputs: dict[str, Any] | None,
    input_args: Iterable[Any] = (),
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    kwargs: dict[str, torch.Tensor] = {}
    for key, value in (inputs or {}).items():
        kwargs[key] = _convert(value, integer=spec.kind == "transformer" and key == "input_ids")
    args = tuple(_convert(value, integer=False) for value in input_args)
    if not kwargs and not args:
        raise InputError("a forward operation requires inputs or input_args")
    if spec.kind == "transformer":
        if "input_ids" not in kwargs:
            raise InputError("transformer forward requires an 'input_ids' keyword input")
        ids = kwargs["input_ids"]
        if ids.dtype != torch.int64:
            raise InputError("input_ids must contain integer token ids")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= spec.vocab_size):
            raise InputError(
                "input_ids contain values outside the model vocabulary",
                details={"vocab_size": spec.vocab_size},
            )
    return kwargs, args


def example_inputs(spec: BuiltinModelSpec) -> dict[str, Any]:
    if spec.kind == "transformer":
        return {"input_ids": [[1, 2, 3, 4]]}
    return {"x": [[0.1 * (index + 1) for index in range(spec.input_dim)]]}
