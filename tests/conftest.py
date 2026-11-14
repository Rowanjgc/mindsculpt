from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from wsp.execution.builtin_models import TinyMLP, TinyTransformer
from wsp.workspace.store import Workspace


def save_model(path: Path, model: torch.nn.Module) -> Path:
    state = {
        name: value.detach().to("cpu").contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(state, str(path))
    return path


@pytest.fixture
def mlp() -> TinyMLP:
    torch.manual_seed(1234)
    model = TinyMLP(input_dim=8, hidden_dim=16, output_dim=4)
    model.eval()
    return model


@pytest.fixture
def transformer() -> TinyTransformer:
    torch.manual_seed(77)
    model = TinyTransformer(
        vocab_size=32, model_dim=16, num_heads=4, num_layers=2, max_seq_len=12
    )
    model.eval()
    return model


@pytest.fixture
def mlp_checkpoint(tmp_path: Path, mlp: TinyMLP) -> Path:
    return save_model(tmp_path / "tiny-mlp.safetensors", mlp)


@pytest.fixture
def transformer_checkpoint(tmp_path: Path, transformer: TinyTransformer) -> Path:
    return save_model(tmp_path / "tiny-transformer.safetensors", transformer)


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "workspace", create=True, name="test")


@pytest.fixture
def input_x() -> list[list[float]]:
    return [[0.1 * (index + 1) for index in range(8)]]
