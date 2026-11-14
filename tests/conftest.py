from __future__ import annotations

import sys
from pathlib import Path

# Test helpers live in ``tests/support`` and are imported by both test tiers.
sys.path.insert(0, str(Path(__file__).parent))

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


@pytest.fixture
def service(workspace: Workspace):
    from wsp.tools.service import ToolService

    return ToolService(workspace=workspace, raise_errors=True)


@pytest.fixture
def make_trace(service):
    def _make(
        text: str,
        *,
        input_text: str | None = None,
        tokens: list[str] | None = None,
        token_ids: list[int] | None = None,
        logprobs: list[float] | None = None,
        termination: str = "eos",
        termination_detail: str | None = None,
        tool_calls: list[dict] | None = None,
        params: dict | None = None,
        privacy: dict | None = None,
        seed: int | None = None,
    ) -> str:
        fixture: dict = {
            "output_text": text,
            "termination": termination,
            "termination_detail": termination_detail,
        }
        if tokens is not None:
            fixture["output_tokens"] = [
                {"text": token, "token_id": (token_ids or [None] * len(tokens))[index]}
                for index, token in enumerate(tokens)
            ]
            if logprobs is not None:
                fixture["token_logprobs"] = logprobs
        if tool_calls is not None:
            fixture["tool_calls"] = tool_calls
        payload = {
            "input_text": input_text or "prompt",
            "backend": {
                "kind": "scripted",
                "scripted": {"default_fixture": fixture},
            },
            "params": params or {"max_new_tokens": 64},
            "seed": seed,
        }
        if privacy is not None:
            payload["privacy"] = privacy
        return service.call("record_trace", payload)["data"]["trace_id"]

    return _make


@pytest.fixture
def trace_store(workspace: Workspace):
    from wsp.behavior.trace import TraceStore

    return TraceStore(workspace)


@pytest.fixture
def tiny_lm():
    """A deterministic causal transformer with a real tokenizer (no download)."""

    from support.tiny_lm import build_runtime

    return build_runtime()


@pytest.fixture
def tiny_trace(tiny_lm):
    def _make(prompt: str, *, max_new_tokens: int = 8):
        from support.tiny_lm import make_trace

        return make_trace(tiny_lm, prompt, max_new_tokens=max_new_tokens)

    return _make
