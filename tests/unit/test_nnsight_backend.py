"""Optional NNsight/nnterp instrumentation backend.

These tests are skipped unless the optional dependencies and a transformers
build with the Qwen3.5 hybrid architecture are both available.  They never
download weights: the tiny hybrid checkpoint is constructed locally.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from wsp.core.budget import Budget
from wsp.core.errors import CompatibilityError, InputError, UnsupportedError
from wsp.tools.service import ToolService
from wsp.workspace.store import Workspace

pytest.importorskip("nnsight")
nnterp = pytest.importorskip("nnterp")
transformers = pytest.importorskip("transformers")

if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip(
        "this transformers build does not expose the Qwen3.5 hybrid architecture",
        allow_module_level=True,
    )

from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from wsp.investigation import nnsight_backend  # noqa: E402

WORDS = (
    "the cat sat on mat loop and that stop hello world a is it was there no yes dog ran"
).split()


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
    for index, word in enumerate(WORDS, start=4):
        vocab[word] = index
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:"
    )
    return tokenizer


@pytest.fixture(scope="module")
def hybrid_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a tiny Qwen3.5 checkpoint with alternating attention layer types."""

    target = tmp_path_factory.mktemp("qwen3_5_hybrid") / "checkpoint"
    target.mkdir()
    config = Qwen3_5TextConfig(
        vocab_size=len(WORDS) + 4,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
    )
    torch.manual_seed(11)
    model = Qwen3_5ForCausalLM(config)
    model.eval()
    model.save_pretrained(target)
    _tokenizer().save_pretrained(target)
    return target


@pytest.fixture
def service(tmp_path: Path) -> ToolService:
    workspace = Workspace(tmp_path / "workspace", create=True, name="nnterp")
    return ToolService(workspace=workspace, raise_errors=True)


@pytest.fixture
def trace_id(service: ToolService, hybrid_checkpoint: Path) -> str:
    envelope = service.call(
        "record_trace",
        {
            "input_text": "the cat sat on the mat",
            "params": {"max_new_tokens": 3},
            "backend": {
                "kind": "huggingface",
                "huggingface": {"model_path": str(hybrid_checkpoint)},
            },
        },
    )
    assert envelope["ok"], envelope
    return envelope["data"]["trace"]["trace_id"]


def _source(checkpoint: Path) -> dict:
    return {"kind": "local_hf", "path": str(checkpoint)}


def test_optional_versions_report_pinned_pair():
    versions = nnsight_backend.optional_versions()

    assert versions["available"] is True
    assert versions["pinned"] == {
        "nnsight": nnsight_backend.PINNED_NNSIGHT,
        "nnterp": nnsight_backend.PINNED_NNTERP,
    }
    assert "nnsight==" in versions["install_hint"]


def test_capture_points_expose_hybrid_layer_types(service, trace_id, hybrid_checkpoint):
    envelope = service.call(
        "list_capture_points",
        {"model": _source(hybrid_checkpoint), "instrumentation": "nnterp"},
    )
    assert envelope["ok"], envelope
    data = envelope["data"]
    points = {point["path"]: point for point in data["points"]}

    assert points["layers.0.linear_attn"]["kind"] == "linear-attention"
    assert points["layers.1.self_attn"]["kind"] == "attention"
    assert points["layers.1"]["kind"] == "block"
    assert points["layers.1.post_attention_layernorm"]["kind"] == "norm"
    assert "layers.0.self_attn" not in points
    assert data["instrumentation"]["backend"] == "nnterp"
    assert any("hybrid" in note.lower() for note in data["notes"])


def test_nnterp_capture_matches_hook_capture(service, trace_id, hybrid_checkpoint):
    """The adapter must agree with the generic PyTorch hook path."""

    source = _source(hybrid_checkpoint)
    nnterp_envelope = service.call(
        "capture_trace_activations",
        {
            "trace_id": trace_id,
            "model": source,
            "modules": ["layers.1", "layers.1.self_attn", "layers.0.linear_attn"],
            "mode": "selected",
            "positions": [1],
            "instrumentation": "nnterp",
        },
    )
    assert nnterp_envelope["ok"], nnterp_envelope
    nnterp_data = nnterp_envelope["data"]
    measurements = {item["module"]: item for item in nnterp_data["measurements"]}
    assert set(measurements) == {"layers.1", "layers.1.self_attn", "layers.0.linear_attn"}
    assert nnterp_data["replay_verification"]["numerical_agreement"] == "verified"
    assert nnterp_data["replay_verification"]["generation_reproducibility"] == "not-claimed"

    stored_record = service.call(
        "inspect_activation_capture",
        {"artifact_id": nnterp_data["capture_artifact_id"]},
    )
    assert stored_record["ok"], stored_record
    assert stored_record["data"]["mode"] == "selected"

    hooks_envelope = service.call(
        "capture_trace_activations",
        {
            "trace_id": trace_id,
            "model": source,
            "modules": ["model.layers.1"],
            "mode": "selected",
            "positions": [1],
        },
    )
    assert hooks_envelope["ok"], hooks_envelope
    hook_measurement = hooks_envelope["data"]["measurements"][0]

    stored = service.workspace.artifacts.read_array(measurements["layers.1"]["value_artifact_id"])
    reference = service.workspace.artifacts.read_array(hook_measurement["value_artifact_id"])
    assert stored.shape == reference.shape
    assert torch.allclose(
        torch.as_tensor(stored, dtype=torch.float64),
        torch.as_tensor(reference, dtype=torch.float64),
        atol=1e-6,
        rtol=1e-5,
    )


def test_nnterp_evidence_records_backend_identity(service, trace_id, hybrid_checkpoint):
    envelope = service.call(
        "capture_trace_activations",
        {
            "trace_id": trace_id,
            "model": _source(hybrid_checkpoint),
            "modules": ["layers.0"],
            "mode": "summary",
            "positions": [1],
            "instrumentation": "nnterp",
        },
    )
    assert envelope["ok"], envelope
    capture_id = envelope["data"]["capture_artifact_id"]
    stored = service.call("inspect_activation_capture", {"artifact_id": capture_id})
    assert stored["ok"], stored

    record = stored["data"]
    assert record["backend"] == "nnterp"
    assert record["backend_version"].startswith("nnterp ")
    assert record["privacy"]["contains_activation_values"] is False
    assert any("nnterp" in warning for warning in record["warnings"])


def test_nnterp_backend_refuses_synthetic_source(service, trace_id):
    with pytest.raises(InputError) as error:
        service.call(
            "capture_trace_activations",
            {
                "trace_id": trace_id,
                "model": {"kind": "synthetic"},
                "modules": ["layers.0"],
                "positions": [1],
                "instrumentation": "nnterp",
            },
        )
    assert "local Hugging Face checkpoint" in str(error.value)


def test_missing_optional_dependency_is_reported(monkeypatch, hybrid_checkpoint):
    def missing():
        raise UnsupportedError(
            "the nnterp instrumentation backend is not installed",
            details={"install_hint": nnsight_backend.INSTALL_HINT},
        )

    monkeypatch.setattr(nnsight_backend, "_require_optional", missing)
    with pytest.raises(UnsupportedError) as error:
        nnsight_backend.load_nnterp_model(str(hybrid_checkpoint))
    assert "nnsight==" in str(error.value.details.get("install_hint", "")) or "nnsight==" in str(
        error.value
    )


def test_unsupported_module_is_refused(hybrid_checkpoint, tmp_path):
    """A plain module without a resolvable architecture is refused, not guessed."""

    module = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model = nnsight_backend.NnterpReplayModel(
        module,
        tokenizer=_tokenizer(),
        name="plain",
    )
    with pytest.raises((UnsupportedError, CompatibilityError)) as error:
        model._build_standardized()
    assert "nnterp" in str(error.value).lower() or "standardis" in str(error.value).lower()


def test_non_eager_attention_is_reported(tmp_path):
    """self_attn output capture needs eager attention; the adapter says so."""

    config = Qwen3_5TextConfig(
        vocab_size=len(WORDS) + 4,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=32,
        layer_types=["full_attention", "full_attention"],
    )
    module = Qwen3_5ForCausalLM(config)
    module.eval()
    model = nnsight_backend.NnterpReplayModel(module, tokenizer=_tokenizer(), name="sdpa")
    with pytest.raises(CompatibilityError) as error:
        model._build_standardized()
    assert "eager" in str(error.value).lower()
