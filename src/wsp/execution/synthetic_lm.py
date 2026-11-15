"""A small, deterministic causal language model with a real tokenizer.

This is a *fixture model*, not an inference engine.  It exists so the workbench
can exercise generation, tracing, replay, alignment, and activation capture
offline, on CPU, without downloading weights.  Two properties make it useful:

* It is an ordinary causal transformer: position ``p`` only attends to tokens
  ``0..p``, so teacher-forced replay is a meaningful operation.
* Generation re-runs the full prefix at every step (no KV cache), so a
  teacher-forced pass over the recorded sequence sees exactly the same context
  the generator saw.  Replay agreement is therefore a sharp check of the
  measurement path rather than a tolerance lottery.

Two documented inductive biases are part of the fixture, not of WSP.  They give
behavioral detectors something real to find and make a controlled pair easy to
construct:

* ``repeat_bias`` nudges the next-token logits toward the token that was just
  consumed, which produces simple repetition loops.
* ``stop_bonus`` makes EOS overwhelmingly likely once the literal token
  ``stop`` appears anywhere in the sequence, which produces an immediate stop
  for a prompt that differs from a looping prompt by that one token.

The model is a plain ``torch.nn.Module`` and the tokenizer is a plain object.
Nothing here is special-cased by the code under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from wsp.core.errors import InputError
from wsp.core.hashing import sha256_bytes

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIAL_IDS = (PAD_ID, BOS_ID, EOS_ID, UNK_ID)

DEFAULT_MODEL_NAME = "tiny-causal-lm"
SYNTHETIC_BACKEND_VERSION = "1.0.0"

_CHARACTERS = "abcdefghijklmnopqrstuvwxyz .,!?'\n"
_MULTI_TOKENS = ("the", "loop", "ing", "and", "that", "stop")


class SyntheticModelSpec(BaseModel):
    """Everything needed to rebuild the fixture model deterministically."""

    model_config = ConfigDict(extra="forbid")

    kind: str = "tiny-causal"
    seed: int = 7
    model_dim: int = 32
    num_heads: int = 4
    num_layers: int = 3
    max_seq_len: int = 96
    repeat_bias: float = 0.25
    stop_bonus: float = 6.0

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.kind != "tiny-causal":
            raise InputError(
                "unsupported synthetic model kind",
                details={"kind": self.kind, "supported": ["tiny-causal"]},
            )
        if self.model_dim <= 0 or self.num_heads <= 0 or self.num_layers <= 0:
            raise InputError("synthetic model dimensions must be positive")
        if self.model_dim % self.num_heads != 0:
            raise InputError(
                "model_dim must be divisible by num_heads",
                details={"model_dim": self.model_dim, "num_heads": self.num_heads},
            )
        if self.max_seq_len < 2:
            raise InputError("max_seq_len must be at least 2")

    def digest(self) -> str:
        from wsp.core.json import canonical_bytes

        return sha256_bytes(canonical_bytes(self.model_dump(mode="json")))


class CharTokenizer:
    """Greedy longest-match tokenizer with explicit character offsets.

    Deliberately small, but not degenerate: several tokens are longer than one
    character, unknown characters still consume exactly one character, and
    offsets are reported for every token so alignment has a real ground truth.
    """

    name = "tiny-char"
    model_max_length = 4_096
    is_fast = True
    chat_template = None
    pad_token_id = PAD_ID
    bos_token_id = BOS_ID
    eos_token_id = EOS_ID
    unk_token_id = UNK_ID

    def __init__(self, extra_tokens: Iterable[str] = ()) -> None:
        tokens: list[str] = []
        for word in list(_MULTI_TOKENS) + list(extra_tokens):
            if word not in tokens:
                tokens.append(word)
        for character in _CHARACTERS:
            if character not in tokens:
                tokens.append(character)
        self._tokens = tokens
        self._ids = {token: index + len(SPECIAL_IDS) for index, token in enumerate(tokens)}
        self._longest = sorted(self._tokens, key=len, reverse=True)
        self.all_special_ids = list(SPECIAL_IDS)
        self.reverse_vocab = {index: token for token, index in self._ids.items()}

    # ------------------------------------------------------------------ basics

    @property
    def vocab_size(self) -> int:
        return len(self._tokens) + len(SPECIAL_IDS)

    def get_vocab(self) -> dict[str, int]:
        vocabulary = {
            "<pad>": PAD_ID,
            "<bos>": BOS_ID,
            "<eos>": EOS_ID,
            "<unk>": UNK_ID,
        }
        vocabulary.update(self._ids)
        return vocabulary

    # ------------------------------------------------------------------ encode

    def _encode_raw(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        index = 0
        while index < len(text):
            match = None
            for token in self._longest:
                if text.startswith(token, index):
                    match = token
                    break
            if match is None:
                ids.append(UNK_ID)
                offsets.append((index, index + 1))
                index += 1
                continue
            ids.append(self._ids[match])
            offsets.append((index, index + len(match)))
            index += len(match)
        return ids, offsets

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids, _ = self._encode_raw(text)
        if add_special_tokens:
            return [BOS_ID] + ids
        return ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        return self._encode_raw(text)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        pieces: list[str] = []
        for token_id in ids:
            value = int(token_id)
            if value in SPECIAL_IDS:
                if skip_special_tokens:
                    continue
                pieces.append({PAD_ID: "", BOS_ID: "", EOS_ID: "", UNK_ID: "\ufffd"}[value])
                continue
            pieces.append(self.reverse_vocab.get(value, "\ufffd"))
        return "".join(pieces)


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.out = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        cache: "TinyKVCache | None" = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        batch, length, dim = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        if cache is None:
            attended = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
        else:
            past = cache.get(layer_index)
            if past is not None:
                if length != 1:
                    raise InputError(
                        "a cached forward must process exactly one new position per call",
                        details={"length": int(length)},
                    )
                past_key, past_value = past
                key = torch.cat([past_key, key], dim=2)
                value = torch.cat([past_value, value], dim=2)
            attended = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=past is None
            )
            # The cache is an explicit object owned by the caller.  The model
            # never stores it on the module, so replay and hooks see the same
            # module tree either way.
            cache.set(layer_index, key, value)
        merged = attended.transpose(1, 2).reshape(batch, length, dim)
        return self.out(merged)


@dataclass
class TinyKVCache:
    """Per-layer key/value state for the fixture's cached decoder.

    Ownership is explicit: the object is passed into ``forward`` and mutated
    there.  Two branches must therefore either rebuild their own state or call
    :meth:`clone`, which copies every tensor.  A shallow copy would let one
    branch corrupt the other, which is exactly the failure the staged
    verification is designed to catch.
    """

    keys: list[torch.Tensor | None]
    values: list[torch.Tensor | None]
    token_ids: list[int]

    @classmethod
    def empty(cls, num_layers: int) -> "TinyKVCache":
        return cls(
            keys=[None for _ in range(num_layers)],
            values=[None for _ in range(num_layers)],
            token_ids=[],
        )

    def get(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        key = self.keys[layer_index]
        value = self.values[layer_index]
        if key is None or value is None:
            return None
        return key, value

    def set(self, layer_index: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.keys[layer_index] = key
        self.values[layer_index] = value

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def clone(self) -> "TinyKVCache":
        return TinyKVCache(
            keys=[None if item is None else item.clone() for item in self.keys],
            values=[None if item is None else item.clone() for item in self.values],
            token_ids=list(self.token_ids),
        )

    def state_hash(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        digest.update(str(self.token_ids).encode("utf-8"))
        for index, (key, value) in enumerate(zip(self.keys, self.values)):
            for tensor in (key, value):
                if tensor is None:
                    digest.update(f"|{index}:none".encode("utf-8"))
                    continue
                contiguous = tensor.detach().to("cpu", torch.float32).contiguous()
                digest.update(f"|{index}:{tuple(int(d) for d in contiguous.shape)}".encode("utf-8"))
                digest.update(contiguous.numpy().tobytes())
        return "sha256:" + digest.hexdigest()

    def state_bytes(self) -> int:
        total = 0
        for tensor in self.keys + self.values:
            if tensor is not None:
                total += int(tensor.numel()) * int(tensor.element_size())
        return total


class TinyBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.ln2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cache: TinyKVCache | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(self.ln1(hidden), cache=cache, layer_index=layer_index)
        return hidden + self.mlp(self.ln2(hidden))


class TinyCausalLM(nn.Module):
    """Causal transformer with named, addressable blocks."""

    def __init__(
        self,
        vocab_size: int,
        *,
        model_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 64,
        seed: int = 7,
        repeat_bias: float = 0.25,
        stop_bonus: float = 6.0,
        stop_token_id: int | None = None,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.repeat_bias = float(repeat_bias)
        self.stop_bonus = float(stop_bonus)
        self.stop_token_id = stop_token_id
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.pos = nn.Embedding(max_seq_len, model_dim)
        self.blocks = nn.ModuleList(
            [TinyBlock(model_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
                    * 0.08
                )
        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: TinyKVCache | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Map identifiers to logits.

        ``cache`` is optional and always caller-owned.  Without it the model is
        the ordinary full-prefix decoder the rest of the workbench replays.  With
        it, the call must carry exactly one new position, which is appended to
        the cache; the returned logits describe only that position.  No state is
        stored on the module, so parameter and buffer fingerprints are unaffected
        and every module still receives a normal forward call.
        """

        length = int(input_ids.shape[1])
        total_length = length + (cache.length if cache is not None else 0)
        if total_length > self.max_seq_len:
            raise ValueError(
                f"sequence length {total_length} exceeds the fixture maximum {self.max_seq_len}"
            )
        positions = (
            torch.arange(length, device=input_ids.device) + int(position_offset)
        ).unsqueeze(0)
        hidden = self.embed(input_ids.long()) + self.pos(positions)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cache=cache, layer_index=index)
        logits = self.head(self.norm(hidden))
        if cache is not None:
            full_ids = torch.tensor(
                [list(cache.token_ids) + [int(item) for item in input_ids[0].tolist()]],
                dtype=torch.long,
                device=input_ids.device,
            )
            cache.token_ids.extend(int(item) for item in input_ids[0].tolist())
        else:
            full_ids = input_ids
            if self.repeat_bias:
                previous = torch.roll(input_ids, shifts=1, dims=1)
                previous[:, 0] = input_ids[:, 0]
                logits = logits.scatter_add(
                    2, previous.unsqueeze(-1), torch.full_like(logits, self.repeat_bias)
                )
        if self.repeat_bias and cache is not None:
            previous = torch.roll(full_ids, shifts=1, dims=1)[:, -length:]
            if length == int(full_ids.shape[1]):
                # Prefill: position 0 has no predecessor, matching the uncached path.
                previous[:, 0] = full_ids[:, 0]
            logits = logits.scatter_add(
                2, previous.unsqueeze(-1), torch.full_like(logits, self.repeat_bias)
            )
        if self.stop_bonus and self.stop_token_id is not None:
            stopped = (full_ids == int(self.stop_token_id)).any(dim=1, keepdim=True)
            if bool(stopped.any()):
                boost = torch.zeros_like(logits).scatter(
                    2,
                    torch.full((logits.shape[0], logits.shape[1], 1), EOS_ID),
                    self.stop_bonus,
                )
                logits = logits + boost * stopped.unsqueeze(-1).to(logits.dtype)
        return logits


@dataclass
class SyntheticRuntime:
    """A built fixture: model, tokenizer, and the spec that reproduces both."""

    model: TinyCausalLM
    tokenizer: CharTokenizer
    spec: SyntheticModelSpec
    name: str = DEFAULT_MODEL_NAME

    def replay_model(self) -> Any:
        from wsp.replay.backends import TorchReplayModel

        return TorchReplayModel(
            self.model,
            name=self.name,
            tokenizer=self.tokenizer,
            backend="synthetic",
            backend_version=SYNTHETIC_BACKEND_VERSION,
        )

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.model.parameters()))

    def structure_fingerprint(self) -> str:
        from wsp.replay.backends import structure_fingerprint

        return structure_fingerprint(self.model)

    def module_map(self) -> dict[str, str]:
        """Addressable module names, for capture-point discovery and docs."""

        return {name: type(module).__name__ for name, module in self.model.named_modules() if name}


def build_runtime(
    spec: SyntheticModelSpec | None = None,
    **overrides: Any,
) -> SyntheticRuntime:
    """Build the fixture model from a spec, deterministically."""

    if spec is None:
        spec = SyntheticModelSpec(**overrides)
    elif overrides:
        spec = SyntheticModelSpec(**{**spec.model_dump(), **overrides})
    tokenizer = CharTokenizer()
    model = TinyCausalLM(
        tokenizer.vocab_size,
        model_dim=spec.model_dim,
        num_heads=spec.num_heads,
        num_layers=spec.num_layers,
        max_seq_len=spec.max_seq_len,
        seed=spec.seed,
        repeat_bias=spec.repeat_bias,
        stop_bonus=spec.stop_bonus,
        stop_token_id=tokenizer.get_vocab().get("stop"),
    )
    return SyntheticRuntime(model=model, tokenizer=tokenizer, spec=spec)


def greedy_generate(
    runtime: SyntheticRuntime,
    input_ids: Sequence[int],
    *,
    max_new_tokens: int = 8,
    eos_id: int = EOS_ID,
) -> tuple[list[int], list[float]]:
    """Greedy decode without a KV cache, recording per-step log-probabilities."""

    ids = [int(value) for value in input_ids]
    generated: list[int] = []
    logprobs: list[float] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = runtime.model(torch.tensor([ids], dtype=torch.long))
            step = torch.log_softmax(logits[0, -1].to(torch.float32), dim=-1)
            token_id = int(torch.argmax(step).item())
            logprobs.append(float(step[token_id].item()))
            generated.append(token_id)
            ids.append(token_id)
            if token_id == eos_id:
                break
    return generated, logprobs


def teacher_forced_logits(runtime: SyntheticRuntime, ids: Sequence[int]) -> torch.Tensor:
    with torch.inference_mode():
        return runtime.model(torch.tensor([list(ids)], dtype=torch.long))


__all__ = [
    "BOS_ID",
    "DEFAULT_MODEL_NAME",
    "EOS_ID",
    "PAD_ID",
    "SPECIAL_IDS",
    "SYNTHETIC_BACKEND_VERSION",
    "UNK_ID",
    "CausalSelfAttention",
    "CharTokenizer",
    "SyntheticModelSpec",
    "SyntheticRuntime",
    "TinyBlock",
    "TinyCausalLM",
    "TinyKVCache",
    "build_runtime",
    "greedy_generate",
    "teacher_forced_logits",
]
