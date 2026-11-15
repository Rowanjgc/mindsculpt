"""Architecture adapters for staged and cached generation.

Two things are deliberately separated here:

1. **Executing** a staged continuation (prefill once, then decode one position at
   a time with a cache).
2. **Proving** that this execution reproduces the uncached reference before any
   intervention result from it is treated as valid.

Only the tiny fixture decoders and ordinary full-attention Hugging Face causal
LMs are supported.  A hybrid architecture such as Qwen3.5, whose generation
state includes Gated DeltaNet recurrent state rather than only key/value
tensors, is rejected explicitly: cloning or patching part of that state would
not produce an independent continuation, and this release will not guess.

State ownership
---------------

A cache object is passed into ``forward`` and mutated there.  It is owned by the
caller.  Branches must be rebuilt (:meth:`StagedDecoder.prefill`) or copied with
:meth:`StagedDecoder.clone`, which copies every tensor.  ``clone_isolation`` is
a separate check because a shallow copy of a mutable cache is exactly the
mistake that silently contaminates a paired comparison.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

import torch

from wsp.core.errors import ExecutionError, InputError, UnsupportedError
from wsp.core.hashing import sha256_bytes

from .synthetic_lm import TinyCausalLM, TinyKVCache

ArchitectureClass = Literal["full-attention-transformer", "hybrid-recurrent", "unknown"]

HYBRID_ARCHITECTURE_NOTE = (
    "this architecture carries recurrent or hybrid generation state in addition to "
    "key/value attention state; staged decoding is not verified for it in this release"
)


@dataclass(frozen=True)
class StagedCapabilities:
    supported: bool
    backend: str
    architecture_class: ArchitectureClass
    state_kind: str
    reason: str | None = None
    position_ids_mode: str | None = None
    attention_mask_mode: str | None = None


class StagedDecoder(Protocol):
    name: str
    backend: str
    architecture_class: ArchitectureClass

    def capabilities(self) -> StagedCapabilities: ...

    def root(self) -> torch.nn.Module: ...

    def prefill(self, token_ids: Sequence[int]) -> tuple[torch.Tensor, Any]: ...

    def step(self, state: Any, token_id: int) -> tuple[torch.Tensor, Any]: ...

    def clone(self, state: Any) -> Any: ...

    def state_hash(self, state: Any) -> str: ...

    def state_bytes(self, state: Any) -> int: ...

    def state_length(self, state: Any) -> int: ...


def _torch_module(model: Any) -> torch.nn.Module:
    if isinstance(model, torch.nn.Module):
        return model
    for attribute in ("module", "model"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate
    raise UnsupportedError(
        "the model wrapper does not expose a torch module for staged execution",
        details={"type": type(model).__name__},
    )


# --------------------------------------------------------------------------- #
# fixture decoder
# --------------------------------------------------------------------------- #


@dataclass
class TinyStagedDecoder:
    """KV-cached decoding for the offline fixture transformer.

    The cached path calls the same modules with the same parameters as the
    uncached path; only attention reads and writes a cache.  That is what lets
    forward hooks observe staged activations at the normal module paths.
    """

    module: TinyCausalLM
    name: str = "tiny-kv-cache"
    backend: str = "torch-kv-cache"
    architecture_class: ArchitectureClass = "full-attention-transformer"

    def capabilities(self) -> StagedCapabilities:
        return StagedCapabilities(
            supported=True,
            backend=self.backend,
            architecture_class=self.architecture_class,
            state_kind="per-layer key/value tensors",
            position_ids_mode="contiguous absolute positions derived from the prefix length",
            attention_mask_mode="implicit causal mask; all prefix positions attended",
        )

    def root(self) -> torch.nn.Module:
        return self.module

    def prefill(self, token_ids: Sequence[int]) -> tuple[torch.Tensor, TinyKVCache]:
        tokens = [int(item) for item in token_ids]
        if not tokens:
            raise InputError("staged prefill requires at least one token")
        cache = TinyKVCache.empty(len(self.module.blocks))
        ids = torch.tensor([tokens], dtype=torch.long)
        with torch.inference_mode():
            logits = self.module(ids, cache=cache, position_offset=0)
        return logits, cache

    def step(self, state: TinyKVCache, token_id: int) -> tuple[torch.Tensor, TinyKVCache]:
        if not isinstance(state, TinyKVCache):
            raise UnsupportedError("the staged state is not a fixture key/value cache")
        ids = torch.tensor([[int(token_id)]], dtype=torch.long)
        with torch.inference_mode():
            logits = self.module(ids, cache=state, position_offset=state.length)
        return logits, state

    def clone(self, state: TinyKVCache) -> TinyKVCache:
        return state.clone()

    def state_hash(self, state: TinyKVCache) -> str:
        return state.state_hash()

    def state_bytes(self, state: TinyKVCache) -> int:
        return state.state_bytes()

    def state_length(self, state: TinyKVCache) -> int:
        return state.length


# --------------------------------------------------------------------------- #
# Hugging Face decoder
# --------------------------------------------------------------------------- #

_HYBRID_MODEL_TYPES = ("qwen3_5", "qwen3_5_moe", "mamba", "mamba2", "zamba", "jamba")


def _hf_text_config(module: Any) -> Any:
    config = getattr(module, "config", None)
    return getattr(config, "text_config", None) or config


def classify_hf_architecture(module: Any) -> tuple[ArchitectureClass, str | None]:
    """Decide whether a HF model's generation state is plain key/value state."""

    config = _hf_text_config(module)
    if config is None:
        return "unknown", "the model exposes no configuration"
    model_type = str(getattr(config, "model_type", "") or "")
    layer_types = [str(item) for item in (getattr(config, "layer_types", None) or [])]
    hybrid_layers = [item for item in layer_types if item != "full_attention"]
    if hybrid_layers:
        return "hybrid-recurrent", f"non-attention layers present: {sorted(set(hybrid_layers))}"
    if any(marker in model_type for marker in _HYBRID_MODEL_TYPES):
        return "hybrid-recurrent", f"model_type '{model_type}' carries recurrent state"
    if bool(getattr(config, "is_encoder_decoder", False)):
        return "unknown", "encoder-decoder models are not supported"
    return "full-attention-transformer", None


def cache_pairs(cache: Any) -> list[tuple[Any, Any]]:
    """Normalise a cache object to ``[(key, value), ...]`` or reject it."""

    legacy = getattr(cache, "to_legacy_cache", None)
    if callable(legacy):
        try:
            return list(legacy())
        except Exception:  # pragma: no cover - version specific
            pass
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        return list(zip(key_cache, value_cache))
    layers = getattr(cache, "layers", None)
    if layers is not None:
        pairs: list[tuple[Any, Any]] = []
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None and values is None:
                continue
            pairs.append((keys, values))
        return pairs
    if isinstance(cache, (tuple, list)):
        return list(cache)
    raise UnsupportedError(
        "the generation state is not a key/value cache this release can verify",
        details={"type": type(cache).__name__},
    )


def _assert_kv_state(cache: Any) -> None:
    for index, pair in enumerate(cache_pairs(cache)):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise UnsupportedError(
                "the generation state contains a non key/value entry",
                details={"layer": index, "entry_type": type(pair).__name__},
            )
        key, value = pair
        if key is None and value is None:
            continue
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise UnsupportedError(
                "the generation state contains a non-tensor entry, so it is not a plain "
                "attention cache and cannot be cloned or verified safely",
                details={"layer": index, "key": type(key).__name__, "value": type(value).__name__},
            )


@dataclass
class HFState:
    cache: Any
    token_ids: list[int]


def _accepts_argument(function: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    return name in parameters


@dataclass
class HFStagedDecoder:
    """Cached decoding through a local ``transformers`` causal LM.

    Only plain key/value caches are accepted.  The adapter never constructs a
    custom attention kernel: it calls the model's own ``use_cache`` path so the
    architecture's own masking and position handling are used.
    """

    module: torch.nn.Module
    name: str = "hf-past-key-values"
    backend: str = "transformers-cache"
    attention_mask_model: str = "explicit all-ones mask over the full cached sequence"

    def __post_init__(self) -> None:
        architecture, reason = classify_hf_architecture(self.module)
        self._architecture = architecture
        self._reason = reason
        self._accepts_cache_position = _accepts_argument(self.module.forward, "cache_position")

    @property
    def architecture_class(self) -> ArchitectureClass:
        return self._architecture

    def capabilities(self) -> StagedCapabilities:
        supported = self._architecture == "full-attention-transformer"
        return StagedCapabilities(
            supported=supported,
            backend=self.backend,
            architecture_class=self._architecture,
            state_kind="past_key_values (key/value tensors per attention layer)",
            reason=self._reason,
            position_ids_mode=(
                "explicit cache_position over the cached prefix"
                if self._accepts_cache_position
                else "derived by the architecture from the full attention mask"
            ),
            attention_mask_mode=self.attention_mask_model,
        )

    def _require_supported(self) -> None:
        capabilities = self.capabilities()
        if not capabilities.supported:
            raise UnsupportedError(
                HYBRID_ARCHITECTURE_NOTE,
                details={
                    "architecture_class": capabilities.architecture_class,
                    "reason": capabilities.reason,
                    "note": (
                        "use full-prefix uncached generation, which this release verifies "
                        "for every supported adapter"
                    ),
                },
            )

    def root(self) -> torch.nn.Module:
        return self.module

    def _forward(
        self,
        ids: list[int],
        cache: Any | None,
        *,
        past_length: int = 0,
    ) -> tuple[torch.Tensor, Any]:
        device = next(self.module.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        # A cached decode forward must advertise every position, not just the new
        # one: an all-ones mask of length 1 makes the architecture derive
        # position 0 for the current token, which silently corrupts the
        # continuation.  The mask therefore spans the cached prefix.
        total = past_length + len(ids)
        attention_mask = torch.ones((1, total), dtype=torch.long, device=device)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "return_dict": True,
        }
        if self._accepts_cache_position:
            kwargs["cache_position"] = torch.arange(
                past_length, total, dtype=torch.long, device=device
            )
        if cache is not None:
            kwargs["past_key_values"] = cache
        with torch.inference_mode():
            output = self.module(**kwargs)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise ExecutionError("the model did not return logits from a cached forward")
        new_cache = getattr(output, "past_key_values", None)
        if new_cache is None:
            raise UnsupportedError("the model returned no generation cache")
        _assert_kv_state(new_cache)
        return logits, new_cache

    def prefill(self, token_ids: Sequence[int]) -> tuple[torch.Tensor, HFState]:
        self._require_supported()
        tokens = [int(item) for item in token_ids]
        if not tokens:
            raise InputError("staged prefill requires at least one token")
        logits, cache = self._forward(tokens, None, past_length=0)
        return logits, HFState(cache=cache, token_ids=tokens)

    def step(self, state: HFState, token_id: int) -> tuple[torch.Tensor, HFState]:
        self._require_supported()
        if not isinstance(state, HFState):
            raise UnsupportedError("the staged state is not a Hugging Face cache state")
        tokens = state.token_ids + [int(token_id)]
        # The cache already holds every position, so only the new token is fed,
        # but the mask and cache position still describe the whole sequence.
        logits, cache = self._forward(
            [int(token_id)], state.cache, past_length=len(state.token_ids)
        )
        return logits, HFState(cache=cache, token_ids=tokens)

    def clone(self, state: HFState) -> HFState:
        return HFState(cache=copy.deepcopy(state.cache), token_ids=list(state.token_ids))

    def state_hash(self, state: HFState) -> str:
        digest = sha256_bytes(str(state.token_ids).encode("utf-8"))
        parts = [digest]
        for index, (key, value) in enumerate(cache_pairs(state.cache)):
            for tensor in (key, value):
                if tensor is None:
                    parts.append(f"{index}:none")
                    continue
                contiguous = tensor.detach().to("cpu", torch.float32).contiguous()
                parts.append(
                    f"{index}:{tuple(int(d) for d in contiguous.shape)}:"
                    + sha256_bytes(contiguous.numpy().tobytes())
                )
        return sha256_bytes("\n".join(parts).encode("utf-8"))

    def state_bytes(self, state: HFState) -> int:
        total = 0
        for key, value in cache_pairs(state.cache):
            for tensor in (key, value):
                if isinstance(tensor, torch.Tensor):
                    total += int(tensor.numel()) * int(tensor.element_size())
        return total

    def state_length(self, state: HFState) -> int:
        return len(state.token_ids)


# --------------------------------------------------------------------------- #
# factory and isolation
# --------------------------------------------------------------------------- #


def build_staged_decoder(model: Any) -> StagedDecoder:
    """Pick the staged adapter for a loaded model, or fail explicitly."""

    module = _torch_module(model)
    if isinstance(module, TinyCausalLM):
        return TinyStagedDecoder(module=module)
    architecture, reason = classify_hf_architecture(module)
    if architecture != "full-attention-transformer":
        raise UnsupportedError(
            HYBRID_ARCHITECTURE_NOTE,
            details={
                "architecture_class": architecture,
                "reason": reason,
                "supported": ["tiny fixture key/value cache", "full-attention Hugging Face LMs"],
            },
        )
    if not hasattr(module, "forward"):
        raise UnsupportedError("the model module has no forward interface")
    return HFStagedDecoder(module=module)


def assert_clone_isolation(
    decoder: StagedDecoder, original: Any, before_hash: str, *, action: str = "branch"
) -> None:
    """Fail when a branch mutated the state it was supposed to copy.

    This is the check that makes ``clone`` trustworthy.  A shallow copy whose
    tensor storage is shared produces a different hash here, because decoding
    appends or rewrites cache entries.
    """

    after = decoder.state_hash(original)
    if after != before_hash:
        raise ExecutionError(
            "the cloned generation state shares mutable storage with its source; the "
            f"{action} contaminated the reference state",
            details={
                "before": before_hash,
                "after": after,
                "hint": "rebuild the state with prefill or deep-copy every cache tensor",
            },
        )


def state_summary(decoder: StagedDecoder, state: Any) -> dict[str, Any]:
    return {
        "backend": decoder.backend,
        "state_hash": decoder.state_hash(state),
        "state_bytes": decoder.state_bytes(state),
        "state_length": decoder.state_length(state),
    }


__all__ = [
    "ArchitectureClass",
    "HYBRID_ARCHITECTURE_NOTE",
    "HFStagedDecoder",
    "HFState",
    "StagedCapabilities",
    "StagedDecoder",
    "TinyStagedDecoder",
    "assert_clone_isolation",
    "build_staged_decoder",
    "cache_pairs",
    "classify_hf_architecture",
    "state_summary",
]
