"""Model adapters used by replay.

Architecture-specific knowledge lives here, not in the replay engine.  Two
adapters exist:

* :class:`HuggingFaceReplayModel` for a local ``transformers`` checkpoint.
* :class:`TorchReplayModel` for any ``nn.Module`` whose forward maps token
  identifiers to logits, which is what the offline fixtures use.

Both expose the same small interface, so the engine never assumes that a model
has a residual stream, an attention block, or a particular module naming
scheme.  Optional backends (NNsight/nnterp) plug in at this seam.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch

from wsp.core.dtypes import torch_dtype_name
from wsp.core.errors import (
    CompatibilityError,
    ExecutionError,
    SafetyError,
    UnsupportedError,
    WspError,
)
from wsp.core.hashing import sha256_bytes, sha256_file
from wsp.replay.models import HYBRID_ARCHITECTURE_NOTE, ReplayModelInfo

_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_HASH_LOCK = threading.Lock()


def _file_digest(path: Path) -> str:
    """Hash a checkpoint file, caching by (path, size, mtime)."""

    stat = path.stat()
    key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    with _HASH_LOCK:
        cached = _FILE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = sha256_file(path)
    with _HASH_LOCK:
        _FILE_HASH_CACHE[key] = digest
    return digest


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _module_dtype(module: torch.nn.Module) -> str | None:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return torch_dtype_name(parameter.dtype)
    return None


def _parameter_count(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def structure_fingerprint(module: torch.nn.Module) -> str:
    """Digest of module tree, parameter names, shapes, and dtypes.

    Identity for modules that have no checkpoint file: it changes when the
    architecture or shapes change, and never when only values change, so it can
    be compared across a rebuilt synthetic model and a recorded trace.
    """

    parts: list[str] = [f"module={type(module).__name__}"]
    for name, child in module.named_modules():
        if name:
            parts.append(f"m:{name}:{type(child).__name__}")
    for name, parameter in sorted(module.named_parameters()):
        parts.append(
            f"p:{name}:{tuple(int(d) for d in parameter.shape)}:{parameter.dtype}"
        )
    for name, buffer in sorted(module.named_buffers()):
        parts.append(f"b:{name}:{tuple(int(d) for d in buffer.shape)}:{buffer.dtype}")
    return sha256_bytes("\n".join(parts).encode("utf-8"))


@dataclass
class TokenizationSupport:
    """What a tokenizer can tell us about text and special tokens.

    ``offsets`` maps token index to ``(char_start, char_end)`` for the *same*
    text that produced the identifiers.  Fast tokenizers provide it directly;
    slow ones do not, and alignment reports ambiguity rather than guessing.
    """

    encode: Any | None = None
    encode_with_offsets: Any | None = None
    decode: Any | None = None
    special_ids: set[int] = field(default_factory=set)


class ReplayModel(Protocol):
    """Structural interface implemented by every replay adapter."""

    def describe(self) -> ReplayModelInfo: ...

    def forward_token_ids(self, token_ids: Sequence[int]) -> torch.Tensor: ...

    def tokenizer_support(self) -> TokenizationSupport: ...


@dataclass
class HuggingFaceReplayModel:
    """A locally loaded ``transformers`` causal LM.

    Loading is always ``local_files_only`` with ``trust_remote_code=False``;
    this class never downloads weights.
    """

    model: Any
    tokenizer: Any
    model_path: str
    device: str = "cpu"
    dtype: str | None = None
    compute_file_sha256: bool = True
    cache_file_sha256: bool = True
    _info: ReplayModelInfo | None = None

    # ------------------------------------------------------------------ describe

    def describe(self) -> ReplayModelInfo:
        if self._info is not None:
            return self._info
        module = self.model
        if not isinstance(module, torch.nn.Module):  # pragma: no cover - defensive
            raise ExecutionError("the loaded Hugging Face model is not a torch module")
        config = getattr(module, "config", None)
        text_config = getattr(config, "text_config", None) or config
        architectures = list(getattr(config, "architectures", None) or [])
        path = Path(self.model_path).expanduser()
        file_sha256 = None
        if self.compute_file_sha256:
            candidates = sorted(path.glob("*.safetensors"))
            if candidates:
                try:
                    file_sha256 = _file_digest(candidates[0])
                except OSError:
                    file_sha256 = None
        chat_template = getattr(self.tokenizer, "chat_template", None)
        from wsp.behavior.backends import tokenizer_fingerprint

        caveats: list[str] = []
        layer_types = list(getattr(text_config, "layer_types", None) or [])
        hybrid = any(str(value) != "full_attention" for value in layer_types)
        if hybrid:
            caveats.append(HYBRID_ARCHITECTURE_NOTE)
            caveats.append(
                "Detected layer_types: " + ", ".join(str(value) for value in layer_types)
            )
        runtime: dict[str, str] = {}
        try:
            import transformers

            runtime["transformers"] = transformers.__version__
        except ImportError:  # pragma: no cover - optional dependency
            pass
        try:
            runtime["torch"] = torch.__version__
        except AttributeError:  # pragma: no cover - defensive
            pass
        if hybrid:
            runtime["architecture"] = "hybrid-recurrent"
        info = ReplayModelInfo(
            name=getattr(config, "_name_or_path", None) or path.name,
            model_class=type(module).__name__,
            parameter_count=_parameter_count(module),
            file_sha256=file_sha256,
            structure_fingerprint=None,
            config_architectures=architectures,
            tokenizer_name=getattr(self.tokenizer, "name_or_path", None),
            tokenizer_kind=type(self.tokenizer).__name__,
            tokenizer_fingerprint=tokenizer_fingerprint(self.tokenizer),
            tokenizer_vocab_size=getattr(self.tokenizer, "vocab_size", None),
            tokenizer_max_length=getattr(self.tokenizer, "model_max_length", None),
            chat_template_sha256=(
                sha256_bytes(chat_template.encode("utf-8"))
                if isinstance(chat_template, str)
                else None
            ),
            device=str(_module_device(module)) if isinstance(module, torch.nn.Module) else self.device,
            dtype=_module_dtype(module) or self.dtype,
            backend="huggingface",
            backend_version=runtime.get("transformers"),
            runtime=runtime,
            capabilities={
                "chat_template": bool(chat_template),
                "token_offsets": bool(getattr(self.tokenizer, "is_fast", False)),
                "generation": hasattr(module, "generate"),
                "hybrid_layers": hybrid,
            },
            caveats=caveats,
        )
        self._info = info
        return info

    # -------------------------------------------------------------------- forward

    def forward_token_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        if not token_ids:
            raise ExecutionError("cannot replay an empty token sequence")
        module = self.model
        device = _module_device(module)
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }
        try:
            with torch.inference_mode():
                output = module(**kwargs)
        except TypeError:
            # Some architectures do not accept ``use_cache`` or ``return_dict``.
            kwargs.pop("use_cache", None)
            kwargs.pop("return_dict", None)
            try:
                with torch.inference_mode():
                    output = module(**kwargs)
            except Exception as exc:  # pragma: no cover - depends on model
                raise ExecutionError(f"replay forward failed: {exc}") from exc
        except Exception as exc:
            raise ExecutionError(f"replay forward failed: {exc}") from exc
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, (tuple, list)) and output:
            logits = output[0]
        if not isinstance(logits, torch.Tensor):
            raise ExecutionError(
                "the model did not return a logits tensor",
                details={"output_type": type(output).__name__},
            )
        return logits

    # ------------------------------------------------------------------ tokenizer

    def tokenizer_support(self) -> TokenizationSupport:
        tokenizer = self.tokenizer

        def _encode(text: str, add_special_tokens: bool = True) -> list[int]:
            encoded = tokenizer(text, add_special_tokens=add_special_tokens)
            return [int(value) for value in encoded["input_ids"]]

        def _encode_with_offsets(text: str) -> tuple[list[int], list[tuple[int, int]]]:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoded.get("offset_mapping")
            if offsets is None:
                raise CompatibilityError(
                    "this tokenizer does not report character offsets",
                    details={"tokenizer": type(tokenizer).__name__},
                )
            pairs = [(int(start), int(end)) for start, end in offsets]
            ids = [int(value) for value in encoded["input_ids"]]
            if len(pairs) != len(ids):
                raise CompatibilityError("tokenizer offsets do not align with token identifiers")
            return ids, pairs

        def _decode(ids: Sequence[int]) -> str:
            return tokenizer.decode(
                [int(value) for value in ids],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        return TokenizationSupport(
            encode=_encode,
            encode_with_offsets=(
                _encode_with_offsets if getattr(tokenizer, "is_fast", False) else None
            ),
            decode=_decode,
            special_ids=_special_token_ids(tokenizer),
        )


def _special_token_ids(tokenizer: Any) -> set[int]:
    values: set[int] = set()
    for attribute in (
        "all_special_ids",
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    ):
        candidate = getattr(tokenizer, attribute, None)
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple, set)):
            values.update(int(item) for item in candidate if item is not None)
        else:
            values.add(int(candidate))
    return values


@dataclass
class TorchReplayModel:
    """Adapter for a plain module whose forward maps identifiers to logits.

    This is what the offline tests use.  ``tokenizer`` is any object exposing
    ``encode``/``decode``/``get_vocab``-shaped methods; when it is absent,
    alignment and offset mapping are reported as unavailable rather than
    invented.
    """

    module: torch.nn.Module
    name: str | None = None
    tokenizer: Any | None = None
    backend: str = "torch-module"
    backend_version: str | None = None
    _info: ReplayModelInfo | None = None
    _special_ids: set[int] | None = None

    def describe(self) -> ReplayModelInfo:
        if self._info is not None:
            return self._info
        tokenizer = self.tokenizer
        from wsp.behavior.backends import tokenizer_fingerprint

        vocab_size = None
        if tokenizer is not None:
            vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(self.module, "vocab_size", None)
        chat_template = getattr(tokenizer, "chat_template", None) if tokenizer else None
        info = ReplayModelInfo(
            name=self.name,
            model_class=type(self.module).__name__,
            parameter_count=_parameter_count(self.module),
            file_sha256=None,
            structure_fingerprint=structure_fingerprint(self.module),
            config_architectures=[type(self.module).__name__],
            tokenizer_name=getattr(tokenizer, "name", None) if tokenizer else None,
            tokenizer_kind=type(tokenizer).__name__ if tokenizer else None,
            tokenizer_fingerprint=tokenizer_fingerprint(tokenizer) if tokenizer else None,
            tokenizer_vocab_size=int(vocab_size) if vocab_size is not None else None,
            tokenizer_max_length=(
                getattr(tokenizer, "model_max_length", None) if tokenizer else None
            ),
            chat_template_sha256=(
                sha256_bytes(chat_template.encode("utf-8"))
                if isinstance(chat_template, str)
                else None
            ),
            device=str(_module_device(self.module)),
            dtype=_module_dtype(self.module),
            backend=self.backend,
            backend_version=self.backend_version,
            runtime={"torch": torch.__version__},
            capabilities={
                "chat_template": bool(chat_template),
                "token_offsets": bool(tokenizer is not None and getattr(tokenizer, "is_fast", False)),
                "generation": hasattr(self.module, "generate"),
                "hybrid_layers": False,
            },
        )
        self._info = info
        return info

    def forward_token_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        if not token_ids:
            raise ExecutionError("cannot replay an empty token sequence")
        device = _module_device(self.module)
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        try:
            with torch.inference_mode():
                output = self.module(input_ids)
        except WspError:
            # Resource and input errors raised by instrumentation must stay
            # recognisable; wrapping them would hide the real cause.
            raise
        except Exception as exc:
            raise ExecutionError(f"replay forward failed: {exc}") from exc
        if isinstance(output, torch.Tensor):
            return output
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, (tuple, list)) and output:
            logits = output[0]
        if not isinstance(logits, torch.Tensor):
            raise ExecutionError(
                "the registered module did not return logits",
                details={"module": type(self.module).__name__},
            )
        return logits

    def tokenizer_support(self) -> TokenizationSupport:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return TokenizationSupport(special_ids=self._special_token_ids())
        encode = getattr(tokenizer, "encode", None)
        decode = getattr(tokenizer, "decode", None)
        offsets = getattr(tokenizer, "encode_with_offsets", None)
        return TokenizationSupport(
            encode=(lambda text, add_special_tokens=True: list(encode(text))) if encode else None,
            encode_with_offsets=offsets,
            decode=(lambda ids: decode(list(ids))) if decode else None,
            special_ids=self._special_token_ids(),
        )

    def _special_token_ids(self) -> set[int]:
        if self._special_ids is not None:
            return self._special_ids
        values: set[int] = set()
        tokenizer = self.tokenizer
        if tokenizer is not None:
            candidate = getattr(tokenizer, "all_special_ids", None)
            if candidate:
                values.update(int(item) for item in candidate)
        self._special_ids = values
        return values


def load_local_replay_model(
    path: str | Path,
    *,
    device: str = "cpu",
    dtype: str | None = None,
    compute_file_sha256: bool = True,
) -> HuggingFaceReplayModel:
    """Load a causal LM from an existing local directory.

    Automatic downloads are disabled, mirroring the V1.1 generation backend.
    """

    local_path = Path(path).expanduser().resolve()
    if not local_path.exists() or not local_path.is_dir():
        raise SafetyError(
            "replay loads only local model directories; automatic downloads are disabled",
            details={"path": str(local_path)},
        )
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise UnsupportedError(
            "transformers is not installed; install the optional 'hf' extra"
        ) from exc
    from wsp.adapters.huggingface import load_local_causal_lm

    adapter = load_local_causal_lm(local_path, device=device, dtype=dtype)
    return HuggingFaceReplayModel(
        model=adapter.model,
        tokenizer=adapter.tokenizer,
        model_path=str(local_path),
        device=device,
        dtype=dtype,
        compute_file_sha256=compute_file_sha256,
    )
