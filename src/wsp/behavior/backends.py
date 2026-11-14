"""Generation backends for trace collection.

Only two backends exist in V1.1: a deterministic scripted backend used by
fixtures and demonstrations, and an optional local Hugging Face backend.  There
is deliberately no inference engine, no download path, and no hosted provider.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from wsp.behavior.models import (
    BackendRef,
    BackendSpec,
    ChatMessage,
    GenerationParams,
    GenerationTrace,
    InputRef,
    ModelRef,
    PrivacyInfo,
    Provenance,
    ScriptedBackendConfig,
    ScriptedFixture,
    TerminationInfo,
    TimingInfo,
    TokenAlternative,
    TokenRecord,
    TokenizerRef,
)
from wsp.behavior.ids import new_trace_id
from wsp.behavior.trace import finalize_trace
from wsp.core.budget import Budget
from wsp.core.errors import ExecutionError, InputError, ResourceLimitError, UnsupportedError
from wsp.core.hashing import sha256_bytes, sha256_file
from wsp.core.json import canonical_bytes
from wsp.core.time import utc_now_iso


class GenerationRequest(Protocol):
    """Structural request type implemented by the tool schema."""

    input_text: str
    system: str | None
    params: GenerationParams
    seed: int | None


class GenerationBackend(Protocol):
    def info(self) -> BackendRef: ...

    def generate(self, request: GenerationRequest, *, budget: Budget) -> GenerationTrace: ...


@dataclass
class ScriptedBackend:
    config: ScriptedBackendConfig

    def info(self) -> BackendRef:
        return BackendRef(
            name="scripted",
            version="1.0.0",
            device="cpu",
            dtype=None,
            deterministic=True,
            runtime={
                "fixtures": len(self.config.fixtures),
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        )

    def _select(self, input_text: str) -> ScriptedFixture:
        for fixture in self.config.fixtures:
            if fixture.input_equals is not None and fixture.input_equals == input_text:
                return fixture
        for fixture in self.config.fixtures:
            if fixture.input_contains is not None and fixture.input_contains in input_text:
                return fixture
        if self.config.default_fixture is not None:
            return self.config.default_fixture
        raise ExecutionError(
            "no scripted fixture matched the input and no default fixture was configured",
            details={"input_chars": len(input_text), "fixtures": len(self.config.fixtures)},
        )

    def generate(self, request: GenerationRequest, *, budget: Budget) -> GenerationTrace:
        started_at = utc_now_iso()
        started = time.perf_counter()
        fixture = self._select(request.input_text)
        if len(fixture.output_text) > budget.max_trace_chars:
            raise ResourceLimitError(
                "scripted output exceeds the configured character limit",
                details={"characters": len(fixture.output_text), "limit": budget.max_trace_chars},
            )
        tokens: list[TokenRecord] = []
        unavailable: dict[str, str] = {}
        if fixture.output_tokens:
            for index, token in enumerate(fixture.output_tokens):
                logprob = None
                if fixture.token_logprobs is not None:
                    logprob = fixture.token_logprobs[index]
                tokens.append(
                    TokenRecord(token_id=token.token_id, text=token.text, logprob=logprob)
                )
            if fixture.token_logprobs is None:
                unavailable["token_logprobs"] = "scripted fixture did not provide log-probabilities"
            if any(token.token_id is None for token in fixture.output_tokens):
                unavailable["token_ids"] = "scripted fixture provided text-only tokens"
        else:
            unavailable["tokens"] = "scripted fixture produced text without token records"
            unavailable["token_logprobs"] = "no token records were produced"
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        trace = GenerationTrace(
            trace_id=new_trace_id(),
            created_at=started_at,
            model=ModelRef(name=self.config.model_name, revision=self.config.model_revision),
            tokenizer=TokenizerRef(
                name=self.config.tokenizer_name,
                kind="synthetic",
                vocab_size=self.config.vocab_size,
            ),
            input=InputRef(
                text=request.input_text,
                system=request.system,
                messages=(
                    [ChatMessage(role="user", content=request.input_text)]
                    if request.system is None
                    else [
                        ChatMessage(role="system", content=request.system),
                        ChatMessage(role="user", content=request.input_text),
                    ]
                ),
            ),
            backend=self.info(),
            params=request.params,
            seed=request.seed,
            tokens=tokens,
            text=fixture.output_text,
            termination=TerminationInfo(
                reason=fixture.termination,
                detail=fixture.termination_detail,
                generated_tokens=len(tokens) if tokens else None,
            ),
            timing=TimingInfo(
                started_at=started_at,
                completed_at=utc_now_iso(),
                duration_ms=duration_ms,
                tokens_per_second=(
                    len(tokens) / (duration_ms / 1000.0)
                    if tokens and duration_ms > 0
                    else None
                ),
            ),
            tool_calls=fixture.tool_calls,
            provenance=Provenance(
                origin="generated",
                source_system="wsp.behavior.scripted",
                source_version="1.0.0",
            ),
            privacy=PrivacyInfo(
                store_input_text=True,
                store_output_text=True,
            ),
            unavailable=unavailable,
        )
        return finalize_trace(trace, budget)


@dataclass
class HuggingFaceBackend:
    """Optional local causal-LM generation with token-level observations."""

    config: Any  # HuggingFaceBackendConfig, kept loose to avoid a hard import
    _adapter: Any | None = None
    _model: Any | None = None
    _tokenizer: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        from wsp.adapters.huggingface import load_local_causal_lm

        adapter = load_local_causal_lm(
            self.config.model_path,
            device=self.config.device,
            dtype=self.config.dtype,
        )
        self._adapter = adapter
        self._model = adapter.model
        self._tokenizer = adapter.tokenizer
        return self._model, self._tokenizer

    def info(self) -> BackendRef:
        details: dict[str, Any] = {}
        try:
            import torch
            import transformers

            details = {"torch": torch.__version__, "transformers": transformers.__version__}
        except ImportError:  # pragma: no cover - only informational
            pass
        return BackendRef(
            name="huggingface",
            version=details.get("transformers"),
            device=self.config.device,
            dtype=self.config.dtype,
            deterministic=None,
            runtime=details,
        )

    def _tokenizer_ref(self, tokenizer: Any) -> TokenizerRef:
        fingerprint = None
        try:
            fingerprint = sha256_bytes(
                canonical_bytes(
                    {
                        "vocab": tokenizer.get_vocab(),
                        "model_max_length": getattr(tokenizer, "model_max_length", None),
                    }
                )
            )
        except Exception:
            fingerprint = None
        init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
        return TokenizerRef(
            name=getattr(tokenizer, "name_or_path", None) or self.config.tokenizer_path,
            revision=init_kwargs.get("_commit_hash"),
            kind=type(tokenizer).__name__,
            vocab_size=getattr(tokenizer, "vocab_size", None),
            fingerprint=fingerprint,
        )

    def _model_ref(self, model: Any, tokenizer: Any) -> ModelRef:
        path = Path(self.config.model_path).expanduser().resolve()
        config = getattr(model, "config", None)
        name = self.config.model_name
        if name is None and config is not None:
            name = getattr(config, "_name_or_path", None) or path.name
        file_sha256 = None
        if self.config.compute_file_sha256:
            candidates = sorted(path.glob("*.safetensors"))
            if candidates:
                file_sha256 = sha256_file(candidates[0])
        architectures = getattr(config, "architectures", None) if config is not None else None
        return ModelRef(
            name=name,
            revision=None,
            checkpoint_path=str(path),
            file_sha256=file_sha256,
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            metadata={"architectures": architectures, "tokenizer_name": getattr(tokenizer, "name_or_path", None)},
        )

    def _render_prompt(self, tokenizer: Any, request: GenerationRequest) -> Any:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.input_text})
        template = getattr(tokenizer, "chat_template", None)
        if template:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        rendered = (
            f"{request.system}\n\n{request.input_text}"
            if request.system
            else request.input_text
        )
        return tokenizer(rendered, return_tensors="pt")

    def generate(self, request: GenerationRequest, *, budget: Budget) -> GenerationTrace:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is a core dependency
            raise UnsupportedError("torch is required for local generation") from exc
        model, tokenizer = self._load()
        encoded = self._render_prompt(tokenizer, request)
        if "input_ids" not in encoded:
            raise ExecutionError("tokenizer did not return input_ids")
        device = model.device
        encoded = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        input_ids = encoded["input_ids"]
        if input_ids.shape[1] > budget.max_trace_input_tokens:
            raise ResourceLimitError(
                "prompt exceeds the configured input-token limit",
                details={
                    "tokens": int(input_ids.shape[1]),
                    "limit": budget.max_trace_input_tokens,
                },
            )
        if request.params.max_new_tokens > budget.max_trace_tokens:
            raise ResourceLimitError(
                "requested generation exceeds the configured trace token limit",
                details={
                    "max_new_tokens": request.params.max_new_tokens,
                    "limit": budget.max_trace_tokens,
                },
            )
        if request.seed is not None:
            torch.manual_seed(request.seed)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": request.params.max_new_tokens,
            "do_sample": request.params.sampling_enabled(),
            "output_scores": bool(request.params.return_logprobs),
            "return_dict_in_generate": True,
        }
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        if request.params.sampling_enabled():
            generate_kwargs["temperature"] = request.params.temperature
            if request.params.top_p is not None:
                generate_kwargs["top_p"] = request.params.top_p
            if request.params.top_k is not None:
                generate_kwargs["top_k"] = request.params.top_k
        if request.params.repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = request.params.repetition_penalty
        if request.params.stop_sequences:
            generate_kwargs["stop_strings"] = list(request.params.stop_sequences)
            generate_kwargs["tokenizer"] = tokenizer

        started_at = utc_now_iso()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                result = model.generate(**encoded, **generate_kwargs)
        except Exception as exc:
            raise ExecutionError(f"local generation failed: {exc}") from exc
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        sequences = getattr(result, "sequences", None)
        if sequences is None:
            raise ExecutionError("generation did not return a sequences tensor")
        input_length = int(input_ids.shape[1])
        generated_ids = sequences[0, input_length:].tolist()
        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        scores = getattr(result, "scores", ()) or ()
        vocab_size = None
        logits = getattr(model, "config", None)
        if logits is not None:
            vocab_size = getattr(logits, "vocab_size", None)
        top_k = request.params.top_logprobs
        tokens: list[TokenRecord] = []
        for index, token_id in enumerate(generated_ids):
            token_text = tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            logprob = None
            alternatives: list[TokenAlternative] = []
            if index < len(scores):
                step = torch.log_softmax(scores[index][0].float(), dim=-1)
                logprob = float(step[token_id].item())
                if top_k:
                    values, indices = torch.topk(step, k=min(top_k, step.shape[-1]))
                    alternatives = [
                        TokenAlternative(
                            token_id=int(alt_id),
                            text=tokenizer.decode(
                                [int(alt_id)],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            ),
                            logprob=float(value),
                        )
                        for value, alt_id in zip(values.tolist(), indices.tolist())
                    ]
            tokens.append(
                TokenRecord(
                    token_id=int(token_id),
                    text=token_text,
                    logprob=logprob,
                    top_logprobs=alternatives,
                )
            )
        unavailable: dict[str, str] = {}
        if not request.params.return_logprobs:
            unavailable["token_logprobs"] = "log-probabilities were disabled for this request"
        elif scores and len(scores) != len(generated_ids):
            unavailable["token_logprobs"] = "backend score count did not align with generated tokens"
        elif not scores:
            unavailable["token_logprobs"] = "backend did not return generation scores"

        eos_ids = tokenizer.eos_token_id
        eos_set = set(eos_ids) if isinstance(eos_ids, (list, tuple)) else {eos_ids}
        reason = "unknown"
        detail = None
        matched_stop = None
        if generated_ids and generated_ids[-1] in eos_set:
            reason = "eos"
        elif any(sequence in text for sequence in request.params.stop_sequences):
            reason = "stop_sequence"
            matched_stop = next(
                sequence for sequence in request.params.stop_sequences if sequence in text
            )
        elif len(generated_ids) >= request.params.max_new_tokens:
            reason = "length_limit"
            detail = f"generated {len(generated_ids)} tokens, reaching max_new_tokens"
        elif not generated_ids:
            reason = "unknown"
            detail = "the backend returned no generated tokens"

        model_ref = self._model_ref(model, tokenizer)
        return finalize_trace(
            GenerationTrace(
                trace_id=new_trace_id(),
                created_at=started_at,
                model=model_ref,
                tokenizer=self._tokenizer_ref(tokenizer),
                input=InputRef(
                    text=request.input_text,
                    system=request.system,
                    messages=(
                        ([ChatMessage(role="user", content=request.input_text)])
                        if not request.system
                        else [
                            ChatMessage(role="system", content=request.system),
                            ChatMessage(role="user", content=request.input_text),
                        ]
                    ),
                ),
                backend=BackendRef(
                    name="huggingface",
                    version=(self.info().runtime or {}).get("transformers"),
                    device=str(getattr(model, "device", self.config.device)),
                    dtype=str(getattr(model, "dtype", self.config.dtype)),
                    deterministic=not request.params.sampling_enabled(),
                    runtime=self.info().runtime,
                ),
                params=request.params,
                seed=request.seed,
                tokens=tokens,
                text=text,
                termination=TerminationInfo(
                    reason=reason,  # type: ignore[arg-type]
                    detail=detail,
                    stop_sequence=matched_stop,
                    generated_tokens=len(tokens),
                ),
                timing=TimingInfo(
                    started_at=started_at,
                    completed_at=utc_now_iso(),
                    duration_ms=duration_ms,
                    tokens_per_second=(
                        len(tokens) / (duration_ms / 1000.0) if tokens and duration_ms > 0 else None
                    ),
                ),
                provenance=Provenance(
                    origin="generated",
                    source_system="wsp.behavior.huggingface",
                    source_version=(self.info().runtime or {}).get("transformers"),
                ),
                privacy=PrivacyInfo(store_input_text=True, store_output_text=True),
                unavailable=unavailable,
                metadata={"vocab_size": vocab_size},
            ),
            budget,
        )


def build_backend(spec: BackendSpec, *, budget: Budget) -> GenerationBackend:
    if spec.kind == "scripted":
        assert spec.scripted is not None
        return ScriptedBackend(spec.scripted)
    if spec.kind == "huggingface":
        assert spec.huggingface is not None
        return HuggingFaceBackend(spec.huggingface)
    raise InputError(f"unknown backend kind {spec.kind!r}")
