"""Test helpers for the offline fixture causal LM.

The model and tokenizer live in :mod:`wsp.execution.synthetic_lm` so that the
same implementation is used by the production synthetic backend, by replay, and
by these tests.  This module only adds trace-recording helpers that are useful
in tests and deliberately not part of the library.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from wsp.behavior.models import (
    BackendRef,
    GenerationParams,
    GenerationTrace,
    InputRef,
    ModelRef,
    PrivacyInfo,
    Provenance,
    TerminationInfo,
    TimingInfo,
    TokenRecord,
    TokenizerRef,
)
from wsp.behavior.trace import finalize_trace
from wsp.core.budget import Budget
from wsp.core.hashing import sha256_bytes
from wsp.core.time import utc_now_iso
from wsp.execution.synthetic_lm import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_IDS,
    UNK_ID,
    CausalSelfAttention,
    CharTokenizer,
    SyntheticModelSpec,
    SyntheticRuntime,
    TinyBlock,
    TinyCausalLM,
    build_runtime as _build_runtime,
    greedy_generate,
    teacher_forced_logits,
)


#: Backwards-compatible alias used by earlier tests.
TinyRuntime = SyntheticRuntime


def build_runtime(
    *,
    seed: int = 7,
    model_dim: int = 32,
    num_layers: int = 3,
    max_seq_len: int = 96,
) -> SyntheticRuntime:
    return _build_runtime(
        SyntheticModelSpec(
            seed=seed,
            model_dim=model_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )
    )


def greedy_generation(
    model: TinyCausalLM,
    input_ids: Sequence[int],
    *,
    max_new_tokens: int = 8,
    eos_id: int = EOS_ID,
) -> tuple[list[int], list[float]]:
    """Greedy decode for a bare model, without a runtime wrapper."""

    tokenizer = CharTokenizer()
    runtime = SyntheticRuntime(
        model=model, tokenizer=tokenizer, spec=SyntheticModelSpec()
    )
    return greedy_generate(
        runtime, input_ids, max_new_tokens=max_new_tokens, eos_id=eos_id
    )


def token_fingerprint(tokenizer: CharTokenizer) -> str:
    """The canonical fingerprint, so fixture traces match replay identity checks."""

    from wsp.behavior.backends import tokenizer_fingerprint as canonical

    value = canonical(tokenizer)
    assert value is not None
    return value


def make_trace(
    runtime: SyntheticRuntime,
    prompt: str,
    *,
    max_new_tokens: int = 8,
    trace_id: str | None = None,
    params: dict[str, Any] | None = None,
    budget: Budget | None = None,
) -> GenerationTrace:
    """Record a generation from the fixture as a valid ``wsp.trace/1.1``."""

    tokenizer = runtime.tokenizer
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    generated, logprobs = greedy_generate(
        runtime, input_ids, max_new_tokens=max_new_tokens
    )
    text = tokenizer.decode(generated, skip_special_tokens=True)
    created_at = utc_now_iso()
    trace = GenerationTrace(
        trace_id=trace_id or f"tr_{sha256_bytes(prompt.encode()).split(':', 1)[1][:16]}",
        created_at=created_at,
        model=ModelRef(
            name=runtime.name,
            parameter_count=runtime.parameter_count(),
            structure_fingerprint=runtime.structure_fingerprint(),
            metadata={
                "architectures": [type(runtime.model).__name__],
                "synthetic_spec": runtime.spec.model_dump(mode="json"),
            },
        ),
        tokenizer=TokenizerRef(
            name=tokenizer.name,
            kind=type(tokenizer).__name__,
            vocab_size=tokenizer.vocab_size,
            fingerprint=token_fingerprint(tokenizer),
        ),
        input=InputRef(
            text=prompt,
            token_ids=input_ids,
            token_encoding="text",
            add_special_tokens=True,
        ),
        backend=BackendRef(
            name="torch-module",
            version="1.0.0",
            device="cpu",
            dtype="float32",
            deterministic=True,
            runtime={"torch": torch.__version__},
        ),
        params=GenerationParams(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            **(params or {}),
        ),
        tokens=[
            TokenRecord(
                token_id=token_id,
                text=tokenizer.decode([token_id], skip_special_tokens=False),
                logprob=logprob,
            )
            for token_id, logprob in zip(generated, logprobs)
        ],
        text=text,
        termination=TerminationInfo(
            reason="eos" if generated and generated[-1] == EOS_ID else "length_limit",
            generated_tokens=len(generated),
        ),
        timing=TimingInfo(started_at=created_at, completed_at=created_at, duration_ms=0.0),
        provenance=Provenance(
            origin="generated",
            source_system="tests.support.tiny_lm",
            source_version="1.0.0",
        ),
        privacy=PrivacyInfo(store_input_text=True, store_output_text=True),
    )
    return finalize_trace(trace, budget or Budget())


def perplexity_style_entropy(logits: torch.Tensor) -> float:
    """Small helper so tests can assert the fixture is not degenerate."""

    probabilities = torch.softmax(logits.to(torch.float64), dim=-1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
    return float(entropy.mean().item())


__all__ = [
    "BOS_ID",
    "EOS_ID",
    "PAD_ID",
    "SPECIAL_IDS",
    "UNK_ID",
    "CausalSelfAttention",
    "CharTokenizer",
    "SyntheticModelSpec",
    "SyntheticRuntime",
    "TinyBlock",
    "TinyRuntime",
    "TinyCausalLM",
    "build_runtime",
    "greedy_generation",
    "make_trace",
    "perplexity_style_entropy",
    "teacher_forced_logits",
    "token_fingerprint",
]
