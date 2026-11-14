"""Optional local Hugging Face causal-LM adapter.

No network access and no remote code execution.  Transformers is an optional
dependency imported only when this adapter is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wsp.core.errors import InputError, SafetyError, UnsupportedError


class HuggingFaceCausalLMAdapter:
    def __init__(self, model: Any, tokenizer: Any, model_path: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path

    def prepare_text(
        self,
        text: str,
        *,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(text, str):
            raise InputError("text inputs must be strings")
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return dict(encoded)


def load_local_causal_lm(
    path: str | Path,
    *,
    device: str = "cpu",
    dtype: str | None = None,
) -> HuggingFaceCausalLMAdapter:
    """Load a causal LM from an existing local directory only."""

    local_path = Path(path).expanduser().resolve()
    if not local_path.exists() or not local_path.is_dir():
        raise SafetyError(
            "Hugging Face models must already exist in a local directory; "
            "automatic downloads are disabled",
            details={"path": str(local_path)},
        )
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise UnsupportedError(
            "transformers is not installed; install the optional 'hf' extra"
        ) from exc
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if dtype is not None:
        from wsp.core.dtypes import torch_dtype_from_name

        kwargs["torch_dtype"] = torch_dtype_from_name(dtype)
    model = AutoModelForCausalLM.from_pretrained(str(local_path), **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_path), local_files_only=True, trust_remote_code=False
    )
    model.to(torch.device(device))
    model.eval()
    return HuggingFaceCausalLMAdapter(model, tokenizer, str(local_path))
