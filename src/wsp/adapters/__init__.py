"""Model adapters.  Generic PyTorch modules remain the foundation."""

from wsp.adapters.huggingface import HuggingFaceCausalLMAdapter, load_local_causal_lm

__all__ = ["HuggingFaceCausalLMAdapter", "load_local_causal_lm"]
