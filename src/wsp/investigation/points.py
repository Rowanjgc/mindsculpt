"""Discover addressable capture points in a loaded model.

Point discovery is deliberately architecture-agnostic: it walks
``named_modules`` and classifies by name and class, so hybrid stacks (Gated
DeltaNet, Mamba-style mixers, plain attention) all appear without the capture
code knowing anything about them.
"""

from __future__ import annotations

from typing import Any, Iterable

from wsp.core.budget import Budget
from wsp.core.errors import InputError
from wsp.investigation.models import CapturePoint, CapturePointList

_MAX_CHILDREN = 32

_ATTENTION = ("attention", "attn", "self_attn", "cross_attn", "q_proj", "k_proj", "v_proj", "o_proj", "qkv")
_LINEAR_ATTENTION = ("linearattention", "linear_attention", "linear_attn", "gateddeltanet", "gated_delta", "deltanet", "selective_scan", "mamba", "ssm")
_RECURRENT = ("recurrent", "mixer", "conv1d", "causal_conv", "state_update", "gated_delta_rule")
_MLP = ("mlp", "feed_forward", "feedforward", "ffn", "gate_proj", "up_proj", "down_proj", "swiglu")
_NORM = ("norm", "layernorm", "rmsnorm", "batchnorm", "groupnorm")
_EMBEDDING = ("embed", "embedding", "wte", "wpe", "embed_tokens")
_HEAD = ("lm_head", "output_proj", "classifier")
_BLOCK = ("block", "layer", "decoderlayer", "encoderlayer", "transformerlayer")


def torch_module_of(model: Any) -> Any:
    """Return the underlying ``nn.Module`` for any accepted model object."""

    for attribute in ("module", "model"):
        candidate = getattr(model, attribute, None)
        if candidate is not None and hasattr(candidate, "named_modules"):
            return candidate
    if hasattr(model, "named_modules"):
        return model
    raise InputError("the supplied object is not a torch module and cannot expose capture points")


def _classify(path: str, class_name: str) -> str:
    lowered = path.lower()
    class_lowered = class_name.lower()
    haystack = f"{lowered} {class_lowered}"
    if any(marker in haystack for marker in _LINEAR_ATTENTION):
        return "linear-attention"
    if any(marker in haystack for marker in _RECURRENT):
        return "recurrent"
    # Norms come before attention on purpose: hybrid stacks name the
    # normalisation that precedes an attention block
    # ``post_attention_layernorm``, and it is a norm, not an attention module.
    if any(marker in lowered for marker in _NORM):
        return "norm"
    if any(marker in lowered for marker in _EMBEDDING):
        return "embedding"
    if any(marker in lowered for marker in _HEAD):
        return "head"
    if any(marker in lowered for marker in _MLP):
        return "mlp"
    if any(marker in lowered for marker in _ATTENTION):
        return "attention"
    if any(marker in lowered for marker in _BLOCK):
        return "block"
    if "aggregate" in haystack:
        return "aggregate"
    return "other"


#: The canonical kind labels produced by :func:`_classify`.  Exposed through
#: ``list_capture_points`` so callers never have to guess the taxonomy.
CAPTURE_KIND_LABELS = (
    "attention",
    "linear-attention",
    "recurrent",
    "norm",
    "embedding",
    "head",
    "mlp",
    "block",
    "aggregate",
    "other",
)


def _parameter_count(module: Any) -> tuple[int, int]:
    parameters = list(module.parameters()) if hasattr(module, "parameters") else []
    return int(sum(int(p.numel()) for p in parameters)), len(parameters)


def discover_points(
    module: Any,
    *,
    include_submodules: bool = True,
    kinds: Iterable[str] | None = None,
    limit: int = 200,
    budget: Budget | None = None,
) -> CapturePointList:
    """Enumerate capture points, bounded by ``limit``."""

    if limit <= 0:
        raise InputError("limit must be positive")
    settings = budget or Budget()
    cap = min(int(limit), int(settings.max_capture_artifacts) * 16)
    wanted = set(kinds) if kinds else None

    named = list(module.named_modules())
    total = len(named)
    children: dict[str, int] = {}
    for name, _ in named:
        if not name:
            continue
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        children[parent] = children.get(parent, 0) + 1

    points: list[CapturePoint] = []
    truncated = False
    seen_kinds: set[str] = set()
    for name, child in named:
        if not include_submodules and name:
            break
        path = name or "<root>"
        class_name = type(child).__name__
        kind = _classify(name, class_name)
        seen_kinds.add(kind)
        if wanted is not None and kind not in wanted:
            continue
        parameter_count, tensor_count = _parameter_count(child)
        child_count = int(children.get(name, 0))
        notes: list[str] = []
        if tensor_count == 0 and child_count == 0:
            notes.append("module has no parameters; capture may be empty or non-numeric")
        if len(points) >= cap:
            truncated = True
            break
        points.append(
            CapturePoint(
                path=path,
                class_name=class_name,
                kind=kind,
                depth=0 if not name else name.count(".") + 1,
                parameter_count=parameter_count,
                numel=parameter_count,
                is_leaf=child_count == 0,
                child_count=child_count,
                notes=notes,
            )
        )

    config = getattr(module, "config", None)
    architecture = None
    for attribute in ("architectures", "model_type"):
        value = getattr(config, attribute, None)
        if isinstance(value, list) and value:
            architecture = str(value[0])
            break
        if isinstance(value, str) and value:
            architecture = value
            break

    notes = [
        "capture points are module paths accepted by capture_trace_activations",
        "a module may be called more than once per forward pass; call_index distinguishes calls",
    ]
    if truncated:
        notes.append(f"listing truncated at {cap} points; raise limit to see more")
    if wanted is not None and not points:
        notes.append(
            "no capture points matched kinds "
            + ", ".join(sorted(wanted))
            + "; canonical kind labels are "
            + ", ".join(CAPTURE_KIND_LABELS)
        )
    if not include_submodules and not points:
        notes.append(
            "include_submodules=false lists only the model root; "
            "set include_submodules=true to enumerate submodules"
        )

    return CapturePointList(
        model_name=getattr(module, "name", None) or type(module).__name__,
        model_class=type(module).__name__,
        architecture=architecture,
        total_modules=total,
        points=points,
        truncated=truncated,
        notes=notes,
    )


def validate_modules(module: Any, paths: Iterable[str], *, budget: Budget) -> list[str]:
    """Return a canonical module path list, raising on unknown names."""

    requested = [str(path) for path in paths]
    if not requested:
        raise InputError("at least one module path is required", details={"hint": "call list_capture_points"})
    if len(requested) > int(budget.max_capture_modules):
        raise InputError(
            "too many modules requested for one capture",
            details={"requested": len(requested), "limit": int(budget.max_capture_modules)},
        )
    named = {name or "<root>" for name, _ in module.named_modules()}
    missing = [path for path in requested if path not in named]
    if missing:
        raise InputError(
            "some module paths are not present in the model",
            details={"missing": missing[:8], "hint": "call list_capture_points for valid paths"},
        )
    return sorted(dict.fromkeys(requested))
