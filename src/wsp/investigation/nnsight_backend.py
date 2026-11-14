"""Optional NNsight / nnterp instrumentation backend.

The core workbench only needs PyTorch: forward hooks are enough to measure a
module's output at chosen token positions.  NNsight and nnterp add two things
that are genuinely useful for transformer work:

* **normalised access** — ``layers[3].self_attn`` means the same thing across
  architectures, instead of ``model.layers.3.self_attn`` versus
  ``transformer.h.3.attn``;
* **hybrid-architecture awareness** — nnterp knows that a Qwen3.5 block may hold
  a gated DeltaNet mixer (``linear_attn``) instead of softmax attention
  (``self_attn``), and gives both a stable name.

This module is deliberately isolated: nothing else in the package imports it,
it is never imported at package import time, and every entry point raises
:class:`~wsp.core.errors.UnsupportedError` with an install hint when the optional
dependencies are missing.  It never substitutes a different backend silently —
the recorded evidence carries ``backend="nnterp"``.

What the adapter is *not*: nnterp is not the canonical representation of WSP
activation evidence.  Measurements are built by
:class:`wsp.investigation.measure.MeasurementBuilder`, the same code the hook
backend uses, so a capture from either backend is comparable field for field.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from wsp.activations.capture import _iter_tensors
from wsp.core.budget import Budget
from wsp.core.errors import (
    CompatibilityError,
    ExecutionError,
    InputError,
    UnsupportedError,
    WspError,
)
from wsp.core.time import utc_now_iso
from wsp.investigation.measure import MeasurementBuilder
from wsp.investigation.models import (
    SCHEMA_POINTS,
    CaptureMode,
    CapturePoint,
    CapturePointList,
    Measurement,
)
from wsp.investigation.points import _classify
from wsp.replay.backends import (
    HuggingFaceReplayModel,
    structure_fingerprint,
    TokenizationSupport,
)
from wsp.replay.models import HYBRID_ARCHITECTURE_NOTE, ReplayModelInfo

BACKEND_NAME = "nnterp"
PINNED_NNSIGHT = "0.8.2"
PINNED_NNTERP = "2.0.0rc1"
INSTALL_HINT = (
    'uv pip install "nnsight==%s" "nnterp==%s"' % (PINNED_NNSIGHT, PINNED_NNTERP)
)

#: Sublayer names nnterp standardises, in the order they run inside one block.
SUBTLAYER_ORDER = (
    "input_layernorm",
    "self_attn",
    "linear_attn",
    "post_attention_layernorm",
    "mlp",
)
_LAYER_OUTPUT_RANK = 4
_ROOT_POINTS = ("embed_tokens", "norm", "lm_head")

ADAPTER_CAVEATS = (
    "capture runs through NNsight tracing and nnterp's name normalisation, which are "
    "optional dependencies; the CPU-only core does not require them",
    "nnterp resolves a checkpoint to a family module and refuses architectures it does "
    "not cover, rather than guessing at a module layout",
    "attention modules must run with the eager attention implementation for nnterp to "
    "expose their outputs; the workbench loads the model that way for this backend",
    "measurements are built by the shared MeasurementBuilder, so they are comparable "
    "with hook-based captures of the same model and positions",
)


# --------------------------------------------------------------------------- deps


def optional_versions() -> dict[str, Any]:
    """Report what is installed without importing either package."""

    def _version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    nnsight = _version("nnsight")
    nnterp = _version("nnterp")
    return {
        "nnsight": nnsight,
        "nnterp": nnterp,
        "available": bool(nnsight and nnterp),
        "pinned": {"nnsight": PINNED_NNSIGHT, "nnterp": PINNED_NNTERP},
        "install_hint": INSTALL_HINT,
    }


def _require_optional() -> tuple[Any, Any]:
    try:
        nnsight = importlib.import_module("nnsight")
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise UnsupportedError(
            "the nnterp instrumentation backend needs the optional 'nnsight' package",
            details={"install": INSTALL_HINT, "backend": BACKEND_NAME},
        ) from exc
    try:
        nnterp = importlib.import_module("nnterp")
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise UnsupportedError(
            "the nnterp instrumentation backend needs the optional 'nnterp' package",
            details={"install": INSTALL_HINT, "backend": BACKEND_NAME},
        ) from exc
    return nnsight, nnterp


def _backend_version() -> str:
    versions = optional_versions()
    return "nnterp %s / nnsight %s" % (versions["nnterp"], versions["nnsight"])


# ---------------------------------------------------------------- path handling


def parse_standard_path(path: str) -> tuple[int | None, str]:
    """Split a standardised capture path into (layer index, name)."""

    if not isinstance(path, str) or not path.strip():
        raise InputError("capture paths must be non-empty strings")
    tokens = [token.strip() for token in path.split(".")]
    if any(not token for token in tokens):
        raise InputError(f"capture path {path!r} is malformed")
    if tokens[0] in _ROOT_POINTS and len(tokens) == 1:
        return None, tokens[0]
    if tokens[0] != "layers" or len(tokens) < 2:
        raise InputError(
            f"capture path {path!r} is not a standardised path",
            details={
                "hint": "use embed_tokens, norm, lm_head, layers.<i>, or layers.<i>.<sublayer>",
                "sublayers": list(SUBTLAYER_ORDER),
            },
        )
    try:
        index = int(tokens[1])
    except ValueError as exc:
        raise InputError(
            f"capture path {path!r} does not name a layer index",
            details={"hint": "layers.<i> takes an integer layer index"},
        ) from exc
    if index < 0:
        raise InputError(f"capture path {path!r} has a negative layer index")
    if len(tokens) == 2:
        return index, ""
    if len(tokens) > 3:
        raise InputError(
            f"capture path {path!r} is deeper than this backend supports",
            details={"hint": "capture a block or one of its sublayers, not a sublayer's child"},
        )
    sublayer = tokens[2]
    if sublayer not in SUBTLAYER_ORDER:
        raise InputError(
            f"capture path {path!r} names an unknown sublayer {sublayer!r}",
            details={"sublayers": list(SUBTLAYER_ORDER)},
        )
    return index, sublayer


def _execution_rank(path: str) -> tuple[int, int, int]:
    """Sort key that follows one forward pass through a transformer stack."""

    layer, sublayer = parse_standard_path(path)
    if layer is None:
        return (0, 0, 0) if path == "embed_tokens" else (2, 0, 0)
    if not sublayer:
        return (1, layer, _LAYER_OUTPUT_RANK)
    return (1, layer, SUBTLAYER_ORDER.index(sublayer))


def _point_kind(path: str, class_name: str) -> str:
    return _classify(path, class_name)


# ------------------------------------------------------------------ model wrapper


@dataclass
class _ResolvedPoint:
    path: str
    envoy: Any
    class_name: str
    rank: tuple[int, int, int]


class NnterpReplayModel:
    """A locally loaded causal LM exposed through nnterp's standardised names.

    Implements the same replay protocol as the plain PyTorch and Hugging Face
    adapters, so the replay engine, verification checks, and capture record are
    unchanged.  Only the *capture* path differs: activations are read inside an
    NNsight trace instead of from forward hooks.
    """

    def __init__(
        self,
        module: Any,
        tokenizer: Any,
        *,
        name: str | None = None,
        model_path: str | None = None,
        device: str = "cpu",
        dtype: str | None = None,
        compute_file_sha256: bool = True,
    ) -> None:
        if not isinstance(module, torch.nn.Module):
            raise InputError("the nnterp backend needs a loaded torch module")
        self.module = module
        self.tokenizer = tokenizer
        self.model_path = model_path or getattr(module, "_wsp_model_path", "<memory>")
        self.name = name
        self.device = device
        self.dtype = dtype
        self._hf = HuggingFaceReplayModel(
            model=module,
            tokenizer=tokenizer,
            model_path=self.model_path,
            device=device,
            dtype=dtype,
            compute_file_sha256=compute_file_sha256,
        )
        self._standardized: Any | None = None
        self._pending_capture: "NnterpCapture | None" = None
        self._info: ReplayModelInfo | None = None

    # -- construction ------------------------------------------------------

    def _build_standardized(self) -> Any:
        _, nnterp = _require_optional()
        config = getattr(self.module, "config", None)
        implementation = getattr(config, "_attn_implementation", None)
        if implementation not in (None, "eager"):
            raise CompatibilityError(
                "the nnterp backend needs the model loaded with the eager attention implementation",
                details={
                    "attn_implementation": str(implementation),
                    "hint": "load the model through the nnterp backend so that eager attention is requested",
                },
            )
        try:
            standardized = nnterp.StandardizedTransformer(
                self.module,
                tokenizer=self.tokenizer,
                task="text-generation",
            )
        except WspError:
            raise
        except Exception as exc:  # pragma: no cover - depends on nnterp internals
            raise UnsupportedError(
                "nnterp could not standardise this model architecture",
                details={
                    "model_class": type(self.module).__name__,
                    "error": f"{type(exc).__name__}: {exc}",
                    "hint": "use the default torch-hooks instrumentation for this model",
                },
            ) from exc
        return standardized

    @property
    def standardized(self) -> Any:
        if self._standardized is None:
            self._standardized = self._build_standardized()
        return self._standardized

    # -- describe ----------------------------------------------------------

    def describe(self) -> ReplayModelInfo:
        if self._info is not None:
            return self._info
        base = self._hf.describe()
        capabilities = dict(base.capabilities)
        capabilities.update(
            {
                "standardized_paths": True,
                "instrumentation": BACKEND_NAME,
            }
        )
        caveats = list(base.caveats) + [ADAPTER_CAVEATS[0], ADAPTER_CAVEATS[1]]
        runtime = dict(base.runtime)
        versions = optional_versions()
        if versions["nnsight"]:
            runtime["nnsight"] = str(versions["nnsight"])
        if versions["nnterp"]:
            runtime["nnterp"] = str(versions["nnterp"])
        model_type = getattr(getattr(self.module, "config", None), "model_type", None)
        if model_type:
            runtime["nnterp_config"] = str(model_type)
        info = base.model_copy(
            update={
                "name": self.name or base.name,
                "backend": BACKEND_NAME,
                "backend_version": _backend_version(),
                "structure_fingerprint": structure_fingerprint(self.module),
                "capabilities": capabilities,
                "caveats": caveats,
                "runtime": runtime,
            }
        )
        self._info = info
        return info

    # -- forward -----------------------------------------------------------

    def forward_token_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        if not token_ids:
            raise ExecutionError("cannot replay an empty token sequence")
        standardized = self.standardized
        capture = self._pending_capture
        sink: dict[str, Any] = {}
        tensor = torch.tensor([list(token_ids)], dtype=torch.long)
        try:
            with standardized.trace(tensor):
                if capture is not None and capture.active:
                    capture.issue_requests(standardized, sink)
                sink["__logits__"] = standardized.logits.save()
        except WspError:
            raise
        except Exception as exc:
            raise ExecutionError(
                f"nnterp replay forward failed: {type(exc).__name__}: {exc}"
            ) from exc
        logits = sink.pop("__logits__", None)
        if capture is not None and capture.active:
            capture.consume(sink)
        if not isinstance(logits, torch.Tensor):
            raise ExecutionError(
                "the nnterp trace did not return logits",
                details={"logits_type": type(logits).__name__},
            )
        return logits

    # -- tokenizer ---------------------------------------------------------

    def tokenizer_support(self) -> TokenizationSupport:
        return self._hf.tokenizer_support()

    # -- capture points ----------------------------------------------------

    def _resolve(self, path: str) -> _ResolvedPoint:
        layer, sublayer = parse_standard_path(path)
        standardized = self.standardized
        if layer is None:
            envoy = getattr(standardized, path, None)
            if envoy is None:
                raise InputError(
                    f"this model has no standardised {path!r} module",
                    details={"hint": "call list_capture_points with instrumentation='nnterp'"},
                )
            class_name = _module_class_name(envoy)
            return _ResolvedPoint(path, envoy, class_name, _execution_rank(path))
        try:
            layer_envoy = standardized.layers[layer]
        except (IndexError, KeyError) as exc:
            raise InputError(
                f"layer {layer} is outside this model",
                details={"layers": len(standardized.layers)},
            ) from exc
        envoy = layer_envoy if not sublayer else getattr(layer_envoy, sublayer, None)
        if envoy is None:
            raise InputError(
                f"layer {layer} has no {sublayer!r} sublayer in this architecture",
                details={
                    "hint": "call list_capture_points with instrumentation='nnterp'",
                    "sublayers": _present_sublayers(layer_envoy),
                },
            )
        class_name = _module_class_name(envoy) if not sublayer else _sublayer_class_name(layer_envoy, sublayer)
        return _ResolvedPoint(path, envoy, class_name, _execution_rank(path))

    def resolve_paths(self, paths: Iterable[str]) -> list[_ResolvedPoint]:
        resolved = [self._resolve(str(path)) for path in paths]
        if not resolved:
            raise InputError("at least one capture module is required")
        return sorted(resolved, key=lambda item: item.rank)

    def capture_points(self, *, limit: int = 200, budget: Budget | None = None) -> CapturePointList:
        settings = budget or Budget()
        standardized = self.standardized
        config = getattr(self.module, "config", None)
        text_config = getattr(config, "text_config", None) or config
        layers = list(getattr(standardized, "layers", []) or [])
        points: list[CapturePoint] = []
        for name in ("embed_tokens", "norm", "lm_head"):
            envoy = getattr(standardized, name, None)
            if envoy is None:
                continue
            class_name = _module_class_name(envoy)
            parameter_count, numel = _param_counts(envoy)
            points.append(
                CapturePoint(
                    path=name,
                    class_name=class_name,
                    kind=_point_kind(name, class_name),  # type: ignore[arg-type]
                    depth=0,
                    parameter_count=parameter_count,
                    numel=numel,
                    is_leaf=True,
                    notes=["standardised root module"],
                )
            )
        for index, layer_envoy in enumerate(layers):
            class_name = _module_class_name(layer_envoy)
            parameter_count, numel = _param_counts(layer_envoy)
            points.append(
                CapturePoint(
                    path=f"layers.{index}",
                    class_name=class_name,
                    kind="block",
                    depth=1,
                    parameter_count=parameter_count,
                    numel=numel,
                    notes=["residual stream after this block"],
                )
            )
            for sublayer in SUBTLAYER_ORDER:
                if not hasattr(layer_envoy, sublayer):
                    continue
                sub_class = _sublayer_class_name(layer_envoy, sublayer)
                sub_params, sub_numel = _param_counts(getattr(layer_envoy, sublayer, None))
                points.append(
                    CapturePoint(
                        path=f"layers.{index}.{sublayer}",
                        class_name=sub_class,
                        kind=_point_kind(sublayer, sub_class),  # type: ignore[arg-type]
                        depth=2,
                        parameter_count=sub_params,
                        numel=sub_numel,
                        is_leaf=True,
                        notes=["standardised sublayer"],
                    )
                )
        cap = min(int(limit), max(1, settings.max_capture_artifacts * 16))
        truncated = len(points) > cap
        architecture: list[str] = []
        model_type = getattr(text_config, "model_type", None)
        if model_type:
            architecture.append(str(model_type))
        for value in getattr(config, "architectures", None) or []:
            architecture.append(str(value))
        notes = [
            "paths are supplied by nnterp's standardised naming, not by the checkpoint's own module names",
        ]
        if any(hasattr(layer, "linear_attn") for layer in layers):
            notes.append(HYBRID_ARCHITECTURE_NOTE)
        return CapturePointList(
            schema_version=SCHEMA_POINTS,
            model_name=self.name or getattr(config, "_name_or_path", None) or "model",
            model_class=type(self.module).__name__,
            architecture=" / ".join(dict.fromkeys(architecture)) or None,
            sequence_dim_hint="hidden states are batch-first: [batch, sequence, hidden]",
            total_modules=len(points),
            points=points[:cap],
            truncated=truncated,
            notes=notes,
        )


def _param_counts(envoy: Any) -> tuple[int, int]:
    """Parameter count and element count for an envoy's underlying module."""

    module = None
    for attribute in ("module", "_module"):
        candidate = getattr(envoy, attribute, None)
        if candidate is not None and hasattr(candidate, "parameters"):
            module = candidate
            break
    if module is None:
        return 0, 0
    tensors = [parameter for parameter in module.parameters()]
    return int(sum(int(item.numel()) for item in tensors)), int(len(tensors))


def _module_class_name(envoy: Any) -> str:
    for attribute in ("module", "_module"):
        module = getattr(envoy, attribute, None)
        if module is not None:
            return type(module).__name__
    path = getattr(envoy, "path", None)
    if isinstance(path, str) and path:
        return path.rsplit(".", 1)[-1]
    return type(envoy).__name__


def _sublayer_class_name(layer_envoy: Any, sublayer: str) -> str:
    envoy = getattr(layer_envoy, sublayer, None)
    return _module_class_name(envoy) if envoy is not None else sublayer


def _present_sublayers(layer_envoy: Any) -> list[str]:
    return [name for name in SUBTLAYER_ORDER if hasattr(layer_envoy, name)]


# ---------------------------------------------------------------- capture object


class NnterpCapture:
    """Capture selected positions through an NNsight trace.

    Presents the same small protocol as
    :class:`wsp.investigation.collector.PositionedActivationCapture`
    (``install``/``close``/``ordered_measurements``/``vectors``/``notes``), which
    is what :func:`wsp.replay.engine.execute_replay` and
    :func:`wsp.investigation.capture.run_capture` consume.  Requests are issued in
    forward-pass order because NNsight rejects a request for a module the model
    has already run past.
    """

    def __init__(
        self,
        model: NnterpReplayModel,
        module_paths: Iterable[str],
        positions: Iterable[int],
        *,
        sequence_length: int,
        mode: CaptureMode = "summary",
        record_inputs: bool = False,
        max_elements_per_tensor: int | None = None,
        max_elements_total: int | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.model = model
        self.module_paths = list(dict.fromkeys(str(path) for path in module_paths))
        self.sequence_length = int(sequence_length)
        self.mode: CaptureMode = mode
        self.record_inputs = bool(record_inputs)
        self.builder = MeasurementBuilder(
            sequence_length=self.sequence_length,
            positions=positions,
            mode=mode,
            max_elements_per_tensor=max_elements_per_tensor,
            max_elements_total=max_elements_total,
            budget=budget or Budget(),
        )
        self.positions = self.builder.positions
        self.points = model.resolve_paths(self.module_paths)
        self.notes: list[str] = self.builder.notes
        self._active = False

    # -- lifecycle ---------------------------------------------------------

    def install(self) -> None:
        if self._active:
            return
        self.model._pending_capture = self
        self._active = True

    def close(self) -> None:
        self._active = False
        if self.model._pending_capture is self:
            self.model._pending_capture = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def installed(self) -> bool:
        return self._active

    def __enter__(self) -> "NnterpCapture":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- trace requests ----------------------------------------------------

    def issue_requests(self, standardized: Any, sink: dict[str, Any]) -> None:
        for point in self.points:
            if self.record_inputs:
                sink[(point.path, "input")] = point.envoy.input.save()
            sink[(point.path, "output")] = point.envoy.output.save()

    def consume(self, sink: dict[str, Any]) -> None:
        for point in self.points:
            for role in ("input", "output") if self.record_inputs else ("output",):
                value = sink.get((point.path, role))
                if value is None:
                    self.builder.skip(
                        point.path,
                        (),
                        f"nnterp returned no {role} value for this module; nothing was measured",
                    )
                    continue
                tensors = list(_iter_tensors(value, "output"))
                if not tensors:
                    self.builder.skip(
                        point.path,
                        (),
                        f"nnterp returned no tensor for the module {role}; nothing was measured",
                    )
                    continue
                if len(tensors) > 1:
                    self.builder.skip(
                        point.path,
                        tuple(int(d) for d in tensors[0][1].shape),
                        (
                            f"the module returns {len(tensors)} tensors (for example attention "
                            "weights alongside the output); only the first was measured"
                        ),
                    )
                tensor_name, tensor = tensors[0]
                self.builder.build(
                    module=point.path,
                    class_name=point.class_name,
                    call_index=0,
                    tensor_role=role,
                    tensor_name=tensor_name,
                    tensor=tensor,
                )

    # -- delegated measurement state ---------------------------------------

    @property
    def measurements(self) -> list[Measurement]:
        return self.builder.measurements

    @property
    def vectors(self) -> dict[tuple[str, int, str, int], Any]:
        return self.builder.vectors

    @property
    def elements_measured(self) -> int:
        return self.builder.elements_measured

    @property
    def elements_retained(self) -> int:
        return self.builder.elements_retained

    @property
    def max_elements_per_tensor(self) -> int:
        return self.builder.max_elements_per_tensor

    @property
    def max_elements_total(self) -> int:
        return self.builder.max_elements_total

    def ordered_measurements(self) -> list[Measurement]:
        return self.builder.ordered_measurements(self.module_paths)


# ----------------------------------------------------------------- loading entry


def load_nnterp_model(
    path: str,
    *,
    device: str = "cpu",
    dtype: str | None = None,
    compute_file_sha256: bool = True,
    name: str | None = None,
) -> NnterpReplayModel:
    """Load a local checkpoint with eager attention and wrap it for nnterp.

    Eager attention is required because nnterp reads attention sublayer outputs
    from the eager forward.  Nothing is downloaded: the checkpoint must already
    exist in a local directory.
    """

    _require_optional()
    from wsp.replay.backends import load_local_replay_model

    base = load_local_replay_model(
        path,
        device=device,
        dtype=dtype,
        compute_file_sha256=compute_file_sha256,
        attn_implementation="eager",
    )
    wrapped = NnterpReplayModel(
        base.model,
        base.tokenizer,
        name=name or getattr(getattr(base.model, "config", None), "_name_or_path", None),
        model_path=str(path),
        device=device,
        dtype=dtype,
        compute_file_sha256=compute_file_sha256,
    )
    return wrapped


def capture_backend_capabilities() -> dict[str, Any]:
    """Describe the optional backend for tool output and documentation."""

    versions = optional_versions()
    return {
        "backend": BACKEND_NAME,
        "available": versions["available"],
        "versions": versions,
        "caveats": list(ADAPTER_CAVEATS),
        "generated_at": utc_now_iso(),
    }
