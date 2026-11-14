"""Tensor-region selection and the four supported intervention operations.

Everything here operates on a *clone* of the activation tensor.  The original
tensor may be shared with a cache or another hook, so in-place modification is
never performed.

The selection rules are intentionally strict:

* positions index the sequence axis (``1`` for ``[batch, seq, hidden]`` and
  ``0`` for ``[seq, hidden]``); a position outside the current sequence is an
  error, not a wrap-around;
* feature selection is the last axis;
* a structured module output is only intervened on when the caller named the
  tuple index or mapping key, or when the output is a single tensor;
* a replacement must match the selected region exactly, except for the documented
  mean-ablation broadcast across selected positions.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch

from wsp.core.dtypes import torch_dtype_name
from wsp.core.errors import CompatibilityError, ExecutionError, UnsupportedError
from wsp.interventions.models import SliceStats, TensorSelector


@dataclass(frozen=True)
class Selection:
    """A validated tensor region on one concrete activation tensor."""

    axis: int
    rows: list[int]
    feature_start: int | None
    feature_end: int | None
    feature_indices: list[int]
    length: int
    width: int
    note: str | None = None


def _as_tensor(value: Any, what: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise UnsupportedError(
            f"{what} is not a tensor",
            details={"type": type(value).__name__},
        )
    return value


def selection_length(tensor: torch.Tensor, axis: int) -> int:
    return int(tensor.shape[axis])


def resolve_selection(
    tensor: torch.Tensor,
    selector: TensorSelector,
    *,
    sequence_length: int,
) -> Selection:
    """Validate that ``selector`` names a well-defined region of ``tensor``."""

    if tensor.ndim == 0:
        raise UnsupportedError("a scalar activation cannot be targeted")
    if selector.expected_shape is not None and list(tensor.shape) != list(
        selector.expected_shape
    ):
        raise CompatibilityError(
            "the activation shape does not match the shape declared by the intervention",
            details={
                "expected": list(selector.expected_shape),
                "observed": [int(item) for item in tensor.shape],
            },
        )
    if selector.expected_dtype is not None:
        observed = torch_dtype_name(tensor)
        if observed != selector.expected_dtype:
            raise CompatibilityError(
                "the activation dtype does not match the dtype declared by the intervention",
                details={"expected": selector.expected_dtype, "observed": observed},
            )

    note = None
    if tensor.ndim == 1:
        axis = 0
        if list(selector.positions) != [0]:
            raise UnsupportedError(
                "a 1-D activation has no sequence axis; use positions=[0] to target it"
            )
        length = 1
        note = "1-D activation targeted as a single vector"
    elif selector.axis is not None:
        axis = int(selector.axis)
    elif tensor.ndim == 3:
        axis = 1
        note = "sequence axis inferred as 1 for a rank-3 activation"
    elif tensor.ndim == 2:
        axis = 0
        note = "sequence axis inferred as 0 for a rank-2 activation"
    else:
        raise UnsupportedError(
            "the workbench cannot infer a sequence axis for this activation rank; "
            "declare selector.axis explicitly",
            details={"shape": [int(item) for item in tensor.shape], "rank": tensor.ndim},
        )

    if axis >= tensor.ndim:
        raise CompatibilityError(
            "the declared selection axis is outside the activation rank",
            details={"axis": axis, "rank": tensor.ndim},
        )
    if tensor.ndim > 1 and axis == tensor.ndim - 1:
        raise UnsupportedError(
            "the sequence axis cannot also be the feature axis; a feature selection is "
            "expressed separately"
        )

    if tensor.ndim > 1:
        length = int(tensor.shape[axis])
        if length != sequence_length:
            raise CompatibilityError(
                "the selected axis length does not equal the recorded sequence length, "
                "so token positions would be ambiguous",
                details={
                    "axis": axis,
                    "axis_length": length,
                    "sequence_length": sequence_length,
                    "shape": [int(item) for item in tensor.shape],
                },
            )
        if selector.positions[-1] >= length:
            raise CompatibilityError(
                "a selected position lies outside the activation sequence",
                details={"position": selector.positions[-1], "length": length},
            )
        width = int(tensor.shape[-1])
    else:
        width = int(tensor.shape[0])

    if selector.feature_indices:
        if selector.feature_indices[-1] >= width:
            raise CompatibilityError(
                "a selected feature index lies outside the activation width",
                details={"index": selector.feature_indices[-1], "width": width},
            )
        region_width = len(selector.feature_indices)
    elif selector.feature_start is not None and selector.feature_end is not None:
        if selector.feature_end > width:
            raise CompatibilityError(
                "the selected feature range lies outside the activation width",
                details={
                    "feature_start": selector.feature_start,
                    "feature_end": selector.feature_end,
                    "width": width,
                },
            )
        region_width = selector.feature_end - selector.feature_start
    else:
        region_width = width

    return Selection(
        axis=axis,
        rows=list(selector.positions),
        feature_start=selector.feature_start,
        feature_end=selector.feature_end,
        feature_indices=list(selector.feature_indices),
        length=length,
        width=region_width,
        note=note,
    )


def selected_region(tensor: torch.Tensor, selection: Selection) -> torch.Tensor:
    """Return the selected slice as a fresh tensor."""

    if tensor.ndim == 1:
        return tensor[_feature_index(selection)].clone()
    base = tensor.index_select(
        selection.axis, torch.tensor(selection.rows, dtype=torch.long, device="cpu").to(tensor.device)
    )
    return base[..., _feature_index(selection)].clone()


def _feature_index(selection: Selection) -> Any:
    if selection.feature_indices:
        return selection.feature_indices
    if selection.feature_start is not None and selection.feature_end is not None:
        return slice(selection.feature_start, selection.feature_end)
    return slice(None)


def slice_stats(values: torch.Tensor) -> SliceStats:
    flat = values.detach().to("cpu", torch.float64).reshape(-1)
    count = int(flat.numel())
    if count == 0:
        return SliceStats(count=0, finite=True)
    finite_mask = torch.isfinite(flat)
    finite = bool(finite_mask.all().item())
    usable = flat[finite_mask]
    if usable.numel() == 0:
        return SliceStats(count=count, finite=False)
    return SliceStats(
        count=count,
        mean=float(usable.mean().item()),
        std=float(usable.std(unbiased=False).item()),
        minimum=float(usable.min().item()),
        maximum=float(usable.max().item()),
        l2_norm=float(torch.linalg.vector_norm(usable).item()),
        finite=finite,
    )


def compare_slices(
    before: torch.Tensor, after: torch.Tensor, positions: Sequence[int]
) -> tuple[SliceStats, SliceStats, float | None, float | None, float | None, float | None]:
    """Return stats plus max/mean absolute difference, relative L2, and cosine."""

    left = before.detach().to("cpu", torch.float64).reshape(-1)
    right = after.detach().to("cpu", torch.float64).reshape(-1)
    max_abs = mean_abs = relative = cosine = None
    if left.numel() == right.numel() and left.numel() > 0:
        delta = right - left
        max_abs = float(delta.abs().max().item())
        mean_abs = float(delta.abs().mean().item())
        denominator = float(torch.linalg.vector_norm(left).item())
        if denominator > 0:
            relative = float(torch.linalg.vector_norm(delta).item() / denominator)
        norm_left = denominator
        norm_right = float(torch.linalg.vector_norm(right).item())
        if norm_left > 0 and norm_right > 0:
            cosine = float(torch.dot(left, right).item() / (norm_left * norm_right))
    return (
        slice_stats(before),
        slice_stats(after),
        max_abs,
        mean_abs,
        relative,
        cosine,
    )


def _validate_replacement(
    replacement: torch.Tensor,
    selection: Selection,
    *,
    operation: Literal["zero", "mean", "patch", "identity"],
    allow_dtype_cast: bool,
    allow_position_broadcast: bool,
    tensor_dtype: torch.dtype,
) -> tuple[torch.Tensor, str | None]:
    """Check finiteness, shape compatibility, and any dtype conversion."""

    if operation in ("zero", "identity"):
        return replacement, None
    if not torch.isfinite(replacement).all().item():
        raise CompatibilityError(
            "the replacement region contains non-finite values",
            details={"operation": operation},
        )
    if operation == "mean":
        if replacement.ndim == 1:
            replacement = replacement.unsqueeze(0)
        if replacement.shape[0] != 1:
            raise CompatibilityError(
                "a mean ablation vector must describe one feature row",
                details={"shape": [int(item) for item in replacement.shape]},
            )
        if int(replacement.shape[1]) != selection.width:
            raise CompatibilityError(
                "the mean vector width does not match the selected feature region",
                details={
                    "mean_width": int(replacement.shape[1]),
                    "region_width": selection.width,
                },
            )
        # A mean ablation replaces every selected position with the same mean.
        # This broadcast is part of the operation's definition, not an inference.
        replacement = replacement.expand(len(selection.rows), -1)
    elif operation == "patch":
        if replacement.ndim == 1:
            replacement = replacement.unsqueeze(0)
        if int(replacement.shape[1]) != selection.width:
            raise CompatibilityError(
                "donor width does not match the selected feature region",
                details={
                    "donor_width": int(replacement.shape[1]),
                    "region_width": selection.width,
                },
            )
        if replacement.shape[0] == 1 and len(selection.rows) > 1:
            if not allow_position_broadcast:
                raise CompatibilityError(
                    "one donor row cannot be broadcast across several selected positions "
                    "without explicit authorization",
                    details={
                        "donor_rows": 1,
                        "selected_positions": len(selection.rows),
                        "hint": "set donor.allow_position_broadcast=true to accept the broadcast",
                    },
                )
            replacement = replacement.expand(len(selection.rows), -1)
        elif replacement.shape[0] != len(selection.rows):
            raise CompatibilityError(
                "donor rows do not match the number of selected positions",
                details={
                    "donor_rows": int(replacement.shape[0]),
                    "selected_positions": len(selection.rows),
                },
            )

    dtype_cast = None
    if replacement.dtype != tensor_dtype:
        if not allow_dtype_cast:
            raise CompatibilityError(
                "the replacement dtype differs from the activation dtype; "
                "set allow_dtype_cast to accept the recorded precision change",
                details={
                    "replacement": torch_dtype_name(replacement),
                    "activation": torch_dtype_name(tensor_dtype),
                },
            )
        dtype_cast = f"{torch_dtype_name(replacement)}->{torch_dtype_name(tensor_dtype)}"
        replacement = replacement.to(tensor_dtype)
    return replacement.contiguous(), dtype_cast


def apply_operation(
    tensor: torch.Tensor,
    selection: Selection,
    *,
    operation: Literal["zero", "mean", "patch", "identity"],
    replacement: torch.Tensor | None = None,
    allow_dtype_cast: bool = False,
    allow_position_broadcast: bool = False,
) -> tuple[torch.Tensor, int, str | None, torch.Tensor, torch.Tensor]:
    """Return ``(new_tensor, elements_modified, dtype_cast, before, after)``."""

    before = selected_region(tensor, selection)
    updated = tensor.clone()
    if operation == "identity":
        after = before.clone()
        return updated, 0, None, before, after

    if operation == "zero":
        # ``before`` keeps the full tensor shape (including a batch axis), so it
        # cannot be indexed by selected-row number.  Build the replacement with
        # exactly one row per selected position instead.
        replacement_tensor = torch.zeros(
            (len(selection.rows), selection.width),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dtype_cast = None
    else:
        if replacement is None:
            raise CompatibilityError(f"operation '{operation}' requires a replacement tensor")
        replacement_tensor, dtype_cast = _validate_replacement(
            replacement,
            selection,
            operation=operation,
            allow_dtype_cast=allow_dtype_cast,
            allow_position_broadcast=allow_position_broadcast,
            tensor_dtype=tensor.dtype,
        )

    if tensor.ndim == 1:
        feature_index = _feature_index(selection)
        if isinstance(feature_index, slice):
            updated[feature_index] = replacement_tensor.reshape(-1)
        else:
            updated[feature_index] = replacement_tensor.reshape(-1)
    else:
        feature_index = _feature_index(selection)
        for row_index, row in enumerate(selection.rows):
            view = updated.select(selection.axis, row)
            if isinstance(feature_index, slice):
                view[feature_index] = replacement_tensor[row_index]
            else:
                view[feature_index] = replacement_tensor[row_index]
    after = selected_region(updated, selection)
    elements = int(after.numel())
    return updated, elements, dtype_cast, before, after


# --------------------------------------------------------------------------- #
# structured outputs and inputs
# --------------------------------------------------------------------------- #


def _single_tensor(items: Sequence[Any]) -> int | None:
    indices = [index for index, item in enumerate(items) if isinstance(item, torch.Tensor)]
    if len(indices) == 1:
        return indices[0]
    return None


def extract_activation(
    value: Any,
    selector: TensorSelector,
    *,
    role: Literal["input", "output"],
) -> tuple[torch.Tensor, Any]:
    """Locate the target tensor inside an input/output structure.

    Returns the tensor and a token identifying where it was found so the caller
    can rebuild the same structure with the replaced tensor.
    """

    if isinstance(value, torch.Tensor):
        return value, ("tensor",)
    if isinstance(value, (tuple, list)):
        index = selector.tensor_index
        if index is None:
            index = _single_tensor(value)
        if index is None:
            raise UnsupportedError(
                f"the module {role} is a sequence containing several tensors; "
                "name the intended element with selector.tensor_index",
                details={"length": len(value)},
            )
        if index >= len(value):
            raise CompatibilityError(
                "selector.tensor_index is outside the module output",
                details={"index": index, "length": len(value)},
            )
        tensor, _ = extract_activation(value[index], selector, role=role)
        return tensor, ("sequence", index)
    if isinstance(value, dict):
        if selector.tensor_name is None:
            raise UnsupportedError(
                f"the module {role} is a mapping; name the intended tensor with "
                "selector.tensor_name"
            )
        if selector.tensor_name not in value:
            raise CompatibilityError(
                "selector.tensor_name is not a key of the module output",
                details={"key": selector.tensor_name, "keys": sorted(map(str, value.keys()))},
            )
        tensor, _ = extract_activation(value[selector.tensor_name], selector, role=role)
        return tensor, ("mapping", selector.tensor_name)
    raise UnsupportedError(
        f"the module {role} is not a tensor or a supported container",
        details={"type": type(value).__name__},
    )


def rebuild_activation(
    value: Any,
    location: Any,
    replacement: torch.Tensor,
) -> Any:
    """Rebuild a structure with exactly one tensor replaced.

    The original container type is preserved when it can be rebuilt without
    guessing (tuples, lists, named tuples, and mutable mappings).  None of the
    ModelOutput subclasses used by Hugging Face are frozen, so a shallow copy
    plus one assignment keeps their downstream attribute access working.
    """

    kind = location[0]
    if kind == "tensor":
        return replacement
    if kind == "sequence":
        if not isinstance(value, (tuple, list)):
            raise UnsupportedError(
                "internal error: the selected activation is not held in a sequence"
            )
        index = int(location[1])
        rebuilt = list(value)
        rebuilt[index] = _replace_in_child(rebuilt[index], replacement)
        if isinstance(value, tuple):
            fields = getattr(value, "_fields", None)
            if fields is not None:
                return type(value)(*rebuilt)
            return tuple(rebuilt)
        return rebuilt
    if kind == "mapping":
        if not hasattr(value, "keys"):
            raise UnsupportedError(
                "internal error: the selected activation is not held in a mapping"
            )
        key = location[1]
        rebuilt = copy.copy(value)
        try:
            rebuilt[key] = _replace_in_child(value[key], replacement)
        except (TypeError, AttributeError) as error:  # pragma: no cover - exotic containers
            raise UnsupportedError(
                "the module output mapping cannot be rebuilt with a replaced value",
                details={"type": type(value).__name__, "error": str(error)},
            ) from error
        return rebuilt
    raise UnsupportedError(f"unsupported activation location {kind!r}")


def _replace_in_child(child: Any, replacement: torch.Tensor) -> Any:
    if isinstance(child, torch.Tensor):
        return replacement
    # Nested containers are not searched: ``extract_activation`` only descends
    # one level and only when the schema names the element, so the location is
    # always direct.
    raise UnsupportedError("internal error: nested activation replacement is not supported")


def finite_or_raise(tensor: torch.Tensor, *, what: str) -> None:
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return
    count = int((~finite).sum().item())
    raise ExecutionError(
        f"{what} contains non-finite values",
        details={"non_finite": count, "numel": int(tensor.numel())},
    )


def vector_values(values: Any, *, what: str) -> torch.Tensor:
    """Convert a JSON list or numpy array to a float64 vector, rejecting NaN."""

    tensor = torch.as_tensor(values, dtype=torch.float64)
    if not torch.isfinite(tensor).all().item():
        raise CompatibilityError(f"{what} contains non-finite values")
    return tensor


def estimated_elements(selection_rows: int, width: int) -> int:
    return max(0, int(selection_rows) * int(width))


def cosine_projection(delta: torch.Tensor, direction: torch.Tensor) -> float | None:
    if delta.numel() != direction.numel():
        raise CompatibilityError(
            "the projection direction length does not match the activation region",
            details={"delta": int(delta.numel()), "direction": int(direction.numel())},
        )
    if not math.isfinite(float(direction.abs().sum().item())):
        raise CompatibilityError("the projection direction contains non-finite values")
    return float(torch.dot(delta.to(torch.float64), direction.to(torch.float64)).item())


__all__ = [
    "Selection",
    "apply_operation",
    "compare_slices",
    "cosine_projection",
    "estimated_elements",
    "extract_activation",
    "finite_or_raise",
    "rebuild_activation",
    "resolve_selection",
    "selected_region",
    "selection_length",
    "slice_stats",
    "vector_values",
]
