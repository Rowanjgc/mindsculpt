"""Bounded tensor selections, resolved against concrete shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from wsp.core.errors import InputError


class AxisSlice(BaseModel):
    """A positive-step slice on one axis.  Negative steps are out of scope."""

    model_config = ConfigDict(extra="forbid")

    axis: int = Field(ge=0)
    start: int | None = Field(default=None, ge=0)
    stop: int | None = Field(default=None, ge=0)
    step: int = Field(default=1, ge=1)


class Selection(BaseModel):
    """An explicit rectangular selection.

    An empty selection means the full tensor.  Axes not mentioned are selected
    in full.  Indices use Python's half-open convention and must be
    non-negative; this avoids ambiguous negative-index behavior in artifacts.
    """

    model_config = ConfigDict(extra="forbid")

    slices: list[AxisSlice] = Field(default_factory=list)

    def resolve(self, shape: list[int] | tuple[int, ...]) -> "ResolvedSelection":
        source_shape = [int(d) for d in shape]
        ndim = len(source_shape)
        by_axis: dict[int, AxisSlice] = {}
        for spec in self.slices:
            if spec.axis >= ndim:
                raise InputError(
                    f"slice axis {spec.axis} is out of range for a {ndim}-D tensor",
                    details={"shape": source_shape, "axis": spec.axis},
                )
            if spec.axis in by_axis:
                raise InputError(
                    f"axis {spec.axis} is selected more than once",
                    details={"shape": source_shape, "axis": spec.axis},
                )
            by_axis[spec.axis] = spec

        dims: list[ResolvedAxisSlice] = []
        selected_shape: list[int] = []
        for axis, dimension in enumerate(source_shape):
            spec = by_axis.get(axis)
            start = 0 if spec is None or spec.start is None else spec.start
            stop = dimension if spec is None or spec.stop is None else spec.stop
            step = 1 if spec is None else spec.step
            if start > dimension or stop > dimension:
                raise InputError(
                    f"slice on axis {axis} exceeds dimension {dimension}",
                    details={"axis": axis, "start": start, "stop": stop, "dimension": dimension},
                )
            if start > stop:
                raise InputError(
                    f"slice start {start} is greater than stop {stop} on axis {axis}",
                    details={"axis": axis, "start": start, "stop": stop},
                )
            length = 0 if start == stop else (stop - start + step - 1) // step
            dims.append(
                ResolvedAxisSlice(
                    axis=axis, start=start, stop=stop, step=step, length=length
                )
            )
            selected_shape.append(length)

        numel = 1
        for dimension in selected_shape:
            numel *= dimension
        return ResolvedSelection(
            source_shape=source_shape,
            selected_shape=selected_shape,
            numel=numel,
            dims=dims,
        )


class ResolvedAxisSlice(BaseModel):
    axis: int
    start: int
    stop: int
    step: int
    length: int


class ResolvedSelection(BaseModel):
    source_shape: list[int]
    selected_shape: list[int]
    numel: int
    dims: list[ResolvedAxisSlice]

    def to_index(self) -> tuple[slice, ...]:
        return tuple(slice(dim.start, dim.stop, dim.step) for dim in self.dims)

    def is_full(self) -> bool:
        return self.numel == _product(self.source_shape)

    def canonical(self) -> dict[str, Any]:
        return {
            "source_shape": self.source_shape,
            "selected_shape": self.selected_shape,
            "numel": self.numel,
            "dims": [dim.model_dump() for dim in self.dims],
        }


def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result
