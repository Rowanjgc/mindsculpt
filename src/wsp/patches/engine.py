"""Patch creation, preview, chaining, application, and export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wsp.core.budget import Budget, enforce_numel
from wsp.core.dtypes import bytes_per_element
from wsp.core.environment import dependency_versions
from wsp.core.errors import (
    CompatibilityError,
    InputError,
    ResourceLimitError,
    SafetyError,
    UnsupportedError,
)
from wsp.core.hashing import sha256_bytes, tensor_fingerprint
from wsp.core.json import canonical_bytes
from wsp.core.models import CheckpointIdentity, StatsSummary
from wsp.core.selection import ResolvedSelection, Selection
from wsp.core.time import utc_now_iso
from wsp.inspection.base import CheckpointSource
from wsp.inspection.safetensors_source import SafeTensorsSource
from wsp.patches.operations import (
    ArtifactLoader,
    apply_operation,
    compute_effect,
    payload_sha256,
    resolve_payload,
)
from wsp.patches.safetensors_export import export_with_replacements
from wsp.patches.spec import (
    AddParams,
    PatchEffect,
    PatchParams,
    PatchPreview,
    PatchProvenance,
    PatchSet,
    PayloadRef,
    PrecisionReport,
    ProjectionParams,
    RankOneParams,
    ReplaceParams,
    ScaleParams,
    WeightPatch,
    payload_identity,
)
from wsp.tensors.stats import basic_stats


def _read_full_target(
    source: CheckpointSource, target: str, budget: Budget
) -> np.ndarray[Any, Any]:
    info = source.tensor_info(target)
    enforce_numel(
        info.numel,
        budget.max_tensor_numel,
        what=f"full tensor {target!r} (needed for an exact source fingerprint)",
        details={"tensor": target},
    )
    return source.read_full(target, max_elements=budget.max_tensor_numel)


def _selected(full: np.ndarray[Any, Any], resolved: ResolvedSelection) -> np.ndarray[Any, Any]:
    if list(full.shape) != resolved.source_shape:
        raise InputError(
            "selected region does not match the tensor shape",
            details={"tensor_shape": list(full.shape), "selection": resolved.source_shape},
        )
    index = resolved.to_index()
    selected = full[index] if index else full
    return np.asarray(selected).reshape(resolved.selected_shape)


def _set_selected(
    full: np.ndarray[Any, Any],
    resolved: ResolvedSelection,
    values: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    result = np.array(full, copy=True)
    index = resolved.to_index()
    if index:
        result[index] = values.reshape(resolved.selected_shape)
    else:
        result[...] = values
    return result


def _normalize_payload_ref(
    payload: PayloadRef,
    *,
    expected_shape: list[int] | None,
    budget: Budget,
    loader: ArtifactLoader | None,
    what: str,
) -> PayloadRef:
    values = resolve_payload(
        payload,
        expected_shape=expected_shape,
        budget=budget,
        loader=loader,
        what=what,
    )
    update: dict[str, Any] = {
        "shape": [int(d) for d in values.shape],
        "numel": int(values.size),
        "dtype": "float64",
        "sha256": payload_sha256(values),
    }
    return payload.model_copy(update=update)


def normalize_params(
    params: PatchParams,
    *,
    resolved: ResolvedSelection,
    budget: Budget,
    loader: ArtifactLoader | None = None,
) -> PatchParams:
    selected_shape = list(resolved.selected_shape)
    if isinstance(params, ReplaceParams):
        return params.model_copy(
            update={
                "values": _normalize_payload_ref(
                    params.values,
                    expected_shape=selected_shape,
                    budget=budget,
                    loader=loader,
                    what="replacement values",
                )
            }
        )
    if isinstance(params, AddParams):
        return params.model_copy(
            update={
                "delta": _normalize_payload_ref(
                    params.delta,
                    expected_shape=selected_shape,
                    budget=budget,
                    loader=loader,
                    what="delta",
                )
            }
        )
    if isinstance(params, ScaleParams):
        return params
    if isinstance(params, RankOneParams):
        if len(selected_shape) != 2:
            raise InputError("rank-one updates require a 2-D selected region")
        rows, columns = selected_shape
        return params.model_copy(
            update={
                "u": _normalize_payload_ref(
                    params.u,
                    expected_shape=[rows],
                    budget=budget,
                    loader=loader,
                    what="rank-one u vector",
                ),
                "v": _normalize_payload_ref(
                    params.v,
                    expected_shape=[columns],
                    budget=budget,
                    loader=loader,
                    what="rank-one v vector",
                ),
            }
        )
    if isinstance(params, ProjectionParams):
        if len(selected_shape) != 2:
            raise InputError("projection edits require a 2-D selected region")
        rows, columns = selected_shape
        dimension = columns if params.axis == 1 else rows
        return params.model_copy(
            update={
                "direction": _normalize_payload_ref(
                    params.direction,
                    expected_shape=[dimension],
                    budget=budget,
                    loader=loader,
                    what="projection direction",
                )
            }
        )
    raise UnsupportedError(f"unsupported patch operation: {type(params).__name__}")


def _patch_id(
    source: CheckpointIdentity,
    target: str,
    resolved: ResolvedSelection,
    params: PatchParams,
    expected_fingerprint: str,
) -> str:
    payload = {
        "schema": "wsp.patch/1.0",
        "source_structure_fingerprint": source.structure_fingerprint,
        "target": target,
        "selection": resolved.canonical(),
        "params": payload_identity(params),
        "expected_tensor_fingerprint": expected_fingerprint,
    }
    digest = sha256_bytes(canonical_bytes(payload))[len("sha256:") :]
    return "p_" + digest[:16]


def create_patch(
    source: CheckpointSource,
    *,
    target: str,
    selection: Selection | None,
    params: PatchParams,
    budget: Budget,
    loader: ArtifactLoader | None = None,
    expected_fingerprint: str | None = None,
    notes: str | None = None,
    request_id: str | None = None,
) -> WeightPatch:
    info = source.tensor_info(target)
    resolved = (selection or Selection()).resolve(info.shape)
    if resolved.numel == 0:
        raise InputError(
            "patch selection is empty",
            details={"tensor": target, "shape": info.shape},
        )
    enforce_numel(
        resolved.numel,
        budget.max_tensor_numel,
        what=f"patch selection in {target!r}",
    )
    normalized = normalize_params(
        params, resolved=resolved, budget=budget, loader=loader
    )
    full = _read_full_target(source, target, budget)
    fingerprint = tensor_fingerprint(
        full, storage_dtype=info.dtype, shape=info.shape
    )
    if expected_fingerprint is not None and expected_fingerprint != fingerprint:
        raise CompatibilityError(
            "tensor fingerprint does not match the expected source tensor",
            details={
                "tensor": target,
                "expected": expected_fingerprint,
                "actual": fingerprint,
            },
        )
    before_values = _selected(full, resolved)
    after_values, computed_values, report = apply_operation(
        before_values,
        normalized,
        resolved,
        storage_dtype=info.dtype,
        budget=budget,
        loader=loader,
    )
    before = basic_stats(before_values, storage_dtype=info.dtype)
    after = basic_stats(after_values, storage_dtype=info.dtype)
    effect = PatchEffect(**compute_effect(before_values, after_values))
    if effect.introduced_non_finite:
        raise InputError("patch introduces non-finite values into the tensor")
    patch_id = _patch_id(source.identity(), target, resolved, normalized, fingerprint)
    return WeightPatch(
        patch_id=patch_id,
        created_at=utc_now_iso(),
        source=source.identity(),
        target=target,
        target_info=info,
        selection=resolved,
        operation=normalized.kind,
        params=normalized,
        expected={
            "fingerprint": fingerprint,
            "dtype": info.dtype,
            "shape": info.shape,
        },
        parameter_count=resolved.numel,
        estimated_memory_bytes=int(report["estimated_memory_bytes"]),
        before=before,
        after=after,
        effect=effect,
        precision=PrecisionReport(
            storage_dtype=info.dtype,
            cast_applied=bool(report["cast_applied"]),
            cast_changed_values=bool(report["cast_changed"]),
            notes=list(report["notes"]),
        ),
        provenance=PatchProvenance(
            tool="create_patch",
            dependencies=dependency_versions(),
            notes=notes,
            request_id=request_id,
        ),
    )


def _validate_patch_source(
    source: CheckpointSource, patch: WeightPatch, budget: Budget
) -> np.ndarray[Any, Any]:
    current = source.identity()
    if current.structure_fingerprint != patch.source.structure_fingerprint:
        raise CompatibilityError(
            "checkpoint structure is incompatible with this patch",
            details={
                "patch_source": patch.source.structure_fingerprint,
                "actual_source": current.structure_fingerprint,
            },
        )
    info = source.tensor_info(patch.target)
    if info.dtype != patch.target_info.dtype:
        raise CompatibilityError(
            f"dtype of {patch.target!r} changed since the patch was created",
            details={"expected": patch.target_info.dtype, "actual": info.dtype},
        )
    if info.shape != patch.target_info.shape:
        raise CompatibilityError(
            f"shape of {patch.target!r} changed since the patch was created",
            details={"expected": patch.target_info.shape, "actual": info.shape},
        )
    if patch.expected.dtype != info.dtype or patch.expected.shape != info.shape:
        raise CompatibilityError("patch expected-tensor metadata is internally inconsistent")
    full = _read_full_target(source, patch.target, budget)
    fingerprint = tensor_fingerprint(full, storage_dtype=info.dtype, shape=info.shape)
    if fingerprint != patch.expected.fingerprint:
        raise CompatibilityError(
            f"tensor fingerprint for {patch.target!r} does not match the patch",
            details={
                "tensor": patch.target,
                "expected": patch.expected.fingerprint,
                "actual": fingerprint,
            },
        )
    return full


def _summary_matches(left: StatsSummary, right: StatsSummary) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    for field in (
        "count",
        "finite_count",
        "nan_count",
        "posinf_count",
        "neginf_count",
        "zero_count",
    ):
        if getattr(left, field) != getattr(right, field):
            return False
    for field in ("mean", "std", "min", "max", "abs_max"):
        a = getattr(left, field)
        b = getattr(right, field)
        if a is None or b is None:
            if a is not b:
                return False
        elif not np.isclose(a, b, rtol=1e-6, atol=1e-9):
            return False
    return True


def preview_patch(
    source: CheckpointSource,
    patch: WeightPatch,
    *,
    budget: Budget,
    loader: ArtifactLoader | None = None,
) -> PatchPreview:
    full = _validate_patch_source(source, patch, budget)
    before_values = _selected(full, patch.selection)
    after_values, _computed, report = apply_operation(
        before_values,
        patch.params,
        patch.selection,
        storage_dtype=patch.target_info.dtype,
        budget=budget,
        loader=loader,
    )
    before = basic_stats(before_values, storage_dtype=patch.target_info.dtype)
    after = basic_stats(after_values, storage_dtype=patch.target_info.dtype)
    effect = PatchEffect(**compute_effect(before_values, after_values))
    warnings: list[str] = []
    current = source.identity()
    if (
        patch.source.file_sha256 is not None
        and current.file_sha256 is not None
        and patch.source.file_sha256 != current.file_sha256
    ):
        warnings.append(
            "source file hash differs from the patch record; tensor structure and target fingerprint matched"
        )
    if not _summary_matches(patch.before, before):
        warnings.append("stored pre-patch summary differs from the current source")
    if not _summary_matches(patch.after, after):
        warnings.append("stored post-patch summary differs from the recomputed preview")
    checks = [
        "source_structure_fingerprint_matches",
        "target_dtype_matches",
        "target_shape_matches",
        "target_fingerprint_matches",
        "selection_within_tensor",
        "patch_recomputed_without_mutation",
    ]
    return PatchPreview(
        patch_id=patch.patch_id,
        source=current,
        target=patch.target,
        selection=patch.selection,
        operation=patch.operation,
        parameter_count=patch.parameter_count,
        before=before,
        after=after,
        effect=effect,
        precision=PrecisionReport(
            storage_dtype=patch.target_info.dtype,
            cast_applied=bool(report["cast_applied"]),
            cast_changed_values=bool(report["cast_changed"]),
            notes=list(report["notes"]),
        ),
        expected=patch.expected,
        compatible=True,
        checks=checks,
        warnings=warnings,
    )


def build_patchset(
    source: CheckpointSource,
    patches: list[WeightPatch],
    *,
    budget: Budget,
    output_path: str | Path | None = None,
) -> PatchSet:
    if not patches:
        raise InputError("a patch set must contain at least one patch")
    enforce_numel(len(patches), budget.max_patchset_tensors, what="patches in a patch set")
    current = source.identity()
    targets: set[str] = set()
    for patch in patches:
        if current.structure_fingerprint != patch.source.structure_fingerprint:
            raise CompatibilityError(
                f"patch {patch.patch_id} was created for a different checkpoint structure"
            )
        if patch.target in targets:
            raise CompatibilityError(
                "multiple patches target the same tensor; apply them in separate sets",
                details={"tensor": patch.target},
            )
        targets.add(patch.target)
    patch_ids = [patch.patch_id for patch in patches]
    identity_payload = {
        "schema": "wsp.patchset/1.0",
        "source_structure_fingerprint": current.structure_fingerprint,
        "patch_ids": patch_ids,
    }
    if output_path is not None:
        identity_payload["output_path"] = str(Path(output_path).expanduser().resolve())
    digest = sha256_bytes(canonical_bytes(identity_payload))[len("sha256:") :]
    return PatchSet(
        patchset_id="ps_" + digest[:16],
        created_at=utc_now_iso(),
        source=current,
        patch_ids=patch_ids,
        patches=patches,
    )


def apply_patchset(
    source: CheckpointSource,
    patchset: PatchSet,
    output_path: str | Path,
    *,
    budget: Budget,
    overwrite: bool = False,
    loader: ArtifactLoader | None = None,
) -> PatchSet:
    source_ref = source.source_ref()
    if source_ref.format != "safetensors" or source_ref.path is None:
        raise UnsupportedError(
            "patch application exports to SafeTensors; use a SafeTensors checkpoint source"
        )
    current = source.identity()
    if current.structure_fingerprint != patchset.source.structure_fingerprint:
        raise CompatibilityError("patch set was created for a different checkpoint structure")
    if len(patchset.patches) > budget.max_patchset_tensors:
        raise ResourceLimitError("patch set exceeds the configured patch-count limit")

    replacements: dict[str, np.ndarray[Any, Any]] = {}
    for patch in patchset.patches:
        full = _validate_patch_source(source, patch, budget)
        before_values = _selected(full, patch.selection)
        after_values, _computed, _report = apply_operation(
            before_values,
            patch.params,
            patch.selection,
            storage_dtype=patch.target_info.dtype,
            budget=budget,
            loader=loader,
        )
        effect = compute_effect(before_values, after_values)
        if effect["introduced_non_finite"]:
            raise InputError(
                f"patch {patch.patch_id} would introduce non-finite values"
            )
        replacements[patch.target] = _set_selected(full, patch.selection, after_values)

    destination = Path(output_path).expanduser().resolve()
    if destination == Path(source_ref.path).expanduser().resolve():
        raise SafetyError("refusing to overwrite the source checkpoint")
    export_with_replacements(
        source_ref.path,
        destination,
        replacements,
        overwrite=overwrite,
    )
    with SafeTensorsSource(
        destination, display_name=destination.name, compute_file_sha256=True
    ) as output_source:
        output_identity = output_source.identity()
        if output_identity.structure_fingerprint != current.structure_fingerprint:
            raise CompatibilityError(
                "exported checkpoint structure unexpectedly differs from the source"
            )
    return patchset.model_copy(
        update={
            "output_path": str(destination),
            "output_identity": output_identity,
            "status": "applied",
        }
    )


def revert_patchset(
    source: CheckpointSource, patchset: PatchSet
) -> CheckpointIdentity:
    """Rollback is source selection: verify the patch set and return the immutable source."""

    current = source.identity()
    if current.structure_fingerprint != patchset.source.structure_fingerprint:
        raise CompatibilityError(
            "the supplied source is not the patch set's source checkpoint"
        )
    return current


def export_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> CheckpointIdentity:
    from wsp.patches.safetensors_export import copy_checkpoint

    destination = copy_checkpoint(source_path, output_path, overwrite=overwrite)
    with SafeTensorsSource(
        destination, display_name=destination.name, compute_file_sha256=True
    ) as source:
        return source.identity()
