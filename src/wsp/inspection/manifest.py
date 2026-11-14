"""Build and persist machine-readable model manifests."""

from __future__ import annotations

from pathlib import Path

from wsp import __version__
from wsp.core.json import atomic_write_json
from wsp.core.models import ManifestTotals, ModelManifest
from wsp.core.time import utc_now_iso
from wsp.inspection.base import CheckpointSource


def build_manifest(
    source: CheckpointSource,
    *,
    include_tensors: bool = True,
    max_tensors: int | None = None,
) -> ModelManifest:
    infos = source.tensor_infos()
    truncated = False
    returned = infos
    if max_tensors is not None and len(infos) > max_tensors:
        returned = infos[:max_tensors]
        truncated = True
    if not include_tensors:
        returned = []
        truncated = bool(infos)

    by_dtype: dict[str, int] = {}
    by_dtype_bytes: dict[str, int] = {}
    for info in infos:
        by_dtype[info.dtype] = by_dtype.get(info.dtype, 0) + info.numel
        by_dtype_bytes[info.dtype] = by_dtype_bytes.get(info.dtype, 0) + info.nbytes

    totals = ManifestTotals(
        tensor_count=len(infos),
        parameter_count=sum(info.numel for info in infos),
        total_bytes=sum(info.nbytes for info in infos),
        by_dtype=by_dtype,
        by_dtype_bytes=by_dtype_bytes,
    )
    return ModelManifest(
        created_at=utc_now_iso(),
        wsp_version=__version__,
        identity=source.identity(),
        source=source.source_ref(),
        tensors=returned,
        tensor_count=len(infos),
        tensors_truncated=truncated,
        metadata=source.metadata(),
        totals=totals,
    )


def export_manifest(
    manifest: ModelManifest,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    return atomic_write_json(path, manifest, overwrite=overwrite)
