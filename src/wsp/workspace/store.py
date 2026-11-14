"""Local experiment workspace with content-addressed artifacts."""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path
from typing import Any, Mapping

from wsp.core.budget import Budget
from wsp.core.errors import ArtifactError, InputError, NotFoundError, SafetyError
from wsp.core.json import atomic_write_bytes, atomic_write_json, read_json
from wsp.core.models import ModelManifest
from wsp.core.time import utc_now_compact, utc_now_iso
from wsp.patches.spec import PatchSet, WeightPatch
from wsp.workspace.artifacts import ArtifactStore
from wsp.workspace.models import ExperimentRecord, RunRecord, WorkspaceConfig

_RUN_ID = re.compile(r"^run_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")
_PATCH_ID = re.compile(r"^p_[0-9a-f]{16}$")
_PATCHSET_ID = re.compile(r"^ps_[0-9a-f]{16}$")
_EXPERIMENT_ID = re.compile(r"^e_[0-9a-f]{16}$")


def _semantic_payload(payload: Any) -> Any:
    """Strip fields that legitimately vary between equivalent records.

    ``created_at`` timestamps, per-run request identifiers, and provenance
    notes describe when and how a record was produced, not what it means.
    Nested objects (for example the patches embedded in a patch set) are
    normalized too.
    """

    if isinstance(payload, dict):
        return {
            key: _semantic_payload(value)
            for key, value in payload.items()
            if key not in {"created_at", "request_id", "provenance"}
        }
    if isinstance(payload, list):
        return [_semantic_payload(value) for value in payload]
    return payload


def _equivalent_except_output_bytes(left: Any, right: Any) -> bool:
    """Compare patch sets while ignoring the exported file's byte checksum.

    The mathematical patch set is identical.  A SafeTensors writer may emit
    tensor metadata in a different order, so two exports can have different
    bytes while representing the same tensors.  The structural fingerprint,
    size, paths, patches, and status must still match; only the whole-file
    digest is ignored.
    """

    def normalized(payload: Any) -> Any:
        value = _semantic_payload(payload)
        identity = value.get("output_identity") if isinstance(value, dict) else None
        if isinstance(identity, dict):
            identity = dict(identity)
            identity.pop("file_sha256", None)
            value = dict(value)
            value["output_identity"] = identity
        return value

    return normalized(left) == normalized(right)


class Workspace:
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        create: bool = False,
        name: str | None = None,
        budget: Budget | None = None,
    ) -> None:
        configured = root or os.environ.get("WSP_WORKSPACE") or ".wsp"
        self.root = Path(configured).expanduser().resolve()
        self.budget = budget or Budget()
        self.config_path = self.root / "workspace.json"
        if create:
            self._create(name=name)
        elif not self.config_path.exists():
            raise NotFoundError(
                f"no workspace found at {self.root}",
                details={"path": str(self.root), "hint": "run 'wsp workspace init'"},
            )
        self.config = WorkspaceConfig.model_validate(read_json(self.config_path))
        # The artifact store is anchored at the workspace root so that
        # ArtifactRef.path is workspace-relative (artifacts/<prefix>/<id>).
        self.artifacts = ArtifactStore(self.root, budget=self.budget)
        for directory in ("patches", "patchsets", "runs", "models", "experiments"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def _create(self, *, name: str | None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            return
        existing = list(self.root.iterdir())
        if existing:
            raise SafetyError(
                "refusing to initialize a workspace in a non-empty directory",
                details={"path": str(self.root), "entries": len(existing)},
            )
        config = WorkspaceConfig(
            workspace_id="ws_" + secrets.token_hex(8),
            name=name or self.root.name,
            created_at=utc_now_iso(),
        )
        atomic_write_json(self.config_path, config.model_dump(), overwrite=False)
        for directory in ("artifacts", "patches", "patchsets", "runs", "models", "experiments"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def _safe(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts).resolve()
        if path != self.root and self.root not in path.parents:
            raise SafetyError("path escapes the workspace")
        return path

    def new_run_id(self) -> str:
        return f"run_{utc_now_compact()}_{secrets.token_hex(4)}"

    def save_run(self, record: RunRecord, *, overwrite: bool = False) -> Path:
        if not _RUN_ID.match(record.run_id):
            raise ArtifactError("invalid run identifier", details={"run_id": record.run_id})
        directory = self._safe("runs", record.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        return atomic_write_json(
            directory / "run.json", record.model_dump(), overwrite=overwrite
        )

    def load_run(self, run_id: str) -> RunRecord:
        if not _RUN_ID.match(run_id):
            raise InputError("invalid run identifier", details={"run_id": run_id})
        path = self._safe("runs", run_id, "run.json")
        if not path.exists():
            raise NotFoundError(f"run {run_id} was not found")
        return RunRecord.model_validate(read_json(path))

    def list_runs(self) -> list[RunRecord]:
        records: list[RunRecord] = []
        for path in sorted(self.root.glob("runs/run_*/run.json")):
            try:
                records.append(RunRecord.model_validate(read_json(path)))
            except Exception as exc:
                raise ArtifactError(f"malformed run record: {path.name}: {exc}") from exc
        return records

    def save_manifest(self, manifest: ModelManifest) -> Path:
        digest = manifest.identity.structure_fingerprint
        short = digest.split(":", 1)[-1][:16]
        path = self._safe("models", short, "manifest.json")
        atomic_write_json(path, manifest.model_dump(), overwrite=True)
        return path

    def load_manifest(self, structure_fingerprint_or_short: str) -> ModelManifest:
        short = structure_fingerprint_or_short.split(":", 1)[-1][:16]
        path = self._safe("models", short, "manifest.json")
        if not path.exists():
            raise NotFoundError("model manifest was not found", details={"id": short})
        return ModelManifest.model_validate(read_json(path))

    def list_manifests(self) -> list[ModelManifest]:
        return [
            ModelManifest.model_validate(read_json(path))
            for path in sorted(self.root.glob("models/*/manifest.json"))
        ]

    def save_patch(self, patch: WeightPatch) -> Path:
        if not _PATCH_ID.match(patch.patch_id):
            raise ArtifactError("invalid patch identifier", details={"patch_id": patch.patch_id})
        path = self._safe("patches", f"{patch.patch_id}.json")
        payload = patch.model_dump_json(indent=2).encode("utf-8")
        if path.exists():
            # Patch identifiers are derived from semantically stable content.
            # ``created_at`` is the one field that changes when the same
            # operation is recreated later, so it must not turn an idempotent
            # save into a collision.
            if _semantic_payload(read_json(path)) != _semantic_payload(patch.model_dump()):
                raise ArtifactError(
                    "patch identifier collides with different patch contents",
                    details={"patch_id": patch.patch_id},
                )
            return path
        return atomic_write_bytes(path, payload, overwrite=False)

    def load_patch(self, patch_id: str) -> WeightPatch:
        if not _PATCH_ID.match(patch_id):
            raise InputError("invalid patch identifier", details={"patch_id": patch_id})
        path = self._safe("patches", f"{patch_id}.json")
        if not path.exists():
            raise NotFoundError(f"patch {patch_id} was not found")
        try:
            patch = WeightPatch.model_validate(read_json(path))
        except Exception as exc:
            raise ArtifactError(f"malformed patch {patch_id}: {exc}") from exc
        if patch.patch_id != patch_id:
            raise ArtifactError("patch filename does not match its patch_id")
        return patch

    def list_patches(self) -> list[WeightPatch]:
        patches: list[WeightPatch] = []
        for path in sorted(self.root.glob("patches/p_*.json")):
            patches.append(self.load_patch(path.stem))
        return patches

    def save_patchset(self, patchset: PatchSet) -> Path:
        if not _PATCHSET_ID.match(patchset.patchset_id):
            raise ArtifactError(
                "invalid patch set identifier", details={"patchset_id": patchset.patchset_id}
            )
        path = self._safe("patchsets", f"{patchset.patchset_id}.json")
        payload = patchset.model_dump_json(indent=2).encode("utf-8")
        if path.exists():
            existing = read_json(path)
            current = patchset.model_dump()
            if _semantic_payload(existing) != _semantic_payload(current):
                if not _equivalent_except_output_bytes(existing, current):
                    raise ArtifactError(
                        "patch set identifier collides with different contents",
                        details={"patchset_id": patchset.patchset_id},
                    )
                # Reapplying an equivalent patch set can legitimately produce
                # different bytes when a SafeTensors header serializes its
                # metadata in a different order.  Keep the original creation
                # time but point the journal at the artifact just produced;
                # each individual application is also recorded as a run.
            current["created_at"] = existing.get("created_at", current["created_at"])
            return atomic_write_json(path, current, overwrite=True)
        return atomic_write_bytes(path, payload, overwrite=False)

    def load_patchset(self, patchset_id: str) -> PatchSet:
        if not _PATCHSET_ID.match(patchset_id):
            raise InputError("invalid patch set identifier", details={"patchset_id": patchset_id})
        path = self._safe("patchsets", f"{patchset_id}.json")
        if not path.exists():
            raise NotFoundError(f"patch set {patchset_id} was not found")
        try:
            patchset = PatchSet.model_validate(read_json(path))
        except Exception as exc:
            raise ArtifactError(f"malformed patch set {patchset_id}: {exc}") from exc
        return patchset

    def list_patchsets(self) -> list[PatchSet]:
        return [
            self.load_patchset(path.stem)
            for path in sorted(self.root.glob("patchsets/ps_*.json"))
        ]

    def artifact_loader(self, artifact_id: str) -> Any:
        ref, path = self.artifacts.get(artifact_id)
        if ref.format == "npy":
            return self.artifacts.read_array(artifact_id)
        if ref.format == "json":
            return self.artifacts.read_json(artifact_id)
        raise ArtifactError(
            "patch payload artifacts must be .npy arrays or JSON values",
            details={"artifact_id": artifact_id, "format": ref.format},
        )

    def create_experiment(
        self,
        name: str,
        *,
        description: str | None = None,
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentRecord:
        record = ExperimentRecord(
            experiment_id="e_" + secrets.token_hex(8),
            name=name,
            description=description,
            created_at=utc_now_iso(),
            tags=list(tags or []),
            metadata=dict(metadata or {}),
        )
        self.save_experiment(record)
        return record

    def save_experiment(self, record: ExperimentRecord) -> Path:
        if not _EXPERIMENT_ID.match(record.experiment_id):
            raise ArtifactError("invalid experiment identifier")
        path = self._safe("experiments", f"{record.experiment_id}.json")
        return atomic_write_json(path, record.model_dump(), overwrite=True)

    def load_experiment(self, experiment_id: str) -> ExperimentRecord:
        if not _EXPERIMENT_ID.match(experiment_id):
            raise InputError("invalid experiment identifier")
        path = self._safe("experiments", f"{experiment_id}.json")
        if not path.exists():
            raise NotFoundError(f"experiment {experiment_id} was not found")
        return ExperimentRecord.model_validate(read_json(path))

    def list_experiments(self) -> list[ExperimentRecord]:
        return [
            ExperimentRecord.model_validate(read_json(path))
            for path in sorted(self.root.glob("experiments/e_*.json"))
        ]

    def add_run_to_experiment(self, experiment_id: str, run_id: str) -> ExperimentRecord:
        experiment = self.load_experiment(experiment_id)
        self.load_run(run_id)  # validate existence
        if run_id not in experiment.run_ids:
            experiment.run_ids.append(run_id)
        self.save_experiment(experiment)
        return experiment

    def status(self) -> dict[str, Any]:
        return {
            "schema_version": "wsp.workspace-status/1.0",
            "workspace": self.config.model_dump(),
            "root": str(self.root),
            "runs": len(self.list_runs()),
            "patches": len(self.list_patches()),
            "patchsets": len(self.list_patchsets()),
            "manifests": len(self.list_manifests()),
            "artifacts": len(self.artifacts.list()),
            "experiments": len(self.list_experiments()),
        }
