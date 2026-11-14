"""Execution-time activation interventions.

This module is deliberately small and conservative.  A session installs at most
one forward hook on one module, replaces only the selected region of one
activation, and never mutates parameters, buffers, or the incoming activation
in place.  The original tensor is cloned before assignment, and the module's
return structure (tensor, tuple, list, mapping, or ``ModelOutput``) is rebuilt
rather than guessed at.

Interventions are execution-scoped.  Nothing here writes a checkpoint, and the
model parameters are fingerprinted before and after every pass so accidental
mutation is detected instead of assumed away.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from wsp.core.budget import Budget, enforce_numel
from wsp.core.dtypes import torch_dtype_from_name, torch_dtype_name, torch_to_numpy
from wsp.core.errors import (
    CompatibilityError,
    ExecutionError,
    InputError,
    ResourceLimitError,
    UnsupportedError,
    WspError,
)
from wsp.core.hashing import sha256_bytes
from wsp.core.models import TensorDiffStats
from wsp.core.time import utc_now_iso
from wsp.replay.engine import parameter_fingerprint
from wsp.tensors.diff import compare_arrays

from .models import (
    ActivationIntervention,
    ConditionSummary,
    CounterfactualComparison,
    DownstreamActivation,
    EffectSummary,
    ExecutionMode,
    IdentityControl,
    InterventionApplication,
    InterventionResult,
    InterventionScope,
    InterventionValidation,
    SliceComparison,
    TensorSelector,
    TokenLogprobDelta,
    ValidationCheck,
    ValidityReport,
    comparison_sha256,
    result_sha256,
)
from .operations import (
    apply_operation,
    compare_slices,
    cosine_projection,
    extract_activation,
    finite_or_raise,
    rebuild_activation,
    resolve_selection,
    selected_region,
    selection_length,
    vector_values,
)

TORCH_HOOKS = "torch_hooks"
SUPPORTED_BACKENDS = (TORCH_HOOKS,)

IDENTITY_ATOL = 1e-6
IDENTITY_RTOL = 1e-5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def torch_module(model: Any) -> torch.nn.Module:
    """Return the underlying ``torch.nn.Module`` of a replay model wrapper."""

    for attribute in ("module", "model"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate
    raise UnsupportedError(
        "the model wrapper does not expose a torch module for hook-based intervention",
        details={"type": type(model).__name__},
    )


def describe_model(model: Any) -> dict[str, Any]:
    """A bounded, JSON-safe identity block for an execution record."""

    describe = getattr(model, "describe", None)
    if callable(describe):
        info = describe()
        dump = getattr(info, "model_dump", None)
        if callable(dump):
            payload = dump(mode="json")
        else:  # pragma: no cover - defensive
            payload = dict(vars(info))
    else:  # pragma: no cover - defensive
        payload = {"name": type(model).__name__}
    keep = (
        "name",
        "model_class",
        "parameter_count",
        "file_sha256",
        "structure_fingerprint",
        "device",
        "dtype",
        "backend",
        "backend_version",
        "config_architectures",
    )
    return {key: payload[key] for key in keep if key in payload}


def _module_lookup(root: torch.nn.Module, path: str) -> torch.nn.Module:
    wanted = "" if path in ("<root>", "") else path
    for name, candidate in root.named_modules():
        if name == wanted:
            return candidate
    raise NotFoundModule(path)


class NotFoundModule(UnsupportedError):
    def __init__(self, path: str) -> None:
        super().__init__(
            f"module '{path}' is not present in the execution model",
            details={"module": path},
        )


def _token_tensor(token_ids: Sequence[int]) -> torch.Tensor:
    if not token_ids:
        raise InputError("an intervention requires at least one token id")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in token_ids):
        raise InputError("token ids must be non-negative integers")
    return torch.tensor([list(token_ids)], dtype=torch.long)


def _check_logits(
    logits: torch.Tensor,
    *,
    spec: ActivationIntervention,
    budget: Budget,
) -> torch.Tensor:
    if logits.ndim != 3 or int(logits.shape[0]) != 1:
        raise CompatibilityError(
            "the execution produced logits with an unexpected shape",
            details={"shape": [int(item) for item in logits.shape]},
        )
    elements = int(logits.numel())
    enforce_numel(
        elements,
        budget.max_replay_logits_elements,
        what="forward-pass logits",
    )
    # ``spec.limits.max_logit_elements`` bounds the logits rows that are actually
    # compared and serialized, not the model's full [1, sequence, vocabulary]
    # tensor.  Paired measurement enforces it on the selected rows.
    cpu = logits.detach().to("cpu")
    finite_or_raise(cpu, what="forward-pass logits")
    return cpu


def _as_declared_dtype(tensor: torch.Tensor, dtype_name: str | None) -> torch.Tensor:
    """Match the dtype recorded by the specification.

    Values loaded from npy artifacts arrive as float64 because that is the lossless
    storage form.  When the specification records the activation dtype, casting
    back is a storage conversion, not an unrecorded precision change.
    """

    if dtype_name is None:
        return tensor
    try:
        target = torch_dtype_from_name(dtype_name)
    except Exception as exc:  # noqa: BLE001 - reported as a specification error
        raise CompatibilityError(
            "the specification records an unsupported dtype",
            details={"dtype": dtype_name, "reason": str(exc)[:120]},
        ) from exc
    return tensor.to(target)


def _replacement_tensor(
    spec: ActivationIntervention,
    store: Any | None,
) -> torch.Tensor | None:
    """Load and validate the mean or donor values for a specification."""

    if spec.operation == "mean":
        source = spec.mean
        if source is None:  # pragma: no cover - model validation
            raise CompatibilityError("a mean ablation requires mean values")
        if source.values is not None:
            return _as_declared_dtype(
                vector_values(source.values, what="mean activation"), source.dtype
            )
        if store is None:
            raise InputError(
                "the mean activation is stored as an artifact but no workspace store "
                "was supplied"
            )
        try:
            array = store.read_array(source.artifact_id)
        except WspError as exc:
            raise CompatibilityError(
                "the mean activation artifact could not be loaded",
                details={"artifact_id": source.artifact_id, "error": str(exc)},
            ) from exc
        return _as_declared_dtype(
            vector_values(array, what="mean activation"), source.dtype
        )
    if spec.operation == "patch":
        donor = spec.donor
        if donor is None:  # pragma: no cover - model validation
            raise CompatibilityError("an activation patch requires donor values")
        tensor: torch.Tensor | None = None
        if donor.values is not None:
            tensor = vector_values(donor.values, what="donor activation")
        elif store is None:
            raise InputError(
                "the donor activation is stored as an artifact but no workspace store "
                "was supplied"
            )
        else:
            try:
                array = store.read_array(donor.values_artifact_id)
            except WspError as exc:
                raise CompatibilityError(
                    "the donor activation artifact could not be loaded",
                    details={"artifact_id": donor.values_artifact_id, "error": str(exc)},
                ) from exc
            tensor = vector_values(array, what="donor activation")
        if donor.shape is not None and int(np.prod(donor.shape)) == int(tensor.numel()):
            tensor = tensor.reshape(tuple(int(item) for item in donor.shape))
        return _as_declared_dtype(tensor, donor.dtype)
    return None


def _rows_for_measurement(
    tensor: torch.Tensor,
    positions: Sequence[int],
    *,
    sequence_length: int,
    tensor_name: str | None,
    tensor_index: int | None,
    value: Any,
    role: str,
) -> torch.Tensor:
    selector = TensorSelector(
        positions=list(positions),
        tensor_name=tensor_name,
        tensor_index=tensor_index,
    )
    selection = resolve_selection(tensor, selector, sequence_length=sequence_length)
    return selected_region(tensor, selection)


# --------------------------------------------------------------------------- #
# the session
# --------------------------------------------------------------------------- #


@dataclass
class Measurement:
    module: str
    call_index: int
    shape: list[int]
    rows: np.ndarray
    note: str | None = None


@dataclass
class SessionOutcome:
    applications: list[InterventionApplication] = field(default_factory=list)
    target_comparison: SliceComparison | None = None
    target_delta: np.ndarray | None = None
    measurements: dict[str, Measurement] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    scope: InterventionScope | None = None


class InterventionSession:
    """Install one reversible activation intervention for one execution."""

    def __init__(
        self,
        root: torch.nn.Module,
        spec: ActivationIntervention,
        *,
        replacement: torch.Tensor | None = None,
        apply: bool = True,
        measure_modules: Sequence[str] = (),
        measure_positions: Sequence[int] | None = None,
        measure_tensor_names: dict[str, str] | None = None,
        resolve_all_calls: bool = False,
        budget: Budget | None = None,
    ) -> None:
        self.budget = budget or Budget()
        if spec.execution.backend not in SUPPORTED_BACKENDS:
            raise UnsupportedError(
                "activation intervention execution is only implemented for the "
                "torch-hooks backend in this release",
                details={
                    "backend": spec.execution.backend,
                    "supported": list(SUPPORTED_BACKENDS),
                    "note": "nnterp capture remains available; intervention support "
                    "there is explicitly deferred",
                },
            )
        self.spec = spec
        self.root = root
        self.replacement = replacement
        self.apply = apply
        self.sequence_length = int(spec.execution.sequence_length)
        self.module_path = "" if spec.target.module in ("", "<root>") else spec.target.module
        self.module = _module_lookup(root, self.module_path)
        self._call_index = 0
        self._applied_calls: set[int] = set()
        self.outcome = SessionOutcome()
        self._handles: list[Any] = []
        self._closed = False
        self._measure_modules = list(dict.fromkeys(measure_modules))
        self._measure_positions = list(measure_positions or [])
        self._measure_tensor_names = dict(measure_tensor_names or {})
        self._measure_call_index: dict[str, int] = {}
        self._resolve_all_calls = resolve_all_calls

        if len(self._measure_modules) > self.budget.max_counterfactual_modules:
            raise ResourceLimitError(
                "too many downstream modules were requested for measurement",
                details={
                    "requested": len(self._measure_modules),
                    "limit": self.budget.max_counterfactual_modules,
                },
            )
        if self._measure_positions and len(self._measure_positions) > self.budget.max_counterfactual_rows:
            raise ResourceLimitError(
                "too many downstream positions were requested for measurement",
                details={
                    "requested": len(self._measure_positions),
                    "limit": self.budget.max_counterfactual_rows,
                },
            )

    # context manager ------------------------------------------------------ #

    def __enter__(self) -> "InterventionSession":
        self.install()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # hooks ---------------------------------------------------------------- #

    def install(self) -> None:
        if self._handles:
            raise InputError("the intervention session is already installed")
        if self.spec.target.tensor_role == "input":
            handle = self.module.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
            self._handles.append(handle)
        else:
            handle = self.module.register_forward_hook(self._post_hook)
            self._handles.append(handle)

        for path in self._measure_modules:
            target = _module_lookup(self.root, path)
            if path == self.module_path:
                continue
            self._handles.append(target.register_forward_hook(self._measurement_hook(path)))

    def close(self) -> None:
        if self._closed:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._closed = True

    def _pre_hook(self, module: torch.nn.Module, args: Any, kwargs: Any) -> Any:
        if len(args) == 1:
            value = args[0]
            single = True
        else:
            value = tuple(args)
            single = False
        updated = self._process(value)
        if updated is value:
            return None
        if single:
            return ((updated,) + tuple(args[1:]), kwargs)
        return (tuple(updated), kwargs)

    def _post_hook(self, module: torch.nn.Module, args: Any, output: Any) -> Any:
        updated = self._process(output)
        if updated is output:
            return None
        return updated

    def _measurement_hook(self, path: str):
        def hook(module: torch.nn.Module, args: Any, output: Any) -> None:
            if not self._measure_positions:
                return
            call_index = self._measure_call_index.get(path, 0)
            self._measure_call_index[path] = call_index + 1
            if call_index != 0 or path in self.outcome.measurements:
                return
            try:
                tensor, _ = extract_activation(
                    output,
                    TensorSelector(
                        positions=list(self._measure_positions),
                        tensor_name=self._measure_tensor_names.get(path),
                    ),
                    role="output",
                )
                rows = _rows_for_measurement(
                    tensor,
                    self._measure_positions,
                    sequence_length=self.sequence_length,
                    tensor_name=self._measure_tensor_names.get(path),
                    tensor_index=None,
                    value=output,
                    role="output",
                )
            except WspError as exc:
                self.outcome.notes.append(
                    f"downstream measurement at '{path}' was skipped: {exc}"
                )
                return
            self.outcome.measurements[path] = Measurement(
                module=path,
                call_index=call_index,
                shape=[int(item) for item in tensor.shape],
                rows=rows.detach().to("cpu", torch.float64).numpy(),
            )

        return hook

    # processing ----------------------------------------------------------- #

    def _process(self, value: Any) -> Any:
        call_index = self._call_index
        self._call_index += 1
        should_apply = (
            self.apply
            and call_index in self.spec.apply_on_calls
            and call_index not in self._applied_calls
        )
        if not should_apply and not (self._resolve_all_calls or not self.apply):
            return value
        tensor, location = extract_activation(
            value,
            self.spec.target.selector,
            role=self.spec.target.tensor_role,
        )
        selection = resolve_selection(
            tensor,
            self.spec.target.selector,
            sequence_length=self.sequence_length,
        )
        self.outcome.scope = _scope_from_selection(self.spec, selection, tensor)
        if not should_apply:
            return value
        updated, elements, dtype_cast, before, after = apply_operation(
            tensor,
            selection,
            operation=self.spec.operation,
            replacement=self.replacement,
            allow_dtype_cast=self.spec.allow_dtype_cast,
            allow_position_broadcast=bool(
                self.spec.donor is not None and self.spec.donor.allow_position_broadcast
            ),
        )
        self._applied_calls.add(call_index)
        payload = after.detach().to("cpu", torch.float64).contiguous().numpy().tobytes()
        self.outcome.applications.append(
            InterventionApplication(
                module=self.module_path,
                call_index=call_index,
                tensor_role=self.spec.target.tensor_role,
                tensor_shape=[int(item) for item in tensor.shape],
                dtype=torch_dtype_name(tensor),
                axis=selection.axis,
                rows=list(selection.rows),
                feature_start=selection.feature_start,
                feature_end=selection.feature_end,
                operation=self.spec.operation,
                elements_modified=elements,
                replacement_sha256=sha256_bytes(payload),
                dtype_cast=dtype_cast,
            )
        )
        self.outcome.target_delta = (
            after.detach().to("cpu", torch.float64) - before.detach().to("cpu", torch.float64)
        ).reshape(-1).numpy()
        if self.outcome.target_comparison is None:
            before_stats, after_stats, max_abs, mean_abs, relative, cosine = compare_slices(
                before, after, selection.rows
            )
            self.outcome.target_comparison = SliceComparison(
                positions=list(selection.rows),
                before=before_stats,
                after=after_stats,
                max_abs=max_abs,
                mean_abs=mean_abs,
                relative_l2=relative,
                cosine=cosine,
            )
        return rebuild_activation(value, location, updated)

    # introspection -------------------------------------------------------- #

    @property
    def applied_calls(self) -> list[int]:
        return sorted(self._applied_calls)

    @property
    def applications(self) -> list[InterventionApplication]:
        return list(self.outcome.applications)


def _scope_from_selection(
    spec: ActivationIntervention,
    selection: Any,
    tensor: torch.Tensor,
) -> InterventionScope:
    replacement_source = {
        "zero": "zeros",
        "mean": "mean-activation",
        "patch": "donor-activation",
        "identity": "original-activation",
    }[spec.operation]
    rows = list(selection.rows)
    lines = [
        f"operation: {spec.operation} ({replacement_source})",
        f"module: {spec.target.module or '<root>'} [{spec.target.tensor_role}]",
        f"activation shape: {[int(item) for item in tensor.shape]} ({torch_dtype_name(tensor)})",
        f"sequence axis: {selection.axis}; positions: {rows}",
        (
            f"feature region: {selection.feature_start}..{selection.feature_end}"
            if selection.feature_start is not None
            else (
                f"feature indices: {selection.feature_indices}"
                if selection.feature_indices
                else "feature region: all"
            )
        ),
        f"elements modified per application: {selection.length if tensor.ndim > 1 else 1} "
        f"x {selection.width}",
        f"applied on calls: {spec.apply_on_calls}",
    ]
    return InterventionScope(
        module=spec.target.module or "<root>",
        tensor_role=spec.target.tensor_role,
        axis=selection.axis,
        rows=rows,
        feature_start=selection.feature_start,
        feature_end=selection.feature_end,
        feature_indices=list(selection.feature_indices),
        operation=spec.operation,
        replacement_source=replacement_source,
        position_semantics=spec.target.position_semantics,
        apply_on_calls=list(spec.apply_on_calls),
        estimated_elements=len(rows) * selection.width,
        expected_shape=[int(item) for item in tensor.shape],
        expected_dtype=torch_dtype_name(tensor),
        human_readable=lines,
    )


# --------------------------------------------------------------------------- #
# single passes
# --------------------------------------------------------------------------- #


@dataclass
class PassOutcome:
    condition: str
    applied: bool
    logits: torch.Tensor
    token_ids: list[int]
    applications: list[InterventionApplication]
    target_comparison: SliceComparison | None
    measurements: dict[str, Measurement]
    notes: list[str]
    elapsed_ms: float
    fingerprint_before: str
    fingerprint_after: str
    forward_calls: int
    resources: dict[str, int]
    outcome_delta: np.ndarray | None = None


def run_pass(
    spec: ActivationIntervention,
    model: Any,
    token_ids: Sequence[int],
    *,
    condition: str,
    store: Any | None = None,
    operation: str | None = None,
    replacement: torch.Tensor | None = None,
    apply: bool = True,
    measure_modules: Sequence[str] = (),
    measure_positions: Sequence[int] | None = None,
    measure_tensor_names: dict[str, str] | None = None,
    budget: Budget | None = None,
) -> PassOutcome:
    """Execute one forward pass, optionally under an intervention."""

    budget = budget or Budget()
    tokens = [int(item) for item in token_ids]
    if len(tokens) != spec.execution.sequence_length:
        raise CompatibilityError(
            "the token sequence length does not match the recorded execution identity",
            details={
                "tokens": len(tokens),
                "execution_sequence_length": spec.execution.sequence_length,
            },
        )
    root = torch_module(model)
    effective = spec
    if operation is not None and operation != spec.operation:
        payload = spec.model_dump(mode="python")
        payload["operation"] = operation
        if operation in ("zero", "identity"):
            payload["mean"] = None
            payload["donor"] = None
        effective = ActivationIntervention.model_validate(payload)
    elif replacement is None:
        replacement = _replacement_tensor(effective, store)

    was_training = root.training
    before = parameter_fingerprint(root)
    started = time.perf_counter()
    calls = {"n": 0}
    if measure_modules and not measure_positions:
        measure_positions = list(effective.target.selector.positions)
    session = InterventionSession(
        root,
        effective,
        replacement=replacement,
        apply=apply,
        measure_modules=measure_modules,
        measure_positions=measure_positions,
        measure_tensor_names=measure_tensor_names,
        budget=budget,
    )
    root.eval()
    try:
        with torch.inference_mode():
            session.install()
            try:
                logits = model.forward_token_ids(tokens)
            finally:
                session.close()
    finally:
        root.train(was_training)
    elapsed = (time.perf_counter() - started) * 1000.0
    calls["n"] = 1
    checked = _check_logits(logits, spec=effective, budget=budget)
    after = parameter_fingerprint(root)
    if before != after:
        raise _mutation_error()
    return PassOutcome(
        condition=condition,
        applied=bool(session.outcome.applications),
        logits=checked,
        token_ids=tokens,
        applications=session.outcome.applications,
        target_comparison=session.outcome.target_comparison,
        measurements=session.outcome.measurements,
        notes=session.outcome.notes,
        elapsed_ms=elapsed,
        fingerprint_before=before,
        fingerprint_after=after,
        forward_calls=calls["n"],
        resources={
            "logits_elements": int(checked.numel()),
            "applications": len(session.outcome.applications),
            "measured_modules": len(session.outcome.measurements),
        },
        outcome_delta=session.outcome.target_delta,
    )


def _mutation_error() -> WspError:
    return ExecutionError(
        "model parameters changed during the intervention; the run is not "
        "trustworthy and no result should be recorded"
    )


# --------------------------------------------------------------------------- #
# validation / preview
# --------------------------------------------------------------------------- #


def inspect_intervention(
    spec: ActivationIntervention,
    model: Any,
    token_ids: Sequence[int],
    *,
    store: Any | None = None,
    budget: Budget | None = None,
) -> InterventionValidation:
    """Validate a specification against a model without changing any activation."""

    budget = budget or Budget()
    checks: list[ValidationCheck] = []
    errors: list[str] = []
    warnings: list[str] = []
    scope: InterventionScope | None = None

    if spec.execution.backend not in SUPPORTED_BACKENDS:
        checks.append(
            ValidationCheck(
                name="backend",
                status="fail",
                detail=f"'{spec.execution.backend}' does not support intervention execution",
            )
        )
        errors.append(
            "the requested instrumentation backend cannot execute interventions in this release"
        )
    else:
        checks.append(ValidationCheck(name="backend", status="pass", detail=spec.execution.backend))

    if len(token_ids) != spec.execution.sequence_length:
        checks.append(
            ValidationCheck(
                name="sequence_length",
                status="fail",
                detail="token ids do not match execution.sequence_length",
            )
        )
        errors.append("the token sequence does not match the recorded execution length")
    else:
        checks.append(ValidationCheck(name="sequence_length", status="pass"))

    root = torch_module(model)
    try:
        module = _module_lookup(root, spec.target.module)
        checks.append(
            ValidationCheck(
                name="capture_point",
                status="pass",
                detail=f"{spec.target.module or '<root>'} :: {type(module).__name__}",
            )
        )
    except WspError as exc:
        checks.append(ValidationCheck(name="capture_point", status="fail", detail=str(exc)))
        errors.append(str(exc))
        module = None

    replacement: torch.Tensor | None = None
    try:
        replacement = _replacement_tensor(spec, store)
        if replacement is not None:
            if not torch.isfinite(replacement).all().item():
                raise CompatibilityError("replacement values are not finite")
            checks.append(
                ValidationCheck(
                    name="replacement",
                    status="pass",
                    detail=f"{spec.operation} values: {tuple(replacement.shape)}",
                )
            )
        else:
            checks.append(ValidationCheck(name="replacement", status="pass", detail=spec.operation))
    except WspError as exc:
        checks.append(ValidationCheck(name="replacement", status="fail", detail=str(exc)))
        errors.append(str(exc))

    if module is not None and not errors:
        was_training = root.training
        root.eval()
        session = InterventionSession(
            root,
            spec,
            replacement=replacement,
            apply=False,
            budget=budget,
        )
        try:
            with torch.inference_mode():
                session.install()
                try:
                    model.forward_token_ids([int(item) for item in token_ids])
                finally:
                    session.close()
        except WspError as exc:
            errors.append(str(exc))
            checks.append(ValidationCheck(name="dry_run", status="fail", detail=str(exc)))
        finally:
            root.train(was_training)
        scope = session.outcome.scope
        if scope is not None:
            checks.append(
                ValidationCheck(
                    name="dry_run",
                    status="pass",
                    detail=(
                        f"resolved shape {scope.expected_shape} axis {scope.axis} "
                        f"rows {scope.rows}"
                    ),
                )
            )
            if scope.estimated_elements > spec.limits.max_elements:
                checks.append(
                    ValidationCheck(
                        name="elements",
                        status="fail",
                        detail=(
                            f"{scope.estimated_elements} elements exceed the intervention "
                            f"limit of {spec.limits.max_elements}"
                        ),
                    )
                )
                errors.append("the selected region exceeds limits.max_elements")
        notes = list(session.outcome.notes)
        warnings.extend(notes)
    valid = not errors
    lines = []
    if scope is not None:
        lines.extend(scope.human_readable)
    lines.append(f"valid: {valid}")
    lines.extend(f"error: {item}" for item in errors)
    lines.extend(f"warning: {item}" for item in warnings)
    return InterventionValidation(
        valid=valid,
        checks=checks,
        scope=scope,
        errors=errors,
        warnings=warnings,
        human_readable=lines,
    )


# --------------------------------------------------------------------------- #
# paired counterfactual measurement
# --------------------------------------------------------------------------- #


@dataclass
class PairedOutcome:
    comparison: CounterfactualComparison
    results: dict[str, InterventionResult]


def _logit_rows(logits: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    index = torch.tensor(list(positions), dtype=torch.long)
    return logits.index_select(1, index)


def _next_token_logprobs(
    logits: torch.Tensor, token_ids: Sequence[int], positions: Sequence[int]
) -> dict[int, tuple[int, float, int]]:
    """Map each position to ``(next_token_id, logprob, rank)``."""

    out: dict[int, tuple[int, float, int]] = {}
    probabilities = torch.log_softmax(logits.to(torch.float64), dim=-1)
    for position in positions:
        if position + 1 >= len(token_ids):
            continue
        token_id = int(token_ids[position + 1])
        row = probabilities[0, position]
        value = float(row[token_id].item())
        rank = int((row > row[token_id]).sum().item()) + 1
        out[position] = (token_id, value, rank)
    return out


def _spec_variant(spec: ActivationIntervention, operation: str) -> ActivationIntervention:
    payload = spec.model_dump(mode="python")
    payload["operation"] = operation
    payload["mean"] = None
    payload["donor"] = None
    return ActivationIntervention.model_validate(payload)


def retarget_spec_execution(
    spec: ActivationIntervention,
    *,
    execution_mode: ExecutionMode,
    sequence_length: int,
    backend_version: str | None = None,
) -> ActivationIntervention:
    """Return a re-validated copy of a specification for another execution mode."""

    payload = spec.model_dump(mode="python")
    payload["execution"]["execution_mode"] = execution_mode
    payload["execution"]["sequence_length"] = int(sequence_length)
    if backend_version is not None:
        payload["execution"]["backend_version"] = backend_version
    for position in payload["target"]["selector"]["positions"]:
        if position >= sequence_length:
            raise CompatibilityError(
                "an intervention position lies outside the new execution sequence",
                details={"position": position, "sequence_length": sequence_length},
            )
    return ActivationIntervention.model_validate(payload)


def run_paired_intervention(
    spec: ActivationIntervention,
    model: Any,
    token_ids: Sequence[int],
    *,
    store: Any | None = None,
    budget: Budget | None = None,
    measure_modules: Sequence[str] = (),
    measure_positions: Sequence[int] | None = None,
    measure_tensor_names: dict[str, str] | None = None,
    logit_positions: Sequence[int] | None = None,
    alternative: ActivationIntervention | None = None,
    direction: Sequence[float] | None = None,
    case_id: str | None = None,
    intervention_artifact_id: str | None = None,
    identity_atol: float = IDENTITY_ATOL,
    identity_rtol: float = IDENTITY_RTOL,
    identity: bool = True,
) -> PairedOutcome:
    """Run untouched, identity, intervention, and optional alternative passes."""

    budget = budget or Budget()
    tokens = [int(item) for item in token_ids]
    if spec.execution.execution_mode != "teacher-forced":
        raise CompatibilityError(
            "paired measurement expects a teacher-forced specification; use the "
            "continuation interface for generation experiments",
            details={"execution_mode": spec.execution.execution_mode},
        )
    if len(tokens) != spec.execution.sequence_length:
        raise CompatibilityError(
            "the token sequence length does not match the recorded execution identity",
            details={"tokens": len(tokens), "expected": spec.execution.sequence_length},
        )
    if spec.execution.backend not in SUPPORTED_BACKENDS:
        raise UnsupportedError(
            "paired intervention execution is only available for the torch-hooks backend",
            details={"backend": spec.execution.backend},
        )

    sequence_length = len(tokens)
    if measure_modules and not measure_positions:
        measure_positions = list(spec.target.selector.positions)
    if logit_positions is None:
        # Default to the intervened positions: comparing every position of a
        # real vocabulary would exceed the logits budget and is rarely useful.
        positions = list(spec.target.selector.positions)
    else:
        positions = [int(item) for item in logit_positions]
        if any(item < 0 or item >= sequence_length for item in positions):
            raise InputError("a logit position lies outside the token sequence")
    if len(set(positions)) != len(positions):
        raise InputError("logit positions must be unique")
    if len(positions) > budget.max_counterfactual_rows:
        raise ResourceLimitError(
            "too many logit positions were requested",
            details={"requested": len(positions), "limit": budget.max_counterfactual_rows},
        )
    model_info = describe_model(model)
    checks: list[ValidationCheck] = []
    warnings: list[str] = []

    def execute(
        effective: ActivationIntervention,
        *,
        condition: str,
        apply: bool,
        operation: str | None = None,
    ) -> PassOutcome:
        return run_pass(
            effective,
            model,
            tokens,
            condition=condition,
            store=store,
            operation=operation,
            apply=apply,
            measure_modules=measure_modules,
            measure_positions=measure_positions,
            measure_tensor_names=measure_tensor_names,
            budget=budget,
        )

    baseline = execute(spec, condition="baseline", apply=False)
    longest = budget.max_counterfactual_rows
    if len(positions) > longest:  # pragma: no cover - guarded above
        raise ResourceLimitError("too many logit positions")
    baseline_rows = _logit_rows(baseline.logits, positions)
    if baseline_rows.numel() > budget.max_intervention_logit_elements:
        raise ResourceLimitError(
            "the selected logits exceed the intervention logit budget",
            details={
                "elements": int(baseline_rows.numel()),
                "limit": budget.max_intervention_logit_elements,
            },
        )

    identity_outcome: PassOutcome | None = None
    identity_control = IdentityControl(performed=False)
    if identity:
        identity_outcome = execute(
            _spec_variant(spec, "identity"),
            condition="identity",
            apply=True,
            operation="identity",
        )
        exact = torch.equal(baseline.logits, identity_outcome.logits)
        stats = compare_arrays(
            torch_to_numpy(baseline.logits),
            torch_to_numpy(identity_outcome.logits),
            atol=identity_atol,
            rtol=identity_rtol,
        )
        if exact:
            agreement = "exact"
            passed = True
            detail = "baseline and identity logits are bitwise identical"
        elif stats.allclose:
            agreement = "within_tolerance"
            passed = True
            detail = (
                "baseline and identity logits agree within the declared tolerance but "
                "are not bitwise identical"
            )
        else:
            agreement = "failed"
            passed = False
            detail = (
                "baseline and identity logits disagree beyond the declared tolerance; "
                "no observed difference can be attributed solely to the intervention"
            )
        identity_control = IdentityControl(
            performed=True,
            passed=passed,
            agreement=agreement,  # type: ignore[arg-type]
            max_abs_logits_diff=stats.max_abs,
            mean_abs_logits_diff=stats.mean_abs,
            atol=identity_atol,
            rtol=identity_rtol,
            detail=detail,
        )
        checks.append(
            ValidationCheck(name="identity_control", status="pass" if passed else "fail", detail=detail)
        )
    else:
        checks.append(
            ValidationCheck(
                name="identity_control",
                status="skipped",
                detail="the identity control was disabled by the caller",
            )
        )
        warnings.append(
            "the identity control was not performed; attribution of any difference to "
            "the requested intervention is not established"
        )

    intervention = execute(spec, condition="intervention", apply=True)
    if not intervention.applied:
        raise ExecutionError(
            "the intervention did not apply on any selected call index; check apply_on_calls"
        )

    alternative_outcome: PassOutcome | None = None
    if alternative is not None:
        alternative_outcome = execute(
            alternative, condition="alternative-control", apply=True
        )

    def compare_to_baseline(outcome: PassOutcome) -> TensorDiffStats:
        return compare_arrays(
            torch_to_numpy(baseline.logits),
            torch_to_numpy(outcome.logits),
            atol=identity_atol,
            rtol=identity_rtol,
        )

    def token_deltas(outcome: PassOutcome) -> list[TokenLogprobDelta]:
        base = _next_token_logprobs(baseline.logits, tokens, positions)
        ident = (
            _next_token_logprobs(identity_outcome.logits, tokens, positions)
            if identity_outcome is not None
            else {}
        )
        other = _next_token_logprobs(outcome.logits, tokens, positions)
        deltas: list[TokenLogprobDelta] = []
        for position in sorted(base):
            token_id, base_value, _ = base[position]
            identity_value = ident.get(position, (token_id, None, 0))[1]
            other_value = other.get(position, (token_id, None, 0))[1]
            deltas.append(
                TokenLogprobDelta(
                    position=position,
                    token_id=token_id,
                    baseline=base_value,
                    identity=identity_value,
                    intervention=other_value,
                    identity_delta=None
                    if identity_value is None
                    else identity_value - base_value,
                    intervention_delta=None
                    if other_value is None
                    else other_value - base_value,
                )
            )
        return deltas

    def downstream(outcome: PassOutcome) -> list[DownstreamActivation]:
        items: list[DownstreamActivation] = []
        for module, measurement in sorted(outcome.measurements.items()):
            reference = baseline.measurements.get(module)
            if reference is None:
                continue
            stats = compare_arrays(
                reference.rows,
                measurement.rows,
                atol=identity_atol,
                rtol=identity_rtol,
            )
            items.append(
                DownstreamActivation(
                    module=module,
                    positions=list(measure_positions or []),
                    shape=measurement.shape,
                    diff=stats,
                )
            )
        return items

    baseline_result = build_result(
        spec,
        baseline,
        condition="baseline",
        model_info=model_info,
        logits_diff=None,
        token_logprobs=[],
        downstream=[],
    )
    results: dict[str, InterventionResult] = {"baseline": baseline_result}
    if identity_outcome is not None:
        results["identity"] = build_result(
            spec,
            identity_outcome,
            condition="identity",
            model_info=model_info,
            logits_diff=compare_to_baseline(identity_outcome),
            token_logprobs=[],
            downstream=[],
        )
    results["intervention"] = build_result(
        spec,
        intervention,
        condition="intervention",
        model_info=model_info,
        logits_diff=compare_to_baseline(intervention),
        token_logprobs=token_deltas(intervention),
        downstream=downstream(intervention),
    )
    if alternative_outcome is not None:
        results["alternative-control"] = build_result(
            spec,
            alternative_outcome,
            condition="alternative-control",
            model_info=model_info,
            logits_diff=compare_to_baseline(alternative_outcome),
            token_logprobs=token_deltas(alternative_outcome),
            downstream=downstream(alternative_outcome),
        )

    target_projection: dict[str, float | None] = {}
    if direction is not None and intervention.outcome_delta is not None:
        vector = vector_values(direction, what="projection direction")
        target_projection["target_activation_delta"] = cosine_projection(
            torch.from_numpy(intervention.outcome_delta), vector
        )

    validity_checks = list(checks)
    reasons: list[str] = []
    status = "valid"
    if identity_control.performed and not identity_control.passed:
        status = "invalid"
        reasons.append(
            "the identity control failed; the observed difference cannot be attributed "
            "solely to the requested intervention"
        )
    if intervention.fingerprint_before != intervention.fingerprint_after:
        status = "invalid"
        reasons.append("model parameters changed during execution")
    if not identity_control.performed:
        status = "limited" if status == "valid" else status
        reasons.append("no identity control was performed")
    if measure_modules and not intervention.measurements:
        status = "limited" if status == "valid" else status
        reasons.append("downstream measurements were requested but none were captured")
    if intervention.notes:
        warnings.extend(intervention.notes)

    conditions = []
    for condition_name, result in results.items():
        outcome = {
            "baseline": baseline,
            "identity": identity_outcome,
            "intervention": intervention,
            "alternative-control": alternative_outcome,
        }[condition_name]
        top1: dict[str, int] = {}
        for position in positions:
            if position < outcome.logits.shape[1]:
                top1[str(position)] = int(outcome.logits[0, position].argmax().item())
        target_l2 = (
            outcome.target_comparison.after.l2_norm
            if outcome.target_comparison is not None
            else None
        )
        conditions.append(
            ConditionSummary(
                condition=condition_name,  # type: ignore[arg-type]
                result_id=result.result_id,
                applied=outcome.applied,
                elapsed_ms=round(outcome.elapsed_ms, 3),
                logits_shape=[int(item) for item in outcome.logits.shape],
                top1_token_ids=top1,
                target_l2_after=target_l2,
            )
        )

    record = CounterfactualComparison(
        comparison_id=_new_comparison_id(),
        created_at=utc_now_iso(),
        case_id=case_id,
        intervention_id=spec.intervention_id,
        intervention_sha256=_intervention_hash(spec),
        intervention_artifact_id=intervention_artifact_id,
        trace_id=spec.execution.trace_id,
        replay_id=spec.execution.replay_id,
        replay_artifact_id=spec.execution.replay_artifact_id,
        capture_artifact_id=spec.execution.capture_artifact_id,
        model=model_info,
        execution_mode=spec.execution.execution_mode,
        backend=spec.execution.backend,
        conditions=conditions,
        identity_control=identity_control,
        effect=EffectSummary(
            logits=results["intervention"].logits,
            target_activation=intervention.target_comparison,
            downstream=results["intervention"].downstream,
            token_logprobs=results["intervention"].token_logprobs,
            direction_projection=target_projection,
        ),
        validity=ValidityReport(
            status=status,  # type: ignore[arg-type]
            checks=validity_checks,
            reasons=reasons,
        ),
        warnings=warnings,
        metadata={
            "token_sequence_sha256": sha256_bytes(
                ("\n".join(str(item) for item in tokens)).encode("utf-8")
            ),
            "logit_positions": positions,
            "alternative_control": alternative is not None,
            "measure_modules": list(measure_modules),
            "measure_positions": list(measure_positions or []),
        },
    )
    record.checksums["comparison_sha256"] = comparison_sha256(record)
    for result in results.values():
        result.checksums["result_sha256"] = result_sha256(result)
    return PairedOutcome(comparison=record, results=results)


def _intervention_hash(spec: ActivationIntervention) -> str:
    from .models import intervention_sha256

    return intervention_sha256(spec)


def _new_comparison_id() -> str:
    from .ids import new_comparison_id

    return new_comparison_id()


def _new_result_id() -> str:
    from .ids import new_result_id

    return new_result_id()


def build_result(
    spec: ActivationIntervention,
    outcome: PassOutcome,
    *,
    condition: str,
    model_info: dict[str, Any],
    logits_diff: TensorDiffStats | None,
    token_logprobs: list[TokenLogprobDelta],
    downstream: list[DownstreamActivation],
) -> InterventionResult:
    application = outcome.applications[0] if outcome.applications else None
    vocabulary = int(outcome.logits.shape[-1]) if outcome.logits.ndim == 3 else None
    checks = [
        ValidationCheck(
            name="parameters_unchanged",
            status="pass" if outcome.fingerprint_before == outcome.fingerprint_after else "fail",
        ),
        ValidationCheck(name="finite_logits", status="pass"),
    ]
    return InterventionResult(
        result_id=_new_result_id(),
        created_at=utc_now_iso(),
        intervention_id=spec.intervention_id,
        intervention_sha256=_intervention_hash(spec),
        condition=condition,  # type: ignore[arg-type]
        applied=outcome.applied,
        execution_mode=spec.execution.execution_mode,
        backend=spec.execution.backend,
        backend_version=spec.execution.backend_version,
        model=model_info,
        sequence_length=len(outcome.token_ids),
        token_ids_sha256=sha256_bytes(
            ("\n".join(str(item) for item in outcome.token_ids)).encode("utf-8")
        ),
        forward_calls=outcome.forward_calls,
        logits_shape=[int(item) for item in outcome.logits.shape],
        vocabulary_size=vocabulary,
        elapsed_ms=round(outcome.elapsed_ms, 3),
        application=application,
        target_activation=outcome.target_comparison,
        logits=logits_diff,
        logit_positions=[],
        token_logprobs=token_logprobs,
        downstream=downstream,
        resources=outcome.resources,
        checks=checks,
        warnings=list(outcome.notes),
        metadata={"condition": condition},
    )


__all__ = [
    "IDENTITY_ATOL",
    "IDENTITY_RTOL",
    "InterventionSession",
    "Measurement",
    "PairedOutcome",
    "PassOutcome",
    "build_result",
    "describe_model",
    "inspect_intervention",
    "retarget_spec_execution",
    "run_paired_intervention",
    "run_pass",
    "torch_module",
]
