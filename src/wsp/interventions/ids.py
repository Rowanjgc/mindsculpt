"""Identifier helpers for intervention and continuation records."""

from __future__ import annotations

import secrets


def _new(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def new_intervention_id() -> str:
    return _new("iv")


def new_result_id() -> str:
    return _new("ir")


def new_comparison_id() -> str:
    return _new("cf")


def new_continuation_id() -> str:
    return _new("cc")


def new_plan_id() -> str:
    return _new("pl")


def new_plan_intervention_id() -> str:
    return _new("pi")


def new_event_log_id() -> str:
    return _new("el")


def new_temporal_run_id() -> str:
    return _new("tr2")


def new_composition_id() -> str:
    return _new("cx")


def new_staged_id() -> str:
    return _new("sc")


def new_staged_verification_id() -> str:
    return _new("sv")


__all__ = [
    "new_comparison_id",
    "new_composition_id",
    "new_continuation_id",
    "new_event_log_id",
    "new_intervention_id",
    "new_plan_id",
    "new_plan_intervention_id",
    "new_result_id",
    "new_staged_id",
    "new_staged_verification_id",
    "new_temporal_run_id",
]
