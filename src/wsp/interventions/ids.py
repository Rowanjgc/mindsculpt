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


__all__ = [
    "new_comparison_id",
    "new_continuation_id",
    "new_intervention_id",
    "new_result_id",
]
