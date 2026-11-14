"""Identifier helpers for behavioral records."""

from __future__ import annotations

import secrets


def _new(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def new_trace_id() -> str:
    return _new("tr")


def new_scan_id() -> str:
    return _new("sc")


def new_evaluation_id() -> str:
    return _new("ev")


def new_contrast_id() -> str:
    return _new("ct")


def new_contrast_result_id() -> str:
    return _new("cr")


def new_case_id() -> str:
    return _new("case")


def new_annotation_id() -> str:
    return _new("an")


def new_hypothesis_id() -> str:
    return _new("hy")
