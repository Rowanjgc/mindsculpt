"""Identifier helpers for alignment, capture, and comparison records."""

from __future__ import annotations

import secrets


def _new(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def new_alignment_id() -> str:
    return _new("al")


def new_span_id() -> str:
    return _new("sp")


def new_capture_id() -> str:
    return _new("tc")


def new_comparison_id() -> str:
    return _new("cmp")


def new_evidence_id() -> str:
    return _new("inv")


__all__ = [
    "new_alignment_id",
    "new_capture_id",
    "new_comparison_id",
    "new_evidence_id",
    "new_span_id",
]
