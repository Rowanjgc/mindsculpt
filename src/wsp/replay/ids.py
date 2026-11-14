"""Identifier helpers for replay plans, replays, and alignment records."""

from __future__ import annotations

import secrets


def _new(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def new_plan_id() -> str:
    return _new("rp")


def new_replay_id() -> str:
    return _new("ry")


def new_alignment_id() -> str:
    return _new("al")


def new_capture_id() -> str:
    return _new("tc")


def new_comparison_id() -> str:
    return _new("cmp")


def new_evidence_id() -> str:
    return _new("inv")
