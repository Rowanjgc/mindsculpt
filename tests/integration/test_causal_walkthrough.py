"""The documented V1.3 causal-investigation walkthrough must keep working."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "causal_investigation.py"


def test_causal_walkthrough_runs_offline(tmp_path: Path):
    workspace = tmp_path / "causal-demo"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--workspace", str(workspace)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout

    for marker in (
        "[1] deterministic transformer",
        "[2] behavioral traces",
        "[3] replay",
        "[4] captured",
        "[5] identity intervention",
        "[6] identity control exact",
        "[8] paired measurement",
        "[9] donor activation",
        "[10] donor patch",
        "[11] downstream logits",
        "[12] greedy continuations",
        "[13] first divergence",
        "[14] case",
        "[15] private export",
        "[16] reloaded into a fresh workspace",
        "[17] checksums and references verified",
        "[18] parameter digest unchanged",
        "[19] redacted export",
    ):
        assert marker in output, output

    assert "leak_scan=clean" in output
    assert "not proof of a causal circuit" in output
    assert (workspace / "case-private.json").exists()
    assert (workspace / "case-redacted.json").exists()
    assert (workspace / "fresh" / "workspace.json").exists()
