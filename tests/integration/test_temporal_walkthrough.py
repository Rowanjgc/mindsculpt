"""The documented V1.4 temporal-intervention walkthrough must keep working."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "temporal_intervention_walkthrough.py"


def test_temporal_walkthrough_runs_offline(tmp_path: Path):
    workspace = tmp_path / "temporal-demo"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--workspace", str(workspace)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout

    for marker in (
        "[1] deterministic transformer",
        "[2] baseline trace",
        "[3] selected generation step",
        "[4] plan pl_",
        "[5] validated",
        "[6] identity plan pl_",
        "[7] baseline and identity conditions run",
        "[8] identity control exact",
        "[9] scheduled intervention executed",
        "[10] event log:",
        "[11] downstream: first divergence at",
        "[12] second plan pl_",
        "[13] individual effects measured",
        "[14] combined plan executed",
        "[15] control comparison:",
        "[16] interaction contrast",
        "[16b]",
        "[17] staged prefill prepared",
        "[18] staged verification",
        "[19] staged intervention through the model's own cache",
        "[20] case case_",
        "[21] private export",
        "[22] reloaded into a fresh workspace",
        "[23] redacted export",
        "[24] model state preserved",
    ):
        assert marker in output, output

    assert "generation step 2" in output
    assert "leak_scan=clean" in output
    assert "not a discovered circuit" in output
    assert "not proof of a causal circuit" in output
    assert (workspace / "case-private.json").exists()
    assert (workspace / "case-redacted.json").exists()
    assert (workspace / "fresh" / "workspace.json").exists()
