"""The documented V1.2 walkthrough must keep working end to end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "investigation_walkthrough.py"


def test_walkthrough_runs_offline(tmp_path: Path):
    workspace = tmp_path / "walkthrough"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--workspace", str(workspace)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout

    # Every stage of the documented chain must have reported something.
    for marker in (
        "[1] observed trace",
        "[4] detector",
        "[5b] finding span resolved",
        "[6] replay plan",
        "[7] replay",
        "[8] captured",
        "[10] identical-context control",
        "[10b] different-input comparison",
        "[11b] evidence",
        "[14] integrity",
        "[15] parameter digest unchanged",
        "[16] redacted export",
    ):
        assert marker in output, output

    assert "generation_reproducibility=not-claimed" in output
    assert "l2=0.0000" in output
    assert "zero difference where the context is identical" in output
    assert (workspace / "case-private.json").exists()
    assert (workspace / "case-redacted.json").exists()
