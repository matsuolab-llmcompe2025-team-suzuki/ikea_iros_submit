"""運営の conformance.py を、今の components/ と ramen/ (本番と同じ受け口・送り口) で回す。

:5555 / :5556 / :5557 を 127.0.0.1 で使うので、同じ machine で他に使っていると落ちる。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SUBMIT_ROOT = Path(__file__).resolve().parents[1]


def test_conformance_passes_on_the_decoupled_lane() -> None:
    result = subprocess.run(
        [sys.executable, "conformance.py", "--lane", "decoupled"],
        cwd=SUBMIT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert "40 accepted, 0 rejected" in output
