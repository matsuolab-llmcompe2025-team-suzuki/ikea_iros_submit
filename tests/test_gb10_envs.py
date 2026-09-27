"""The GPU probe must fail closed; these tests never load torch or use a GPU."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FAKE_PROBE = """\
import os, sys
if sys.argv[0].endswith('/python'):
    name = 'pick'
elif '-e' in sys.argv:
    name = sys.argv[sys.argv.index('-e') + 1]
else:
    name = 'desktop'
if name == os.environ.get('FAIL_PROBE'):
    print('CUDA probe failed', file=sys.stderr)
    sys.exit(19)
print(name + ' matmul_ok True')
"""


def run_probe(tmp_path, failing=""):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for executable in (
        binaries / "pixi",
        tmp_path / "model/subtask_policy_training/.venv/bin/python",
    ):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text(f"#!{sys.executable}\n" + FAKE_PROBE)
        executable.chmod(0o755)
    return subprocess.run(
        ["bash", str(ROOT / "tools/gb10/check_envs.sh")],
        env={**os.environ, "PATH": f"{binaries}:/usr/bin:/bin",
             "RAMEN_ROOT": str(tmp_path), "FAIL_PROBE": failing},
        capture_output=True, text=True, timeout=10,
    )


def test_all_four_environments_must_pass(tmp_path):
    result = run_probe(tmp_path)
    assert result.returncode == 0, result.stderr
    for name in ("runtime", "desktop", "vlm", "pick"):
        assert f"[{name}] {name} matmul_ok True" in result.stdout


@pytest.mark.parametrize("name", ["runtime", "desktop", "vlm", "pick"])
def test_any_gpu_environment_error_fails_the_script(tmp_path, name):
    result = run_probe(tmp_path, name)
    assert result.returncode != 0
    assert f"[{name}] FAILED" in result.stderr
    assert "CUDA probe failed" in result.stderr
