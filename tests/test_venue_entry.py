"""提出 image の起動口 (docker/venue_entry.sh) の test。

偽の pixi で script を走らせ、entrypoint に何が渡るかを見る。
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SUBMIT_ROOT = Path(__file__).resolve().parents[1]
ENTRY = SUBMIT_ROOT / "docker" / "venue_entry.sh"
FIXED = [
    "--action-sink", "boundary",
    "--synthetic-hand-state",
    "--boundary-host", "0.0.0.0",
    "--spawn-vlm-server",
    "--gpu-models", "all",
]
DOCKERFILE = SUBMIT_ROOT / "docker" / "Dockerfile.thor"


def _run(tmp_path: Path, *args: str, orin_host: str | None = "192.168.123.164"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    pixi = bin_dir / "pixi"
    pixi.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        'print(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd()}))\n'
    )
    pixi.chmod(pixi.stat().st_mode | stat.S_IXUSR)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "RAMEN_ROOT": str(tmp_path)}
    if orin_host is not None:
        env["IROS_ORIN_HOST"] = orin_host
    return subprocess.run(
        ["bash", str(ENTRY), *args], env=env, capture_output=True, text=True, timeout=30
    )


def test_a_venue_run_adds_the_fixed_options_before_the_operators_args(tmp_path) -> None:
    result = _run(tmp_path, "--stage", "2", "--actuate")
    assert result.returncode == 0, result.stderr
    ran = json.loads(result.stdout)
    assert ran["cwd"] == str(tmp_path.resolve())
    assert ran["argv"] == [
        "run", "--as-is", "-e", "runtime", "python", "-m", "inference.desktop.entrypoint",
        *FIXED, "--stage", "2", "--actuate",
    ]


def test_a_venue_run_without_the_pc2_address_stops(tmp_path) -> None:
    result = _run(tmp_path, "--stage", "2", orin_host=None)
    assert result.returncode == 2
    assert "IROS_ORIN_HOST" in result.stderr
    assert result.stdout == ""  # pixi は呼ばれていない


def test_other_commands_run_as_given(tmp_path) -> None:
    result = _run(tmp_path, "bash", "-c", "echo passthrough", orin_host=None)
    assert result.returncode == 0
    assert result.stdout.strip() == "passthrough"


def test_no_arguments_print_the_usage(tmp_path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 2
    assert "--stage N" in result.stderr


def _image_env() -> dict[str, str]:
    """Dockerfile.thor の ENV 命令 (行の継続を含む) を KEY=VALUE で集める。"""
    text = DOCKERFILE.read_text().replace("\\\n", " ")
    env: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("ENV "):
            for item in line[len("ENV "):].split():
                key, _, value = item.partition("=")
                env[key] = value
    return env


def test_the_image_keeps_every_library_offline() -> None:
    """会場は実行時にネットに出ない。重み (HF / transformers) と YOLO の問い合わせを image で止める。

    ultralytics は import 時に DNS でネットの有無を調べ、推論の開始時に Google Analytics へ
    利用統計を送っていた (GB10 で strace、2026-09-25)。YOLO_OFFLINE=true で両方止まる。
    """
    env = _image_env()
    assert env.get("HF_HUB_OFFLINE") == "1"
    assert env.get("TRANSFORMERS_OFFLINE") == "1"
    assert env.get("YOLO_OFFLINE") == "true"


def test_the_fixed_options_exist_in_the_copied_entrypoint() -> None:
    """本体側で option の名前が変わったら、ここで分かる (ramen/ は本体のコピー)。"""
    result = subprocess.run(
        [sys.executable, "-m", "inference.desktop.entrypoint", "--help"],
        cwd=SUBMIT_ROOT / "ramen",
        capture_output=True,
        text=True,
        timeout=120,
        # ramen/ は本体のコピーなので、bytecode も残さない
        env={**os.environ, "PYTHONPATH": str(SUBMIT_ROOT / "ramen"), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    for option in (o for o in FIXED if o.startswith("--")):
        assert option in result.stdout, option
