"""pick-leg GR00T worker が vendor tree の階層で起動できることを確認する。

GPU/Thor/HF 不要。`_PickLegsWorkerClient.__init__` を実際に走らせ、Popen 直前まで
到達したコマンドを見る。

なぜ要るか:
  本家 `groot_pick_legs.py` は `repo_root = Path(__file__).resolve().parents[4]` で
  repo root を取るが、vendor tree は `components/ramen/vendor/{desktop,model}` と
  desktop/ が 1 段余分に挟まるので parents[4] は `vendor/desktop` を指し、その下に
  model/ は無い。加えて worker interpreter が dev 専用の
  `model/subtask_policy_training/.venv/bin/python` 決め打ちで container には無い。
  どちらも fallback が無いため `pick_table_leg` = **Stage 1 の先頭**が
  FileNotFoundError で落ちる。`tools/vendor_patches.py` がこれを直している。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE.parent / "vendor"
_VENDOR_DESKTOP = _VENDOR / "desktop"
for _p in (str(_VENDOR_DESKTOP), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference.desktop.lower_policy.policies import groot_pick_legs as _mod
from inference.desktop.lower_policy.policies.base import PolicyConfig

_WORKER_SCRIPT = (
    _VENDOR / "model/subtask_policy_training/deployment/real_groot_n17_worker.py"
)


class _Spawned(Exception):
    """Popen まで到達した (= runtime 解決が通った) ことを示す sentinel。"""


@pytest.fixture
def spawn_capture(monkeypatch):
    """subprocess.Popen を差し替えて引数だけ捕まえる。worker は起動しない。"""
    captured: dict = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["cwd"] = kwargs.get("cwd")
        raise _Spawned

    monkeypatch.setattr(_mod.subprocess, "Popen", _fake_popen)
    return captured


def _cfg(ckpt_dir: Path) -> PolicyConfig:
    # ckpt_ref を実在 directory にすると _resolve_checkpoint が HF download を
    # 迂回する (reference.is_dir() 分岐)。ネットワークに触らせない。
    # 中身は見られないが存在チェックはあるので、空ファイルを置いておく。
    for name in (
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
    ):
        (ckpt_dir / name).touch()
    return PolicyConfig(mode="none", ckpt_ref=str(ckpt_dir))


def test_worker_script_resolves_inside_the_vendor_tree(
    tmp_path, monkeypatch, spawn_capture
):
    """parents[4] のずれを吸収して vendor/model/ の worker script に届く。"""
    monkeypatch.setenv("RAMEN_WORKER_PYTHON", sys.executable)

    with pytest.raises(_Spawned):
        _mod._PickLegsWorkerClient(_cfg(tmp_path))

    cmd = spawn_capture["cmd"]
    assert cmd[0] == sys.executable
    assert Path(cmd[1]) == _WORKER_SCRIPT.resolve()
    assert Path(cmd[1]).is_file(), "vendor tree の worker script が見つからない"
    # worker は vendor/ を cwd に期待する (components/ramen/groot_worker.py と同じ)。
    assert Path(spawn_capture["cwd"]).resolve() == _VENDOR.resolve()


def test_worker_python_comes_from_the_env(tmp_path, monkeypatch, spawn_capture):
    """container には .venv が無いので RAMEN_WORKER_PYTHON を使う。

    PATH 上の名前 (`python3` 等) で渡されても解決できること。
    """
    monkeypatch.setenv("RAMEN_WORKER_PYTHON", "python3")

    with pytest.raises(_Spawned):
        _mod._PickLegsWorkerClient(_cfg(tmp_path))

    resolved = Path(spawn_capture["cmd"][0])
    assert resolved.is_file(), f"PATH 名を解決できていない: {resolved}"


def test_missing_worker_python_names_the_env_var(tmp_path, monkeypatch, spawn_capture):
    """env も .venv も無ければ、何を設定すべきか言って落ちる。"""
    monkeypatch.delenv("RAMEN_WORKER_PYTHON", raising=False)

    with pytest.raises(FileNotFoundError, match="RAMEN_WORKER_PYTHON"):
        _mod._PickLegsWorkerClient(_cfg(tmp_path))


def test_vendor_patches_are_applied_to_the_committed_tree():
    """rsync 後に tools/vendor_patches.py を流し忘れた状態を検出する。

    sync_vendor_desktop.sh は rsync 直後に呼ぶが、手で rsync した場合に漏れる。
    """
    sys.path.insert(0, str(_HERE.parents[2] / "tools"))
    import vendor_patches

    for patch in vendor_patches.PATCHES:
        status = vendor_patches.apply_patch(_VENDOR_DESKTOP, patch)
        assert status == "already-applied", (
            f"{patch.path} に vendor patch が当たっていない。"
            f" tools/vendor_patches.py を流すこと。\n  {patch.why}"
        )
