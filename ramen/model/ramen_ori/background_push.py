"""ckpt の HF push を学習の process の外で 1 本ずつ流す (Issue #141)。

push を学習 loop の中で呼ぶと、3.5 GB の ckpt を 100 Mbps で上げる約 300 s の間、学習が止まる
(10k step ごと、1 run で約 49 分 = 壁時計時間の約 2 割。2026-09-14 の c32 / c16 で計測)。

thread ではなく別 process にする理由:
- val の DataLoader は 1k step ごとに worker を fork する。upload の thread が lock を持ったまま fork すると、
  子の worker が詰まることがある
- upload (hf_xet) が異常終了しても、学習の process は巻き込まれない
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PUSH_SCRIPT = Path(__file__).resolve().parent / "scripts" / "push_ckpt_to_hf.py"


class BackgroundCkptPusher:
    """`scripts/push_ckpt_to_hf.py` を別 process で起動し、push を投げた順に 1 本ずつ流す。

    - `submit` は push の終わりを待たない。ただし前の push がまだ走っていれば、それが終わるまで待つ
      (HF の commit を step の順に積み、2 本で帯域を取り合わないため。ckpt は 5k step ごと・push は 10k step ごとなので、
      普段は前の push は終わっている)
    - 学習の終わりに `wait` を呼ぶ。最後の ckpt の push が終わるまで待つので、run の `.done` は
      「全 ckpt が HF にある」の意味のまま
    - 子 process の stdout / stderr は学習の log にそのまま流れる。HF の進捗バー (tqdm、改行なし) は
      `[step ...]` の行に混ざって行頭で拾えなくなるので、子 process の環境変数で止める
    - 学習の process が落ちても子 process は push を続ける (再開用の ckpt が HF に届く)
    """

    def __init__(self, repo_id: str, private: bool = True, script: Path = PUSH_SCRIPT) -> None:
        self.repo_id = repo_id
        self.private = private
        self.script = Path(script)
        # push の script が無いと 10k step 目まで気づかないので、学習を始める前に止める
        if not self.script.is_file():
            raise FileNotFoundError(f"push script not found: {self.script}")
        self.failed: list[str] = []  # push が失敗した ckpt の名前
        self._proc: subprocess.Popen | None = None
        self._ckpt_name: str | None = None

    def submit(self, ckpt_path: Path, step: int) -> None:
        """ckpt の push を別 process で始める (前の push が走っていれば、その終わりだけ待つ)。"""
        name = Path(ckpt_path).name
        waited = self._wait_current()
        if waited >= 1.0:
            print(f"[hf_autopush] waited {waited:.0f}s for the previous push before {name}", flush=True)
        cmd = [
            sys.executable, str(self.script),
            "--ckpt", str(ckpt_path), "--repo-id", self.repo_id, "--step", str(step),
        ]
        if not self.private:
            cmd.append("--public")
        try:
            self._proc = subprocess.Popen(cmd, env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        except OSError as e:
            # 起動できなくても学習は止めない (push の失敗と同じ扱い)
            self.failed.append(name)
            print(f"[hf_autopush] WARN: could not start push of {name} ({type(e).__name__}: {e})", flush=True)
            return
        self._ckpt_name = name
        print(f"[hf_autopush] pushing {name} in background (pid {self._proc.pid})", flush=True)

    def wait(self) -> None:
        """走っている push が終わるまで待つ (学習の最後に 1 回呼ぶ)。"""
        waited = self._wait_current()
        print(
            f"[hf_autopush] all pushes finished (waited {waited:.0f}s at the end, "
            f"failed {len(self.failed)}: {self.failed})",
            flush=True,
        )

    def _wait_current(self) -> float:
        """走っている push の終わりを待ち、待った秒数を返す。exit code が 0 でなければ failed に残す。"""
        if self._proc is None:
            return 0.0
        t0 = time.time()
        rc = self._proc.wait()
        if rc != 0:
            self.failed.append(self._ckpt_name)
            print(f"[hf_autopush] WARN: push of {self._ckpt_name} exited with {rc}", flush=True)
        self._proc = None
        self._ckpt_name = None
        return time.time() - t0
