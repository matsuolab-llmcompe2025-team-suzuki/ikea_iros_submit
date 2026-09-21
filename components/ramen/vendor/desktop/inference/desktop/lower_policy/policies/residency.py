"""GPU に置く model の数を決める (Issue #141 束 1-12 / D7-2)。

# なぜ要るか

stage の途中で skill が切り替わるたびに次の model を読むと、その skill の 1 tick 目が
model の読み込み (RAMEN-Ori で 3.2 s = 30 Hz の 100 tick 分) を待つことになる。かといって
全部を GPU に載せると、実機の Desktop でも載り切らない組み合わせがある。

そこで「実行中の model + その次の N-1 個」を GPU に置く形にして、N を起動引数で選べる
ようにする (`--gpu-models`)。N=1 は今までどおり「実行中の 1 個だけ」。

# 読み込みの走らせ方

先読みは **1 本の worker thread** で順番に行う。skill の切れ目を待つ形にはしない
(rotate → pick → insert → rotate_leg のように頭の手順が入らない繋ぎがあるため)。
同時に 2 つ読まないので、GPU に一時的に N+1 個載ることはない。

既定は N=1 (2026-09-15)。実機で「先読み中の tick の周期」と「GPU の使用量」を測って
から上げる。
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Any, Optional


class ModelResidency:
    """skill の列に沿って、GPU に置く model を N 個に保つ。

    Args:
        order: stage の中で回る skill 名の列 (頭の手順を含んでよい。model を持たない
            skill は無視される)。
        policies: skill 名 → `DeferredPolicy`。`prepare()` と `close()` を持つもの。
        resident: GPU に置く model の数 (1 以上)。列の長さ以上なら全部載せる。
    """

    def __init__(
        self,
        order: list[str],
        policies: dict[str, Any],
        *,
        resident: int,
    ) -> None:
        if resident < 1:
            raise ValueError(f"resident must be >= 1, got {resident}")
        self._order = [name for name in order if name in policies]
        self._policies = dict(policies)
        self._resident = int(resident)
        self._loaded: set[str] = set()
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

    @property
    def resident(self) -> int:
        return self._resident

    @property
    def loaded(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._loaded))

    def prime(self) -> None:
        """列の最初の model をこの thread で読み、残りを先読みに積む。

        stage を始める前に呼ぶ。最初の 1 個だけは同期で読む (1 tick 目が読み込みを
        待つと、開始直後に 3 秒動かないことになる)。
        """
        first = next((name for name in self._order), None)
        if first is None:
            return
        prepare = getattr(self._policies[first], "prepare", None)
        if callable(prepare):
            print(f"[residency] loading {first} before the stage starts", file=sys.stderr)
            prepare()
        with self._lock:
            self._loaded.add(first)
        self.on_skill_started(first)

    def on_skill_started(self, skill_name: Optional[str]) -> None:
        """active な skill が変わったときに呼ぶ。保つ範囲を更新する。

        範囲から外れた model はその場で解放し (呼び出し元の thread)、範囲に入って
        まだ読んでいない model は worker thread に積む。
        """
        if skill_name is None or skill_name not in self._order:
            return
        index = self._order.index(skill_name)
        keep = self._order[index : index + self._resident]
        with self._lock:
            release = [name for name in self._loaded if name not in keep]
            # ⚠️ 自分の集合だけを信じない。`VlaSkill._on_stop()` が skill 停止時に
            # `release_after_skill()` を呼び、**ここを通らずに解放する**ので、
            # 「読んだつもりで実は無い」が起きる。そうなると先読みが積まれず、
            # 次に要るときその場で読む (実測 8 秒)。policy が言える場合は実体を聞く。
            missing = [
                name
                for name in keep
                if name not in self._loaded or not self._is_loaded(name)
            ]
            # 先に「読んだ」ことにしておく (同じ model を 2 度積まない)。
            self._loaded.update(missing)
        for name in release:
            self._release(name)
        if not missing:
            return
        self._ensure_worker()
        for name in missing:
            self._queue.put(name)

    def _is_loaded(self, name: str) -> bool:
        """policy が `is_loaded` を持つならそれを、無ければ集合を信じる。"""
        loaded = getattr(self._policies[name], "is_loaded", None)
        return bool(loaded) if isinstance(loaded, bool) else True

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run, name="model-preload", daemon=True
        )
        self._worker.start()

    def _run(self) -> None:
        while True:
            name = self._queue.get()
            if name is None:
                return
            try:
                prepare = getattr(self._policies[name], "prepare", None)
                if callable(prepare):
                    print(f"[residency] preloading {name}", file=sys.stderr)
                    prepare()
                    print(f"[residency] preloaded {name}", file=sys.stderr)
            except Exception as exc:     # 先読みの失敗で run を落とさない
                # 実際に使うときに同じ読み込みが走り、そこで例外が出る。
                with self._lock:
                    self._loaded.discard(name)
                print(f"[residency] preload failed for {name}: {exc!r}", file=sys.stderr)
            finally:
                self._queue.task_done()

    def _release(self, name: str) -> None:
        with self._lock:
            self._loaded.discard(name)
        close = getattr(self._policies[name], "close", None)
        if callable(close):
            print(f"[residency] releasing {name}", file=sys.stderr)
            close()

    def close(self) -> None:
        """worker を止め、この管理が載せた model を解放する。

        stage ごとに作り直す (`--phase3-full`) ので、ここで解放しないと次の stage の
        `_loaded` は空のまま前の stage の model が GPU に残り、誰も解放できなくなる。
        """
        if self._worker is not None and self._worker.is_alive():
            self._queue.put(None)
            self._worker.join(timeout=30.0)
        self._worker = None
        with self._lock:
            remaining = tuple(self._loaded)
        for name in remaining:
            self._release(name)
