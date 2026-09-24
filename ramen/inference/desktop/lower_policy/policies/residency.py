"""GPU に置く model の数を決める (Issue #141 束 1-12 / D7-2)。

# なぜ要るか

stage の途中で skill が切り替わるたびに次の model を読むと、その skill の 1 tick 目が
model の読み込み (RAMEN-Ori で 3.2 s = 30 Hz の 100 tick 分) を待つことになる。かといって
全部を GPU に載せると、実機の Desktop でも載り切らない組み合わせがある。

そこで「実行中の model + その次の N-1 個」を GPU に置く形にして、N を起動引数で選べる
ようにする (`--gpu-models`)。N=1 は今までどおり「実行中の 1 個だけ」。

# 読み込みの走らせ方

読込は **1 本の worker thread** で順番に行う。実機の model 間に有限
transition がある場合、旧 model を保持したまま腕・手を次 model の
frame-zero 姿勢へ収束させる。最後の ``hold_transition_*`` に入ってから
初めて旧 model を解放し、次 model を読む。

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
        self._timeline = list(order)
        self._order = [name for name in self._timeline if name in policies]
        self._policies = dict(policies)
        self._resident = int(resident)
        self._loaded: set[str] = set()
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[tuple[str, str]]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        # 積んだまだ終わっていない操作 (操作, 名前)。**先読み中の model には
        # `is_loaded` を聞かない** (Issue #159 B1b-1)。聞くと worker が `_load()` 中に
        # 握っている lock を待ち、呼び出し元 (制御 loop / driver の request thread) が
        # 読み込み完了まで止まる (pod 実測 8〜27 s)。
        self._pending: set[tuple[str, str]] = set()
        self._errors: dict[str, str] = {}
        # 先読みが失敗した model → そのとき走っていた skill。**同じ skill の間は
        # 積み直さない** (Issue #159 B3a-01)。積み直すと失敗する読み込みを延々と
        # 繰り返す。skill が変われば 1 回だけ試し直す。
        self._failed: dict[str, Optional[str]] = {}
        self._active_skill: Optional[str] = None
        self._switch_release_error: Optional[str] = None
        for policy in self._policies.values():
            setter = getattr(policy, "set_residency_managed", None)
            if callable(setter):
                setter(True)

    @property
    def resident(self) -> int:
        return self._resident

    @property
    def loaded(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._loaded))

    @property
    def first_model_waits_for_switch_hold(self) -> bool:
        """最初の model の前に model 載せ替えの HOLD (`hold_transition_*`) があるか。

        ある stage (通し実行の Stage 2〜5) では、最初の model は HOLD の中で読む。
        `prime()` で先に同期で読むと、次の初期姿勢へ動く前に制御 loop の外で
        読み込むことになる (Codex #9)。
        """
        first = next((name for name in self._order), None)
        if first is None:
            return False
        before = self._timeline[: self._timeline.index(first)]
        return any(name.startswith("hold_transition_") for name in before)

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

        範囲から外れた model の解放も、新たな model の読込も worker thread に積む。
        checkpoint worker の終了待ちを制御 tick で行ってはいけない。
        """
        if skill_name is None or skill_name not in self._timeline:
            return
        # Loading a multi-GiB checkpoint during hand/arm motion can starve the
        # control loop.  The final hold is the sole model-switch point.
        if skill_name.startswith(
            ("hand_release_", "arm_transition_", "hand_transition_")
        ):
            return
        timeline_index = self._timeline.index(skill_name)
        next_model = next(
            (
                name
                for name in self._timeline[timeline_index:]
                if name in self._policies
            ),
            None,
        )
        if next_model is None:
            return
        index = self._order.index(next_model)
        keep = self._order[index : index + self._resident]
        # Never preload across another physical pose boundary, even when the
        # configured residency count is greater than one.
        if len(keep) > 1:
            second_model_at = next(
                (
                    i
                    for i, name in enumerate(
                        self._timeline[timeline_index + 1 :],
                        start=timeline_index + 1,
                    )
                    if name == keep[1]
                ),
                None,
            )
            if second_model_at is not None and any(
                name.startswith("hold_transition_")
                for name in self._timeline[timeline_index + 1 : second_model_at]
            ):
                keep = keep[:1]
        with self._lock:
            self._active_skill = skill_name
            release = [
                name
                for name in self._loaded
                if name not in keep and ("release", name) not in self._pending
            ]
            # 自分の集合だけを信じない。初期化失敗や外部 cleanup が直接 close
            # した場合もあるので、policy が言える場合は実体を確認する。ただし
            # 先読み中のもの (`("prepare", name)`) には聞かない (上の B1b-1。条件の
            # 順番が大事: `_is_loaded` を最後に評価する)。
            missing = [
                name
                for name in keep
                if ("prepare", name) not in self._pending
                and self._failed.get(name, "\0") != skill_name
                and (name not in self._loaded or not self._is_loaded(name))
            ]
            # 先に「読んだ」ことにしておく (同じ model を 2 度積まない)。
            self._loaded.update(missing)
            for name in release:
                self._loaded.discard(name)
                self._pending.add(("release", name))
            if release:
                self._switch_release_error = None
            for name in missing:
                self._errors.pop(name, None)
                self._failed.pop(name, None)
                self._pending.add(("prepare", name))
        if not release and not missing:
            return
        if skill_name.startswith("hold_transition_"):
            print(
                f"[residency] frame-zero HOLD reached; switching to {next_model}",
                file=sys.stderr,
            )
        self._ensure_worker()
        # Release precedes prepare so resident=N never transiently becomes N+1.
        for name in release:
            self._queue.put(("release", name))
        for name in missing:
            self._queue.put(("prepare", name))

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
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            operation, name = item
            try:
                if operation == "release":
                    self._release(name, update_bookkeeping=False)
                else:
                    with self._lock:
                        release_error = self._switch_release_error
                    if release_error is not None:
                        raise RuntimeError(
                            "previous model release failed; refusing to load the "
                            f"next model: {release_error}"
                        )
                    prepare = getattr(self._policies[name], "prepare", None)
                    if callable(prepare):
                        print(f"[residency] preloading {name}", file=sys.stderr)
                        prepare()
                        print(f"[residency] preloaded {name}", file=sys.stderr)
                    with self._lock:
                        self._errors.pop(name, None)
            except Exception as exc:     # 先読みの失敗で run を落とさない
                # 実際に使うときに同じ読み込みが走り、そこで例外が出る。
                with self._lock:
                    self._loaded.discard(name)
                    self._errors[name] = repr(exc)
                    if operation == "release":
                        self._switch_release_error = repr(exc)
                    else:
                        self._failed[name] = self._active_skill
                    skill = self._active_skill
                failure_label = "preload" if operation == "prepare" else operation
                print(
                    f"[residency] 🔴 {failure_label} failed for {name}: {exc!r} "
                    f"(not retried while {skill!r} is running)",
                    file=sys.stderr,
                )
            finally:
                with self._lock:
                    self._pending.discard((operation, name))
                self._queue.task_done()

    def is_ready(self, name: str) -> bool:
        """Return true only after ``name`` finished loading successfully."""

        if name not in self._policies:
            return True
        with self._lock:
            tracked = name in self._loaded
            pending = ("prepare", name) in self._pending
            failed = name in self._errors
        return tracked and not pending and not failed and self._is_loaded(name)

    def error_for(self, name: str) -> Optional[str]:
        """Return the last asynchronous load failure for ``name``."""

        with self._lock:
            return self._errors.get(name)

    def _release(self, name: str, *, update_bookkeeping: bool = True) -> None:
        if update_bookkeeping:
            with self._lock:
                self._loaded.discard(name)
        close = getattr(self._policies[name], "release_from_residency", None)
        if not callable(close):
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
            self._queue.join()
            self._queue.put(None)
            self._worker.join(timeout=30.0)
        self._worker = None
        with self._lock:
            remaining = tuple(self._loaded)
        for name in remaining:
            self._release(name)
        for policy in self._policies.values():
            setter = getattr(policy, "set_residency_managed", None)
            if callable(setter):
                setter(False)
