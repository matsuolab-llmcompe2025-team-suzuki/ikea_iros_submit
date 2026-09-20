"""One-at-a-time policy loading for the Phase 3 physical orchestrator.

Phase 3 contains several large GR00T experts.  Keeping all of them resident at
once exceeds the memory-safe load envelope of the 32 GiB workstation GPU.  This
adapter preserves the model-specific policy classes and configs, but owns at
most one concrete policy for the lifetime of its active skill.
"""

from __future__ import annotations

import gc
import sys
import threading
from typing import Any, Callable


class DeferredPolicy:
    """Load a concrete policy on skill start and release it on skill stop.

    ``loader`` builds the concrete policy.  Issue #141 routes every load through
    ``assembly.load_policy`` so a deferred expert gets the same warm-up as an
    eagerly loaded one; without that, the first tick after a skill switch pays
    the初回 forward cost while the robot is already moving.
    """

    def __init__(
        self, policy_cls: type, *, label: str, loader: Callable[[], Any]
    ) -> None:
        self._policy_cls = policy_cls
        self._loader = loader
        self._label = str(label)
        self._inner: Any | None = None
        self.loaded_during_last_reset = False
        # 先読みの worker thread (ModelResidency) と tick の thread が同じ instance を
        # 触る。守らないと二重に読む / 解放した後に読み込みが終わって孤立する。
        self._lock = threading.RLock()

    @property
    def EXECUTION_HORIZON(self) -> int:  # noqa: N802 - policy protocol constant
        return int(getattr(self._policy_cls, "EXECUTION_HORIZON", 1))

    def build_state_from_raw(self, raw: Any) -> Any:
        """読み込んだ policy に委譲する。

        state の作り方は ckpt によって変わる (Issue #141 P8-4: 約束つきの ckpt は
        版 2)。class 単位では決まらないので instance に聞く。VlaSkill が呼ぶのは
        skill が始まった後 (= reset で読み込み済み) なので、ここで読み込みが走ることは
        通常は無い。
        """
        return self._load().build_state_from_raw(raw)

    def _load(self) -> Any:
        with self._lock:
            if self._inner is None:
                print(f"[policy] loading deferred expert: {self._label}", file=sys.stderr)
                self._inner = self._loader()
                print(f"[policy] deferred expert ready: {self._label}", file=sys.stderr)
            return self._inner

    def reset(self) -> None:
        with self._lock:
            was_loaded = self._inner is not None
            inner = self._load()
            self.loaded_during_last_reset = not was_loaded
        reset = getattr(inner, "reset", None)
        if callable(reset):
            reset()

    def predict(self, observation: Any) -> Any:
        return self._load().predict(observation)

    def release_after_skill(self) -> None:
        self.close()

    def validate_load_and_release(self) -> None:
        """Command-free artifact load check used by ``--stage`` dry-runs."""

        self._load()
        self.close()

    def prepare(self) -> None:
        """Preload the first expert before the physical actuation gate."""

        self._load()

    def close(self) -> None:
        with self._lock:
            inner = self._inner
            self._inner = None
            self.loaded_during_last_reset = False
        if inner is None:
            return
        try:
            close = getattr(inner, "close", None)
            if callable(close):
                close()
        finally:
            del inner
            gc.collect()
            # Do not import torch solely for cleanup.  If a policy imported it,
            # release allocator caches before the next expert is constructed.
            torch = sys.modules.get("torch")
            if torch is not None:
                cuda = getattr(torch, "cuda", None)
                if cuda is not None and cuda.is_available():
                    cuda.empty_cache()
            print(
                f"[policy] deferred expert released: {self._label}",
                file=sys.stderr,
            )
