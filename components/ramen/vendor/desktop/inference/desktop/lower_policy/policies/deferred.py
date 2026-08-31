"""One-at-a-time policy loading for the Phase 3 physical orchestrator.

Phase 3 contains several large GR00T experts.  Keeping all of them resident at
once exceeds the memory-safe load envelope of the 32 GiB workstation GPU.  This
adapter preserves the model-specific policy classes and configs, but owns at
most one concrete policy for the lifetime of its active skill.
"""

from __future__ import annotations

import gc
import sys
from typing import Any


class DeferredPolicy:
    """Load a concrete policy on skill start and release it on skill stop."""

    def __init__(self, policy_cls: type, policy_config: Any, *, label: str) -> None:
        self._policy_cls = policy_cls
        self._policy_config = policy_config
        self._label = str(label)
        self._inner: Any | None = None
        self.loaded_during_last_reset = False

    @property
    def EXECUTION_HORIZON(self) -> int:  # noqa: N802 - policy protocol constant
        return int(getattr(self._policy_cls, "EXECUTION_HORIZON", 1))

    def build_state_from_raw(self, raw: Any) -> Any:
        return self._policy_cls.build_state_from_raw(raw)

    def _load(self) -> Any:
        if self._inner is None:
            print(f"[policy] loading deferred expert: {self._label}", file=sys.stderr)
            self._inner = self._policy_cls.from_ckpt(self._policy_config)
            print(f"[policy] deferred expert ready: {self._label}", file=sys.stderr)
            return self._inner
        return self._inner

    def reset(self) -> None:
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
