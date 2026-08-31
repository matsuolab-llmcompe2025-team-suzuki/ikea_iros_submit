"""Model-agnostic asynchronous action-chunk replanning.

The physical command loop must never block on neural-network inference.  This
module deliberately knows nothing about cameras, model state, action units, or
robot joints: each model runner owns those contracts and submits a callable
that returns an already decoded and validated ``[horizon, action_dim]`` chunk.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import math
import sys
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Generic, Iterable, TypeVar

import numpy as np


PredictionCallable = Callable[
    [], tuple[np.ndarray, float, dict[str, Any]]
]
CleanupStep = tuple[str, Callable[[], None]]
PredictionResult = TypeVar("PredictionResult")


@dataclass(frozen=True)
class ReplanningProfile:
    lead_steps: int
    max_prediction_age_s: float


FAMILY_REPLANNING_PROFILES = MappingProxyType({
    "act_absolute_joint16_v1": ReplanningProfile(4, 0.30),
    "groot_absolute_joint_v1": ReplanningProfile(4, 0.30),
    "groot_relative_eef_v1": ReplanningProfile(4, 0.35),
    "diffusion_chunk_relative_v1": ReplanningProfile(2, 0.20),
    "diffusion_absolute_v1": ReplanningProfile(2, 0.20),
})


def family_replanning_schedule(
    family: str,
    execution_steps: int,
) -> tuple[int, float]:
    """Return the trusted local prefetch boundary and request-age budget."""

    try:
        profile = FAMILY_REPLANNING_PROFILES[family]
    except KeyError as exc:
        raise ValueError(f"no asynchronous replanning profile for {family!r}") from exc
    if execution_steps < 1:
        raise ValueError("execution_steps must be positive")
    return (
        max(0, execution_steps - profile.lead_steps),
        profile.max_prediction_age_s,
    )


class AsyncPredictionTask(Generic[PredictionResult]):
    """One daemonized prediction with bounded, abortable shutdown.

    This lower-level primitive is for policy families whose temporal ensemble
    owns chunk promotion. Most runners should use
    :class:`AsyncActionChunkPipeline` instead.
    """

    def __init__(
        self,
        predictor: Callable[[], PredictionResult],
        *,
        thread_name: str,
    ) -> None:
        if not thread_name:
            raise ValueError("thread_name must not be empty")
        self._future: Future[PredictionResult] = Future()

        def run_and_publish() -> None:
            try:
                self._future.set_result(predictor())
            except BaseException as exc:  # preserve the exact model failure
                self._future.set_exception(exc)

        self._thread = threading.Thread(
            target=run_and_publish,
            name=thread_name,
            daemon=True,
        )
        self._thread.start()
        self._closed = False
        self.shutdown_aborted = False

    def done(self) -> bool:
        return self._future.done()

    def result(self) -> PredictionResult:
        return self._future.result()

    def close(
        self,
        *,
        timeout_s: float = 0.5,
        abort_pending: Callable[[], None] | None = None,
    ) -> None:
        if self._closed:
            return
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        self._closed = True
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_s)
        aborted = False
        if self._thread.is_alive() and abort_pending is not None:
            abort_pending()
            aborted = True
            self.shutdown_aborted = True
            self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise TimeoutError(
                "asynchronous prediction did not stop within the bounded "
                "shutdown interval"
            )
        try:
            self._future.result()
        except BaseException:
            if aborted:
                return
            raise


@dataclass(frozen=True)
class CompletedPrediction:
    """One model-specific, validated prediction produced off the command loop."""

    actions: np.ndarray
    inference_ms: float
    diagnostics: dict[str, Any]
    anchor_generation: tuple[int, ...]
    submitted_monotonic_ns: int
    completed_monotonic_ns: int


class AsyncActionChunkPipeline:
    """Double-buffer action chunks while keeping command cadence deterministic.

    A runner begins executing ``initial_actions`` immediately.  It may submit
    one model-specific prediction after ``replan_after_steps`` have been
    consumed.  The replacement is promoted only after ``execution_steps``;
    if inference is late, the final transmitted model target is held instead
    of blocking or emitting a burst of catch-up commands.
    """

    def __init__(
        self,
        initial_actions: np.ndarray,
        *,
        execution_steps: int,
        replan_after_steps: int,
        max_prediction_age_s: float,
        thread_name_prefix: str = "policy-replan",
    ) -> None:
        values = self._validated_chunk(initial_actions, expected_shape=None)
        horizon = int(values.shape[0])
        if not 1 <= execution_steps <= horizon:
            raise ValueError("execution_steps must lie in [1, action horizon]")
        if not 0 <= replan_after_steps < execution_steps:
            raise ValueError("replan_after_steps must be smaller than execution_steps")
        if not math.isfinite(max_prediction_age_s) or max_prediction_age_s <= 0.0:
            raise ValueError("max_prediction_age_s must be finite and positive")
        if not thread_name_prefix:
            raise ValueError("thread_name_prefix must not be empty")

        self._chunk_shape = values.shape
        self._actions = values.copy()
        self._execution_steps = int(execution_steps)
        self._replan_after_steps = int(replan_after_steps)
        self._max_prediction_age_ns = int(max_prediction_age_s * 1.0e9)
        self._index = 0
        self._last_action = self._actions[0].copy()
        self._thread_name_prefix = thread_name_prefix
        self._thread: threading.Thread | None = None
        self._pending: Future[CompletedPrediction] | None = None
        self._pending_submitted_monotonic_ns: int | None = None
        self._closed = False
        self.deadline_miss_ticks = 0
        self.completed_chunks = 0
        self.stale_discard_count = 0
        self.last_stale_discard_age_ms: float | None = None
        self.shutdown_aborted_prediction = False

    @staticmethod
    def _validated_chunk(
        actions: np.ndarray,
        *,
        expected_shape: tuple[int, int] | None,
    ) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("action chunk must be a non-empty rank-2 array")
        if expected_shape is not None and values.shape != expected_shape:
            raise ValueError(
                f"replanned action shape changed: {values.shape} != {expected_shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("action chunk must contain only finite values")
        return values

    @property
    def chunk_shape(self) -> tuple[int, int]:
        return self._chunk_shape

    @property
    def execution_steps(self) -> int:
        return self._execution_steps

    @property
    def prediction_pending(self) -> bool:
        return self._pending is not None

    @property
    def action_index(self) -> int:
        return self._index

    @property
    def wants_prediction(self) -> bool:
        return (
            not self._closed
            and self._pending is None
            and self._index >= self._replan_after_steps
        )

    def submit(
        self,
        predictor: PredictionCallable,
        *,
        anchor_generation: tuple[int, ...],
    ) -> None:
        if self._closed:
            raise RuntimeError("asynchronous replanning pipeline is closed")
        if not self.wants_prediction:
            raise RuntimeError("an asynchronous prediction is already pending or not due")
        submitted_ns = time.monotonic_ns()
        immutable_anchor = tuple(int(value) for value in anchor_generation)

        def run() -> CompletedPrediction:
            actions, latency, diagnostics = predictor()
            values = self._validated_chunk(
                actions,
                expected_shape=self._chunk_shape,
            )
            inference_ms = float(latency)
            if not np.isfinite(inference_ms) or inference_ms < 0.0:
                raise ValueError("inference latency must be finite and non-negative")
            return CompletedPrediction(
                actions=values.copy(),
                inference_ms=inference_ms,
                diagnostics=dict(diagnostics),
                anchor_generation=immutable_anchor,
                submitted_monotonic_ns=submitted_ns,
                completed_monotonic_ns=time.monotonic_ns(),
            )

        pending: Future[CompletedPrediction] = Future()

        def run_and_publish() -> None:
            try:
                pending.set_result(run())
            except BaseException as exc:  # propagate the exact worker failure
                pending.set_exception(exc)

        self._pending = pending
        self._pending_submitted_monotonic_ns = submitted_ns
        # The production predictor waits on an isolated subprocess.  A daemon
        # thread, combined with the runner's abort callback, prevents an
        # unresponsive GPU worker from keeping the Python process alive.
        self._thread = threading.Thread(
            target=run_and_publish,
            name=f"{self._thread_name_prefix}-0",
            daemon=True,
        )
        self._thread.start()

    def promote_if_ready(self) -> CompletedPrediction | None:
        """Promote at a chunk boundary, or report one non-blocking hold tick."""

        if self._closed:
            raise RuntimeError("asynchronous replanning pipeline is closed")
        if self._index < self._execution_steps:
            return None
        if self._pending is None:
            self.deadline_miss_ticks += 1
            return None
        if not self._pending.done():
            submitted_ns = self._pending_submitted_monotonic_ns
            if (
                submitted_ns is not None
                and time.monotonic_ns() - submitted_ns
                > self._max_prediction_age_ns
            ):
                raise TimeoutError(
                    "asynchronous model request exceeded its observation-age "
                    "budget before producing a usable chunk"
                )
            self.deadline_miss_ticks += 1
            return None
        completed = self._pending.result()
        self._pending = None
        self._pending_submitted_monotonic_ns = None
        self._thread = None
        promotion_age_ns = time.monotonic_ns() - completed.submitted_monotonic_ns
        if promotion_age_ns > self._max_prediction_age_ns:
            self.stale_discard_count += 1
            self.last_stale_discard_age_ms = promotion_age_ns / 1.0e6
            return None
        self._actions = completed.actions.copy()
        self._index = 0
        self.completed_chunks += 1
        return completed

    def next_action(self) -> tuple[np.ndarray, int | None]:
        """Return the next model step, or the last step while inference is late."""

        if self._closed:
            raise RuntimeError("asynchronous replanning pipeline is closed")
        if self._index >= self._execution_steps:
            return self._last_action.copy(), None
        index = self._index
        self._last_action = self._actions[index].copy()
        self._index += 1
        return self._last_action.copy(), index

    def close(
        self,
        *,
        timeout_s: float = 0.5,
        abort_pending: Callable[[], None] | None = None,
    ) -> None:
        """Bounded shutdown that can abort an unresponsive model subprocess.

        The predictor thread is daemonized as a final process-exit safeguard,
        but normal physical runners also provide ``abort_pending`` to terminate
        the isolated worker and unblock its pipe read.
        """

        if self._closed:
            return
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        pending = self._pending
        thread = self._thread
        self._closed = True
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
        aborted = False
        if thread is not None and thread.is_alive() and abort_pending is not None:
            abort_pending()
            aborted = True
            self.shutdown_aborted_prediction = True
            thread.join(timeout=timeout_s)
        self._pending = None
        self._pending_submitted_monotonic_ns = None
        self._thread = None
        if thread is not None and thread.is_alive():
            raise TimeoutError(
                "asynchronous model request did not stop within the bounded "
                "shutdown interval"
            )
        if pending is not None:
            try:
                pending.result()
            except BaseException:
                if aborted:
                    # The request is no longer needed after the physical arm
                    # handoff. Terminating its isolated worker is a normal,
                    # bounded cancellation rather than a policy failure.
                    return
                raise


def advance_periodic_deadline(
    scheduled_s: float,
    finished_s: float,
    period_s: float,
) -> float:
    """Skip missed ticks while retaining the original command-loop phase."""

    if not all(math.isfinite(value) for value in (scheduled_s, finished_s, period_s)):
        raise ValueError("periodic deadline inputs must be finite")
    if period_s <= 0.0:
        raise ValueError("period_s must be positive")
    if scheduled_s >= finished_s:
        return scheduled_s
    skipped = math.floor((finished_s - scheduled_s) / period_s) + 1
    return scheduled_s + skipped * period_s


def run_cleanup_steps(
    steps: Iterable[CleanupStep],
    *,
    primary_exception: BaseException | None,
) -> None:
    """Run every cleanup step and preserve an already-active primary failure."""

    failures: list[tuple[str, BaseException]] = []
    for label, callback in steps:
        try:
            callback()
        except BaseException as exc:  # cleanup must survive a second Ctrl+C too
            failures.append((label, exc))
    if not failures:
        return
    summary = "; ".join(
        f"{label}: {type(exc).__name__}: {exc}" for label, exc in failures
    )
    if primary_exception is not None:
        # Python 3.10 has no BaseException.add_note(). Preserve the original
        # exception while making every cleanup failure visible and inspectable.
        setattr(primary_exception, "cleanup_failures", summary)
        print(f"[cleanup] additional failures: {summary}", file=sys.stderr)
        return
    interrupted = next(
        (
            exc
            for _, exc in failures
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
        ),
        None,
    )
    if interrupted is not None:
        setattr(interrupted, "cleanup_failures", summary)
        raise interrupted
    raise RuntimeError(f"physical policy cleanup failed: {summary}") from failures[0][1]
