"""Recoverable, one-shot preparation for the organizer joint lane only.

The wire protocol has no acceptance acknowledgement. Arrival is measured, not
inferred from a successful publish. The adapter holds the endpoint after the
planned trajectory; a timeout here never advances to another waypoint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import sys
import time

import numpy as np


@dataclass(frozen=True)
class PreparationSettings:
    max_speed_rad_s: float
    max_goto_duration_s: float
    adapter_hz: float
    arrival_error_rad: float
    arrival_speed_rad_s: float
    arrival_dwell_s: float
    settle_grace_s: float
    state_max_age_s: float
    diagnostic_period_s: float

    @classmethod
    def from_config(cls, config: dict) -> "PreparationSettings":
        result = cls(**config["boundary_preparation"])
        if not all(np.isfinite(v) and v > 0 for v in vars(result).values()):
            raise ValueError("boundary_preparation values must be positive and finite")
        if not (0.01 <= result.max_speed_rad_s <= 0.3):
            raise ValueError("preparation goto speed must be in [0.01, 0.3] rad/s")
        if not (1 / result.adapter_hz < result.max_goto_duration_s <= 15):
            raise ValueError("preparation goto duration exceeds the organizer contract")
        return result


class BoundaryPreparation:
    def __init__(
        self, waypoints, *, sink, settings: PreparationSettings,
        record_hold, state_getter, console=None, clock=time.monotonic,
        stop_navigation=lambda: None,
    ):
        # Preserve the startup yaw/wrists at the forward-outward waypoint too.
        self.waypoints = tuple(
            replace(w, preserve_initial=(2, 4, 5, 6, 9, 11, 12, 13))
            if w.name == "forward_outward_clearance" else w for w in waypoints
        )
        self.sink, self.cfg = sink, settings
        self.record_hold, self.state_getter = record_hold, state_getter
        self.console, self.clock = console, clock
        self.stop_navigation = stop_navigation
        self.reset()

    def reset(self):
        self.phase = "idle"
        self.complete = False
        self.initial = None
        self.targets = ()
        self.index = 0
        self.segment = None
        self.started_at = None
        self.stable_since = None
        self.last_stamp = None
        self.last_measurement = None
        self.last_diagnostic = -float("inf")
        self.recovered = False
        self.owned = False

    @property
    def retreat_waypoints(self):
        if self.initial is None:
            return ()
        return tuple(q.copy() for q in reversed(self.targets[:self.index])) + (self.initial.copy(),)

    def _phase(self, phase, **details):
        if self.phase != phase:
            self.phase = phase
            self.sink.preparation_event(
                "preparation_state", phase=phase, waypoint=self.waypoints[self.index].name,
                **details,
            )
            print(f"[preparation] {self.waypoints[self.index].name}: {phase}", file=sys.stderr)
        if self.console is not None:
            allowed = ("enter", "r") if phase == "ready" else ("r",) if phase == "holding" else ()
            self.console.show(
                self.console.stage,
                f"準備／{self.waypoints[self.index].name}／{phase}", allowed,
            )

    def _publish(self, measured, now):
        self._phase("sending")
        self.segment = self.sink.send_preparation_goto(
            self, self.targets[self.index], measured,
            max_speed=self.cfg.max_speed_rad_s,
            max_duration=self.cfg.max_goto_duration_s, adapter_hz=self.cfg.adapter_hz,
        )
        self.started_at = now
        self.stable_since = None
        self._phase("moving")

    def step(self, obs):
        if self.complete:
            return
        now = self.clock()
        state = obs.get("joint_state")
        stamp = getattr(state, "received_monotonic_ns", None)
        if stamp is None or not (0 <= now - stamp / 1e9 <= self.cfg.state_max_age_s):
            self.stable_since = None
            raise RuntimeError("preparation requires fresh joint state; no goto was issued")
        body = np.asarray(state.position, dtype=np.float64)
        velocity = np.asarray(state.velocity, dtype=np.float64)
        if body.shape != (29,) or velocity.shape != (29,) or not (
            np.isfinite(body).all() and np.isfinite(velocity).all()
        ):
            raise RuntimeError("preparation requires finite 29-D position and velocity")
        measured, speed = body[15:29], float(np.max(np.abs(velocity[15:29])))
        # Freshness uses ``stamp``; sample spacing (distinct samples, dwell) uses the
        # measurement time, which a delayed PC2-guard reply does not shift.
        measured_at = getattr(state, "measured_monotonic_ns", None)
        if measured_at is None:
            measured_at = stamp
        self.last_measurement = body.copy()
        if self.initial is None:
            self.initial = measured.copy()
            self.targets = tuple(w.resolve(self.initial, venue=True) for w in self.waypoints)
            self.sink.acquire_preparation(self)
            self.owned = True
            self.stop_navigation()
            self._publish(measured, now)
        # Reusing a cached observation is not another convergence sample.
        distinct = measured_at != self.last_stamp
        if distinct:
            if self.last_stamp is not None and (
                measured_at < self.last_stamp
                or (measured_at - self.last_stamp) / 1e9 > self.cfg.state_max_age_s
            ):
                self.stable_since = None
            self.last_stamp = measured_at
        intent = np.asarray(self.segment["intent"])
        errors = np.abs(intent - measured)
        error = float(np.max(errors))
        elapsed = now - self.started_at
        duration = self.segment["planned_duration_s"]
        within = error <= self.cfg.arrival_error_rad and speed <= self.cfg.arrival_speed_rad_s
        if not within:
            self.stable_since = None
        elif distinct:
            if self.stable_since is None:
                self.stable_since = measured_at / 1e9
        reached = bool(
            distinct and within and elapsed >= duration
            and self.stable_since is not None
            and measured_at / 1e9 - self.stable_since >= self.cfg.arrival_dwell_s
        )
        if now - self.last_diagnostic >= self.cfg.diagnostic_period_s:
            self.last_diagnostic = now
            self.sink.preparation_event(
                "preparation_progress", phase=self.phase, waypoint=self.waypoints[self.index].name,
                error_rad=error, worst_arm_index=int(np.argmax(errors)), speed_rad_s=speed,
                elapsed_s=elapsed, planned_duration_s=duration,
                intended=intent.tolist(), measured=measured.tolist(), state_received_ns=stamp,
            )
            print(
                f"[preparation] {self.waypoints[self.index].name}: {elapsed:.1f}/{duration:.1f}s "
                f"error={error:.5f}rad joint={int(np.argmax(errors))} speed={speed:.3f}rad/s",
                file=sys.stderr,
            )
        if self.phase == "moving" and elapsed > duration + self.cfg.settle_grace_s and not reached:
            self.recovered = True
            self._phase("holding", reason="not_arrived; R retries this waypoint, Ctrl+C interrupts")
        if self.phase in ("holding", "ready") or (self.recovered and reached):
            ready = reached or (self.phase == "ready" and within and not distinct)
            self._phase("ready" if ready else "holding")
            key = self.console.poll() if self.console is not None else None
            if key == "r":
                self._publish(measured, now)
                return
            if key != "enter" or not reached:
                return
            # A new Enter acknowledges late arrival; it never starts a policy.
            self.recovered = False
        elif self.console is not None:
            self.console.poll()  # Detect EOF, discard all premature input.
        if not reached:
            return
        self.sink.preparation_event("preparation_arrived", waypoint=self.waypoints[self.index].name)
        if not self.segment["final_segment"]:
            self._publish(measured, now)
            return
        self.record_hold(self.targets[self.index])
        if self.index + 1 == len(self.targets):
            self.complete = True
            self._phase("complete")
            self.sink.release_preparation(self)
            self.owned = False
        else:
            self.index += 1
            self._publish(measured, now)

    def cancel(self):
        if not self.owned:
            return
        snapshot = self.state_getter()
        body = None if snapshot is None else np.asarray(snapshot.position, dtype=np.float64)
        if body is None or body.shape != (29,) or not np.isfinite(body).all():
            body = self.last_measurement
        if body is None:
            raise RuntimeError("cannot interrupt preparation without a measured pose")
        stamp = getattr(snapshot, "received_monotonic_ns", None)
        self.sink.preparation_event(
            "preparation_interrupt_requested", state_received_ns=stamp,
            state_stale=stamp is None or not (0 <= self.clock() - stamp / 1e9 <= self.cfg.state_max_age_s),
        )
        self.sink.interrupt_preparation(self, body)
        self.record_hold(body[15:29])
        self.sink.release_preparation(self)
        self.owned = False
        self._phase("interrupted")


def cancel_boundary_preparations(registry):
    """Cancel active preparation before main's ordinary shutdown commands."""
    for skill in registry.values():
        cancel = getattr(skill, "cancel_boundary_preparation", None)
        if callable(cancel):
            cancel()
