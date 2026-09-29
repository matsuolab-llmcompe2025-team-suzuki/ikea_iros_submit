"""Real guard protocol fed by synthetic lowstate, bound only to loopback.

No SDK or DDS is loaded. The production attestor/relay are used unchanged;
only the lowstate source is synthetic. Bridge publication stays independent.
"""

import threading
from types import SimpleNamespace

from inference.desktop.perception.venue_state_guard import StateAttestor, serve


class LoopbackStateGuard:
    def __init__(self, *, upstream_port=5557, guard_port=5558):
        self.profile = {"guard_port": guard_port, "source_max_age_s": .25,
                        "upstream_endpoint": f"tcp://127.0.0.1:{upstream_port}"}
        self.attestor = StateAttestor({"left": [-.72, 4.64], "right": [0., 5.38]})
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.tick = 0
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        if not self.ready.wait(3):
            self.close()
            raise RuntimeError(f"Loopback guard did not start: {self.error}")

    def _serve(self):
        try:
            serve(self.attestor, self.profile, "127.0.0.1",
                  stop_event=self.stop, ready=self.ready)
        except Exception as exc:
            self.error = exc

    def observe(self, payload):
        self.check()
        self.tick = (self.tick + 1) % 2**32
        motors = [SimpleNamespace(q=0.) for _ in range(35)]
        for motor, q in zip(motors, payload["body_q"]):
            motor.q = q
        for side, index in (("left", 31), ("right", 33)):
            closed, opened = self.attestor.calibration[side]
            motors[index].q = closed + payload["gripper_q"][side]["q"] * (opened - closed) / -5.3
        self.attestor.observe(SimpleNamespace(
            tick=self.tick, motor_state=motors,
            imu_state=SimpleNamespace(quaternion=payload["base_quat"])))

    def regress_tick(self):
        self.tick = 0

    def check(self):
        if not self.thread.is_alive() or self.error is not None:
            raise RuntimeError(f"Loopback guard failed: {self.error}")

    def close(self):
        self.stop.set()
        self.thread.join(3)
        if self.thread.is_alive():
            raise RuntimeError("Loopback guard did not stop")
