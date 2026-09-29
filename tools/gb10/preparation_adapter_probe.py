#!/usr/bin/env python3
"""Exercise the production preparation against the pinned adapter, without I/O."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml


class Console:
    stage = 1

    def __init__(self):
        self.view = None
        self.key = None

    def show(self, stage, phase, allowed):
        view = (phase, allowed)
        if view != self.view:
            self.key = None
            self.view = view

    def poll(self):
        key, self.key = self.key, None
        return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from official_follower import REVISION
    assert subprocess.check_output(
        ["git", "-C", str(args.organizer), "rev-parse", "HEAD"], text=True).strip() == REVISION
    sys.path.insert(0, str(args.organizer / "reference/wbc_adapter/tests"))
    sys.path.insert(0, "/app/ramen")
    import test_joint_lane as official
    import inference.desktop.boundary as boundary
    from inference.desktop.lower_policy.actuators.boundary_sink import BoundaryActionSink
    from inference.desktop.lower_policy.skills.boundary_preparation import BoundaryPreparation, PreparationSettings
    from inference.desktop.lower_policy.skills.collision_aware_pre_motion import ArmWaypoint

    config = yaml.safe_load(Path("/app/ramen/inference/desktop/lower_policy/configs/skill_config.yaml").read_text())
    results = []

    def run(case):
        ctx = official.make_ctx(dex1=official.FakeDex1())
        clock = SimpleNamespace(now=10.0)
        q = np.zeros(14)
        target = np.full(14, 0.5)
        if case == "split":
            q[0], target[0] = -3.0, 1.5
        if case == "clamp":
            ctx.arm_limits.upper[:] = 0.3
        console = Console()
        events, holds, sent = [], [], []
        snapshot = None
        segment_start = q.copy()
        segment_time = clock.now

        class Transport:
            def send_goto(self, left, right, **kw):
                nonlocal segment_start, segment_time
                sent.append(np.r_[left, right])
                if case == "drop_retry" and len(sent) == 1:
                    return
                segment_start, segment_time = q.copy(), clock.now
                official.wbc_driver._handle_goto(ctx, official.goto_msg(left, right, **kw))

            def send_chunk(self, chunk, **kw):
                raise AssertionError("Ordinary chunk overwrote an owned preparation")

        with patch.object(boundary, "JointSink", return_value=Transport()):
            sink = BoundaryActionSink(lane="joint", log_fn=events.append, clock=lambda: clock.now)
        engine = BoundaryPreparation(
            [ArmWaypoint("target", tuple(target))], sink=sink,
            settings=PreparationSettings.from_config(config), record_hold=holds.append,
            state_getter=lambda: snapshot, console=console, clock=lambda: clock.now)

        def tick(bias=0.0, duplicate=False):
            nonlocal snapshot, q
            clock.now += 0.05
            old = q.copy()
            if ctx.in_flight:
                g = ctx.in_flight
                upper = np.asarray(g["upper_body"])
                points = np.vstack([segment_start, np.c_[upper[:, :7], upper[:, 14:21]]])
                times = np.r_[segment_time, g["times"]]
                desired = np.array([np.interp(clock.now, times, points[:, j]) for j in range(14)])
                q += np.clip((desired - bias - q) * 0.3, -0.015, 0.015)
            ctx.backend.q[15:22], ctx.backend.q[29:36] = q[:7], q[7:]
            if not duplicate:
                snapshot = SimpleNamespace(position=np.r_[np.zeros(15), q],
                    velocity=np.r_[np.zeros(15), (q-old)/0.05], received_monotonic_ns=int(clock.now*1e9))
            engine.step({"joint_state": snapshot})
            if engine.owned:
                assert not sink.send_action(np.zeros(19), np.zeros(29), force=True)

        with patch.object(time, "monotonic", lambda: clock.now), patch.object(time, "time", lambda: 1700000000+clock.now):
            ctx.backend.q[15:22], ctx.backend.q[29:36] = q[:7], q[7:]
            tick()
            if case == "interrupt":
                for _ in range(10):
                    tick()
                measured = q.copy()
                engine.cancel()
                np.testing.assert_allclose(sent[-1], measured)
                np.testing.assert_allclose(ctx.in_flight["upper_body"][-1][:7], measured[:7])
                assert engine.phase == "interrupted" and not engine.complete
            elif case == "stale":
                for _ in range(9):
                    tick(duplicate=True)
                try:
                    for _ in range(3):
                        tick(duplicate=True)
                except RuntimeError as exc:
                    assert "fresh joint state" in str(exc)
                else:
                    raise AssertionError("Stale observation was accepted")
                assert not engine.complete
                engine.cancel()
            else:
                bias = 0.1162 if case == "late" else 0
                for _ in range(350 if case == "split" else 170):
                    tick(bias)
                    if engine.complete:
                        break
                if case in ("late", "drop_retry", "clamp"):
                    assert engine.phase == "holding" and not engine.complete
                    console.key = "enter"
                    tick(bias)
                    assert not engine.complete
                if case == "clamp":
                    engine.cancel()
                else:
                    if case == "drop_retry":
                        console.key = "r"
                        tick()
                    for _ in range(350):
                        tick(0.08445 if case == "late" else 0)
                        if engine.phase == "ready":
                            assert not engine.complete
                            console.key = "enter"
                            tick(0.08445 if case == "late" else 0)
                        if engine.complete:
                            break
                    assert engine.complete, (case, engine.phase, q.tolist())
                    np.testing.assert_allclose(q, target, atol=0.1)
            assert not ctx.dex1_pub.sent, "goto must not change Dex1 commands"
        return {"case": case, "passed": True, "final_phase": engine.phase,
                "goto_sent": len(sent), "goto_accepted": ctx.stats.goto_accepted,
                "stats": ctx.stats.report(), "events": events}

    for case in ("nominal", "split", "late", "drop_retry", "clamp", "interrupt", "stale"):
        results.append(run(case))
        print(json.dumps({"case": case, "passed": True}), flush=True)
    args.output.write_text(json.dumps({"passed": True, "organizer_revision": REVISION,
        "physical_commands_sent": False, "physics_validated": False, "cases": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()
