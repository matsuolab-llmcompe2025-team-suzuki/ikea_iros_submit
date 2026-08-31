"""boundary↔orchestrator I/O アダプタ層の unit test (GPU 不要)。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from components.ramen.orchestrator_io import (
    DEX1_OPEN_VALUE,
    G1_JOINT_NAMES,
    BoundaryDex1StateSource,
    BoundaryJointStateSource,
    InterceptorActuator,
    assemble_19d,
    build_frame_data,
)


class TestJointStateSource:
    def test_update_get(self) -> None:
        src = BoundaryJointStateSource()
        assert src.get() is None
        q = np.arange(29, dtype=np.float64)
        src.update(q, t=5)
        js = src.get()
        assert js is not None
        assert js.name == G1_JOINT_NAMES and len(js.name) == 29
        assert np.allclose(js.position, q) and js.t == 5
        assert js.velocity.shape == (29,) and js.effort.shape == (29,)

    def test_bad_shape(self) -> None:
        with pytest.raises(ValueError):
            BoundaryJointStateSource().update(np.zeros(19), t=0)


class TestDex1StateSource:
    def test_open_fraction_to_rad(self) -> None:
        src = BoundaryDex1StateSource(open_fraction=(1.0, 0.0))
        d = src.get()
        assert np.allclose(d.position_rad, [DEX1_OPEN_VALUE, 0.0])

    def test_update_clips(self) -> None:
        src = BoundaryDex1StateSource()
        src.update((2.0, -1.0), t=3)
        assert np.allclose(src.get().position_rad, [DEX1_OPEN_VALUE, 0.0])
        assert src.get().t == 3


class TestInterceptorActuator:
    def test_capture(self) -> None:
        a = InterceptorActuator("waist")
        assert a.last is None
        a.send_action(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(a.last, [1.0, 2.0, 3.0]) and a.count == 1
        a.start(); a.stop(); a.reset()          # lifecycle no-op
        assert a.last is None


class TestBuildFrameData:
    def test_packed_stereo_duplicates_width(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        fd = build_frame_data(img, t=7, packed_stereo=True)
        assert fd.rgb.shape == (480, 1280, 3) and fd.t == 7

    def test_mono(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        fd = build_frame_data(img, t=1, packed_stereo=False)
        assert fd.rgb.shape == (480, 640, 3)


class TestAssemble19d:
    def test_full(self) -> None:
        out = assemble_19d(
            np.array([1, 2, 3]), np.arange(14) + 10, np.array([0.1, 0.2])
        )
        assert out.shape == (19,)
        assert np.allclose(out[0:3], [1, 2, 3])
        assert np.allclose(out[3:17], np.arange(14) + 10)
        assert np.allclose(out[17:19], [0.1, 0.2])

    def test_missing_waist_hand_zero(self) -> None:
        out = assemble_19d(None, np.ones(14), None)
        assert np.allclose(out[0:3], 0.0) and np.allclose(out[17:19], 0.0)
        assert np.allclose(out[3:17], 1.0)
