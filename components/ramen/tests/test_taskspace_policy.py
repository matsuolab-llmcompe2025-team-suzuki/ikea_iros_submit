"""段階1 PoC 検証: GrootPickTaskspacePolicy が boundary decoupled 契約を満たす。

GPU/Thor 不要。mock obs → act → boundary/actions.py DecoupledSink.validate_chunk
(実 boundary の検証器) が通ることを確認する。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# repo root を import path に (components.* / boundary.* 解決用)。
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from boundary.actions import DecoupledSink, TASKSPACE_DIM, TASKSPACE_SLICES
from components.ramen.policy import GrootPickTaskspacePolicy, HoldPoseBackend
from components.ramen.taskspace_adapter import DEX1_OPEN_VALUE


def _obs(body_q: np.ndarray, prompt: str = "pick table leg") -> dict:
    return {
        "images": {},
        "body_q": body_q.astype(np.float32),
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "prompt": prompt,
        "t": 0.0,
    }


@pytest.fixture(scope="module")
def policy() -> GrootPickTaskspacePolicy:
    return GrootPickTaskspacePolicy(lane="decoupled")


class TestMetadata:
    def test_decoupled_and_keys(self, policy) -> None:
        md = policy.metadata
        assert md["lane"] == "decoupled"
        assert md["wants_state"] and md["wants_prompt"]
        assert md["action_chunk_size"] == GrootPickTaskspacePolicy.ACTION_CHUNK
        assert "ego_view" in md["camera_keys"]

    def test_rejects_non_decoupled(self) -> None:
        with pytest.raises(ValueError):
            GrootPickTaskspacePolicy(lane="sonic")


class TestActBoundaryContract:
    def _body(self, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).uniform(-0.5, 0.5, 29)

    def test_shape_and_validate_chunk(self, policy) -> None:
        out = policy.act(_obs(self._body(1)))
        actions = out["actions"]
        assert actions.shape == (policy.ACTION_CHUNK, TASKSPACE_DIM)
        assert actions.dtype == np.float32
        # 実 boundary の検証器 (socket 不要な staticmethod) を直接通す
        validated = DecoupledSink.validate_chunk(actions)
        assert validated.shape == actions.shape

    def test_multiple_seeds_all_valid(self, policy) -> None:
        for seed in range(6):
            actions = policy.act(_obs(self._body(seed)))["actions"]
            DecoupledSink.validate_chunk(actions)   # raises on violation

    def test_hold_pose_hands_open_by_default(self, policy) -> None:
        actions = policy.act(_obs(self._body(2)))["actions"]
        hands = np.concatenate(
            [actions[:, TASKSPACE_SLICES["left_hand"]],
             actions[:, TASKSPACE_SLICES["right_hand"]]],
            axis=1,
        )
        assert np.allclose(hands, -1.0)   # 既定 = 全開 → boundary -1

    def test_chunk_rows_identical_for_hold(self, policy) -> None:
        actions = policy.act(_obs(self._body(3)))["actions"]
        assert np.allclose(actions, actions[0:1])   # HoldPose = 全 row 同一

    def test_bad_body_q_shape(self, policy) -> None:
        with pytest.raises(ValueError):
            policy.act(_obs(np.zeros(19)))


class TestReset:
    def test_reset_ok(self, policy) -> None:
        policy.act(_obs(np.zeros(29)))
        assert policy.reset() == {"ok": True}
        assert policy._steps == 0


class TestHoldPoseBackend:
    def test_infer_shape_and_hand_value(self) -> None:
        b = HoldPoseBackend(hand_open_fraction=0.0)   # 全閉
        chunk = b.infer({"body_q": np.zeros(29)}, horizon=8)
        assert chunk.shape == (8, 38)
        assert np.allclose(chunk[:, 36:38], DEX1_OPEN_VALUE * 0.0)   # closed→model 0
