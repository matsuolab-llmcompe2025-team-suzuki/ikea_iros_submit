"""decoupled-lane policy: obs → GR00T action → (T, 25) task-space。

段階1 PoC (案C = pick 単体固定)。model 推論は `InferenceBackend` 抽象の背後に
置き、obs → 絶対 body29 → FK → (T, 25) の plumbing を **GPU/Thor 無し**で検証できる。

- `HoldPoseBackend`: obs の現在姿勢をそのまま action として返す stub。実 model が
  載る前でも「現在 EE pose を task-space で保持する」有効な fallback にもなる。
- `GrootWorkerBackend`(将来): iros_2026_ramen の `Gr00tPolicyPickLegs` worker を
  呼ぶ本番 backend。ここでは interface のみ定義 (GPU + desktop repo 依存のため PoC 外)。

boundary Policy 契約 (`components/server.py`):
    metadata / act(obs)->{"actions": (T,25)} / reset()
"""

from __future__ import annotations

import abc

import numpy as np

from .g1_urdf_fk import G1WristFK
from .taskspace_adapter import (
    DEX1_OPEN_VALUE,
    GROOT_ACTION_DIM,
    groot_chunk_to_taskspace,
)

# groot_pick_leg_contract.REAL_ROOT_PROXY_XYZ_WXYZ (静止 root proxy、xyz+wxyz)。
# RealDdsBackend は global translation / base height を観測できないため、訓練時の
# 立位高さの session-local 静止 root を使う。root 予測は robot に送られない。
_ROOT_PROXY_XYZ_WXYZ = np.array(
    [0.0, 0.0, 0.70, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
)
_BODY_DIM = 29
_HAND_DIM = 2


class InferenceBackend(abc.ABC):
    """obs → GR00T raw action chunk (T, 38) = robot_q(36) + hand(2)。"""

    @abc.abstractmethod
    def infer(self, obs: dict, horizon: int) -> np.ndarray:
        """(horizon, 38) float の action chunk を返す。"""

    def reset(self) -> None:
        """attempt 毎の内部状態リセット (default no-op)。"""


class HoldPoseBackend(InferenceBackend):
    """現在姿勢保持 backend (PoC + 安全 fallback)。

    obs.body_q(29) をそのまま robot_q_desired の body に据え、root は静止 proxy、
    hand は既定開度で全 horizon 埋める。→ adapter を通すと「現在 EE pose を保持する」
    task-space chunk になる。GPU/model 不要。
    """

    def __init__(self, hand_open_fraction: float = 1.0):
        # 1.0 = 全開 (boundary -1 = open)。DEX1 model 空間の絶対値に変換して保持。
        self._hand_value = float(np.clip(hand_open_fraction, 0.0, 1.0)) * DEX1_OPEN_VALUE

    def infer(self, obs: dict, horizon: int) -> np.ndarray:
        body_q = np.asarray(obs["body_q"], dtype=np.float64)
        if body_q.shape != (_BODY_DIM,):
            raise ValueError(f"obs.body_q must be (29,), got {body_q.shape}")
        row = np.concatenate(
            [
                _ROOT_PROXY_XYZ_WXYZ,                       # root(7)
                body_q,                                     # body(29)
                np.full(_HAND_DIM, self._hand_value, np.float64),  # hand(2)
            ]
        )
        assert row.shape == (GROOT_ACTION_DIM,)
        return np.tile(row, (horizon, 1))


class GrootWorkerBackend(InferenceBackend):
    """実 GR00T pick 推論 backend (self-contained、直接 worker protocol)。

    vendored worker (`components/ramen/vendor` + `GrootPickWorker`) を使い、boundary
    obs → 38D state + 4 cam → **raw 38D** action chunk を得て返す (adapter が直接 (T,25) 化)。
    desktop policy stack (base / config_loader / Observation) には依存しない。

    camera: boundary は RGB (ego_view + left/right_wrist)。worker/model は BGR 期待なので
    RGB→BGR に戻し、head 単一 ego_view を cam_0/cam_1 両方へ、wrist 欠落は zero-image。
    worker venv は env `RAMEN_WORKER_PYTHON` (実 Thor container では image 内 venv)。
    """

    def __init__(
        self,
        dex1_open_fraction: tuple[float, float] = (1.0, 1.0),
        task: str = "pick table leg",
        worker_python: str | None = None,
    ):
        from .groot_worker import GrootPickWorker

        self._dex1_open = np.clip(np.asarray(dex1_open_fraction, np.float32), 0.0, 1.0)
        self._worker = GrootPickWorker(worker_python=worker_python, task=task)

    @staticmethod
    def _bgr(images: dict, key: str) -> np.ndarray:
        img = images.get(key)
        if img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[:, :, ::-1])

    def infer(self, obs: dict, horizon: int) -> np.ndarray:
        body_q = np.asarray(obs["body_q"], dtype=np.float32)
        if body_q.shape != (_BODY_DIM,):
            raise ValueError(f"obs.body_q must be (29,), got {body_q.shape}")
        state38 = self._worker.build_state(body_q, self._dex1_open)

        images = obs.get("images", {})
        ego = self._bgr(images, "ego_view")   # head 単一 ego_view を cam_0/cam_1 両方へ
        frames_bgr = {
            "head_left": ego,
            "head_right": ego,
            "left_wrist": self._bgr(images, "left_wrist"),
            "right_wrist": self._bgr(images, "right_wrist"),
        }
        chunk38 = self._worker.predict(state38, frames_bgr)   # (T_model, 38) raw

        if chunk38.shape[0] >= horizon:
            return chunk38[:horizon]
        pad = np.tile(chunk38[-1:], (horizon - chunk38.shape[0], 1))
        return np.concatenate([chunk38, pad])

    def close(self) -> None:
        self._worker.close()


class Groot53Backend(InferenceBackend):
    """53D LeRobot GR00T backend (rotate_table_base / insert / rotate_leg / flip)。

    pick(38D Isaac native)と異なり、これらは REAL_G1_RELATIVE_EEF 53D (arm relative、
    post_processor が current arm state 加算で absolute 化)。desktop `Gr00tPolicy` を
    再利用: obs → 49D state (build_state_from_raw + G1WristFK ee_state) → predict →
    19D executable (waist3 + arm14 + hand2、絶対) → body29 → (T,38)。camera 3個。

    ⚠️ 現状 **desktop 依存**(RAMEN_DESKTOP_REPO + inference/desktop pixi env=lerobot0.6.1)。
    self-contained 化 (components/ramen/vendor/groot53 + groot53_worker) は 2 blocker で
    保留: (1) 53D は lerobot 0.6.1、pick は 0.6.0 = container で env 分裂、(2) takada 53D
    checkpoint は embed_tokens tied-weight で原さん vanilla server の from_pretrained が
    strict load 失敗 = desktop の custom load (raw_config+streaming) が必須。→ VENDOR_NOTES。
    """

    _WAIST = slice(0, 3)
    _ARMS = slice(3, 17)
    _HANDS = slice(17, 19)
    _DEX1_OPEN_VALUE = 4.5

    def __init__(
        self,
        variant: str,
        desktop_repo: str | None = None,
        dex1_open_fraction: tuple[float, float] = (1.0, 1.0),
        task: str | None = None,
    ):
        import os
        import sys
        from pathlib import Path

        # 既定は vendored desktop subtree (self-contained)。RAMEN_DESKTOP_REPO で override 可。
        vendored = str(Path(__file__).resolve().parent / "vendor" / "desktop")
        repo = desktop_repo or os.environ.get("RAMEN_DESKTOP_REPO") or vendored
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from inference.desktop.lower_policy.policies.base import (  # noqa: E402
            CameraKey,
            Observation,
            RawRobotState,
        )
        from inference.desktop.lower_policy.policies.config_loader import (  # noqa: E402
            load_policy_variant,
            resolve_policy_class,
        )
        from inference.desktop.lower_policy.policies.groot import (  # noqa: E402
            build_state_from_raw,
        )
        from inference.desktop.perception.g1_urdf_fk import G1WristFK  # noqa: E402

        self._CameraKey = CameraKey
        self._Observation = Observation
        self._RawRobotState = RawRobotState
        self._build_state = build_state_from_raw
        self._fk = G1WristFK.from_urdf()
        self._dex1_open = np.clip(np.asarray(dex1_open_fraction, np.float32), 0.0, 1.0)
        self._task = task

        cfg_path = os.path.join(
            repo, "inference/desktop/lower_policy/configs/policy_config.yaml"
        )
        entry = load_policy_variant(cfg_path, variant)
        cls = resolve_policy_class(entry.policy_type)
        self._policy = cls.from_ckpt(entry.policy_config)

    @staticmethod
    def _bgr(images: dict, key: str) -> np.ndarray:
        img = images.get(key)
        if img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[:, :, ::-1])

    def infer(self, obs: dict, horizon: int) -> np.ndarray:
        body_q = np.asarray(obs["body_q"], dtype=np.float32)
        if body_q.shape != (_BODY_DIM,):
            raise ValueError(f"obs.body_q must be (29,), got {body_q.shape}")
        ee_state = self._fk.compute_ee_state(body_q)
        hand_state = (self._dex1_open * self._DEX1_OPEN_VALUE).astype(np.float32)
        state49 = self._build_state(self._RawRobotState(
            joint_positions=body_q, hand_state=hand_state, ee_state=ee_state,
        ))
        images = obs.get("images", {})
        frames = {
            self._CameraKey.HEAD_LEFT: self._bgr(images, "ego_view"),
            self._CameraKey.WRIST_LEFT: self._bgr(images, "left_wrist"),
            self._CameraKey.WRIST_RIGHT: self._bgr(images, "right_wrist"),
        }
        observation = self._Observation(
            frames_bgr=frames, frames_bgr_prev=None, state=state49,
            skill_id=None, language=self._task or obs.get("prompt"),
            obb_detections=None, timestamp_ns=0,
        )
        action19 = np.asarray(
            self._policy.predict(observation).action_chunk, dtype=np.float64
        )
        if action19.ndim != 2 or action19.shape[1] != 19:
            raise ValueError(f"expected 53D-policy 19D chunk, got {action19.shape}")

        legs12 = body_q[:12].astype(np.float64)
        rows = []
        for row in action19:
            body29 = np.concatenate([legs12, row[self._WAIST], row[self._ARMS]])
            rows.append(
                np.concatenate([_ROOT_PROXY_XYZ_WXYZ, body29, row[self._HANDS]])
            )
        chunk38 = np.stack(rows)
        if chunk38.shape[0] >= horizon:
            return chunk38[:horizon]
        pad = np.tile(chunk38[-1:], (horizon - chunk38.shape[0], 1))
        return np.concatenate([chunk38, pad])

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def close(self) -> None:
        close = getattr(self._policy, "close", None)
        if callable(close):
            close()


class GrootPickTaskspacePolicy:
    """boundary decoupled Policy: obs → backend → adapter → (T, 25)。

    段階1 では pick 単体 (案C) を想定。model 選択・skill 遷移は将来 (案A/B)。
    """

    ACTION_CHUNK = 16     # metadata.action_chunk_size (= 返す T)
    OBS_CHUNK = 1

    def __init__(
        self,
        lane: str = "decoupled",
        backend: InferenceBackend | None = None,
        ee_frame_transform: np.ndarray | None = None,
        urdf_path: str | None = None,
    ):
        if lane != "decoupled":
            raise ValueError(f"GrootPickTaskspacePolicy is decoupled-only, got {lane!r}")
        self.lane = lane
        self._backend = backend if backend is not None else HoldPoseBackend()
        self._ee_frame_transform = ee_frame_transform
        self._fk = (
            G1WristFK.from_urdf(urdf_path) if urdf_path else G1WristFK.from_urdf()
        )
        self._steps = 0

    @property
    def metadata(self) -> dict:
        return {
            "lane": self.lane,
            "action_chunk_size": self.ACTION_CHUNK,
            "obs_chunk_size": self.OBS_CHUNK,
            # organizer が publish する camera key で宣言する (model cam への写像は
            # backend 側で行う)。ego_view のみ保証、wrist は欠落あり。
            "camera_keys": ["ego_view", "left_wrist", "right_wrist"],
            "wants_state": True,
            "wants_prompt": True,
        }

    def act(self, obs: dict) -> dict:
        self._steps += 1
        chunk_38 = self._backend.infer(obs, self.ACTION_CHUNK)
        actions = groot_chunk_to_taskspace(
            chunk_38, self._fk, ee_frame_transform=self._ee_frame_transform
        )
        return {"actions": actions}

    def reset(self) -> dict:
        self._steps = 0
        self._backend.reset()
        return {"ok": True}


# prompt keyword → skill_key (先勝ち)。organizer が skill 毎に prompt を変える前提 (Q5)。
_PROMPT_SKILL_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("flip",), "flip"),
    (("insert",), "insert"),
    (("tighten",), "rotate_leg"),
    (("pick",), "pick"),
    (("rotate", "move", "base"), "rotate_table_base"),
)

# skill_key → backend 生成 (lazy)。pick=38D self-contained worker、他=53D。
_SKILL_BACKENDS: dict[str, "callable"] = {
    "pick": lambda: GrootWorkerBackend(),
    "rotate_table_base": lambda: Groot53Backend("groot_overlay"),
    "insert": lambda: Groot53Backend("groot_insert_leg_200k"),
    "rotate_leg": lambda: Groot53Backend("groot_rotate_leg_200k"),
    "flip": lambda: Groot53Backend("groot_flip_table_n17_2_baseline"),
}


def resolve_skill_from_prompt(prompt: str | None, default_skill: str) -> str:
    """prompt 文字列 → skill_key。keyword 一致 (先勝ち)、無ければ default。"""
    text = (prompt or "").lower()
    for keywords, skill in _PROMPT_SKILL_RULES:
        if any(k in text for k in keywords):
            return skill
    return default_skill


class MultiSkillTaskspacePolicy:
    """prompt 駆動で 5 skill の backend を切替える decoupled Policy (案A)。

    act(obs) 毎に obs["prompt"] から skill を判定し、対応 backend で (T,25) を返す。
    backend は skill 初回使用時に lazy load しキャッシュ (pick=lerobot0.6.0 worker /
    53D=lerobot0.6.1 worker、各自別 subprocess)。organizer が skill 毎に prompt を
    変えれば「一通り全部」も追従する。prompt が skill を示さない時は default_skill。

    ⚠️ organizer が run 内で prompt をどう与えるか (Q5) 未確認。prompt が固定なら
    実質 1 skill (=default) になる。GPU memory: skill を跨ぐと複数 model が常駐しうる
    (Thor 128GB 想定)。default_skill は env RAMEN_DEFAULT_SKILL で指定。
    """

    ACTION_CHUNK = 16
    OBS_CHUNK = 1

    def __init__(
        self,
        lane: str = "decoupled",
        default_skill: str = "pick",
        ee_frame_transform: np.ndarray | None = None,
    ):
        if lane != "decoupled":
            raise ValueError(f"MultiSkill is decoupled-only, got {lane!r}")
        if default_skill not in _SKILL_BACKENDS:
            raise ValueError(
                f"unknown default_skill {default_skill!r}; "
                f"known: {sorted(_SKILL_BACKENDS)}"
            )
        self.lane = lane
        self._default_skill = default_skill
        self._ee_frame_transform = ee_frame_transform
        self._fk = G1WristFK.from_urdf()
        self._backends: dict[str, InferenceBackend] = {}
        self._steps = 0

    @property
    def metadata(self) -> dict:
        return {
            "lane": self.lane,
            "action_chunk_size": self.ACTION_CHUNK,
            "obs_chunk_size": self.OBS_CHUNK,
            "camera_keys": ["ego_view", "left_wrist", "right_wrist"],
            "wants_state": True,
            "wants_prompt": True,
        }

    def _backend(self, skill: str) -> InferenceBackend:
        backend = self._backends.get(skill)
        if backend is None:
            print(f"[server] MultiSkill: loading backend for skill={skill}")
            backend = _SKILL_BACKENDS[skill]()
            self._backends[skill] = backend
        return backend

    def act(self, obs: dict) -> dict:
        self._steps += 1
        skill = resolve_skill_from_prompt(obs.get("prompt"), self._default_skill)
        chunk_38 = self._backend(skill).infer(obs, self.ACTION_CHUNK)
        actions = groot_chunk_to_taskspace(
            chunk_38, self._fk, ee_frame_transform=self._ee_frame_transform
        )
        return {"actions": actions}

    def reset(self) -> dict:
        self._steps = 0
        for backend in self._backends.values():
            backend.reset()
        return {"ok": True}

    def close(self) -> None:
        for backend in self._backends.values():
            close = getattr(backend, "close", None)
            if callable(close):
                close()


class OrchestratorTaskspacePolicy:
    """full orchestrator を boundary Policy 契約に包む (RAMEN_POLICY=groot_orchestrator)。

    原さんの orchestrator を OrchestratorDriver 経由で act(obs) 毎に 1 tick 駆動し、
    perception(YOLO)+dwell+is_complete の skill 遷移をそのまま使って (T,25) を返す。
    orchestrator は per-tick 1 step なので action_chunk_size=1 (organizer が補間)。
    """

    ACTION_CHUNK = 1
    OBS_CHUNK = 1

    def __init__(self, lane: str = "decoupled", **kwargs):
        if lane != "decoupled":
            raise ValueError(f"orchestrator policy is decoupled-only, got {lane!r}")
        from .orchestrator_driver import OrchestratorDriver

        self.lane = lane
        self._driver = OrchestratorDriver(**kwargs)

    @property
    def metadata(self) -> dict:
        return {
            "lane": self.lane,
            "action_chunk_size": self.ACTION_CHUNK,
            "obs_chunk_size": self.OBS_CHUNK,
            "camera_keys": ["ego_view", "left_wrist", "right_wrist"],
            "wants_state": True,
            "wants_prompt": True,
        }

    def act(self, obs: dict) -> dict:
        return {"actions": self._driver.act(obs)["actions"]}

    def reset(self) -> dict:
        self._driver.reset()
        return {"ok": True}

    def close(self) -> None:
        self._driver.close()
