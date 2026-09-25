"""ACT / Diffusion Policy (LeRobot native、19D absolute action) の推論 adapter (Issue #139)。

# 学習時の契約 (HF `Team-RAMEN/IROS2026_RAMEN_hara_rotate_table_base_{act,diffusion}_v1`)

- 観測: `observation.state` 19D (waist 3 → 左腕 7 → 右腕 7 → gripper 左右、
  subtask_training.json:state.names 順) と 3 cam (head_left / left_wrist /
  right_wrist、640x480)。画像は jpg を BGR で読み → RGB → CHW float [0, 1]
  (`lerobot_frame_cache_patch._read_jpg_as_frame_tensor`)。
- 出力: `action` 19D の絶対値。並びは VlaSkill の 19D 契約と同じ。
- 正規化: ckpt の pre / post processor がそのまま持つ (ACT MEAN_STD、DP MIN_MAX)。
- overlay: head_left のみ (学習 frame cache に焼き込み済)。
- 観測履歴: ACT 1 frame / DP 2 frame (1 tick = 1/30s 前)。n_obs_steps / chunk 長 /
  カメラ名はコードに固定せず ckpt の config.json から読む (再学習 ckpt を
  `ckpt_ref` の差し替えだけで使うため)。

# 構成

- `ActDiffusionModel`: ckpt と processor を読み、観測列 → 19D chunk を返す。lerobot
  (Python 3.12) が要る。`act_diffusion_worker` と、lerobot がある env での同一
  process 実行の両方で使う。
- `ActDiffusionPolicy`: Policy protocol。観測履歴と overlay を持ち、実行層は
  `ChunkExecutor` (async 再計画 + temporal ensemble)。lerobot を import できない
  実機 runtime (Python 3.10) では worker を別 process で起動する (GR00T と同じ境界)。
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from inference.desktop.lower_policy.chunk_executor import ChunkExecutor
from inference.desktop.lower_policy.policies.base import (
    G1_UPPER_BODY_JOINT_INDICES,
    CameraKey,
    Observation,
    PolicyAction,
    PolicyConfig,
    RawRobotState,
)
from inference.desktop.lower_policy.policies.groot_worker_protocol import (
    receive_archive,
    scalar_text,
    send_archive,
)
from inference.desktop.lower_policy.policies.ramen_ori import (
    match_training_jpeg,
    overlay_obb_on_frame,
)


# 学習と同じ 3 cam (subtask_training.json:cameras = head_left / left_wrist / right_wrist)
CAMERAS: tuple[CameraKey, ...] = (
    CameraKey.HEAD_LEFT,
    CameraKey.WRIST_LEFT,
    CameraKey.WRIST_RIGHT,
)
STATE_DIM: int = 19
ACTION_DIM: int = 19
IMAGE_SHAPE_HWC: tuple[int, int, int] = (480, 640, 3)

# CameraKey → ckpt の image feature key (学習 view 名)
_CAM_TO_FEATURE_KEY: dict[CameraKey, str] = {
    CameraKey.HEAD_LEFT: "observation.images.head_left",
    CameraKey.WRIST_LEFT: "observation.images.left_wrist",
    CameraKey.WRIST_RIGHT: "observation.images.right_wrist",
}
_STATE_KEY = "observation.state"
_SUPPORTED_POLICY_KINDS = ("act", "diffusion")

# worker との往復の上限 [s]。起動は ckpt の load、初回の predict は CUDA 初期化と cuDNN の
# autotune が乗るので長め。以降の predict (DP で ~150 ms) が返らなければ worker の hang とみなす。
_WORKER_STARTUP_TIMEOUT_S = 300.0
_FIRST_PREDICT_TIMEOUT_S = 300.0
_PREDICT_TIMEOUT_S = 10.0
_CONTROL_TIMEOUT_S = 10.0


def build_state_from_raw(raw: RawRobotState) -> np.ndarray:
    """RawRobotState → 学習時の `observation.state` 19D。

    joint_positions は G1_JOINT_NAMES 順 (waist 12-14 / 左腕 15-21 / 右腕 22-28) で、
    学習の state.names (waist 3 → 左腕 7 → 右腕 7) と同じ並び。gripper は Dex1 の
    motor-output rad で、学習 source の hand_state と同じ単位。
    """
    joints = np.asarray(raw.joint_positions, dtype=np.float32)
    hand = np.asarray(raw.hand_state, dtype=np.float32)
    if joints.shape != (29,) or hand.shape != (2,):
        raise ValueError(
            f"expected joint_positions (29,) and hand_state (2,), "
            f"got {joints.shape} and {hand.shape}"
        )
    return np.concatenate([joints[list(G1_UPPER_BODY_JOINT_INDICES)], hand])


def validate_act_diffusion_config(cfg: PolicyConfig) -> None:
    """ckpt を読む前に分かる設定ミスを落とす。"""
    if cfg.mode not in ("none", "overlay"):
        raise ValueError(
            f"ACT / Diffusion Policy supports mode 'none' or 'overlay', got {cfg.mode!r}"
        )
    if cfg.dtype != "fp32":
        raise ValueError(
            f"ACT / Diffusion Policy runs in fp32 (trained in fp32), got dtype={cfg.dtype!r}"
        )
    if set(cfg.cams) != set(CAMERAS):
        raise ValueError(
            f"ACT / Diffusion Policy needs cams {[cam.value for cam in CAMERAS]}, "
            f"got {[cam.value for cam in cfg.cams]}"
        )


def _resolve_checkpoint_root(ckpt_ref: str, checkpoint_subdir: str | None) -> Path:
    """local directory か `HF-repo[@revision]` を 1 つの snapshot directory に解決する。

    repo root に最新 model、`checkpoints/<step>/pretrained_model` に途中 ckpt がある
    (Issue #139 の ckpt_uploader)。root を使う時は途中 ckpt を落とさない。
    """
    local_root = Path(ckpt_ref).expanduser()
    if local_root.is_dir():
        snapshot_root = local_root
    else:
        # lazy: env-isolated dependencies (huggingface_hub は推論 env にのみある)
        from huggingface_hub import constants as hf_constants
        from huggingface_hub import snapshot_download

        repo_id, separator, revision = ckpt_ref.partition("@")
        if not repo_id or (separator and not revision):
            raise ValueError(
                "ckpt_ref must be a local directory, HF repo, or HF-repo@revision; "
                f"got {ckpt_ref!r}"
            )
        patterns = (
            {"allow_patterns": [f"{checkpoint_subdir.rstrip('/')}/**"]}
            if checkpoint_subdir is not None
            else {"ignore_patterns": ["checkpoints/**"]}
        )
        if hf_constants.HF_HUB_OFFLINE:
            # 会場は実行時オフライン。worker の desktop env (huggingface_hub 1.28) は commit hash
            # 指定でも file 一覧の記録 (trees/<commit>.json) が無いとネットへ取りに行く。事前取得は
            # runtime env (1.20.1) で行うので記録は無い。cache の snapshot だけを見る (GR00T と同じ)。
            patterns["local_files_only"] = True
        snapshot_root = Path(
            snapshot_download(repo_id=repo_id, revision=revision or None, **patterns)
        )
    checkpoint_root = (
        snapshot_root / checkpoint_subdir if checkpoint_subdir is not None else snapshot_root
    )
    if not (checkpoint_root / "config.json").is_file():
        raise FileNotFoundError(f"LeRobot checkpoint not found: {checkpoint_root}")
    return checkpoint_root.resolve()


def _validate_checkpoint_config(config) -> None:
    """ckpt の入出力が runtime の 19D / 3 cam 契約と一致するかを重み load 前に確かめる。"""
    if config.type not in _SUPPORTED_POLICY_KINDS:
        raise ValueError(
            f"checkpoint policy type must be one of {_SUPPORTED_POLICY_KINDS}, "
            f"got {config.type!r}"
        )
    expected_images = {key: (3, *IMAGE_SHAPE_HWC[:2]) for key in _CAM_TO_FEATURE_KEY.values()}
    actual_images = {key: tuple(feature.shape) for key, feature in config.image_features.items()}
    if actual_images != expected_images:
        raise ValueError(
            f"checkpoint image features {actual_images} do not match the runtime "
            f"cameras {expected_images}"
        )
    state_shape = tuple(config.robot_state_feature.shape)
    action_shape = tuple(config.action_feature.shape)
    if state_shape != (STATE_DIM,) or action_shape != (ACTION_DIM,):
        raise ValueError(
            f"checkpoint state/action must be ({STATE_DIM},)/({ACTION_DIM},), "
            f"got {state_shape}/{action_shape}"
        )


class ActDiffusionModel:
    """ckpt の policy + processor を持ち、観測列 → 19D 絶対値 chunk を返す (lerobot 必須)。

    Attributes:
        policy_kind: "act" / "diffusion" (ckpt config.json の type)。
        n_obs_steps: 1 回の推論に使う観測数 (ACT 1 / DP 2)。
        chunk_len: 返す chunk の行数 (ACT chunk_size / DP horizon - n_obs_steps + 1)。
    """

    def __init__(self, policy, preprocessor, postprocessor) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        config = policy.config
        self._device = str(config.device)
        self.policy_kind: str = config.type
        self.n_obs_steps: int = int(config.n_obs_steps)
        self.chunk_len: int = int(
            config.chunk_size if config.type == "act" else config.n_action_steps
        )
        # 初回の同期推論と async 推論が重なると VRAM ピークが倍になるので直列化する
        # (GR00T / RAMEN-Ori と同じ)。
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_ref: str,
        checkpoint_subdir: str | None = None,
        device: str = "cuda",
    ) -> "ActDiffusionModel":
        # lazy: env-isolated dependencies (lerobot は推論 env = Python 3.12 のみ)
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        checkpoint_root = _resolve_checkpoint_root(ckpt_ref, checkpoint_subdir)
        config = PreTrainedConfig.from_pretrained(checkpoint_root)
        _validate_checkpoint_config(config)
        config.device = device
        # 学習時の設定 (ResNet18_Weights.IMAGENET1K_V1) のままだと、model を作る時点で torchvision が
        # ImageNet の重みをネットから取りに行く (会場は実行時オフライン)。backbone も ckpt の
        # state dict に全部入っていて strict=True で上書きされるので、初期値は使われない。
        config.pretrained_backbone_weights = None
        if config.type == "diffusion":
            # 標準の generate_actions は現在 tick から n_action_steps (学習 8) 行しか
            # 返さない。execution_steps 8 と同じ長さでは temporal ensemble の重なりが
            # 無くなるので、horizon のうち現在以降を全部返させる。
            config.n_action_steps = config.horizon - config.n_obs_steps + 1
        policy = get_policy_class(config.type).from_pretrained(
            checkpoint_root, config=config, strict=True
        )
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint_root),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        print(
            f"[act_diffusion] loaded {config.type} from {checkpoint_root} "
            f"(n_obs_steps={config.n_obs_steps}, device={device})",
            file=sys.stderr,
        )
        return cls(policy, preprocessor, postprocessor)

    def predict_chunk(
        self,
        frames_seq: Sequence[Mapping[CameraKey, np.ndarray]],
        states_seq: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """観測列 (古い順に n_obs_steps 個) → (chunk_len, 19) の絶対値 chunk と latency_ms。

        frames は model 入力そのもの (BGR uint8 480x640、overlay 済)。chunk の row 0 は
        最新観測の tick の action。
        """
        # lazy: env-isolated dependencies (torch は推論 env 前提)
        import torch

        if len(frames_seq) != self.n_obs_steps or len(states_seq) != self.n_obs_steps:
            raise ValueError(
                f"expected {self.n_obs_steps} observations, "
                f"got {len(frames_seq)} frames / {len(states_seq)} states"
            )
        t0 = time.monotonic_ns()
        with self._lock, torch.inference_mode():
            steps = [
                self._preprocessor(_to_lerobot_observation(frames, state, self._device))
                for frames, state in zip(frames_seq, states_seq)
            ]
            if self.n_obs_steps == 1:
                batch = steps[0]
            else:
                # 各 tick を正規化してから時間軸 (dim 1) に積む。DP の queue は空の
                # ままなので predict_action_chunk はこの batch をそのまま使う。
                keys = (_STATE_KEY, *_CAM_TO_FEATURE_KEY.values())
                batch = {key: torch.stack([step[key] for step in steps], dim=1) for key in keys}
            actions = self._postprocessor(self._policy.predict_action_chunk(batch))
            chunk = actions[0].detach().cpu().numpy().astype(np.float32)
        return chunk, (time.monotonic_ns() - t0) / 1e6

    def close(self) -> None:
        # lazy: env-isolated dependencies
        import torch

        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _to_lerobot_observation(
    frames: Mapping[CameraKey, np.ndarray], state: np.ndarray, device: str
) -> dict:
    """BGR uint8 HWC → RGB float [0, 1] CHW (学習時の jpg 読み込みと同じ変換)。

    画像は uint8 のまま device に送ってから float 化する (float 化してから送ると
    転送量が 4 倍になり、DP の 2 観測 × 3 cam で前処理が律速になる)。
    """
    # lazy: env-isolated dependencies
    import torch

    observation = {_STATE_KEY: torch.from_numpy(np.asarray(state, dtype=np.float32))}
    # scalar で割ると CUDA は逆数の乗算になり、CPU で割った学習時と 1 ulp ずれる。
    # device 上の tensor で割ると CPU と bit 単位で一致する。
    scale = torch.tensor(255.0, device=device)
    for cam, key in _CAM_TO_FEATURE_KEY.items():
        rgb = torch.from_numpy(np.ascontiguousarray(frames[cam][..., ::-1])).to(device)
        observation[key] = rgb.permute(2, 0, 1).float() / scale
    return observation


class _ActDiffusionWorkerClient:
    """Python 3.10 の DDS runtime から Python 3.12 の推論 env への proxy。

    `ActDiffusionModel` と同じ属性・`predict_chunk` を持ち、Policy からは区別しない。
    """

    def __init__(self, cfg: PolicyConfig) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        manifest = repo_root / "inference/desktop/pixi.toml"
        pixi = shutil.which("pixi") or str(Path.home() / ".pixi/bin/pixi")
        if not Path(pixi).is_file() or not manifest.is_file():
            raise FileNotFoundError(
                f"ACT / Diffusion worker runtime is unavailable: pixi={pixi} manifest={manifest}"
            )
        self._socket_path = Path(
            f"/tmp/iros_2026_ramen_act_diffusion_{os.getpid()}_{uuid.uuid4().hex}.sock"
        )
        command = [
            # --as-is: 実行時に環境の install も lock の更新もしない (会場は実行時オフライン。
            # image の build で --frozen で入れた環境をそのまま使う。GR00T の worker と同じ)
            pixi, "run", "--as-is", "--manifest-path", str(manifest),
            "python", "-m", "inference.desktop.lower_policy.policies.act_diffusion_worker",
            "--socket", str(self._socket_path),
            "--ckpt-ref", str(cfg.ckpt_ref),
            "--device", cfg.device,
        ]
        if cfg.checkpoint_subdir is not None:
            command.extend(["--checkpoint-subdir", cfg.checkpoint_subdir])
        print(
            "[act_diffusion] lerobot is isolated from the DDS runtime; "
            "starting Python 3.12 worker",
            file=sys.stderr,
        )
        self._process = subprocess.Popen(command, cwd=repo_root, start_new_session=True)
        self._connection: socket.socket | None = None
        self._predicted_once = False
        # async 推論と初回の同期推論が別 thread から来るので、要求と応答の組を直列化する。
        self._request_lock = threading.Lock()
        try:
            self._connect(timeout_s=_WORKER_STARTUP_TIMEOUT_S)
            info = self._request(timeout_s=_CONTROL_TIMEOUT_S, kind=np.asarray("describe"))
        except BaseException:
            # worker は別 session なので端末の Ctrl+C が届かない。起動中の中断
            # (KeyboardInterrupt) でも落としておかないと GPU を掴んだまま残る。
            self.close()
            raise
        self.policy_kind: str = scalar_text(info["policy_kind"], "policy_kind")
        self.n_obs_steps: int = int(info["n_obs_steps"].reshape(-1)[0])
        self.chunk_len: int = int(info["chunk_len"].reshape(-1)[0])

    def _connect(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"ACT / Diffusion worker exited during startup (exit_code={return_code})"
                )
            if self._socket_path.is_socket():
                connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                connection.settimeout(_CONTROL_TIMEOUT_S)
                try:
                    connection.connect(str(self._socket_path))
                except OSError:
                    connection.close()
                else:
                    self._connection = connection
                    return
            time.sleep(0.1)
        raise TimeoutError(f"timed out waiting {timeout_s:.0f}s for the ACT / Diffusion worker")

    def _request(self, *, timeout_s: float, **arrays) -> dict[str, np.ndarray]:
        with self._request_lock:
            if self._connection is None:
                raise RuntimeError("ACT / Diffusion worker connection is closed")
            self._connection.settimeout(timeout_s)
            try:
                send_archive(self._connection, **arrays)
                response = receive_archive(self._connection)
            except TimeoutError as exc:
                # 応答が途中で届くと次の要求と組がずれるので、この接続はもう使わない
                self._connection.close()
                self._connection = None
                raise TimeoutError(
                    f"ACT / Diffusion worker did not answer within {timeout_s:.0f}s"
                ) from exc
        if response is None:
            raise ConnectionError("ACT / Diffusion worker closed without a response")
        ok = response.get("ok")
        if ok is None or ok.size != 1 or int(ok.reshape(-1)[0]) != 1:
            error = response.get("error")
            detail = "unknown error" if error is None else str(error.reshape(-1)[0])
            raise RuntimeError(f"ACT / Diffusion worker failed: {detail}")
        return response

    def predict_chunk(
        self,
        frames_seq: Sequence[Mapping[CameraKey, np.ndarray]],
        states_seq: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """`ActDiffusionModel.predict_chunk` と同じ契約。latency は IPC 込みの往復時間。"""
        t0 = time.monotonic_ns()
        arrays: dict[str, np.ndarray] = {
            "kind": np.asarray("predict"),
            "states": np.stack([np.asarray(s, dtype=np.float32) for s in states_seq]),
        }
        for index, frames in enumerate(frames_seq):
            for cam in CAMERAS:
                arrays[f"{cam.value}_{index}"] = np.ascontiguousarray(frames[cam])
        timeout_s = _PREDICT_TIMEOUT_S if self._predicted_once else _FIRST_PREDICT_TIMEOUT_S
        response = self._request(timeout_s=timeout_s, **arrays)
        self._predicted_once = True
        chunk = np.asarray(response["action_chunk"], dtype=np.float32)
        if chunk.shape != (self.chunk_len, ACTION_DIM) or not np.isfinite(chunk).all():
            raise RuntimeError(
                f"ACT / Diffusion worker returned an invalid chunk: shape={chunk.shape}"
            )
        return chunk, (time.monotonic_ns() - t0) / 1e6

    def abort(self) -> None:
        """終了時に推論中の要求を解くため、接続と worker を落とす。"""
        connection = self._connection
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
            self._connection = None
        if self._process.poll() is None:
            self._process.terminate()

    def close(self) -> None:
        # 接続前 (起動中の中断など) は close 要求を送れないので、待たずに terminate する
        graceful = self._connection is not None
        if self._connection is not None:
            try:
                self._request(timeout_s=_CONTROL_TIMEOUT_S, kind=np.asarray("close"))
            except Exception:
                pass
            if self._connection is not None:
                self._connection.close()
                self._connection = None
        if self._process.poll() is None:
            try:
                self._process.wait(timeout=10.0 if graceful else 0.1)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5.0)
        self._socket_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _TickObservation:
    """1 tick 分の model 入力の素 (overlay は推論する時に描く)。"""

    frames: dict[CameraKey, np.ndarray]
    detections: tuple | None  # head_left の OBB。mode=none なら None
    state: np.ndarray


class ActDiffusionPolicy:
    """ACT / Diffusion Policy の Policy protocol 実装。

    毎 tick の観測を n_obs_steps 個まで保持し、`ChunkExecutor` に「その時点の履歴で
    推論する関数」を渡す。VlaSkill には blend 済みの 1 行だけを返す。
    """

    STATE_DIM = STATE_DIM
    ACTION_DIM = ACTION_DIM
    CAMERAS = CAMERAS
    # ChunkExecutor が chunk を合流させるので、VlaSkill 側の queue 経路は使わない。
    EXECUTION_HORIZON = 1
    build_state_from_raw = staticmethod(build_state_from_raw)

    def __init__(self, cfg: PolicyConfig, backend) -> None:
        validate_act_diffusion_config(cfg)
        self.cfg = cfg
        self._backend = backend
        self._executor = ChunkExecutor(
            action_dim=ACTION_DIM,
            temporal_lambda=cfg.temporal_lambda,
            replan_family=cfg.replan_family,
            execution_steps=cfg.execution_steps,
            thread_name_prefix="act-diffusion-replan",
        )
        self._history: deque[_TickObservation] = deque(maxlen=backend.n_obs_steps)

    @classmethod
    def from_ckpt(cls, cfg: PolicyConfig) -> "ActDiffusionPolicy":
        validate_act_diffusion_config(cfg)
        # 実機 runtime は Unitree DDS のため Python 3.10、lerobot 0.6 は 3.12 必須。
        # lerobot が無い env では worker process に分ける (GR00T と同じ境界)。
        if importlib.util.find_spec("lerobot") is None:
            backend = _ActDiffusionWorkerClient(cfg)
        else:
            backend = ActDiffusionModel.from_checkpoint(
                cfg.ckpt_ref, cfg.checkpoint_subdir, cfg.device
            )
        return cls(cfg, backend)

    def warmup(self, n_iter: int = 5) -> None:
        """初回 forward の CUDA 初期化 / cuDNN autotune を制御 loop の前に済ませる。"""
        frames = {cam: np.zeros(IMAGE_SHAPE_HWC, dtype=np.uint8) for cam in CAMERAS}
        state = np.zeros(STATE_DIM, dtype=np.float32)
        n_obs = self._backend.n_obs_steps
        for _ in range(n_iter):
            self._backend.predict_chunk([frames] * n_obs, [state] * n_obs)
        self.reset()

    def predict(self, obs: Observation) -> PolicyAction:
        t0 = time.monotonic_ns()
        tick = self._snapshot(obs)
        if self._history:
            self._history.append(tick)
        else:
            # skill 開始直後は最初の観測で履歴を埋める (LeRobot の select_action と同じ)
            self._history.extend([tick] * self._history.maxlen)
        history = tuple(self._history)
        target, metadata = self._executor.step(lambda: self._predict_from_history(history))
        target = target.astype(np.float32)
        return PolicyAction(
            action_chunk=target[None, :],
            latency_ms=(time.monotonic_ns() - t0) / 1e6,
            metadata={
                "mode": self.cfg.mode,
                "policy_kind": self._backend.policy_kind,
                "chunk_len": 1,
                "action_dim": ACTION_DIM,
                "model_chunk_len": self._backend.chunk_len,
                "overlay_detection_count": (
                    len(tick.detections) if tick.detections is not None else 0
                ),
                "action_min_19d": float(target.min()),
                "action_max_19d": float(target.max()),
                "action_arms_absmax_19d": float(np.abs(target[3:17]).max()),
                **metadata,
            },
        )

    def reset(self) -> None:
        """skill 遷移 / episode 開始時 (VlaSkill._on_start)。前 skill の chunk と観測を捨てる。"""
        self._executor.reset()
        self._history.clear()

    def close(self) -> None:
        self._executor.close(abort_pending=getattr(self._backend, "abort", None))
        self._backend.close()

    def _snapshot(self, obs: Observation) -> _TickObservation:
        frames = {}
        for cam in CAMERAS:
            frame = np.asarray(obs.frames_bgr[cam])
            if frame.shape != IMAGE_SHAPE_HWC or frame.dtype != np.uint8:
                raise ValueError(
                    f"{cam.value} must be uint8 {IMAGE_SHAPE_HWC}, "
                    f"got {frame.shape} {frame.dtype}"
                )
            # async 推論は後の tick に別 thread で読むので、camera buffer から切り離す
            frames[cam] = frame.copy()
        detections = None
        if self.cfg.mode == "overlay":
            if obs.obb_detections is None:
                raise RuntimeError(
                    "overlay mode requires live YOLO-OBB detections; refusing to "
                    "evaluate the overlay checkpoint on unannotated images"
                )
            detections = tuple(obs.obb_detections.get(CameraKey.HEAD_LEFT, []))
        state = np.array(obs.state, dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state must be ({STATE_DIM},), got {state.shape}")
        return _TickObservation(frames=frames, detections=detections, state=state)

    def _predict_from_history(
        self, history: tuple[_TickObservation, ...]
    ) -> tuple[np.ndarray, float]:
        frames_seq = []
        for tick in history:
            frames = tick.frames
            if tick.detections is not None:
                frames = dict(frames)
                overlaid = overlay_obb_on_frame(
                    tick.frames[CameraKey.HEAD_LEFT].copy(), list(tick.detections)
                )
                frames[CameraKey.HEAD_LEFT] = match_training_jpeg(
                    overlaid, self.cfg.overlay_jpeg_subsampling
                )
            frames_seq.append(frames)
        return self._backend.predict_chunk(frames_seq, [tick.state for tick in history])
