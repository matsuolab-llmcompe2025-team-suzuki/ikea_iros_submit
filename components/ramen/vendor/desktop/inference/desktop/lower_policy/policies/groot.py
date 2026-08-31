"""GR00T policy loader (Issue #125 Phase 2、C-axis inference integration)。

# 訓練時仕様の厳密追従

Batch key / dim / shape の唯一の真実:
- `model/subtask_policy_training/configs/subtask_training.json` (combined_task5_7 の
  cameras / state / action / groot config)
- `model/subtask_policy_training/gr00t/g1_full_body_mapping.py` (REAL_G1_RELATIVE_EEF
  state/action 組立)
- `lerobot.policies.groot.modeling_groot.GrootPolicy` + `processor_groot.py`

# GR00T combined_task5_7 契約

- **Base model**: `nvidia/GR00T-N1.7-3B` (revision `2fc962b...`)
- **Embodiment tag**: `real_g1_relative_eef_relative_joints`
- **State dim**: 49 (REAL_G1_RELATIVE_EEF、wrist EE 9D + arm/waist joints)
- **Action dim**: 53
- **chunk_size / n_action_steps**: 16
- **Image size**: (224, 224)、per-cam 3 cam layout:
    - `observation.images.head_left`
    - `observation.images.left_wrist`
    - `observation.images.right_wrist`
  (**head_right = cam_1 は subtask_training.json:ignored_video_keys で drop**)
- **Language prompt**: `subtask_training.json:subtasks[combined_task5_7].task` =
  "rotate and move table base (combined 5+7)"
- **Precision**: bf16 (use_bf16=true)

# 実装状態

- ✅ `build_state_from_raw` = 29-dim joint_state + 12D ee_state + 2D hand_state
  → 49D REAL_G1_RELATIVE_EEF state layout mapping (default env で test 通過)
- ✅ `_pre_processor` / model forward / `_post_processor` = lerobot 委譲 pipeline
- ✅ 53D → 19D action adapter (`slice_53d_to_19d`) = `Gr00tPolicy.predict()` の
  出力を VlaSkill 19D contract に落とす、default env で test 通過。同 adapter は
  `Gr00tPolicy` を使う全 GR00T skill (baseline/overlay/insert/rotate_leg/flip)
  で自動的に有効。`Gr00tPolicyPickLegs` は別 embodiment=別 class で対象外。
- ⚠️ RELATIVE→ABSOLUTE 復元 verify 未完: `_post_processor` が
  `action_configs.rep="RELATIVE"` の arm slice を state 加算で abs 化する仕様と
  想定して adapter は arm slice をそのまま渡すが、実 model 出力の実測 verify は
  未完。**実機初回起動時に read-only preflight** で
  `PolicyAction.metadata.action_arms_absmax_19d` を確認:
  - |max| > ~0.3 rad オーダー = 通常 (abs 化済)、そのまま dispatch OK
  - |max| < ~0.1 rad ばかり = post_processor が delta のまま返してる疑い、
    dispatch せず `slice_53d_to_19d` の後に `+ current_arm_joints` の補正 patch
    を追加して再 verify

Env 手順は `inference/desktop/pixi.toml` header +
`docs/handoff/issue_125_c_axis_inference_task.md §Env` 参照。

# 使用例 (integration 後、Sakura H100 想定)

```python
cfg = PolicyConfig(
    mode="none",
    ckpt_ref="Team-RAMEN/IROS2026_RAMEN_hara_task_5_7_groot_baseline_100k_v1",
    dtype="bf16",
    cams=Gr00tPolicy.CAMERAS,
)
policy = Gr00tPolicy.from_ckpt(cfg)
policy.warmup(n_iter=5)
raw = joint_state_source.get_raw()  # dict[str, np.ndarray]
state = Gr00tPolicy.build_state_from_raw(raw)
obs = Observation(
    frames_bgr={
        CameraKey.HEAD_LEFT: head_left_bgr,
        CameraKey.WRIST_LEFT: wrist_left_bgr,
        CameraKey.WRIST_RIGHT: wrist_right_bgr,
    },
    state=state,
    skill_id=None,
    language="rotate and move table base (combined 5+7)",
    obb_detections=None,   # or overlay-drawn frames_bgr instead
    timestamp_ns=time.monotonic_ns(),
)
action = policy.predict(obs)  # PolicyAction with action_chunk (16, 53)
```
"""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import json
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

import numpy as np

from inference.desktop.lower_policy.policies.base import (
    CameraKey,
    G1_LEFT_ARM_SLICE,
    G1_RIGHT_ARM_SLICE,
    G1_WAIST_SLICE,
    Observation,
    PolicyAction,
    PolicyConfig,
    RawRobotState,
)


# ---- GR00T combined_task5_7 契約定数 (subtask_training.json 唯一の真実) ---- #

STATE_DIM: int = 49   # REAL_G1_RELATIVE_EEF_STATE_DIM (g1_full_body_mapping.py)
ACTION_DIM: int = 53  # REAL_G1_RELATIVE_EEF_ACTION_DIM
CHUNK_LEN: int = 16   # subtask_training.json:groot.chunk_size / n_action_steps
IMAGE_HW: tuple[int, int] = (224, 224)  # GrootConfig.image_size default

# GR00T combined_task5_7 で使う 3 cam layout (head_right は drop)
CAMERAS: tuple[CameraKey, ...] = (
    CameraKey.HEAD_LEFT,
    CameraKey.WRIST_LEFT,
    CameraKey.WRIST_RIGHT,
)

# CameraKey → GR00T LeRobot key (subtask_training.json:cameras)
_CAM_TO_LEROBOT_KEY: dict[CameraKey, str] = {
    CameraKey.HEAD_LEFT: "observation.images.head_left",
    CameraKey.WRIST_LEFT: "observation.images.left_wrist",
    CameraKey.WRIST_RIGHT: "observation.images.right_wrist",
}

# subtask_training.json:subtasks.combined_task5_7.task
DEFAULT_LANGUAGE_PROMPT: str = "rotate and move table base (combined 5+7)"

# g1_full_body_mapping.py:REAL_G1_RELATIVE_EEF_EMBODIMENT_ID
EMBODIMENT_ID: int = 25


# ---- 53D REAL_G1_RELATIVE_EEF action → 19D VlaSkill contract mapping ---- #
#
# GR00T (task_5+7 combined_task5_7) の action は REAL_G1_RELATIVE_EEF layout の
# 53 dim (g1_full_body_mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES)。実機 arm-only
# 制御 (Regular Mode + G1ArmActuator + WaistActuator + HandActuator) は
# vla_skill.ACTION_DIM_TOTAL=19 (waist3 + arm14 + hand2) を受け取る契約。
# 差分の 34 dim は下記の 3 種で構成 = 全て実機 dispatch では捨てる/畳み込む:
#   - EEF 18D (left_wrist_eef_9d[0:9] + right_wrist_eef_9d[9:18]):
#     whole-body IK 用の Cartesian 目標、実機は joint 直接指令のため drop。
#   - hand 14D (left_hand[18:25] + right_hand[25:32]):
#     Dex1-1 は per side 1 DOF (open/close)、GR00T layout は 7 joint synergy
#     (index_0/1, middle_0/1, thumb_0/1/2) で学習。**逆射影 (least-squares
#     projection) で 1D Dex1 に畳み込む**、dex1_hand_synergy.hand_to_dex1() 使用。
#   - base 4D (base_height_command 1 + navigate_command 3):
#     下半身 / 歩行制御用、Regular Mode arm-only では非対応。drop。
#
# 【Hand encoding の verify 結果】(2026-08-30、user 指示で HF 実測):
# 全 4 GR00T ckpt (hara_task_5_7 / takada_insert / takada_rotate / suzuki_flip)
# の policy_postprocessor.json:raw_stats.action.{left,right}_hand は 7 dim 全て
# active (range 0.7〜1.5、mean nonzero) = **全 variant で synergy encoding**。
# 単純に position [18] / [25] を取ると 7D 情報の 1/7 しか使わず精度劣化するため
# hand_to_dex1() で必ず逆射影する (variant 分岐なし)。詳細は
# model/subtask_policy_training/gr00t/dex1_hand_synergy.py と
# assets/dex1_g1_synergy.json (Dex1 open=4.5 / closed=0.0 rad の calibration)。
#
# 【RELATIVE rep の扱い】(重要):
# g1_full_body_mapping.REAL_G1_RELATIVE_EEF_ACTION_CONFIGS で
# left_arm / right_arm は "rep": "RELATIVE" として訓練済。Isaac-GR00T processor
# (make_groot_pre_post_processors の post pipeline) は action_config.rep を読み、
# RELATIVE の場合 state[state_key] を加算して absolute 化して返す仕様。
# 本 adapter は post_processor が **absolute を返す前提** で slice のみ行う。
# 実機で action_chunk[:, 3:17] (arm 14D) が **明らかに delta 相当の小さい値**
# ばかりの場合、この前提が壊れているので caller が current_arm_joints を加算
# する要 (docs/inference/realmachine_smoke_checklist.md の GR00T セクション参照)。
_ACTION_53D_WAIST_SLICE: slice = slice(46, 49)      # waist yaw/roll/pitch (ABSOLUTE)
_ACTION_53D_LEFT_ARM_SLICE: slice = slice(32, 39)   # 7D (訓練 rep=RELATIVE、post で ABS 化想定)
_ACTION_53D_RIGHT_ARM_SLICE: slice = slice(39, 46)  # 7D (同上)
_ACTION_53D_LEFT_HAND_SLICE: slice = slice(18, 25)  # 7D synergy → hand_to_dex1 で 1D 化
_ACTION_53D_RIGHT_HAND_SLICE: slice = slice(25, 32) # 同上

_VLASKILL_ACTION_DIM: int = 19  # vla_skill.ACTION_DIM_TOTAL と一致すべき

# GR00T N1.7 checkpoint は約 12 GiB の FP32 safetensors。LeRobot の通常 load は
# config.device="cuda" の場合、safetensors を直接 GPU へ読み込んだ後に model.to(cuda)
# も行うため、ロード中だけ GPU / pinned host memory のピークが大きくなる。この PC
# (RAM 30 GiB / RTX 5090 32 GiB) では NVIDIA driver の NV_ERR_NO_MEMORY と desktop
# freeze を実測したため、CPU staging に必要な余白を fail-closed で要求する。
_GROOT_HOST_LOAD_OVERHEAD_BYTES: int = 8 * 1024**3
_GROOT_GPU_LOAD_OVERHEAD_BYTES: int = 12 * 1024**3

# Qwen ties the token embedding and language-model head.  Some Team-RAMEN
# training environments serialized both aliases, while LeRobot 0.6.1's current
# module state_dict exposes only lm_head.  The extra key is safe to ignore only
# when its bytes are exactly identical to the canonical lm_head tensor.
_GROOT_ALLOWED_TIED_CHECKPOINT_ALIASES: dict[str, str] = {
    "_groot_model.backbone.model.model.language_model.embed_tokens.weight":
        "_groot_model.backbone.model.lm_head.weight",
}


@dataclass(frozen=True)
class GrootArtifactContract:
    """Deployment-relevant fields sealed into one checkpoint snapshot."""

    policy_type: str
    chunk_size: int
    n_action_steps: int
    video_horizon: int
    history_offset_ticks: int
    expert_role: str | None


def _read_groot_artifact_contract(checkpoint_root: Path) -> GrootArtifactContract:
    """Validate dimensions and temporal semantics before allocating a model."""

    config = json.loads((checkpoint_root / "config.json").read_text(encoding="utf-8"))
    processor = json.loads(
        (checkpoint_root / "policy_preprocessor.json").read_text(encoding="utf-8")
    )
    policy_type = str(config.get("type", "groot"))
    if policy_type not in {"groot", "furniture_groot"}:
        raise ValueError(f"unsupported GR00T artifact type: {policy_type!r}")
    if config.get("input_features", {}).get("observation.state", {}).get("shape") != [STATE_DIM]:
        raise ValueError("GR00T artifact must expose the 49-D REAL_G1 state")
    if config.get("output_features", {}).get("action", {}).get("shape") != [ACTION_DIM]:
        raise ValueError("GR00T artifact must expose the 53-D logical action")
    if config.get("embodiment_tag") != "real_g1_relative_eef_relative_joints":
        raise ValueError("GR00T artifact embodiment does not match the physical adapter")
    if not bool(config.get("use_relative_actions", False)):
        raise ValueError("GR00T artifact must retain relative EEF/arm decoding")

    pack_steps = [
        step for step in processor.get("steps", [])
        if step.get("registry_name") == "groot_n1_7_pack_inputs_v1"
    ]
    if len(pack_steps) != 1:
        raise ValueError("GR00T artifact must contain exactly one pack-input step")
    video_horizon = int(pack_steps[0].get("config", {}).get("video_horizon", 1))
    if video_horizon not in {1, 2}:
        raise ValueError(f"unsupported GR00T video_horizon={video_horizon}")
    history_offset = 20 if policy_type == "furniture_groot" else 1
    if policy_type == "furniture_groot":
        if video_horizon != 2:
            raise ValueError(
                "Furniture-GR00T must preserve the two-frame [-20,0] image pair"
            )
        if int(config.get("chunk_size", -1)) != 40:
            raise ValueError("Furniture-GR00T must retain H40")
        if int(config.get("max_state_dim", -1)) != 132 or int(
            config.get("max_action_dim", -1)
        ) != 132:
            raise ValueError("Furniture-GR00T packed dimensions must be 132D")
        if int(config.get("valid_action_dim", -1)) != 46:
            raise ValueError("Furniture-GR00T valid action span must be slots 0:46")
    return GrootArtifactContract(
        policy_type=policy_type,
        chunk_size=int(config.get("chunk_size", CHUNK_LEN)),
        n_action_steps=int(config.get("n_action_steps", CHUNK_LEN)),
        video_horizon=video_horizon,
        history_offset_ticks=history_offset,
        expert_role=config.get("expert_role"),
    )


def _safetensor_ranges_equal(path: Path, left: str, right: str) -> bool:
    """Compare two tensors in one safetensors file without materializing them."""

    with path.open("rb") as header_file:
        header_size = int.from_bytes(header_file.read(8), "little")
        header = json.loads(header_file.read(header_size))
    try:
        left_offsets = header[left]["data_offsets"]
        right_offsets = header[right]["data_offsets"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"checkpoint tensor metadata is missing for tied alias {left!r}/{right!r}"
        ) from exc
    left_size = int(left_offsets[1]) - int(left_offsets[0])
    right_size = int(right_offsets[1]) - int(right_offsets[0])
    if left_size != right_size:
        return False
    data_start = 8 + header_size
    with path.open("rb") as left_file, path.open("rb") as right_file:
        left_file.seek(data_start + int(left_offsets[0]))
        right_file.seek(data_start + int(right_offsets[0]))
        remaining = left_size
        while remaining:
            chunk_size = min(8 * 1024**2, remaining)
            if left_file.read(chunk_size) != right_file.read(chunk_size):
                return False
            remaining -= chunk_size
    return True


def _validate_groot_checkpoint_keys(
    checkpoint_file: Path, expected_keys: set[str]
) -> None:
    """Fail closed on every checkpoint schema mismatch except an exact tied alias."""

    from safetensors import safe_open

    with safe_open(checkpoint_file, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
    missing = expected_keys - actual_keys
    # Some serializers keep only the canonical lm_head side of the tied Qwen
    # embedding. It is safe to reconstruct the missing alias from that exact
    # tensor; the opposite case is checked byte-for-byte below.
    for alias, canonical in _GROOT_ALLOWED_TIED_CHECKPOINT_ALIASES.items():
        if alias in missing and canonical in actual_keys:
            missing.remove(alias)
    unexpected = actual_keys - expected_keys
    allowed_unexpected = set(_GROOT_ALLOWED_TIED_CHECKPOINT_ALIASES)
    disallowed = unexpected - allowed_unexpected
    if missing or disallowed:
        raise RuntimeError(
            "GR00T checkpoint schema mismatch: "
            f"missing={sorted(missing)[:10]} unexpected={sorted(disallowed)[:10]}"
        )
    for alias in sorted(unexpected):
        canonical = _GROOT_ALLOWED_TIED_CHECKPOINT_ALIASES[alias]
        if canonical not in actual_keys or not _safetensor_ranges_equal(
            checkpoint_file, alias, canonical
        ):
            raise RuntimeError(
                "GR00T tied checkpoint alias differs from its canonical tensor: "
                f"alias={alias!r} canonical={canonical!r}"
            )


def _resolve_groot_checkpoint_root(
    checkpoint_ref: str, checkpoint_subdir: str | None = None
) -> Path:
    """Resolve a local directory or an immutable ``HF-repo@revision`` snapshot.

    Loading ``model.safetensors`` from one revision while restoring config and
    processors from a moving ``main`` can silently change action decoding.  A
    single resolved snapshot directory is therefore the source of truth for all
    checkpoint-owned files.
    """

    local_root = Path(checkpoint_ref).expanduser()
    if local_root.is_dir():
        snapshot_root = local_root.resolve()
        checkpoint_root = (
            snapshot_root / checkpoint_subdir
            if checkpoint_subdir is not None
            else snapshot_root
        )
        if not checkpoint_root.is_dir():
            raise FileNotFoundError(
                f"GR00T checkpoint directory not found: {checkpoint_root}"
            )
        return checkpoint_root.resolve()

    from huggingface_hub import snapshot_download

    repo_id = checkpoint_ref
    revision: str | None = None
    if "@" in checkpoint_ref:
        repo_id, revision = checkpoint_ref.rsplit("@", 1)
        if not repo_id or not revision:
            raise ValueError(
                "GR00T ckpt_ref must be a local directory, HF repo, or "
                f"HF-repo@revision; got {checkpoint_ref!r}"
            )
    download_kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "revision": revision,
    }
    if checkpoint_subdir is not None:
        # Multi-step Trainer repositories can contain tens of GiB per step.
        # Fetch only the explicitly selected immutable checkpoint.
        download_kwargs["allow_patterns"] = [
            f"{checkpoint_subdir.rstrip('/')}/**"
        ]
    snapshot_root = Path(snapshot_download(**download_kwargs)).resolve()
    checkpoint_root = (
        snapshot_root / checkpoint_subdir
        if checkpoint_subdir is not None
        else snapshot_root
    )
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(
            f"GR00T checkpoint directory not found: {checkpoint_root}"
        )
    return checkpoint_root.resolve()


def _host_mem_available_bytes() -> int | None:
    """Linux MemAvailable。取得不能時はNone（platform依存testを壊さない）。"""

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _process_rss_bytes() -> int | None:
    """Current process RSS from /proc, for load diagnostics only."""

    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _runtime_memory_snapshot() -> str:
    """Compact host/CUDA memory diagnostic for the first heavy prediction."""

    rss = _process_rss_bytes()
    available = _host_mem_available_bytes()
    parts = [
        f"rss={rss / 1024**3:.2f}GiB" if rss is not None else "rss=unknown",
        (
            f"host_available={available / 1024**3:.2f}GiB"
            if available is not None
            else "host_available=unknown"
        ),
    ]
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            parts.extend(
                (
                    f"cuda_allocated={torch.cuda.memory_allocated() / 1024**3:.2f}GiB",
                    f"cuda_reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GiB",
                    f"cuda_free={free / 1024**3:.2f}/{total / 1024**3:.2f}GiB",
                )
            )
    except ImportError:
        pass
    return ", ".join(parts)


def _release_host_load_memory() -> tuple[int | None, int | None, bool]:
    """Return CPU staging allocations to the OS after the CUDA transfer."""

    rss_before = _process_rss_bytes()
    gc.collect()
    trimmed = False
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        trimmed = bool(malloc_trim(0))
    except (AttributeError, OSError):
        pass
    rss_after = _process_rss_bytes()
    return rss_before, rss_after, trimmed


def _require_groot_load_headroom(
    *,
    checkpoint_bytes: int,
    host_available_bytes: int | None,
    gpu_free_bytes: int | None,
) -> None:
    """重いmodel構築前にRAM/VRAM不足を検出し、desktop freezeを防ぐ。"""

    host_required = checkpoint_bytes + _GROOT_HOST_LOAD_OVERHEAD_BYTES
    if host_available_bytes is not None and host_available_bytes < host_required:
        raise MemoryError(
            "insufficient host memory for memory-safe GR00T load: "
            f"available={host_available_bytes / 1024**3:.1f}GiB "
            f"required>={host_required / 1024**3:.1f}GiB "
            f"checkpoint={checkpoint_bytes / 1024**3:.1f}GiB"
        )

    gpu_required = checkpoint_bytes + _GROOT_GPU_LOAD_OVERHEAD_BYTES
    if gpu_free_bytes is not None and gpu_free_bytes < gpu_required:
        raise MemoryError(
            "insufficient free GPU memory for GR00T load/warmup: "
            f"free={gpu_free_bytes / 1024**3:.1f}GiB "
            f"required>={gpu_required / 1024**3:.1f}GiB "
            f"checkpoint={checkpoint_bytes / 1024**3:.1f}GiB"
        )


def _wait_for_groot_host_load_headroom(
    *,
    checkpoint_bytes: int,
    timeout_s: float = 15.0,
    poll_s: float = 0.5,
) -> int | None:
    """Wait briefly for transient HF download/reconstruction memory to leave.

    Hugging Face/Xet can finish reconstructing a multi-GB safetensors file only
    milliseconds before this process reaches the load guard.  Its helper
    processes and reclaimable mappings may therefore make ``MemAvailable``
    momentarily miss the conservative guard by a few MiB.  Do not weaken the
    guard: release this process' garbage, wait a bounded interval, and require
    the exact same headroom before model construction.
    """

    if timeout_s < 0.0 or poll_s <= 0.0:
        raise ValueError("GR00T headroom wait parameters are invalid")
    required = checkpoint_bytes + _GROOT_HOST_LOAD_OVERHEAD_BYTES
    deadline = time.monotonic() + timeout_s
    last_available = _host_mem_available_bytes()
    while last_available is not None and last_available < required:
        gc.collect()
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass
        if time.monotonic() >= deadline:
            _require_groot_load_headroom(
                checkpoint_bytes=checkpoint_bytes,
                host_available_bytes=last_available,
                gpu_free_bytes=None,
            )
        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))
        last_available = _host_mem_available_bytes()
    return last_available


def _load_groot_policy_streaming(
    *,
    load_cfg: object,
    checkpoint_file: Path,
    device: str,
    target_dtype: object,
    policy_base_class: type | None = None,
) -> object:
    """Construct on meta and stream the validated checkpoint directly to GPU.

    This is the same loading strategy used by the physical N1.7 deployment
    worker.  It changes loading mechanics only: architecture, checkpoint
    tensors, processors and action semantics remain checkpoint-owned.
    """

    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config
    from lerobot.policies.groot.modeling_groot import (
        GrootPolicy as _LrGroot,
        _tie_unused_qwen_lm_head,
    )
    from safetensors import safe_open

    runtime_policy_base = policy_base_class or _LrGroot

    class _StreamingInferenceGrootPolicy(runtime_policy_base):
        def _create_groot_model(self):
            raw_config = GR00TN17Config.from_pretrained(
                self.config.base_model_path,
                local_files_only=True,
            )
            raw_config.tune_llm = False
            raw_config.tune_visual = False
            raw_config.tune_projector = False
            raw_config.tune_diffusion_model = False
            raw_config.tune_vlln = False
            raw_config.tune_top_llm_layers = 0
            raw_config.use_flash_attention = self.config.use_flash_attention
            raw_config.load_bf16 = target_dtype == torch.bfloat16
            raw_config.backbone_trainable_params_fp32 = False
            with init_empty_weights():
                model = GR00TN17(
                    raw_config,
                    load_backbone_weights=False,
                    transformers_loading_kwargs={
                        "trust_remote_code": True,
                        "local_files_only": True,
                    },
                )
                backbone = getattr(model, "backbone", None)
                qwen_model = getattr(backbone, "model", None)
                if qwen_model is not None:
                    _tie_unused_qwen_lm_head(qwen_model)
            print("[groot-worker] meta model constructed", file=sys.stderr)

            inner_keys = set(model.state_dict())
            allowed_aliases = set(_GROOT_ALLOWED_TIED_CHECKPOINT_ALIASES)
            with safe_open(checkpoint_file, framework="pt", device="cpu") as reader:
                for wrapper_key in sorted(reader.keys()):
                    prefix = "_groot_model."
                    if not wrapper_key.startswith(prefix):
                        # Furniture-GR00T stores diagnostic heads on the outer
                        # policy. They are loaded after this base model exists.
                        continue
                    inner_key = wrapper_key[len(prefix):]
                    # Some LeRobot versions omit the tied embedding alias from
                    # state_dict; others expose it as a separate meta
                    # parameter. Load it when present so retie() cannot retain
                    # a meta tensor, and skip it only for the omitted form.
                    if wrapper_key in allowed_aliases and inner_key not in inner_keys:
                        continue
                    tensor = reader.get_tensor(wrapper_key)
                    dtype = (
                        target_dtype
                        if torch.is_floating_point(tensor)
                        else tensor.dtype
                    )
                    set_module_tensor_to_device(
                        model,
                        inner_key,
                        device,
                        value=tensor,
                        dtype=dtype,
                    )
                    # Reconstruct an omitted tied input-embedding alias from
                    # the canonical lm_head tensor before retie().
                    for alias, canonical in _GROOT_ALLOWED_TIED_CHECKPOINT_ALIASES.items():
                        if wrapper_key == canonical:
                            alias_inner = alias[len(prefix):]
                            if alias_inner in inner_keys:
                                set_module_tensor_to_device(
                                    model,
                                    alias_inner,
                                    device,
                                    value=tensor,
                                    dtype=dtype,
                                )
                    del tensor
            print("[groot-worker] checkpoint tensors streamed", file=sys.stderr)

            backbone = getattr(model, "backbone", None)
            qwen_model = getattr(backbone, "model", None)
            if qwen_model is not None:
                _tie_unused_qwen_lm_head(qwen_model)
            missing_meta = [
                f"parameter:{name}"
                for name, value in model.named_parameters()
                if value.device.type == "meta"
            ] + [
                f"buffer:{name}"
                for name, value in model.named_buffers()
                if value.device.type == "meta"
            ]
            if missing_meta:
                raise RuntimeError(
                    "GR00T checkpoint left tensors on the meta device: "
                    + ", ".join(missing_meta[:20])
                )
            model.to(device)
            model.backbone.set_trainable_parameters(
                tune_visual=False,
                tune_llm=False,
                tune_top_llm_layers=0,
            )
            model.action_head.set_trainable_parameters(
                tune_projector=False,
                tune_diffusion_model=False,
                tune_vlln=False,
            )
            model.eval()
            print("[groot-worker] model dispatched", file=sys.stderr)
            return model

    policy = _StreamingInferenceGrootPolicy(load_cfg)
    expected_keys = set(policy.state_dict())
    _validate_groot_checkpoint_keys(checkpoint_file, expected_keys)
    with safe_open(checkpoint_file, framework="pt", device="cpu") as reader:
        for key in sorted(reader.keys()):
            if key.startswith("_groot_model.") or key in _GROOT_ALLOWED_TIED_CHECKPOINT_ALIASES:
                continue
            tensor = reader.get_tensor(key)
            dtype = target_dtype if torch.is_floating_point(tensor) else tensor.dtype
            set_module_tensor_to_device(
                policy,
                key,
                device,
                value=tensor,
                dtype=dtype,
            )
            del tensor
    policy.to(device)
    missing_meta = [
        f"parameter:{name}"
        for name, value in policy.named_parameters()
        if value.device.type == "meta"
    ] + [
        f"buffer:{name}"
        for name, value in policy.named_buffers()
        if value.device.type == "meta"
    ]
    if missing_meta:
        raise RuntimeError(
            "GR00T outer policy left tensors on the meta device: "
            + ", ".join(missing_meta[:20])
        )
    return policy


class _GrootWorkerClient:
    """Proxy from the Python 3.10 DDS runtime to the Python 3.12 model env."""

    def __init__(self, cfg: PolicyConfig) -> None:
        from inference.desktop.lower_policy.policies.groot_worker_protocol import (
            receive_archive,
            send_archive,
        )

        repo_root = Path(__file__).resolve().parents[4]
        checkpoint_root = _resolve_groot_checkpoint_root(
            str(cfg.ckpt_ref), cfg.checkpoint_subdir
        )
        self.artifact_contract = _read_groot_artifact_contract(checkpoint_root)
        self.video_horizon = self.artifact_contract.video_horizon
        self.history_offset_ticks = self.artifact_contract.history_offset_ticks
        self._mode = cfg.mode
        self._socket_path = Path(
            f"/tmp/iros_2026_ramen_groot_{os.getpid()}_{uuid.uuid4().hex}.sock"
        )
        # VENDOR PATCH (Team RAMEN、boundary container 用):
        # RAMEN_WORKER_PYTHON_53D (lerobot 0.6.1 の python) 指定時はそれで worker を
        # 直接起動 (container は pixi 不在、pick=0.6.0 と別 env のため専用変数)。
        # 互換のため RAMEN_WORKER_PYTHON も見る。未指定なら従来の pixi run にフォールバック。
        worker_python = os.environ.get("RAMEN_WORKER_PYTHON_53D") or os.environ.get(
            "RAMEN_WORKER_PYTHON"
        )
        if worker_python and not Path(worker_python).is_file():
            worker_python = shutil.which(worker_python) or worker_python
        if worker_python and Path(worker_python).is_file():
            launch_prefix = [str(worker_python)]
        else:
            manifest = repo_root / "inference/desktop/pixi.toml"
            pixi = shutil.which("pixi") or "/home/ubuntu/.pixi/bin/pixi"
            if not Path(pixi).is_file() or not manifest.is_file():
                raise FileNotFoundError(
                    f"GR00T worker runtime is unavailable: pixi={pixi} manifest={manifest}"
                )
            launch_prefix = [pixi, "run", "--manifest-path", str(manifest), "python"]
        command = [
            *launch_prefix,
            "-m",
            "inference.desktop.lower_policy.policies.groot_worker",
            "--socket",
            str(self._socket_path),
            # Overlay is rendered in this Python 3.10 client before frames
            # cross the worker boundary.  The worker therefore always receives
            # model-ready images and must not attempt a second overlay pass.
            "--mode",
            "none",
            "--ckpt-ref",
            str(cfg.ckpt_ref),
            "--device",
            cfg.device,
            "--dtype",
            cfg.dtype,
        ]
        if cfg.checkpoint_subdir is not None:
            command.extend(["--checkpoint-subdir", cfg.checkpoint_subdir])
        print(
            "[groot] lerobot is isolated from the DDS runtime; starting Python 3.12 worker",
            file=sys.stderr,
        )
        self._process = subprocess.Popen(
            command,
            cwd=repo_root,
            start_new_session=True,
        )
        self._connection: socket.socket | None = None
        self._receive_archive = receive_archive
        self._send_archive = send_archive
        # Async replanning can submit a background inference while the control
        # thread checks/falls back.  The Unix stream protocol is strictly
        # request/response, so serialize complete exchanges.
        self._request_lock = threading.Lock()
        self._last_metadata: dict[str, object] = {}
        deadline = time.monotonic() + 300.0
        try:
            while time.monotonic() < deadline:
                return_code = self._process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"GR00T worker exited during startup (exit_code={return_code})"
                    )
                if self._socket_path.is_socket():
                    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    connection.settimeout(600.0)
                    try:
                        connection.connect(str(self._socket_path))
                    except OSError:
                        connection.close()
                    else:
                        self._connection = connection
                        return
                time.sleep(0.1)
            raise TimeoutError("timed out waiting 300s for the GR00T worker")
        except Exception:
            self.close()
            raise

    def _request(self, **arrays) -> dict[str, np.ndarray]:
        with self._request_lock:
            if self._connection is None:
                raise RuntimeError("GR00T worker connection is closed")
            self._send_archive(self._connection, **arrays)
            response = self._receive_archive(self._connection)
            if response is None:
                raise ConnectionError("GR00T worker closed without a response")
            ok = response.get("ok")
            if ok is None or ok.size != 1 or int(ok.reshape(-1)[0]) != 1:
                error = response.get("error")
                detail = "unknown error" if error is None else str(error.reshape(-1)[0])
                raise RuntimeError(f"GR00T worker failed: {detail}")
            return response

    def warmup(self, n_iter: int) -> None:
        self._request(
            kind=np.asarray("warmup"),
            count=np.asarray([n_iter], dtype=np.int64),
        )

    def predict(self, obs: Observation) -> PolicyAction:
        frames, overlay_detection_count = _prepare_gr00t_frames(obs, self._mode)
        request = dict(
            kind=np.asarray("predict"),
            state=np.asarray(obs.state, dtype=np.float32),
            head_left=np.asarray(frames[CameraKey.HEAD_LEFT]),
            wrist_left=np.asarray(frames[CameraKey.WRIST_LEFT]),
            wrist_right=np.asarray(frames[CameraKey.WRIST_RIGHT]),
            language=np.asarray(obs.language or DEFAULT_LANGUAGE_PROMPT),
            timestamp_ns=np.asarray([obs.timestamp_ns], dtype=np.int64),
        )
        if self.video_horizon == 2:
            if obs.frames_bgr_prev is None:
                raise RuntimeError("temporal GR00T observation is missing its history frame")
            request.update(
                head_left_prev=np.asarray(obs.frames_bgr_prev[CameraKey.HEAD_LEFT]),
                wrist_left_prev=np.asarray(obs.frames_bgr_prev[CameraKey.WRIST_LEFT]),
                wrist_right_prev=np.asarray(obs.frames_bgr_prev[CameraKey.WRIST_RIGHT]),
            )
        response = self._request(**request)
        chunk = np.asarray(response["action_chunk"], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != _VLASKILL_ACTION_DIM:
            raise RuntimeError(f"GR00T worker returned invalid action shape {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise RuntimeError("GR00T worker returned a non-finite action")
        metadata = json.loads(str(response["metadata_json"].reshape(-1)[0]))
        metadata["mode"] = self._mode
        metadata["overlay_detection_count"] = overlay_detection_count
        self._last_metadata = dict(metadata)
        latency_ms = float(response["latency_ms"].reshape(-1)[0])
        return PolicyAction(action_chunk=chunk, latency_ms=latency_ms, metadata=metadata)

    def abort(self) -> None:
        """Force an in-flight worker request to unblock during bounded shutdown."""

        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass
            self._connection = None
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                self._request(kind=np.asarray("close"))
            except Exception:
                pass
            try:
                connection.close()
            except OSError:
                pass
            self._connection = None
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        socket_path = getattr(self, "_socket_path", None)
        if socket_path is not None:
            socket_path.unlink(missing_ok=True)

def slice_53d_to_19d(action_chunk_53d: np.ndarray) -> np.ndarray:
    """GR00T 53D REAL_G1_RELATIVE_EEF action chunk → 19D VlaSkill contract。

    Layout mapping (上記 module docstring 参照):
        19D[0:3]   = 53D[46:49]                       (waist、ABSOLUTE)
        19D[3:10]  = 53D[32:39]                       (left arm、post で ABS 化想定)
        19D[10:17] = 53D[39:46]                       (right arm、同上)
        19D[17]    = hand_to_dex1(53D[18:25], left)   (Dex1 L grip = 7D synergy 逆射影)
        19D[18]    = hand_to_dex1(53D[25:32], right)  (Dex1 R grip 同上)

    Args:
        action_chunk_53d: (chunk_len, 53) float、GR00T `_post_processor` 通過後の
            denormalize + relative→absolute 済 action chunk。

    Returns:
        (chunk_len, 19) float、VlaSkill が dispatch できる layout。dtype は入力を
        保持 (VlaSkill 側で np.float32 に統一される)。hand slice のみ float 精度で
        projection 計算のため dtype 保持は arm/waist に対して。

    Raises:
        ValueError: 入力 shape が (*, 53) でない場合。
    """
    if action_chunk_53d.ndim != 2 or action_chunk_53d.shape[1] != ACTION_DIM:
        raise ValueError(
            f"slice_53d_to_19d expects (chunk_len, {ACTION_DIM}), got "
            f"shape={action_chunk_53d.shape}"
        )
    # lazy: env-isolated dependency は無いが、adapter code path の可読性のため
    # module-top で import しない (循環回避 + テストで monkey-patch しやすい)
    from model.subtask_policy_training.gr00t.dex1_hand_synergy import (
        hand_to_dex1,
    )

    chunk_len = action_chunk_53d.shape[0]
    out = np.zeros((chunk_len, _VLASKILL_ACTION_DIM), dtype=action_chunk_53d.dtype)
    out[:, 0:3]   = action_chunk_53d[:, _ACTION_53D_WAIST_SLICE]
    out[:, 3:10]  = action_chunk_53d[:, _ACTION_53D_LEFT_ARM_SLICE]
    out[:, 10:17] = action_chunk_53d[:, _ACTION_53D_RIGHT_ARM_SLICE]
    # Hand: 7D synergy → 1D Dex1 逆射影 (per-step、chunk 短いのでベクトル化不要)
    left_hand_7d  = action_chunk_53d[:, _ACTION_53D_LEFT_HAND_SLICE]
    right_hand_7d = action_chunk_53d[:, _ACTION_53D_RIGHT_HAND_SLICE]
    for t in range(chunk_len):
        out[t, 17] = hand_to_dex1(left_hand_7d[t].tolist(), side="left", kind="action")
        out[t, 18] = hand_to_dex1(right_hand_7d[t].tolist(), side="right", kind="action")
    return out


# ---- Preprocessing helpers (lazy import せず default env で testable) ---- #


def validate_gr00t_config(cfg: PolicyConfig) -> None:
    """`PolicyConfig` が GR00T の contract を満たすかを検証する。

    - cams が (HEAD_LEFT, WRIST_LEFT, WRIST_RIGHT) 3 cam layout であること
      (head_right は subtask_training.json:ignored_video_keys で drop)
    - mode が "none" or "overlay" (GR00T は explicit OBB token 非対応)
    - dtype が "bf16" 推奨 (subtask_training.json:groot.use_bf16=true)。
      fp32 も許容 (base config の smoke 用)、fp16 は非推奨で warning。
    """
    if tuple(cfg.cams) != CAMERAS:
        raise ValueError(
            f"GR00T requires cams={tuple(c.value for c in CAMERAS)!r} "
            f"(3 cam layout per subtask_training.json), got "
            f"{tuple(c.value for c in cfg.cams)!r}"
        )
    if cfg.mode == "precomputed_token":
        raise ValueError(
            "GR00T does not support mode='precomputed_token' (no OBB token channel). "
            "Use mode='overlay' to draw OBB rectangles onto frames, or mode='none'."
        )


def build_batch_dict(
    obs: Observation,
    embodiment_id: int = EMBODIMENT_ID,
    video_horizon: int = 1,
) -> dict:
    """Observation → GR00T pre-processor が受け取る raw dict。

    processor_groot.py:Gr00tPacker が期待する key を組む:
        - observation.images.<cam_name>: (H, W, 3) uint8 BGR (processor が RGB
          変換 + resize 224 する)
        - observation.state: (state_dim,) float32
        - task: str
        - embodiment_id: int

    Note: 本 helper は torch を要さず、後段 `predict()` で tensor 化する
    (default env の unit test は本 helper の shape/key を verify する)。

    Args:
        obs: Skill wrapper が assemble 済の Observation。state は既に (49,) に
            build_state_from_raw で組んである前提。
        embodiment_id: subtask_training.json:groot.embodiment_tag に対応する
            integer id (REAL_G1_RELATIVE_EEF_EMBODIMENT_ID = 25)。

    Returns:
        GR00T pre-processor 用の dict。tensor 化 (torch) は predict() 内部で
        遅延実行。
    """
    if obs.state.shape != (STATE_DIM,):
        raise ValueError(
            f"Observation.state must have shape ({STATE_DIM},) for GR00T, got "
            f"{obs.state.shape}"
        )
    if obs.language is None:
        raise ValueError(
            "GR00T requires obs.language (natural language prompt). Skill wrapper "
            "must set it (default: DEFAULT_LANGUAGE_PROMPT)."
        )

    batch: dict = {
        "observation.state": obs.state.astype(np.float32, copy=False),
        "task": obs.language,
        "embodiment_id": int(embodiment_id),
    }
    for cam in CAMERAS:
        if cam not in obs.frames_bgr:
            raise KeyError(
                f"Observation.frames_bgr missing required cam {cam.value!r} "
                f"(GR00T needs {tuple(c.value for c in CAMERAS)!r})"
            )
        frame = obs.frames_bgr[cam]
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame for {cam.value!r} must be (H, W, 3) uint8 BGR, got "
                f"shape={frame.shape} dtype={frame.dtype}"
            )
        if video_horizon == 1:
            batch[_CAM_TO_LEROBOT_KEY[cam]] = frame
        elif video_horizon == 2:
            if obs.frames_bgr_prev is None or cam not in obs.frames_bgr_prev:
                raise ValueError(
                    f"two-frame GR00T input is missing history for {cam.value}"
                )
            previous = obs.frames_bgr_prev[cam]
            if previous.shape != frame.shape or previous.dtype != np.uint8:
                raise ValueError(
                    f"history frame for {cam.value} does not match current frame"
                )
            batch[_CAM_TO_LEROBOT_KEY[cam]] = np.stack((previous, frame), axis=0)
        else:
            raise ValueError(f"unsupported GR00T video_horizon={video_horizon}")

    return batch


def _prepare_gr00t_frames(
    obs: Observation, mode: str
) -> tuple[dict[CameraKey, np.ndarray], int]:
    """Return model-ready frames, rendering the training-time OBB overlay.

    GR00T's overlay checkpoint was trained with rectangles on head-left only;
    wrist images remain raw.  ``None`` detections means that no detector is
    connected and is therefore a configuration error.  An empty list is a
    valid YOLO result and produces an unchanged head image.
    """

    frames = dict(obs.frames_bgr)
    if mode != "overlay":
        return frames, 0
    if obs.obb_detections is None:
        raise RuntimeError(
            "GR00T overlay mode requires live YOLO-OBB detections; refusing "
            "to evaluate the overlay checkpoint on unannotated images"
        )
    detections = list(obs.obb_detections.get(CameraKey.HEAD_LEFT, []))
    if CameraKey.HEAD_LEFT not in frames:
        raise KeyError("GR00T overlay mode requires a head_left frame")
    from inference.desktop.lower_policy.policies.ramen_ori import (
        overlay_obb_on_frame,
    )

    frames[CameraKey.HEAD_LEFT] = overlay_obb_on_frame(
        frames[CameraKey.HEAD_LEFT].copy(), detections
    )
    return frames, len(detections)


def _image_bgr_hwc_to_rgb_chw(frame: np.ndarray) -> np.ndarray:
    """Convert a live OpenCV frame to LeRobot's unbatched image contract.

    LeRobot's ``GrootN17PackInputsStep`` accepts image tensors as ``(C,H,W)``
    (the following batch step turns them into ``(B,C,H,W)``). Passing the live
    ``(H,W,C)`` frame through unchanged makes the processor interpret image
    width as channels; a 640x480 RGB frame then becomes a 480-channel image and
    the first Qwen preprocessing call can allocate tens of GiB.
    """

    if frame.dtype != np.uint8 or frame.ndim not in {3, 4} or frame.shape[-1] != 3:
        raise ValueError(
            "GR00T image conversion expects (H, W, 3) uint8 BGR, got "
            f"shape={frame.shape} dtype={frame.dtype}"
        )
    # OpenCV/DDS observations are BGR; LeRobot training videos are RGB.
    if frame.ndim == 3:
        return np.ascontiguousarray(frame[..., ::-1].transpose(2, 0, 1))
    # Temporal Furniture-GR00T input: (T,H,W,C) -> (T,C,H,W).
    return np.ascontiguousarray(frame[..., ::-1].transpose(0, 3, 1, 2))


def _euler_xyz_to_rot6d(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler XYZ (radians) → ROT6D (先頭 2 row of Rz@Ry@Rx rotation matrix、flatten)。

    **訓練時仕様の厳密追従** (`g1_full_body_mapping.py:_euler_xyz_matrix` +
    `source_euler_xyz_pose_to_xyz_rot6d` inline)。
    順序は scipy `Rotation.from_euler("xyz")` = Rz(yaw) @ Ry(pitch) @ Rx(roll)。

    Returns: (6,) float32 = [row0_x, row0_y, row0_z, row1_x, row1_y, row1_z]。
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    row0 = np.array(
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dtype=np.float32
    )
    row1 = np.array(
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dtype=np.float32
    )
    return np.concatenate([row0, row1])


def _wrist_pose_to_xyz_rot6d(pose6d: np.ndarray) -> np.ndarray:
    """(xyz + euler xyz) 6D pose → (xyz + rot6d) 9D。training と 1e-6 一致想定。

    Args:
        pose6d: (6,) float32 = [x, y, z, roll, pitch, yaw] (radians、root frame)。

    Returns:
        (9,) float32 = [x, y, z, row0(3), row1(3)]。
    """
    xyz = pose6d[:3].astype(np.float32, copy=False)
    rot6d = _euler_xyz_to_rot6d(
        float(pose6d[3]), float(pose6d[4]), float(pose6d[5])
    )
    return np.concatenate([xyz, rot6d])


def build_state_from_raw(raw: RawRobotState) -> np.ndarray:
    """Raw robot state → GR00T 49D REAL_G1_RELATIVE_EEF state。

    Layout (g1_full_body_mapping.py:REAL_G1_RELATIVE_EEF_STATE_SLICES):
        [0:9]    left_wrist_eef_9d  = ee_state[0:6] → XYZ(3) + ROT6D(6)
        [9:18]   right_wrist_eef_9d = ee_state[6:12] → XYZ + ROT6D
        [18:25]  left_hand  = official seven-joint synergy from Dex1 left
        [25:32]  right_hand = official seven-joint synergy from Dex1 right
        [32:39]  left_arm   = joint_positions[15:22]
        [39:46]  right_arm  = joint_positions[22:29]
        [46:49]  waist      = joint_positions[12:15]

    Phase A-3 (現在): 全 slice を実装。inference/ 単体 deploy 想定のため
    training module (g1_full_body_mapping) は import せず、euler→rot6d 変換も
    ここに inline (`_euler_xyz_to_rot6d` / `_wrist_pose_to_xyz_rot6d`)。

    Args:
        raw: orchestrator の RawRobotState (Orin 実データ layout)。ee_state は
            root frame の (xyz+euler)×2、Phase A-5 で pinocchio FK が埋める予定。

    Returns:
        (49,) float32、REAL_G1_RELATIVE_EEF layout の state。
    """
    if raw.joint_positions.shape != (29,):
        raise ValueError(
            f"joint_positions must be (29,), got {raw.joint_positions.shape}"
        )
    if raw.hand_state.shape != (2,):
        raise ValueError(f"hand_state must be (2,), got {raw.hand_state.shape}")
    if raw.ee_state.shape != (12,):
        raise ValueError(f"ee_state must be (12,), got {raw.ee_state.shape}")

    joint_positions = raw.joint_positions.astype(np.float32, copy=False)
    hand_state = raw.hand_state.astype(np.float32, copy=False)
    ee_state = raw.ee_state.astype(np.float32, copy=False)
    from model.subtask_policy_training.gr00t.dex1_hand_synergy import dex1_to_hand

    out = np.zeros(STATE_DIM, dtype=np.float32)
    # [0:9] left_wrist_eef_9d
    out[0:9] = _wrist_pose_to_xyz_rot6d(ee_state[0:6])
    # [9:18] right_wrist_eef_9d
    out[9:18] = _wrist_pose_to_xyz_rot6d(ee_state[6:12])
    # The training materializer expands the physical Dex1 coordinate into the
    # official seven-joint G1 hand synergy. Zero-padding here makes every real
    # observation out-of-distribution even though action decoding uses the
    # inverse synergy projection.
    out[18:25] = dex1_to_hand(float(hand_state[0]), side="left", kind="state")
    out[25:32] = dex1_to_hand(float(hand_state[1]), side="right", kind="state")
    # [32:39] left_arm (joint_positions[15:22])
    out[32:39] = joint_positions[G1_LEFT_ARM_SLICE]
    # [39:46] right_arm (joint_positions[22:29])
    out[39:46] = joint_positions[G1_RIGHT_ARM_SLICE]
    # [46:49] waist (joint_positions[12:15])
    out[46:49] = joint_positions[G1_WAIST_SLICE]
    return out


# ---- Gr00tPolicy (Skeleton、実 forward は integration test 経由) ---- #


class Gr00tPolicy:
    """GR00T inference loader (lerobot.policies.groot.GrootPolicy に委譲)。

    実 model load + forward は `lerobot` + `isaac-gr00t` 相当の重い deps を
    要するため、`from_ckpt` / `predict` は lazy import。default env / runtime
    env の unit test は preprocessing helper と config validation のみ verify、
    実 forward は `@pytest.mark.integration` marker で sakura env に飛ばす。

    Attributes:
        cfg: 生成時 config (immutable)。
        _lerobot_policy: lerobot.policies.groot.GrootPolicy instance (lazy load)。
        _pre_processor / _post_processor: processor_groot pipelines。
    """

    # Public constants (Skill wrapper が config assemble する時に参照)
    STATE_DIM = STATE_DIM
    ACTION_DIM = ACTION_DIM
    CHUNK_LEN = CHUNK_LEN
    # This adapter's own temporal ensembler/async pipeline consumes the native
    # chunk and deliberately returns one truthful transmitted target per tick.
    # VlaSkill must therefore not queue that one-row result as if it were the
    # original H16/H40 model chunk.
    EXECUTION_HORIZON = 1
    IMAGE_HW = IMAGE_HW
    CAMERAS = CAMERAS
    DEFAULT_LANGUAGE_PROMPT = DEFAULT_LANGUAGE_PROMPT
    EMBODIMENT_ID = EMBODIMENT_ID
    build_state_from_raw = staticmethod(build_state_from_raw)

    def __init__(
        self,
        cfg: PolicyConfig,
        _lerobot_policy=None,
        _pre_processor=None,
        _post_processor=None,
        _worker_client: _GrootWorkerClient | None = None,
        _artifact_contract: GrootArtifactContract | None = None,
    ) -> None:
        validate_gr00t_config(cfg)
        self.cfg = cfg
        self._lerobot_policy = _lerobot_policy
        self._pre_processor = _pre_processor
        self._post_processor = _post_processor
        self._worker_client = _worker_client
        self._artifact_contract = (
            getattr(_worker_client, "artifact_contract", _artifact_contract)
            if _worker_client is not None
            else _artifact_contract
        )
        self._video_horizon = (
            self._artifact_contract.video_horizon
            if self._artifact_contract is not None
            else 1
        )
        self._history_offset_ticks = (
            self._artifact_contract.history_offset_ticks
            if self._artifact_contract is not None
            else 1
        )
        self._frame_history: deque[dict[CameraKey, np.ndarray]] = deque(
            maxlen=self._history_offset_ticks + 1
        )
        self._prediction_count = 0
        self._last_sync_metadata: dict[str, object] = {}
        # Phase 2 (Issue #128): temporal ensemble state。
        # cfg.temporal_lambda=None なら ensembler は作るが lambda=None passthrough、
        # blend せず最新 chunk[0] 相当を返す (実装 simplicity)。
        # decay_lambda=-0.1 (default) で age に応じた exp decay blend。
        # 詳細は model.subtask_policy_training.gr00t.temporal_ensemble の docstring。
        from model.subtask_policy_training.gr00t.temporal_ensemble import (
            TargetTemporalEnsembler,
        )
        self._ensembler = TargetTemporalEnsembler(
            dim=_VLASKILL_ACTION_DIM,      # 19 (waist3 + arms14 + hand2)
            decay_lambda=cfg.temporal_lambda,
        )
        self._current_step: int = 0
        # Phase 4 (Issue #128): async_replanning pipeline。
        # cfg.replan_family=None なら pipeline=None、predict() は毎 tick sync 実行
        # (Phase 2 と同一動作、後方互換)。
        # cfg.replan_family 指定時は issue-70 FAMILY_REPLANNING_PROFILES を lookup、
        # 初回 sync predict で seed chunk を取得してから pipeline 生成。**pipeline
        # 生成は初回 predict() 呼出時に遅延化** (from_ckpt 直後は _lerobot_policy 未 load
        # の可能性、初回 predict まで待つ)。
        self._pipeline = None
        self._pipeline_lead_steps: int | None = None      # replan_after_steps 計算用
        self._pipeline_max_age_s: float | None = None
        self._pending_submit_step: int | None = None
        # 初期 chunk seed 用 obs cache (async submit の predictor に渡す)
        self._last_seen_obs: Observation | None = None

    def reset(self) -> None:
        """Skill 遷移 / episode 開始時に呼ぶ (VlaSkill._on_start から)。

        temporal ensemble の candidate buffer と step counter + async pipeline を
        初期化。pipeline が active なら bounded shutdown (0.5s timeout、pending
        prediction は promote 待ち)。
        """
        self._ensembler.reset()
        self._current_step = 0
        if self._pipeline is not None:
            # bounded close: pending inference が終わるまで最大 0.5s 待つ、以降は
            # daemon thread として process 終了と共に消える
            try:
                self._pipeline.close(timeout_s=0.5)
            except Exception:
                pass
            self._pipeline = None
        self._last_seen_obs = None
        self._pending_submit_step = None
        self._frame_history.clear()

    @classmethod
    def from_ckpt(cls, cfg: PolicyConfig) -> "Gr00tPolicy":
        """HF or local から fine-tuned GR00T ckpt を load。

        Lazy import: `lerobot.policies.groot.modeling_groot.GrootPolicy` +
        `processor_groot.make_groot_pre_post_processors`。

        - `cfg.ckpt_ref` が dir path なら local safetensors 検出
        - HF repo string なら hf_hub_download で safetensors 存在検証 → fine-tuned
          ckpt として load
        - Base model (nvidia/GR00T-N1.5-3B 等) を渡した場合は fresh ロード

        Args:
            cfg: PolicyConfig (mode / dtype / cams / ckpt_ref)。

        Returns:
            初期化済 Gr00tPolicy。model は cuda に配置、eval mode。
        """
        # The physical-G1 runtime is Python 3.10 for Unitree DDS, while LeRobot
        # 0.6.1 requires Python 3.12. Keep that ABI boundary explicit and use a
        # local worker rather than installing incompatible model packages into
        # the actuator process.
        if importlib.util.find_spec("lerobot") is None:
            return cls(cfg=cfg, _worker_client=_GrootWorkerClient(cfg))

        # Lazy import: default env / runtime env は lerobot 未 install。
        # 実 env (sakura .venv_lerobot060) で本 method が走る前提。
        # lazy: env-isolated dependencies (lerobot は sakura training env pin)
        import torch
        from lerobot.policies.groot.configuration_groot import (
            GrootConfig as _LrGrootConfig,
        )
        from lerobot.policies.groot.processor_groot import (
            make_groot_pre_post_processors_from_pretrained,
        )

        checkpoint_ref = str(cfg.ckpt_ref)
        checkpoint_root = _resolve_groot_checkpoint_root(
            checkpoint_ref, cfg.checkpoint_subdir
        )
        artifact_contract = _read_groot_artifact_contract(checkpoint_root)
        checkpoint_file = checkpoint_root / "model.safetensors"
        if not checkpoint_file.is_file():
            raise FileNotFoundError(
                f"GR00T model.safetensors not found: {checkpoint_file}"
            )

        gpu_free_bytes: int | None = None
        if str(cfg.device).startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("GR00T requested CUDA but torch.cuda is unavailable")
            torch.cuda.empty_cache()
            gpu_free_bytes = int(torch.cuda.mem_get_info()[0])
        checkpoint_bytes = checkpoint_file.stat().st_size
        host_available_bytes = _host_mem_available_bytes()
        # Streaming from safetensors needs only one tensor at a time in host
        # memory.  Preserve a conservative 4 GiB operating-system/runtime
        # reserve instead of requiring a second full checkpoint-sized CPU
        # model.  The GPU guard below remains unchanged.
        streaming_host_min = 4 * 1024**3
        if (
            host_available_bytes is not None
            and host_available_bytes < streaming_host_min
        ):
            raise MemoryError(
                "insufficient host memory for streaming GR00T load: "
                f"available={host_available_bytes / 1024**3:.1f}GiB "
                f"required>={streaming_host_min / 1024**3:.1f}GiB"
            )
        _require_groot_load_headroom(
            checkpoint_bytes=checkpoint_bytes,
            # The legacy CPU-staging host guard does not apply to the
            # one-tensor streaming loader; GPU headroom is still fail-closed.
            host_available_bytes=None,
            gpu_free_bytes=gpu_free_bytes,
        )

        # Keep the immutable checkpoint in its official FP32 master format, but honor
        # PolicyConfig.dtype for inference. The uploaded checkpoint declares
        # use_bf16=true while storing 11.7 GiB of FP32 master parameters; ignoring cfg.dtype
        # made both host and GPU memory peak high enough to freeze this 30/32-GiB PC.
        # Load on CPU first, then cast+transfer once so no full FP32 CUDA copy is created.
        policy_base_class = None
        if artifact_contract.policy_type == "furniture_groot":
            from inference.desktop.lower_policy.policies.furniture_groot_runtime import (
                FurnitureGrootRuntimeConfig,
                FurnitureGrootRuntimePolicy,
            )

            load_cfg = FurnitureGrootRuntimeConfig.from_pretrained(str(checkpoint_root))
            policy_base_class = FurnitureGrootRuntimePolicy
        else:
            load_cfg = _LrGrootConfig.from_pretrained(str(checkpoint_root))
        # Uploaded checkpoints may retain a training-workstation absolute
        # path. Resolve the pinned official base snapshot locally instead.
        base_revision = str(
            getattr(
                load_cfg,
                "base_model_revision",
                "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
            )
        )
        load_cfg.base_model_path = str(
            _resolve_groot_checkpoint_root(
                f"nvidia/GR00T-N1.7-3B@{base_revision}"
            )
        )
        load_cfg.device = "cpu"
        host_available_text = (
            f"{host_available_bytes / 1024**3:.1f}GiB"
            if host_available_bytes is not None
            else "unknown"
        )
        print(
            "[groot] memory-safe load: meta model -> tensor-streamed CUDA "
            f"(checkpoint={checkpoint_bytes / 1024**3:.1f}GiB, "
            f"host_available={host_available_text})",
            file=sys.stderr,
        )
        target_dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32
        policy = _load_groot_policy_streaming(
            load_cfg=load_cfg,
            checkpoint_file=checkpoint_file,
            device=cfg.device,
            target_dtype=target_dtype,
            policy_base_class=policy_base_class,
        )
        policy.config.device = cfg.device
        policy.config.model_params_fp32 = target_dtype == torch.float32
        policy.eval()
        rss_before, rss_after, malloc_trimmed = _release_host_load_memory()
        before_text = (
            f"{rss_before / 1024**2:.0f}MiB" if rss_before is not None else "unknown"
        )
        after_text = (
            f"{rss_after / 1024**2:.0f}MiB" if rss_after is not None else "unknown"
        )
        print(
            "[groot] released CPU staging memory "
            f"(rss={before_text}->{after_text}, malloc_trim={malloc_trimmed}, "
            f"runtime_dtype={target_dtype})",
            file=sys.stderr,
        )

        # A fine-tuned LeRobot checkpoint serializes the exact normalization,
        # relative-action horizon, and decode pipeline next to the model. Rebuilding
        # processors from config loses those horizon-preserving statistics and is
        # rejected by LeRobot 0.6.1. Always restore the checkpoint-owned pipelines.
        pre, post = make_groot_pre_post_processors_from_pretrained(
            policy.config,
            str(checkpoint_root),
        )

        return cls(
            cfg=cfg,
            _lerobot_policy=policy,
            _pre_processor=pre,
            _post_processor=post,
            _artifact_contract=artifact_contract,
        )

    def warmup(self, n_iter: int = 5) -> None:
        """cuDNN autotune + Eagle cache warm-up (30Hz loop 前に消化)。

        dummy Observation で `predict()` を n_iter 回叩く。initial call の JIT
        コスト (~500ms-2s) を loop 突入前に消す。
        """
        if self._worker_client is not None:
            self._worker_client.warmup(n_iter)
            return
        if self._lerobot_policy is None:
            raise RuntimeError(
                "Gr00tPolicy is not loaded. Call from_ckpt() first."
            )
        dummy_obs = self._make_dummy_observation()
        for _ in range(n_iter):
            self.predict(dummy_obs)

    def _sync_predict_chunk_19d(self, obs: Observation) -> tuple[np.ndarray, float, np.ndarray]:
        """1 tick observation を synchronous inference → 19D chunk + latency + raw 53D。

        Phase 1/2/4 の共通 sync inference path。
        - Phase 2 (replan_family=None): 毎 tick 呼ばれる (predict() 内)
        - Phase 4 (replan_family 設定時): 初回 seed / async worker predictor callback /
          candidate 欠損時 fallback で呼ばれる

        Returns:
            (chunk_19d, latency_ms, raw_action_chunk_53d):
                chunk_19d = slice_53d_to_19d 通過後、(16, 19) float32
                latency_ms = inference 経過時間 [ms]
                raw_action_chunk_53d = post_processor 直後、(16, 53) — debug 用
        """
        if self._worker_client is not None:
            action = self._worker_client.predict(obs)
            raw_shape = tuple(
                int(value)
                for value in action.metadata.get(
                    "raw_action_shape_53d", (action.action_chunk.shape[0], ACTION_DIM)
                )
            )
            if len(raw_shape) != 2 or raw_shape[1] != ACTION_DIM:
                raise RuntimeError(
                    f"GR00T worker returned an invalid raw action shape: {raw_shape}"
                )
            self._last_sync_metadata = dict(action.metadata)
            self._prediction_count += 1
            # The parent only needs the raw shape for diagnostics; avoid
            # copying the discarded 53-D tensor across the process boundary.
            raw_shape_proxy = np.empty(raw_shape, dtype=np.float32)
            return action.action_chunk, action.latency_ms, raw_shape_proxy
        if self._lerobot_policy is None:
            raise RuntimeError(
                "Gr00tPolicy is not loaded. Call from_ckpt() first."
            )
        # lazy: env-isolated dependencies (torch は runtime env 以上でのみ available)
        import torch

        t0 = time.monotonic_ns()
        first_prediction = self._prediction_count == 0
        if first_prediction:
            print(
                f"[groot] first prediction start ({_runtime_memory_snapshot()})",
                file=sys.stderr,
            )
        frames, overlay_detection_count = _prepare_gr00t_frames(obs, self.cfg.mode)
        effective_obs = Observation(
            frames_bgr=frames,
            frames_bgr_prev=obs.frames_bgr_prev,
            state=obs.state,
            skill_id=obs.skill_id,
            language=obs.language,
            obb_detections=obs.obb_detections,
            timestamp_ns=obs.timestamp_ns,
        )
        raw_batch = build_batch_dict(
            effective_obs, video_horizon=self._video_horizon
        )

        # numpy → torch tensor、batch 次元追加 (B=1)
        torch_batch: dict = {}
        for key, val in raw_batch.items():
            if isinstance(val, np.ndarray):
                if key == "observation.state":
                    # The serialized pipeline contains AddBatchDimensionProcessorStep.
                    # Keep this unbatched so it becomes (1, D) exactly once.
                    torch_batch[key] = torch.from_numpy(val)  # (D,)
                else:
                    # Image step expects unbatched RGB (C, H, W); its own batch
                    # processor adds B=1. Do not pass BHWC here.
                    rgb_chw = _image_bgr_hwc_to_rgb_chw(val)
                    torch_batch[key] = torch.from_numpy(rgb_chw)  # (C, H, W)
            elif key == "task":
                # AddBatchDimensionProcessorStep wraps this string in list[str].
                torch_batch[key] = val
            elif key == "embodiment_id":
                torch_batch[key] = torch.tensor(val, dtype=torch.long)
            else:
                torch_batch[key] = val

        # Pre-process: normalize + Eagle encode → eagle_* tensors 追加
        processed = self._pre_processor(torch_batch)
        if first_prediction:
            print(
                f"[groot] first preprocessor complete ({_runtime_memory_snapshot()})",
                file=sys.stderr,
            )

        # Predict action chunk: (1, chunk_len, max_action_dim=32) が返る
        action_chunk_padded = self._lerobot_policy.predict_action_chunk(processed)
        if first_prediction:
            print(
                f"[groot] first model forward complete ({_runtime_memory_snapshot()})",
                file=sys.stderr,
            )

        # Post-process (denormalize) + max_action_dim (=32 default) → real 53 で slice
        # NOTE: max_action_dim < ACTION_DIM の場合、GrootPolicy 内で自動 slice される
        # (`actions = actions[:, :, :original_action_dim]`)
        # NOTE (RELATIVE rep): post_processor は action_config.rep="RELATIVE" の
        # left_arm / right_arm に対し state[state_key] を加算して absolute 化する
        # 仕様 (Isaac-GR00T processor pipeline)。詳細は module 冒頭の adapter comment。
        # LeRobot 0.6 PolicyProcessorPipeline's postprocessor input/output is a
        # PolicyAction, which is an alias of torch.Tensor. Passing the older
        # ``{"action": tensor}`` wrapper is rejected by policy_action_to_transition.
        denorm = self._post_processor(action_chunk_padded)
        action_chunk_53d = denorm[0].detach().cpu().numpy()  # (chunk_len, 53)

        # 53D REAL_G1_RELATIVE_EEF → 19D VlaSkill contract。
        # 実機 arm-only 制御では EEF / hand padding / base の 34 dim は使わないため
        # ここで drop する (詳細な mapping / 前提は slice_53d_to_19d docstring)。
        chunk_19d = slice_53d_to_19d(action_chunk_53d).astype(
            np.float32, copy=False
        )
        latency_ms = (time.monotonic_ns() - t0) / 1e6
        self._last_sync_metadata = {
            "overlay_detection_count": int(overlay_detection_count),
        }
        self._prediction_count += 1
        return chunk_19d, latency_ms, action_chunk_53d

    def predict(self, obs: Observation) -> PolicyAction:
        """1 tick observation → GR00T action chunk。

        Modes:
            - **cfg.replan_family=None**: Phase 2 sync per-tick predict → ensembler
              blend → 19D. 命令的シンプル、実機で inference latency 100ms が
              command loop cadence を遅らせる。
            - **cfg.replan_family=<family>**: Phase 4 async pipeline replan。
              初回 sync seed → chunk pipeline 生成 → 以降 tick で pipeline.
              wants_prediction が真なら async submit、完了した chunk があれば
              ensembler に統合。command loop は毎 tick non-blocking、
              cadence 維持。詳細は module docstring の Phase 4 説明。
        """
        obs = self._with_model_history(obs)
        # Phase 4: async pipeline 使用時は pipeline 経由、そうでなければ従来 sync
        chunk_source: str  # "sync" / "async_promoted" / "async_none_this_tick"
        latency_ms: float = 0.0
        raw_action_shape_53d: tuple | None = None
        pipeline_index: int | None = None

        if self.cfg.replan_family is None:
            # -------- Phase 2 legacy: per-tick sync predict --------
            chunk_19d, latency_ms, raw_53d = self._sync_predict_chunk_19d(obs)
            self._ensembler.add_chunk(
                origin_step=self._current_step, absolute_targets=chunk_19d
            )
            raw_action_shape_53d = tuple(raw_53d.shape)
            chunk_source = "sync"
        else:
            # -------- Phase 4: async pipeline replan --------
            # 初回 predict: sync seed chunk を取得して pipeline を build
            if self._pipeline is None:
                seed_19d, latency_ms, raw_53d = self._sync_predict_chunk_19d(obs)
                self._ensembler.add_chunk(
                    origin_step=self._current_step, absolute_targets=seed_19d
                )
                raw_action_shape_53d = tuple(raw_53d.shape)
                self._init_pipeline(seed_19d)
                chunk_source = "sync"  # seed = sync
            else:
                # 完了した async chunk があれば ensembler に追加
                promoted = self._pipeline.promote_if_ready()
                if promoted is not None and self._pending_submit_step is not None:
                    self._ensembler.add_chunk(
                        origin_step=self._pending_submit_step,
                        absolute_targets=promoted.actions.astype(np.float32),
                    )
                    self._pending_submit_step = None
                    chunk_source = "async_promoted"
                else:
                    chunk_source = "async_none_this_tick"
                # Pipeline の index を進める (wants_prediction 計算に必要)
                _pipeline_action, pipeline_index = self._pipeline.next_action()
                # replan タイミングに来たら async submit
                if (
                    self._pipeline.wants_prediction
                    and self._pending_submit_step is None
                ):
                    obs_snapshot = obs  # capture at submit time
                    submit_step = self._current_step

                    def predictor():
                        chunk_19d, ms, _raw = self._sync_predict_chunk_19d(obs_snapshot)
                        return (
                            chunk_19d.astype(np.float64),
                            float(ms),
                            {"submit_step": submit_step},
                        )

                    self._pipeline.submit(
                        predictor, anchor_generation=(submit_step,)
                    )
                    self._pending_submit_step = submit_step

                # candidate が current step に無い場合 = pipeline stall (inference 完了
                # 遅れ + seed chunk が使い切られた) → sync fallback で hard stall 回避
                if self._ensembler.candidate_count(self._current_step) == 0:
                    chunk_19d, fallback_ms, raw_53d = self._sync_predict_chunk_19d(obs)
                    self._ensembler.add_chunk(
                        origin_step=self._current_step,
                        absolute_targets=chunk_19d,
                    )
                    raw_action_shape_53d = tuple(raw_53d.shape)
                    latency_ms = fallback_ms
                    chunk_source = "sync_fallback"

        # 現 step の blended target 取得
        blended_target = self._ensembler.target(step=self._current_step)  # (19,)
        blended_chunk = blended_target[None, :].astype(np.float32, copy=False)
        candidate_count = self._ensembler.candidate_count(self._current_step)
        # 事後: 次 tick に向けて step 進める
        self._current_step += 1

        return PolicyAction(
            action_chunk=blended_chunk,
            latency_ms=latency_ms,
            metadata={
                "mode": self.cfg.mode,
                "chunk_len": blended_chunk.shape[0],
                "action_dim": blended_chunk.shape[1],
                # 【Phase 2】ensembler の diagnostics
                "blended_from_n_candidates": int(candidate_count),
                "temporal_lambda": self.cfg.temporal_lambda,
                # 【Phase 4】async pipeline の diagnostics
                "replan_family": self.cfg.replan_family,
                "chunk_source": chunk_source,  # sync / async_promoted / async_none_this_tick / sync_fallback
                "pipeline_index": pipeline_index,
                "pending_submit_step": self._pending_submit_step,
                # 【Phase 1】raw 53D shape (この tick で sync predict 発火時のみ埋まる)
                "raw_action_shape_53d": raw_action_shape_53d,
                # Debug 用: 実 dispatch した 19D の min/max
                "action_min_19d": float(blended_target.min()),
                "action_max_19d": float(blended_target.max()),
                "action_arms_absmax_19d": float(
                    np.abs(blended_target[3:17]).max()
                ),
                "overlay_detection_count": int(
                    self._last_sync_metadata.get("overlay_detection_count", 0)
                ),
            },
        )

    def _with_model_history(self, obs: Observation) -> Observation:
        """Attach the checkpoint-requested image history without changing callers.

        Furniture-GR00T was trained with frames ``[-20, 0]``.  The generic
        VLA wrapper only exposes the immediately previous frame, so keeping the
        model-specific ring here prevents RAMEN-Ori's ``t-1`` semantics from
        being changed globally.  At an episode boundary, the earliest live
        frame is repeated until 20 ticks exist, matching index clamping during
        dataset sampling.
        """

        if self._video_horizon == 1:
            return obs
        snapshot = {
            cam: np.asarray(obs.frames_bgr[cam]).copy() for cam in CAMERAS
        }
        self._frame_history.append(snapshot)
        previous = self._frame_history[0]
        return Observation(
            frames_bgr=snapshot,
            frames_bgr_prev=previous,
            state=obs.state,
            skill_id=obs.skill_id,
            language=obs.language,
            obb_detections=obs.obb_detections,
            timestamp_ns=obs.timestamp_ns,
        )

    def _init_pipeline(self, seed_chunk_19d: np.ndarray) -> None:
        """初回 sync seed chunk から AsyncActionChunkPipeline を build。

        cfg.replan_family + cfg.execution_steps を issue-70 FAMILY_REPLANNING_PROFILES
        で lookup、replan_after_steps + max_prediction_age_s を決定。
        """
        from inference.desktop.lower_policy.async_replanning import (
            AsyncActionChunkPipeline,
            family_replanning_schedule,
        )
        replan_after, max_age = family_replanning_schedule(
            self.cfg.replan_family, self.cfg.execution_steps
        )
        self._pipeline_lead_steps = self.cfg.execution_steps - replan_after
        self._pipeline_max_age_s = max_age
        self._pipeline = AsyncActionChunkPipeline(
            initial_actions=seed_chunk_19d.astype(np.float64),
            execution_steps=self.cfg.execution_steps,
            replan_after_steps=replan_after,
            max_prediction_age_s=max_age,
            thread_name_prefix="gr00t-replan",
        )
        # submit 中の origin_step tracking (promote 時に ensembler.add_chunk へ渡す)
        self._pending_submit_step: int | None = None

    def close(self) -> None:
        """GPU memory 解放 (long-running orchestrator の graceful shutdown)。"""
        if self._pipeline is not None:
            abort_pending = (
                self._worker_client.abort
                if self._worker_client is not None
                else None
            )
            self._pipeline.close(timeout_s=0.5, abort_pending=abort_pending)
            self._pipeline = None
        if self._worker_client is not None:
            self._worker_client.close()
            self._worker_client = None
            return
        # lazy: env-isolated dependencies
        try:
            import torch
        except ImportError:
            self._lerobot_policy = None
            return
        self._lerobot_policy = None
        self._pre_processor = None
        self._post_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- private helpers ---- #

    def _make_dummy_observation(self) -> Observation:
        """warmup 用 dummy observation (unit test の Observation._make と別)。"""
        H, W = 480, 640
        state = np.zeros(STATE_DIM, dtype=np.float32)
        # The GR00T state contract stores each wrist orientation as ROT6D.
        # Six zeros are not a rotation and make the official relative-action
        # postprocessor divide by zero.  Use identity for both dummy wrists so
        # warmup exercises the real processor without producing NaNs/warnings.
        identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        state[3:9] = identity_rot6d
        state[12:18] = identity_rot6d
        return Observation(
            frames_bgr={cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in CAMERAS},
            frames_bgr_prev=(
                {cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in CAMERAS}
                if self._video_horizon == 2
                else None
            ),
            state=state,
            skill_id=None,
            language=DEFAULT_LANGUAGE_PROMPT,
            obb_detections=(
                {CameraKey.HEAD_LEFT: []} if self.cfg.mode == "overlay" else None
            ),
            timestamp_ns=time.monotonic_ns(),
        )
