"""RAMEN-Ori policy loader (Issue #125 Phase 3、C-axis inference integration)。

# 訓練時仕様の厳密追従

Batch key / dim / shape の唯一の真実:
- `model/ramen_ori/model.py` 冒頭 docstring (batch format contract)
- `model/ramen_ori/data_lerobot.py` (real dataset loader、image transforms)
- `model/ramen_ori/state_derive.py` (`derive_state_71d` を直接再利用)
- `model/ramen_ori/skill_mapping.py` (skill_id canonical map)
- `model/ramen_ori/configs/base.yaml` (num_cams / chunk_len / action_dim / etc.)

# RAMEN-Ori 契約

- **State dim**: 71 (E-δ default、[joint19 + tracking_err19 + velocity19 +
   hand_state2 + ee_pose12])
- **Action dim**: 19 (waist3 + arm7 + arm7 + hand2)
- **Chunk len**: 16 (Flow Matching action chunk)
- **Image size**: (224, 224)、resize with antialias=True
- **N_cams**: 4 (HEAD_LEFT, HEAD_RIGHT, WRIST_LEFT, WRIST_RIGHT)
- **Cam vocabulary**: num_cams=6 (model side embedding、data 側 N_cams=4 のみ埋まる)
- **Normalize**: ImageNet mean/std (LingBot / DINOv2 標準)
- **OBB**: 7 classes × 4 top_K per cam (mode=none では valid_mask 全 False で埋める)
- **Precision**: fp32 (base.yaml、Phase 2 で bf16 検討中)
- **Skill_id**: 6 canonical (skill_mapping.py):
    0=insert_table_leg / 1=flip_table / 2=rotate_leg_to_tighten /
    3=pick_table_leg / 4=rotate_table_base / 5=move_table_base

# Phase 3 (本 module) scope = mode="none" のみ

- OBB signal は zeros で埋め (valid_mask 全 False)、model は precomputed_token
  channel を受け取るが実質 skip する形。
- overlay mode (Phase 5) と precomputed_token mode (Phase 6) は本 module に
  add-on、build_batch_dict の分岐で対応する。
- 実 ckpt load + forward smoke は本 module の integration test で行う (Sakura で
  train した Run 1 default ckpt = `Team-RAMEN/..._ramen_ori_default_100k_v1`、
  ~200-300M params、local 8GB VRAM 十分)。

# ImageNet normalize (data_lerobot.py:IMAGENET_MEAN/STD と一致)
"""

from __future__ import annotations

import time

import numpy as np

from inference.desktop.lower_policy.policies.base import (
    CameraKey,
    G1_UPPER_BODY_JOINT_DIM,
    G1_UPPER_BODY_JOINT_INDICES,
    Observation,
    PolicyAction,
    PolicyConfig,
    RawRobotState,
)


# UPPER_BODY joint (17 dim = 3 waist + 7 left_arm + 7 right_arm) + hand (2 dim) = 19 dim
# = RAMEN-Ori state [0:19] joint slice の source
_UPPER_INDEX_NP = np.asarray(G1_UPPER_BODY_JOINT_INDICES, dtype=np.int64)


# ---- RAMEN-Ori 契約定数 (base.yaml + model.py 唯一の真実) ---- #

STATE_DIM: int = 71     # E-δ (state_derive.py:RAMEN_ORI_STATE_DIM)
ACTION_DIM: int = 19    # base.yaml:model.action_dim
CHUNK_LEN: int = 16     # base.yaml:model.chunk_len
IMAGE_HW: tuple[int, int] = (224, 224)  # data_lerobot.py:make_image_transform default
NUM_CAMS_VOCAB: int = 6  # base.yaml:model.num_cams (global embedding vocab)
NUM_CLASSES: int = 7    # base.yaml:model.obb.num_classes
TOP_K: int = 4          # data_lerobot.py default top_K
NUM_SKILLS: int = 6     # skill_mapping.NUM_SKILLS

# ---- OBB overlay palette (C-11、training OverlayRenderer と一致) ---- #

# BGR (cv2 convention)、model/ramen_ori/overlay.py:CLASS_COLORS_BGR と同値。
# training 側と 1 pixel drift しないよう独立に inline (inference/ 単体 deploy 想定)。
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (128, 128, 128),  # workspace  = gray
    1: (0, 255, 0),      # leg        = green
    2: (0, 255, 255),    # leg_tip    = yellow
    3: (0, 0, 255),      # hole       = red
    4: (255, 0, 0),      # table_top  = blue
    5: (0, 128, 255),    # hand_right = orange
    6: (255, 0, 255),    # hand_left  = magenta
}

# training OverlayRenderer default (Issue #122 D-3 preview 決定値)
DEFAULT_OVERLAY_CONF_THRESHOLD: float = 0.30
DEFAULT_OVERLAY_LINE_THICKNESS: int = 2

# training が overlay を掛ける cam (subtask_training.json の precompute cache
# = head_left/right の 2 cam のみ、wrist は cache 無し = overlay skip)
OVERLAY_TARGET_CAMS: tuple[CameraKey, ...] = (
    CameraKey.HEAD_LEFT,
    CameraKey.HEAD_RIGHT,
)

# RAMEN-Ori 4 cam layout (data_lerobot.py:default_camera_keys 順)
CAMERAS: tuple[CameraKey, ...] = (
    CameraKey.HEAD_LEFT,
    CameraKey.HEAD_RIGHT,
    CameraKey.WRIST_LEFT,
    CameraKey.WRIST_RIGHT,
)

# ImageNet normalize (LingBot / DINOv2 系標準、data_lerobot.py:IMAGENET_MEAN/STD)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# skill_mapping.py canonical (subtask_training.json とは別、RAMEN-Ori 独自)
SKILL_MOVE_TABLE_BASE: int = 5
SKILL_ROTATE_TABLE_BASE: int = 4


# ---- state_dict load 診断 helper (Phase B follow-up、Finding 1) ---- #


def _log_state_dict_load(
    label: str,
    missing: list[str],
    unexpected: list[str],
    ckpt_path: str,
) -> None:
    """`model.load_state_dict(..., strict=False)` の返り値を stderr に log。

    strict=False は checkpoint schema drift (training/inference の model 定義乖離)
    で missing/unexpected keys を silent に無視するため、明示的な log が必要
    (PR #126 Finding 1)。ゼロ件なら 1 行に抑える。
    """
    import sys as _sys

    if not missing and not unexpected:
        print(
            f"[RamenOriPolicy] loaded {label} from {ckpt_path!r} (all keys matched)",
            file=_sys.stderr,
        )
        return
    print(
        f"[RamenOriPolicy] loaded {label} from {ckpt_path!r} with schema drift: "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        file=_sys.stderr,
    )
    if missing:
        print(f"  missing keys (first 10): {missing[:10]}", file=_sys.stderr)
    if unexpected:
        print(f"  unexpected keys (first 10): {unexpected[:10]}", file=_sys.stderr)


def _validate_architecture_state_dict_load(
    missing: list[str],
    unexpected: list[str],
    ckpt_path: str,
) -> None:
    """Reject architecture drift while allowing the separately loaded backbone.

    Training checkpoints intentionally omit the frozen vision backbone, which is
    restored by ``load_vision_backbone``. Any other missing key, or any unexpected
    key, means the inference model was instantiated with a different architecture.
    Continuing in that state silently evaluates random modules instead of the named
    variant, so physical evaluation must fail closed.
    """
    disallowed_missing = [
        key for key in missing if not key.startswith("vision.backbone.")
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "RAMEN-Ori checkpoint architecture mismatch: "
            f"missing_non_backbone={len(disallowed_missing)} "
            f"unexpected={len(unexpected)} checkpoint={ckpt_path!r}. "
            "Refusing to evaluate a partially loaded model."
        )


# ---- Preprocessing helpers (default env で testable、torch 不要) ---- #


def validate_ramen_ori_config(cfg: PolicyConfig) -> None:
    """`PolicyConfig` が RAMEN-Ori の contract を満たすかを検証する。

    - cams が 4 cam layout (HEAD_LEFT, HEAD_RIGHT, WRIST_LEFT, WRIST_RIGHT)
      であること (data_lerobot.py:default_camera_keys 順、cam_id 割当と一致)
    - mode が "none" / "precomputed_token" / "overlay" のいずれか (base.py の
      PolicyConfig.__post_init__ でも validate されるが RAMEN-Ori は 3 mode 全対応)
    - dtype fp32 or bf16 (fp16 は非推奨)
    """
    if tuple(cfg.cams) != CAMERAS:
        raise ValueError(
            f"RAMEN-Ori requires cams={tuple(c.value for c in CAMERAS)!r} "
            f"(4 cam layout per data_lerobot.py:default_camera_keys), got "
            f"{tuple(c.value for c in cfg.cams)!r}"
        )


def build_state_from_raw(raw: RawRobotState) -> np.ndarray:
    """Raw robot state → RAMEN-Ori 71D E-δ state。

    E-δ 71D layout (training-side state_derive.py:derive_state_71d docstring):
        [0:19]   joint       q_current[UPPER_BODY]  (waist3+arm7+arm7+hand2)
        [19:38]  tracking_err (last_action_19d - q_current[UPPER_BODY])
        [38:57]  velocity    (q_current - q_prev)[UPPER_BODY]
        [57:59]  hand_state  raw.hand_state (joint 末尾 2 と重複、explicit token)
        [59:71]  ee_pose     raw.ee_state (left+right xyz+euler xyz、root frame)

    Phase A-2 (現在): joint / hand_state / ee_pose slice を実装。
    Phase A-4 で tracking_err / velocity slice を追加予定 (現状 zeros)。

    Args:
        raw: orchestrator の RawRobotState (Orin 実データ layout)。
             last_action_19d / joint_positions_prev は Phase A-4 まで未使用。

    Returns:
        (71,) float32、E-δ layout の state。
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

    # [0:19] joint = q_current[UPPER_BODY 17 dim] + hand_state 2 dim
    upper_current = joint_positions[_UPPER_INDEX_NP]  # (17,)
    joint_slice = np.concatenate([upper_current, hand_state])  # (19,)
    assert joint_slice.shape == (G1_UPPER_BODY_JOINT_DIM + 2,)

    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[0:19] = joint_slice

    # [19:38] tracking_err = last_action_19d - joint_slice (19D)
    # 先頭 tick (last_action_19d=None) は zeros のまま。
    if raw.last_action_19d is not None:
        if raw.last_action_19d.shape != (19,):
            raise ValueError(
                f"last_action_19d must be (19,), got {raw.last_action_19d.shape}"
            )
        out[19:38] = raw.last_action_19d.astype(np.float32, copy=False) - joint_slice

    # [38:57] velocity = joint_slice - prev_joint_slice (19D)
    # 学習側と同じく、腕・腰とDex1を同じ前tick実測値から差分化する。
    if raw.joint_positions_prev is not None:
        if raw.joint_positions_prev.shape != (29,):
            raise ValueError(
                f"joint_positions_prev must be (29,), got "
                f"{raw.joint_positions_prev.shape}"
            )
        prev_upper = raw.joint_positions_prev[_UPPER_INDEX_NP].astype(
            np.float32, copy=False
        )
        if raw.hand_state_prev is None:
            prev_hand = hand_state
        else:
            if raw.hand_state_prev.shape != (2,):
                raise ValueError(
                    f"hand_state_prev must be (2,), got {raw.hand_state_prev.shape}"
                )
            prev_hand = raw.hand_state_prev.astype(np.float32, copy=False)
        prev_joint_slice = np.concatenate([prev_upper, prev_hand])
        out[38:57] = joint_slice - prev_joint_slice

    out[57:59] = hand_state
    out[59:71] = ee_state
    return out


def pack_obb_tokens(
    obb_detections,  # dict[CameraKey, list[OBBDetection]]
    cams: tuple[CameraKey, ...] = CAMERAS,
    top_k: int = TOP_K,
    num_classes: int = NUM_CLASSES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """C-2 precomputed_token: per-cam OBB list → RAMEN-Ori model 用 5-tuple tensor。

    **訓練時仕様の厳密追従** (Issue #125 Phase 6):
    model.py:_encode の `obb` 呼出しは
    `self.obb(obb_verts, obb_conf, obb_class_id, obb_cam_id, obb_valid_mask)`。
    obb.py:ObbTokenizer が期待する shape / dtype:
        - obb_verts:      (N_cams, top_K, 8) float32  [0,1] normalized xyxyxyxy
        - obb_conf:       (N_cams, top_K, 1) float32  [0,1]
        - obb_class_id:   (N_cams, top_K)    int64    [0..num_classes-1]
        - obb_valid_mask: (N_cams, top_K)    bool     True = 実 det、False = padding

    training 側は precompute cache から fill、inference 側は per-cam YOLO 結果を
    conf desc で sort → top_K 個を取る (data_lerobot.py:_obb_from_cache と同挙動)。
    padding slot の class_id は 0 で埋める (embedding lookup が valid index を要
    求、valid_mask で downstream attention が無視する前提)。

    Args:
        obb_detections: cam_key → list[OBBDetection]。cam key が dict に無い場合
            (YOLO 未走行 cam = wrist 等) は該当 cam slot 全 invalid。
        cams: model expected cam order (N_cams,)。dict から取り出す順序を規定。
        top_k: 各 cam の最大 det 数 (data_lerobot.py:top_K=4)。
        num_classes: class_id validation 上限 (num_classes 以上は skip)。

    Returns:
        (verts, conf, class_id, valid_mask) tuple、shapes 上記の通り。
        obb_cam_id は caller 側で `np.broadcast_to(arange(N)[:, None], (N, top_k))`
        で自動生成 (data_lerobot.py と同じ pattern)。
    """
    N = len(cams)
    verts = np.zeros((N, top_k, 8), dtype=np.float32)
    conf = np.zeros((N, top_k, 1), dtype=np.float32)
    class_id = np.zeros((N, top_k), dtype=np.int64)
    valid_mask = np.zeros((N, top_k), dtype=bool)

    for cam_idx, cam in enumerate(cams):
        dets = obb_detections.get(cam)
        if not dets:
            continue
        # conf desc で sort、num_classes 上限を超える class_id は skip
        filtered = [d for d in dets if 0 <= int(d.class_id) < num_classes]
        filtered.sort(key=lambda d: float(d.confidence), reverse=True)
        for slot_idx, det in enumerate(filtered[:top_k]):
            # verts (4, 2) normalized → flat 8D xyxyxyxy
            v = det.verts.reshape(-1).astype(np.float32, copy=False)
            if v.shape != (8,):
                raise ValueError(
                    f"OBBDetection.verts must be (4, 2) or (8,), got {det.verts.shape}"
                )
            verts[cam_idx, slot_idx] = v
            conf[cam_idx, slot_idx, 0] = float(det.confidence)
            class_id[cam_idx, slot_idx] = int(det.class_id)
            valid_mask[cam_idx, slot_idx] = True
    return verts, conf, class_id, valid_mask


def overlay_obb_on_frame(
    frame_bgr: np.ndarray,
    detections,  # list[OBBDetection] — lazy typing for default env
    conf_threshold: float = DEFAULT_OVERLAY_CONF_THRESHOLD,
    line_thickness: int = DEFAULT_OVERLAY_LINE_THICKNESS,
    class_filter: set[int] | None = None,
    class_colors_bgr: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """C-11 overlay: BGR uint8 frame に OBB rectangle を色分け描画 (in-place)。

    **訓練時仕様の厳密追従** (Issue #125 Phase 5):
    `model/ramen_ori/overlay.py:OverlayRenderer.__call__` の cv2.polylines 呼出と
    厳密一致。差異:
        - training: `ObbPrecomputedCache.lookup_mp4_frame` から det 取得
        - inference: `list[OBBDetection]` (YoloObbPerception の per-frame 結果)
          を直接受け取る (cache 経由不要)
    描画 logic (color / thickness / conf filter / class filter / polylines args)
    は完全に同じ。

    Args:
        frame_bgr: (H, W, 3) uint8 BGR image (cv2 native)。**in-place で書き換わる**。
        detections: `list[OBBDetection]` (yolo_obb.OBBDetection)、`.verts` は
            (4, 2) normalized [0, 1]。空 list なら no-op。
        conf_threshold: この conf 未満の det は描画しない。default 0.30 は training
            preview 決定値 (Issue #122 D-3)。
        line_thickness: cv2.polylines の thickness (default 2 = training default)。
        class_filter: None なら全 class 描画、set なら該当 class_id のみ描画。
        class_colors_bgr: None なら CLASS_COLORS_BGR (module default) を使う。

    Returns:
        frame_bgr そのもの (in-place 描画済)。呼出側は返り値を使う。
    """
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"frame_bgr must be (H, W, 3) uint8 BGR, got shape={frame_bgr.shape} "
            f"dtype={frame_bgr.dtype}"
        )
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError(f"conf_threshold must be in [0, 1], got {conf_threshold}")
    if line_thickness < 1:
        raise ValueError(f"line_thickness must be >= 1, got {line_thickness}")

    if not detections:
        return frame_bgr

    palette = dict(class_colors_bgr or CLASS_COLORS_BGR)

    # lazy: env-isolated dependencies (opencv は runtime env or model env のみ)
    import cv2

    H, W = frame_bgr.shape[:2]
    for det in detections:
        if float(det.confidence) < conf_threshold:
            continue
        cid = int(det.class_id)
        if class_filter is not None and cid not in class_filter:
            continue
        color = palette.get(cid, (255, 255, 255))
        # verts (4, 2) normalized [0,1] → pixel (int32) → cv2.polylines format
        v = det.verts.reshape(4, 2) * np.array([W, H], dtype=np.float32)
        pts = v.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            frame_bgr,
            [pts],
            isClosed=True,
            color=color,
            thickness=line_thickness,
        )
    return frame_bgr


def split_head_stereo(head_packed_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """480x1280 packed head stereo → (head_left 480x640, head_right 480x640)。

    G1_WBT / HIW-500 の head camera は左右 packed stereo (640 + 640 = 1280 幅)。
    training-side data pipeline が cam_0 (head_left) と cam_1 (head_right) に split
    する。inference 側は orchestrator が split して VlaSkill に渡す責務 (この
    helper で行う)。

    Args:
        head_packed_bgr: (480, 1280, 3) uint8 BGR。

    Returns:
        (head_left, head_right) tuple of (480, 640, 3) uint8 BGR。
    """
    if head_packed_bgr.shape[:2] != (480, 1280):
        raise ValueError(
            f"head_packed_bgr must be (480, 1280, 3), got shape={head_packed_bgr.shape}"
        )
    return (
        np.ascontiguousarray(head_packed_bgr[:, :640, :]),
        np.ascontiguousarray(head_packed_bgr[:, 640:, :]),
    )


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (H, W, 3) → normalized (3, 224, 224) float32。

    **訓練時仕様の厳密追従** (Phase 4 で pixel diff < 1e-5 検証):
    training-side data_lerobot.py の pipeline を **torchvision** で再現:
      1. BGR → RGB (cv2 BGR native → training LeRobot は RGB return)
      2. HWC uint8 → CHW float32 [0, 1]  (LeRobot Dataset return format と一致)
      3. `torchvision.transforms.functional.resize([224, 224], antialias=True)`
         (data_lerobot.py:make_image_transform と同 antialias 挙動)
      4. `torchvision.transforms.functional.normalize(mean, std)`
         (data_lerobot.py:ImageAugPipeline.normalize と一致、val mode = aug 無し)

    cv2.resize + numpy normalize では torchvision antialias と 1 pixel 差が出るため
    使わない。training と inference の gap を silent に起こさないための徹底。

    Args:
        frame_bgr: (H, W, 3) uint8 BGR image (cv2 native、YoloObbPerception の入力
                   と同 format)。head の場合、caller が split_head_stereo() で
                   L/R 単 cam (480x640) に分けたものを渡す。

    Returns:
        (3, 224, 224) float32、ImageNet normalized、CHW layout。
    """
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"frame_bgr must be (H, W, 3) uint8 BGR, got shape={frame_bgr.shape} "
            f"dtype={frame_bgr.dtype}"
        )

    # lazy: env-isolated dependencies (torch/torchvision は runtime env or model env のみ)
    import torch
    import torchvision.transforms.functional as TF

    # 1. BGR → RGB (contiguous copy for from_numpy)
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])

    # 2. HWC uint8 → CHW float32 [0, 1] (LeRobot Dataset return format と一致)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0

    # 3. Resize with antialias (data_lerobot.py:make_image_transform 準拠)
    tensor = TF.resize(tensor, [IMAGE_HW[0], IMAGE_HW[1]], antialias=True)

    # 4. ImageNet normalize (data_lerobot.py:ImageAugPipeline.normalize、val mode)
    tensor = TF.normalize(tensor, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))

    return tensor.numpy()


def build_batch_dict(obs: Observation, mode: str) -> dict:
    """Observation → RAMEN-Ori model.predict_action() が受け取る batch dict。

    Batch key layout (model.py 冒頭 docstring):
        - images:        (N_cams, 3, 224, 224) float, ImageNet-normalized I_t
        - images_prev:   (N_cams, 3, 224, 224) float, I_{t-1} (先頭 tick は zeros)
        - cam_id:        (N_cams,) long, cam vocabulary global id
        - obb_verts:     (N_cams, top_K, 8) float
        - obb_conf:      (N_cams, top_K, 1) float [0,1]
        - obb_class_id:  (N_cams, top_K) long
        - obb_cam_id:    (N_cams, top_K) long
        - obb_valid_mask:(N_cams, top_K) bool
        - state:         (state_dim,) float
        - skill_id:      scalar long

    Note: 返り値は **numpy dict** (torch 化は predict() 内部で行う)、default env
    で shape/dtype 検証可能。

    Args:
        obs: Skill wrapper が assemble 済の Observation。
        mode: "none" のみ Phase 3 対応。"overlay" / "precomputed_token" は Phase
              5/6 で拡張。

    Returns:
        numpy dict、batch 次元 (B=1) 追加前。predict() 側で unsqueeze(0)。
    """
    if mode not in ("none", "precomputed_token", "overlay"):
        raise ValueError(
            f"mode must be one of 'none' / 'precomputed_token' / 'overlay', "
            f"got {mode!r}"
        )

    if obs.state.shape != (STATE_DIM,):
        raise ValueError(
            f"Observation.state must have shape ({STATE_DIM},) for RAMEN-Ori, got "
            f"{obs.state.shape}"
        )
    if obs.skill_id is None:
        raise ValueError(
            "RAMEN-Ori requires obs.skill_id (skill embedding index). Skill wrapper "
            "must set it (0..NUM_SKILLS-1 per skill_mapping.py)."
        )
    if not (0 <= obs.skill_id < NUM_SKILLS):
        raise ValueError(
            f"skill_id must be in [0, {NUM_SKILLS}), got {obs.skill_id}"
        )
    for cam in CAMERAS:
        if cam not in obs.frames_bgr:
            raise KeyError(
                f"Observation.frames_bgr missing required cam {cam.value!r} "
                f"(RAMEN-Ori needs {tuple(c.value for c in CAMERAS)!r})"
            )

    N = len(CAMERAS)

    # mode=overlay: 描画は raw resolution (480x640) の frames_bgr に対して行い、
    # その後 preprocess_frame (resize→normalize) を通す = training-side D-3 hook と
    # 同順 (post-decode hook → LeRobot Resize が antialias)。
    # ※ in-place で書き換わるため、caller の frames_bgr を汚染しないよう copy。
    def _maybe_overlay(cam: CameraKey, frame: np.ndarray) -> np.ndarray:
        if mode != "overlay" or not obs.obb_detections:
            return frame
        if cam not in OVERLAY_TARGET_CAMS:
            # training-side cache に無い cam (wrist) は overlay 対象外
            return frame
        # per-cam det list を dict から取得 (未提供 cam は overlay skip)
        cam_dets = obs.obb_detections.get(cam)
        if not cam_dets:
            return frame
        # in-place 汚染回避のため copy してから描画
        return overlay_obb_on_frame(frame.copy(), cam_dets)

    # Current frames (I_t)
    images = np.stack(
        [
            preprocess_frame(_maybe_overlay(cam, obs.frames_bgr[cam]))
            for cam in CAMERAS
        ],
        axis=0,
    )  # (N, 3, 224, 224)

    # Previous frames (I_{t-1})、None なら zeros (data_lerobot.py "先頭 frame は zeros" 追従)
    if obs.frames_bgr_prev is None:
        images_prev = np.zeros_like(images)
    else:
        images_prev = np.stack(
            [
                preprocess_frame(_maybe_overlay(cam, obs.frames_bgr_prev[cam]))
                if cam in obs.frames_bgr_prev
                else np.zeros((3, IMAGE_HW[0], IMAGE_HW[1]), dtype=np.float32)
                for cam in CAMERAS
            ],
            axis=0,
        )

    # cam_id: 0..N_cams-1 (data_lerobot.py:_cam_id = torch.arange(N_cams))
    cam_ids_1d = np.arange(N, dtype=np.int64)
    cam_ids_2d = np.broadcast_to(cam_ids_1d[:, None], (N, TOP_K)).copy()

    # OBB signals: mode=none / overlay では zeros + valid_mask 全 False。
    # mode=precomputed_token では per-cam det を top_K 個 pack (conf desc)。
    if mode == "precomputed_token" and obs.obb_detections:
        obb_verts, obb_conf, obb_class_id, obb_valid_mask = pack_obb_tokens(
            obs.obb_detections, cams=CAMERAS, top_k=TOP_K
        )
    else:
        obb_verts = np.zeros((N, TOP_K, 8), dtype=np.float32)
        obb_conf = np.zeros((N, TOP_K, 1), dtype=np.float32)
        obb_class_id = np.zeros((N, TOP_K), dtype=np.int64)
        obb_valid_mask = np.zeros((N, TOP_K), dtype=bool)
    # mode == "overlay" は build_batch_dict 上流の _maybe_overlay で処理済 (OBB
    # channel は全 invalid、model は overlay された画像を vision 経由で読む)

    return {
        "images": images,
        "images_prev": images_prev,
        "cam_id": cam_ids_1d,
        "obb_verts": obb_verts,
        "obb_conf": obb_conf,
        "obb_class_id": obb_class_id,
        "obb_cam_id": cam_ids_2d,
        "obb_valid_mask": obb_valid_mask,
        "state": obs.state.astype(np.float32, copy=False),
        "skill_id": np.int64(obs.skill_id),
    }


# ---- RamenOriPolicy (Skeleton、実 forward は integration test / local smoke) ---- #


class RamenOriPolicy:
    """RAMEN-Ori inference loader (`model.ramen_ori.model.RamenOriPolicy` 委譲)。

    実 model load + forward は torch + LeRobot fork + model.ramen_ori 依存。
    default env / runtime env の unit test は preprocessing helper と config
    validation のみ verify、実 forward は `@pytest.mark.integration` marker で
    model/ramen_ori env or local sakura env に飛ばす。

    Attributes:
        cfg: 生成時 config (immutable)。
        _model: model.ramen_ori.model.RamenOriPolicy instance (lazy load)。
        _device: torch device 文字列。
    """

    # Public constants (Skill wrapper が config assemble する時に参照)
    STATE_DIM = STATE_DIM
    ACTION_DIM = ACTION_DIM
    CHUNK_LEN = CHUNK_LEN
    IMAGE_HW = IMAGE_HW
    CAMERAS = CAMERAS
    NUM_SKILLS = NUM_SKILLS
    SKILL_MOVE_TABLE_BASE = SKILL_MOVE_TABLE_BASE
    SKILL_ROTATE_TABLE_BASE = SKILL_ROTATE_TABLE_BASE
    # VlaSkill only receives the common policy object.  Expose the canonical
    # module-level mapper on the class so online and offline state assembly use
    # the exact same 71D contract.
    build_state_from_raw = staticmethod(build_state_from_raw)

    def __init__(
        self,
        cfg: PolicyConfig,
        _model=None,
        _device: str | None = None,
    ) -> None:
        validate_ramen_ori_config(cfg)
        self.cfg = cfg
        self._model = _model
        self._device = _device or cfg.device

    @classmethod
    def from_ckpt(
        cls,
        cfg: PolicyConfig,
        *,
        config_name: str = "base",
        use_ema: bool = True,
        ckpt_filename: str | None = None,
    ) -> "RamenOriPolicy":
        """HF or local から fine-tuned RAMEN-Ori ckpt を load + Hydra で model 組立。

        訓練時 (`model/ramen_ori/train.py:_build_model`) と厳密同じ手順で sub-module
        を instantiate し、ckpt (`torch.save({"model_state_dict": ...})`) を load。

        Args:
            cfg: PolicyConfig。ckpt_ref は HF repo (`Team-RAMEN/...`) or local dir /
                 .pt file path。
            config_name: Hydra config 名 (default "base" = base.yaml、real_task5_7
                 も可)。model 構造は base で 100% 決まるので base default で OK
                 (real_task5_7 は training-specific override が主で model 構造は同じ)。
            use_ema: True なら ckpt の `ema_state_dict` を model に load (val 時と
                 同じ挙動)。False なら `model_state_dict` (raw training weights)。
                 default True (training ema decay=0.9999 は inference で優先すべき)。
            ckpt_filename: HF repo 内で load する .pt file 名 (例
                 "ckpt_step_100000.pt")。None なら latest step の ckpt を auto
                 pick (repo 内 `ckpt_step_*.pt` の最大 step)。

        Returns:
            初期化済 RamenOriPolicy。model は cfg.device (cuda) 配置、eval mode。

        Note: 依存する model/ramen_ori/* nn.Module classes + Hydra + LingBot
        backbone downloader (HF 経由) が必要。runtime env には無いので
        model/ramen_ori pixi env で走らせる想定。
        """
        # lazy: env-isolated dependencies (torch / hydra / model.ramen_ori.*)
        import torch
        from huggingface_hub import HfApi, hf_hub_download
        from hydra import compose, initialize_config_dir

        from model.ramen_ori.model import RamenOriPolicy as _NnModule
        from model.ramen_ori.vision_backbone import load_vision_backbone

        # ---- 1. Hydra config load ----
        # model/ramen_ori/configs/ を initialize_config_dir 経由で読む
        import os
        from pathlib import Path

        config_dir = str(
            (
                Path(__file__).parent.parent.parent.parent.parent
                / "model"
                / "ramen_ori"
                / "configs"
            ).resolve()
        )
        if not os.path.isdir(config_dir):
            raise FileNotFoundError(
                f"model/ramen_ori/configs not found at {config_dir}"
            )

        # Hydra は global singleton なので既 initialize 済なら clear
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            hydra_cfg = compose(
                config_name=config_name,
                overrides=list(cfg.hydra_overrides),
            )

        # ---- 2. Model instantiate (train.py:_build_model と同手順) ----
        import hydra as _hydra

        backbone, embed_dim = load_vision_backbone(
            variant=hydra_cfg.vision_backbone.variant,
            device=cfg.device,
            dtype=hydra_cfg.vision_backbone.dtype,
        )
        vision = _hydra.utils.instantiate(
            hydra_cfg.model.vision, backbone=backbone, embed_dim=embed_dim
        )
        temporal = _hydra.utils.instantiate(hydra_cfg.model.temporal)
        obb = _hydra.utils.instantiate(hydra_cfg.model.obb)
        state = _hydra.utils.instantiate(hydra_cfg.model.state)
        skill = _hydra.utils.instantiate(hydra_cfg.model.skill)
        fusion = _hydra.utils.instantiate(hydra_cfg.model.fusion)
        action_expert = _hydra.utils.instantiate(hydra_cfg.model.action_expert)
        aux_head = None
        if hydra_cfg.model.get("aux_head") is not None:
            aux_head = _hydra.utils.instantiate(hydra_cfg.model.aux_head)

        model = _NnModule(
            vision=vision,
            temporal=temporal,
            obb=obb,
            state=state,
            skill=skill,
            fusion=fusion,
            action_expert=action_expert,
            aux_head=aux_head,
            aux_weight=hydra_cfg.model.get("aux_weight", 0.1),
            d_model=hydra_cfg.model.d_model,
            num_cams=hydra_cfg.model.num_cams,
            chunk_len=hydra_cfg.model.chunk_len,
            action_dim=hydra_cfg.model.action_dim,
            sample_n_steps=hydra_cfg.model.sample_n_steps,
        )
        model = model.to(cfg.device)

        # ---- 3. Ckpt DL + state_dict load ----
        ckpt_ref = cfg.ckpt_ref
        if os.path.isfile(ckpt_ref):
            ckpt_path = ckpt_ref
        elif os.path.isdir(ckpt_ref):
            # local dir 内で最新 step の ckpt を picking
            step_files = sorted(
                Path(ckpt_ref).glob("ckpt_step_*.pt"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            if not step_files:
                raise FileNotFoundError(f"no ckpt_step_*.pt in {ckpt_ref}")
            ckpt_path = str(step_files[-1])
        else:
            # HF repo: latest step ckpt auto pick or filename 指定
            api = HfApi()
            files = api.list_repo_files(repo_id=ckpt_ref)
            if ckpt_filename is None:
                step_ckpts = sorted(
                    [f for f in files if f.startswith("ckpt_step_") and f.endswith(".pt")],
                    key=lambda f: int(f.replace("ckpt_step_", "").replace(".pt", "")),
                )
                if not step_ckpts:
                    raise FileNotFoundError(
                        f"no ckpt_step_*.pt in HF repo {ckpt_ref}"
                    )
                ckpt_filename = step_ckpts[-1]
            ckpt_path = hf_hub_download(repo_id=ckpt_ref, filename=ckpt_filename)

        import sys as _sys

        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        if use_ema and "ema_state_dict" in ckpt:
            # EMA state_dict を model に反映 (data_lerobot.py val flow と同じ)
            # ema_state_dict は EMA class の shadow weights を key で保持、
            # model.state_dict() 形式に合わせて load
            ema_state = ckpt["ema_state_dict"]
            # EMA class の save format 次第、"shadow" key 内が直接 state_dict の
            # 場合と、EMA object そのままの場合がある — 両方対応
            if "shadow_params" in ema_state:
                # EMAdiff 系 (torch_ema)
                shadow = ema_state["shadow_params"]
                # 長さ検証: zip は silent truncate なので事前 assert (Issue #125 PR 126
                # Finding 2、partial EMA load 事故予防)。
                n_params = sum(1 for _ in model.named_parameters())
                if len(shadow) != n_params:
                    raise RuntimeError(
                        f"EMA shadow_params length mismatch: shadow={len(shadow)}, "
                        f"model.named_parameters()={n_params}. ckpt {ckpt_path!r} "
                        f"may be from a different model architecture."
                    )
                for (name, param), s in zip(model.named_parameters(), shadow):
                    param.data.copy_(s)
                print(
                    f"[RamenOriPolicy] loaded EMA shadow_params ({n_params} params)",
                    file=_sys.stderr,
                )
            else:
                # 生 state_dict
                missing, unexpected = model.load_state_dict(ema_state, strict=False)
                _log_state_dict_load("EMA state_dict", missing, unexpected, ckpt_path)
                _validate_architecture_state_dict_load(missing, unexpected, ckpt_path)
            del ema_state
        else:
            missing, unexpected = model.load_state_dict(
                ckpt["model_state_dict"], strict=False
            )
            _log_state_dict_load("model_state_dict", missing, unexpected, ckpt_path)
            _validate_architecture_state_dict_load(missing, unexpected, ckpt_path)

        del ckpt  # free CPU memory
        model.eval()

        # dtype 変換 (bf16 / fp16 指定時)
        if cfg.dtype == "bf16":
            model = model.to(torch.bfloat16)
        elif cfg.dtype == "fp16":
            model = model.to(torch.float16)

        return cls(cfg=cfg, _model=model, _device=cfg.device)

    def warmup(self, n_iter: int = 5) -> None:
        """cuDNN autotune + LingBot cache warm-up。

        dummy Observation で predict() を n_iter 回叩く。
        """
        if self._model is None:
            raise RuntimeError(
                "RamenOriPolicy is not loaded. Call from_ckpt() first."
            )
        dummy_obs = self._make_dummy_observation()
        for _ in range(n_iter):
            self.predict(dummy_obs)

    def predict(self, obs: Observation) -> PolicyAction:
        """1 tick observation → RAMEN-Ori action chunk。

        Pipeline:
            1. `build_batch_dict(obs, mode)` → numpy dict
            2. numpy → torch tensor、batch 次元追加 (B=1)
            3. `self._model.predict_action(batch)` → (1, 16, 19)
            4. → PolicyAction wrap で return (denorm 無し = training が action 未
               normalize 前提、Phase 4 で action stats 確認)
        """
        if self._model is None:
            raise RuntimeError(
                "RamenOriPolicy is not loaded. Call from_ckpt() first."
            )
        # lazy: env-isolated dependencies (torch は model env only)
        import torch

        t0 = time.monotonic_ns()
        raw_batch = build_batch_dict(obs, mode=self.cfg.mode)

        # numpy → torch tensor、batch 次元追加 (B=1)
        torch_batch: dict = {}
        for key, val in raw_batch.items():
            t = torch.from_numpy(np.ascontiguousarray(val)) if isinstance(val, np.ndarray) else torch.tensor(val)
            torch_batch[key] = t.unsqueeze(0).to(self._device)

        with torch.inference_mode():
            action_chunk = self._model.predict_action(torch_batch)
        action_np = action_chunk[0].detach().cpu().numpy().astype(np.float32, copy=False)

        latency_ms = (time.monotonic_ns() - t0) / 1e6
        return PolicyAction(
            action_chunk=action_np,
            latency_ms=latency_ms,
            metadata={
                "mode": self.cfg.mode,
                "chunk_len": action_np.shape[0],
                "action_dim": action_np.shape[1],
                "skill_id": obs.skill_id,
            },
        )

    def close(self) -> None:
        """GPU memory 解放 (long-running orchestrator の graceful shutdown)。"""
        try:
            import torch  # lazy: env-isolated
        except ImportError:
            self._model = None
            return
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- private helpers ---- #

    def _make_dummy_observation(self) -> Observation:
        H, W = 480, 640
        return Observation(
            frames_bgr={cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in CAMERAS},
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=SKILL_MOVE_TABLE_BASE,
            language=None,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )
