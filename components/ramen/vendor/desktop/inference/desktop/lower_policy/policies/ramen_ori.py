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

import statistics
import sys
import threading
import time
from collections import deque

import numpy as np

from inference.desktop.lower_policy.rtc import (
    AUTO_FROZEN_STEPS,
    ChunkLeftoverBuffer,
    DelayEstimator,
    build_velocity_strength,
    validate_rtc_against_chunk_len,
)
from inference.desktop.lower_policy.policies.base import (
    CameraKey,
    G1_UPPER_BODY_JOINT_DIM,
    G1_UPPER_BODY_JOINT_INDICES,
    OVERLAY_JPEG_SUBSAMPLINGS,
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

# RAMEN-Ori 4 cam layout (Phase 2、data_lerobot.py:default_camera_keys 順)
CAMERAS: tuple[CameraKey, ...] = (
    CameraKey.HEAD_LEFT,
    CameraKey.HEAD_RIGHT,
    CameraKey.WRIST_LEFT,
    CameraKey.WRIST_RIGHT,
)

# RAMEN-Ori 3 cam layout (Phase K = Issue #129 commit 652c31b、head_right 除外)。
# GR00T と apples-to-apples 比較性 + vision backbone forward の compute -33% 目的。
# Phase K 6 run (Run 5 winner 含む) はこの layout で学習、slot YAML で明示指定して
# 4 cam default を override する。cam_id は 0..N-1 で割当 (build_batch_dict L572 arange)。
CAMERAS_3CAM_PHASE_K: tuple[CameraKey, ...] = (
    CameraKey.HEAD_LEFT,
    CameraKey.WRIST_LEFT,
    CameraKey.WRIST_RIGHT,
)

# RAMEN-Ori variant で許容する cam layout 集合 (validate_ramen_ori_config が使う)。
VALID_CAM_LAYOUTS: tuple[tuple[CameraKey, ...], ...] = (
    CAMERAS,
    CAMERAS_3CAM_PHASE_K,
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


_COMPILE_PREFIX = "_orig_mod."


def strip_compile_prefix(state_dict: dict) -> dict:
    """`torch.compile` 由来の `_orig_mod.` prefix を key から除去する。

    `train.py` は `torch.compile(model)` 後の `model.state_dict()` を保存するため、
    `speedup.torch_compile=true` で学習した ckpt は全 key に `_orig_mod.` が付く
    (Issue #137、Phase K R-6 全 6 run が該当)。prefix 付きのまま
    `load_state_dict(strict=False)` すると **1 key も一致せず**、モデルは初期化の
    まま残る (実測: 非 backbone param 374 個中 135 個が zero-init のまま)。

    Issue #137 Phase B 以降の ckpt は保存側で正規化済なので no-op になる。
    """
    if not any(k.startswith(_COMPILE_PREFIX) for k in state_dict):
        return state_dict
    return {
        k[len(_COMPILE_PREFIX):] if k.startswith(_COMPILE_PREFIX) else k: v
        for k, v in state_dict.items()
    }


def ema_state_dict_is_stale(ema_state: dict, model_state: dict) -> bool:
    """EMA shadow が一度も更新されていない ckpt かを判定する (Issue #137)。

    `train.py` が EMA を `torch.compile` **前**に構築し、`update()` を compile
    **後**の `named_parameters()` で回していたため、key が一致せず shadow が
    step 0 の初期化値のまま保存されていた。この ckpt を推論で使うと
    `action_expert.output_proj` が zero-init のままになり、Flow Matching の
    `v_pred = 0` → `sample_action` が noise `x_0` をそのまま返す。

    判定は「model 側が非ゼロなのに EMA 側が完全ゼロ」の key が 1 つでもあるか。
    zero-init 層 (output_proj / AdaLN-Zero) は学習で必ず非ゼロになるので、
    正常に更新された EMA でこの条件が立つことはない。閾値やヒューリスティックは
    使わない。

    Args:
        ema_state / model_state: いずれも prefix 正規化済の state_dict。

    Returns:
        True なら shadow が未更新 = raw weights を使うべき。
    """
    for key, ema_tensor in ema_state.items():
        model_tensor = model_state.get(key)
        if model_tensor is None:
            continue
        if not (hasattr(ema_tensor, "numel") and hasattr(model_tensor, "numel")):
            continue
        if ema_tensor.numel() == 0 or ema_tensor.shape != model_tensor.shape:
            continue
        if float(ema_tensor.abs().max()) == 0.0 and float(model_tensor.abs().max()) != 0.0:
            return True
    return False


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

    Issue #134 Phase H-4: rel action model の `_relative_arms_mean/std` は EMA
    state_dict に含まれない (buffer は EMA shadow の対象外)。init 時に
    ``relative_stats`` から埋めるので、EMA path で missing 判定されても値は有効
    → allow-list に加える。
    """
    _ALLOWED_MISSING = frozenset({"_relative_arms_mean", "_relative_arms_std"})
    disallowed_missing = [
        key for key in missing
        if not key.startswith("vision.backbone.") and key not in _ALLOWED_MISSING
    ]
    # Issue #137: `fk.*` は L4 FK anchor loss 用の学習専用 buffer
    # (model/ramen_ori/fk.py:G1WristFKTorch)。inference model は FK を持たないので
    # unexpected に出るが、action 出力には一切関与しない → 許容する。
    disallowed_unexpected = [key for key in unexpected if not key.startswith("fk.")]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "RAMEN-Ori checkpoint architecture mismatch: "
            f"missing_non_backbone={len(disallowed_missing)} "
            f"unexpected={len(disallowed_unexpected)} checkpoint={ckpt_path!r}. "
            "Refusing to evaluate a partially loaded model."
        )


# ---- Preprocessing helpers (default env で testable、torch 不要) ---- #


CONTRACT_STATE_VARIANT_V2 = 2
# memory が見るカメラ (学習は `observation.images.cam_0` = head_left、
# `model/ramen_ori/memory_features.py:MEMORY_CAMERA`)。
_MEMORY_CAMERA_KEY = CameraKey.HEAD_LEFT
# 学習の camera key (`observation.images.cam_N`) → 推論の CameraKey。
# cam_0 head_left / cam_1 head_right / cam_2 左手首 / cam_3 右手首。
_CONTRACT_CAM_INDEX_TO_KEY = {
    0: CameraKey.HEAD_LEFT,
    1: CameraKey.HEAD_RIGHT,
    2: CameraKey.WRIST_LEFT,
    3: CameraKey.WRIST_RIGHT,
}


def _contract_camera_keys(camera_keys) -> tuple[str, ...]:
    """約束の camera key を推論の CameraKey の値に直す。並びも保つ。"""
    resolved: list[str] = []
    for key in camera_keys:
        index = str(key).rsplit("cam_", 1)[-1]
        if not index.isdigit() or int(index) not in _CONTRACT_CAM_INDEX_TO_KEY:
            raise RuntimeError(f"ckpt contract has an unknown camera key: {key!r}")
        resolved.append(_CONTRACT_CAM_INDEX_TO_KEY[int(index)].value)
    return tuple(resolved)


def validate_contract(contract: dict, cfg: PolicyConfig) -> None:
    """ckpt の約束と推論の設定が合っているかを、重みを読む前に確かめる (Issue #141 P8-3)。

    合っていなければここで止める。黙って違う入力で動かすと、実機で「なぜか下手」な
    run を 1 本使ってしまう。

    見るもの:
        - slot の `skill_id` がその ckpt の学習した skill に含まれるか
        - overlay を焼いた YOLO の `repo@revision` が `policy_config.yaml` と同じか
        - overlay の conf と線の太さ (推論の描画と同じか)
        - カメラの並び / 画像の大きさ / chunk の長さ / action の空間

    Args:
        contract: `ckpt["contract"]` (`model/ramen_ori/contract.py` が学習時に作る)。
        cfg: 推論の slot の設定。
    """
    version = contract.get("version")
    if version != 1:
        raise RuntimeError(
            f"ckpt contract version {version!r} is not supported by this inference "
            "build (expected 1)"
        )
    problems: list[str] = []

    skill_ids = {int(skill["id"]) for skill in contract.get("skills", [])}
    if cfg.skill_id is None:
        problems.append(
            f"slot does not set skill_id; the ckpt was trained on {sorted(skill_ids)}"
        )
    elif int(cfg.skill_id) not in skill_ids:
        problems.append(
            f"slot skill_id={cfg.skill_id} is not in the ckpt's skills {sorted(skill_ids)}"
        )

    state = contract.get("state") or {}
    state_shape = state.get("variant")
    if state_shape is not None and str(state_shape) != "71d":
        problems.append(
            f"state layout: ckpt={state_shape!r} inference=71d (E-δ)"
        )
    state_version = state.get("version")
    if state_version is not None and int(state_version) not in (1, 2):
        problems.append(
            f"state version {state_version} is not supported (inference knows 1 and 2)"
        )

    action = contract.get("action") or {}
    if action.get("space") != cfg.action_space:
        problems.append(
            f"action space: ckpt={action.get('space')!r} slot={cfg.action_space!r}"
        )

    images = contract.get("images") or {}
    ckpt_cams = _contract_camera_keys(images.get("camera_keys") or ())
    slot_cams = tuple(cam.value for cam in cfg.cams)
    if ckpt_cams and ckpt_cams != slot_cams:
        problems.append(f"cameras: ckpt={list(ckpt_cams)} slot={list(slot_cams)}")
    img_size = images.get("img_size")
    if img_size is not None and int(img_size) != IMAGE_HW[0]:
        problems.append(f"image size: ckpt={img_size} inference={IMAGE_HW[0]}")

    overlay = images.get("overlay")
    if overlay:
        ckpt_yolo = overlay.get("yolo_ckpt")
        if ckpt_yolo and cfg.yolo_ckpt_ref and ckpt_yolo != cfg.yolo_ckpt_ref:
            problems.append(
                f"overlay YOLO weight: ckpt={ckpt_yolo!r} policy_config={cfg.yolo_ckpt_ref!r}"
            )
        conf = overlay.get("conf")
        if conf is not None and abs(float(conf) - DEFAULT_OVERLAY_CONF_THRESHOLD) > 1e-9:
            problems.append(
                f"overlay conf: ckpt={conf} inference={DEFAULT_OVERLAY_CONF_THRESHOLD}"
            )
        thickness = overlay.get("thickness")
        if thickness is not None and int(thickness) != DEFAULT_OVERLAY_LINE_THICKNESS:
            problems.append(
                f"overlay line thickness: ckpt={thickness} inference={DEFAULT_OVERLAY_LINE_THICKNESS}"
            )

    if problems:
        raise RuntimeError(
            "ckpt contract does not match the inference configuration:\n  - "
            + "\n  - ".join(problems)
        )


def validate_ramen_ori_config(cfg: PolicyConfig) -> None:
    """`PolicyConfig` が RAMEN-Ori の contract を満たすかを検証する。

    - cams が VALID_CAM_LAYOUTS のいずれかであること:
        * 4 cam (CAMERAS) = Phase 2 5 variant (data_lerobot.py:default_camera_keys 順)
        * 3 cam (CAMERAS_3CAM_PHASE_K) = Phase K 6 run (head_right 除外)
    - mode が "none" / "precomputed_token" / "overlay" のいずれか (base.py の
      PolicyConfig.__post_init__ でも validate されるが RAMEN-Ori は 3 mode 全対応)
    - dtype fp32 or bf16 (fp16 は非推奨)
    """
    if tuple(cfg.cams) not in VALID_CAM_LAYOUTS:
        valid_str = "\n".join(
            f"  - {tuple(c.value for c in layout)!r}" for layout in VALID_CAM_LAYOUTS
        )
        raise ValueError(
            f"RAMEN-Ori cams layout {tuple(c.value for c in cfg.cams)!r} not "
            f"supported. Valid layouts:\n{valid_str}"
        )
    if (
        cfg.action_space == "rel"
        and cfg.rtc.enabled
        and not cfg.rtc.allow_experimental_relative_action
    ):
        raise ValueError(
            "RAMEN-Ori RTC is not safe with action_space='rel': the RTC soft ramp "
            "operates on normalized delta-q rows whose errors accumulate during "
            "absolute-joint reconstruction. Use async replanning/temporal ensemble "
            "with rtc.enabled=false."
        )
    if (
        cfg.action_space == "rel"
        and cfg.rtc.enabled
        and cfg.rtc.allow_experimental_relative_action
    ):
        print(
            "[ramen_ori] WARNING: experimental RTC on cumulative relative "
            "delta-q is enabled. Issue #137 B-3 produced 827/899 safety HOLD "
            "ticks; this setting is for the isolated B-4 comparison only.",
            file=sys.stderr,
        )


def build_state_from_raw(raw: RawRobotState, *, state_variant: int = 1) -> np.ndarray:
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
        state_variant: state の定義の版 (`ckpt["contract"]["state"]["variant"]`)。
            1 = Phase K まで。2 = Issue #141 の再学習 (tracking_err の腰 3 次元は 0。
            腰の指令は出していないので、学習では常に 0 が入っている)。

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
        if state_variant >= CONTRACT_STATE_VARIANT_V2:
            # 腰は指令を出していない (action は腕 14 + hand 2)。学習の state は
            # 腰の tracking_err を 0 で作っているので、推論も 0 にする (Issue #141 P8-4)。
            out[19:22] = 0.0

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


def match_training_jpeg(frame_bgr: np.ndarray, subsampling: str) -> np.ndarray:
    """overlay 対象 cam の画像を、学習 overlay cache の jpg 保存と同じ設定で 1 回通す。

    学習の overlay cache は box 描画後の画像を quality 90 で jpg 保存し、学習時は cv2 で
    読んでいる。色差 (`subsampling`) は焼き込みの時期で違う:

    - "4:4:4": Issue #139 の統合 cache (2026-09-12 以降)。GPU の nvJPEG は色差を間引かない
    - "4:2:0": それより前 (GR00T #129 / RAMEN-Ori Phase K)。PIL の既定

    2px の色線は 4:2:0 の間引きで色が滲むので、推論時にそのまま描いた線とは明確に違う
    (box の線で平均 16/255、背景は 1/255 未満)。逆に 4:4:4 の cache で学習した ckpt に
    4:2:0 を通すと、box の画素が平均 5.9/255 (上位 5% で 56/255) ずれる。box が 0 件の
    frame も cache では同じ保存を通っているので、overlay mode では描画の有無にかかわらず通す。

    WHY: 焼き込みの保存設定を変えた cache で学習した ckpt を使う時は、slot の
    `overlay_jpeg_subsampling` を合わせる。
    """
    if subsampling not in OVERLAY_JPEG_SUBSAMPLINGS:
        raise ValueError(
            f"subsampling must be one of {tuple(OVERLAY_JPEG_SUBSAMPLINGS)}, got {subsampling!r}"
        )
    # lazy: env-isolated dependencies (Pillow / opencv は runtime env と推論 env のみ)
    import io

    import cv2
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame_bgr[..., ::-1])).save(
        buffer, format="JPEG", quality=90, subsampling=OVERLAY_JPEG_SUBSAMPLINGS[subsampling]
    )
    return cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)


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


def build_batch_dict(
    obs: Observation,
    mode: str,
    cams: tuple[CameraKey, ...] = CAMERAS,
    overlay_jpeg_subsampling: str | None = None,
    with_prev_frames: bool = True,
) -> dict:
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
        cams: 使う cam layout。default は 4 cam (Phase 2 5 variant)、Phase K
              Run 5 は 3 cam (CAMERAS_3CAM_PHASE_K) を渡す。
        overlay_jpeg_subsampling: overlay 画像を通す jpg の色差 ("4:4:4" / "4:2:0")。
              mode="overlay" では必須 (slot の PolicyConfig.overlay_jpeg_subsampling)。

    Returns:
        numpy dict、batch 次元 (B=1) 追加前。predict() 側で unsqueeze(0)。
    """
    if mode not in ("none", "precomputed_token", "overlay"):
        raise ValueError(
            f"mode must be one of 'none' / 'precomputed_token' / 'overlay', "
            f"got {mode!r}"
        )
    if mode == "overlay" and overlay_jpeg_subsampling is None:
        raise ValueError(
            "mode='overlay' requires overlay_jpeg_subsampling (the training cache's "
            "JPEG chroma subsampling)"
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
    for cam in cams:
        if cam not in obs.frames_bgr:
            raise KeyError(
                f"Observation.frames_bgr missing required cam {cam.value!r} "
                f"(RAMEN-Ori needs {tuple(c.value for c in cams)!r})"
            )

    N = len(cams)

    # mode=overlay: 描画は raw resolution (480x640) の frames_bgr に対して行い、
    # その後 preprocess_frame (resize→normalize) を通す = training-side D-3 hook と
    # 同順 (post-decode hook → LeRobot Resize が antialias)。
    # ※ in-place で書き換わるため、caller の frames_bgr を汚染しないよう copy。
    def _maybe_overlay(cam: CameraKey, frame: np.ndarray) -> np.ndarray:
        if mode != "overlay" or cam not in OVERLAY_TARGET_CAMS:
            # training-side cache に無い cam (wrist) は overlay 対象外
            return frame
        # per-cam det list を dict から取得 (未提供 cam は描画 skip)。
        # in-place 汚染回避のため copy してから描画
        cam_dets = (obs.obb_detections or {}).get(cam)
        drawn = overlay_obb_on_frame(frame.copy(), cam_dets) if cam_dets else frame
        return match_training_jpeg(drawn, overlay_jpeg_subsampling)

    # Current frames (I_t)
    images = np.stack(
        [
            preprocess_frame(_maybe_overlay(cam, obs.frames_bgr[cam]))
            for cam in cams
        ],
        axis=0,
    )  # (N, 3, 224, 224)

    # Previous frames (I_{t-1})。
    # 前の frame が無いとき (skill の 1 tick 目、reset の直後) は **今の frame をそのまま**
    # 入れて差分を 0 にする (Issue #141 束 1-3 / INF-4)。学習も区間の頭では
    # `images_prev = images.clone()` (data_lerobot.py の ep 境界) で差分 0 にしている。
    # zeros を入れると「画像がまるごと現れた」差分になり、学習に無い入力で 1 tick 目を出すことになる。
    # 画像差分を使わない ckpt (Issue #141 の再学習、`model.temporal: null`) では、
    # 前処理そのものを飛ばす (overlay の描画と resize が 1 tick ぶん浮く、P8-5)。
    if not with_prev_frames:
        images_prev = np.zeros_like(images)
    elif obs.frames_bgr_prev is None:
        images_prev = images.copy()
    else:
        # 前の frame に無い cam も、その cam の差分だけ 0 にする (同上)。
        images_prev = np.stack(
            [
                preprocess_frame(_maybe_overlay(cam, obs.frames_bgr_prev[cam]))
                if cam in obs.frames_bgr_prev
                else images[index]
                for index, cam in enumerate(cams)
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
            obs.obb_detections, cams=cams, top_k=TOP_K
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


    def __init__(
        self,
        cfg: PolicyConfig,
        _model=None,
        _device: str | None = None,
        _contract: dict | None = None,
    ) -> None:
        validate_ramen_ori_config(cfg)
        self.cfg = cfg
        self._model = _model
        self._device = _device or cfg.device
        # ckpt の約束 (Issue #141 P8)。Phase K までの ckpt では None で、
        # state の版も入力の作り方も従来のまま。
        self.contract = _contract
        # 約束の `state` は 2 つ持つ: `variant` は state の形 ("71d" = E-δ / "73d" = E-β)、
        # `version` は中身の定義の版 (2 = 腰の tracking_err を 0 にする)。P8-4 が見るのは版。
        self._state_version = int(
            ((_contract or {}).get("state") or {}).get("version", 1)
        )
        # Issue #141 P8-6: memory の token を持つ ckpt では、学習と同じ tracker を
        # tick ごとに回す。skill の開始 (reset) で忘れる。
        self._memory_tracker = None
        self._memory_ticks = 0
        # Issue #137 Phase C: async replanning + temporal ensemble。
        # **cfg.replan_family=None (既定) なら一切使わず、毎 tick 同期推論して
        # chunk 全体を返す従来動作のまま**。設定した時だけ GR00T と同じ
        # 「policy 内部で pipeline + ensembler を持ち 1 行だけ返す」構造になる。
        from model.subtask_policy_training.gr00t.temporal_ensemble import (
            TargetTemporalEnsembler,
        )
        self._ensembler = TargetTemporalEnsembler(
            dim=ACTION_DIM, decay_lambda=cfg.temporal_lambda
        )
        self._current_step: int = 0
        self._pipeline = None
        self._pipeline_lead_steps: int | None = None
        self._pipeline_max_age_s: float | None = None
        self._pending_submit_step: int | None = None
        self._last_seen_obs: Observation | None = None
        # Issue #137 Phase C-3: RTC。cfg.rtc.enabled=False (既定) なら buffer も
        # 作らず prefix 経路に入らない = 従来動作と同一。
        self._rtc_leftover = (
            ChunkLeftoverBuffer(action_dim=ACTION_DIM) if cfg.rtc.enabled else None
        )
        self._rtc_delay = DelayEstimator()
        self._rtc_tick_deltas: deque[float] = deque(maxlen=32)
        self._rtc_last_tick_ns: int | None = None
        self._rtc_disabled_after_error = False
        self._last_rtc_metadata: dict = {}
        # model の action 空間 (16=Phase K waist 除外 / 19)。prefix を model space に
        # 戻すのに要るが ckpt を読むまで確定しないので初回 predict で確定させる。
        self._model_action_dim: int | None = None
        self._model_chunk_len: int | None = None
        # predict は stateless (tick 間で持ち越す可変状態は無い) ので GR00T のような
        # correctness 上の必要は無いが、async worker と sync_fallback が同時に
        # forward すると VRAM ピークが倍になる。構造も GR00T と揃えて直列化する。
        self._inference_lock = threading.Lock()

    def _memory_enabled(self) -> bool:
        """この ckpt が memory の token を使うか (Issue #141 P8-6)。"""
        return getattr(self._model, "memory", None) is not None

    def _memory_vector(self, obs: Observation) -> np.ndarray:
        """1 tick ぶん memory を進めて (51,) を返す。

        学習と同じ `MemoryTracker` を回す。入力は
            - 検出: この tick の検出 (policy 用の filter の後、conf は tracker 側で 0.30 で切る)
            - 手先: **前の tick に送った指令**を FK した左右の xyz (6)

        前の指令は 71 次元の state から戻す (Issue #141 P8-6 の案 a):
        `state[0:19] + state[19:38]` = joint + tracking_err = 前の tick の指令。
        skill の最初の tick は指令が無いので None (学習の t=0 と同じ)。
        """
        # lazy: env-isolated dependencies (torch / model package は runtime env のみ)
        import torch

        if self._memory_tracker is None:
            from model.ramen_ori.memory_features import MemoryTracker

            self._memory_tracker = MemoryTracker()

        camera = _MEMORY_CAMERA_KEY
        detections = (obs.obb_detections or {}).get(camera) or []
        if detections:
            class_id = np.asarray([d.class_id for d in detections], dtype=np.int64)
            conf = np.asarray([d.confidence for d in detections], dtype=np.float64)
            verts = np.stack([np.asarray(d.verts, dtype=np.float64) for d in detections])
        else:
            class_id = np.zeros(0, dtype=np.int64)
            conf = np.zeros(0, dtype=np.float64)
            verts = np.zeros((0, 4, 2), dtype=np.float64)

        hand_pos = None
        if self._memory_ticks > 0:      # skill の最初の tick は学習の t=0 と同じで None
            # この tick の state が持っているのが「前の tick に送った指令」なので、
            # ここで戻す。控えて次の tick で使うと、学習より 1 tick (33 ms) 古くなる。
            state = np.asarray(obs.state, dtype=np.float32)
            fk = getattr(self._model, "fk", None)
            if fk is None:
                from model.ramen_ori.fk import G1WristFKTorch

                fk = G1WristFKTorch.from_default_urdf()
                self._model.fk = fk
            command = torch.as_tensor(
                state[0:19] + state[19:38], dtype=torch.float32
            ).unsqueeze(0).to(self._device)
            with torch.no_grad():
                detailed = fk.forward_detailed(command)
            hand_pos = (
                torch.cat([detailed["left_hand"], detailed["right_hand"]], dim=-1)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        memory = self._memory_tracker.update(class_id, conf, verts, hand_pos)
        self._memory_ticks += 1
        return np.asarray(memory, dtype=np.float32)

    def _model_uses_prev_frames(self) -> bool:
        """model が画像差分 (temporal) を持っているか (Issue #141 P8-5)。

        持っていない ckpt では、前の frame の前処理を丸ごと飛ばせる。
        """
        return getattr(self._model, "temporal", None) is not None

    def build_state_from_raw(self, raw: RawRobotState) -> np.ndarray:
        """71 次元の state を組む。state の版は ckpt の約束から (Issue #141 P8-4)。

        VlaSkill は policy の instance を通して呼ぶので、ckpt ごとに版が変わってよい。
        約束の無い ckpt (Phase K まで) は版 1。
        """
        return build_state_from_raw(raw, state_variant=self._state_version)

    # 自前で pipeline / ensembler を持ち VlaSkill には 1 行だけ返すため、
    # VlaSkill 側の queue 経路 (EXECUTION_HORIZON > 1) は使わない。
    EXECUTION_HORIZON = 1

    def reset(self) -> None:
        """Skill 遷移 / episode 開始時に呼ぶ (VlaSkill._on_start、optional protocol)。

        前 skill の chunk が新 skill の blend / prefix に混入しないよう、
        ensembler・step counter・async pipeline を初期化する。
        """
        self._ensembler.reset()
        self._current_step = 0
        if self._pipeline is not None:
            try:
                self._pipeline.close(timeout_s=0.5)
            except Exception:
                pass
            self._pipeline = None
        self._pending_submit_step = None
        self._last_seen_obs = None
        if self._rtc_leftover is not None:
            self._rtc_leftover.reset()
        self._rtc_delay.reset()
        self._rtc_tick_deltas.clear()
        self._rtc_last_tick_ns = None
        # Issue #141 P8-6: memory は区間 (skill) ごとに作り直す。前の skill の
        # 基準や EMA を持ち越すと、学習の「区間の頭で reset」と食い違う。
        if self._memory_tracker is not None:
            self._memory_tracker.reset()
        self._memory_ticks = 0

    def _rtc_tick_period_s(self) -> float | None:
        """predict() の呼出間隔から実測した tick 周期 [s]。sample 不足なら None。"""
        if len(self._rtc_tick_deltas) < 3:
            return None
        return float(statistics.median(self._rtc_tick_deltas))

    def _build_rtc_prefix(self, torch_batch: dict, rtc_step: int | None) -> tuple[dict, dict]:
        """RTC の prefix と velocity_strength を組んで predict_action の kwargs を返す。

        `_in_process_predict_chunk_19d` の中から、**現 tick の state を組んだ後**に
        呼ぶこと。prefix の起点となる `arms_current` をそこから取るため
        (これが re-anchor そのもの)。

        Returns:
            (kwargs, metadata)。RTC を使わない tick では kwargs は空 dict。
        """
        if (
            self._rtc_leftover is None
            or rtc_step is None
            or self._rtc_disabled_after_error
            or self._model_action_dim is None
            or self._model_chunk_len is None
        ):
            return {}, {"rtc_enabled": bool(self.cfg.rtc.enabled)}

        chunk_len = int(self._model_chunk_len)
        overlap = self.cfg.rtc.overlap_steps
        if overlap is None:
            overlap = self.cfg.execution_steps
        overlap = max(0, min(int(overlap), chunk_len))

        leftover = self._rtc_leftover.remaining(int(rtc_step))
        if leftover is None or overlap < 1:
            return {}, {"rtc_enabled": True, "rtc_prefix_rows": 0}
        rows = min(int(leftover.shape[0]), overlap)
        leftover = leftover[:rows]

        tick_period_s = self._rtc_tick_period_s()
        if self.cfg.replan_family is None:
            # 同期実行では推論中ロボットが保持され、新 chunk から先行送信される
            # action が無い。frozen は定義上 0 (詳細は rtc.py / groot.py と同じ)。
            frozen = 0
        elif self.cfg.rtc.frozen_steps == AUTO_FROZEN_STEPS:
            if tick_period_s is None:
                return {}, {"rtc_enabled": True, "rtc_prefix_rows": 0}
            frozen = self._rtc_delay.frozen_steps(tick_period_s, cap=rows)
            if frozen is None:
                return {}, {"rtc_enabled": True, "rtc_prefix_rows": 0}
        else:
            frozen = max(0, min(int(self.cfg.rtc.frozen_steps), rows))

        try:
            prefix = self._encode_prefix_to_model_space(leftover, torch_batch)
            strength = build_velocity_strength(
                chunk_len=chunk_len,
                frozen_steps=frozen,
                overlap_steps=rows,
                ramp_rate=self.cfg.rtc.ramp_rate,
            )
        except Exception as exc:  # noqa: BLE001 - 実機 slot を落とさない
            self._rtc_disabled_after_error = True
            print(
                f"[ramen_ori] WARNING: RTC prefix build failed ({type(exc).__name__}: "
                f"{exc}). Continuing with RTC disabled for this policy instance.",
                file=sys.stderr,
            )
            return {}, {"rtc_enabled": False, "rtc_error": type(exc).__name__}

        # lazy: env-isolated dependencies (torch は model env only)
        import torch

        metadata = {
            "rtc_enabled": True,
            "rtc_prefix_rows": int(rows),
            "rtc_frozen_steps": int(frozen),
            "rtc_overlap_steps": int(overlap),
            "rtc_ramp_rate": float(self.cfg.rtc.ramp_rate),
            "rtc_async_execution": self.cfg.replan_family is not None,
            "rtc_tick_period_ms": (
                None if tick_period_s is None else float(tick_period_s * 1000.0)
            ),
            "rtc_delay_max_ms": (
                None
                if self._rtc_delay.max_latency_s is None
                else float(self._rtc_delay.max_latency_s * 1000.0)
            ),
        }
        return (
            {
                "prefix": prefix,
                "velocity_strength": torch.from_numpy(strength).to(self._device),
            },
            metadata,
        )

    def _encode_prefix_to_model_space(self, leftover_19d, torch_batch):
        """絶対 19D leftover → model action space の prefix。

        - `abs`: model 出力は絶対関節角そのもの (正規化なし) なので、waist pad を
          落とすだけの恒等変換。
        - `rel`: model 出力は正規化 Δq。`reconstruct_arms_abs_from_dq_norm`
          (`arms_current + cumsum`) の逆で、**現 tick の `arms_current` を起点に
          差分を取り直す**のが re-anchor そのもの。学習側と同じ
          `compute_teacher_arms_dq` / `normalize_arms_dq` を使い、正規化の解釈が
          学習とずれないようにする。
        """
        # lazy: env-isolated dependencies (torch / model.ramen_ori は model env only)
        import torch

        from model.ramen_ori.relative_action import (
            compute_teacher_arms_dq,
            normalize_arms_dq,
        )

        chunk = torch.from_numpy(np.ascontiguousarray(leftover_19d)).to(self._device)
        if self._model_action_dim == 16:
            chunk = chunk[:, 3:]          # waist pad を落とす
            arms_slice = slice(0, 14)
        elif self._model_action_dim == 19:
            arms_slice = slice(3, 17)
        else:
            raise RuntimeError(
                f"unexpected model action_dim {self._model_action_dim}, "
                "expected 16 (Phase K) or 19 (Phase 1/2)"
            )
        if self.cfg.action_space == "rel":
            arms_current = torch_batch["state"][0, 3:17]      # (14,) 現 tick の state
            dq = compute_teacher_arms_dq(chunk[:, arms_slice], arms_current)
            chunk = chunk.clone()
            chunk[:, arms_slice] = normalize_arms_dq(
                dq,
                self._model._relative_arms_mean,
                self._model._relative_arms_std,
            )
        return chunk

    def _init_pipeline(self, seed_chunk_19d: np.ndarray) -> None:
        """初回 sync seed chunk から AsyncActionChunkPipeline を build。"""
        from inference.desktop.lower_policy.async_replanning import (
            AsyncActionChunkPipeline,
            family_replanning_schedule,
        )

        replan_after, max_age = family_replanning_schedule(
            self.cfg.replan_family, self.cfg.execution_steps
        )
        chunk_len = int(seed_chunk_19d.shape[0])
        if (
            self.cfg.temporal_lambda is not None
            and self.cfg.execution_steps >= chunk_len
        ):
            # 重なり = chunk_len - execution_steps。0 なら候補が常に 1 個で
            # blend が一度も起きない (GR00T で実測した落とし穴と同じ)。
            print(
                f"[ramen_ori] WARNING: execution_steps={self.cfg.execution_steps} "
                f">= chunk_len={chunk_len}: consecutive chunks never overlap, so "
                f"the temporal ensemble (temporal_lambda={self.cfg.temporal_lambda}) "
                "can never blend.",
                file=sys.stderr,
            )
        self._pipeline_lead_steps = self.cfg.execution_steps - replan_after
        self._pipeline_max_age_s = max_age
        self._pipeline = AsyncActionChunkPipeline(
            initial_actions=seed_chunk_19d.astype(np.float64),
            execution_steps=self.cfg.execution_steps,
            replan_after_steps=replan_after,
            max_prediction_age_s=max_age,
            thread_name_prefix="ramen-ori-replan",
        )
        self._pending_submit_step = None

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

        # ---- 2. Ckpt path resolve + torch.load (Phase C で先出し) ----
        # rel action の場合は state_dict の buffer `_relative_arms_mean/std` を
        # 抽出して _NnModule init に渡す必要 = ckpt を model 生成前に load する。
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
            # HF repo: latest step ckpt auto pick or filename 指定。
            # Issue #134: `@sha` pin syntax (Issue #132 Phase C+ で導入) を revision
            # parameter に split。GR00T 側 `groot.py:355-374` と同 pattern、HF API の
            # repo_id は `@` を含めない (HFValidationError 回避)。
            if "@" in ckpt_ref:
                repo_id, revision = ckpt_ref.rsplit("@", 1)
                if not repo_id or not revision:
                    raise ValueError(
                        f"ckpt_ref must be a local file/dir, HF repo, or "
                        f"HF-repo@revision; got {ckpt_ref!r}"
                    )
            else:
                repo_id, revision = ckpt_ref, None
            api = HfApi()
            files = api.list_repo_files(repo_id=repo_id, revision=revision)
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
            ckpt_path = hf_hub_download(
                repo_id=repo_id, filename=ckpt_filename, revision=revision
            )

        import sys as _sys

        # ckpt は **CPU に**読む。file には optimizer の状態も入っていて (c32 の実測で
        # file 3.3 GB)、GPU に読むとその分がそのまま VRAM を占める。重みは
        # load_state_dict が CPU → GPU に写すので、GPU に置くのは model だけでよい
        # (Issue #141 P8 の確認で、読み込み時の GPU の山が 4.36 → 1.49 GiB)。
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # ---- 2b. 約束つき ckpt (Issue #141 の再学習) は `ckpt["cfg"]` から組み立てる ----
        # Phase K までの ckpt には `contract` が無いので、従来の経路 (base.yaml +
        # slot の hydra_overrides) をそのまま通る。
        contract = ckpt.get("contract")
        if contract is not None:
            return cls._from_contract_ckpt(
                cfg, ckpt=ckpt, ckpt_path=ckpt_path, contract=contract, use_ema=use_ema
            )

        # ---- 3. Phase C: rel action の relative_stats を ckpt buffer から抽出 ----
        # buffer は param ではないので EMA shadow_params に含まれない = 常に
        # model_state_dict から取得。abs slot (default) では skip。
        # Issue #134 Phase H-3 / #137: torch.compile 経由の ckpt は key に
        # "_orig_mod." prefix が付く (Phase K R-6 全 6 run が該当)。#137 で
        # `strip_compile_prefix` に一本化し、prefix 有無どちらも accept する。
        relative_stats = None
        if cfg.action_space == "rel":
            msd = ckpt.get("model_state_dict")
            if not isinstance(msd, dict):
                raise RuntimeError(
                    f"action_space='rel' expects model_state_dict dict in ckpt "
                    f"{ckpt_path!r}, got {type(msd).__name__}."
                )
            msd = strip_compile_prefix(msd)
            _mean_key = "_relative_arms_mean"
            _std_key = "_relative_arms_std"
            if _mean_key not in msd or _std_key not in msd:
                raise RuntimeError(
                    f"action_space='rel' expects buffers '_relative_arms_mean' / "
                    f"'_relative_arms_std' (with optional '_orig_mod.' prefix from "
                    f"torch.compile) in ckpt {ckpt_path!r} model_state_dict, "
                    "but they were not found. This ckpt was likely trained with "
                    "abs action space — verify action_space matches training recipe."
                )
            _mean = msd[_mean_key]
            _std = msd[_std_key]
            if tuple(_mean.shape) != (14,) or tuple(_std.shape) != (14,):
                raise RuntimeError(
                    f"relative_stats shape mismatch: mean={tuple(_mean.shape)}, "
                    f"std={tuple(_std.shape)}, expected (14,) for both"
                )
            relative_stats = {"mean": _mean, "std": _std}

        # ---- 4. Model instantiate (train.py:_build_model と同手順) ----
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
        # 【Issue #137】RTC の overlap は chunk 長を超えられない。chunk_len は
        # hydra config を読むまで確定しないので config_loader では検証できないが、
        # ここは重み load の前なので設定ミスは即座に落ちる。
        validate_rtc_against_chunk_len(cfg, int(hydra_cfg.model.chunk_len))
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
            # Phase C: rel action space (RAMEN-Ori Phase K Run 3/5 対応)
            use_relative_action=(cfg.action_space == "rel"),
            relative_stats=relative_stats,
        )
        model = model.to(cfg.device)

        # ---- 5. state_dict load (ckpt は §2 で load 済) ----
        # Issue #137: `torch.compile` 経由の ckpt は全 key に `_orig_mod.` prefix が
        # 付く。prefix 付きのままだと 1 key も一致せずモデルが初期化のまま残るので、
        # EMA / raw 双方を load 前に正規化する。
        raw_state = strip_compile_prefix(ckpt["model_state_dict"])
        if use_ema and "ema_state_dict" in ckpt:
            ckpt["ema_state_dict"] = strip_compile_prefix(ckpt["ema_state_dict"])
        # Issue #137: EMA shadow が未更新な ckpt (Phase K R-6 全 6 run) は raw に
        # フォールバックする。fail-fast にしないのは、既存 ckpt を再 upload せずに
        # 正しい重みで評価できるようにするため。
        if (
            use_ema
            and isinstance(ckpt.get("ema_state_dict"), dict)
            and "shadow_params" not in ckpt["ema_state_dict"]
            and ema_state_dict_is_stale(ckpt["ema_state_dict"], raw_state)
        ):
            print(
                f"[RamenOriPolicy] WARNING: ema_state_dict in {ckpt_path!r} is at its "
                "initialization (EMA was never updated — trained with "
                "speedup.torch_compile=true before the Issue #137 fix). "
                "Falling back to model_state_dict (raw training weights).",
                file=_sys.stderr,
            )
            use_ema = False
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
            missing, unexpected = model.load_state_dict(raw_state, strict=False)
            _log_state_dict_load("model_state_dict", missing, unexpected, ckpt_path)
            _validate_architecture_state_dict_load(missing, unexpected, ckpt_path)
        del raw_state

        del ckpt  # free CPU memory
        model.eval()

        # dtype 変換 (bf16 / fp16 指定時)
        if cfg.dtype == "bf16":
            model = model.to(torch.bfloat16)
        elif cfg.dtype == "fp16":
            model = model.to(torch.float16)

        return cls(cfg=cfg, _model=model, _device=cfg.device)

    @classmethod
    def _from_contract_ckpt(
        cls,
        cfg: PolicyConfig,
        *,
        ckpt: dict,
        ckpt_path: str,
        contract: dict,
        use_ema: bool,
    ) -> "RamenOriPolicy":
        """約束 (`ckpt["contract"]`) つきの ckpt を読む (Issue #141 P8-1 / P8-2 / P8-3)。

        - 組み立ては学習と同じ関数 (`model.ramen_ori.build.build_model`) に
          `ckpt["cfg"]` をそのまま渡す。推論側で構造を組み直さない
        - 重みは raw の `model_state_dict` を先に読む (正規化と FK の統計は buffer なので
          raw にしかない)。その上から EMA の学習 param を重ねる
        - 設定と食い違っていれば、重みを読む前に止める
        """
        import sys as _sys

        from omegaconf import DictConfig, OmegaConf

        validate_contract(contract, cfg)
        if cfg.hydra_overrides:
            raise RuntimeError(
                "hydra_overrides cannot be used with a contract ckpt: the model is "
                f"built from ckpt['cfg'] (got {list(cfg.hydra_overrides)})"
            )
        if cfg.dtype != "fp32":
            # 正規化の統計まで bf16 になり、入力の縮尺が変わる (Issue #141 P8-7)。
            raise RuntimeError(
                f"contract ckpts must run in fp32 (slot dtype={cfg.dtype!r}); "
                "the normalization statistics live in buffers and would be cast too"
            )

        hydra_cfg = ckpt.get("cfg")
        if hydra_cfg is None:
            raise RuntimeError(
                f"ckpt {ckpt_path!r} has a contract but no 'cfg'; cannot rebuild the model"
            )
        if not isinstance(hydra_cfg, DictConfig):
            hydra_cfg = OmegaConf.create(hydra_cfg)
        validate_rtc_against_chunk_len(cfg, int(hydra_cfg.model.chunk_len))

        from model.ramen_ori.build import build_model

        model = build_model(hydra_cfg, cfg.device)

        raw_state = strip_compile_prefix(ckpt["model_state_dict"])
        missing, unexpected = model.load_state_dict(raw_state, strict=False)
        _log_state_dict_load("model_state_dict", missing, unexpected, ckpt_path)
        _validate_architecture_state_dict_load(missing, unexpected, ckpt_path)

        ema_state = ckpt.get("ema_state_dict")
        if use_ema and isinstance(ema_state, dict) and "shadow_params" in ema_state:
            shadow = ema_state["shadow_params"]
            n_params = sum(1 for _ in model.named_parameters())
            if len(shadow) != n_params:
                raise RuntimeError(
                    f"EMA shadow_params length mismatch: shadow={len(shadow)}, "
                    f"model.named_parameters()={n_params}. ckpt {ckpt_path!r} "
                    "may be from a different model architecture."
                )
            for (_name, param), value in zip(model.named_parameters(), shadow):
                param.data.copy_(value)
            print(
                f"[RamenOriPolicy] loaded raw weights (buffers) + EMA shadow_params "
                f"({n_params} params)",
                file=_sys.stderr,
            )
        elif use_ema:
            print(
                f"[RamenOriPolicy] WARNING: no EMA shadow_params in {ckpt_path!r}; "
                "using the raw training weights",
                file=_sys.stderr,
            )
        del raw_state, ckpt
        model.eval()
        return cls(cfg=cfg, _model=model, _device=cfg.device, _contract=contract)

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
        # 【Issue #137】warmup は dummy observation で、初回 JIT (~500ms-2s) も
        # 含む。これを RTC の遅延推定 (window 内 max) に混ぜると d が過大評価され、
        # overlap 全域が凍結されて応答性を失う (実測: 966ms の warmup latency で
        # frozen が overlap と同値 8 になった)。dummy chunk が RTC prefix や
        # temporal ensemble に流れ込むのも防ぐため、warmup 由来の状態は全て捨てる。
        self.reset()

    def _sync_predict_chunk_19d(
        self, obs: Observation, *, rtc_step: int | None = None
    ) -> tuple[np.ndarray, float]:
        """`_in_process_predict_chunk_19d` を直列化して呼ぶ。

        predict 自体は stateless なので GR00T のような correctness 上の必要は
        無いが、async worker と sync_fallback が同時に forward すると VRAM
        ピークが倍になる。構造も GR00T と揃える。
        """
        with self._inference_lock:
            return self._in_process_predict_chunk_19d(obs, rtc_step=rtc_step)

    def _in_process_predict_chunk_19d(
        self, obs: Observation, *, rtc_step: int | None = None
    ) -> tuple[np.ndarray, float]:
        """1 tick observation → 19D absolute action chunk + latency [ms]。

        Pipeline:
            1. `build_batch_dict(obs, mode)` → numpy dict
            2. numpy → torch tensor、batch 次元追加 (B=1)
            3. `self._model.predict_action(batch)` → (1, 16, 19)
            4. → PolicyAction wrap で return。**denorm はしない** (Issue #141 P8-7):
               Phase K までの ckpt は action を正規化せずに学習しており、再学習の
               ckpt は正規化を model の中に持つので、どちらも出力は実単位。
        """
        if self._model is None:
            raise RuntimeError(
                "RamenOriPolicy is not loaded. Call from_ckpt() first."
            )
        # lazy: env-isolated dependencies (torch は model env only)
        import torch

        t0 = time.monotonic_ns()
        raw_batch = build_batch_dict(
            obs,
            mode=self.cfg.mode,
            cams=tuple(self.cfg.cams),
            overlay_jpeg_subsampling=self.cfg.overlay_jpeg_subsampling,
            with_prev_frames=self._model_uses_prev_frames(),
        )
        if self._memory_enabled():
            raw_batch["memory"] = self._memory_vector(obs)

        # numpy → torch tensor、batch 次元追加 (B=1)
        torch_batch: dict = {}
        for key, val in raw_batch.items():
            t = torch.from_numpy(np.ascontiguousarray(val)) if isinstance(val, np.ndarray) else torch.tensor(val)
            torch_batch[key] = t.unsqueeze(0).to(self._device)

        with torch.inference_mode():
            rtc_kwargs, rtc_metadata = self._build_rtc_prefix(torch_batch, rtc_step)
            action_chunk_t = self._model.predict_action(torch_batch, **rtc_kwargs)
        # model の action 空間を確定 (prefix を model space に戻すのに要る)
        self._model_chunk_len = int(action_chunk_t.shape[-2])
        self._model_action_dim = int(action_chunk_t.shape[-1])

        # ---- Phase C: rel action space の abs 復元 (Phase K Run 3/5 対応) ----
        # model 出力の arms 14 dim は normalized Δq、hand 2 dim は abs pass-through。
        # arms を `arms_current + cumsum(dq_norm*std + mean)` で abs に戻す。
        # 学習側 action_dim = 16 (Phase K、waist 除外) or 19 (Phase 1/2、waist 込み) で
        # arms slice 位置が異なる。
        if self.cfg.action_space == "rel":
            # lazy: env-isolated dependency (relative_action は model env only)
            from model.ramen_ori.relative_action import (
                reconstruct_arms_abs_from_dq_norm,
            )
            arms_current_t = torch_batch["state"][:, 3:17]  # (B=1, 14)
            mean = self._model._relative_arms_mean  # (14,) tensor (buffer)
            std = self._model._relative_arms_std    # (14,)
            if action_chunk_t.shape[-1] == 16:
                arms_slice = slice(0, 14)  # [arms 14, hand 2]
            elif action_chunk_t.shape[-1] == 19:
                arms_slice = slice(3, 17)  # [waist 3, arms 14, hand 2]
            else:
                raise RuntimeError(
                    "rel reconstruction expects action_dim 16 or 19, "
                    f"got {action_chunk_t.shape[-1]}"
                )
            dq_norm = action_chunk_t[..., arms_slice]  # (B, chunk, 14)
            arms_abs = reconstruct_arms_abs_from_dq_norm(
                dq_norm, arms_current_t, mean, std
            )
            action_chunk_t = action_chunk_t.clone()  # inference_mode でも clone 可
            action_chunk_t[..., arms_slice] = arms_abs

        action_np = action_chunk_t[0].detach().cpu().numpy().astype(np.float32, copy=False)

        # ---- Phase C: 16D → 19D pad (vla_skill contract に合わせる) ----
        # 学習側 (Phase K) が waist 3D を loss 圏外化して action_dim=16 にした結果、
        # inference 出力も 16D。vla_skill は 19D chunk を expect するので pad 必須。
        # skill_config で rotate_table_base.dispatch_waist=false 運用のため、pad 値は
        # zeros で OK (waist chunk は vla_skill 側で drop、snap リスクなし)。
        # Phase 1/2 5 variant (19D 学習) は if 分岐入らず現状挙動維持 (backward compat)。
        if action_np.shape[1] == 16:
            waist_pad = np.zeros((action_np.shape[0], 3), dtype=np.float32)
            action_np = np.concatenate([waist_pad, action_np], axis=1)  # (chunk, 19)
        elif action_np.shape[1] != 19:
            raise RuntimeError(
                f"unexpected action_dim {action_np.shape[1]}, "
                "expected 16 (Phase K) or 19 (Phase 1/2)"
            )

        latency_ms = (time.monotonic_ns() - t0) / 1e6
        if self._rtc_leftover is not None and rtc_step is not None:
            self._rtc_leftover.store(action_np, origin_step=int(rtc_step))
        self._rtc_delay.add(latency_ms / 1000.0)
        self._last_rtc_metadata = rtc_metadata
        return action_np, latency_ms

    def predict(self, obs: Observation) -> PolicyAction:
        """1 tick observation → PolicyAction。

        - **cfg.replan_family=None (既定)**: 毎 tick 同期推論して chunk 全体を
          返す従来動作。VlaSkill が row 0 を使う。挙動は Phase C 以前と同一。
        - **cfg.replan_family 設定時**: async pipeline + temporal ensemble。
          GR00T と同じく policy 内部で chunk を合流させ、VlaSkill には blend 済み
          1 行だけ返す (`EXECUTION_HORIZON = 1`)。command loop は毎 tick
          non-blocking になり、学習と同じ 30Hz cadence を狙える。

        学習データは 30fps だが、同期経路は毎 tick 推論 (実測 50.8ms) が cadence を
        律速して 16.2Hz しか出ていない。async 化はそのズレを埋めるためのもの。
        """
        # RTC の d は latency ÷ tick 周期。周期は policy ごとに違うので実測する。
        tick_now_ns = time.monotonic_ns()
        if self._rtc_last_tick_ns is not None:
            delta_s = (tick_now_ns - self._rtc_last_tick_ns) / 1e9
            if 0.0 < delta_s < 1.0:  # 一時停止や skill 遷移の穴は捨てる
                self._rtc_tick_deltas.append(delta_s)
        self._rtc_last_tick_ns = tick_now_ns

        base_metadata = {
            "mode": self.cfg.mode,
            "action_space": self.cfg.action_space,
            "skill_id": obs.skill_id,
        }
        if self.cfg.replan_family is None:
            chunk_19d, latency_ms = self._sync_predict_chunk_19d(
                obs, rtc_step=self._current_step
            )
            self._current_step += 1
            return PolicyAction(
                action_chunk=chunk_19d,
                latency_ms=latency_ms,
                metadata={
                    **base_metadata,
                    "chunk_len": int(chunk_19d.shape[0]),
                    "action_dim": int(chunk_19d.shape[1]),
                    "chunk_source": "sync",
                    "predict_latency_ms": float(latency_ms),
                    **self._last_rtc_metadata,
                },
            )

        self._last_seen_obs = obs
        latency_ms = 0.0
        pipeline_index: int | None = None
        if self._pipeline is None:
            seed_19d, latency_ms = self._sync_predict_chunk_19d(
                obs, rtc_step=self._current_step
            )
            self._ensembler.add_chunk(
                origin_step=self._current_step, absolute_targets=seed_19d
            )
            self._init_pipeline(seed_19d)
            chunk_source = "sync"
        else:
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
            _pipeline_action, pipeline_index = self._pipeline.next_action()
            if self._pipeline.wants_prediction and self._pending_submit_step is None:
                obs_snapshot = obs
                submit_step = self._current_step

                def predictor():
                    chunk_19d, ms = self._sync_predict_chunk_19d(
                        obs_snapshot, rtc_step=submit_step
                    )
                    return (
                        chunk_19d.astype(np.float64),
                        float(ms),
                        {"submit_step": submit_step},
                    )

                self._pipeline.submit(predictor, anchor_generation=(submit_step,))
                self._pending_submit_step = submit_step
            if self._ensembler.candidate_count(self._current_step) == 0:
                # inference 完了遅れ + seed chunk 使い切り。hard stall を避ける
                # ため同期推論に落とす (GR00T と同じ fallback)。
                chunk_19d, latency_ms = self._sync_predict_chunk_19d(
                    obs, rtc_step=self._current_step
                )
                self._ensembler.add_chunk(
                    origin_step=self._current_step, absolute_targets=chunk_19d
                )
                chunk_source = "sync_fallback"

        blended_target = self._ensembler.target(step=self._current_step)
        candidate_count = self._ensembler.candidate_count(self._current_step)
        self._current_step += 1
        return PolicyAction(
            action_chunk=blended_target[None, :].astype(np.float32, copy=False),
            latency_ms=latency_ms,
            metadata={
                **base_metadata,
                "chunk_len": 1,
                "action_dim": ACTION_DIM,
                "blended_from_n_candidates": int(candidate_count),
                "temporal_lambda": self.cfg.temporal_lambda,
                "replan_family": self.cfg.replan_family,
                "chunk_source": chunk_source,
                "pipeline_index": pipeline_index,
                "pending_submit_step": self._pending_submit_step,
                "async_deadline_miss_ticks": int(self._pipeline.deadline_miss_ticks),
                "async_stale_discard_count": int(self._pipeline.stale_discard_count),
                "async_last_stale_discard_age_ms": (
                    self._pipeline.last_stale_discard_age_ms
                ),
                "predict_latency_ms": float(latency_ms),
                **self._last_rtc_metadata,
            },
        )

    def close(self) -> None:
        """GPU memory 解放 (long-running orchestrator の graceful shutdown)。"""
        # 【Issue #137】async pipeline の daemon thread が生きたまま model を
        # 落とすと、thread 側が保持する CUDA tensor の解放と競合して
        # プロセス終了時に abort する (実測: 実 ckpt smoke で core dump)。
        # GR00T と同じく bounded close で先に畳む。
        if self._pipeline is not None:
            self._pipeline.close(timeout_s=0.5)
            self._pipeline = None
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
            frames_bgr={
                cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in self.cfg.cams
            },
            frames_bgr_prev=None,
            state=np.zeros(STATE_DIM, dtype=np.float32),
            skill_id=SKILL_MOVE_TABLE_BASE,
            language=None,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )
