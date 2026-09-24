"""env-driven OBB overlay setup (Issue #122 D-4: GR00T + overlay 拡張)。

RAMEN-Ori (`model/ramen_ori/data_lerobot.py`) は `obb_source="overlay"` config で
自動 hook register するが、GR00T は既存の pre-trained model を fine-tune するため
model 側 config 経由での register が困難。そこで `lerobot_train_with_frame_cache`
wrapper (GR00T が学習 CLI として invoke するもの) の起動時に env vars を読んで
overlay hook を register する薄い setup layer を提供する。

# 使い方 (H100 での GR00T 学習例)

    export OBB_OVERLAY_ENABLE=true
    export OBB_PRECOMPUTED_ROOT=outputs/yolo_obb_cache
    # hash 未指定なら root 直下の唯一 hash dir を auto-pick、複数あれば error
    # export OBB_PRECOMPUTED_HASH=abc123def456

    SUBTASK=combined_task5_7 POLICY_TYPE=groot USE_MERGED_HF_REPO=true \\
      bash model/subtask_policy_training/scripts/train_lerobot.sh

# 動作

- `OBB_OVERLAY_ENABLE` が truthy でなければ何もせず None を return (backward compat)。
- そうでなければ:
  1. `clear_post_decode_hooks()` (単一 hook 保証、prior register を wipe)
  2. `ObbPrecomputedCache` init (manifest 検証、build_episode_mapping は skip
     — overlay hook は mp4 semantic 直接 lookup のみ使用、ep mapping 不要)
  3. `OverlayRenderer` init + `register_post_decode_hook`
  4. `OverlayRenderer` instance を return (呼出側で保持 → GC で dead reference 化防止)

# 対応 env vars

    OBB_OVERLAY_ENABLE       required、"true"/"1"/"yes" で有効化
    OBB_PRECOMPUTED_ROOT     required、precompute cache root (例 outputs/yolo_obb_cache)
    OBB_PRECOMPUTED_HASH     optional、None なら root 直下 hash dir auto-pick
    OBB_OVERLAY_CAMERAS      optional、default "observation.images.cam_0,observation.images.cam_1"
    OBB_OVERLAY_TOP_K        optional、default 4
    OBB_OVERLAY_NUM_CLASSES  optional、default 7
    OBB_OVERLAY_FPS          optional、default 30
    OBB_OVERLAY_CONF         optional、default 0.30 (preview mosaic + user 判定)
    OBB_OVERLAY_THICKNESS    optional、default 2  (preview mosaic + user 判定)
    OBB_OVERLAY_CLASS_FILTER optional、JSON list of int (例 "[3,4]" = hole+table_top のみ)、
                             未指定 = 全 class 描画

# 併存関係

- RAMEN-Ori side (`data_lerobot.py`) は自身の `obb_source="overlay"` で register。
  GR00T pipeline とは別プロセス想定なので registry 上競合しないが、setup は
  `clear_post_decode_hooks()` で明示 wipe → 「1 プロセス = 単一 overlay hook」
  の invariant を守る。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CAMERAS = "observation.images.cam_0,observation.images.cam_1"
DEFAULT_TOP_K = 4
DEFAULT_NUM_CLASSES = 7
DEFAULT_FPS = 30
DEFAULT_CONF = 0.30
DEFAULT_THICKNESS = 2

TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in TRUTHY


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return float(v)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _env_json_list_int(name: str) -> list[int] | None:
    """`OBB_OVERLAY_CLASS_FILTER` 用 JSON list-of-int 解析。未指定 or 空文字 → None。"""
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    parsed = json.loads(v)
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise ValueError(
            f"{name} must be a JSON list of int, got {v!r} (parsed={parsed!r})"
        )
    return parsed


def is_enabled() -> bool:
    """`OBB_OVERLAY_ENABLE` env を bool 化。"""
    return _env_bool("OBB_OVERLAY_ENABLE", False)


def setup_from_env() -> Any | None:
    """env vars を読んで overlay hook を register。

    - Enable flag false → 何もせず None (backward compat)
    - True の場合、cache + renderer 作成 + hook register し、renderer instance を return。
      呼出側 (wrapper) は返り値を local 変数に保持して GC で dead reference 化しないよう
      に注意 (module-level `_registered_renderer` にも保存)。

    Returns:
        OverlayRenderer instance if enabled, None otherwise.
    """
    if not is_enabled():
        return None

    root_str = _env_str("OBB_PRECOMPUTED_ROOT", "")
    if not root_str:
        raise ValueError(
            "OBB_OVERLAY_ENABLE=true requires OBB_PRECOMPUTED_ROOT to be set "
            "(precompute cache dir、例 outputs/yolo_obb_cache)"
        )
    root = Path(root_str)

    hash_str = os.environ.get("OBB_PRECOMPUTED_HASH", "")
    ckpt_hash = hash_str if hash_str else None

    cameras_csv = _env_str("OBB_OVERLAY_CAMERAS", DEFAULT_CAMERAS)
    cameras = [c.strip() for c in cameras_csv.split(",") if c.strip()]
    if not cameras:
        raise ValueError("OBB_OVERLAY_CAMERAS resolved to empty list")

    top_k = _env_int("OBB_OVERLAY_TOP_K", DEFAULT_TOP_K)
    num_classes = _env_int("OBB_OVERLAY_NUM_CLASSES", DEFAULT_NUM_CLASSES)
    fps = _env_int("OBB_OVERLAY_FPS", DEFAULT_FPS)
    conf = _env_float("OBB_OVERLAY_CONF", DEFAULT_CONF)
    thickness = _env_int("OBB_OVERLAY_THICKNESS", DEFAULT_THICKNESS)
    class_filter = _env_json_list_int("OBB_OVERLAY_CLASS_FILTER")

    # lazy import: 呼出側 (GR00T wrapper) の env 未 setup 時 import 失敗を避ける
    from model.ramen_ori.obb_cache import ObbPrecomputedCache  # noqa: PLC0415
    from model.ramen_ori.overlay import OverlayRenderer  # noqa: PLC0415
    from model.subtask_policy_training.scripts.lerobot_frame_cache_patch import (  # noqa: PLC0415
        clear_post_decode_hooks,
        register_post_decode_hook,
    )

    cache = ObbPrecomputedCache(
        root=root,
        ckpt_hash=ckpt_hash,
        camera_keys=cameras,
        top_K=top_k,
        num_classes=num_classes,
        fps=fps,
    )
    # ep mapping は overlay hook では未使用 (lookup_mp4_frame 直接、mp4 semantic)
    # → build_episode_mapping を skip する事で LeRobot meta 依存を切り離す。

    renderer = OverlayRenderer(
        cache=cache,
        conf_threshold=conf,
        line_thickness=thickness,
        class_filter=class_filter,
    )

    # 単一 hook 保証 (GR00T プロセスで overlay は 1 個のみ想定)
    clear_post_decode_hooks()
    register_post_decode_hook(renderer)

    # module-level に保持 (呼出側の local 変数に頼らず、GC で dead 化させない)
    global _registered_renderer
    _registered_renderer = renderer

    print(
        f"[obb_overlay_setup] registered OBB overlay hook: "
        f"root={root}, hash={cache.ckpt_hash}, cams={cameras}, "
        f"top_K={top_k}, conf={conf}, thickness={thickness}, "
        f"class_filter={class_filter}",
        flush=True,
    )
    return renderer


# module-level に生存させる (単一 GR00T プロセス想定、複数 setup 呼出は clear + 上書き)
_registered_renderer: Any | None = None
