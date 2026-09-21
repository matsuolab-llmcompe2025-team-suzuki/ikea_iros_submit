"""C-11 (Selective overlay) 描画用 OverlayRenderer (Issue #122 D-3、Issue #129 Phase D 2026-08-31)。

# 2 経路 (共存)

1. **Hook 経路** (Issue #122 D-3、GR00T training + `obb_overlay_setup.py` で継続使用):
   JPG frame cache patch の post-decode hook として `OverlayRenderer.__call__` を register。
   JPG cv2.imread 直後 (uint8 HWC BGR、raw 640×480) の image に対して OBB rectangle を
   描画、その後 LeRobot の Resize(224) が anti-alias を効かせて滑らかな線が model 入力に届く。

2. **Inline 経路** (Issue #129 Phase D 2026-08-31、RAMEN-Ori dataloader が使用):
   dataloader が `_transform_item` 内で明示的に `draw_overlay_on_tensor(...)` を呼び、
   photometric aug 後 / Normalize 前の tensor に overlay を描画する。
   目的: aug (ColorJitter / RandomErasing) が overlay を破壊しない正しい順序を実現。
   Trade-off: 224×224 tensor 段階で描画 → hook 経路より線が chunky、model 学習への影響は小。

hook 経路は GR00T training と inference palette 参照で継続、`OverlayRenderer.__call__`
は draw_overlay_on_tensor と同じ core 描画 kernel を呼ぶ形に refactor 済 (DRY、
cv2.polylines 実装を単一化)。

# 責務 (共通 core)
- ObbPrecomputedCache.lookup_mp4_frame or 直接 tensor で該当 frame の top_K det 取得
- conf_threshold + class_filter 適用
- cv2.polylines で色分け枠線描画 (fill/label なし)
- cache に無い cam (wrist) → 描画 skip (raw image をそのまま return)

# 色 palette (BGR, cv2 convention)
    workspace   = gray    (128, 128, 128)
    leg         = green   (  0, 255,   0)
    leg_tip     = yellow  (  0, 255, 255)  ← cyan から変更 (leg green と近似だったため)
    hole        = red     (  0,   0, 255)
    table_top   = blue    (255,   0,   0)
    hand_right  = orange  (  0, 128, 255)
    hand_left   = magenta (255,   0, 255)

# class_filter の運用 (v1 = global static)
- None → 全 class 描画 (default)
- set of class_id → 該当 class のみ描画
- v1 は skill-conditional dispatch はしない (task_index を hook context 内で解決する
  仕組みが要、YAGNI で skip)。効果測定で必要になったら別 issue で拡張。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    import torch  # inline 経路の tensor helper 用

    from model.ramen_ori.obb_cache import ObbPrecomputedCache


# BGR (cv2 convention)、overlay_style_preview.py の palette と一致
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (128, 128, 128),  # workspace  = gray
    1: (0, 255, 0),      # leg        = green
    2: (0, 255, 255),    # leg_tip    = yellow
    3: (0, 0, 255),      # hole       = red
    4: (255, 0, 0),      # table_top  = blue
    5: (0, 128, 255),    # hand_right = orange
    6: (255, 0, 255),    # hand_left  = magenta
}


def _derive_cam_and_mp4_from_video_path(video_path: Path) -> tuple[str, str] | None:
    """`<root>/videos/<cam_key>/chunk-XXX/file-YYY.mp4` から (cam_key, "chunk-XXX/file-YYY") 抽出。

    JPG cache patch の `_derive_cache_dir_from_path` と同じ path structure 前提。
    形式外なら None (overlay skip)。
    """
    parts = video_path.parts
    if len(parts) < 4 or parts[-4] != "videos":
        return None
    cam_key = parts[-3]
    chunk = parts[-2]
    file_stem = video_path.stem
    return cam_key, f"{chunk}/{file_stem}"


class OverlayRenderer:
    """post-decode hook として register する OBB overlay 描画器。

    Args:
        cache: ObbPrecomputedCache (D-2 で既に build 済 instance を共有)。
            cache.covered_cameras に含まれる cam のみ描画、それ以外 (wrist) は skip。
        conf_threshold: この conf 未満の det は描画しない (default 0.30)。
        line_thickness: cv2.polylines の thickness (default 2)。
        class_filter: None なら全 class 描画、set なら該当 class_id のみ描画。
        class_colors_bgr: color palette override (default = module 定数)。
    """

    def __init__(
        self,
        cache: "ObbPrecomputedCache",
        conf_threshold: float = 0.30,
        line_thickness: int = 2,
        class_filter: set[int] | None = None,
        class_colors_bgr: dict[int, tuple[int, int, int]] | None = None,
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(f"conf_threshold must be in [0,1], got {conf_threshold}")
        if line_thickness < 1:
            raise ValueError(f"line_thickness must be >= 1, got {line_thickness}")
        self.cache = cache
        self.conf_threshold = float(conf_threshold)
        self.line_thickness = int(line_thickness)
        self.class_filter = set(class_filter) if class_filter is not None else None
        self.class_colors_bgr = dict(class_colors_bgr or CLASS_COLORS_BGR)

    def __call__(
        self, img_bgr: np.ndarray, video_path: Path, frame_idx: int
    ) -> np.ndarray:
        """Hook 経路 (Issue #122 D-3): post-decode hook signature。

        JPG cv2.imread 直後の uint8 BGR image に対して OBB を in-place で描画。
        GR00T training + `obb_overlay_setup.py` で継続使用。Phase D 以降、内部
        描画は `_draw_polylines_bgr` 経路に統一 (`draw_overlay_on_tensor` と共通)。

        Return は img_bgr 自身 (or 描画 skip 時は そのまま return)。
        """
        parsed = _derive_cam_and_mp4_from_video_path(Path(video_path))
        if parsed is None:
            return img_bgr
        cam_key, mp4_relative = parsed

        per = self.cache.lookup_mp4_frame(cam_key, mp4_relative, frame_idx)
        if per is None:
            # cache に無い cam (wrist) or frame 範囲外 → 描画 skip
            return img_bgr

        _draw_polylines_bgr(
            img_bgr=img_bgr,
            verts=per["verts"],
            conf=per["conf"],
            class_id=per["class_id"],
            valid=per["valid"],
            conf_threshold=self.conf_threshold,
            line_thickness=self.line_thickness,
            class_filter=self.class_filter,
            class_colors_bgr=self.class_colors_bgr,
        )
        return img_bgr


# ---------------------------------------------------------------------------
# Shared drawing kernel (hook 経路 + inline 経路 共通)
# ---------------------------------------------------------------------------


def _draw_polylines_bgr(
    img_bgr: np.ndarray,
    verts: np.ndarray,
    conf: np.ndarray,
    class_id: np.ndarray,
    valid: np.ndarray,
    conf_threshold: float,
    line_thickness: int,
    class_filter: set[int] | None,
    class_colors_bgr: dict[int, tuple[int, int, int]],
) -> None:
    """cv2.polylines で BGR image に OBB rectangle を in-place 描画 (低レベル kernel)。

    Args:
        img_bgr: (H, W, 3) uint8 BGR。in-place 変更。
        verts:   (top_K, 8) float、normalized [0,1] xyxyxyxy
        conf:    (top_K,) float
        class_id:(top_K,) int
        valid:   (top_K,) bool
        conf_threshold: この conf 未満は skip
        line_thickness: cv2.polylines thickness
        class_filter:   None なら全 class 描画、set なら該当のみ
        class_colors_bgr: class_id → BGR color 辞書

    Note:
        - lazy `import cv2` (default env に無い可能性、training / runtime env 前提)
        - verts / conf / class_id / valid は numpy 想定、torch tensor は事前に .numpy() 要
    """
    import cv2  # noqa: PLC0415

    H, W = img_bgr.shape[:2]
    scale = np.array([W, H], dtype=np.float32)

    for k in range(len(valid)):
        if not bool(valid[k]):
            continue
        if float(conf[k]) < conf_threshold:
            continue
        cid = int(class_id[k])
        if class_filter is not None and cid not in class_filter:
            continue
        color = class_colors_bgr.get(cid, (255, 255, 255))
        v = np.asarray(verts[k]).reshape(4, 2) * scale
        pts = v.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            img_bgr,
            [pts],
            isClosed=True,
            color=color,
            thickness=line_thickness,
        )


def draw_overlay_on_tensor(
    img_rgb_float: "torch.Tensor",
    verts,
    conf,
    class_id,
    valid,
    conf_threshold: float = 0.30,
    line_thickness: int = 2,
    class_filter: set[int] | None = None,
    class_colors_bgr: dict[int, tuple[int, int, int]] | None = None,
) -> "torch.Tensor":
    """Inline 経路 (Issue #129 Phase D 2026-08-31): tensor image に OBB overlay を描画。

    Photometric aug 後 / Normalize 前の float [0,1] RGB tensor に対して cv2.polylines
    による overlay を描画し、float [0,1] RGB tensor で return。dataloader の
    `_transform_item` から photometric-aug → RandomErasing 済み tensor を受けて呼ぶ想定。

    Wrist cam (cache 未収録) は `valid` が全 False な OBB data を渡すことで自然に no-op
    (polylines が呼ばれない)、明示的な cam type 分岐は呼出側の責務。

    Args:
        img_rgb_float: (3, H, W) torch.Tensor float [0,1] RGB。gradient graph に居ない想定
                       (aug 後、model input 前の中間 stage、Normalize 前)
        verts / conf / class_id / valid: torch or numpy、`_draw_polylines_bgr` に渡る
        conf_threshold, line_thickness, class_filter: OverlayRenderer と同じ意味
        class_colors_bgr: None なら module CLASS_COLORS_BGR を使う

    Returns:
        (3, H, W) torch.Tensor float [0,1] RGB (overlay 描画済)
    """
    import torch  # noqa: PLC0415

    if img_rgb_float.dim() != 3 or img_rgb_float.shape[0] != 3:
        raise ValueError(
            f"img_rgb_float must be (3, H, W), got {tuple(img_rgb_float.shape)}"
        )

    # (3, H, W) float [0,1] RGB → (H, W, 3) uint8 BGR (cv2 前提)
    img_hwc_rgb = img_rgb_float.clamp(0.0, 1.0).permute(1, 2, 0).contiguous()
    img_hwc_uint8_rgb = (img_hwc_rgb * 255.0).to(torch.uint8).cpu().numpy()
    img_hwc_uint8_bgr = img_hwc_uint8_rgb[..., ::-1].copy()  # RGB→BGR (cv2 convention)

    # numpy 化 (torch tensor でも動くように)
    def _to_np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    _draw_polylines_bgr(
        img_bgr=img_hwc_uint8_bgr,
        verts=_to_np(verts),
        conf=_to_np(conf),
        class_id=_to_np(class_id),
        valid=_to_np(valid),
        conf_threshold=float(conf_threshold),
        line_thickness=int(line_thickness),
        class_filter=class_filter,
        class_colors_bgr=class_colors_bgr or CLASS_COLORS_BGR,
    )

    # (H, W, 3) uint8 BGR → (3, H, W) float [0,1] RGB
    result_rgb = img_hwc_uint8_bgr[..., ::-1].copy()  # BGR→RGB
    result_tensor = torch.from_numpy(result_rgb).to(
        dtype=img_rgb_float.dtype, device=img_rgb_float.device
    )
    result_tensor = result_tensor.permute(2, 0, 1).contiguous() / 255.0
    return result_tensor
