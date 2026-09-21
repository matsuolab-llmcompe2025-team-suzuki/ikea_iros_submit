"""YOLO-OBB overlay を VLM 入力に載せる (Issue #136、determinded.md D-7)。

`obs["cleaned"]` (YOLO-OBB のクリーンアップ済み検出) は orchestrator が既に
毎 tick 渡している。**新しい配線は要らない。** ここでやるのは、その検出を
使って画像に枠を描き、base64 JPEG に直すところだけ。

描画は `policies/ramen_ori.py:overlay_obb_on_frame` を再利用する
(GR00T overlay / RAMEN-Ori overlay で実運用中の実装、色・線幅・conf 閾値が
training 側の `OverlayRenderer` と一致している)。

参照画像にも**同じ描画**をかけること。現在画像と見た目を揃えないと、
VLM の強制二択が「関係の違い」ではなく「絵柄の違い」で決まってしまう。

cv2 は runtime env にしか無いので import はすべて関数内に閉じる。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from inference.desktop.pick_leg_hybrid.config import OverlayConfig
from inference.desktop.pick_leg_hybrid.vlm import encode_jpeg_base64


def draw_overlay(
    frame_bgr: np.ndarray,
    detections: Sequence,
    cfg: Optional[OverlayConfig] = None,
) -> np.ndarray:
    """OBB 枠を描いた **新しい** frame を返す (入力は破壊しない)。

    Args:
        frame_bgr: (H, W, 3) uint8 BGR。
        detections: `list[OBBDetection]`。空なら素通し (copy は返す)。
        cfg: overlay 設定。None なら既定。

    Returns:
        描画済みの (H, W, 3) uint8 BGR。`cfg.enabled=False` なら copy のみ。
    """
    cfg = cfg or OverlayConfig()
    out = np.array(frame_bgr, copy=True)
    if not cfg.enabled or not detections:
        return out
    # lazy: overlay_obb_on_frame は cv2 を使う (runtime env のみ)
    from inference.desktop.lower_policy.policies.ramen_ori import (
        overlay_obb_on_frame,
    )

    return overlay_obb_on_frame(
        out,
        detections=list(detections),
        conf_threshold=cfg.conf_threshold,
        line_thickness=cfg.line_thickness,
        class_filter=set(cfg.class_ids),
    )


def build_vlm_images(
    frames_bgr: Sequence[np.ndarray],
    detections_per_frame: Sequence[Sequence],
    *,
    overlay_cfg: Optional[OverlayConfig] = None,
    jpeg_quality: int = 85,
) -> list[str]:
    """[過去…, 現在] の frame 列 → overlay 済み base64 JPEG 列。

    Args:
        frames_bgr: 古い順に並べた frame。**最後が現在**。
        detections_per_frame: 各 frame に対応する検出列。長さは
            `frames_bgr` と一致させること。
        overlay_cfg: overlay 設定。
        jpeg_quality: JPEG 品質。

    Returns:
        base64 JPEG の list (入力と同じ順)。

    Raises:
        ValueError: frame と検出列の長さが合わない。
    """
    if len(frames_bgr) != len(detections_per_frame):
        raise ValueError(
            "frames_bgr and detections_per_frame must have the same length, "
            f"got {len(frames_bgr)} and {len(detections_per_frame)}"
        )
    return [
        encode_jpeg_base64(
            draw_overlay(frame, dets, overlay_cfg), quality=jpeg_quality
        )
        for frame, dets in zip(frames_bgr, detections_per_frame)
    ]


class FrameHistory:
    """直近 N 枚の frame と検出を保持する固定長バッファ。

    VLM には現在画像だけでなく過去数枚も渡す (D-9)。どれが現在かは
    **並び順で表す** (最後が現在)。質問文もその順を前提にしている。
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self._size = int(size)
        self._frames: list[np.ndarray] = []
        self._dets: list[Sequence] = []

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def size(self) -> int:
        return self._size

    def push(self, frame_bgr: np.ndarray, detections: Sequence = ()) -> None:
        """1 枚積む。溢れたら古い方から捨てる。"""
        # Camera adapters may reuse their receive buffer.  Keep an immutable
        # generation here so history never silently becomes N copies of the
        # newest frame before the VLM request is encoded.
        self._frames.append(np.array(frame_bgr, copy=True))
        self._dets.append(list(detections))
        if len(self._frames) > self._size:
            self._frames.pop(0)
            self._dets.pop(0)

    def clear(self) -> None:
        self._frames.clear()
        self._dets.clear()

    def as_vlm_images(
        self,
        *,
        overlay_cfg: Optional[OverlayConfig] = None,
        jpeg_quality: int = 85,
    ) -> list[str]:
        """保持中の frame を overlay 済み base64 JPEG 列にする。"""
        return build_vlm_images(
            self._frames,
            self._dets,
            overlay_cfg=overlay_cfg,
            jpeg_quality=jpeg_quality,
        )
