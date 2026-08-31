"""YOLO-OBB inference wrapper。

Epic #43 / Issue #47。学習済 YOLO-OBB weight (default: `m_lowaug_v11b`) を load し、
BGR frame から OBB detection list を返す薄い wrapper。

`research/scripts/predict_video_ep64.py` の YOLO 呼び出し部分を再利用可能な形に
切り出したもの。座標は image size 非依存の normalized [0,1] で返す (overlap 計算
しやすく、overlay 時に image size 掛け戻す)。

`ultralytics` は `YoloObbPerception.__init__` 内で lazy import する (runtime env のみ
install)。default env から本 module を import しても OBBDetection dataclass は使える
(unit-test collection 用、CLAUDE.md の lazy import 方針)。

Usage:
    from inference.desktop.perception.yolo_obb import YoloObbPerception

    perception = YoloObbPerception("model/yolo_obb/runs/m_lowaug_v11b/weights/best.pt")
    dets = perception.predict(frame_bgr)  # list[OBBDetection]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OBBDetection:
    """1 個の OBB 検出結果。

    Attributes:
        class_id: 0..N-1、model.names のキー
        class_name: model.names[class_id] (例: "leg", "hole", "hand_left")
        confidence: [0, 1]
        verts: shape (4, 2) normalized xy in [0, 1]。時計回りまたは反時計回りの
            4 頂点 (ultralytics の xyxyxyxyn 準拠)。実 pixel に戻すときは
            `verts * np.array([W, H])`
    """

    class_id: int
    class_name: str
    confidence: float
    verts: np.ndarray


class YoloObbPerception:
    """YOLO-OBB weight を wrap し、frame → OBB list を返す。

    - weight は init 時 fail-fast (存在しなければ FileNotFoundError)
    - class name は `model.names` から動的取得 (train dataset 変更に自動追従)
    - inference は 1 frame ずつ (batch は YAGNI、iteration 用途に速度足りる)

    Args:
        weight_path: `.pt` file への path
        conf: minimum confidence threshold (default 0.25、predict_video_ep64.py と同じ)
        device: `"cuda"` / `"cpu"` / None (None は ultralytics 自動選択)
    """

    def __init__(
        self,
        weight_path: Path | str,
        conf: float = 0.25,
        device: str | None = None,
    ) -> None:
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"weight not found: {weight_path}")
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"conf must be in [0, 1], got {conf}")

        self._weight_path = weight_path
        self._conf = conf
        self._device = device
        # ultralytics は runtime env のみ install。YoloObbPerception を instantiate
        # する時だけ必要なので lazy import (default env の unit-test collection を通す)。
        from ultralytics import YOLO

        self._model = YOLO(str(weight_path))
        # model.names は dict[int, str] (ultralytics 慣例)
        self._class_names: dict[int, str] = dict(self._model.names)

    @property
    def class_names(self) -> dict[int, str]:
        """weight に紐付いた class_id → class_name map。"""
        return dict(self._class_names)

    def predict(self, frame_bgr: np.ndarray) -> list[OBBDetection]:
        """1 frame の BGR uint8 image から OBB list を返す。

        Args:
            frame_bgr: shape (H, W, 3) uint8 BGR image (cv2 慣例)

        Returns:
            検出 0 個なら空 list。検出順は ultralytics の出力順そのまま。
        """
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"frame_bgr must be (H, W, 3), got shape {frame_bgr.shape}"
            )

        predict_kwargs: dict = {"conf": self._conf, "verbose": False}
        if self._device is not None:
            predict_kwargs["device"] = self._device
        results = self._model(frame_bgr, **predict_kwargs)
        r = results[0]

        detections: list[OBBDetection] = []
        # 検出 0 の場合 r.obb は None ではなく空を持つ形になるが、防御的に None も許容
        if r.obb is None or r.obb.cls is None or len(r.obb.cls) == 0:
            return detections

        cls_arr = r.obb.cls.cpu().numpy()
        conf_arr = r.obb.conf.cpu().numpy()
        verts_arr = r.obb.xyxyxyxyn.cpu().numpy()  # shape (N, 4, 2) normalized

        for cls_id, confidence, verts in zip(cls_arr, conf_arr, verts_arr):
            cls_id_int = int(cls_id)
            detections.append(
                OBBDetection(
                    class_id=cls_id_int,
                    class_name=self._class_names.get(cls_id_int, f"cls_{cls_id_int}"),
                    confidence=float(confidence),
                    verts=np.asarray(verts, dtype=np.float32),
                )
            )
        return detections
