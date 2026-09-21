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

import sys
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


def resolve_yolo_ckpt_ref(ckpt_ref: str) -> Path:
    """`policy_config.yaml` の `yolo.ckpt_ref` (`repo@revision`) の重みを HF の cache に取り、path を返す。

    学習の overlay の焼き込み (`unified_frame_cache.yaml` の yolo) と同じ重みを推論で使うため (Issue #141 INF-10)。
    repo の中の `.pt` はちょうど 1 つである前提 (複数あれば止める)。token は huggingface_hub の通常の経路 (HF_TOKEN など)。
    """
    # lazy: huggingface_hub は runtime env と学習の env にだけある (default env の unit-test collection を通す)
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    repo_id, sep, revision = ckpt_ref.partition("@")
    if not sep or not repo_id or not revision:
        raise ValueError(f"yolo ckpt_ref must be 'repo@revision', got {ckpt_ref!r}")
    try:
        weights = [
            f for f in HfApi().list_repo_files(repo_id, revision=revision) if f.endswith(".pt")
        ]
    except Exception as exc:
        # HF に問い合わせできないとき (会場でネットが無い、HF の障害) でも、cache に
        # 落ちていれば起動できるようにする。cache も無ければ snapshot_download が上げる。
        print(f"[yolo] HF に問い合わせできないので cache から探す: {exc!r}", file=sys.stderr)
        cached = sorted(
            Path(
                snapshot_download(repo_id, revision=revision, local_files_only=True)
            ).glob("*.pt")
        )
        if len(cached) != 1:
            raise ValueError(
                f"{ckpt_ref}: cache の .pt は 1 つの前提だが {[w.name for w in cached]}"
            ) from exc
        return cached[0]
    if len(weights) != 1:
        raise ValueError(f"{ckpt_ref}: expected exactly one .pt file, found {weights}")
    return Path(hf_hub_download(repo_id, weights[0], revision=revision))


class YoloObbPerception:
    """YOLO-OBB weight を wrap し、frame → OBB list を返す。

    - weight は init 時 fail-fast (存在しなければ FileNotFoundError)
    - class name は `model.names` から動的取得 (train dataset 変更に自動追従)
    - inference は 1 frame ずつ (batch は YAGNI、iteration 用途に速度足りる)

    Args:
        weight_path: `.pt` file への path
        conf: minimum confidence threshold (default 0.25、predict_video_ep64.py と同じ)
        imgsz: 推論の入力解像度。None なら ultralytics の解決に任せる (重みに保存された
            値、無ければ 640)。学習 cache の焼き込みと揃えるため、推論では
            `policy_config.yaml` の `yolo.imgsz` を明示して渡す (Issue #141 INF-10)。
        device: `"cuda"` / `"cpu"` / None (None は ultralytics 自動選択)
    """

    def __init__(
        self,
        weight_path: Path | str,
        conf: float = 0.25,
        device: str | None = None,
        imgsz: int | None = None,
    ) -> None:
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"weight not found: {weight_path}")
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"conf must be in [0, 1], got {conf}")
        if imgsz is not None and (imgsz <= 0 or imgsz % 32 != 0):
            raise ValueError(f"imgsz must be a positive multiple of 32, got {imgsz}")

        self._weight_path = weight_path
        self._conf = conf
        self._device = device
        self._imgsz = imgsz
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
        if self._imgsz is not None:
            predict_kwargs["imgsz"] = self._imgsz
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
