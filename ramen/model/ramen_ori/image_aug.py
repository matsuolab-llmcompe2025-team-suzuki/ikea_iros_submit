"""RAMEN-Ori image aug pipeline module (Issue #129 Phase I-0-1、2026-09-01)。

`data_lerobot.py` から `ImageAugPipeline` + helper 群を切り出し。offline aug bake
precompute (`precompute_augmented_frame_cache_ramen_ori.py`) と training loop の両方
から同じ pipeline を呼べるようにする。

# 経緯
- Phase D (2026-08-31): overlay inline pipeline + aug 順序 refactor
- Phase G (2026-08-31): sharpness/JPEG/GaussianNoise/geometric head only + OBB coord warp
- Phase I-0-1 (2026-09-01): module 化、offline bake から import 可能に

# 契約
- `ImageAugPipeline.apply(img_2, cam_key, idx, is_train, overlay_callback=None)`
  → (imgs (2, 3, H, W) aug + Normalize 済、affine_matrix (2, 3) normalized [0,1] 空間)
- overlay_callback は per-frame で呼ばれ、overlay 描画済 tensor を return する契約 (Phase D)
- geometric aug は head only (`is_wrist(cam_key)` で判定)、wrist は identity affine
- `apply` の中身は `photometric` / `geometric_params` / `apply_geometric` の組合せ。offline
  bake はこの 3 つを直接呼び、token / overlay で photometric を 1 回だけ計算する (Issue #139)
- 既存 `data_lerobot.py` からは re-export で backward compat 維持

再 export された symbol は data_lerobot.py の top-level から import 可能:

    from model.ramen_ori.data_lerobot import ImageAugPipeline, IMAGENET_MEAN, ...
"""

from __future__ import annotations

import math
import zlib
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2 as tv2
from torchvision.transforms.v2 import functional as tvF


# ImageNet stats (LingBot / DINOv2 系 backbone 標準)。
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FloatJPEG(nn.Module):
    """v2.JPEG (uint8 前提) を float [0,1] tensor で扱う thin wrapper (Issue #129 Phase G、2026-08-31)。

    Real robot camera stream の JPEG artifact を training aug で模倣する目的。
    Float [0,1] → clamp → uint8 [0,255] → v2.JPEG → float [0,1] round trip。
    """

    def __init__(self, quality: tuple[int, int] | int) -> None:
        super().__init__()
        self.jpeg = tv2.JPEG(quality=quality)

    def forward(self, img_float: torch.Tensor) -> torch.Tensor:
        # (C, H, W) float [0,1] → uint8 → JPEG → float [0,1]
        img_u8 = (img_float.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        # JPEG encode/decode は CPU で行う (GPU bake でも同じ経路、CPU tensor では no-op)
        out_u8 = self.jpeg(img_u8.cpu()).to(img_float.device)
        return out_u8.float() / 255.0


def _identity_affine_matrix() -> torch.Tensor:
    """(2, 3) identity affine matrix (geometric skip / wrist cam の default)。"""
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)


def _make_head_affine_matrix(
    angle_deg: float, tx_frac: float, ty_frac: float
) -> torch.Tensor:
    """head geometric aug の (angle, translate) から normalized 座標 [0,1] 空間の (2, 3) affine を生成。

    OBB coord warp (Phase E `warp_obb_coord_under_affine`) に渡す用。image への実描画は
    torchvision v2 の `functional.affine(img, angle, translate=[tx_pix, ty_pix], ...)` を
    別途呼び (translate は pixel 単位)、両者は同 (angle, tx_frac, ty_frac) に基づく。
    """
    # obb_warp モジュールの遅延 import (循環回避)。
    from model.ramen_ori.obb_warp import make_rotation_translation_affine  # lazy

    return make_rotation_translation_affine(
        angle_rad=math.radians(angle_deg),
        tx=tx_frac,
        ty=ty_frac,
        center=(0.5, 0.5),
    )


class GeometricParams(NamedTuple):
    """1 sample × 1 cam の geometric aug params (prev / current で共通)。"""

    use: bool
    angle_deg: float
    tx_frac: float
    ty_frac: float
    affine_matrix: torch.Tensor  # (2, 3) normalized [0,1] 空間


def make_image_transform(img_size: int) -> transforms.Compose:
    """LeRobot の image_transforms 用、Resize のみ (Normalize + aug は ImageAugPipeline で cam type 別 dispatch)。

    Issue #122 で aug 導入時に Normalize を Dataset 側 (`ImageAugPipeline`) へ移動。
    LeRobot は uint8 (H, W, 3) を返すが LeRobotDataset 側で torch.Tensor float32 [0,1]
    (C, H, W) に変換済のことが多い。Resize のみここで実行し、Normalize は cam type
    (head vs wrist) 別 aug と合わせて後段で行う。
    """
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), antialias=True),
        ]
    )


def _cfg_to_dict(cfg: Any) -> dict:
    """OmegaConf DictConfig or dict を pure dict に変換 (test で dict 直渡し許容)。"""
    if hasattr(cfg, "to_container") or hasattr(cfg, "keys"):
        try:
            from omegaconf import OmegaConf

            if hasattr(cfg, "to_container"):
                return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
        except ImportError:
            pass
    return dict(cfg)


class ImageAugPipeline:
    """cam type (head vs wrist) 別 image aug + ImageNet normalize (Issue #122)。

    設計判断:
    - **prev + current の 2 frame に同 aug params を適用** (`fork_rng` で seed 固定)
      → delta encoder (D-3) が「aug 由来の変化」を学習しない
    - **head 強 / wrist 弱** ([[feedback_head_wrist_light_gradient]]): head の方が天井≈遠光源で
      環境照明変動を強く受ける、wrist は物体近接で局所照明のため弱い aug で十分
    - **順序**: Aug (color, blur) → Normalize → RandomErasing (torchvision standard)
    - **val は plain**: `is_train=False` で aug スキップ、Normalize のみ実行

    Args:
        aug_cfg: base.yaml の augmentation section (`enabled=false` で aug スキップ)
            - wrist_key_substring: cam key に含まれるとき wrist 扱い (default "wrist")
            - wrist_cam_keys: wrist 扱いする cam key の明示 list。source 名
              (`observation.images.cam_2` 等) は substring で判定できないため (Issue #139)
        mean/std: ImageNet stats (LingBot / DINOv2 系 backbone 標準)
    """

    def __init__(
        self,
        aug_cfg: dict | None,
        mean: list[float] = IMAGENET_MEAN,
        std: list[float] = IMAGENET_STD,
    ) -> None:
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.enabled = bool(aug_cfg and aug_cfg.get("enabled", False))
        self.wrist_key_substring = "wrist"
        self.wrist_cam_keys: frozenset[str] = frozenset()
        # Issue #129 Phase G: geometric aug 設定 (head only、config で override 可)
        self.geometric_enabled: bool = False
        self.geometric_degrees: tuple[float, float] = (-5.0, 5.0)
        self.geometric_translate: tuple[float, float] = (0.05, 0.05)
        if not self.enabled:
            return

        # OmegaConf DictConfig を dict 化して pop 事故防止
        cfg = _cfg_to_dict(aug_cfg)
        self.wrist_key_substring = cfg.get("wrist_key_substring", "wrist")
        self.wrist_cam_keys = frozenset(cfg.get("wrist_cam_keys") or [])

        head = cfg["head"]
        wrist = cfg["wrist"]
        erase = cfg["random_erasing"]

        self.head_color_jitter = transforms.ColorJitter(
            brightness=head["color_jitter"]["brightness"],
            contrast=head["color_jitter"]["contrast"],
            saturation=head["color_jitter"]["saturation"],
            hue=head["color_jitter"]["hue"],
        )
        self.wrist_color_jitter = transforms.ColorJitter(
            brightness=wrist["color_jitter"]["brightness"],
            contrast=wrist["color_jitter"]["contrast"],
            saturation=wrist["color_jitter"]["saturation"],
            hue=wrist["color_jitter"]["hue"],
        )
        self.head_gaussian_blur = transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=head["gaussian_blur"]["kernel_size"],
                    sigma=(head["gaussian_blur"]["sigma_min"], head["gaussian_blur"]["sigma_max"]),
                )
            ],
            p=head["gaussian_blur"]["p"],
        )
        self.wrist_gaussian_blur = transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=wrist["gaussian_blur"]["kernel_size"],
                    sigma=(wrist["gaussian_blur"]["sigma_min"], wrist["gaussian_blur"]["sigma_max"]),
                )
            ],
            p=wrist["gaussian_blur"]["p"],
        )
        self.random_erasing = transforms.RandomErasing(
            p=erase["p"],
            scale=(erase["scale_min"], erase["scale_max"]),
        )

        # Issue #129 Phase G (2026-08-31): 追加 aug (Session 32 論点 M4 決定通り)
        # ・sharpness    (head + wrist、GR00T 由来、backbone 高周波特徴保護)
        # ・camera domain (GaussianNoise + JPEG、real robot sensor noise / JPEG stream 対応)
        # ・geometric aug (head only、camera 取付位置ズレ吸収、OBB coord も同 affine で warp
        #    → Run 2/6 で `warp_obb_coord_under_affine` に渡す)
        # config schema は base.yaml Phase G section を参照。keys が無ければ safe default。
        _sharp_head = head.get("sharpness", {"factor": 1.5, "p": 0.5})
        _sharp_wrist = wrist.get("sharpness", {"factor": 1.5, "p": 0.5})
        self.head_sharpness = tv2.RandomAdjustSharpness(
            sharpness_factor=float(_sharp_head["factor"]), p=float(_sharp_head["p"])
        )
        self.wrist_sharpness = tv2.RandomAdjustSharpness(
            sharpness_factor=float(_sharp_wrist["factor"]), p=float(_sharp_wrist["p"])
        )
        _noise_head = head.get("gaussian_noise", {"sigma": 0.010, "p": 0.5})
        _noise_wrist = wrist.get("gaussian_noise", {"sigma": 0.010, "p": 0.5})
        self.head_gaussian_noise = tv2.RandomApply(
            [tv2.GaussianNoise(sigma=float(_noise_head["sigma"]), clip=True)],
            p=float(_noise_head["p"]),
        )
        self.wrist_gaussian_noise = tv2.RandomApply(
            [tv2.GaussianNoise(sigma=float(_noise_wrist["sigma"]), clip=True)],
            p=float(_noise_wrist["p"]),
        )
        _jpeg_head = head.get("jpeg", {"quality": [60, 90], "p": 0.3})
        _jpeg_wrist = wrist.get("jpeg", {"quality": [60, 90], "p": 0.3})
        self.head_jpeg = tv2.RandomApply(
            [FloatJPEG(quality=tuple(_jpeg_head["quality"]))],
            p=float(_jpeg_head["p"]),
        )
        self.wrist_jpeg = tv2.RandomApply(
            [FloatJPEG(quality=tuple(_jpeg_wrist["quality"]))],
            p=float(_jpeg_wrist["p"]),
        )

        # Geometric aug: head only、config で on/off 可。sampled params は apply() で
        # (prev, current) 共通適用 (delta encoder guard)。affine matrix (2, 3) を
        # normalized [0,1] 空間で return、OBB coord warp と一貫。
        geom = cfg.get("geometric", {"enabled": False, "degrees": [-5.0, 5.0], "translate": [0.05, 0.05]})
        self.geometric_enabled = bool(geom.get("enabled", False))
        _deg = geom.get("degrees", [-5.0, 5.0])
        _trn = geom.get("translate", [0.05, 0.05])
        self.geometric_degrees = (float(_deg[0]), float(_deg[1]))
        self.geometric_translate = (float(_trn[0]), float(_trn[1]))

    def is_wrist(self, cam_key: str) -> bool:
        return cam_key in self.wrist_cam_keys or self.wrist_key_substring in cam_key

    def uses_geometric(self, cam_key: str) -> bool:
        """この cam に geometric aug が掛かるか (skip_geometric 指定なしの場合)。"""
        return self.enabled and self.geometric_enabled and not self.is_wrist(cam_key)

    @staticmethod
    def _seed(idx: int | str, cam_key: str) -> int:
        # per-sample per-cam seed。CRC32 で python hash 依存 (PYTHONHASHSEED) を回避
        return zlib.crc32(f"{idx}:{cam_key}".encode()) & 0x7FFFFFFF

    def photometric(self, img: torch.Tensor, cam_key: str, idx: int | str) -> torch.Tensor:
        """1 frame (3, H, W) float [0,1] に photometric → camera domain → RandomErasing を適用。

        overlay 描画前までの処理。seed は (idx, cam_key) で決まり、同じ引数なら同じ結果
        (prev / current や token / overlay で共通に使える)。
        """
        is_wrist = self.is_wrist(cam_key)
        color_jitter = self.wrist_color_jitter if is_wrist else self.head_color_jitter
        gaussian_blur = self.wrist_gaussian_blur if is_wrist else self.head_gaussian_blur
        sharpness = self.wrist_sharpness if is_wrist else self.head_sharpness
        gaussian_noise = self.wrist_gaussian_noise if is_wrist else self.head_gaussian_noise
        jpeg = self.wrist_jpeg if is_wrist else self.head_jpeg

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._seed(idx, cam_key))
            img = color_jitter(img)
            img = gaussian_blur(img)
            img = sharpness(img)             # Phase G
            img = gaussian_noise(img)        # Phase G (RandomApply、内部 seed で p 判定)
            img = jpeg(img)                  # Phase G (float wrapper、RandomApply で p 判定)
            # Phase D: RandomErasing を Normalize 前に配置 (overlay を消さない)
            img = self.random_erasing(img)
        return img

    def geometric_params(
        self, cam_key: str, idx: int | str, skip_geometric: bool = False
    ) -> GeometricParams:
        """geometric params を per-sample-per-cam で 1 回 sample (prev/current 共通)。

        head cam + geometric.enabled のみ実適用、それ以外は identity affine。
        Phase I-0-3 (2026-09-01): skip_geometric=True で強制 off (precompute token 側用)。
        """
        use_geometric = self.uses_geometric(cam_key) and not skip_geometric
        if not use_geometric:
            return GeometricParams(False, 0.0, 0.0, 0.0, _identity_affine_matrix())
        geom_rng = np.random.RandomState(self._seed(idx, cam_key))
        angle_deg = float(geom_rng.uniform(self.geometric_degrees[0], self.geometric_degrees[1]))
        tx_frac = float(geom_rng.uniform(-self.geometric_translate[0], self.geometric_translate[0]))
        ty_frac = float(geom_rng.uniform(-self.geometric_translate[1], self.geometric_translate[1]))
        affine_matrix = _make_head_affine_matrix(angle_deg, tx_frac, ty_frac)
        return GeometricParams(True, angle_deg, tx_frac, ty_frac, affine_matrix)

    @staticmethod
    def apply_geometric(img: torch.Tensor, params: GeometricParams) -> torch.Tensor:
        """geometric aug (head only) を 1 frame に適用。params.use=False なら no-op。"""
        if not params.use:
            return img
        H, W = img.shape[-2:]
        tx_pix = int(round(params.tx_frac * W))
        ty_pix = int(round(params.ty_frac * H))
        return tvF.affine(
            img,
            angle=params.angle_deg,
            translate=[tx_pix, ty_pix],
            scale=1.0,
            shear=[0.0, 0.0],
        )

    def apply(
        self,
        img_2: torch.Tensor,
        cam_key: str,
        idx: int | str,
        is_train: bool,
        overlay_callback=None,
        skip_normalize: bool = False,
        skip_geometric: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """img_2: (T, 3, H, W) float [0,1]。T=2 (prev + current) または T=1 (current だけ、Issue #141 RO-11)。

        Issue #129 Phase D (2026-08-31) refactor: overlay 保護のため pipeline 順を
          Photometric → Camera domain → RandomErasing → **overlay_callback** → Geometric → Normalize
        に変更 (旧 = Photometric → Normalize → RandomErasing で overlay を後段が汚染)。

        Issue #129 Phase G (2026-08-31) refactor: 追加 aug 統合 + return を `(imgs, affine)` に:
        - Photometric に **sharpness** 追加
        - Camera domain (**GaussianNoise + JPEG**) を新設、Photometric と RandomErasing の間
        - **Geometric aug (RandomAffine head only)** を overlay 後 / Normalize 前に追加、
          affine matrix (normalized [0,1] 空間、(2, 3)) を return して OBB coord warp と一貫
          (wrist / disabled は identity affine を return)

        Args:
            overlay_callback: None (デフォルト、overlay skip) or callable
                (img_frame_tensor: torch.Tensor (3, H, W) float [0,1] RGB, frame_i: int)
                -> torch.Tensor (3, H, W) float [0,1] RGB
                per-frame で呼ばれ、overlay 描画済 tensor を return する契約。
            skip_normalize: True で最終 Normalize を skip (Issue #129 Phase I-0-1、2026-09-01)。
                offline aug bake precompute では [0,1] float の bake 済 image を jpg 保存
                したいので True で呼ぶ。default False (train/val 経路は Normalize 済 tensor を
                model input として使う)。
            skip_geometric: True で geometric aug を強制 off (Issue #129 Phase I-0-3、
                2026-09-01)。offline aug bake precompute の token 側 (coord grounding 用) は
                box coord と image の幾何一致を保つため geometric を skip する。affine は
                identity を return。config で `geometric.enabled=False` にするより per-call
                で切替可能で、単一 config yaml で token/overlay 両側の precompute を回せる。
                default False (train/val 経路 + overlay 側 precompute は config どおり適用)。

        Returns:
            (imgs, affine_matrix):
              imgs: (T, 3, H, W) float32、aug + overlay + Normalize 済 (全 frame に同じ aug)
              affine_matrix: (2, 3) float32、normalized [0,1] 空間 (identity for wrist / disabled)。
                             Run 2/6 (coord token + head geometric) で
                             `warp_obb_coord_under_affine(verts, affine_matrix)` に渡す想定。
        """
        # val path (aug 無し): overlay_callback あれば適用してから normalize、affine は identity
        if not self.enabled or not is_train:
            results = []
            for frame_i in range(img_2.shape[0]):
                img = img_2[frame_i]
                if overlay_callback is not None:
                    img = overlay_callback(img, frame_i)
                results.append(img if skip_normalize else self.normalize(img))
            return torch.stack(results, dim=0), _identity_affine_matrix()

        geom = self.geometric_params(cam_key, idx, skip_geometric=skip_geometric)

        results = []
        for frame_i in range(img_2.shape[0]):
            # prev/current 同 seed で photometric 系を同 params 適用
            # (delta encoder が aug 変化を学習しないよう保証)
            img = self.photometric(img_2[frame_i], cam_key, idx)
            # Phase D: overlay_callback は Erasing 後 / Geometric 前
            if overlay_callback is not None:
                img = overlay_callback(img, frame_i)
            # Phase G: geometric aug (head only)、prev/current 共通 params で warp
            img = self.apply_geometric(img, geom)
            if not skip_normalize:
                img = self.normalize(img)
            results.append(img)
        return torch.stack(results, dim=0), geom.affine_matrix
