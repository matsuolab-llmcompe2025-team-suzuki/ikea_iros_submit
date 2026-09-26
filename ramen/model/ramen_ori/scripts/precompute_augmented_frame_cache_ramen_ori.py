"""RAMEN-Ori 用 offline aug + overlay bake precompute (Issue #129 Phase I-0-2、2026-09-01)。

# 目的

RAMEN-Ori R-6 bench の training loop で online aug + inline overlay を回すと
`__getitem__` の CPU cost が支配的になり (GR00T 実測 aug on 1.6s/step vs aug off 0.9s/step、
100k step で ~19h 差)、GPU が待たされる。両方を offline で bake:

- `<output_token_root>/<cam>/<chunk>/<file>/frame_XXXXXX_v{i}.jpg`
    aug 適用済 image (overlay 無し、coord grounding = Run 2/6 で使う)
- `<output_overlay_root>/<cam>/<chunk>/<file>/frame_XXXXXX_v{i}.jpg`
    aug + overlay bake 済 image (overlay grounding = Run 1/3/4/5 で使う)
- `<output_common_root>/<cam>/<chunk>/<file>/frame_XXXXXX_v{i}.jpg` (任意、Issue #139)
    token と overlay で同一になる cam (OBB overlay 無し + geometric 無し = wrist) を 1 回だけ書く。
    token / overlay root 側には `<cam> -> <common>/<cam>` の symlink を張る

v0 (raw jpg) は無変更、rerun 用 raw source として保持 (GR00T `precompute_augmented_frame_cache.py`
と同じ設計)。Issue #139 以降、GR00T / ACT / DP もこの script の overlay 出力を読む (1 本化)。

# GR00T script との差分

- 使う aug は **`model/ramen_ori/image_aug.py:ImageAugPipeline`** を通す。GR00T の
  hard-coded `_build_aug_pools()` ではなく、YAML config (Phase G で確定した base.yaml
  の augmentation section) を渡す。sharpness / JPEG / GaussianNoise / geometric head only
  が Phase G で追加されており、GR00T pool との差異がある。
- **token / overlay (/ common) の output root** に同時 write。
  token 側は overlay skip、overlay 側は overlay bake。
- **seed derivation**: `idx = f"{frame_global_uid}:{variant_i}"` を str で渡し、
  ImageAugPipeline 内の `zlib.crc32(f"{idx}:{cam_key}".encode())` で最終 seed 化。
  training loader は同 seed で affine matrix を再現し、raw OBB (obb_yolo/) に
  `warp_obb_coord_under_affine` を適用して geometric 一致を保つ (I-0-3 で実装)。

# Aug pipeline order (photometric → overlay → geometric)

  Photometric (color/sharpness/blur) → Camera domain (GaussianNoise/JPEG) → RandomErasing
    → **overlay** (overlay 側のみ、per_frame OBB verts を描画) → Geometric affine
    (head cam only、wrist は identity affine)

Issue #139: photometric は `ImageAugPipeline.photometric` で 1 frame 分だけ 1 回計算し、
token (そのまま) と overlay (描画 + geometric) に分岐する。`apply()` を token / overlay で
2 回、しかも 2 frame 分呼んでいた旧実装と出力はビット一致する (test_image_aug.py で検証)。
[0,1] float tensor のまま jpg 保存 (Normalize は training loader で on-the-fly)。

# OBB lookup の cam 名

v0 dir の cam 名は training view 名 (head_left 等) でも source 名 (cam_0 等) でもよい。
training view marker / fallback の camera_map で source 名に翻訳し、対応が無くても
OBB cache が同名の cam を持っていればそのまま引く。OBB cache が覆う cam で lookup が
1 件も当たらない場合は途中で止める (silent に overlay 無しで bake しないため)。

# 完了マーカー (Issue #139)

mp4 dir (`<cam>/<chunk>/<file>`) の全 frame を bake し終えたら、出力先の同 dir に
`_bake_done.json` ({"frames": N, "num_variants": V}) を書く。`pack_frame_cache_tars.py
--watch` はこれを見て、焼き上がった mp4 から順に tar 化する。

# Cache layout

    <input_cache_root>/{cam}/{chunk-XXX}/{file-YYY}/
        frame_000000.jpg           # v0 raw、無変更で保持 (rerun source)

    <output_{token,overlay,common}_root>/{cam}/{chunk-XXX}/{file-YYY}/
        frame_000000_v1.jpg
        ...
        frame_000000_v{N-1}.jpg
        _bake_done.json

# Usage

    python model/ramen_ori/scripts/precompute_augmented_frame_cache_ramen_ori.py \\
        --input-cache-root <path>/frame_cache \\
        --output-token-root <path>/frame_cache_v2/token \\
        --output-overlay-root <path>/frame_cache_v2/overlay \\
        [--output-common-root <path>/frame_cache_v2/common] \\
        --aug-config-yaml data/bitrobot_lerobot_subtask_datasets/configs/unified_frame_cache.yaml \\
        --obb-root <path>/obb_yolo \\
        --num-variants 10 \\
        --jobs 22 \\
        [--cuda-workers 8] [--gpu-jpeg-encode] [--torch-threads 1] [--chunksize 16]
        [--max-frames 2000]                # 速度計測用
        [--obb-hash <hash>] [--obb-top-k 32]
        [--overlay-conf 0.30]
        [--overlay-thickness 2]
        [--overlay-class-filter '[3,4]']
        [--force]

# Reproducibility

Precompute で使った seed derivation を training loader (I-0-3) が同じ formula で再現する:
    seed = zlib.crc32(f"{frame_global_uid}:{variant_i}:{cam_key}".encode()) & 0x7FFFFFFF

これで token side の baked jpg + `obb_yolo/` の raw OBB を組み合わせて、
`warp_obb_coord_under_affine` で geometric 一致した warped OBB coord を再構成できる。
geometric params は numpy RandomState で決まるので GPU worker でも同じ値になる。
photometric の乱数 (GaussianNoise) は CPU / GPU で出方が異なるので、`--cuda-workers` を使うと
どの worker が処理したかで画像が変わる (分布は同じ、token / overlay は同じ計算から分岐するので対応は保たれる)。

# CPU / GPU 混成 (Issue #139)

`--jobs N --cuda-workers K` で、worker のうち K 個が aug を GPU で、残り N-K 個が CPU で計算する。
task は空いた worker が順に取る (imap_unordered) ので、速い worker ほど多く処理し、CPU と GPU の
両方が埋まる。K と N は Sakura で `--max-frames` を使って測って決める。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# Repo root を sys.path に注入 (mp.Pool fork でも継承される)。
# `<repo>/model/ramen_ori/scripts/<this>` から辿って repo root を導出。
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_JPG_QUALITY = 90
DEFAULT_OVERLAY_CONF = 0.30
DEFAULT_OVERLAY_THICKNESS = 2
DONE_MARKER_NAME = "_bake_done.json"
CACHE_META_NAME = "_cache_meta.json"

MARKER_REL_PATH = Path("meta") / "team_ramen_training_view.json"

# subtask_training.json:source_dataset.camera_map の hardcoded 版 (marker 不在時 fallback)。
# training view で使う cam 名 → source dataset cam 名 (OBB cache の key)。
FALLBACK_CAMERA_MAP: dict[str, str] = {
    "observation.images.head_left": "observation.images.cam_0",
    "observation.images.head_right": "observation.images.cam_1",
    "observation.images.left_wrist": "observation.images.cam_2",
    "observation.images.right_wrist": "observation.images.cam_3",
}


@dataclass(frozen=True)
class CamPlan:
    """1 cam の bake 方針。"""

    source_cam: str | None  # OBB cache の cam key (None = overlay 描画なし)
    shared: bool  # token / overlay が同一 → common root に 1 回だけ書く


def _load_aug_cfg(yaml_path: Path) -> dict:
    """base.yaml から augmentation section を抽出。ImageAugPipeline に渡せる dict にする。"""
    import yaml  # noqa: PLC0415

    with yaml_path.open() as f:
        full_cfg = yaml.safe_load(f)
    aug = full_cfg.get("augmentation") if isinstance(full_cfg, dict) else None
    if aug is None:
        raise ValueError(
            f"aug config yaml missing top-level 'augmentation' key: {yaml_path}"
        )
    return aug


def _derive_frame_info(v0_path: Path, cache_root: Path) -> tuple[str, str, int, str]:
    """`<cache_root>/{cam}/{chunk-XXX}/{file-YYY}/frame_XXXXXX.jpg` から
    (training_cam_key, mp4_relative, frame_idx_in_mp4, frame_global_uid) を抽出。

    mp4_relative は "chunk-XXX/file-YYY" (OBB cache lookup 用)。
    frame_global_uid = f"{cam}:{chunk}:{file}:{frame_idx}" (seed derivation の再現性用、
    training loader (I-0-3) がこの UID を再構成できるよう path-based に定義)。
    """
    rel = v0_path.relative_to(cache_root)
    parts = rel.parts
    if len(parts) != 4:
        raise ValueError(
            f"v0 path structure unexpected (expect <cam>/<chunk>/<file>/frame_XXX.jpg): "
            f"{v0_path}"
        )
    training_cam = parts[0]
    chunk = parts[1]
    file_stem = parts[2]
    stem = v0_path.stem
    if not stem.startswith("frame_"):
        raise ValueError(f"unexpected jpg name: {v0_path.name}")
    frame_idx = int(stem[len("frame_") :])
    frame_uid = f"{training_cam}:{chunk}:{file_stem}:{frame_idx:06d}"
    return training_cam, f"{chunk}/{file_stem}", frame_idx, frame_uid


def _load_camera_map(cache_root: Path) -> dict[str, str]:
    """Training view marker から camera_map を読取、無ければ hardcoded fallback。"""
    marker_path = cache_root.parent / MARKER_REL_PATH
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        camera_map = marker.get("camera_map")
        if isinstance(camera_map, dict) and camera_map:
            return dict(camera_map)
    return dict(FALLBACK_CAMERA_MAP)


def _resolve_source_cam(
    training_cam: str, camera_map: dict[str, str], covered_cameras: set[str]
) -> str | None:
    """training cam → OBB cache の cam key。cache が覆わない cam は None。

    camera_map に対応が無くても、cache が同名の cam を持っていればそのまま使う
    (v0 dir が source 名 `observation.images.cam_0` の場合、Issue #139)。
    """
    source = camera_map.get(training_cam)
    if source is None and training_cam in covered_cameras:
        source = training_cam
    if source is None or source not in covered_cameras:
        return None
    return source


def _collect_all_v0_frames(cache_root: Path) -> list[Path]:
    """`<cache_root>/{cam}/{chunk}/{file}/frame_*.jpg` (v0 のみ、_v サフィックス除く) を列挙。

    mp4 dir 単位で完了させるため path 順に並べる (pack の逐次 tar 化が早く始まる)。
    """
    frames: list[Path] = []
    for jpg in cache_root.rglob("frame_*.jpg"):
        if "_v" in jpg.stem.split("frame_")[-1]:
            continue
        frames.append(jpg)
    return sorted(frames)


def _list_input_cams(cache_root: Path) -> list[str]:
    """input cache root 直下の cam dir 名 (symlink は除く)。"""
    return sorted(
        p.name for p in cache_root.iterdir() if p.is_dir() and not p.is_symlink()
    )


def build_cam_plans(
    cams: list[str],
    *,
    camera_map: dict[str, str],
    covered_cameras: set[str],
    uses_geometric,
    share_enabled: bool,
) -> dict[str, CamPlan]:
    """cam ごとに OBB lookup key と common 共有の可否を決める。

    共有できるのは overlay 描画も geometric も無い cam (token == overlay になる cam) だけ。
    """
    plans: dict[str, CamPlan] = {}
    for cam in cams:
        source = _resolve_source_cam(cam, camera_map, covered_cameras)
        shared = share_enabled and source is None and not uses_geometric(cam)
        plans[cam] = CamPlan(source_cam=source, shared=shared)
    return plans


def _link_shared_cams(
    plans: dict[str, CamPlan], common_root: Path, link_roots: list[Path]
) -> None:
    """共有 cam について `<root>/<cam> -> <common_root>/<cam>` の相対 symlink を張る。"""
    for cam, plan in plans.items():
        if not plan.shared:
            continue
        target = common_root / cam
        target.mkdir(parents=True, exist_ok=True)
        for root in link_roots:
            root.mkdir(parents=True, exist_ok=True)
            link = root / cam
            rel_target = os.path.relpath(target, root)
            if link.is_symlink():
                if os.readlink(link) == rel_target:
                    continue
                raise FileExistsError(f"symlink {link} points elsewhere: {os.readlink(link)}")
            if link.exists():
                raise FileExistsError(
                    f"{link} は実 dir として存在する (共有 cam は {target} に置く設計)。"
                    "旧出力を消してから再実行してください"
                )
            link.symlink_to(rel_target, target_is_directory=True)


def _parse_class_filter(spec: str | None) -> list[int] | None:
    """`--overlay-class-filter '[3,4]'` の JSON list-of-int を parse。無指定なら None。"""
    if spec is None or spec.strip() == "":
        return None
    parsed = json.loads(spec)
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise argparse.ArgumentTypeError(
            f"--overlay-class-filter must be JSON list of int, got {spec!r}"
        )
    return parsed


def _count_drawable_boxes(
    per_frame: dict[str, Any], conf_threshold: float, class_filter: set[int] | None
) -> int:
    """overlay で実際に描かれる box 数 (valid かつ conf 以上、class filter 通過)。"""
    mask = np.asarray(per_frame["valid"], dtype=bool) & (
        np.asarray(per_frame["conf"], dtype=np.float32) >= conf_threshold
    )
    if class_filter is not None:
        mask &= np.isin(np.asarray(per_frame["class_id"]), list(class_filter))
    return int(mask.sum())


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

# Worker-global (mp.Pool initializer で set、per-worker 一度だけ生成)
_worker_state: dict[str, Any] = {}


def _assign_device(counter: Any, cuda_workers: int) -> str:
    """worker の device。起動順に先頭 cuda_workers 個が cuda、残りが cpu。

    counter は process 間共有の mp.Value (main process 内実行では None)。
    """
    if counter is None:
        return "cuda" if cuda_workers > 0 else "cpu"
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    return "cuda" if idx < cuda_workers else "cpu"


def _worker_init(
    obb_root: str,
    obb_hash: str | None,
    obb_camera_keys: list[str],
    obb_top_k: int,
    obb_num_classes: int,
    obb_fps: int,
    cam_plans: dict[str, CamPlan],
    overlay_conf: float,
    overlay_thickness: int,
    overlay_class_filter_list: list[int] | None,
    jpg_quality: int,
    aug_cfg: dict,
    cache_root: str,
    output_token_root: str | None,
    output_overlay_root: str | None,
    output_common_root: str | None,
    cuda_workers: int,
    torch_threads: int,
    gpu_encode: bool = False,
    device_counter: Any = None,
) -> None:
    """Pool worker 起動時に 1 回だけ呼ばれる。OBB cache + ImageAugPipeline を初期化。"""
    global _worker_state
    import cv2  # noqa: PLC0415

    from model.ramen_ori.image_aug import ImageAugPipeline  # noqa: PLC0415
    from model.ramen_ori.obb_cache import ObbPrecomputedCache  # noqa: PLC0415

    # worker 数 × 内部スレッドで CPU を取り合わないよう 1 worker あたりのスレッドを絞る
    torch.set_num_threads(max(1, int(torch_threads)))
    cv2.setNumThreads(1)
    device = _assign_device(device_counter, int(cuda_workers))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--cuda-workers が指定されたが CUDA が使えない")

    cache = ObbPrecomputedCache(
        root=Path(obb_root),
        ckpt_hash=obb_hash,
        camera_keys=obb_camera_keys,
        top_K=obb_top_k,
        num_classes=obb_num_classes,
        fps=obb_fps,
    )
    pipe = ImageAugPipeline(aug_cfg)
    if not pipe.enabled:
        raise ValueError(
            "ImageAugPipeline reports enabled=False; aug bake precompute では "
            "augmentation.enabled=true が必須"
        )

    _worker_state = {
        "cache": cache,
        "cam_plans": dict(cam_plans),
        "pipe": pipe,
        "cfg": {
            "overlay_conf": float(overlay_conf),
            "overlay_thickness": int(overlay_thickness),
            "overlay_class_filter": (
                set(overlay_class_filter_list) if overlay_class_filter_list is not None else None
            ),
            "jpg_quality": int(jpg_quality),
            "gpu_encode": bool(gpu_encode),
        },
        "cache_root": Path(cache_root),
        "output_token_root": Path(output_token_root) if output_token_root else None,
        "output_overlay_root": Path(output_overlay_root) if output_overlay_root else None,
        "output_common_root": Path(output_common_root) if output_common_root else None,
        "device": torch.device(device),
    }


def _load_v0_as_tensor(v0_path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    """v0 jpg を (3, H, W) float [0,1] tensor に load、(H, W) も返す。"""
    img_pil = Image.open(v0_path).convert("RGB")
    arr = np.asarray(img_pil, dtype=np.uint8)  # (H, W, 3) RGB
    img_pil.close()
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # (3, H, W) [0,1]
    return tensor, (arr.shape[0], arr.shape[1])


_gpu_encode_failed = False


def _save_tensor_as_jpg(
    tensor_3hw: torch.Tensor, out_path: Path, quality: int, gpu_encode: bool = False
) -> None:
    """(3, H, W) float [0,1] tensor を jpg に save (親 dir 自動作成)。

    tmp に書いてから rename する: 途中で止まっても半端な jpg が残らない (resume 時は
    既存 file を skip するため、半端な file が残るとそのまま tar に入ってしまう)。
    gpu_encode=True かつ CUDA tensor なら nvJPEG で encode (失敗したら以降 CPU)。
    """
    global _gpu_encode_failed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    if gpu_encode and tensor_3hw.is_cuda and not _gpu_encode_failed:
        try:
            from torchvision.io import encode_jpeg  # noqa: PLC0415

            u8 = (tensor_3hw.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            tmp.write_bytes(encode_jpeg(u8, quality=quality).cpu().numpy().tobytes())
            os.replace(tmp, out_path)
            return
        except Exception as e:  # noqa: BLE001
            _gpu_encode_failed = True
            print(f"[precompute_aug_ramen_ori] WARN: GPU jpeg encode 失敗、CPU に切替: {e}", flush=True)
    arr = tensor_3hw.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3) [0,1]
    img_pil = Image.fromarray((arr * 255.0).astype(np.uint8))
    # --gpu-jpeg-encode の cache は nvJPEG と同じ色差 4:4:4 に揃える。CPU worker や nvJPEG 失敗後の
    # fallback だけ PIL の既定 (4:2:0) になると、1 つの cache に色差が混ざり、推論側の
    # `overlay_jpeg_subsampling` (slot に 1 つ) では合わせきれない
    save_kwargs = {"subsampling": 0} if gpu_encode else {}
    img_pil.save(tmp, format="JPEG", quality=quality, **save_kwargs)
    os.replace(tmp, out_path)


def _lookup_overlay_for_frame(
    source_cam: str | None, mp4_relative: str, frame_idx: int
) -> dict[str, Any] | None:
    """OBB cache を引く。cache が覆わない cam (source_cam=None) や範囲外 frame は None。"""
    if source_cam is None:
        return None
    return _worker_state["cache"].lookup_mp4_frame(source_cam, mp4_relative, frame_idx)


def _process_one_frame(
    args_tuple: tuple[Path, int, bool],
) -> tuple[Path, int, int, bool | None, int, str]:
    """1 frame について v1..v(N-1) の token / overlay (/ common) を bake。v0 は変更しない。

    Returns:
        (v0_path, saved, skipped, obb_hit, n_boxes)
          saved: 新規 write した jpg 数 / skipped: 出力が全て既存で飛ばした variant 数
          obb_hit: OBB cache が覆う cam なら lookup 成否、覆わない cam は None
          n_boxes: overlay で描かれる box 数
          device: 処理した worker の device ("cpu" / "cuda")、混成時の配分確認用
    """
    v0_path, num_variants, force = args_tuple
    st = _worker_state
    cache_root: Path = st["cache_root"]
    cfg = st["cfg"]
    pipe = st["pipe"]

    training_cam, mp4_relative, frame_idx, frame_uid = _derive_frame_info(v0_path, cache_root)
    plan: CamPlan = st["cam_plans"][training_cam]
    per_frame = _lookup_overlay_for_frame(plan.source_cam, mp4_relative, frame_idx)
    obb_hit = None if plan.source_cam is None else per_frame is not None
    n_boxes = (
        _count_drawable_boxes(per_frame, cfg["overlay_conf"], cfg["overlay_class_filter"])
        if per_frame is not None
        else 0
    )

    rel = v0_path.relative_to(cache_root)
    parent_rel = rel.parent  # <cam>/<chunk>/<file>
    stem = v0_path.stem       # frame_XXXXXX
    jpg_quality = cfg["jpg_quality"]
    gpu_encode = cfg["gpu_encode"]

    v0_tensor: torch.Tensor | None = None
    saved = 0
    skipped = 0
    for v_idx in range(1, num_variants):
        name = f"{stem}_v{v_idx}.jpg"
        if plan.shared:
            common_out = st["output_common_root"] / parent_rel / name
            token_out = overlay_out = None
            need_common = force or not common_out.exists()
            need_token = need_overlay = False
        else:
            common_out = None
            need_common = False
            token_root = st["output_token_root"]
            overlay_root = st["output_overlay_root"]
            token_out = token_root / parent_rel / name if token_root is not None else None
            overlay_out = overlay_root / parent_rel / name if overlay_root is not None else None
            need_token = token_out is not None and (force or not token_out.exists())
            need_overlay = overlay_out is not None and (force or not overlay_out.exists())
        if not (need_common or need_token or need_overlay):
            skipped += 1
            continue

        if v0_tensor is None:
            v0_tensor = _load_v0_as_tensor(v0_path)[0].to(st["device"])

        # Seed derivation: (frame_uid, variant_i) を str idx として ImageAugPipeline に渡す。
        # Training loader (I-0-3) は同 (frame_uid, variant_i) で seed 再構成する契約。
        idx_str = f"{frame_uid}:v{v_idx}"
        # photometric は 1 回だけ。token / overlay / common はここから分岐する
        base = pipe.photometric(v0_tensor, training_cam, idx_str)

        if need_common:
            # overlay も geometric も無い cam → token == overlay、1 回だけ書く
            _save_tensor_as_jpg(base, common_out, jpg_quality, gpu_encode)
            saved += 1
        # token 側 (overlay skip + geometric skip)
        # geometric aug を skip する理由: token grounding では box coord (raw OBB) が別 token
        # として入力されるので、image に geometric aug を適用すると box と image で幾何
        # 不一致になる。photometric aug のみ bake し、box coord は raw obb_yolo/ を identity
        # affine 前提で load すれば一貫。
        if need_token:
            _save_tensor_as_jpg(base, token_out, jpg_quality, gpu_encode)
            saved += 1
        # overlay 側 (per_frame OBB 描画 bake + geometric 有効)
        # overlay grounding では box が pixel level で描画済 → geometric aug で image と
        # overlay が一緒に warp されて幾何一致を保つ (rotate handoff §2.5 と同 pattern)。
        if need_overlay:
            img = base
            if per_frame is not None:
                from model.ramen_ori.overlay import draw_overlay_on_tensor  # noqa: PLC0415

                img = draw_overlay_on_tensor(
                    base,
                    verts=per_frame["verts"],
                    conf=per_frame["conf"],
                    class_id=per_frame["class_id"],
                    valid=per_frame["valid"],
                    conf_threshold=cfg["overlay_conf"],
                    line_thickness=cfg["overlay_thickness"],
                    class_filter=cfg["overlay_class_filter"],
                )
            img = pipe.apply_geometric(img, pipe.geometric_params(training_cam, idx_str))
            _save_tensor_as_jpg(img, overlay_out, jpg_quality, gpu_encode)
            saved += 1

    return v0_path, saved, skipped, obb_hit, n_boxes, st["device"].type


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class _CamStats:
    frames: int = 0      # OBB が覆う cam で処理した frame 数
    hits: int = 0        # OBB lookup が当たった frame 数
    box_frames: int = 0  # box が 1 つ以上描かれた frame 数


def _write_done_markers(
    rel_dir: Path,
    plan: CamPlan,
    *,
    frames: int,
    num_variants: int,
    token_root: Path | None,
    overlay_root: Path | None,
    common_root: Path | None,
    jpg_subsampling: str,
) -> None:
    """mp4 dir の bake 完了を各出力先に記録 (pack --watch が拾う)。

    jpg の色差も残す。推論 slot の `overlay_jpeg_subsampling` をこれに合わせる。
    """
    roots = [common_root] if plan.shared else [token_root, overlay_root]
    payload = json.dumps(
        {"frames": frames, "num_variants": num_variants, "jpg_subsampling": jpg_subsampling}
    )
    for root in roots:
        if root is None:
            continue
        marker = root / rel_dir / DONE_MARKER_NAME
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(payload, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-cache-root",
        type=Path,
        required=True,
        help="v0 raw frame cache root (例: outputs/training_views/ramen_ori_task0/frame_cache)",
    )
    parser.add_argument(
        "--output-token-root",
        type=Path,
        default=None,
        help="Aug 適用済 (overlay 無し) の出力 root。coord grounding = Run 2/6 用",
    )
    parser.add_argument(
        "--output-overlay-root",
        type=Path,
        default=None,
        help="Aug + overlay bake 済 の出力 root。overlay grounding = Run 1/3/4/5 用",
    )
    parser.add_argument(
        "--output-common-root",
        type=Path,
        default=None,
        help="token と overlay で同一になる cam (wrist 等) を 1 回だけ書く root (Issue #139)。"
        "token / overlay root には symlink を張る",
    )
    parser.add_argument(
        "--aug-config-yaml",
        type=Path,
        required=True,
        help="ImageAugPipeline に渡す config yaml (例: model/ramen_ori/configs/base.yaml)、"
        "top-level 'augmentation' section を読取",
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=10,
        help="Total variants per frame (v0 + v1..v_(N-1))。Default 10",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=DEFAULT_JPG_QUALITY,
        help="JPG encoding quality (0-100)。Default 90 (元 cache と一致)",
    )
    parser.add_argument(
        "--jobs", type=int, default=24, help="Multiprocessing worker 数 (<=1 で main process 内実行)"
    )
    parser.add_argument(
        "--cuda-workers",
        type=int,
        default=0,
        help="--jobs のうち aug を GPU で計算する worker 数 (残りは CPU)。>0 で worker を spawn で起動",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=1, help="worker 1 つあたりの torch スレッド数"
    )
    parser.add_argument(
        "--chunksize", type=int, default=16, help="imap_unordered の chunksize"
    )
    parser.add_argument(
        "--gpu-jpeg-encode",
        action="store_true",
        help="GPU worker では jpg encode も GPU (nvJPEG) で行う",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="先頭 N 個の v0 frame だけ処理 (速度計測用)。途中までの mp4 dir には完了マーカーを書かない",
    )
    parser.add_argument(
        "--guard-min-frames",
        type=int,
        default=200,
        help="OBB が覆う cam でこの frame 数を処理しても lookup が 1 件も当たらなければ中断",
    )
    parser.add_argument(
        "--force", action="store_true", help="既存 v_i を無視して再生成"
    )
    # === OBB overlay params ===
    parser.add_argument(
        "--obb-root",
        type=Path,
        required=True,
        help="OBB precompute cache root (例: /nvme/rotate_table_base/obb_yolo)",
    )
    parser.add_argument(
        "--obb-hash",
        type=str,
        default=None,
        help="OBB cache hash dir 名。未指定なら root 直下唯一の hash dir auto-pick",
    )
    parser.add_argument(
        "--obb-cameras",
        type=str,
        default="observation.images.cam_0,observation.images.cam_1",
        help="OBB cache の cam key list (comma-sep)",
    )
    parser.add_argument("--obb-top-k", type=int, default=4)
    parser.add_argument("--obb-num-classes", type=int, default=7)
    parser.add_argument("--obb-fps", type=int, default=30)
    parser.add_argument(
        "--overlay-conf", type=float, default=DEFAULT_OVERLAY_CONF, help="Overlay conf threshold"
    )
    parser.add_argument(
        "--overlay-thickness",
        type=int,
        default=DEFAULT_OVERLAY_THICKNESS,
        help="cv2.polylines thickness",
    )
    parser.add_argument(
        "--overlay-class-filter",
        type=str,
        default=None,
        help="JSON list of int (例 '[3,4]' = hole+table_top のみ描画)、未指定 = 全 class",
    )
    args = parser.parse_args()

    if not args.input_cache_root.is_dir():
        print(f"ERROR: input-cache-root not found: {args.input_cache_root}", file=sys.stderr)
        return 2
    if args.output_token_root is None and args.output_overlay_root is None:
        print(
            "ERROR: at least one of --output-token-root / --output-overlay-root must be set",
            file=sys.stderr,
        )
        return 2
    if args.num_variants < 2:
        print(
            f"ERROR: --num-variants must be >= 2 (got {args.num_variants}); "
            f"1 = v0 のみ、この script 不要",
            file=sys.stderr,
        )
        return 2
    if not args.obb_root.is_dir():
        print(f"ERROR: obb-root not found: {args.obb_root}", file=sys.stderr)
        return 2
    if not args.aug_config_yaml.is_file():
        print(f"ERROR: aug-config-yaml not found: {args.aug_config_yaml}", file=sys.stderr)
        return 2
    if args.cuda_workers < 0 or (args.jobs > 1 and args.cuda_workers > args.jobs):
        print(
            f"ERROR: --cuda-workers {args.cuda_workers} は 0..--jobs ({args.jobs}) の範囲で指定",
            file=sys.stderr,
        )
        return 2

    aug_cfg = _load_aug_cfg(args.aug_config_yaml)
    if not bool(aug_cfg.get("enabled", False)):
        print(
            f"ERROR: aug config yaml has augmentation.enabled=false: {args.aug_config_yaml}",
            file=sys.stderr,
        )
        return 2

    from model.ramen_ori.image_aug import ImageAugPipeline  # noqa: PLC0415
    from model.ramen_ori.obb_cache import ObbPrecomputedCache  # noqa: PLC0415

    obb_cameras = [c.strip() for c in args.obb_cameras.split(",") if c.strip()]
    # main でも 1 回組み立てて manifest (top_k / cameras / classes) を worker 起動前に検証する
    main_cache = ObbPrecomputedCache(
        root=args.obb_root,
        ckpt_hash=args.obb_hash,
        camera_keys=obb_cameras,
        top_K=args.obb_top_k,
        num_classes=args.obb_num_classes,
        fps=args.obb_fps,
    )
    main_pipe = ImageAugPipeline(aug_cfg)

    camera_map = _load_camera_map(args.input_cache_root)
    cams = _list_input_cams(args.input_cache_root)
    plans = build_cam_plans(
        cams,
        camera_map=camera_map,
        covered_cameras=set(main_cache.covered_cameras),
        uses_geometric=main_pipe.uses_geometric,
        share_enabled=args.output_common_root is not None,
    )
    print(f"[precompute_aug_ramen_ori] camera_map: {camera_map}", flush=True)
    for cam, plan in plans.items():
        dest = "common" if plan.shared else "token/overlay"
        print(
            f"[precompute_aug_ramen_ori] cam {cam}: obb={plan.source_cam} "
            f"wrist={main_pipe.is_wrist(cam)} geometric={main_pipe.uses_geometric(cam)} "
            f"→ {dest}",
            flush=True,
        )
    if args.output_common_root is not None:
        if not any(p.shared for p in plans.values()):
            print("[precompute_aug_ramen_ori] WARN: common 共有できる cam が無い", flush=True)
        _link_shared_cams(
            plans,
            args.output_common_root,
            [r for r in (args.output_token_root, args.output_overlay_root) if r is not None],
        )

    # Issue #129 Phase I-0-3 (2026-09-01): lerobot_frame_cache_patch は cache root 直下の
    # `_cache_meta.json` から fps を読む。各 output root にも copy が必要。
    meta_src = args.input_cache_root / CACHE_META_NAME
    out_roots = [
        r
        for r in (args.output_token_root, args.output_overlay_root, args.output_common_root)
        if r is not None
    ]
    if meta_src.is_file():
        for out_root in out_roots:
            out_root.mkdir(parents=True, exist_ok=True)
            meta_dst = out_root / CACHE_META_NAME
            if not meta_dst.is_file():
                meta_dst.write_bytes(meta_src.read_bytes())
                print(
                    f"[precompute_aug_ramen_ori] copied {CACHE_META_NAME} to {out_root}",
                    flush=True,
                )
    else:
        print(
            f"[precompute_aug_ramen_ori] WARN: {meta_src} not found; "
            "training-side patch will need meta present to use baked cache",
            flush=True,
        )

    class_filter_list = _parse_class_filter(args.overlay_class_filter)

    print(
        f"[precompute_aug_ramen_ori] scanning {args.input_cache_root} for v0 frames...",
        flush=True,
    )
    v0_frames = _collect_all_v0_frames(args.input_cache_root)
    n_frames = len(v0_frames)
    if n_frames == 0:
        print(f"ERROR: no v0 frames found under {args.input_cache_root}", file=sys.stderr)
        return 2

    expected_per_dir: Counter[Path] = Counter(
        v0.relative_to(args.input_cache_root).parent for v0 in v0_frames
    )
    print(
        f"[precompute_aug_ramen_ori] {n_frames:,} v0 frames in {len(expected_per_dir)} mp4 dirs "
        f"× {args.num_variants - 1} variants (jobs={args.jobs}, cuda_workers={args.cuda_workers}, "
        f"overlay conf={args.overlay_conf}, class_filter={class_filter_list})",
        flush=True,
    )

    if args.max_frames is not None:
        v0_frames = v0_frames[: args.max_frames]
        n_frames = len(v0_frames)
        print(f"[precompute_aug_ramen_ori] --max-frames: first {n_frames:,} frames only", flush=True)
    tasks = [(v0, args.num_variants, args.force) for v0 in v0_frames]
    initargs = (
        str(args.obb_root),
        args.obb_hash,
        obb_cameras,
        args.obb_top_k,
        args.obb_num_classes,
        args.obb_fps,
        plans,
        args.overlay_conf,
        args.overlay_thickness,
        class_filter_list,
        args.jpg_quality,
        aug_cfg,
        str(args.input_cache_root),
        str(args.output_token_root) if args.output_token_root else None,
        str(args.output_overlay_root) if args.output_overlay_root else None,
        str(args.output_common_root) if args.output_common_root else None,
        args.cuda_workers,
        args.torch_threads,
        args.gpu_jpeg_encode,
    )

    done_per_dir: Counter[Path] = Counter()
    mismatch_dirs: set[Path] = set()
    stats: dict[str, _CamStats] = {cam: _CamStats() for cam, p in plans.items() if p.source_cam}
    t0 = time.monotonic()
    total_saved = 0
    total_skipped = 0

    per_device: Counter[str] = Counter()
    pool = None
    if args.jobs <= 1:
        _worker_init(*initargs)
        results = map(_process_one_frame, tasks)
    else:
        # CUDA は fork 後の子で初期化できないので、GPU worker を含むときは spawn
        ctx = mp.get_context("spawn") if args.cuda_workers > 0 else mp.get_context()
        counter = ctx.Value("i", 0)
        pool = ctx.Pool(
            processes=args.jobs, initializer=_worker_init, initargs=(*initargs, counter)
        )
        results = pool.imap_unordered(_process_one_frame, tasks, chunksize=args.chunksize)

    try:
        for i, (v0_path, saved, skipped, obb_hit, n_boxes, device) in enumerate(results, 1):
            total_saved += saved
            total_skipped += skipped
            per_device[device] += 1
            rel_dir = v0_path.relative_to(args.input_cache_root).parent
            cam = rel_dir.parts[0]
            if obb_hit is not None:
                s = stats[cam]
                s.frames += 1
                s.hits += int(obb_hit)
                s.box_frames += int(n_boxes > 0)
                if not obb_hit:
                    mismatch_dirs.add(rel_dir)
                if s.frames >= args.guard_min_frames and s.hits == 0:
                    print(
                        f"ERROR: {cam} は OBB cache が覆う cam なのに {s.frames} frame で lookup が "
                        "1 件も当たらない (cam 名 / mp4 layout の不一致)。中断します",
                        file=sys.stderr,
                    )
                    return 3
            done_per_dir[rel_dir] += 1
            if done_per_dir[rel_dir] == expected_per_dir[rel_dir] and rel_dir not in mismatch_dirs:
                _write_done_markers(
                    rel_dir,
                    plans[cam],
                    frames=expected_per_dir[rel_dir],
                    num_variants=args.num_variants,
                    token_root=args.output_token_root,
                    overlay_root=args.output_overlay_root,
                    common_root=args.output_common_root,
                    jpg_subsampling="4:4:4" if args.gpu_jpeg_encode else "4:2:0",
                )
            if i % 500 == 0 or i == n_frames:
                elapsed = time.monotonic() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (n_frames - i) / rate if rate > 0 else float("inf")
                cuda_share = per_device["cuda"] / i
                print(
                    f"[precompute_aug_ramen_ori] {i:,}/{n_frames:,} frames processed, "
                    f"saved={total_saved:,} skipped={total_skipped:,}, "
                    f"rate={rate:.1f} frame/s, eta={eta/60:.1f} min, "
                    f"gpu_share={cuda_share:.0%}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    elapsed = time.monotonic() - t0
    print(
        f"[precompute_aug_ramen_ori] DONE in {elapsed/60:.1f} min. "
        f"total saved: {total_saved:,}, skipped (already existed): {total_skipped:,}, "
        f"frames by device: {dict(per_device)}"
    )
    for cam, s in stats.items():
        print(
            f"[precompute_aug_ramen_ori] overlay {cam}: frames={s.frames:,} "
            f"obb_hits={s.hits:,} frames_with_boxes={s.box_frames:,}",
            flush=True,
        )
    if mismatch_dirs:
        print(
            f"ERROR: OBB lookup が外れた frame を含む mp4 dir が {len(mismatch_dirs)} 個 "
            f"(完了マーカーは書いていない): {sorted(str(d) for d in mismatch_dirs)[:5]}",
            file=sys.stderr,
        )
        return 4
    empty = [cam for cam, s in stats.items() if s.frames > 0 and s.box_frames == 0]
    if empty:
        print(f"ERROR: overlay が 1 枚も描かれていない cam: {empty}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
