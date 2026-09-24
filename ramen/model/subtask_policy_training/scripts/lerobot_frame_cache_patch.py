"""LeRobot ``decode_video_frames`` の JPG frame cache monkey-patch (Issue #122)。

# 目的
H.264 圧縮 mp4 の random access decode は per-frame 30-100ms かかり train iter の
律速 (data_s 78%) になっている。事前展開した per-frame JPG cache を read することで
decode を 10-30x 高速化する。

# 対応 LeRobot version
- **0.5.1** (RAMEN-Ori が使う fork ramen branch): `decode_video_frames(path, ts, tol, backend=None)`
- **0.6.0** (GR00T が pip install する PyPI 版): `decode_video_frames(path, ts, tol, backend=None,
  return_uint8=False, is_depth=False)`

追加引数を ``**extra_kwargs`` で受け、特別 mode (``return_uint8``/``is_depth``) の時は
fallback (元関数を呼ぶ) で挙動不変を保証。default RGB float32 mode でのみ cache を効かせる。

# 使い方
- 環境変数 ``LEROBOT_FRAME_CACHE_ENABLE=true`` を set
- ``apply_patch()`` を lerobot import 後に呼ぶ (idempotent)。
    - RAMEN-Ori: ``data_lerobot.py`` の ``_patch_lerobot_load_tasks()`` 直後
    - GR00T: ``lerobot_train_with_frame_cache.py`` wrapper が invoke 前に apply

# Cache layout (data/bitrobot_lerobot_subtask_datasets/scripts/precompute_frame_cache.py と対応)
    <lerobot_root>/frame_cache/
        _cache_meta.json                    # {"fps": 30.0, "video_keys": [...], ...}
        <cam_key>/<chunk-XXX>/<file-YYY>/
            _frame_count.txt
            frame_000000.jpg
            ...

# Fallback 段階
1. env var 未設定 or false → 元関数
2. is_depth=True (0.6.0 depth encoding、dequantize が特殊、future work) → 元関数
3. video_path 構造が想定外 → 元関数
4. cache dir 未作成 or _cache_meta.json 欠 → 元関数
5. 対応 frame_idx.jpg 欠 (partial cache) → 元関数 (whole call、mixed 結果を返さない)

`FRAME_CACHE_STRICT=true` の時は上記いずれも元関数を呼ばず RuntimeError (video を読まない)。

# 対応 mode
- default (RGB float32 [0,1] CHW): cache hit
- return_uint8=True (0.6.0 GR00T factory の常用 mode): cache hit
  _read_jpg_as_frame_tensor が normalize skip で uint8 CHW を返す
  (torchcodec return_uint8=True 出力形式に一致、mean abs diff 1.02/255 = visually lossless)
- is_depth=True: fallback (上記 fallback 2)

# 追加 patch (Issue #122)
1. `from .video_utils import decode_video_frames` local binding も差替
   (dataset_reader / lerobot_dataset module が実 decode に使う binding)
2. `_check_cached_episodes_sufficient` の video 存在 loop skip
mp4 auto drop 後は lerobot original の `_check_cached_episodes_sufficient` が
video_path.exists() で False を返し、`_download()` → snapshot_download が
materialized flat view を raw sub-feature parquet で上書きしてしまう。
Patch は video 存在 loop だけ skip し、parquet 側 check は残す。JPG cache 欠損時は
`cached_decode_video_frames` の fallback で torchcodec 経由の loud fail に落ちる。

# decode_video_frames の local binding 差替 (Issue #122 追加)
`from .video_utils import decode_video_frames` は import 時に元関数を module
attribute として bind する。実 decode 経路 (0.6.0: dataset_reader._decode_single,
0.5.1: LeRobotDataset method) はその local binding を使うため、video_utils だけ
差替ても cache 素通り。identity 一致 (元関数と同じ object) の場合のみ差替、
user override は保護する。
"""

from __future__ import annotations

import importlib
import json
import os
import random
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

CACHE_META_NAME = "_cache_meta.json"
ENV_ENABLE = "LEROBOT_FRAME_CACHE_ENABLE"
# Issue #129: offline aug precompute で生成した N variants から random 選択
# (>=2 で有効化、default 1 = v0 のみ = 従来挙動 backward compat)
ENV_NUM_VARIANTS = "FRAME_CACHE_NUM_VARIANTS"
# Issue #129 Phase I-0-3 (2026-09-01): baked cache dir を明示的に指定する env。
# 通常は video_path 構造から `<root>/frame_cache/<cam>/<chunk>/<file>/` を導出するが、
# RAMEN-Ori では token/ と overlay/ の 2 種 baked cache を切替える必要があるため、
# `<override_root>/<cam>/<chunk>/<file>/` に redirect する。
# 空文字列 or 未設定 = 従来挙動 (video_path 由来)、backward compat 完全保持。
ENV_ROOT_OVERRIDE = "FRAME_CACHE_ROOT_OVERRIDE"
# Issue #139: true で cache miss 時に mp4 decode へ落とさず RuntimeError にする。
# 「学習中に video を読まない」を保証したい run (ACT / DP) 用。既定 off で従来挙動。
ENV_STRICT = "FRAME_CACHE_STRICT"

_patch_lock = threading.Lock()
_patch_applied = False
_original_decode: Callable[..., torch.Tensor] | None = None

# Issue #122 D-3: post-decode hook registry (OBB overlay 描画用)。
# Hook signature: `fn(img_bgr: np.ndarray[H, W, 3] uint8, video_path: Path, frame_idx: int) -> np.ndarray[H, W, 3] uint8`
# `_read_jpg_as_frame_tensor` が cv2.imread 直後 (uint8 HWC BGR)、tensor 化前に順次適用。
# hook は image を書き換えて return する (in-place でも copy でも可)。
# 単一 dataloader instance 前提 (register/clear は dataloader init 時のみ想定)。
PostDecodeHook = Callable[[np.ndarray, "Path", int], np.ndarray]
_post_decode_hooks: list[PostDecodeHook] = []


def register_post_decode_hook(fn: PostDecodeHook) -> None:
    """`_read_jpg_as_frame_tensor` の cv2.imread 直後に走らせる hook を登録。

    OBB overlay 描画 (raw 640×480 uint8 HWC の段階で cv2 polylines) で使う。
    dataloader init 時に 1 回だけ呼ぶ想定。既存 hooks には append される。
    """
    _post_decode_hooks.append(fn)


def clear_post_decode_hooks() -> None:
    """登録済 hook を全 clear (test / dataloader teardown 用)。"""
    _post_decode_hooks.clear()


# Issue #129 Phase K (2026-09-01): source-root prefix keyed override registry。
# 単一 process 内で複数 RamenOriLerobotDataset (multi wrapper) を build する時、
# env `FRAME_CACHE_ROOT_OVERRIDE` は global 状態で per-instance 上書きになるため
# 2 sub 目以降が 1 sub 目の env を潰し、両 sub とも最後に登録された override を使う
# = 両 sub が同じ (誤った) cache を読む bug が起きる。registry で source_root prefix
# ごとに独立した override を保持し、video_path の prefix match で lookup する。
# {source_root_resolved: {"override_root": Path, "num_variants": int}}
_OVERRIDE_REGISTRY: dict[Path, dict[str, Any]] = {}


def register_frame_cache_override(
    source_root: Path | str,
    override_root: Path | str,
    num_variants: int = 1,
) -> None:
    """Per-source-root cache override を登録 (multi-dataset safe)。

    `source_root` は sub-dataset の LeRobot dataset root (mp4 videos の親 dir、
    e.g. `/nvme/rotate_table_base`)。この prefix 配下の video_path が渡された時、
    cache dir は `<override_root>/<cam>/<chunk>/<file>/` に redirect され、
    variant sampling は `num_variants` を使う。

    冪等: 同一 source_root を再登録すると値が更新される。空文字列 override / 0 variants
    は valid (fallback 用途)、env 経路の意味論と揃える。
    """
    resolved = Path(source_root).resolve()
    _OVERRIDE_REGISTRY[resolved] = {
        "override_root": Path(override_root),
        "num_variants": int(num_variants),
    }


def unregister_frame_cache_override(source_root: Path | str) -> None:
    """Registry から entry を削除 (test teardown 用)。存在しない source_root は無視。"""
    resolved = Path(source_root).resolve()
    _OVERRIDE_REGISTRY.pop(resolved, None)


def clear_frame_cache_registry() -> None:
    """Registry を全 clear (test 用)。"""
    _OVERRIDE_REGISTRY.clear()


def _lookup_registry_for(video_path: Path) -> tuple[Path, int] | None:
    """video_path の prefix match で登録済 override を lookup。

    複数 prefix が match する場合、最長 prefix が勝つ (nested source root 対応)。
    return (override_root, num_variants) or None (未登録)。
    """
    if not _OVERRIDE_REGISTRY:
        return None
    try:
        resolved = video_path.resolve()
    except (OSError, RuntimeError):
        resolved = video_path
    best_entry: dict | None = None
    best_len = -1
    for source_root, entry in _OVERRIDE_REGISTRY.items():
        try:
            resolved.relative_to(source_root)
        except ValueError:
            continue
        # 長い prefix (深い path) が優先
        depth = len(source_root.parts)
        if depth > best_len:
            best_entry = entry
            best_len = depth
    if best_entry is None:
        return None
    return best_entry["override_root"], int(best_entry["num_variants"])

# _check_cached_episodes_sufficient の original を version 別 class 単位で保持 (rollback 用)。
# 0.5.1 (RAMEN-Ori fork) と 0.6.0 (GR00T pip) で class 位置が違う (下記 _check_patch_targets)
_original_checks: dict[type, Callable[..., bool]] = {}
_check_patch_targets: list[tuple[str, str]] = [
    ("lerobot.datasets.dataset_reader", "DatasetReader"),      # 0.6.0
    ("lerobot.datasets.lerobot_dataset", "LeRobotDataset"),    # 0.5.1
]

# `from .video_utils import decode_video_frames` で local bind した module 名。
# apply_patch() は video_utils.decode_video_frames を差替るが、この 2 module は
# import 時に元関数を local attribute として bind 済のため、その binding を使う
# 実 decode 経路 (dataset_reader._decode_single / LeRobotDataset method) が cache
# patch を素通りする (Issue #122)。identity 一致 (元関数と同じ object) の場合のみ差替。
_decode_bind_targets: list[str] = [
    "lerobot.datasets.dataset_reader",   # 0.6.0
    "lerobot.datasets.lerobot_dataset",  # 0.5.1
]
_patched_bind_modules: list[str] = []  # 差替した module 名 (reset_for_test rollback 用)


def is_enabled() -> bool:
    """Env var で patch が enable されているか。"""
    return os.environ.get(ENV_ENABLE, "false").lower() in ("1", "true", "yes")


def _derive_cache_dir_from_path(video_path: Path) -> Path | None:
    """Path structure から cache subdir を pure に導出 (実在 check せず、shape のみ)。

    `<root>/videos/<cam>/chunk-XXX/file-YYY.mp4` → `<root>/frame_cache/<cam>/chunk-XXX/file-YYY/`
    precompute_frame_cache.py の cache layout と対応。
    """
    parts = video_path.parts
    if len(parts) < 4:
        return None
    if parts[-4] != "videos":
        return None
    root = Path(*parts[:-4])
    cam_key = parts[-3]
    chunk = parts[-2]
    file_stem = video_path.stem
    return root / "frame_cache" / cam_key / chunk / file_stem


def _get_root_override() -> Path | None:
    """`FRAME_CACHE_ROOT_OVERRIDE` env を Path で return、未設定 or 空文字列は None。"""
    v = os.environ.get(ENV_ROOT_OVERRIDE)
    if v is None or not v.strip():
        return None
    return Path(v.strip())


def _derive_override_cache_dir(video_path: Path, override_root: Path) -> Path | None:
    """override_root 下に `<cam>/<chunk>/<file>/` subdir を join した path を return。

    video_path は `<root>/videos/<cam>/chunk-XXX/file-YYY.mp4` 構造前提。
    override 導出は shape のみ、実在 check は呼出側で行う。
    """
    parts = video_path.parts
    if len(parts) < 4:
        return None
    if parts[-4] != "videos":
        return None
    cam_key = parts[-3]
    chunk = parts[-2]
    file_stem = video_path.stem
    return override_root / cam_key / chunk / file_stem


def _cache_dir_for_video_path(video_path: Path) -> Path | None:
    """Cache dir を探す (Issue #122 revised): symlink resolve で real path 側も探す。

    LeRobot は training view の path (symlink) で `decode_video_frames` を呼ぶが、
    cache は merged_source (symlink target) 側に置かれる (policy_type 越しに共有
    可能にするため)。両方 try:
      1. Given path そのままの cache dir (in-place cache 想定)
      2. Resolved (symlink 追尾後) の cache dir (merged_source 共有 cache 想定)

    Issue #129 Phase I-0-3 (2026-09-01): `FRAME_CACHE_ROOT_OVERRIDE` が set されて
    いれば、両 candidate の <cam>/<chunk>/<file>/ 部分を override_root 下に join した
    path を最優先で試す。RAMEN-Ori の token/ vs. overlay/ 切替に使う。
    Phase K (2026-09-01): registry 経路 (register_frame_cache_override) が先、env は
    fallback。multi-dataset 内で複数 sub が異なる override を要する場合、env global
    上書きでは対応できないため registry で source_root prefix keyed に管理する。

    どちらかが存在すれば return、無ければ None (呼出側で fallback)。
    """
    p = Path(video_path)
    candidates: list[Path] = [p]
    try:
        resolved = p.resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = p
    if resolved != p:
        candidates.append(resolved)

    # Issue #129 Phase K: registry 経路が最優先 (per-source-root override、multi safe)。
    reg_lookup = _lookup_registry_for(p)
    override_root: Path | None = None
    if reg_lookup is not None:
        override_root, _ = reg_lookup
    else:
        # Issue #129 Phase I-0-3: env fallback (単一 dataset の従来経路、backward compat)
        override_root = _get_root_override()

    if override_root is not None:
        for cand in candidates:
            cd = _derive_override_cache_dir(cand, override_root)
            if cd is not None and cd.is_dir():
                return cd

    # 通常 path 導出 (backward compat、override 無しの挙動)
    for cand in candidates:
        cd = _derive_cache_dir_from_path(cand)
        if cd is not None and cd.is_dir():
            return cd
    return None


@lru_cache(maxsize=None)
def _cache_fps(cache_root: Path) -> float | None:
    """`<cache_root>/_cache_meta.json` から fps を読む (LRU cache、per cache root)。"""
    meta_path = cache_root / CACHE_META_NAME
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    fps = meta.get("fps")
    if not isinstance(fps, (int, float)):
        return None
    return float(fps)


def _resolve_cache_root(cache_dir: Path) -> Path:
    """`<root>/frame_cache/<cam>/<chunk>/<file>` → `<root>/frame_cache`。"""
    return cache_dir.parent.parent.parent


def _get_num_variants() -> int:
    """Issue #129: env `FRAME_CACHE_NUM_VARIANTS` を parse。無効 or <2 なら 1 (v0 のみ)。

    ~2 以上で offline aug variant 選択が有効化される。precompute_augmented_frame_cache.py
    で生成した variant 数と一致させる (mismatch = variant idx 選ばれても jpg 不在 fallback)。
    """
    v = os.environ.get(ENV_NUM_VARIANTS)
    if v is None:
        return 1
    try:
        n = int(v)
    except ValueError:
        return 1
    return max(1, n)


def _resolve_variant_jpg_path(cache_dir: Path, frame_idx: int, num_variants: int) -> Path:
    """Frame idx から variant 込みの jpg path を返す (Issue #129 case F)。

    num_variants=1 (default) → 従来通り `frame_XXXXXX.jpg` (v0 = raw)
    num_variants>=2         → random.randint(1, num_variants-1) で v idx pick:
                              - v0 (raw、無 aug、無 overlay) は skip
                              - v_i (i=1..N-1) は precompute で aug + overlay baked 済

    Rationale: precompute_augmented_frame_cache.py (case F) は v0 を無変更で残し、
    v_1..v_(N-1) に photometric + OBB overlay + geometric を bake する。v0 を残す事で
    rerun 時の overlay 二重描画を避け、v0 は raw source として再生成 base に使う。
    Training では v0 を選択しない (canonical + overlay の inference-condition 画像は
    v_i の中で偶発的に aug が empty になる ~10% が近似)。
    """
    base_name = f"frame_{frame_idx:06d}"
    if num_variants <= 1:
        return cache_dir / f"{base_name}.jpg"
    v_idx = random.randint(1, num_variants - 1)  # v0 skip
    return cache_dir / f"{base_name}_v{v_idx}.jpg"


def _read_jpg_as_frame_tensor(
    jpg_path: Path,
    return_uint8: bool = False,
    hook_context: tuple[Path, int] | None = None,
) -> torch.Tensor:
    """JPG → torch.Tensor。default = RGB float32 CHW [0, 1] (元 decode_video_frames 出力に一致)、
    ``return_uint8=True`` 時は uint8 CHW をそのまま返す
    (torchcodec の ``return_uint8=True`` 出力形式に一致、GR00T factory が要求)。

    Issue #122 D-3: ``hook_context=(video_path, frame_idx)`` が渡された場合、
    cv2.imread 直後 (uint8 HWC BGR、raw 解像度) に登録済 post-decode hook を順次適用
    してから tensor 化する。OBB overlay 描画用 (raw で描画 → 後段 Resize で anti-alias)。
    """
    # cv2 の import は一部 platform で libgl race するため lazy (multiprocess fork 前に触らない)
    import cv2  # noqa: PLC0415

    img_bgr = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread failed for {jpg_path}")
    if hook_context is not None and _post_decode_hooks:
        video_path, frame_idx = hook_context
        for hook in _post_decode_hooks:
            img_bgr = hook(img_bgr, video_path, frame_idx)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # HWC uint8
    tensor = torch.from_numpy(np.ascontiguousarray(img_rgb)).permute(2, 0, 1)  # CHW uint8
    if return_uint8:
        return tensor
    return tensor.to(dtype=torch.float32) / 255.0


def is_strict() -> bool:
    return os.environ.get(ENV_STRICT, "").strip().lower() in ("1", "true", "yes", "on")


def _fallback(video_path, timestamps, tolerance_s, backend, extra_kwargs, reason: str) -> torch.Tensor:
    """元関数呼び出し (0.5.1/0.6.0 の signature 差分を吸収)。

    strict mode では mp4 を decode せずに止める。黙って decode に落ちると GPU が
    dataloader 待ちで遊ぶだけで、run は遅いまま進んでしまうため。
    """
    if is_strict():
        raise RuntimeError(
            f"frame cache miss ({reason}) under {ENV_STRICT}=true; refusing to decode "
            f"{video_path} at timestamps {list(timestamps)[:4]}"
        )
    assert _original_decode is not None
    return _original_decode(video_path, timestamps, tolerance_s, backend, **extra_kwargs)


def cached_decode_video_frames(
    video_path,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
    **extra_kwargs: Any,
) -> torch.Tensor:
    """LeRobot ``decode_video_frames`` の drop-in replacement (0.5.1 + 0.6.0 両対応)。

    Cache hit した frame は JPG から O(1) load、miss や special mode は 元関数に fallback。
    Default (RGB float32 [0,1] CHW) mode でのみ cache を効かせる。
    """
    if _original_decode is None:
        raise RuntimeError("lerobot_frame_cache_patch.apply_patch() has not been called")

    # env off → 全て fallback (挙動不変保証)
    if not is_enabled():
        return _fallback(video_path, timestamps, tolerance_s, backend, extra_kwargs, "cache disabled")

    # 0.6.0 の is_depth は cache 未対応 (depth encoding dequantize が特殊、future work)
    # return_uint8 は _read_jpg_as_frame_tensor で normalize skip 対応 (JPG は元々 uint8)
    if extra_kwargs.get("is_depth"):
        return _fallback(video_path, timestamps, tolerance_s, backend, extra_kwargs, "depth frames")

    cache_dir = _cache_dir_for_video_path(Path(video_path))
    if cache_dir is None:
        return _fallback(video_path, timestamps, tolerance_s, backend, extra_kwargs, "no cache dir")
    cache_root = _resolve_cache_root(cache_dir)
    fps = _cache_fps(cache_root)
    if fps is None:
        return _fallback(video_path, timestamps, tolerance_s, backend, extra_kwargs, "no cache meta")

    return_uint8 = bool(extra_kwargs.get("return_uint8", False))
    # Issue #122 D-3: post-decode hook (OBB overlay) が register 済なら
    # (video_path, frame_idx) を _read_jpg_as_frame_tensor に渡す。未 register 時
    # は context=None で hook 適用パスは走らない (挙動不変)。
    pass_hook_context = bool(_post_decode_hooks)
    # Issue #129: offline aug variant 選択 (N>=2 で有効化、per-call で random)。
    # 各 timestamp 独立に variant idx を pick、cache hit 時のみ suffix 付き jpg を試す。
    # Phase K (2026-09-01): multi-dataset で sub 別 num_variants が違う場合に備え、
    # registry 経由 lookup (video_path の source prefix match) を先に試す。未登録なら env fallback。
    reg_lookup = _lookup_registry_for(Path(video_path))
    if reg_lookup is not None:
        _, num_variants = reg_lookup
    else:
        num_variants = _get_num_variants()
    frames: list[torch.Tensor] = []
    for ts in timestamps:
        frame_idx = int(round(float(ts) * fps))
        jpg_path = _resolve_variant_jpg_path(cache_dir, frame_idx, num_variants)
        if not jpg_path.is_file():
            # partial cache miss → whole call fallback (mixed 結果を返さない)
            return _fallback(
                video_path, timestamps, tolerance_s, backend, extra_kwargs, f"missing {jpg_path}"
            )
        hook_context = (Path(video_path), frame_idx) if pass_hook_context else None
        frames.append(
            _read_jpg_as_frame_tensor(
                jpg_path, return_uint8=return_uint8, hook_context=hook_context
            )
        )
    return torch.stack(frames, dim=0)


def _make_patched_check(
    original: Callable[[Any], bool],
) -> Callable[[Any], bool]:
    """Video 不在を許容する ``_check_cached_episodes_sufficient`` (Issue #122)。

    Frame cache active 時、mp4 は precompute の auto drop で不在なのが正常状態。
    lerobot original は video_path.exists() を strict に見て False を返し、
    LeRobotDataset.__init__ が ``_download()`` を invoke → snapshot_download が
    raw sub-feature parquet を materialized flat view に上書きする (train 落ち)。

    Parquet 側 check (hf_dataset presence + requested episodes 完備) はそのまま残し、
    video path 存在 loop だけ skip する。JPG cache 欠損時は
    ``cached_decode_video_frames`` が torchcodec fallback で loud fail するので、
    silent 上書き risk は排除しつつ data 欠損検知は保つ。

    Env ``LEROBOT_FRAME_CACHE_ENABLE=false`` の時は original に fall through
    (patch install 済でも実挙動は変わらない、backward compat)。
    """

    def patched(self: Any) -> bool:
        if not is_enabled():
            return original(self)
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False
        # 0.5.1: self.meta / 0.6.0: self._meta (attribute name 差分)
        meta = getattr(self, "_meta", None) or getattr(self, "meta", None)
        if meta is None:
            return original(self)  # unexpected shape → 安全側で原挙動
        available = {
            (ep.item() if isinstance(ep, torch.Tensor) else ep)
            for ep in self.hf_dataset.unique("episode_index")
        }
        requested = (
            set(range(meta.total_episodes))
            if self.episodes is None
            else set(self.episodes)
        )
        return requested.issubset(available)

    return patched


def _install_decode_bind_patch() -> None:
    """`from .video_utils import decode_video_frames` で local bind した module 側の
    binding を差替 (Issue #122)。

    apply_patch() が ``video_utils.decode_video_frames`` を差替ても、
    ``dataset_reader`` / ``lerobot_dataset`` module は import 時に元関数を local
    attribute として bind 済で、実 decode 経路 (``_decode_single`` / class method)
    はその local binding を使う。この関数はその binding も差替る。

    Identity 一致 (``_original_decode`` と同じ object) の場合のみ差替、既に別関数に
    置換済 (user override 等) なら touch しない。idempotent: 差替済 module は skip。
    """
    if _original_decode is None:
        return  # apply_patch 前呼び出しは no-op
    for mod_name in _decode_bind_targets:
        if mod_name in _patched_bind_modules:
            continue
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        current = getattr(mod, "decode_video_frames", None)
        if current is _original_decode:
            setattr(mod, "decode_video_frames", cached_decode_video_frames)
            _patched_bind_modules.append(mod_name)


def _install_check_patch() -> None:
    """Both 0.5.1/0.6.0 の ``_check_cached_episodes_sufficient`` を duck typing で差替。

    片方が import 不可 or class 不在なら silent skip (両版が同 env に共存する事は無い)。
    Idempotent: 既に差替済 class は skip。
    """
    for module_name, class_name in _check_patch_targets:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        if cls in _original_checks:
            continue
        original = getattr(cls, "_check_cached_episodes_sufficient", None)
        if original is None:
            continue
        _original_checks[cls] = original
        setattr(
            cls,
            "_check_cached_episodes_sufficient",
            _make_patched_check(original),
        )


def apply_patch() -> bool:
    """``lerobot.datasets.video_utils.decode_video_frames`` と
    ``_check_cached_episodes_sufficient`` を monkey-patch (idempotent)。

    Env var ``LEROBOT_FRAME_CACHE_ENABLE=true`` が set されていなくても patch install
    自体は行う (実挙動は runtime で切替 = is_enabled() ガード、default off で backward compat)。
    重複呼び出しは no-op。

    # Worker propagation
    Linux default の ``fork`` start method では main process で apply_patch した状態が
    child worker に丸ごと継承される (patched module attribute 込み)。従って
    DataLoader(num_workers>0) でも各 worker で cache が効く (fork 前に apply_patch を
    呼んでいれば OK)。``spawn`` start method (macOS default / Windows / user が明示的に
    設定した Linux 環境) は worker が clean な interpreter で立ち上がるため patch が
    継承されない → cache 素通り。spawn を使う環境では DataLoader の
    ``worker_init_fn=lambda _: apply_patch()`` で per-worker install が必要。
    (この project は Sakura H100 Linux + RAMEN-Ori 別サーバ Linux 前提、fork のため対応不要)

    Returns:
        True: patch を適用した (or 既に適用済)。False: import 失敗 (lerobot 未 install)。
    """
    global _patch_applied, _original_decode
    with _patch_lock:
        if _patch_applied:
            return True
        try:
            from lerobot.datasets import video_utils  # noqa: PLC0415
        except ImportError:
            return False

        current = getattr(video_utils, "decode_video_frames", None)
        if current is None:
            return False
        if current is not cached_decode_video_frames:
            _original_decode = current
            video_utils.decode_video_frames = cached_decode_video_frames
        # `from .video_utils import decode_video_frames` で local bind した module も差替
        # (dataset_reader._decode_single / LeRobotDataset method 経路が cache を使う)
        _install_decode_bind_patch()
        # _check_cached_episodes_sufficient patch install (video 不在許容)
        _install_check_patch()
        _patch_applied = True
        return True


def reset_for_test() -> None:
    """Test 用 (patch 状態を initial に戻す)。production では使わない。"""
    global _patch_applied, _original_decode
    with _patch_lock:
        if _patch_applied and _original_decode is not None:
            try:
                from lerobot.datasets import video_utils  # noqa: PLC0415

                video_utils.decode_video_frames = _original_decode
            except ImportError:
                pass
        # bind 差替 (decode_video_frames の local binding) を rollback
        if _original_decode is not None:
            for mod_name in _patched_bind_modules:
                try:
                    mod = importlib.import_module(mod_name)
                    if getattr(mod, "decode_video_frames", None) is cached_decode_video_frames:
                        setattr(mod, "decode_video_frames", _original_decode)
                except ImportError:
                    pass
        _patched_bind_modules.clear()
        # _check_cached_episodes_sufficient patch を rollback
        for cls, original in _original_checks.items():
            try:
                setattr(cls, "_check_cached_episodes_sufficient", original)
            except (AttributeError, TypeError):
                pass
        _original_checks.clear()
        _patch_applied = False
        _original_decode = None
        _cache_fps.cache_clear()


if __name__ == "__main__":
    # CLI: apply patch and print status (setup verify 用)
    import sys as _sys

    ok = apply_patch()
    print(f"apply_patch: {ok}, enabled: {is_enabled()}", file=_sys.stderr)
    _sys.exit(0 if ok else 1)
