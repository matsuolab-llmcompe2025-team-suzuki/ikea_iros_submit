"""OBB precompute cache reader (Issue #122 D-2)。

`data/yolo_obb/scripts/precompute_yolo_obb.py` の出力を dataloader (RamenOriLerobotDataset)
から lookup するための薄い wrapper。

# 責務
- Manifest 検証 (schema_version / top_K / num_classes / cameras subset)
- LeRobot v3 の episode meta から `episode_index → {cam: (mp4_relative, start_frame_in_mp4)}` map を build
- `(episode_index, frame_index_in_ep, cam_key)` → top_K padded OBB tensor 一発 lookup
- Parquet load は per-(cam, mp4) で in-memory キャッシュ (LRU 相当、簡易 dict)

# データ受け渡し方針
- Precompute cache には head Left/Right (cam_0/1) の 2 cam 分しか parquet が無い前提
  (wrist に対しては YOLO 推論しない design decision、Issue #122 D-2 議論参照)
- 学習側 (camera_keys) は 4 cam で回るので、cache に無い cam (wrist) は
  dataloader 側で zero-fill + valid_mask=False で対応 → 本 module は
  `cam_key not in covered_cameras` の場合 `None` を返し、上位に丸投げ
- padding row の class_id は precompute 側で 0 (workspace) に統一、valid=False で下流 mask
  この collision (workspace vs padding が同 class_id) は Fusion / ActionExpert の
  2 段 key_padding_mask で gradient が完全に分離される (`obb_source="none"` mode と
  同じ safety guarantee、Issue #122 D-2 議論参照)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch


CACHE_SCHEMA_VERSION = "team_ramen_yolo_obb_cache_v1"
CACHE_MANIFEST_NAME = "manifest.json"


class ObbPrecomputedCache:
    """YOLO OBB precompute cache の read-only view。

    Args:
        root: precompute cache root (`outputs/yolo_obb_cache/`、その下に <hash> dir が並ぶ)
        ckpt_hash: 使用する hash dir 名。None なら root 配下に hash dir が
            1 個しか無い時それを auto-pick、複数あれば error (曖昧性を fail-fast)。
        camera_keys: 学習側 config で使う全 cam key list (例 4 cams)。
            manifest.cameras は必ずこの subset である事を verify。
        top_K: dataloader の top_K。manifest と一致 verify。
        num_classes: 期待 class 数。manifest と一致 verify。
        fps: dataset の fps (frame_in_mp4 = round(from_timestamp * fps) + frame_in_ep)。
    """

    def __init__(
        self,
        root: str | Path,
        ckpt_hash: str | None,
        camera_keys: list[str],
        top_K: int | None,
        num_classes: int,
        fps: int,
    ) -> None:
        """top_K: None なら manifest の値を使う (memory の入力は cache にある検出を全部読む、Issue #141 Phase 7)。"""
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"obb precompute root not found: {root}")

        if ckpt_hash is None:
            candidates = [p for p in root.iterdir() if p.is_dir() and (p / CACHE_MANIFEST_NAME).is_file()]
            if not candidates:
                raise FileNotFoundError(
                    f"no <hash>/manifest.json found under {root} "
                    f"(precompute_yolo_obb.py を実行して cache 生成する)"
                )
            if len(candidates) > 1:
                names = sorted(p.name for p in candidates)
                raise ValueError(
                    f"multiple hash dirs under {root}: {names}. "
                    f"obb_precomputed_hash= で明示指定してください "
                    f"(YOLO ckpt を差替えて複数併存する運用のため fail-fast)。"
                )
            ckpt_hash = candidates[0].name

        cache_dir = root / ckpt_hash
        manifest_path = cache_dir / CACHE_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported cache schema_version: got {manifest.get('schema_version')!r}, "
                f"expected {CACHE_SCHEMA_VERSION!r} (precompute script との version 不整合)"
            )
        if top_K is None:
            top_K = int(manifest["top_k"])
        if int(manifest["top_k"]) != int(top_K):
            raise ValueError(
                f"top_K mismatch: cache={manifest['top_k']}, dataloader={top_K}"
            )
        if int(manifest["num_classes"]) != int(num_classes):
            raise ValueError(
                f"num_classes mismatch: cache={manifest['num_classes']}, "
                f"dataloader={num_classes}"
            )
        covered_cameras = list(manifest["cameras"])
        # Issue #129 Phase K (2026-09-01): cam 集合の整合チェックを緩和。
        # 元設計: cache に無い cam (wrist 等) は dataloader が None fallback で 0-tensor 化、
        #        cache extras (dataloader が使わない cam) は禁止 (config error 検知目的)。
        # 変更後: 「intersection が非空」だけ要求 (完全 disjoint な cache = 別 task 用の
        #        誤 load を検知)、subset / superset の非対称は許容:
        #          - cache subset of dataloader: wrist 等の穴埋め 0 fallback (元設計)
        #          - cache superset of dataloader: Phase A の 4→3 cam 縮小 (今 relax する側)
        intersect = [c for c in covered_cameras if c in camera_keys]
        if not intersect:
            raise ValueError(
                f"cache/dataloader camera_keys disjoint: cache covered={covered_cameras}, "
                f"dataloader camera_keys={camera_keys}. 別 task 用 OBB cache の誤 load 疑い、"
                "obb_precomputed_root / obb_precomputed_hash を確認。"
            )
        extras = [c for c in covered_cameras if c not in camera_keys]
        if extras:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "ObbPrecomputedCache: cache has extra cams %s not used by dataloader "
                "(camera_keys=%s); ignoring silently (Phase A 4→3 cam 縮小前の cache でも動く)",
                extras, camera_keys,
            )

        self.root = root
        self.ckpt_hash = ckpt_hash
        self.cache_dir = cache_dir
        self.manifest = manifest
        self.camera_keys = list(camera_keys)
        self.covered_cameras: set[str] = set(covered_cameras)
        self.top_K = int(top_K)
        self.num_classes = int(num_classes)
        self.fps = int(fps)

        # episode mapping: {ep_idx: {cam_key: (mp4_relative_str, start_frame_in_mp4)}}
        self._ep_mapping: dict[int, dict[str, tuple[str, int]]] = {}
        # parquet in-memory cache: {(cam_key, mp4_relative_str): {frame_idx: np.ndarray[(top_K, ...)]}}
        # 実際は per-frame lookup を np.array で index するので (frame_index → row range) の
        # 高速 index 用に dict[int, dict{verts,conf,class_id,valid}[array]] を持つ
        self._parquet_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Episode mapping build
    # ------------------------------------------------------------------

    def build_episode_mapping(self, lerobot_meta_episodes: Any) -> None:
        """LeRobot v3 の meta.episodes (HF Dataset or pandas DataFrame) から map を構築。

        必要な per-episode row column:
          - episode_index
          - videos/<cam_key>/chunk_index
          - videos/<cam_key>/file_index
          - videos/<cam_key>/from_timestamp

        covered_cameras に無い cam は無視 (dataloader 側で zero padding)。
        """
        iterator = (
            lerobot_meta_episodes.iterrows()
            if hasattr(lerobot_meta_episodes, "iterrows")
            else lerobot_meta_episodes
        )
        mapping: dict[int, dict[str, tuple[str, int]]] = {}
        for row in iterator:
            if isinstance(row, tuple):  # pandas iterrows は (idx, row)
                row = row[1]
            ep_idx = int(row["episode_index"])
            per_cam: dict[str, tuple[str, int]] = {}
            for cam in sorted(self.covered_cameras):
                try:
                    chunk_idx = int(row[f"videos/{cam}/chunk_index"])
                    file_idx = int(row[f"videos/{cam}/file_index"])
                    from_ts = float(row[f"videos/{cam}/from_timestamp"])
                except KeyError as e:
                    raise KeyError(
                        f"episode {ep_idx}: meta row missing video column {e} "
                        f"(cache camera {cam} not present in LeRobot meta)"
                    ) from e
                mp4_rel = f"chunk-{chunk_idx:03d}/file-{file_idx:03d}"
                start_frame = int(round(from_ts * self.fps))
                per_cam[cam] = (mp4_rel, start_frame)
            mapping[ep_idx] = per_cam
        self._ep_mapping = mapping

    # ------------------------------------------------------------------
    # Parquet load + lookup
    # ------------------------------------------------------------------

    def _load_parquet(self, cam_key: str, mp4_relative: str) -> dict[str, np.ndarray]:
        """cam × mp4 の parquet を初回のみ read + numpy 配列に materialize、以降 cache。

        Return: {"verts": (N_frames, top_K, 8), "conf": (N_frames, top_K),
                 "class_id": (N_frames, top_K), "valid": (N_frames, top_K),
                 "n_frames": int}
        """
        key = (cam_key, mp4_relative)
        if key in self._parquet_cache:
            return self._parquet_cache[key]

        parquet_path = self.cache_dir / cam_key / f"{mp4_relative}.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"OBB parquet not found: {parquet_path} "
                f"(precompute_yolo_obb.py の出力を確認、ckpt_hash / cameras 不一致?)"
            )
        table = pq.read_table(parquet_path)
        n_rows = table.num_rows
        expected_rows_per_frame = self.top_K
        if n_rows % expected_rows_per_frame != 0:
            raise ValueError(
                f"parquet row count {n_rows} not divisible by top_K={self.top_K}: "
                f"{parquet_path}"
            )
        n_frames = n_rows // expected_rows_per_frame

        # rows are ordered by (frame_index asc, det_idx asc) from precompute script
        # Python の list を経由しない (top_k 32 の cache は 1 file で数百万行、Issue #141 Phase 7)
        frame_index_arr = table.column("frame_index").to_numpy().astype(np.int32)
        det_idx_arr = table.column("det_idx").to_numpy().astype(np.int8)
        # verts は fixed_size_list<float32, 8>: 中身を平らにして (n_rows, 8) に戻す
        verts_flat = (
            table.column("verts").combine_chunks().flatten().to_numpy().astype(np.float32).reshape(n_rows, 8)
        )
        conf_flat = table.column("conf").to_numpy().astype(np.float32)
        class_id_flat = table.column("class_id").to_numpy().astype(np.int8)
        valid_flat = table.column("valid").to_numpy().astype(bool)

        # verify ordering (frame_index が [0, 0, ..., 0(x top_K), 1, 1, ...])
        expected_frame = np.repeat(np.arange(n_frames, dtype=np.int32), self.top_K)
        if not np.array_equal(frame_index_arr, expected_frame):
            raise ValueError(
                f"parquet row order unexpected in {parquet_path}: "
                f"first {min(8, n_rows)} frame_index = {frame_index_arr[:8].tolist()}"
            )
        expected_det = np.tile(np.arange(self.top_K, dtype=np.int8), n_frames)
        if not np.array_equal(det_idx_arr, expected_det):
            raise ValueError(f"parquet det_idx order unexpected in {parquet_path}")

        data = {
            "verts": verts_flat.reshape(n_frames, self.top_K, 8),
            "conf": conf_flat.reshape(n_frames, self.top_K),
            "class_id": class_id_flat.reshape(n_frames, self.top_K),
            "valid": valid_flat.reshape(n_frames, self.top_K),
            "n_frames": n_frames,
        }
        self._parquet_cache[key] = data
        return data

    def lookup_mp4_frame(
        self, cam_key: str, mp4_relative: str, frame_idx_in_mp4: int
    ) -> dict[str, np.ndarray] | None:
        """Direct lookup by mp4 semantic (cache-native, ep mapping 不要)。

        overlay hook 用 (JPG cache patch level で `(cam_key, mp4_relative,
        frame_idx_in_mp4)` から直接引く)。

        - cam_key が cache に無い (wrist) → None
        - frame_idx が range 外 → None (overlay 側で skip、error にしない)

        Return: {"verts": (top_K, 8) float32, "conf": (top_K,) float32,
                 "class_id": (top_K,) int8, "valid": (top_K,) bool}
        """
        if cam_key not in self.covered_cameras:
            return None
        try:
            data = self._load_parquet(cam_key, mp4_relative)
        except FileNotFoundError:
            return None
        if not (0 <= frame_idx_in_mp4 < data["n_frames"]):
            return None
        return {
            "verts": data["verts"][frame_idx_in_mp4],
            "conf": data["conf"][frame_idx_in_mp4],
            "class_id": data["class_id"][frame_idx_in_mp4],
            "valid": data["valid"][frame_idx_in_mp4],
        }

    def episode_arrays(self, episode_index: int, length: int, cam_key: str) -> dict[str, np.ndarray]:
        """1 episode 分 (frame 0..length-1) の検出をまとめて返す (memory の表を作る用、Issue #141 Phase 7)。

        Return: {"verts": (length, top_K, 8), "conf": (length, top_K), "class_id": (length, top_K), "valid": (length, top_K)}
        """
        mp4_rel, start_frame = self._ep_mapping[episode_index][cam_key]
        data = self._load_parquet(cam_key, mp4_rel)
        if start_frame + length > data["n_frames"]:
            raise IndexError(
                f"ep={episode_index} frames [{start_frame}, {start_frame + length}) out of range "
                f"[0, {data['n_frames']}) in {cam_key}/{mp4_rel}"
            )
        sl = slice(start_frame, start_frame + length)
        return {k: data[k][sl] for k in ("verts", "conf", "class_id", "valid")}

    def lookup_frame(
        self, episode_index: int, frame_index_in_ep: int, cam_key: str
    ) -> dict[str, np.ndarray] | None:
        """(ep, frame_in_ep, cam) の top_K OBB を返す。cache に無い cam は None。

        Return: {"verts": (top_K, 8) float32, "conf": (top_K,) float32,
                 "class_id": (top_K,) int8, "valid": (top_K,) bool}
        """
        if cam_key not in self.covered_cameras:
            return None
        if episode_index not in self._ep_mapping:
            raise KeyError(
                f"episode {episode_index} not in mapping "
                f"(build_episode_mapping 未実行 or ep_idx が meta と不整合)"
            )
        mp4_rel, start_frame = self._ep_mapping[episode_index][cam_key]
        frame_in_mp4 = start_frame + int(frame_index_in_ep)
        data = self._load_parquet(cam_key, mp4_rel)
        if not (0 <= frame_in_mp4 < data["n_frames"]):
            raise IndexError(
                f"frame_in_mp4={frame_in_mp4} out of range [0, {data['n_frames']}) "
                f"for ep={episode_index} frame_in_ep={frame_index_in_ep} cam={cam_key} mp4={mp4_rel}"
            )
        return {
            "verts": data["verts"][frame_in_mp4],
            "conf": data["conf"][frame_in_mp4],
            "class_id": data["class_id"][frame_in_mp4],
            "valid": data["valid"][frame_in_mp4],
        }

    # ------------------------------------------------------------------
    # Batch dataloader 用: 全 cam × top_K の tensor を組み立てる
    # ------------------------------------------------------------------

    def make_obb_tensors(
        self, episode_index: int, frame_index_in_ep: int, cam_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """全 cam 分の (verts, conf, class_id, cam_id, valid_mask) tensor を返す。

        cache に無い cam (wrist) は zeros + valid=False で埋める。dummy 実装と同じ shape。

        Args:
            episode_index: LeRobot ep index (build_episode_mapping で key 済み)
            frame_index_in_ep: ep 内 frame index (0..length-1)
            cam_ids: (N_cams,) long tensor、_cam_id と同 layout (dataloader 側で保持)

        Returns:
            dict with:
              verts:      (N_cams, top_K, 8)  float32
              conf:       (N_cams, top_K, 1)  float32
              class_id:   (N_cams, top_K)     long
              cam_id:     (N_cams, top_K)     long
              valid_mask: (N_cams, top_K)     bool
        """
        N_cams = len(self.camera_keys)
        top_K = self.top_K
        verts = np.zeros((N_cams, top_K, 8), dtype=np.float32)
        conf = np.zeros((N_cams, top_K), dtype=np.float32)
        class_id = np.zeros((N_cams, top_K), dtype=np.int64)
        valid_mask = np.zeros((N_cams, top_K), dtype=bool)

        for i, cam in enumerate(self.camera_keys):
            per = self.lookup_frame(episode_index, frame_index_in_ep, cam)
            if per is None:
                # cache 外の cam (wrist 等) → zeros + valid False で埋め済
                continue
            verts[i] = per["verts"]
            conf[i] = per["conf"]
            class_id[i] = per["class_id"].astype(np.int64)
            valid_mask[i] = per["valid"]

        # padding row の class_id が -1 なら 0 (workspace) にクランプ (embedding OOB 回避)。
        # Fusion の key_padding_mask が valid=False の gradient を完全 filter する前提。
        class_id = np.where(class_id < 0, 0, class_id)

        cam_ids_2d = cam_ids.unsqueeze(1).expand(N_cams, top_K).contiguous()
        return {
            "verts": torch.from_numpy(verts),
            "conf": torch.from_numpy(conf).unsqueeze(-1),  # (N_cams, top_K, 1)
            "class_id": torch.from_numpy(class_id),
            "cam_id": cam_ids_2d,
            "valid_mask": torch.from_numpy(valid_mask),
        }
