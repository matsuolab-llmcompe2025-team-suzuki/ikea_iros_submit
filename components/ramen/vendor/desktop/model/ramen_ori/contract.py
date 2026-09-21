"""学習と推論の約束 (Issue #141 束 2-14)。

学習が定義を決めて ckpt (`ckpt["contract"]`) に書き、推論はそれを読んで同じ変換をする
(読み取りと起動時の確認は推論側)。正規化と FK の統計は model の buffer (model_state_dict) に入るので、ここには書かない。
aug の設定は学習でしか使わないので書かない。

中身 (version 1):

| key | 中身 |
|---|---|
| skills | 学習した skill_id と skill ごとの frame 数 (推論は slot の skill がここに含まれるかを確かめる) |
| state | 71D の並びと定義の版 (`state_derive.STATE71_DEFINITION`) |
| action | abs / rel、次元 (腕 14 + hand 2)、chunk_len |
| normalization | state / action を model の中で正規化するか |
| images | カメラ、画像の大きさ、前 frame の有無、画像の出どころ、overlay の条件 (YOLO の重み・閾値・線の太さ) |
| memory | memory の入力の定義 (版、51 個の並び、class、conf、τ、カメラ、OBB の YOLO の重み)。使わない run は null |
| provenance | 学習のコードの git commit と、未 commit の変更があったか |

baked_overlay の画像は焼き込み済みなので、overlay の条件は焼き込みの設定 file (`cfg.contract.bake_config`、
`data/bitrobot_lerobot_subtask_datasets/configs/unified_frame_cache.yaml`) から読む。
cache 自体は焼き込みの条件を持たない (`_cache_meta.json` は fps などだけ)。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import yaml
from omegaconf import DictConfig

from model.ramen_ori.memory_features import (
    CONF_MIN,
    FPS,
    MEMORY_CAMERA,
    MEMORY_LAYOUT,
    MEMORY_VERSION,
    MULTI_CLASSES,
    SINGLE_CLASSES,
    TAUS_S,
)
from model.ramen_ori.skill_mapping import skill_id_name
from model.ramen_ori.state_derive import STATE71_DEFINITION

CONTRACT_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_contract(cfg: DictConfig, dataset) -> dict:
    """学習の開始時に 1 回だけ作る。dataset は RamenOriLerobotDataset / RamenOriMultiDataset。"""
    skill_ids, frames = np.unique(dataset.sample_skill_ids(), return_counts=True)
    return {
        "version": CONTRACT_VERSION,
        "skills": [
            {"id": int(s), "name": skill_id_name(int(s)), "frames": int(n)}
            for s, n in zip(skill_ids, frames)
        ],
        "state": {"variant": dataset.state_variant, **STATE71_DEFINITION},
        "action": {
            "space": "rel" if dataset.use_relative_action else "abs",
            "dim": int(cfg.model.action_dim),
            "layout": "左腕 7 + 右腕 7 + hand 2 (腰は出力しない)",
            "chunk_len": int(cfg.model.chunk_len),
        },
        "normalization": bool(cfg.model.normalization.enabled),
        "images": _images(cfg, dataset),
        "memory": _memory(dataset),
        "provenance": _git_provenance(),
    }


def _memory(dataset) -> dict | None:
    """memory の入力 (Issue #141 Phase 7)。推論は同じ MemoryTracker を skill の開始時に reset して tick ごとに回す。
    切り詰めの範囲と統計は model の buffer (memory.clip_* / memory.normalizer.*)。"""
    if not dataset.memory:
        return None
    return {
        "version": MEMORY_VERSION,
        "layout": list(MEMORY_LAYOUT),
        "camera": MEMORY_CAMERA,
        "yolo_ckpt": dataset.memory_yolo_ckpt,
        "conf_min": CONF_MIN,
        "fps": FPS,
        "taus_s": list(TAUS_S),
        "single_classes": dict(SINGLE_CLASSES),
        "multi_classes": dict(MULTI_CLASSES),
    }


def _images(cfg: DictConfig, dataset) -> dict:
    mode = dataset.frame_cache_mode
    if mode == "baked_overlay":
        overlay = _overlay_from_bake_config(cfg.contract.bake_config, dataset.frame_cache_num_variants)
    elif dataset.obb_source == "overlay":
        # online の overlay は dataset が描く。YOLO の重みは OBB の precompute 側で決まり、ここでは分からない
        overlay = {
            "camera": None,
            "yolo_ckpt": None,
            "conf": dataset.overlay_conf_threshold,
            "thickness": dataset.overlay_line_thickness,
            "class_filter": sorted(dataset.overlay_class_filter) if dataset.overlay_class_filter else None,
        }
    else:
        overlay = None
    return {
        "camera_keys": list(dataset.camera_keys),
        "img_size": int(dataset.img_size),
        "prev_frame": bool(dataset.load_prev_image),
        "source": mode,
        "obb_source": dataset.obb_source,
        "overlay": overlay,
    }


def _overlay_from_bake_config(bake_config: str | None, num_variants: int) -> dict:
    if bake_config is None:
        raise ValueError(
            "frame_cache_mode=baked_overlay には contract.bake_config (焼き込みの設定 file) が要る。"
            "overlay の YOLO の重み・閾値を ckpt の約束に書くため"
        )
    path = REPO_ROOT / bake_config
    bake = yaml.safe_load(path.read_text())
    if bake["bake"]["num_variants"] != num_variants:
        raise ValueError(
            f"data.frame_cache_num_variants={num_variants} が焼き込みの num_variants="
            f"{bake['bake']['num_variants']} ({bake_config}) と合わない"
        )
    yolo = bake["yolo"]
    return {
        "camera": bake["cameras"]["head"],
        "yolo_ckpt": f"{yolo['ckpt']}@{yolo['revision']}",   # 推論の policy_config.yaml の yolo.ckpt_ref と同じ形式
        "top_k": int(yolo["top_k"]),
        "conf": float(bake["bake"]["overlay_conf"]),
        "thickness": int(bake["bake"]["overlay_thickness"]),
        "class_filter": None,   # 焼き込みは全 class を描く
        "bake_config": bake_config,
    }


def _git_provenance() -> dict:
    def git(*args: str) -> str | None:
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}
