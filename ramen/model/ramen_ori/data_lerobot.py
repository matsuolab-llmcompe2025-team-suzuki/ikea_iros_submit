"""RAMEN-Ori 実 LeRobot dataset loader (Issue #120 Phase A-2)。

`Team-RAMEN/IROS2026_RAMEN_HARA_task_{5,7}_...` chunk repos を
`MultiLeRobotDataset` で束ね、model.py の batch contract に合わせた dict を返す。

# LeRobot delta_timestamps で prev-frame + action chunk を一括取得

- observation.state.robot_q_current / hand_state: [-dt, 0.0] → shape (2, dim)
  で index 0 = prev, index 1 = current
- observation.state.ee_state: [0.0] → shape (1, dim)
- action.robot_q_desired / hand_cmd: [-dt, 0, dt, ..., (chunk_len-1)*dt]
  → shape (chunk_len+1, dim)。row 0 = 1 frame 前の指令 (tracking_err 用)、row 1.. = 正解 chunk。
  区間の外の frame は LeRobot が区間端に clamp し、`is_pad` flag が真値を持つ
- observation.images.cam_{0..3}: [-dt, 0.0] → shape (2, 3, H, W)

# 71D state derive

`state_derive.derive_state_71d` に渡す 4 引数を LeRobot dict から組み立てる:

- state_current (38,) = concat(robot_q_current[1], hand_state[1])
- action_prev (38,) = concat(robot_q_desired[0], hand_cmd[0]) (1 frame 前の指令) or None if is_pad
- ee_state (12,) = ee_state[0]
- state_prev (38,) = concat(robot_q_current[0], hand_state[0]) or None if is_pad

# action target (16-D chunk_len、Issue #129 Phase A 2026-08-31 で 19→16 に縮小)

action = concat(robot_q_desired, hand_cmd)[1:] → apply ARMS_HAND_SOURCE_INDEX_MAP
→ (chunk_len, 16)  # arms 14 + hand 2 (waist 3 除外、loss 圏外)
action_is_pad = robot_q_desired_is_pad[1:] → (chunk_len,) bool、区間末尾の埋め草 (最終 frame の繰り返し) の行

# skill_id (Issue #141 RO-1)

episode meta の `source_task_index` (BitRobot の元の task 番号、merge しても変わらない) を
`skill_mapping.task_index_to_skill_id` で変換する。frame の `task_index` は dataset ごとのローカル番号
(merge で 0 始まりに振り直される) なので使わない。model の入力・sampler・curriculum が同じ変換を通る。

# OBB (Phase A は dummy、Phase D で precomputed に切替)

`obb_source="none"` で OBB path 完全 skip (zeros + valid_mask=False、Fusion attention で
filter 済 = 学習に寄与しない)、`"precomputed_token"` で real YOLO OBB を Fusion に流す
(C-2)、`"overlay"` は Phase D 追加予定 (C-11)。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data

# Issue #129 Phase I-0-1 (2026-09-01): image aug pipeline を model/ramen_ori/image_aug.py
# に切り出し。既存 import 経路 (tests など `from model.ramen_ori.data_lerobot import
# ImageAugPipeline`) を保つため re-export する。
from model.ramen_ori.image_aug import (  # noqa: F401 (re-export for backward compat)
    IMAGENET_MEAN,
    IMAGENET_STD,
    FloatJPEG,
    ImageAugPipeline,
    _cfg_to_dict,
    _identity_affine_matrix,
    _make_head_affine_matrix,
    make_image_transform,
)
from model.ramen_ori.memory_features import MEMORY_CAMERA, MEMORY_DIM, MemoryTracker
from model.ramen_ori.skill_mapping import skill_id_name, task_index_to_skill_id
from model.ramen_ori.state_derive import (
    ACTION16_ARMS_SLICE,
    ARMS_HAND_SOURCE_INDEX_MAP,
    DEPTH_CONTACT_DIM,
    RAMEN_ORI_ACTION_DIM,
    STATE71_ARMS_SLICE,
    WAIST_SOURCE_INDEX_MAP,
    derive_state_71d,
    derive_state_73d,
)
from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    SOURCE_EEF_DIM,
    UPPER_BODY_SOURCE_INDEX_MAP,
)


# G1_WBT_Dex1_Building-Children-Table (BitRobot 公式 source) の fps。
# curated chunk はこれを継承するため hard-coded。
G1_WBT_FPS = 30


def default_camera_keys() -> list[str]:
    """RAMEN-Ori Phase A default = RGB 3 cam (head_left + L/R wrist、head_right / IR skip)。

    Issue #129 Phase A (2026-08-31): 4 cam → 3 cam に縮小。head_right (cam_1) を
    除外、GR00T Run 1 と apples-to-apples 比較性を確保、compute -33%。
    head stereo baseline 63mm は狭く head_left/right の parallax 小 (depth 精度冗長)。
    """
    return [
        "observation.images.cam_0",   # head_left
        "observation.images.cam_2",   # left_wrist
        "observation.images.cam_3",   # right_wrist
    ]


def build_delta_timestamps(
    chunk_len: int,
    camera_keys: list[str],
    fps: int = G1_WBT_FPS,
    load_prev_image: bool = True,
) -> dict[str, list[float]]:
    """LeRobot delta_timestamps を構築 (prev-frame + 1 frame 前の指令 + action chunk 一括 spec)。

    load_prev_image=False ならカメラは今の frame だけ (画像差分を使わない model 用、Issue #141 RO-11)。
    """
    dt = 1.0 / fps
    # 数値誤差回避のため round (tolerance_s は 1/fps - 1e-4)。
    # action は -dt (1 frame 前の指令、tracking_err 用) から chunk_len 行分。
    action_ts = [round(i * dt, 6) for i in range(-1, chunk_len)]
    delta_ts: dict[str, list[float]] = {
        "observation.state.robot_q_current": [-dt, 0.0],
        "observation.state.hand_state": [-dt, 0.0],
        "observation.state.ee_state": [0.0],
        "action.robot_q_desired": action_ts,
        "action.hand_cmd": list(action_ts),
    }
    for key in camera_keys:
        delta_ts[key] = [-dt, 0.0] if load_prev_image else [0.0]
    return delta_ts


def split_episode_indices(
    num_episodes: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    source_episode_names: list[str] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Episode 単位で train/val/test を split、EPISODE index (0..N-1) を返す (Issue #122)。

    `split_train_val_test` の episode 単位 subset。GR00T との apples-to-apples 比較のため
    export_split_json.py が使用、materialize (GR00T 側) と同じ episode 割当を実現する。

    Args:
        num_episodes: 総 episode 数
        val_ratio: val 比率 (default 0.1)
        test_ratio: test 比率 (default 0.1)
        seed: permutation seed (default 42)
        source_episode_names: 各 episode の source recording ID (len==num_episodes)。
            指定時は source-grouped split (同一 source は同一 split に集約、leakage-safe)。
            None → per-episode shuffle (後方互換)。
            task 5+7 combined などは curated segment が同一 raw 収録を共有するため、
            grouping key として ``sample_episode_metadata()["source_episode_names"]`` を渡す。

    Returns:
        (train_eps, val_eps, test_eps): 各 List[int] of episode indices (昇順、disjoint、全部で N)
    """
    if num_episodes < 3:
        raise ValueError(f"need at least 3 episodes, got {num_episodes}")
    if source_episode_names is not None:
        if len(source_episode_names) != num_episodes:
            raise ValueError(
                f"source_episode_names length {len(source_episode_names)} != num_episodes {num_episodes}"
            )
        return _split_source_grouped(source_episode_names, val_ratio, test_ratio, seed)
    rng = np.random.default_rng(seed)
    ep_order = rng.permutation(num_episodes)
    n_val = max(1, int(round(num_episodes * val_ratio)))
    n_test = max(1, int(round(num_episodes * test_ratio)))
    n_train = num_episodes - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"train split empty (num_eps={num_episodes}, val={n_val}, test={n_test})"
        )
    train_eps = sorted(int(x) for x in ep_order[:n_train])
    val_eps = sorted(int(x) for x in ep_order[n_train : n_train + n_val])
    test_eps = sorted(int(x) for x in ep_order[n_train + n_val :])
    return train_eps, val_eps, test_eps


def _split_source_grouped(
    source_episode_names: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Source-grouped episode split (Issue #122)。同一 source_name は同一 split に集約。

    Materialize (GR00T 側) の ``build_grouped_episode_split`` と同 semantics:
    group を shuffle → n_val/n_test/n_train を group 数から計算 → 各 group の
    全 episode index を担当 split に配布 (index は昇順で返る)。
    """
    groups: dict[str, list[int]] = {}
    for ep_i, name in enumerate(source_episode_names):
        if not isinstance(name, str) or not name:
            raise ValueError(f"source_episode_names[{ep_i}] must be non-empty str, got {name!r}")
        groups.setdefault(name, []).append(ep_i)
    group_keys = sorted(groups)
    n_groups = len(group_keys)
    if n_groups < 3:
        raise ValueError(
            f"need at least 3 distinct source_episode_names for grouped split, got {n_groups}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)
    shuffled = [group_keys[i] for i in perm]
    n_val = max(1, int(round(n_groups * val_ratio)))
    n_test = max(1, int(round(n_groups * test_ratio)))
    n_train = n_groups - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"train split empty (n_groups={n_groups}, val={n_val}, test={n_test})"
        )
    train_keys = shuffled[:n_train]
    val_keys = shuffled[n_train : n_train + n_val]
    test_keys = shuffled[n_train + n_val :]
    train_eps = sorted(ep_i for k in train_keys for ep_i in groups[k])
    val_eps = sorted(ep_i for k in val_keys for ep_i in groups[k])
    test_eps = sorted(ep_i for k in test_keys for ep_i in groups[k])
    return train_eps, val_eps, test_eps


def split_train_val_test(
    episode_lengths: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    source_episode_names: list[str] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Episode 単位で train/val/test を split し、frame index 配列を返す (Issue #122)。

    Frame 単位 split だと同一 episode が train/val に漏れる leakage → episode 単位で分離。
    ConcatLeRobot 内 frame の global index に対応した List[int] を返し、torch.utils.data.Subset
    で wrap 可能。GR00T 側 export_split_json.py と同じ episode 割当を保証 (episode-level split
    を `split_episode_indices` に集約)。

    Args:
        episode_lengths: `sample_episode_metadata()["episode_lengths"]` の返り値
        val_ratio: val に割く episode 比率 (default 0.1 = 10%)
        test_ratio: test に割く episode 比率 (default 0.1 = 10%)
        seed: episode 順列の乱数 seed (default 42)
        source_episode_names: 各 episode の source recording ID (len==num_eps)。
            指定時は source-grouped split (同一 source は同一 split に集約、leakage-safe)。
            None → per-episode shuffle (後方互換)。

    Returns:
        (train_indices, val_indices, test_indices): 各 List[int]、
        Global frame index (ConcatLeRobot indexing)。
    """
    num_eps = len(episode_lengths)
    train_eps_list, val_eps_list, test_eps_list = split_episode_indices(
        num_eps,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        source_episode_names=source_episode_names,
    )
    train_eps = set(train_eps_list)
    val_eps = set(val_eps_list)
    test_eps = set(test_eps_list)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    cursor = 0
    for ep_i, length in enumerate(episode_lengths):
        length_int = int(length)
        end = cursor + length_int
        rng_range = range(cursor, end)
        if ep_i in train_eps:
            train_idx.extend(rng_range)
        elif ep_i in val_eps:
            val_idx.extend(rng_range)
        else:
            test_idx.extend(rng_range)
        cursor = end
    return train_idx, val_idx, test_idx


class RamenOriSplitView(torch.utils.data.Dataset):
    """Episode-split subset を is_train flag 付きで expose (Issue #122)。

    `RamenOriLerobotDataset._transform_item(is_train=...)` に flag を伝達し、
    train は aug 適用、val/test は plain Normalize のみ。
    """

    def __init__(
        self,
        base: "RamenOriLerobotDataset",
        indices: list[int],
        is_train: bool,
    ) -> None:
        self.base = base
        self.indices = list(indices)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        global_idx = self.indices[i]
        item = self.base._base[global_idx]
        return self.base._transform_item(item, global_idx, is_train=self.is_train)


class RamenOriLerobotDataset(torch.utils.data.Dataset):
    """LeRobot v3 merged dataset を wrap し、model.py の batch contract に整形する。

    Issue #122 refactor: 元々 9 個の curated chunk repo を `_ConcatLeRobot` で
    束ねていたが、GR00T と同じ merged_source (1 dataset) を直接 load するように
    変更。効果:
    - GR00T ↔ RAMEN-Ori で **物理データ完全一致** (両者同じ crf=18 mp4)、
      apples-to-apples 比較が strict
    - JPG frame cache を GR00T と共有 (precompute 1 回で両側に効く、~50GB 節約)
    - `MultiLeRobotDataset` の aggregate_stats 非互換 workaround が不要になり
      code 単純化

    Args:
        merged_source_root: LeRobot v3 merged dataset root (meta/, data/, videos/)。
            通常は `outputs/merged_sources/<subtask>/` を指す。base_dataset 注入時は None 可。
        repo_id: LeRobotDataset 内部 logging 用の任意 ID (実 HF アクセスなし)、default
            "local/ramen_ori_merged"
        chunk_len: 予測する action horizon
        img_size: model 入力画像 size (224 default)
        camera_keys: 使用する RGB camera key (None → default 4 cam)
        top_K: OBB dummy shape (Phase A は dummy)
        num_classes: OBB dummy class 数
        num_cams: model side cam vocabulary global size
        obb_source: "none" (OBB path 完全 skip、default) / "precomputed_token" (C-2 real) / "overlay" (C-11、Phase D 予定)
        base_dataset: dependency injection (test 用)、None → merged_source から load。
            `meta.episodes` (episode_index / length / source_task_index) を持つこと
        auto_precompute_frame_cache: True (default) → constructor で precompute 実行
            (hash skip 対応、既存 cache なら即 return)。False → skip、後段 patch は fallback。

    skill_id は episode meta の `source_task_index` から作る (module docstring 参照)。
    meta に `source_task_index` が無い / skill_mapping に無い番号の episode があれば、構築時にエラー。
    """

    def __init__(
        self,
        merged_source_root: str | Path | None = None,
        repo_id: str = "local/ramen_ori_merged",
        chunk_len: int = 16,
        img_size: int = 224,
        camera_keys: list[str] | None = None,
        top_K: int = 4,
        num_classes: int = 7,
        # Issue #129 Phase A (2026-08-31): default を 3 cam + 1 buffer に変更、
        # 旧値は num_cams=6 / depth_target_num_cams=4。config yaml から明示 override 想定。
        num_cams: int = 4,
        obb_source: str = "none",
        obb_precomputed_root: str | Path | None = None,  # D-2/D-3: precomputed_token / overlay 時 required
        obb_precomputed_hash: str | None = None,         # D-2/D-3: None なら root 直下の唯一 hash dir を auto-pick
        overlay_conf_threshold: float = 0.30,            # D-3: overlay mode 時の conf threshold
        overlay_line_thickness: int = 2,                 # D-3: overlay mode 時の cv2.polylines 太さ
        overlay_class_filter: list[int] | None = None,   # D-3: None なら全 class 描画、list なら該当 class_id のみ
        state_variant: str = "71d",
        # Issue #141 RO-11: 前 frame の画像 (画像差分用) を読むか。False なら今の frame だけ読み、
        # batch に images_prev を入れない (model.temporal: null と組にする)
        load_prev_image: bool = True,
        include_depth_target: bool = False,       # Alt-3 I-6 aux 用、SGBM 未実装は zeros
        depth_target_size: tuple[int, int] = (32, 32),
        depth_target_num_cams: int = 3,   # Issue #129 Phase A: 4→3 (head_right 除外に整合)
        # Issue #129 Phase F (2026-08-31): relative action space (Run 3/5 用)。
        # arms 14 dim を Δq に、hand 2 dim は absolute pass-through。arms Δq は
        # normalize (Flow Matching SNR 対策)。None (default) なら従来 absolute。
        use_relative_action: bool = False,
        relative_stats_path: str | Path | None = None,
        base_dataset: torch.utils.data.Dataset | None = None,
        augmentation_cfg: dict | None = None,   # Issue #122: None → Normalize のみ (backward compat)
        auto_precompute_frame_cache: bool = True,  # Issue #122: auto precompute (skip via env FRAME_CACHE_PRECOMPUTE=false)
        # Issue #129 Phase I-0-3 (2026-09-01): offline aug bake cache 選択機構
        # - "online" (default): 従来挙動、__getitem__ で aug pipeline を毎回適用 (backward compat)
        # - "baked_token": token/ (photometric aug 済、overlay 無し) baked cache を load、
        #                  aug skip、raw OBB (identity affine 前提) を coord grounding に流す
        # - "baked_overlay": overlay/ (photometric + overlay + geometric aug 済) baked cache
        #                    を load、aug skip、OBB は使わない (画像に overlay bake 済)
        frame_cache_mode: str = "online",
        # baked_* mode で使う cache root。`<root>/<cam>/<chunk>/<file>/frame_XXX_v{i}.jpg`
        # 構造を持つ dir を指す。lerobot_frame_cache_patch の FRAME_CACHE_ROOT_OVERRIDE env
        # に set され、cache_dir 導出を上書き。
        frame_cache_root_baked: str | Path | None = None,
        # v0 (raw) を除く baked variant 数。precompute で `--num-variants N` 指定した値と
        # 同じにする。>=2 で FRAME_CACHE_NUM_VARIANTS env に set され、patch が v_1..v_(N-1)
        # から uniform random で pick する。online mode では 1 (v0 only、default) のまま。
        frame_cache_num_variants: int = 1,
        # Issue #141 Phase 7: memory の入力 (memory_features.py)。true なら起動時に区間ごとの表を作り、item に
        # "memory" (51,) を入れる。head_left の OBB (obb_precomputed_root) と教師の指令から作る (obb_source とは別)
        memory: bool = False,
    ) -> None:
        _valid_obb_sources = {"none", "precomputed_token", "overlay"}
        if obb_source not in _valid_obb_sources:
            raise ValueError(
                f"obb_source={obb_source!r} not supported; choices={sorted(_valid_obb_sources)}"
            )
        if obb_source in {"precomputed_token", "overlay"} and obb_precomputed_root is None:
            raise ValueError(
                f"obb_source={obb_source!r} requires obb_precomputed_root "
                "(precompute_yolo_obb.py の output dir、例: outputs/yolo_obb_cache)"
            )
        if memory and obb_precomputed_root is None:
            raise ValueError("memory=True requires obb_precomputed_root (head_left の OBB から memory を作る)")
        # Issue #129 Phase I-0-3: frame_cache_mode validation + env set
        _valid_frame_cache_modes = {"online", "baked_token", "baked_overlay"}
        if frame_cache_mode not in _valid_frame_cache_modes:
            raise ValueError(
                f"frame_cache_mode={frame_cache_mode!r} not supported; "
                f"choices={sorted(_valid_frame_cache_modes)}"
            )
        self.frame_cache_mode = frame_cache_mode
        self.frame_cache_num_variants = int(frame_cache_num_variants)
        if frame_cache_mode in {"baked_token", "baked_overlay"}:
            if frame_cache_root_baked is None:
                raise ValueError(
                    f"frame_cache_mode={frame_cache_mode!r} requires frame_cache_root_baked "
                    "(precompute_augmented_frame_cache_ramen_ori.py の "
                    f"--output-{'token' if frame_cache_mode == 'baked_token' else 'overlay'}-root)"
                )
            if self.frame_cache_num_variants < 2:
                raise ValueError(
                    f"frame_cache_mode={frame_cache_mode!r} requires "
                    f"frame_cache_num_variants>=2 (got {self.frame_cache_num_variants}); "
                    "precompute で生成した variant 数と一致させる (10 default)"
                )
            baked_root = Path(frame_cache_root_baked)
            self.frame_cache_root_baked = baked_root
            # Phase K (2026-09-01): env global 上書きから registry へ移行 (multi-dataset safe)。
            # merged_source_root が判るのは 下段 else block での LeRobotDataset 構築後だが、
            # register は source_root prefix keyed なので構築前に呼ぶ必要がある。ここでは
            # merged_source_root が None (DI test 経路) の時は registry skip、そうでない時は
            # register する。env 側にも set しておく事で、単一 dataset 経路の backward compat
            # を保つ (patch 側の lookup は registry → env の 2 段 fallback)。
            os.environ["FRAME_CACHE_ROOT_OVERRIDE"] = str(baked_root)
            os.environ["FRAME_CACHE_NUM_VARIANTS"] = str(self.frame_cache_num_variants)
            # Issue #141: baked の画像は焼き込み済みの jpg だけから読む。jpg が欠けていたら mp4 を decode せず止める
            # (strict でないと、overlay の無い mp4 の画像を黙って学習する)。起動時の env の付け忘れを防ぐため dataset が立てる
            os.environ["LEROBOT_FRAME_CACHE_ENABLE"] = "true"
            os.environ["FRAME_CACHE_STRICT"] = "true"
            if merged_source_root is not None:
                from model.subtask_policy_training.scripts.lerobot_frame_cache_patch import (
                    register_frame_cache_override as _register_override,
                )
                _register_override(
                    source_root=merged_source_root,
                    override_root=baked_root,
                    num_variants=self.frame_cache_num_variants,
                )
        else:
            self.frame_cache_root_baked = None
        if state_variant not in {"71d", "73d"}:
            raise ValueError(
                f"state_variant must be '71d' (E-δ) or '73d' (E-β)、got {state_variant!r}"
            )
        if camera_keys is None:
            camera_keys = default_camera_keys()
        self.camera_keys = camera_keys
        self.chunk_len = chunk_len
        self.img_size = img_size
        self.top_K = top_K
        self.num_classes = num_classes
        self.num_cams = num_cams
        self.obb_source = obb_source
        self.obb_precomputed_root = (
            Path(obb_precomputed_root) if obb_precomputed_root is not None else None
        )
        self.obb_precomputed_hash = obb_precomputed_hash
        self.overlay_conf_threshold = float(overlay_conf_threshold)
        self.overlay_line_thickness = int(overlay_line_thickness)
        self.overlay_class_filter = (
            set(overlay_class_filter) if overlay_class_filter is not None else None
        )
        # D-2/D-3: precomputed_token / overlay 用 cache handle。'none' mode では常に None。
        # base_dataset 経路 (DI test) では ep mapping build / hook register を skip
        # (test 側で inject or 直接 hook 呼ぶ)。
        self._obb_cache = None  # type: ignore[assignment]
        self._overlay_renderer = None  # type: ignore[assignment]
        self.state_variant = state_variant
        self.include_depth_target = include_depth_target
        self.depth_target_size = tuple(depth_target_size)
        self.depth_target_num_cams = depth_target_num_cams
        # Issue #129 Phase F (2026-08-31): relative action space
        self.use_relative_action = bool(use_relative_action)
        self._relative_stats: dict[str, np.ndarray] | None = None
        if self.use_relative_action:
            if relative_stats_path is None:
                raise ValueError(
                    "use_relative_action=True requires relative_stats_path "
                    "(precompute output from scripts/compute_relative_action_stats.py)"
                )
            from model.ramen_ori.relative_action import load_relative_stats

            stats = load_relative_stats(relative_stats_path)
            # numpy 化 (data_lerobot は numpy 前提)、float32
            self._relative_stats = {
                "mean": stats["mean"].numpy().astype(np.float32),
                "std": stats["std"].numpy().astype(np.float32),
            }
        if include_depth_target:
            import logging
            logging.getLogger(__name__).warning(
                "include_depth_target=True but SGBM+WLS pipeline not implemented — "
                "depth_target = zeros placeholder (aux loss will be near-constant)."
            )
        if state_variant == "73d":
            # Alt-2 Phase: SGBM+WLS pipeline 未実装。depth_contact は placeholder zero。
            # 実 signal 対応時は _fetch_depth_contact() を実装差替。
            import logging
            logging.getLogger(__name__).warning(
                "state_variant='73d' (E-β) but depth pipeline not implemented — "
                "using zero placeholder for depth_contact (effectively equivalent to E-δ)."
            )

        self.N_cams = len(camera_keys)
        # cam id = 0..N_cams-1 (camera_keys 順)
        self._cam_id = torch.arange(self.N_cams, dtype=torch.long)

        # Issue #122: 光系 aug + Normalize pipeline (head 強 / wrist 弱、prev/current 同 seed)
        self.aug_pipeline = ImageAugPipeline(augmentation_cfg)

        self.load_prev_image = bool(load_prev_image)
        self.delta_timestamps = build_delta_timestamps(
            chunk_len=chunk_len, camera_keys=camera_keys, fps=G1_WBT_FPS,
            load_prev_image=self.load_prev_image,
        )

        if base_dataset is not None:
            # DI 経路 (test 用)。base_dataset は実 LeRobotDataset と同じく meta.episodes を持つ
            self._base = base_dataset
        else:
            if not merged_source_root:
                raise ValueError(
                    "merged_source_root required when base_dataset is None. "
                    "Issue #122 refactor で repo_ids list は廃止、GR00T と共有の merged "
                    "dataset (outputs/merged_sources/<subtask>/) を指定する事。"
                )
            merged_source_root = Path(merged_source_root)
            # lazy import: LeRobot は import で ffmpeg 依存を舐めるため
            # test で mock injection しやすいよう constructor 内で late import
            _patch_lerobot_load_tasks()  # fork 0.5.1 vs curation_tool 0.6.0 tasks.parquet 形式差分吸収
            # Issue #122: JPG frame cache patch (env LEROBOT_FRAME_CACHE_ENABLE=true で active、
            # 未設定なら install 済でも full fallback = 挙動不変)
            from model.subtask_policy_training.scripts.lerobot_frame_cache_patch import (
                apply_patch as _apply_frame_cache_patch,
            )
            _apply_frame_cache_patch()

            # Issue #122: Auto precompute frame cache (GR00T が既に precompute 済みなら
            # hash skip で即 return、未 precompute なら 5-10 min の待ち)。
            # 環境変数 FRAME_CACHE_PRECOMPUTE=false で skip 可能 (既 cache 前提)。
            # baked は焼き込み済みの cache だけを読むので展開しない (Issue #141)
            if (
                auto_precompute_frame_cache
                and self.frame_cache_root_baked is None
                and os.environ.get("FRAME_CACHE_PRECOMPUTE", "true").lower() != "false"
            ):
                _auto_precompute_frame_cache(merged_source_root)

            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            image_tf = make_image_transform(img_size)
            # Issue #122: merged_source は既に stats.json + tasks.parquet が正規化済
            # (task_index remap + canonical "task" column)、MultiLeRobotDataset の
            # aggregate_stats 非互換 workaround も不要。1 個の LeRobotDataset で完結。
            self._base = LeRobotDataset(
                repo_id,
                root=merged_source_root,
                delta_timestamps=self.delta_timestamps,
                image_transforms=image_tf,
                video_backend="pyav",  # torchcodec は system libavutil 解決困難 → pyav (av==15.1.0)
            )
            # LeRobot が item ごとに読みに行く動画を camera_keys だけにする (IR / head_right を読まない)
            _restrict_video_keys(self._base.meta, self.camera_keys)

            # D-2/D-3: precomputed_token / overlay 用 cache init。ep mapping は
            # meta.episodes 経由で build (LeRobot 実 load 後にしか読めない)。overlay は
            # 加えて JPG cache patch に post-decode hook を register (raw uint8 段階で
            # cv2.polylines 描画 → 後段 Resize で anti-alias)。
            if self.obb_source in {"precomputed_token", "overlay"}:
                from model.ramen_ori.obb_cache import ObbPrecomputedCache

                cache = ObbPrecomputedCache(
                    root=self.obb_precomputed_root,
                    ckpt_hash=self.obb_precomputed_hash,
                    camera_keys=self.camera_keys,
                    top_K=self.top_K,
                    num_classes=self.num_classes,
                    fps=G1_WBT_FPS,
                )
                cache.build_episode_mapping(self._base.meta.episodes)
                self._obb_cache = cache

            if self.obb_source == "overlay":
                # Issue #129 Phase D (2026-08-31): hook 経路を unwire、inline 経路に移行。
                # 旧 register_post_decode_hook 呼出しは削除、renderer は config store
                # (conf_threshold / line_thickness / class_filter / palette) として保持のみ。
                # 実描画は _transform_item 内で `draw_overlay_on_tensor` を Photometric+Erasing 後に呼ぶ。
                # GR00T training 側 (`obb_overlay_setup.py`) は hook 経路を継続使用。
                from model.ramen_ori.overlay import OverlayRenderer

                renderer = OverlayRenderer(
                    cache=self._obb_cache,
                    conf_threshold=self.overlay_conf_threshold,
                    line_thickness=self.overlay_line_thickness,
                    class_filter=self.overlay_class_filter,
                )
                self._overlay_renderer = renderer

        # sample_episode_metadata が iterate する形を保つため、single 要素 list
        self._sub_datasets = [self._base]

        # skill_id の表 (episode_index → skill_id)。_transform_item が frame の episode_index で引く。
        # 表に無い番号・source_task_index の欠損はここで止める (学習を始めてからでは気づけない)
        self._skill_id_by_episode: dict[int, int] = {}
        frames_by_skill: dict[int, int] = {}
        for row in _iter_episode_rows(self._base.meta.episodes):
            skill_id = _episode_skill_id(row)
            self._skill_id_by_episode[int(row["episode_index"])] = skill_id
            frames_by_skill[skill_id] = frames_by_skill.get(skill_id, 0) + _episode_length(row)
        source = merged_source_root if merged_source_root is not None else "base_dataset (injected)"
        print(
            f"[data] {source}: "
            + ", ".join(
                f"skill {sid} ({skill_id_name(sid)}) {n} frames"
                for sid, n in sorted(frames_by_skill.items())
            )
        )

        # Issue #141 Phase 7: memory の表 (dataset の idx 順、(N, 51))。使った OBB の YOLO の重みは約束に書く
        self.memory = bool(memory)
        self.memory_yolo_ckpt: str | None = None
        self._memory_table: np.ndarray | None = None
        if self.memory:
            t0 = time.time()
            self._memory_table, self.memory_yolo_ckpt = self._build_memory_table()
            print(f"[data] {source}: memory の表 {self._memory_table.shape} を {time.time() - t0:.0f}s で作成")

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict:
        item = self._base[idx]
        # is_train=True default (backward compat)。RamenOriSplitView 経由の場合は False も来る。
        return self._transform_item(item, idx, is_train=True)

    def make_train_val_test_split(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        split_json: str | Path | None = None,
    ) -> tuple[RamenOriSplitView, RamenOriSplitView, RamenOriSplitView]:
        """Episode 単位 split (Issue #122)、is_train flag 付き 3 view を返す。

        - train view: `is_train=True` (aug 適用)
        - val/test view: `is_train=False` (Normalize のみ)

        Args:
            split_json: Path to a pre-computed split JSON (export_split_json.py 出力形式)。
                指定時は JSON から episode indices を読み込み、cross-server で strict
                identical split を保証 (GR00T の SPLIT_JSON と同じ物を使う)。None →
                従来通り on-the-fly 計算 (seed=42、source-grouped、決定的だが「同 code +
                同 HF revision」前提)。

        Raises:
            RuntimeError: `sample_episode_metadata` が使えない (base_dataset 注入時) 場合、
                または split_json の episode 数が dataset の実 episode 数と mismatch
        """
        meta = self.sample_episode_metadata()

        if split_json is not None:
            train_eps, val_eps, test_eps = _load_split_from_json(
                Path(split_json), num_episodes=meta["num_episodes"]
            )
            # episode-level → frame-level index に展開
            train_idx, val_idx, test_idx = _expand_episode_split_to_frames(
                train_eps, val_eps, test_eps, meta["episode_lengths"]
            )
        else:
            train_idx, val_idx, test_idx = split_train_val_test(
                meta["episode_lengths"],
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                source_episode_names=meta.get("source_episode_names"),
            )
        return (
            RamenOriSplitView(self, train_idx, is_train=True),
            RamenOriSplitView(self, val_idx, is_train=False),
            RamenOriSplitView(self, test_idx, is_train=False),
        )

    def state_action_item(self, idx: int) -> dict:
        """動画を読まずに state / action / skill_id などを返す (正規化の統計用)。

        LeRobot の表の読み出し (`_query_hf_dataset` は video key を飛ばす) だけを使い、
        `_transform_item` と同じ `_state_action` を通す。
        """
        base = self._base
        base._ensure_hf_dataset_loaded()
        item = base.hf_dataset[idx]
        query_indices, padding = base._get_query_indices(
            item["index"].item(), item["episode_index"].item()
        )
        item = {**item, **padding, **base._query_hf_dataset(query_indices)}
        return self._state_action(item, idx)

    def _build_memory_table(self) -> tuple[np.ndarray, str]:
        """区間 (episode) ごとに MemoryTracker を frame 順に回した表 ((N, 51)、dataset の idx 順) と YOLO の重み。

        frame t の memory は、head_left の frame 0..t の検出と、frame t−1 までの教師の指令 (19D を FK した手先) から作る
        (推論の tick t と同じ: 今の画像の検出と、前の tick に送った指令)。
        """
        from model.ramen_ori.fk import G1WristFKTorch
        from model.ramen_ori.obb_cache import ObbPrecomputedCache

        cache = ObbPrecomputedCache(
            root=self.obb_precomputed_root,
            ckpt_hash=self.obb_precomputed_hash,
            camera_keys=[MEMORY_CAMERA],
            top_K=None,   # cache にある検出を全部読む (統合 cache は 32 個)
            num_classes=self.num_classes,
            fps=G1_WBT_FPS,
        )
        cache.build_episode_mapping(self._base.meta.episodes)

        # 教師の指令 (各 frame の q_desired + hand_cmd → 19D) を FK して左右の手先 (N, 6)
        self._base._ensure_hf_dataset_loaded()
        columns = self._base.hf_dataset.with_format("numpy")
        command38 = np.concatenate(
            [np.asarray(columns["action.robot_q_desired"]), np.asarray(columns["action.hand_cmd"])], axis=1
        )
        command19 = torch.from_numpy(command38[:, np.asarray(UPPER_BODY_SOURCE_INDEX_MAP)].astype(np.float32))
        fk = G1WristFKTorch.from_default_urdf()
        with torch.no_grad():
            hands = [fk.forward_detailed(c) for c in command19.split(65536)]
        hand_pos = torch.cat(
            [torch.cat([h["left_hand"], h["right_hand"]], dim=-1) for h in hands]
        ).numpy()

        rows = list(_iter_episode_rows(self._base.meta.episodes))
        episode_ids = [int(r["episode_index"]) for r in rows]
        lengths = [_episode_length(r) for r in rows]
        # 表の行 = hf_dataset の行。episode が meta の順に連続して並んでいることを確かめる
        if not np.array_equal(
            np.asarray(columns["episode_index"]).reshape(-1), np.repeat(episode_ids, lengths)
        ):
            raise ValueError("hf_dataset の行が meta.episodes の順に並んでいない (memory の表の行が frame とずれる)")

        table = np.zeros((len(hand_pos), MEMORY_DIM), dtype=np.float32)
        tracker = MemoryTracker()
        start = 0
        for ep, length in zip(episode_ids, lengths):
            det = cache.episode_arrays(ep, length, MEMORY_CAMERA)
            verts = det["verts"].reshape(length, -1, 4, 2)
            tracker.reset()
            for t in range(length):
                valid = det["valid"][t]
                table[start + t] = tracker.update(
                    det["class_id"][t][valid],
                    det["conf"][t][valid],
                    verts[t][valid],
                    hand_pos[start + t - 1] if t > 0 else None,
                )
            start += length
        yolo_ckpt = str(cache.manifest["yolo_ckpt_ref"]).removeprefix("hf:")   # 推論の policy_config.yaml と同じ形式
        return table, yolo_ckpt

    def memory_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """memory の表と frame ごとの skill_id (memory の切り詰めと正規化の統計用)。"""
        return self._memory_table, self.sample_skill_ids()

    def _state_action(self, item: dict, idx: int) -> dict:
        """LeRobot 生 dict → state / action / action_is_pad / action_waist_teacher / skill_id (numpy)。

        `item` は LeRobotDataset が返す形 (画像以外):
            - observation.state.robot_q_current: (2, 36)
            - observation.state.hand_state: (2, 2)
            - observation.state.ee_state: (1, 12)
            - action.robot_q_desired: (chunk_len+1, 36)  row 0 = 1 frame 前の指令
            - action.hand_cmd: (chunk_len+1, 2)
            - {above key}_is_pad: bool tensor
            - episode_index: scalar long (skill_id の表を引く)
        """
        # --- state derive (71D) ---
        q_current = _to_numpy(item["observation.state.robot_q_current"])   # (2, 36)
        hand_state = _to_numpy(item["observation.state.hand_state"])       # (2, 2)
        ee_state = _to_numpy(item["observation.state.ee_state"])           # (1, 12)
        q_desired_all = _to_numpy(item["action.robot_q_desired"])          # (chunk_len+1, 36)
        hand_cmd_all = _to_numpy(item["action.hand_cmd"])                  # (chunk_len+1, 2)
        action_pad_all = _to_numpy(item["action.robot_q_desired_is_pad"]).astype(bool)  # (chunk_len+1,)

        # is_pad flags: True → ep 境界で clamp された frame
        q_current_pad = _to_numpy(item.get("observation.state.robot_q_current_is_pad"))
        # prev index = 0、prev が pad なら ep 先頭 → state_prev=None (velocity=0)
        prev_is_pad = bool(q_current_pad[0]) if q_current_pad is not None else False

        state_current_38 = np.concatenate([q_current[1], hand_state[1]]).astype(np.float32)
        state_prev_38 = (
            None
            if prev_is_pad
            else np.concatenate([q_current[0], hand_state[0]]).astype(np.float32)
        )
        # 1 frame 前の指令 (tracking_err 用)。区間先頭では区間の外なので None (tracking_err=0)
        action_prev_38 = (
            None
            if action_pad_all[0]
            else np.concatenate([q_desired_all[0], hand_cmd_all[0]]).astype(np.float32)
        )
        # row 1.. が正解 chunk
        q_desired = q_desired_all[1:]
        hand_cmd = hand_cmd_all[1:]
        action_is_pad = action_pad_all[1:]
        ee_state_12 = ee_state[0].astype(np.float32)

        if self.state_variant == "73d":
            depth_contact = self._fetch_depth_contact(item, idx)
            state_out = derive_state_73d(
                state_current=state_current_38,
                action_prev=action_prev_38,
                ee_state=ee_state_12,
                state_prev=state_prev_38,
                depth_contact=depth_contact,
            )
        else:
            state_out = derive_state_71d(
                state_current=state_current_38,
                action_prev=action_prev_38,
                ee_state=ee_state_12,
                state_prev=state_prev_38,
            )

        # --- action target (chunk_len, 16) ---
        # Issue #129 Phase A (2026-08-31): waist 3 dim を除外、arms 14 + hand 2 = 16D。
        # inference で waist を使わないため loss 圏外に外し、arms/hand への gradient 集中を狙う。
        # concat(robot_q_desired, hand_cmd) → (chunk_len, 38) → ARMS_HAND index
        action_full = np.concatenate([q_desired, hand_cmd], axis=1).astype(np.float32)
        arms_hand_idx = np.asarray(ARMS_HAND_SOURCE_INDEX_MAP)
        action_16 = action_full[:, arms_hand_idx]   # absolute arms 14 + hand 2
        assert action_16.shape == (self.chunk_len, RAMEN_ORI_ACTION_DIM), (
            f"action_16 shape mismatch: {action_16.shape}, expected ({self.chunk_len}, {RAMEN_ORI_ACTION_DIM})"
        )
        # Issue #129 Phase F (2026-08-31): relative action space (Run 3/5)。
        # arms 14 dim を Δq に、hand 2 dim は absolute pass-through。
        # Δq[0] = teacher[0] - current、Δq[k>0] = teacher[k] - teacher[k-1]。
        # arms Δq を per-dim mean/std で unit-variance normalize (Flow Matching SNR 対策)。
        if self.use_relative_action:
            arms_teacher_14 = action_16[:, 0:14]                 # (chunk_len, 14) absolute
            hand_abs_2 = action_16[:, 14:16]                     # (chunk_len, 2) absolute pass-through
            # arms_current は state joint slice (state_out[3:17])。E-δ 71D layout の joint 部分
            # は UPPER_BODY 順序 (waist 3 + arms 14 + hand 2)、arms は index [3:17]。
            arms_current_14 = state_out[STATE71_ARMS_SLICE].astype(np.float32)   # (14,)
            # dq[0] = teacher[0] - current
            dq_0 = (arms_teacher_14[0] - arms_current_14).reshape(1, 14)
            dq_rest = arms_teacher_14[1:] - arms_teacher_14[:-1]  # (chunk_len-1, 14)
            dq_arms = np.concatenate([dq_0, dq_rest], axis=0).astype(np.float32)  # (chunk_len, 14)
            # normalize: (dq - mean) / max(std, 1e-6)
            mean = self._relative_stats["mean"]                  # (14,)
            std = np.maximum(self._relative_stats["std"], 1e-6)  # (14,) clip
            dq_arms_norm = ((dq_arms - mean) / std).astype(np.float32)
            action_16 = np.concatenate([dq_arms_norm, hand_abs_2], axis=1).astype(np.float32)
            assert action_16.shape == (self.chunk_len, RAMEN_ORI_ACTION_DIM)
        # --- Issue #129 Phase B (2026-08-31): teacher waist target (chunk_len, 3)
        # for L4 FK anchor loss ---
        # L4 は 19D FK に (teacher_waist, pred_arms_hand) を concat して食わせる。
        # waist 予測はしないが、teacher の future waist target は FK chain に必要。
        # action.robot_q_desired[:, WAIST_SOURCE_INDEX_MAP] = 3 waist joint targets。
        waist_idx = np.asarray(WAIST_SOURCE_INDEX_MAP)
        action_waist_teacher_3 = action_full[:, waist_idx]
        assert action_waist_teacher_3.shape == (self.chunk_len, 3), (
            f"action_waist_teacher_3 shape mismatch: {action_waist_teacher_3.shape}"
        )

        out = {
            "state": state_out,
            "action": action_16,
            "action_is_pad": action_is_pad,
            "action_waist_teacher": action_waist_teacher_3,
            # episode meta の source_task_index 由来 (構築時に作った表)
            "skill_id": self._skill_id_by_episode[_scalar_from(item["episode_index"])],
        }
        if self._memory_table is not None:
            out["memory"] = self._memory_table[idx]
        return out

    def _transform_item(self, item: dict, idx: int, is_train: bool = True) -> dict:
        """LeRobot 生 dict → RAMEN-Ori batch dict。

        `item` は LeRobotDataset が返す形 (`_state_action` の key に加えて):
            - observation.images.cam_{i}: (2, 3, H, W) - image_transforms 適用済
        """
        sa = self._state_action(item, idx)

        # --- images (N_cams, 3, H, W) + prev ---
        # LeRobot は各 camera_key ごとに (2, 3, H, W) を返す (delta -dt, 0)。
        # Issue #122: image_transforms は Resize のみ、Normalize + aug を aug_pipeline で cam type
        # 別 dispatch (prev/current 同 seed で delta encoder を守る)。
        # Issue #129 Phase D (2026-08-31): overlay 経路は inline に移行、hook 未登録。
        # obb_source='overlay' 時は per-frame の OBB tensor を事前 fetch し、cam ごと
        # overlay_callback で aug_pipeline.apply に注入 (Photometric → Erasing → overlay → Normalize)。
        # Issue #129 Phase I-0-3 (2026-09-01): baked mode 判定。
        # baked_overlay: 画像に overlay 済 → inline overlay 描画 skip
        # baked_token / online: 従来ルート
        _is_baked = self.frame_cache_mode in {"baked_token", "baked_overlay"}

        overlay_obb_by_cam = None
        if (
            not _is_baked
            and self.obb_source == "overlay"
            and self._overlay_renderer is not None
        ):
            overlay_obb_by_cam = self._fetch_overlay_obb_by_cam(item)

        # Issue #129 Phase G (2026-08-31): aug_pipeline.apply が (imgs, affine) の tuple
        # を返す形に refactor 済み。head cam + geometric aug が実適用された場合、affine
        # matrix は normalized [0,1] 空間の非-identity (2,3)、それ以外は identity。
        # obb_source='precomputed_token' で head cam に affine が乗った場合、OBB coord も
        # 同 affine で warp して image と Fusion 入力の空間を一貫させる (Phase E helper)。
        affine_per_cam: list[torch.Tensor] = []

        images_list = []
        images_prev_list = []
        for cam_key_index, cam_key in enumerate(self.camera_keys):
            # (2, 3, H, W) = 前 + 今 (load_prev_image)。今だけのときは LeRobot が frame の次元を落として
            # (3, H, W) で返すので (1, 3, H, W) に戻す。torch tensor float32 [0,1]
            img_2 = item[cam_key]
            if not isinstance(img_2, torch.Tensor):
                img_2 = torch.as_tensor(np.asarray(img_2))
            if img_2.ndim == 3:
                img_2 = img_2.unsqueeze(0)

            overlay_callback = None
            if overlay_obb_by_cam is not None:
                cam_obb = overlay_obb_by_cam[cam_key_index]
                if cam_obb is not None:
                    overlay_callback = self._make_overlay_callback(cam_obb)

            # baked mode: aug + overlay 全部 precompute 済 → is_train=False path で
            # aug skip + Normalize のみ (affine は identity return、OBB warp path も no-op)。
            _effective_is_train = False if _is_baked else is_train
            img_2, affine_matrix = self.aug_pipeline.apply(
                img_2, cam_key, idx, _effective_is_train, overlay_callback=overlay_callback
            )
            affine_per_cam.append(affine_matrix)
            images_list.append(img_2[-1])
            if self.load_prev_image:
                images_prev_list.append(img_2[0])
        images = torch.stack(images_list, dim=0)         # (N_cams, 3, H, W)

        images_prev = None
        if self.load_prev_image:
            images_prev = torch.stack(images_prev_list, dim=0)  # (N_cams, 3, H, W)
            # images_prev の ep 境界は LeRobot が clamp (前 frame = 現 frame と同じ index)
            # → images_prev ≒ images になり、Temporal Delta ≒ 0 で無害。zeros に置き換える
            # 必要は無い (design doc D-3 の delta encoder が学習で吸収)
            # ただし safety check: pad flag が付いてるなら images_prev = images に強制
            cam_pad = _to_numpy(item.get(f"{self.camera_keys[0]}_is_pad"))
            if cam_pad is not None and bool(cam_pad[0]):
                images_prev = images.clone()

        # --- OBB ---
        if self.obb_source == "precomputed_token":
            obb = self._make_precomputed_obb(item)
            # Issue #129 Phase G (2026-08-31): head cam に geometric aug が乗った場合、
            # OBB coord も同 affine で warp して image と Fusion 入力の空間を一貫。
            # affine が identity (wrist / disabled) の cam は変化なし。
            obb = self._warp_obb_verts_under_affines(obb, affine_per_cam)
        else:
            # obb_source in {"none", "overlay"}: OBB token 供給せず (zeros + valid_mask=False)。
            # overlay mode の場合 YOLO 情報は既に image に描画済 (Phase D inline)、
            # token path は unused。overlay 単独効果を測るため precomputed_token とは
            # 排他 (同時に model に流したければ config で明示併用、v1 は排他運用)。
            obb = self._make_none_obb(idx)

        sample = {
            "images": images,
            "cam_id": self._cam_id,
            "obb_verts": obb["verts"],
            "obb_conf": obb["conf"],
            "obb_class_id": obb["class_id"],
            "obb_cam_id": obb["cam_id"],
            "obb_valid_mask": obb["valid_mask"],
            "state": torch.from_numpy(sa["state"]),
            "skill_id": torch.tensor(sa["skill_id"], dtype=torch.long),
            "action": torch.from_numpy(sa["action"]),
            # Issue #141 RO-2: 区間末尾の埋め草の行 (loss から外す)
            "action_is_pad": torch.from_numpy(sa["action_is_pad"]),
            # Issue #129 Phase B (2026-08-31): L4 FK anchor loss 用 teacher waist target
            "action_waist_teacher": torch.from_numpy(sa["action_waist_teacher"]),
        }
        if images_prev is not None:
            sample["images_prev"] = images_prev
        if "memory" in sa:
            sample["memory"] = torch.from_numpy(sa["memory"])
        if self.include_depth_target:
            # Alt-3 I-6 aux 用 placeholder。SGBM+WLS pipeline landed 後は
            # _fetch_depth_target(item, idx) 実装で切替。
            H, W = self.depth_target_size
            sample["depth_target"] = torch.zeros(
                self.depth_target_num_cams, 1, H, W, dtype=torch.float32
            )
        return sample

    def _fetch_depth_contact(self, item: dict, idx: int) -> np.ndarray:
        """wrist depth contact bit (2D) を取得。SGBM+WLS pipeline 未実装のため placeholder zero。

        将来実装案 (別 Alt):
        - Precomputed: chunk repo に depth_contact.parquet を precompute で埋めておく → LeRobot item に含まれる
        - On-the-fly: wrist camera image から SGBM+WLS で depth → threshold で contact bit
        """
        return np.zeros(DEPTH_CONTACT_DIM, dtype=np.float32)

    def sample_episode_metadata(self) -> dict:
        """全 episode の meta (length + skill_id + source_episode_name) を集約。

        Returns:
            {
                "episode_lengths": np.ndarray[num_eps] int64,
                "episode_skill_ids": np.ndarray[num_eps] int64,
                "source_episode_names": list[str][num_eps],
                "num_episodes": int,
            }

        `source_episode_names` は curated dataset の ``source_episode_index`` を
        BitRobot G1_WBT (raw 533ep) 上の stable ID として利用。同じ ``source_episode_index``
        は同じ raw 収録から抽出された segment を指すため、task 5+7 combined の場合でも
        train/val leakage-safe な source-grouped split の grouping key として使える。
        Fallback として ``source_episode_index`` が row に無い場合は
        ``f"local_{sub_idx}_{ep_i}"`` (per-episode 一意) を返す。
        """
        lengths = []
        skill_ids = []
        source_names: list[str] = []
        for sub_idx, sub in enumerate(self._sub_datasets):
            for ep_i, row in enumerate(_iter_episode_rows(sub.meta.episodes)):
                lengths.append(_episode_length(row))
                skill_ids.append(_episode_skill_id(row))
                source_names.append(_extract_source_episode_name(row, sub_idx, ep_i))
        return {
            "episode_lengths": np.asarray(lengths, dtype=np.int64),
            "episode_skill_ids": np.asarray(skill_ids, dtype=np.int64),
            "source_episode_names": source_names,
            "num_episodes": len(lengths),
        }

    def sample_skill_ids(self) -> np.ndarray:
        """全 sample の skill_id 配列を返す (balancing / curriculum の sampler 用)。

        episode meta の skill_id (_episode_skill_id、model の入力と同じ変換) を episode の
        length で frame-level に expand して concat。1 回 dataset 全体を舐めるので init 直後に
        1 度だけ呼ぶこと。

        Returns:
            np.ndarray shape=(len(self),) dtype=int64
        """
        arrs = [
            np.full(_episode_length(row), _episode_skill_id(row), dtype=np.int64)
            for sub in self._sub_datasets
            for row in _iter_episode_rows(sub.meta.episodes)
        ]
        result = np.concatenate(arrs)
        if len(result) != len(self):
            raise RuntimeError(
                f"sample_skill_ids length {len(result)} != dataset len {len(self)}"
            )
        return result

    def _warp_obb_verts_under_affines(
        self,
        obb: dict[str, torch.Tensor],
        affine_per_cam: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Issue #129 Phase G (2026-08-31): OBB verts を cam 別 affine で warp。

        Run 2/6 (obb_source=precomputed_token) + head cam に geometric aug が乗った場合、
        image warp と OBB coord 空間を一貫させる。affine が identity の cam は verts 不変。

        Args:
            obb:            {"verts":(N_cams, top_K, 8), "conf":..., "class_id":..., "cam_id":..., "valid_mask":...}
                            `_make_precomputed_obb` 出力
            affine_per_cam: length N_cams、各 (2, 3) affine (normalized [0,1] 空間)。

        Returns:
            obb と同 shape の dict、verts のみ warp、他 field は passthrough。
        """
        from model.ramen_ori.obb_warp import warp_obb_coord_under_affine  # lazy

        assert len(affine_per_cam) == obb["verts"].shape[0], (
            f"affine_per_cam length {len(affine_per_cam)} != N_cams {obb['verts'].shape[0]}"
        )
        warped_verts = obb["verts"].clone()   # (N_cams, top_K, 8)
        identity = _identity_affine_matrix()
        for cam_i, affine in enumerate(affine_per_cam):
            # identity check (fast path)、微小誤差含み比較
            if torch.allclose(affine, identity, atol=1e-6):
                continue
            warped_verts[cam_i] = warp_obb_coord_under_affine(
                obb["verts"][cam_i],   # (top_K, 8)
                affine,
                clamp=True,             # affine 後の座標を [0,1] 内に、範囲外 box は 0/1 に clip
            )
        return {**obb, "verts": warped_verts}

    def _fetch_overlay_obb_by_cam(self, item: dict) -> list[dict[str, torch.Tensor] | None]:
        """Issue #129 Phase D (2026-08-31): overlay inline 経路用の per-cam OBB fetch。

        `_make_precomputed_obb` と同じ cache lookup を経由し、cam 別に slice して返す。
        wrist cam (cache 未収録) は valid 全 False で返るため、下流の
        `draw_overlay_on_tensor` が自然に no-op となる (polylines skip)。

        Returns:
            list of length N_cams、各 entry は dict(verts=(top_K,8), conf=(top_K,),
            class_id=(top_K,), valid=(top_K,)) or None (cache miss で明示 skip)。
        """
        if self._obb_cache is None:
            raise RuntimeError(
                "obb_source='overlay' だが _obb_cache が None "
                "(base_dataset 経路 DI では test 側で inject が必要)"
            )
        episode_index = _scalar_from(item["episode_index"])
        frame_index_in_ep = _scalar_from(item["frame_index"])
        full = self._obb_cache.make_obb_tensors(
            episode_index=episode_index,
            frame_index_in_ep=frame_index_in_ep,
            cam_ids=self._cam_id,
        )
        # full: {"verts":(N,K,8), "conf":(N,K,1), "class_id":(N,K), "cam_id":(N,K), "valid":(N,K)}
        # cam 別 slice、conf は (K,) に squeeze
        per_cam: list[dict[str, torch.Tensor] | None] = []
        for i in range(self.N_cams):
            per_cam.append(
                {
                    "verts": full["verts"][i],           # (top_K, 8)
                    "conf": full["conf"][i].squeeze(-1),  # (top_K,)
                    "class_id": full["class_id"][i],      # (top_K,)
                    "valid": full["valid"][i],            # (top_K,)
                }
            )
        return per_cam

    def _make_overlay_callback(self, cam_obb: dict[str, torch.Tensor]):
        """Issue #129 Phase D (2026-08-31): aug_pipeline.apply に渡す overlay_callback を生成。

        `OverlayRenderer` の config (conf / thickness / filter / palette) を capture し、
        `draw_overlay_on_tensor` を per-frame で呼ぶ closure を返す。
        """
        from model.ramen_ori.overlay import draw_overlay_on_tensor  # lazy: cv2 依存

        renderer = self._overlay_renderer
        verts = cam_obb["verts"]
        conf = cam_obb["conf"]
        class_id = cam_obb["class_id"]
        valid = cam_obb["valid"]

        def _cb(img: torch.Tensor, frame_i: int) -> torch.Tensor:
            # prev/current で同一 OBB (同 frame_idx 前提)。将来 delta frame を
            # 独立 OBB で描画したい場合はここで frame_i 別 obb を lookup する。
            return draw_overlay_on_tensor(
                img,
                verts=verts,
                conf=conf,
                class_id=class_id,
                valid=valid,
                conf_threshold=renderer.conf_threshold,
                line_thickness=renderer.line_thickness,
                class_filter=renderer.class_filter,
                class_colors_bgr=renderer.class_colors_bgr,
            )

        return _cb

    def _make_precomputed_obb(self, item: dict) -> dict[str, torch.Tensor]:
        """D-2: precompute cache から real OBB tensor を lookup。

        item["episode_index"] と item["frame_index"] から (ep, frame_in_ep) を取り、
        ObbPrecomputedCache.make_obb_tensors に委譲。cache に無い cam (wrist) は
        zeros + valid_mask=False で埋まる (padding class_id=0 の workspace collision は
        Fusion / ActionExpert 2 段 key_padding_mask で gradient 分離、Issue #122 議論)。
        """
        if self._obb_cache is None:
            raise RuntimeError(
                "obb_source='precomputed_token' だが _obb_cache が None "
                "(base_dataset 経路 DI では test 側で inject が必要)"
            )
        episode_index = _scalar_from(item["episode_index"])
        frame_index_in_ep = _scalar_from(item["frame_index"])
        return self._obb_cache.make_obb_tensors(
            episode_index=episode_index,
            frame_index_in_ep=frame_index_in_ep,
            cam_ids=self._cam_id,
        )

    def _make_none_obb(self, idx: int) -> dict[str, torch.Tensor]:
        """OBB path 完全 skip 用の zero tensor 生成 (C-1 相当、`obb_source="none"`)。

        Design (Issue #122):
        - 全 zeros + valid_mask 全 False
        - fusion attention (key_padding_mask=~valid_mask) が OBB tokens を完全 filter
        - ObbTokenizer への gradient も 0 → params は init のまま (実質 no-op)
        - class_id=0 (workspace) を padding にも使うが、valid=False で attention 排除
          されるため class_embed[0] への spurious 学習は起きない (D-2 議論参照)

        旧 "dummy" mode の rename (D-2 実装で 3-mode 化 = none / precomputed_token /
        overlay(D-3))。旧 idx-seeded 乱数 dummy は spurious correlation の risk があった
        ため Issue #122 で all-zeros に revised 済 → その後 D-2 で "none" に rename。

        `idx` は shape の determinism 保証用の signature 互換で受けるが未使用。
        """
        del idx  # zeros なので seed 不要 (signature backward compat のみ)
        N_cams = self.N_cams
        top_K = self.top_K
        cam_ids_2d = self._cam_id.unsqueeze(1).expand(N_cams, top_K).contiguous()
        return {
            "verts": torch.zeros(N_cams, top_K, 8, dtype=torch.float32),
            "conf": torch.zeros(N_cams, top_K, 1, dtype=torch.float32),
            "class_id": torch.zeros(N_cams, top_K, dtype=torch.long),
            "cam_id": cam_ids_2d,
            "valid_mask": torch.zeros(N_cams, top_K, dtype=torch.bool),
        }


class RamenOriMultiSplitView(torch.utils.data.Dataset):
    """複数 RamenOriSplitView を concat する view (RamenOriMultiDataset.make_train_val_test_split 出力)。

    `indices` は multi 全体空間 (0..sum(len(sub))-1) 上の flat indices を expose する。
    これにより train.py の train_indices 経路 (frame-level weights を train subset で slice) と
    そのまま互換。
    """

    def __init__(
        self,
        sub_views: list[RamenOriSplitView],
        multi_offsets: np.ndarray,
    ) -> None:
        self.sub_views = list(sub_views)
        # sub_views 側 (per-sub 空間) の cumulative length: [0, N0, N0+N1, ...]
        self._sub_view_offsets = np.cumsum([0] + [len(v) for v in self.sub_views])
        # multi 全体空間 flat indices を build (train.py train_indices で使う)
        flat_indices: list[int] = []
        for k, v in enumerate(self.sub_views):
            base_offset = int(multi_offsets[k])
            flat_indices.extend(base_offset + i for i in v.indices)
        self.indices = flat_indices

    def __len__(self) -> int:
        return int(self._sub_view_offsets[-1])

    def __getitem__(self, i: int) -> dict:
        # sub-view 空間で routing (offsets 側 index、view の __getitem__ に渡す)
        k = int(np.searchsorted(self._sub_view_offsets[1:], i, side="right"))
        local_i = i - int(self._sub_view_offsets[k])
        return self.sub_views[k][local_i]


# multi の sub ごとに書ける key (data の場所)。base_dataset は test の DI 用
_SUB_DATASET_KEYS = frozenset(
    {
        "merged_source_root",
        "repo_id",
        "frame_cache_root_baked",
        "obb_precomputed_root",
        "obb_precomputed_hash",
        "base_dataset",
    }
)
# 全 sub で同じはずの設定 (multi も同じ名前で持つ)
_SHARED_SUB_ATTRS = (
    "chunk_len",
    "img_size",
    "camera_keys",
    "state_variant",
    "load_prev_image",
    "use_relative_action",
    "frame_cache_mode",
    "frame_cache_num_variants",
    "obb_source",
    "overlay_conf_threshold",
    "overlay_line_thickness",
    "overlay_class_filter",
    "memory",
    "memory_yolo_ckpt",   # memory を作った OBB の YOLO の重み (sub で違えば止める)
)


class RamenOriMultiDataset(torch.utils.data.Dataset):
    """複数の RamenOriLerobotDataset を wrap する ConcatDataset 相当。

    Issue #129 Phase K (2026-09-01): task 別に baked frame_cache_v2 が既に生成済
    (task5=/nvme/rotate_table_base, task7=/nvme/task7, task0=/nvme/merged_sources/combined_task0)
    の状態で、それらを再 merge せずに multi-dataset として dataloader 側 mix する。
    task5+7 の "merge → aug bake" 一連 (~5h 作業 + mp4 再 download) を skip し、既存 baked
    cache を pointing だけで α (task5+7) / β (task5+7+0) 両方を回せる。

    Design:
    - `sub_datasets`: list of per-sub config dict。書けるのは data の場所だけ (`_SUB_DATASET_KEYS`:
      `merged_source_root`、`repo_id`、`frame_cache_root_baked`、`obb_precomputed_root`、`obb_precomputed_hash`)。
      それ以外を書くとエラー (Issue #141 RO-10)
    - shared kwargs (`chunk_len`, `img_size`, `camera_keys`, `state_variant`, `frame_cache_mode`,
      `use_relative_action`, `relative_stats_path`, `augmentation_cfg` 等) は 全 sub に broadcast
    - 各 sub は独立の `RamenOriLerobotDataset` として instantiate、baked cache は各自の path
    - 全 sub で同じはずの設定 (`_SHARED_SUB_ATTRS`) は multi も同じ名前の属性で持つ (ckpt の約束が読む)

    1:1 mix の実現方法:
    - 各 sub の skill_id は、その sub の episode meta の `source_task_index` から作る (config に表は持たない)
    - train.py の `balancing.strategy=task_uniform` は `sample_skill_ids()` の値で
      1:1 balance するので、各 sub が単一 skill_id (rotate=4, move=5, insert=0 等) の場合、
      自動的に sub-dataset uniform mix になる (= handoff §5.1 "1:1 mix" 相当)

    train.py 経路との互換:
    - `__len__` / `__getitem__` は ConcatDataset 相当 (searchsorted で sub routing)
    - `sample_episode_metadata()` は 各 sub の meta を concat
    - `sample_skill_ids()` は skill_id per frame を concat 返却
    - `make_train_val_test_split()` は sub 別に split → concat + multi 空間 index offset

    現状 caveat:
    - split_json は sub-dataset 別に json を要求する schema が未整備、Phase K では
      val.seed=42 の on-the-fly split のみ対応 (issue #129 の R-6 bench で split_json は未使用)
    """

    def __init__(
        self,
        sub_datasets: list[dict],
        augmentation_cfg: dict | None = None,
        # shared kwargs: 全 sub に broadcast。RamenOriLerobotDataset の __init__ シグネチャに準ずる
        chunk_len: int = 16,
        img_size: int = 224,
        camera_keys: list[str] | None = None,
        top_K: int = 4,
        num_classes: int = 7,
        num_cams: int = 4,
        obb_source: str = "none",
        overlay_conf_threshold: float = 0.30,
        overlay_line_thickness: int = 2,
        overlay_class_filter: list[int] | None = None,
        state_variant: str = "71d",
        load_prev_image: bool = True,
        include_depth_target: bool = False,
        depth_target_size: tuple[int, int] = (32, 32),
        depth_target_num_cams: int = 3,
        use_relative_action: bool = False,
        # Issue #129 Phase K (2026-09-01): rel action 用 stats は shared kwarg として全 sub に
        # broadcast (unified stats)。sub では上書きできない (Issue #141 RO-10)。
        # rationale: batch は sub 間 mix、model 側 (train.py の cfg.data.relative_stats_path 経由)
        # denormalize は 1 個の stats しか使えないため、dataset 側 normalize も同じ stats に統一
        # した方が model 出力の意味論が consistent。task5 vs task7 の std 差 ~1.4x なので
        # unified 化の精度損失は小さい。
        relative_stats_path: str | None = None,
        frame_cache_mode: str = "online",
        frame_cache_num_variants: int = 1,
        auto_precompute_frame_cache: bool = True,
        memory: bool = False,
    ) -> None:
        if not sub_datasets or len(sub_datasets) < 1:
            raise ValueError("RamenOriMultiDataset requires at least 1 sub_dataset config")

        shared_kwargs = dict(
            chunk_len=chunk_len,
            img_size=img_size,
            camera_keys=camera_keys,
            top_K=top_K,
            num_classes=num_classes,
            num_cams=num_cams,
            obb_source=obb_source,
            overlay_conf_threshold=overlay_conf_threshold,
            overlay_line_thickness=overlay_line_thickness,
            overlay_class_filter=overlay_class_filter,
            state_variant=state_variant,
            load_prev_image=load_prev_image,
            include_depth_target=include_depth_target,
            depth_target_size=depth_target_size,
            depth_target_num_cams=depth_target_num_cams,
            use_relative_action=use_relative_action,
            relative_stats_path=relative_stats_path,
            frame_cache_mode=frame_cache_mode,
            frame_cache_num_variants=frame_cache_num_variants,
            auto_precompute_frame_cache=auto_precompute_frame_cache,
            augmentation_cfg=augmentation_cfg,
            memory=memory,
        )

        self.subs: list[RamenOriLerobotDataset] = []
        for sub_i, sub_item in enumerate(sub_datasets):
            # test DI 経路: 既に build 済 RamenOriLerobotDataset を直接 pass 可能
            if isinstance(sub_item, RamenOriLerobotDataset):
                self.subs.append(sub_item)
                continue
            # config driven 経路: sub に書けるのは data の場所だけ (Issue #141 RO-10)。
            # 旧実装は sub の値 (null を含む) で共通の設定を上書きし、Phase K の rel では sub の
            # relative_stats_path: null が共通の統計を消して、学習と推論が別の統計を使っていた
            sub_cfg = dict(sub_item) if not isinstance(sub_item, dict) else sub_item
            shared_in_sub = sorted(set(sub_cfg) - _SUB_DATASET_KEYS)
            if shared_in_sub:
                raise ValueError(
                    f"sub_datasets[{sub_i}] に sub では書けない key {shared_in_sub} がある。"
                    f"sub に書けるのは {sorted(_SUB_DATASET_KEYS)} だけで、それ以外は全 sub 共通の設定に書く"
                )
            try:
                self.subs.append(RamenOriLerobotDataset(**shared_kwargs, **sub_cfg))
            except Exception as e:
                raise RuntimeError(
                    f"failed to instantiate sub_datasets[{sub_i}] with keys {list(sub_cfg.keys())}: {e}"
                ) from e

        # 全 sub で同じはずの設定を multi も持つ (batch の key・形と ckpt の約束がこれで決まる)。
        # config 経路では共通の設定なので必ず揃う。揃わないのは build 済みの sub を渡したとき
        for attr in _SHARED_SUB_ATTRS:
            values = [getattr(sub, attr) for sub in self.subs]
            if any(v != values[0] for v in values[1:]):
                raise ValueError(f"sub_datasets の {attr} が揃っていない: {values}")
            setattr(self, attr, values[0])
        # multi 全体 offset: [0, N0, N0+N1, ..., sum]
        lens = [len(s) for s in self.subs]
        self._offsets = np.cumsum([0] + lens)
        self._total = int(self._offsets[-1])
        # curriculum / AWR 用 metadata cache (2 回目呼出しで再計算しない)
        self._cached_ep_meta: dict | None = None
        self._cached_skill_ids: np.ndarray | None = None

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, i: int) -> dict:
        # searchsorted([N0, N0+N1, ...], i, side="right") で sub index を確定
        # (offsets[k+1] > i の最小 k)
        k = int(np.searchsorted(self._offsets[1:], i, side="right"))
        local_i = i - int(self._offsets[k])
        return self.subs[k][local_i]

    def state_action_item(self, i: int) -> dict:
        """動画を読まずに state / action / skill_id などを返す (正規化の統計用、sub に routing)。"""
        k = int(np.searchsorted(self._offsets[1:], i, side="right"))
        return self.subs[k].state_action_item(i - int(self._offsets[k]))

    def sample_episode_metadata(self) -> dict:
        """全 sub の episode meta を concat。

        source_episode_names は `sub{k}::<name>` 形式にして sub 間で globally unique に保つ
        (source-grouped split の grouping key が sub を跨いで leak しないよう)。
        """
        if self._cached_ep_meta is not None:
            return self._cached_ep_meta
        metas = [sub.sample_episode_metadata() for sub in self.subs]
        meta = {
            "episode_lengths": np.concatenate([m["episode_lengths"] for m in metas]),
            "episode_skill_ids": np.concatenate([m["episode_skill_ids"] for m in metas]),
            "source_episode_names": [
                f"sub{k}::{n}" for k, m in enumerate(metas) for n in m["source_episode_names"]
            ],
            "num_episodes": int(sum(m["num_episodes"] for m in metas)),
        }
        self._cached_ep_meta = meta
        return meta

    def memory_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """全 sub の memory の表と frame ごとの skill_id を concat。"""
        parts = [sub.memory_arrays() for sub in self.subs]
        return np.concatenate([t for t, _ in parts]), np.concatenate([s for _, s in parts])

    def sample_skill_ids(self) -> np.ndarray:
        """全 frame の skill_id 配列を返す (各 sub の sample_skill_ids を concat)。"""
        if self._cached_skill_ids is not None:
            return self._cached_skill_ids
        result = np.concatenate([sub.sample_skill_ids() for sub in self.subs])
        if len(result) != self._total:
            raise RuntimeError(
                f"MultiDataset sample_skill_ids length {len(result)} != dataset len {self._total}"
            )
        self._cached_skill_ids = result
        return result

    def make_train_val_test_split(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        split_json: str | Path | None = None,
    ) -> tuple[RamenOriMultiSplitView, RamenOriMultiSplitView, RamenOriMultiSplitView]:
        """各 sub 別に episode-level split を実施 → concat して multi 空間 view を返す。

        split_json は現状未対応 (schema が sub 別 json list を要求するため)。R-6 bench では
        val.seed=42 の on-the-fly split のみ使用。
        """
        if split_json is not None:
            raise NotImplementedError(
                "RamenOriMultiDataset は split_json 未対応。val.seed=... で on-the-fly split を使う事 "
                "(将来対応時は per-sub json list schema に拡張)"
            )
        train_views: list[RamenOriSplitView] = []
        val_views: list[RamenOriSplitView] = []
        test_views: list[RamenOriSplitView] = []
        for sub in self.subs:
            t, v, tt = sub.make_train_val_test_split(
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                split_json=None,
            )
            train_views.append(t)
            val_views.append(v)
            test_views.append(tt)
        return (
            RamenOriMultiSplitView(train_views, self._offsets),
            RamenOriMultiSplitView(val_views, self._offsets),
            RamenOriMultiSplitView(test_views, self._offsets),
        )


_LOAD_TASKS_PATCHED = False


def _patch_lerobot_load_tasks() -> None:
    """fork lerobot 0.5.1 と curation_tool の lerobot 0.6.0 で書かれた tasks.parquet 形式差分の
    monkey-patch (module load 時 1 回、冪等)。

    差分:
    - 0.6.0 build の chunks: column = ``task_index`` + ``__index_level_0__`` (task 名)
    - 0.5.1 fork の load_tasks: ``task`` column を期待、set_index("task") で index 化
    - `_getitem_inner` は ``self.meta.tasks.iloc[task_idx].name`` で lookup、
      task_index=7 の chunk では iloc[7] で out-of-bounds

    修正: `__index_level_0__` を `task` に rename + 0..max_tid で sparse pad
    (missing tid は ``__unused_task_N__`` プレースホルダ)、iloc[tid].name が
    正しい task 名を返す形にする。
    """
    global _LOAD_TASKS_PATCHED
    if _LOAD_TASKS_PATCHED:
        return
    import pandas as pd
    from lerobot.datasets import lerobot_dataset as _lerobot_dataset
    from lerobot.datasets import utils as _lerobot_utils

    def _patched(local_dir):
        tasks = pd.read_parquet(local_dir / _lerobot_utils.DEFAULT_TASKS_PATH)
        if "__index_level_0__" in tasks.columns and "task" not in tasks.columns:
            tasks = tasks.rename(columns={"__index_level_0__": "task"})
        if "task_index" not in tasks.columns:
            raise ValueError(
                f"tasks.parquet missing 'task_index' column: got {list(tasks.columns)}"
            )
        # iloc[task_idx] が正しい .name を返すよう sparse pad
        rows_by_tid = {
            int(row["task_index"]): str(row["task"]) for _, row in tasks.iterrows()
        }
        max_tid = max(rows_by_tid)
        padded_rows = [
            {"task": rows_by_tid.get(i, f"__unused_task_{i}__"), "task_index": i}
            for i in range(max_tid + 1)
        ]
        result = pd.DataFrame(padded_rows).set_index("task")
        result.index.name = "task"
        return result

    _lerobot_utils.load_tasks = _patched
    _lerobot_dataset.load_tasks = _patched
    _LOAD_TASKS_PATCHED = True


def _load_split_from_json(
    split_json_path: Path, *, num_episodes: int
) -> tuple[list[int], list[int], list[int]]:
    """External split JSON (export_split_json.py 出力形式) を読み、episode indices を返す。

    Cross-server で strict identical split を保証するため、Sakura で作った JSON を
    RAMEN-Ori server に scp してこの function 経由で使う。JSON schema は
    ``team_ramen_grouped_episode_split_v1``: `splits.{train,val,test}.episode_indices`
    (0..num_episodes-1 の subset、disjoint、union が全 episode に一致)。

    Args:
        split_json_path: Sakura の GR00T-side で使った split JSON path
        num_episodes: dataset 実 episode 数 (mismatch なら error、HF revision 差検知)

    Returns:
        (train_eps, val_eps, test_eps): 各 sorted list of episode indices
    """
    import json  # lazy
    if not split_json_path.is_file():
        raise FileNotFoundError(f"split_json not found: {split_json_path}")
    payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "team_ramen_grouped_episode_split_v1":
        raise ValueError(
            f"unsupported split_json schema: {payload.get('schema_version')!r}"
        )
    if int(payload.get("num_episodes", -1)) != num_episodes:
        raise RuntimeError(
            f"split_json episode count {payload.get('num_episodes')} != dataset {num_episodes}; "
            "HF revision or merged_source content mismatch の疑い (GR00T server と dataset "
            "revision 揃えて再走 or split_json 再生成)"
        )
    splits = payload.get("splits", {})
    result: list[list[int]] = []
    for name in ("train", "validation", "test"):
        eps = splits.get(name, {}).get("episode_indices")
        if not isinstance(eps, list) or not all(
            isinstance(v, int) and 0 <= v < num_episodes for v in eps
        ):
            raise ValueError(f"split_json invalid episode_indices for {name!r}")
        result.append(sorted(eps))
    train_eps, val_eps, test_eps = result
    all_eps = set(train_eps) | set(val_eps) | set(test_eps)
    if all_eps != set(range(num_episodes)):
        raise ValueError(
            f"split_json coverage mismatch: sets union {len(all_eps)} != {num_episodes}"
        )
    return train_eps, val_eps, test_eps


def _expand_episode_split_to_frames(
    train_eps: list[int],
    val_eps: list[int],
    test_eps: list[int],
    episode_lengths: np.ndarray,
) -> tuple[list[int], list[int], list[int]]:
    """Episode-level split → frame-level global index list (ConcatLeRobot indexing 互換)。"""
    train_set = set(train_eps)
    val_set = set(val_eps)
    test_set = set(test_eps)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    cursor = 0
    for ep_i, length in enumerate(episode_lengths):
        length_int = int(length)
        end = cursor + length_int
        rng_range = range(cursor, end)
        if ep_i in train_set:
            train_idx.extend(rng_range)
        elif ep_i in val_set:
            val_idx.extend(rng_range)
        elif ep_i in test_set:
            test_idx.extend(rng_range)
        # ep_i がどこにも無い場合は無視 (split JSON coverage check で先に error)
        cursor = end
    return train_idx, val_idx, test_idx


def _auto_precompute_frame_cache(merged_source_root: Path) -> None:
    """Merged source に対し JPG frame cache を precompute (Issue #122 refactor)。

    GR00T の train_lerobot.sh も同じ merged_source に対し precompute する。両者
    hash-based invalidation で「既に済みなら skip」= どっち先に走らせても片方は
    5-10min の待ちなし、cache 共有で disk 50GB 節約。

    Precompute の副作用:
    - 初回: 5-10 min の decode 待ち (200k frames × 3-8 cam)
    - 完走後 mp4 常時 auto drop (Issue #122: lerobot_frame_cache_patch の
      _check_cached_episodes_sufficient 差替で mp4 不在でも train 成立、disk 節約)
    - 2 回目以降 (hash 一致): 即 return (~1s)

    Env override:
    - FRAME_CACHE_PRECOMPUTE=false: precompute step skip (既存 cache 前提、
      GR00T 側で既に済んでいる想定でこの env を使う)
    """
    # 起動時 log は precompute script 側の stdout を継承
    import subprocess  # lazy import
    import sys as _sys

    script = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "bitrobot_lerobot_subtask_datasets"
        / "scripts"
        / "precompute_frame_cache.py"
    )
    if not script.is_file():
        # data pipeline 配下の script が見つからないなら silently skip (test 環境等)
        return

    cmd = [_sys.executable, str(script), "--lerobot-root", str(merged_source_root)]
    print(
        f"[ramen_ori] auto precompute frame cache for {merged_source_root} "
        f"(env FRAME_CACHE_PRECOMPUTE=false で skip 可能)",
        flush=True,
    )
    # 完了まで block、exit code non-zero なら例外
    subprocess.run(cmd, check=True)


def _restrict_video_keys(meta, keep: list[str]) -> None:
    """LeRobot が item ごとに読みに行く動画を keep (camera_keys) だけにする (Issue #141)。

    LeRobot は meta の features のうち video を全部読みに行く (時刻の問い合わせ・decode・image_transforms)。
    RAMEN-Ori が使うのは head_left / 左右の手首の 3 本だけで、IR 4 本と head_right は捨てていた。
    統合 frame cache は IR / head_right を展開せず mp4 も落とさないので、残すと最初の item で落ちる。
    dataset を作った後 (parquet の読み込みと cache の確認の後) に meta の一覧から外す。

    Raises:
        ValueError: keep に dataset の動画に無い key がある
    """
    videos = set(meta.video_keys)
    missing = [k for k in keep if k not in videos]
    if missing:
        raise ValueError(f"camera_keys {missing} are not video features of the dataset: {sorted(videos)}")
    for key in videos - set(keep):
        del meta.info["features"][key]


def _iter_episode_rows(episodes):
    """LeRobot meta.episodes (HF datasets.Dataset / pandas DataFrame / list of dict) の row を順に返す。"""
    if hasattr(episodes, "iterrows"):  # pandas iterrows は (idx, row)
        for _, row in episodes.iterrows():
            yield row
    else:
        yield from episodes


def _episode_length(row) -> int:
    if "length" in row and row["length"] is not None:
        return int(row["length"])
    if "dataset_to_index" in row and "dataset_from_index" in row:
        return int(row["dataset_to_index"]) - int(row["dataset_from_index"])
    raise ValueError(f"episode row lacks length/dataset_index: {row}")


def _episode_skill_id(row) -> int:
    """LeRobot v3 episode row → RAMEN-Ori skill_id (Issue #141 RO-1)。

    episode meta の ``source_task_index`` (BitRobot の元の task 番号、build_curated_task_dataset.py
    が出力し merge でも変わらない) を skill_mapping の表で変換する。frame / episode の
    ``task_index`` は dataset ごとのローカル番号 (merge で 0 始まりに振り直される) なので代用しない。

    Raises:
        ValueError: ``source_task_index`` が無い
        KeyError: skill_mapping に無い番号 (tid=1 move_to_table、tid=6 intro など)
    """
    v = row["source_task_index"] if "source_task_index" in row else None
    if v is None:
        raise ValueError(
            "episode row lacks source_task_index (skill_id の元)。frame の task_index は "
            f"ローカル番号なので代用しない (available keys: {list(row.keys())[:20]}...)"
        )
    if hasattr(v, "__iter__") and not isinstance(v, str):
        v = next(iter(v))
    return task_index_to_skill_id(int(v))


def _extract_source_episode_name(row, sub_idx: int, ep_i: int) -> str:
    """LeRobot v3 episode row → source_episode_name (leakage-safe grouping key)。

    curated chunk repo (build_curated_task_dataset.py) は ``source_episode_index``
    (BitRobot G1_WBT の raw 533ep への stable ref) を持つ。同じ raw 収録から複数の
    curated segment が抽出される (task 5 + task 7 が同 session に共存) ため、これを
    grouping key として使うと source recording 単位の train/val 分離を保証できる。

    Fallback (``source_episode_index`` が row に無い場合)、``f"local_{sub_idx}_{ep_i}"``
    (per-episode 一意) を返す。leakage-safety は失われるが、少なくとも grouping key として
    unique になり、per-episode shuffle と等価な挙動になる。
    """
    if "source_episode_index" in row and row["source_episode_index"] is not None:
        v = row["source_episode_index"]
        if hasattr(v, "__iter__") and not isinstance(v, str):
            v = next(iter(v))
        return f"src{int(v):05d}"
    return f"local_{sub_idx}_{ep_i}"


def compute_task_uniform_weights(task_indices: np.ndarray) -> np.ndarray:
    """skill_id array → per-sample weight (skill ごと 1/(N_skill * n_frames_in_skill))。

    Sum(weights * n_frames_in_skill) = 1 per skill → sampler で skill 間 1:1 balance。
    """
    unique, counts = np.unique(task_indices, return_counts=True)
    weight_by_task = {int(tid): 1.0 / (len(unique) * int(cnt)) for tid, cnt in zip(unique, counts)}
    return np.array([weight_by_task[int(t)] for t in task_indices], dtype=np.float64)


def _to_numpy(x) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _scalar_from(x) -> int:
    """torch/np scalar or 0-D or (1,) から int を取り出す。"""
    if isinstance(x, torch.Tensor):
        return int(x.reshape(-1)[0].item())
    arr = np.asarray(x).reshape(-1)
    return int(arr[0])
