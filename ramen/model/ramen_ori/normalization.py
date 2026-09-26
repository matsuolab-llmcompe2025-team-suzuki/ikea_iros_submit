"""RAMEN-Ori の state / action 正規化 (Issue #141 RO-4 / RO-6)。

# 方針

- 次元ごとに平均・std で正規化する (z-score、切り詰めなし、std に下限)
- 統計は全 sub (skill) をまとめて 1 つ。skill ごとに同じ重み (balancing の task_uniform と同じ 1:1)。
  frame 数で混ぜると β では insert が統計を決めてしまう
- 統計は学習の開始時に dataset と同じコード (`state_action_item`、動画は読まない) で全 frame から計算し、
  model の buffer に入れる → ckpt に入り、dataset・L4・val・推論が同じ値を使う
- model は元の単位の state / action を受け取り、中で正規化し、元の単位で action を返す

# skill ごとに同じ重みのまとめ方

skill s の平均 m_s・分散 v_s から、skill を等確率で引いたときの分布の平均・分散:

    mean = (1/S) Σ_s m_s
    var  = (1/S) Σ_s (v_s + (m_s − mean)²)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SkillBalancedMoments:
    """skill ごとに件数・和・二乗和 (float64) を貯め、skill を同じ重みで混ぜた分布の平均・std を返す。"""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._count: dict[int, int] = {}
        self._sum: dict[int, np.ndarray] = {}
        self._sumsq: dict[int, np.ndarray] = {}

    def update(self, values: np.ndarray, skill_ids: np.ndarray) -> None:
        """values: (N, dim)、skill_ids: (N,)"""
        values = np.asarray(values, dtype=np.float64)
        skill_ids = np.asarray(skill_ids)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"values must be (N, {self.dim}), got {values.shape}")
        for sid in np.unique(skill_ids):
            v = values[skill_ids == sid]
            key = int(sid)
            if key not in self._count:
                self._count[key] = 0
                self._sum[key] = np.zeros(self.dim)
                self._sumsq[key] = np.zeros(self.dim)
            self._count[key] += len(v)
            self._sum[key] += v.sum(axis=0)
            self._sumsq[key] += (v * v).sum(axis=0)

    @property
    def counts(self) -> dict[int, int]:
        """skill_id → 集計した件数"""
        return dict(self._count)

    def per_skill(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """skill_id → (平均, 分散)"""
        out = {}
        for sid, n in self._count.items():
            mean = self._sum[sid] / n
            var = np.maximum(self._sumsq[sid] / n - mean * mean, 0.0)
            out[sid] = (mean, var)
        return out

    def finalize(self, std_min: float) -> tuple[np.ndarray, np.ndarray]:
        """skill を同じ重みで混ぜた分布の (平均, std)。std は std_min で下限を付ける。"""
        if not self._count:
            raise ValueError("no samples accumulated")
        per = self.per_skill()
        means = np.stack([m for m, _ in per.values()])
        variances = np.stack([v for _, v in per.values()])
        mean = means.mean(axis=0)
        var = (variances + (means - mean) ** 2).mean(axis=0)
        std = np.maximum(np.sqrt(var), std_min)
        return mean.astype(np.float32), std.astype(np.float32)


class Normalizer(nn.Module):
    """次元ごとの平均・std を buffer に持ち、正規化と逆変換を行う。

    統計は `set_stats` か `load_state_dict` で入る。入る前に使うとエラー (EMA の重みだけを
    strict=False で読んだ場合など、buffer が初期値のまま残る事故を黙って通さない)。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("ready", torch.tensor(False))
        # buffer の ready を毎 forward で CPU に読むと同期が入るので、確認は 1 回だけ
        self._checked = False

    def set_stats(self, mean, std) -> None:
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.shape != (self.dim,) or std.shape != (self.dim,):
            raise ValueError(
                f"stats must be ({self.dim},), got mean={tuple(mean.shape)} std={tuple(std.shape)}"
            )
        if not bool(torch.all(std > 0)):
            raise ValueError("std must be positive")
        self.mean.copy_(mean)
        self.std.copy_(std)
        self.ready.fill_(True)
        self._checked = False

    def _check_ready(self) -> None:
        if self._checked:
            return
        if not bool(self.ready):
            raise RuntimeError(
                "Normalizer used before stats were set (set_stats / load_state_dict で "
                "mean・std・ready を入れること。EMA の重みだけでは buffer は入らない)"
            )
        self._checked = True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._check_ready()
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        self._check_ready()
        return x * self.std + self.mean


class _StateActionView(Dataset):
    """dataset の state_action_item (動画を読まない) を DataLoader で並列に回すための view。"""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> dict:
        item = self.dataset.state_action_item(i)
        return {
            "state": item["state"],
            "action0": item["action"][0],   # その frame の指令 (正解 chunk の各行はどれかの frame の指令)
            "waist0": item["action_waist_teacher"][0],   # その frame の腰の指令 (FK 用)
            "skill_id": item["skill_id"],
        }


def compute_normalization_moments(
    dataset,
    *,
    fk: nn.Module | None = None,
    num_workers: int = 0,
    batch_size: int = 1024,
) -> dict[str, SkillBalancedMoments]:
    """dataset の全 frame の state・指令 (action の row 0)・FK の量を skill ごとに集計する (1 回の pass)。

    dataset は `state_action_item(i)` (state / action / action_waist_teacher / skill_id を返す、
    学習と同じ変換) と `__len__` を持つこと (RamenOriLerobotDataset / RamenOriMultiDataset)。
    fk (G1WristFKTorch) を渡すと、教師の指令 (腰 + 腕 + hand の 19D) を FK に通した 27 次元
    (`fk.features`) も集める (FK の loss の単位そろえ用、Issue #141 RO-16)。

    Returns:
        {"state": ..., "action": ..., "fk": ... (fk を渡したときだけ)}
    """
    from model.ramen_ori.fk import assemble_action19  # lazy: fk 依存を optional に

    loader = DataLoader(
        _StateActionView(dataset), batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    moments: dict[str, SkillBalancedMoments] = {}

    def _update(key: str, values: np.ndarray, skill_ids: np.ndarray) -> None:
        if key not in moments:
            moments[key] = SkillBalancedMoments(values.shape[1])
        moments[key].update(values, skill_ids)

    for b in loader:
        skill_ids = b["skill_id"].numpy()
        _update("state", b["state"].numpy(), skill_ids)
        _update("action", b["action0"].numpy(), skill_ids)
        if fk is not None:
            with torch.no_grad():
                q19 = assemble_action19(b["waist0"].float(), b["action0"].float())
                _update("fk", fk.features(q19).numpy(), skill_ids)
    if not moments:
        raise ValueError("dataset is empty")
    return moments


def skill_balanced_percentiles(
    values: np.ndarray, skill_ids: np.ndarray, q: tuple[float, ...], seed: int = 0
) -> list[np.ndarray]:
    """skill ごとに同じ数 (一番少ない skill の frame 数) の frame を引いて、次元ごとの percentile を返す。

    memory の切り詰めの範囲用 (Issue #141 Phase 7)。平均・std の「skill ごとに同じ重み」と揃える。
    """
    rng = np.random.default_rng(seed)
    skill_ids = np.asarray(skill_ids)
    by_skill = [np.flatnonzero(skill_ids == s) for s in np.unique(skill_ids)]
    n = min(len(i) for i in by_skill)
    picked = np.concatenate([rng.choice(i, n, replace=False) for i in by_skill])
    return [np.percentile(values[picked], p, axis=0) for p in q]


def normalized_std_by_skill(
    moments: SkillBalancedMoments, std: np.ndarray
) -> dict[int, np.ndarray]:
    """skill ごとの正規化後の std (skill 内の std ÷ 全体の std)。起動時の log 用。"""
    return {sid: np.sqrt(var) / std for sid, (_, var) in moments.per_skill().items()}
