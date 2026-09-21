"""RAMEN-Ori Policy: 7 sub-module 統合の VLA (Issue #115、Phase 1 M8)。

Vision (A-1 LingBot frozen) + Temporal (D-3 Δ encoder) + OBB (C-2 coord token) +
State (E-δ 71D) + Skill (F-3 embedding) → Fusion (G-1 6-layer transformer) →
Action Expert (B-7 Flow Matching DiT) の統合 policy。

# Batch format (data.py が供給)

```
batch = {
    # Vision + Delta
    "images":        (B, N_cams, 3, 224, 224)  float, ImageNet-normalize 済 I_t
    "images_prev":   (B, N_cams, 3, 224, 224)  float, I_{t-1} (temporal があるときだけ使う)
    "cam_id":        (B, N_cams)               long, cam vocabulary global id
    # OBB
    "obb_verts":      (B, N_cams, top_K, 8)    float, normalized xyxyxyxy
    "obb_conf":       (B, N_cams, top_K, 1)    float [0,1]
    "obb_class_id":   (B, N_cams, top_K)       long
    "obb_cam_id":     (B, N_cams, top_K)       long
    "obb_valid_mask": (B, N_cams, top_K)       bool
    # State + Skill
    "state":    (B, state_dim)                 float, joint + tracking + vel + hand + EE (元の単位)
    "skill_id": (B,)                           long, [0, num_skills)
    # Training only
    "action":   (B, chunk_len, action_dim)     float, ground truth action chunk (元の単位)
}
```

# Method surface

- `compute_loss(batch) -> Tensor`: training 用 loss (`compute_loss_parts(batch)["loss"]`)
- `compute_loss_parts(batch) -> dict`: loss と内訳 (BC、FK の 3 項と重み、aux)。batch["action_is_pad"] の行は除く
- `predict_action(batch) -> Tensor`: inference 用 Euler 積分 sampling、`(B, chunk_len, action_dim)` (元の単位)

# 正規化 (Issue #141 RO-4 / RO-6、`normalization=True`)

state と action の平均・std を Normalizer の buffer に持つ (学習の開始時に `set_normalization_stats`、
ckpt の model_state_dict に入る)。入出力は元の単位のまま、中で state を正規化して StateEncoder に入れ、
flow matching は正規化した action で行い、`predict_action` は元の単位に戻して返す。
既定 (False) は Phase K と同じ (推論が base.yaml から組み立てるため、既定値は Phase K の構造に保つ)。

# Sub-module DI (Issue #120 Phase A-0 で DI 化)

全 sub-module (vision / temporal / obb / state / skill / fusion / action_expert) を
constructor で instance を受ける形。Hydra 側で `_target_` swap するだけで G-2/G-3
Fusion 系や B-8 MeanFlow などの alternative を差替可能。sub-module の内部 config は
それぞれの `_target_` block で完結、model.py は「global dispatch」だけ持つ。
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ramen_ori.normalization import Normalizer

# Alt-7: action_expert が compute_loss/sample method を持つ contract (Flow Matching / MeanFlow 共通)

_log = logging.getLogger(__name__)


class RamenOriPolicy(nn.Module):
    def __init__(
        self,
        vision: nn.Module,
        temporal: nn.Module | None,     # None = 画像差分を使わない (Issue #141 RO-11)
        obb: nn.Module,
        state: nn.Module,
        skill: nn.Module,
        fusion: nn.Module,
        action_expert: nn.Module,
        aux_head: nn.Module | None = None,   # I-6 Depth aux (Alt-3、None = I-1 baseline)
        aux_weight: float = 0.1,             # aux loss weight (aux_head 有効時)
        # FK の loss (L4、Issue #129 Phase B → Issue #141 RO-16 で 3 項に作り直し)。
        # 1-step Euler で復元した予測 x̂1 を元の単位に戻し、教師の腰と合わせて 19D → URDF FK。
        # 左側 / 右側 = 肘の位置・手先の位置・手の向き (rot6d) を教師の std で割って 1/3 ずつ平均、
        # 両側 = 両手先の位置の差。重みは毎 step `(fk_loss_ratio / 3) × BC ÷ 項` (値だけ、上限 fk_weight_max)
        fk: nn.Module | None = None,         # G1WristFKTorch instance、None = auto-load if l4_enabled
        l4_enabled: bool = False,
        fk_loss_ratio: float = 0.2,          # FK の 3 項の合計を BC の何割にするか (RO-12)
        fk_weight_max: float = 1.2,          # 各項の重みの上限 (学習の後半で項が小さくなると重みが大きくなるため、base.yaml と同じ)
        # Issue #129 Phase F (2026-08-31): relative action space (Run 3/5)。
        # arms 14 dim を Δq (normalized) で予測、hand 2 は absolute pass-through。
        # Issue #141 で保留中 (abs 集中)。FK の loss・正規化とは併用できない (組み立て時にエラー)。
        use_relative_action: bool = False,
        relative_stats: dict | None = None,  # {"mean": (14,), "std": (14,)} tensor、use_relative_action=True 時 required
        # Issue #141 RO-4 / RO-6: state / action を model の中で平均・std 正規化 (統計は set_normalization_stats)
        normalization: bool = False,
        # 正規化した後の state を ±state_clip に切り詰める (正規化のときだけ)。std の小さい次元 (velocity の腰など) は
        # 正規化後に 30 を超える値が出て、入口の層に勾配が集中して学習が崩れた (Issue #141、c32 の 30k step)
        state_clip: float | None = None,
        # Issue #141 Phase 7: memory の token (memory.MemoryEncoder)。None = memory を使わない (Phase K と同じ構造)。
        # batch に "memory" (B, 51) が要る (統計は set_memory_stats)
        memory: nn.Module | None = None,
        # global dispatch 用 (sub-module 内部 hyperparam は各 module で保持)
        d_model: int = 512,
        num_cams: int = 4,
        chunk_len: int = 16,
        action_dim: int = 16,
        sample_n_steps: int = 6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_cams = num_cams
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.sample_n_steps = sample_n_steps
        self.aux_weight = float(aux_weight)
        self.l4_enabled = bool(l4_enabled)
        self.fk_loss_ratio = float(fk_loss_ratio)
        self.fk_weight_max = float(fk_weight_max)

        self.vision = vision
        self.temporal = temporal
        self.obb = obb
        self.state = state
        self.skill = skill
        self.fusion = fusion
        self.action_expert = action_expert
        self.memory = memory  # type: ignore[assignment]
        # None のまま登録すると nn.Module 側で "not a module" 警告なので条件付き register
        self.aux_head = aux_head  # type: ignore[assignment]

        # L4 が有効で fk 未指定 → default URDF (model/ramen_ori/assets/urdf/) から auto-load
        if self.l4_enabled and fk is None:
            from model.ramen_ori.fk import G1WristFKTorch  # lazy: fk import は heavy (URDF parse)
            fk = G1WristFKTorch.from_default_urdf()
        self.fk = fk  # type: ignore[assignment]

        # Issue #129 Phase F (2026-08-31): relative action space
        self.use_relative_action = bool(use_relative_action)
        if self.use_relative_action:
            if relative_stats is None:
                raise ValueError(
                    "use_relative_action=True requires relative_stats "
                    "(load via model.ramen_ori.relative_action.load_relative_stats and pass as dict)"
                )
            mean = relative_stats["mean"]
            std = relative_stats["std"]
            if not isinstance(mean, torch.Tensor):
                mean = torch.as_tensor(mean, dtype=torch.float32)
            if not isinstance(std, torch.Tensor):
                std = torch.as_tensor(std, dtype=torch.float32)
            if mean.shape != (14,) or std.shape != (14,):
                raise ValueError(
                    f"relative_stats mean/std must be (14,), got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
                )
            # buffer 登録 (device 移動 / state_dict 保存対応、gradient 対象外)
            self.register_buffer("_relative_arms_mean", mean.float())
            self.register_buffer("_relative_arms_std", std.float().clamp(min=1e-6))
        else:
            self._relative_arms_mean = None  # type: ignore[assignment]
            self._relative_arms_std = None  # type: ignore[assignment]

        # Issue #141 RO-4 / RO-6: rel は保留中 (abs 集中) で、rel の正規化 (RO-5 / RO-10) は別の形になるので併用しない
        if normalization and self.use_relative_action:
            raise ValueError(
                "normalization=True is for abs action only (rel は Issue #141 で保留、RO-5 / RO-10 参照)"
            )
        self.state_normalizer = Normalizer(state.state_dim) if normalization else None
        self.action_normalizer = Normalizer(action_dim) if normalization else None
        self.state_clip = float(state_clip) if normalization and state_clip is not None else None
        # Issue #141 Phase 6: state の dropout は隠した値を 0 = 正規化後の平均にする
        if getattr(state, "dropout_groups", ()) and not normalization:
            raise ValueError("state の dropout には model の正規化 (normalization=True) が要る (隠した 0 を平均として使う)")

        # Issue #141 RO-16: FK の量 (27 次元) を教師の std で割るための統計 (FK の loss を使うときだけ)
        if self.l4_enabled:
            if self.use_relative_action:
                raise ValueError(
                    "l4_enabled=True is for abs action only (rel の FK の loss は rel を再開するときに作り直す)"
                )
            if not hasattr(action_expert, "compute_loss_with_prediction"):
                # MeanFlow は 1-step 復元の予測を返さない。学習の最初の step で落ちる前に止める (RO-7)
                raise ValueError(
                    f"l4_enabled=True requires {type(action_expert).__name__}.compute_loss_with_prediction"
                )
            from model.ramen_ori.fk import FK_FEATURE_DIM

            self.fk_normalizer = Normalizer(FK_FEATURE_DIM)
        else:
            self.fk_normalizer = None

    def set_normalization_stats(
        self, state_mean, state_std, action_mean, action_std
    ) -> None:
        """学習の開始時に計算した統計を入れる (`normalization.compute_normalization_moments`)。"""
        if self.state_normalizer is None or self.action_normalizer is None:
            raise RuntimeError("model was built with normalization=False")
        self.state_normalizer.set_stats(state_mean, state_std)
        self.action_normalizer.set_stats(action_mean, action_std)

    def set_memory_stats(self, clip_low, clip_high, mean, std) -> None:
        """memory の切り詰めの範囲と、切り詰めた後の平均・std を入れる (学習の開始時に memory の表から計算)。"""
        if self.memory is None:
            raise RuntimeError("model was built without memory")
        self.memory.set_stats(clip_low, clip_high, mean, std)

    def set_fk_stats(self, fk_mean, fk_std) -> None:
        """FK の量 (27 次元) の教師の統計を入れる (FK の loss の単位そろえ用)。"""
        if self.fk_normalizer is None:
            raise RuntimeError("model was built with l4_enabled=False")
        self.fk_normalizer.set_stats(fk_mean, fk_std)

    def _encode(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """全 path の token を統合、Fusion 出力 (context) + attention_mask + vision_features を返す。

        vision_features は aux_head (Alt-3 I-6 Depth) から再利用する用。
        Vision backbone は frozen で重い (85M)、compute_loss 内で再 forward すると
        wasteful なので `_encode` で 1 回計算した token を pass 回す。
        """
        vis_tokens = self.vision(batch["images"])
        obb_tokens, obb_mask = self.obb(
            batch["obb_verts"],
            batch["obb_conf"],
            batch["obb_class_id"],
            batch["obb_cam_id"],
            batch["obb_valid_mask"],
        )
        state = batch["state"]
        if self.state_normalizer is not None:
            state = self.state_normalizer.normalize(state)
            if self.state_clip is not None:
                state = state.clamp(-self.state_clip, self.state_clip)
        state_token = self.state(state)
        skill_token = self.skill(batch["skill_id"])

        # 画像差分 (Δ = I_t − I_{t−1}) の token は temporal があるときだけ (Issue #141 RO-11 で外せるようにした)
        del_tokens = (
            []
            if self.temporal is None
            else [self.temporal(batch["images"], batch["images_prev"], batch["cam_id"])]
        )
        # memory の token は memory があるときだけ (Issue #141 Phase 7、state の token の次に置く)
        memory_tokens = [] if self.memory is None else [self.memory(batch["memory"])]
        tokens = torch.cat(
            [vis_tokens, *del_tokens, obb_tokens, state_token, *memory_tokens, skill_token], dim=1
        )

        B = tokens.shape[0]
        device = tokens.device
        # unified attention_mask: vision/delta/state/memory/skill は全 True、OBB は per-detection valid_mask
        vis_mask = torch.ones(B, vis_tokens.shape[1], dtype=torch.bool, device=device)
        del_mask = [torch.ones(B, t.shape[1], dtype=torch.bool, device=device) for t in del_tokens]
        state_mask = torch.ones(B, 1 + len(memory_tokens), dtype=torch.bool, device=device)
        skill_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        attention_mask = torch.cat(
            [vis_mask, *del_mask, obb_mask, state_mask, skill_mask], dim=1
        )

        # fusion variant (G-1/G-2/G-3) は (context, context_mask) を返す。
        # G-1: mask pass-through (context tokens 数 = input と同)
        # G-2/G-3: mask は全 True (context = num_queries、全 valid)
        context, context_mask = self.fusion(tokens, attention_mask)
        return context, context_mask, vis_tokens

    def compute_loss(self, batch: dict) -> torch.Tensor:
        return self.compute_loss_parts(batch)["loss"]

    def compute_loss_parts(self, batch: dict) -> dict[str, torch.Tensor]:
        """学習用の loss と、その内訳 (log 用) を返す。

        Returns:
            {"loss": 合計 (backward する), "bc": flow matching の loss,
             FK の loss が on なら "fk_left" / "fk_right" / "fk_both" (各項) と
             "w_left" / "w_right" / "w_both" (各項の重み), aux_head があれば "aux_depth"}
            loss 以外は勾配を持たない

        batch["action_is_pad"] (B, chunk) の行 (区間末尾の埋め草) は BC と FK の loss から外す (Issue #141 RO-2)。
        """
        context, attention_mask, vis_tokens = self._encode(batch)
        row_mask = ~batch["action_is_pad"]

        # flow matching の正解 (正規化 on なら正規化した action、FK の教師は元の単位のまま)
        target = batch["action"]
        if self.action_normalizer is not None:
            target = self.action_normalizer.normalize(target)

        parts: dict[str, torch.Tensor] = {}
        if self.l4_enabled:
            # BC forward の v_pred から 1-step Euler で x̂1 を復元 (BC forward 1 回を再利用)
            bc, x_1_hat = self.action_expert.compute_loss_with_prediction(
                target, context, attention_mask, row_mask=row_mask
            )
            if self.action_normalizer is not None:
                # FK は関節角 (元の単位) で計算する
                x_1_hat = self.action_normalizer.denormalize(x_1_hat)
            terms = self._fk_loss_terms(
                x_1_hat, batch["action"], batch["action_waist_teacher"], row_mask
            )
            total = bc
            bc_value = bc.detach()
            for name, term in terms.items():
                # 重みは値だけ使い、勾配は流さない。項が小さくなるほど大きくなるので上限で止める (RO-12)
                w = (self.fk_loss_ratio / len(terms)) * bc_value / term.detach().clamp_min(1e-12)
                w = w.clamp(max=self.fk_weight_max)
                total = total + w * term
                parts[f"fk_{name}"] = term.detach()
                parts[f"w_{name}"] = w
        else:
            bc = self.action_expert.compute_loss(
                target, context, attention_mask, row_mask=row_mask
            )
            total = bc

        if self.aux_head is not None and "depth_target" in batch:
            aux_loss = self._depth_aux_loss(self.aux_head(vis_tokens), batch)
            total = total + self.aux_weight * aux_loss
            parts["aux_depth"] = aux_loss.detach()

        parts["bc"] = bc.detach()
        parts["loss"] = total
        return parts

    def _fk_loss_terms(
        self,
        pred_arms_hand: torch.Tensor,
        teacher_arms_hand: torch.Tensor,
        teacher_waist: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """FK の loss の 3 項 (Issue #141 RO-16)。

        予測と教師を教師の腰と合わせて 19D にし、FK の 27 次元 (肘の位置・手先の位置・手の向き rot6d、
        左右、両手先の差) を教師の std で割ってから差を取る。
        左側 / 右側 = 肘・手先・向きの誤差 (各成分の二乗の平均) を 1/3 ずつ平均、両側 = 両手先の差の誤差。
        埋め草の行を除き、chunk の行と batch で平均する。hand (掴む・離す) は BC だけで扱う。

        Args:
            pred_arms_hand:    (B, chunk, 16) x̂1 (元の単位)
            teacher_arms_hand: (B, chunk, 16) 教師 (元の単位)
            teacher_waist:     (B, chunk, 3)  教師の腰の指令
            row_mask:          (B, chunk) True = 埋め草でない行

        Returns:
            {"left", "right", "both"}: 各 scalar
        """
        from model.ramen_ori.fk import FK_FEATURE_SLICES, assemble_action19  # lazy: fk 依存を optional に

        f_pred = self.fk_normalizer.normalize(
            self.fk.features(assemble_action19(teacher_waist, pred_arms_hand))
        )
        with torch.no_grad():
            f_teacher = self.fk_normalizer.normalize(
                self.fk.features(assemble_action19(teacher_waist, teacher_arms_hand))
            )
        se = (f_pred - f_teacher) ** 2                           # (B, chunk, 27)
        m = row_mask.to(se.dtype)
        n_rows = m.sum()

        def _err(name: str) -> torch.Tensor:
            return (se[..., FK_FEATURE_SLICES[name]].mean(dim=-1) * m).sum() / n_rows

        return {
            "left": (_err("left_elbow") + _err("left_hand") + _err("left_rot6d")) / 3,
            "right": (_err("right_elbow") + _err("right_hand") + _err("right_rot6d")) / 3,
            "both": _err("hand_diff"),
        }

    @staticmethod
    def _depth_aux_loss(depth_pred: torch.Tensor, batch: dict) -> torch.Tensor:
        """L1 depth loss、valid mask 提供時は valid pixel のみ平均。"""
        target = batch["depth_target"]
        if "depth_target_mask" in batch:
            mask = batch["depth_target_mask"]
            diff = (depth_pred - target).abs()
            # mask を broadcast、valid pixel だけの平均
            valid_count = mask.sum().clamp(min=1)
            return (diff * mask.float()).sum() / valid_count
        return F.l1_loss(depth_pred, target)

    @torch.no_grad()
    def predict_action(
        self,
        batch: dict,
        *,
        prefix: torch.Tensor | None = None,
        velocity_strength: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """1 batch → action chunk (元の単位)。

        `prefix` / `velocity_strength` は RTC (Issue #137、inference 専用)。
        両方 None (既定) なら従来と完全に同一の経路を通る。詳細は
        `action_expert.sample_action` の docstring。prefix も元の単位で渡す (正規化 on なら中で正規化)。
        """
        context, attention_mask, _ = self._encode(batch)  # vis_tokens unused at inference
        if prefix is None and velocity_strength is None:
            out = self.action_expert.sample(
                context, attention_mask, n_steps=self.sample_n_steps,
            )
        else:
            if not getattr(self.action_expert, "SUPPORTS_RTC_GUIDANCE", False):
                # MeanFlow 等の 1-step sampler は ramp を刻む余地が無い。silent に
                # 無視すると「RTC を有効にしたのに効いていない」事故になるので落とす。
                raise ValueError(
                    f"{type(self.action_expert).__name__} does not support RTC guidance "
                    "(prefix / velocity_strength). Disable rtc for this variant."
                )
            if prefix is not None and self.action_normalizer is not None:
                prefix = self.action_normalizer.normalize(prefix)
            out = self.action_expert.sample(
                context,
                attention_mask,
                n_steps=self.sample_n_steps,
                prefix=prefix,
                velocity_strength=velocity_strength,
            )
        if self.action_normalizer is not None:
            out = self.action_normalizer.denormalize(out)
        return out
