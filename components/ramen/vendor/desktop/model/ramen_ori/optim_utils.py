"""RAMEN-Ori Phase C (Issue #129、2026-08-31): partial-train + LLRD optimizer builder。

Vision backbone (LingBot ViT-B/16 or RADIO v2.5-B) の **last N blocks** を unfreeze
して、block ごとに layer-wise learning rate decay (LLRD) を掛ける partial-train を
実現する。head/fusion/action_expert 側は通常 LR で全 param 学習。

# なぜ partial-train

Vision backbone を完全 frozen だと overlay (Axis C の C-11 描画 hint) や coord token
の指す "task-specific attention pattern" が backbone に染み込まない。逆に全 unfreeze
だと RADIO agglomerative teacher (SAM + DINOv2 + CLIP + SigLIP の distilled) が
catastrophic forgetting しかねない。妥協案として **last 4/12 blocks + LLRD 0.75 +
backbone LR = head LR × 0.2** を採用 (OpenVLA §5.2、MAE / BEiT v2、GR00T N1.6 の
慣行の合成、Session 32 案 A)。

# 使い方

    from model.ramen_ori.optim_utils import make_optimizer

    optim = make_optimizer(model, cfg.optim)
    # cfg.optim.partial_train.enabled=false: 従来通り single group AdamW
    # cfg.optim.partial_train.enabled=true : head group + backbone LLRD groups の合成

WarmupCosineScheduler は per-group の initial LR を snapshot する形に refactor 済
(train.py 側)、multi-group でも各 group の LR ratio を保ったまま warmup + cosine。

# Backbone アクセス path (実測 verified 2026-08-31)

  LingBot ViT-B/16:  backbone.blocks             (ModuleList, len=12)
  RADIO v2.5-B (RadioAdapter wrap):
                     backbone.radio.model.blocks (Sequential,  len=12)

新 backbone variant を追加する場合は `find_backbone_blocks` に path を追加する。

# Muon (Issue #141 6 本目、`optim.name: muon`)

中間の 2D の重みを `SplitMuon`、それ以外 (入口・出口の層、Embedding、1D) を AdamW で更新する (`MuonWithAdamW`)。
lr・weight decay は AdamW と共通 (Moonshot の match_rms_adamw で更新の大きさを AdamW に合わせる)。
まとめて持っている行列 (MHA の q/k/v、DiT の adaLN の 9 個) は分けて直交化する。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def find_backbone_blocks(backbone: nn.Module) -> nn.Module:
    """Vision backbone の transformer blocks (ModuleList / Sequential) を返す。

    LingBot / RADIO 対応、その他は明示 raise (実装時に path 追加要)。
    """
    # LingBot ViT-B/16 (lingbot_vision.LingBotVisionTransformer)
    if hasattr(backbone, "blocks"):
        return backbone.blocks
    # RADIO v2.5-B (RadioAdapter で wrap 済、raw model は self.radio.model)
    if hasattr(backbone, "radio") and hasattr(backbone.radio, "model"):
        model = backbone.radio.model
        if hasattr(model, "blocks"):
            return model.blocks
    raise ValueError(
        f"cannot locate .blocks in backbone of type {type(backbone).__name__}. "
        f"Add access path in optim_utils.find_backbone_blocks if new backbone variant."
    )


def apply_partial_train_freeze(
    backbone: nn.Module,
    last_n_blocks: int,
) -> None:
    """Backbone 全 param を freeze してから last N blocks のみ unfreeze。

    - Non-block (patch_embed / pos_embed / patch_generator / norm / etc.) は freeze 継続
    - Lower blocks (0..total-last_n-1) は freeze 継続
    - Last N blocks は requires_grad=True

    In-place operation、backbone param の requires_grad を書き換える。
    冪等 (何度呼んでも同じ状態に収束)。
    """
    blocks = find_backbone_blocks(backbone)
    total = len(blocks)
    if last_n_blocks < 0 or last_n_blocks > total:
        raise ValueError(
            f"last_n_blocks={last_n_blocks} out of range [0, {total}] (backbone has {total} blocks)"
        )

    # 1. 全 backbone param を freeze
    for p in backbone.parameters():
        p.requires_grad = False

    # 2. Last N blocks の param を unfreeze
    unfreeze_start = total - last_n_blocks
    for i in range(unfreeze_start, total):
        for p in blocks[i].parameters():
            p.requires_grad = True


def make_partial_train_backbone_groups(
    backbone: nn.Module,
    last_n_blocks: int,
    base_lr: float,
    llrd_decay: float,
) -> list[dict[str, Any]]:
    """Last N blocks の param を LLRD 適用済 param_group リストで返す。

    LLRD (layer-wise learning rate decay) 慣行 (MAE / BEiT v2 §4.1):
      Last block               → base_lr
      Last-1 block             → base_lr * llrd_decay
      Last-2 block             → base_lr * llrd_decay^2
      ...
      block[total - last_n]    → base_lr * llrd_decay^(last_n - 1)

    Non-block param (patch_embed / norm etc.) は本 groups に含まれない
    (apply_partial_train_freeze で frozen 済のため optimizer 対象外)。

    Args:
        backbone: apply_partial_train_freeze 済 (unfrozen blocks あり) を想定
        last_n_blocks: 4 (design doc §Axis A 決定、last 4/12)
        base_lr: backbone base LR (= head_lr * backbone_lr_ratio、Session 32 = 0.2)
        llrd_decay: 0.75 (MAE / BEiT v2 ViT-B 標準)

    Returns:
        [{"params": [...], "lr": float, "name": "backbone.block_i"}, ...] を last N 個
    """
    blocks = find_backbone_blocks(backbone)
    total = len(blocks)
    groups: list[dict[str, Any]] = []
    for i in range(total - last_n_blocks, total):
        dist_from_last = (total - 1) - i
        lr = base_lr * (llrd_decay ** dist_from_last)
        block_params = [p for p in blocks[i].parameters() if p.requires_grad]
        if not block_params:
            # 万一 apply_partial_train_freeze が未実行 or 該当 block が動的に空 → skip
            continue
        groups.append(
            {"params": block_params, "lr": float(lr), "name": f"backbone.block_{i}"}
        )
    return groups


NS_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
NS_IMPLS = ("baddbmm", "matmul")


def _orthogonalize(
    update: torch.Tensor,
    split: int,
    ns_steps: int,
    eps: float,
    dtype: torch.dtype = torch.bfloat16,
    impl: str = "baddbmm",
) -> torch.Tensor:
    """(rows, cols) の更新を行方向に split 個の (rows/split, cols) に分け、それぞれを Newton–Schulz で直交化する。

    既定 (bf16、baddbmm) で split=1 なら torch.optim.Muon (torch 2.10) の `_zeropower_via_newtonschulz` と同じ計算 (係数も同じ)。
    Issue #141: c32_muon が 2 回とも、この辺りで GPU の kernel の異常 (Xid 13 / 43) を起こして落ちたので、別の計算の経路を選べるようにした。
    `impl="matmul"` は転置の view を行列積に渡さず (毎回連続した memory に写す)、baddbmm も使わない。`dtype=float32` は bf16 の行列積を使わない
    """
    a, b, c = 3.4445, -4.775, 2.0315
    x = update.to(dtype).unflatten(0, (split, -1))  # (split, rows/split, cols)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    if impl == "matmul":
        x = x.contiguous()
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(ns_steps):
        if impl == "baddbmm":
            gram = x @ x.mT
            x = torch.baddbmm(x, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
        else:
            gram = torch.matmul(x, x.mT.contiguous())
            x = a * x + torch.matmul(b * gram + c * torch.matmul(gram, gram), x)
    if transposed:
        x = x.mT
    return x.flatten(0, 1)


class SplitMuon(torch.optim.Optimizer):
    """Muon の、まとめて持っている行列を分けて直交化できる版 (Issue #141)。

    更新は torch.optim.Muon (torch 2.10、`adjust_lr_fn="match_rms_adamw"`) と同じ: momentum は lerp、nesterov、
    Newton–Schulz 5 回 (bf16)、更新の大きさは lr × 0.2√max(行, 列) (AdamW の更新の RMS に合わせる)、weight decay は分離型。
    違いは param group の `split` だけで、(rows, cols) の重みを行方向に split 個の行列に分けて別々に直交化する
    (MHA の in_proj は q/k/v の 3 個、DiT の adaLN は 9 個)。torch の Muon には分ける仕組みが無い。

    `ns_dtype` / `ns_impl` は直交化の計算の経路 (`_orthogonalize`)。param group ではなく optimizer に持たせる
    (再開の load_state_dict は保存した group の値で上書きするので、group に置くと設定を変えた再開で黙って元に戻る)。
    """

    def __init__(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        split: int = 1,
        eps: float = 1e-7,
        ns_dtype: str = "bfloat16",
        ns_impl: str = "baddbmm",
    ) -> None:
        if ns_dtype not in NS_DTYPES or ns_impl not in NS_IMPLS:
            raise ValueError(
                f"SplitMuon の ns_dtype は {list(NS_DTYPES)}、ns_impl は {list(NS_IMPLS)} のどれか: {ns_dtype!r} / {ns_impl!r}"
            )
        self.ns_dtype, self.ns_impl = ns_dtype, ns_impl
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, split=split, eps=eps,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2 or p.shape[0] % group["split"] != 0:
                    raise ValueError(
                        f"SplitMuon は行数が split={group['split']} で割り切れる 2D の重みだけ: {tuple(p.shape)}"
                    )

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        for group in self.param_groups:
            momentum, split = group["momentum"], group["split"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - momentum)
                update = p.grad.lerp(buf, momentum) if group["nesterov"] else buf
                update = _orthogonalize(
                    update, split, group["ns_steps"], group["eps"], NS_DTYPES[self.ns_dtype], self.ns_impl
                )
                scale = 0.2 * math.sqrt(max(p.shape[0] // split, p.shape[1]))
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"] * scale)


class MuonWithAdamW:
    """SplitMuon と AdamW を 1 つの optimizer として扱う (train.py と WarmupCosineScheduler が使う形)。

    `param_groups` は両方の group を並べた list (scheduler が書き換える dict は元の optimizer のもの)。
    load_state_dict で torch が group の dict を差し替えるので、毎回元の optimizer から取り直す。
    state_dict は {"muon": ..., "adamw": ...}。
    """

    def __init__(self, muon: SplitMuon, adamw: torch.optim.AdamW) -> None:
        self.muon, self.adamw = muon, adamw

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


def make_muon_optimizer(
    model: nn.Module, lr: float, betas: tuple[float, float], weight_decay: float, muon_cfg: Any
) -> MuonWithAdamW:
    """学習する param を Muon (中間の 2D の重み) と AdamW (それ以外) に分ける。

    AdamW に回すもの: 2D でない param、nn.Embedding の重み、`muon_cfg.adamw_modules` の module の param
    (生の入力を受ける入口の層と、出口の層。Muon の作者の指針)。Muon の重みは `muon_cfg.split` の
    [param 名の末尾, 個数] に当たれば、その個数に分けて直交化する。どちらの一覧も、当たる param が無ければ止める (綴りの誤り)。
    """
    adamw_prefixes = [f"{m}." for m in muon_cfg.adamw_modules]
    split_rules = [(str(suffix), int(n)) for suffix, n in muon_cfg.split]
    embedding_ids = {
        id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()
    }
    muon_params: dict[int, list[torch.nn.Parameter]] = {}
    adamw_params: list[torch.nn.Parameter] = []
    used: set[str] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        prefix = next((pre for pre in adamw_prefixes if name.startswith(pre)), None)
        if prefix is not None or p.ndim != 2 or id(p) in embedding_ids:
            used.add(prefix)
            adamw_params.append(p)
            continue
        rule = next(((s, n) for s, n in split_rules if name.endswith(s)), None)
        if rule is not None:
            used.add(rule[0])
        muon_params.setdefault(1 if rule is None else rule[1], []).append(p)
    unused = [pre[:-1] for pre in adamw_prefixes if pre not in used] + [
        s for s, _ in split_rules if s not in used
    ]
    if unused:
        raise ValueError(f"optim.muon の adamw_modules / split に、当たる param が無いものがある: {unused}")
    muon = SplitMuon(
        [{"params": ps, "split": n, "name": f"muon_split{n}"} for n, ps in sorted(muon_params.items())],
        lr=lr,
        weight_decay=weight_decay,
        momentum=float(muon_cfg.momentum),
        nesterov=bool(muon_cfg.nesterov),
        ns_steps=int(muon_cfg.ns_steps),
        ns_dtype=str(muon_cfg.ns_dtype),
        ns_impl=str(muon_cfg.ns_impl),
    )
    adamw = torch.optim.AdamW(
        [{"params": adamw_params, "name": "adamw"}], lr=lr, betas=betas, weight_decay=weight_decay
    )
    return MuonWithAdamW(muon, adamw)


def make_optimizer(
    model: nn.Module,
    cfg_optim: Any,
) -> torch.optim.AdamW | MuonWithAdamW:
    """統一 optimizer factory (partial-train / legacy 両対応、Issue #129 Phase C)。

    Args:
        model: RamenOriPolicy (model.vision.backbone が RADIO or LingBot)
        cfg_optim: base.yaml の optim section。
            name: adamw | muon (Issue #141、muon なら make_muon_optimizer)
            partial_train:
                enabled: bool = false
                last_n_blocks: int = 4
                llrd_decay: float = 0.75
                backbone_lr_ratio: float = 0.2   # backbone LR = optim.lr * this

    Returns:
        AdamW instance (name=muon なら MuonWithAdamW)。
        - partial_train.enabled=false: 従来通り single group (frozen 以外の全 trainable)
        - partial_train.enabled=true:  head group + backbone LLRD groups の複数 group

    Side effect (partial_train.enabled=true 時):
        model.vision.backbone の requires_grad を書き換える (last N blocks のみ True)。
    """
    head_lr = float(cfg_optim.get("lr", 1e-4))
    betas_raw = cfg_optim.get("betas", (0.9, 0.95))
    betas = (float(betas_raw[0]), float(betas_raw[1]))
    weight_decay = float(cfg_optim.get("weight_decay", 1e-2))

    pt_cfg = cfg_optim.get("partial_train", None)
    partial = pt_cfg is not None and bool(pt_cfg.get("enabled", False))
    if cfg_optim.get("name", "adamw") == "muon":
        if partial:
            raise ValueError("optim.name=muon と optim.partial_train は組み合わせていない (backbone の LLRD は AdamW だけ)")
        return make_muon_optimizer(model, head_lr, betas, weight_decay, cfg_optim.muon)
    if not partial:
        # Legacy: single group AdamW (5-run 継承 behavior)
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=head_lr, betas=betas, weight_decay=weight_decay
        )

    # Partial-train enabled
    last_n = int(pt_cfg.get("last_n_blocks", 4))
    llrd_decay = float(pt_cfg.get("llrd_decay", 0.75))
    backbone_lr_ratio = float(pt_cfg.get("backbone_lr_ratio", 0.2))
    backbone_base_lr = head_lr * backbone_lr_ratio

    if not hasattr(model, "vision") or not hasattr(model.vision, "backbone"):
        raise ValueError(
            "model.vision.backbone not found — partial_train は RAMEN-Ori (VisionEncoder wrap) 前提"
        )
    backbone = model.vision.backbone

    apply_partial_train_freeze(backbone, last_n_blocks=last_n)
    backbone_groups = make_partial_train_backbone_groups(
        backbone,
        last_n_blocks=last_n,
        base_lr=backbone_base_lr,
        llrd_decay=llrd_decay,
    )

    # head = backbone 以外の全 trainable。id で set 引きして重複除外。
    backbone_param_ids: set[int] = set()
    for g in backbone_groups:
        for p in g["params"]:
            backbone_param_ids.add(id(p))
    head_params = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_param_ids
    ]

    param_groups: list[dict[str, Any]] = [
        {"params": head_params, "lr": head_lr, "name": "head"},
        *backbone_groups,
    ]
    return torch.optim.AdamW(
        param_groups, lr=head_lr, betas=betas, weight_decay=weight_decay
    )
