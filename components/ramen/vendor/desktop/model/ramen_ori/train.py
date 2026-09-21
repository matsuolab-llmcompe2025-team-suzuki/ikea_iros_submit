"""RAMEN-Ori BC training script (Issue #115、Phase 1 M10)。

Hydra CLI entry。DummyRamenOriDataset → DataLoader → RamenOriPolicy → AdamW →
Flow Matching L2 loss を回す、100 iter smoke でも real 数万 iter train でも
config 差し替えだけで共通運用。

# 使用例

    cd model/ramen_ori
    pixi run train
    # or CLI override
    pixi run train training.max_steps=1000 training.batch_size=8 wandb.enabled=true
    # CPU smoke (LingBot backbone は cuda 前提だが device=cpu で強制切替可)
    pixi run train device=cpu training.max_steps=2

# Vision backbone load

`lingbot_vision.load_pretrained_backbone(variant=cfg.vision.variant, device=cfg.device, dtype=cfg.vision.dtype)`
で LingBot を DL + load、RamenOriPolicy の vision_backbone / vision_embed_dim に inject。
"""

from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Issue #122: PyTorch DataLoader multi-worker sharing strategy 選択
# (詳細と実測 leak rate は docs/infra/pytorch_shm_leak.md 参照)。
#
# file_descriptor (default): kernel-level `shm_open + immediate shm_unlink` の
#   anonymous fd 経由で share、**fd close で kernel が file を自動 unlink 保証** =
#   pytorch#17499 等の refcount 追跡 bug を回避 (bug 経路を通らない = "都度消す")。
#   要 `ulimit -n 1048576` (Sakura は sakura_setup.md で設定済)。
#
# file_system: /dev/shm に名前付き file を置き torch_shm_manager が refcount 追跡。
#   pytorch#91252 #13246 #17499 の複合 bug で shm leak 発生 (実測 ~1.2 MB/batch)。
#   pin_memory + persistent_workers との組合せで trigger。緊急退避用に env で override 可。
#
# 過去 file_system に切ってた背景: `ulimit -n 1024` 時代に fd 3993 保持で crash した実測
# あり。現在 `1048576` (1024×) に上げているため file_descriptor に戻せる想定。
# fd 累積が再発したら `PYTORCH_SHM_STRATEGY=file_system` で fallback、C (safe_shm_cleanup) 併用可。
_SHM_STRATEGY = os.environ.get("PYTORCH_SHM_STRATEGY", "file_descriptor")
torch.multiprocessing.set_sharing_strategy(_SHM_STRATEGY)
print(f"[train] torch multiprocessing sharing strategy: {_SHM_STRATEGY}")

from model.ramen_ori.background_push import BackgroundCkptPusher
from model.ramen_ori.contract import build_contract
from model.ramen_ori.data import DummyRamenOriDataset
from model.ramen_ori.build import build_model
from model.ramen_ori.model import RamenOriPolicy

_BASE_CONFIG = Path(__file__).resolve().parent / "configs" / "base.yaml"


def _unknown_config_keys(cfg: DictConfig) -> list[str]:
    """base.yaml に無い設定の key を返す (Issue #141)。

    Hydra は `+key=value` や yaml で足した key をそのまま通すので、読むコードが無い key は黙って無視される
    (Phase K の handoff の `+model.l4_weight=15` など)。`data` と `_target_` を持つ block は constructor が
    知らない引数をエラーにし、base で null の block は有効にするときに中身を書くので、その中は見ない。
    """
    base = OmegaConf.to_container(OmegaConf.load(_BASE_CONFIG), resolve=False)
    tree = OmegaConf.to_container(cfg, resolve=False)
    tree.pop("data", None)

    def walk(node: dict, base_node: dict, prefix: str) -> list[str]:
        unknown = []
        for key, value in node.items():
            path = f"{prefix}{key}"
            if key not in base_node:
                unknown.append(path)
            elif isinstance(value, dict) and isinstance(base_node[key], dict) and "_target_" not in base_node[key]:
                unknown.extend(walk(value, base_node[key], f"{path}."))
        return unknown

    return walk(tree, base, "")


# ============================================================================
# Issue #122: LR schedule / EMA / val loop / precision
# ============================================================================


class WarmupCosineScheduler:
    """Linear warmup + cosine decay to end_lr (Issue #122)。

    - step < warmup: lr = base_lr * (step + 1) / warmup
    - step >= warmup: lr = end_lr + 0.5 * (base_lr - end_lr) * (1 + cos(pi * t))
      ここで t = (step - warmup) / (max_steps - warmup)

    Issue #129 Phase C (2026-08-31): **multi-group 対応 refactor**。partial-train +
    LLRD 導入で optimizer が head group + backbone LLRD groups (block ごと異なる LR)
    を持つ場合、各 group の initial LR を snapshot して同一 ratio でスケール適用する。
    - Legacy 呼出し (base_lr + end_lr): end_lr_ratio = end_lr / base_lr で ratio-based に変換
    - LLRD 呼出し (end_lr_ratio 直接指定): initial LR は各 group の param_group["lr"] を snapshot

    どちらも single-group 時は従来 behavior と完全一致 (backward compat 維持)。

    torch.optim.lr_scheduler は epoch/step 混同しがち + warmup + cosine 併用の
    off-the-shelf が torch にないため自作 (~50 行、依存無し)。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        base_lr: float | None = None,       # legacy: 従来の base LR (single group 前提)
        end_lr: float = 1e-5,               # legacy: 従来の cosine 到達 LR
        end_lr_ratio: float | None = None,  # 新: cosine 到達 LR / initial LR の比率 (multi-group 対応)
    ) -> None:
        if warmup_steps < 0 or max_steps <= warmup_steps:
            raise ValueError(
                f"invalid schedule: warmup={warmup_steps}, max={max_steps} "
                f"(need max > warmup >= 0)"
            )
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        # end_lr_ratio 未指定なら legacy 呼出し → base_lr / end_lr から derive。
        # base_lr 未指定なら optimizer の最初の group の LR を base とする (single group 前提)。
        if end_lr_ratio is not None:
            self.end_lr_ratio = float(end_lr_ratio)
        else:
            derived_base = base_lr if base_lr is not None else optimizer.param_groups[0]["lr"]
            self.end_lr_ratio = float(end_lr) / max(float(derived_base), 1e-12)

        # 各 param group の initial LR を snapshot (warmup + cosine で比例スケール)
        self._initial_lrs: list[float] = [float(pg["lr"]) for pg in optimizer.param_groups]
        # legacy 属性 (test / log 用)、head group (index 0) の base_lr を保持
        self.base_lr = float(base_lr) if base_lr is not None else self._initial_lrs[0]
        self.end_lr = float(end_lr)
        self._step = 0

    def get_ratio(self, step: int) -> float:
        """LR scale ratio ∈ [0, 1]。全 group 共通、initial LR に掛けて実 LR に。"""
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        t = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        t = min(1.0, t)
        # cosine 1.0 → end_lr_ratio に減衰
        return self.end_lr_ratio + 0.5 * (1.0 - self.end_lr_ratio) * (1 + math.cos(math.pi * t))

    def get_lr(self, step: int) -> float:
        """Legacy API: 最初の (= head) group の実 LR を返す。"""
        return self._initial_lrs[0] * self.get_ratio(step)

    def step(self) -> float:
        """全 group の LR を initial_lr * ratio(step) で更新、head group の LR を返す。"""
        ratio = self.get_ratio(self._step)
        for pg, init_lr in zip(self.optimizer.param_groups, self._initial_lrs):
            pg["lr"] = init_lr * ratio
        self._step += 1
        return self.optimizer.param_groups[0]["lr"]


class EMA:
    """Exponential Moving Average of learnable params (Issue #122)。

    Flow Matching / Diffusion 系はほぼ必須 (train weight より EMA weight の方が eval 精度良い実績)。
    Frozen backbone (LingBot) は EMA 対象外 (requires_grad=False)。

    使用:
        ema = EMA(model, decay=0.9999)
        for step:
            optim.step()
            ema.update(model)
        with ema.applied(model):  # context manager で ema weight 一時 swap
            val_loss = eval(model)
    """

    # Issue #137: `torch.compile(model)` 後の `named_parameters()` は全 key に
    # この prefix を付ける。EMA を compile 前に構築して compile 後の model で
    # update() すると key が一致せず、shadow が初期化値のまま固まる (Phase K の
    # 6 run が該当、推論が noise を出す原因になった)。key を正規化して吸収する。
    _COMPILE_PREFIX = "_orig_mod."

    @classmethod
    def _canonical(cls, name: str) -> str:
        return name[len(cls._COMPILE_PREFIX):] if name.startswith(cls._COMPILE_PREFIX) else name

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        # learnable param のみ deep copy (frozen backbone は skip)
        self.shadow: dict[str, torch.Tensor] = {
            self._canonical(name): p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self._backup: dict[str, torch.Tensor] | None = None

    def update(self, model: torch.nn.Module) -> None:
        updated: set[str] = set()
        unknown: list[str] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            key = self._canonical(name)
            if key not in self.shadow:
                unknown.append(key)
                continue
            self.shadow[key].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            updated.add(key)
        # Issue #137: key 空間の不整合 (compile/DDP wrapper 等) で shadow が初期化値のまま保存されると、
        # 推論が未学習重みで走る (Phase K の 100k step を無駄にした)。
        # Issue #141 RO-9: 1 key も一致しない場合だけでなく、一部でもずれたら止める
        # (partial-train で学習する param の範囲が変わるので、一部だけの不一致も起こり得る)。
        stale = set(self.shadow) - updated
        if unknown or stale:
            raise RuntimeError(
                f"EMA.update key mismatch: {len(updated)} of {len(self.shadow)} shadow tensors updated, "
                f"learnable params without shadow {unknown[:5]}, shadow not updated {sorted(stale)[:5]} — "
                "model.named_parameters() key space does not match the EMA shadow "
                "(Issue #137 / #141 RO-9)."
            )

    def apply_to(self, model: torch.nn.Module) -> None:
        """model の param を EMA weight に swap (backup を _backup に保持)。"""
        if self._backup is not None:
            raise RuntimeError("EMA.apply_to called twice without restore")
        params = {self._canonical(name): p for name, p in model.named_parameters()}
        # Issue #137: swap が起きないと val は raw weight を EMA と誤記録する
        # (Phase K の val/ema/* が実質 raw だった)。Issue #141 RO-9: 一部だけの swap も止める。
        # 入れ替える前に確かめる (途中で止めると raw と EMA が混ざった model が残る)。
        missing = sorted(set(self.shadow) - set(params))
        if missing:
            raise RuntimeError(
                f"EMA.apply_to matched {len(self.shadow) - len(missing)} of {len(self.shadow)} shadow tensors "
                f"(missing {missing[:5]}) — val would evaluate a mix of raw and EMA weights "
                "(Issue #137 / #141 RO-9)."
            )
        self._backup = {}
        for key, shadow in self.shadow.items():
            p = params[key]
            self._backup[key] = p.detach().clone()
            p.data.copy_(shadow)

    def restore(self, model: torch.nn.Module) -> None:
        if self._backup is None:
            raise RuntimeError("EMA.restore called without prior apply_to")
        for name, p in model.named_parameters():
            key = self._canonical(name)
            if key in self._backup:
                p.data.copy_(self._backup[key])
        self._backup = None

    class _Applied:
        def __init__(self, ema: "EMA", model: torch.nn.Module) -> None:
            self.ema = ema
            self.model = model

        def __enter__(self):
            self.ema.apply_to(self.model)
            return self.ema

        def __exit__(self, *args):
            self.ema.restore(self.model)

    def applied(self, model: torch.nn.Module) -> "EMA._Applied":
        return EMA._Applied(self, model)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        # PR #127 review HIGH #4: shadow と state の key 非対称は silent skip せず raise。
        # arch 変更 (aux_head 追加等) 後の resume で EMA 部分壊れが起きるのを事前検知。
        shadow_keys = set(self.shadow.keys())
        state_keys = set(state.keys())
        missing_in_state = shadow_keys - state_keys
        extra_in_state = state_keys - shadow_keys
        if missing_in_state or extra_in_state:
            raise RuntimeError(
                f"EMA.load_state_dict key mismatch: "
                f"in shadow but not in state ({len(missing_in_state)}): {sorted(missing_in_state)[:5]}..., "
                f"in state but not in shadow ({len(extra_in_state)}): {sorted(extra_in_state)[:5]}... "
                f"arch 変更後の resume の場合は EMA を再初期化してください (ema.enabled=false で load skip も可)"
            )
        for k, v in state.items():
            self.shadow[k].copy_(v)


def _resolve_autocast_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    lookup = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}
    if name not in lookup:
        raise ValueError(f"precision.autocast_dtype={name!r} not in {sorted(lookup)}")
    return lookup[name]


def _safe_shm_cleanup_and_recreate_loader(loader, make_loader) -> tuple[object, int, int]:
    """DataLoader workers を完全 shutdown → /dev/shm/torch_* を安全 rm → DataLoader 再構築。

    Ckpt save 直後 (main thread が torch.save 中、workers が queue-block で idle) の
    quiescent point で呼ぶ想定。この timing なら worker→main の in-flight tensor 転送が
    無いため、rm しても main が open 失敗する race window がゼロになる。

    背景: 我々の `_cleanup_orphan_shm()` (Run 3 crash 原因) や Monitor の `rm -f
    /dev/shm/torch_*` (Run 4 crash 原因) は training loop 走行中に rm するため race hit。
    本 helper は workers を _shutdown_workers() で完全に殺してから rm する。
    caller は返却された new_loader を再代入して古い loader ref を解放する必要がある。

    Returns (new_loader, unlinked_count, freed_bytes)。
    _shutdown_workers() 失敗時は fail-safe に旧 loader を返し (0, 0) で cleanup skip。
    """
    import glob

    # 1. Workers 完全 shutdown (persistent_workers=True でも kill)
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        try:
            iterator._shutdown_workers()
        except Exception as e:
            # PR #127 review HIGH #1: shutdown 失敗を silent 握り潰すと stale workers 残存
            # のまま unlink → 新 loader 起動で Run 3/4 と同型 race crash 再発しうる。
            # WARN 出して旧 loader を返し cleanup+respawn を skip する fail-safe に。
            print(
                f"[safe_shm_cleanup] WARN: _shutdown_workers() failed "
                f"({type(e).__name__}: {e}), skip cleanup+respawn to avoid race"
            )
            return loader, 0, 0
    gc.collect()
    time.sleep(3)  # OS-level worker exit + shm mmap release 待ち (実測 workers 500ms 前後、余裕見て 3sec)

    # 2. Workers shutdown 済みで rm (race window ゼロ)
    freed = 0
    unlinked = 0
    for path in glob.glob("/dev/shm/torch_*"):
        try:
            freed += os.path.getsize(path)
            os.unlink(path)
            unlinked += 1
        except FileNotFoundError:
            continue

    # 3. Fresh workers で DataLoader respawn
    new_loader = make_loader()
    return new_loader, unlinked, freed


def _cleanup_orphan_shm() -> tuple[int, int]:
    """/dev/shm/torch_* の中で「どの process からも mmap 参照されていない」 file を unlink する。

    Returns (unlinked_count, freed_bytes)。

    背景 (Issue #122 Run 2 実測): torch DataLoader (num_workers=8, prefetch=2,
    persistent_workers=True) の long-running で /dev/shm に torch_XXX file が
    数千個蓄積 (真の orphan、どの python process の /proc/PID/maps にも fd にも
    出てこない = shm leak)。gc.collect() では unlink 発火せず、file_system
    sharing strategy でも回収漏れが発生。Sakura H100 で 7h で 56GB 消費、
    118GB cap 突破リスク。この helper で /proc/*/maps を走査し、参照ゼロの
    orphan だけ os.unlink で明示回収する (POSIX unlink 挙動: mmap 参照が
    残っていれば file 内容は消えず、参照ゼロの場合のみ inode 解放)。
    """
    import glob
    import os

    shm_files = set(glob.glob("/dev/shm/torch_*"))
    if not shm_files:
        return 0, 0

    referenced: set[str] = set()
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        maps_path = f"/proc/{pid_str}/maps"
        try:
            with open(maps_path) as f:
                for line in f:
                    # maps 行末が pathname (絶対 path)、"/dev/shm/torch_" で始まるものだけ拾う
                    idx = line.rfind("/dev/shm/torch_")
                    if idx >= 0:
                        referenced.add(line[idx:].rstrip())
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue

    orphans = shm_files - referenced
    freed_bytes = 0
    unlinked = 0
    for path in orphans:
        try:
            size = os.path.getsize(path)
            os.unlink(path)
            freed_bytes += size
            unlinked += 1
        except (FileNotFoundError, PermissionError):
            continue
    return unlinked, freed_bytes


@torch.no_grad()
def _run_val_loop(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str,
    max_batches: int,
    autocast_dtype: torch.dtype | None,
    fk: torch.nn.Module,
    ee_error_rows: int,
) -> tuple[float, dict[str, float]]:
    """Val split で avg loss 計算 + loss の内訳 + 補助 metric (V1: EE error mm、V2: motion energy)。

    Issue #129 Phase H (2026-08-31): metric-only、training loss / gradient に影響なし。

    Args:
        model: RamenOriPolicy (compute_loss_parts / predict_action 保有)
        val_loader / device / max_batches / autocast_dtype: 既存 val loop 通り
        fk: V1 用の G1WristFKTorch (FK の loss が off の run でも手首の誤差を比べられるように別に渡す)
        ee_error_rows: V1 で使う先頭の行数 (推論の execution_steps)

    Returns:
        (avg_loss, extra_metrics):
          avg_loss: val loss
          extra_metrics: dict of val metric name → mean value across batches
            - "bc" / "fk_left" / "fk_right" / "fk_both" / "w_*" (loss の内訳、埋め草の行を除く)
            - "ee_error_exec_mm/left_mm/right_mm/avg_mm" (V1、先頭 ee_error_rows 行の埋め草でない行)
            - "motion/pred_dq_L/R/LR_dq/ptp_L/R/LR_asym" (V2、pred action から)
            - "motion/teacher_dq_L/R/LR_dq/ptp_L/R/LR_asym" (V2、teacher action から)
    """
    from model.ramen_ori.state_derive import STATE71_ARMS_SLICE
    from model.ramen_ori.val_metrics import compute_ee_error_mm, compute_motion_energy

    was_training = model.training
    model.eval()
    total_loss = 0.0
    n_batches = 0
    # Extra metric accumulators
    extra_sums: dict[str, float] = {}
    extra_counts: dict[str, int] = {}

    use_relative = bool(getattr(model, "use_relative_action", False))

    def _accumulate(prefix: str, metric_dict: dict[str, float]) -> None:
        for k, v in metric_dict.items():
            key = f"{prefix}/{k}" if prefix else k
            extra_sums[key] = extra_sums.get(key, 0.0) + float(v)
            extra_counts[key] = extra_counts.get(key, 0) + 1

    for b, batch in enumerate(val_loader):
        if b >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        if autocast_dtype is not None:
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                parts = model.compute_loss_parts(batch)
        else:
            parts = model.compute_loss_parts(batch)
        total_loss += float(parts["loss"].item())
        n_batches += 1
        _accumulate("", {k: v.item() for k, v in parts.items() if k != "loss"})

        # V2: motion energy (teacher は必ず計算、pred も predict_action で計算)
        _accumulate("motion/teacher", compute_motion_energy(batch["action"]))
        pred_action = model.predict_action(batch)   # (B, chunk, 16)、元の単位
        _accumulate("motion/pred", compute_motion_energy(pred_action))

        # V1: EE error mm (推論で実行する先頭の行、埋め草を除く)
        arms_current = batch["state"][:, STATE71_ARMS_SLICE] if use_relative else None
        ee = compute_ee_error_mm(
            pred_action=pred_action,
            teacher_action=batch["action"],
            teacher_waist=batch["action_waist_teacher"],
            fk=fk,
            arms_current=arms_current,
            rel_mean=getattr(model, "_relative_arms_mean", None),
            rel_std=getattr(model, "_relative_arms_std", None),
            use_relative_action=use_relative,
            row_mask=~batch["action_is_pad"],
            n_rows=ee_error_rows,
        )
        _accumulate("ee_error_exec_mm", ee)

    if was_training:
        model.train()

    extra_metrics = {k: extra_sums[k] / max(1, extra_counts[k]) for k in extra_sums}
    return total_loss / max(1, n_batches), extra_metrics


def _build_model(cfg: DictConfig, device: str) -> RamenOriPolicy:
    """`ckpt["cfg"]` から model を組み立てる (実体は `model/ramen_ori/build.py`)。

    Issue #141 P8-1: 推論も同じ関数で組み立てる (推論側に別の組み立てを書かない)。
    train.py 経由の呼び出しはそのまま残す (学習の script を変えないため)。
    """
    return build_model(cfg, device)


def _set_training_stats(
    model: RamenOriPolicy, dataset, std_min: float, num_workers: int
) -> None:
    """全 frame から skill ごとに同じ重みの統計を計算して model に入れる (Issue #141 RO-4 / RO-6 / RO-16)。

    - 正規化 (model.state_normalizer がある): state / 指令の平均・std
    - FK の loss (model.fk_normalizer がある): 教師の指令を FK に通した 27 次元の平均・std
    dataset の `state_action_item` (学習と同じ変換、動画は読まない) を 1 回だけ通す。val の frame も含める
    (統計は平均・std だけで、LeRobot / GR00T の dataset 統計と同じ扱い)。
    起動時に skill ごとの正規化後の std を中身ごとに log する。action で 0.5 未満の次元は WARN
    (その skill の動きが loss で 1/4 以下に軽く数えられる。loss の重みを考え直す目安)。
    state と FK は表示だけ (state は入力で縮尺の違いは最初の層が吸収する、FK は単位そろえ用)。
    """
    from model.ramen_ori.fk import FK_FEATURE_SLICES, G1WristFKTorch
    from model.ramen_ori.normalization import (
        compute_normalization_moments,
        normalized_std_by_skill,
    )
    from model.ramen_ori.skill_mapping import skill_id_name
    from model.ramen_ori.state_derive import (
        ACTION16_ARMS_SLICE,
        ACTION16_HAND_SLICE,
        STATE71_GROUPS,
    )

    need_norm = model.state_normalizer is not None
    need_fk = model.fk_normalizer is not None
    t0 = time.time()
    # stats の pass は CPU の batch なので、FK も CPU で別に作る (model.fk は GPU にある)
    moments = compute_normalization_moments(
        dataset,
        fk=G1WristFKTorch.from_default_urdf() if need_fk else None,
        num_workers=num_workers,
    )
    stats = {name: m.finalize(std_min) for name, m in moments.items()}
    if need_norm:
        model.set_normalization_stats(*stats["state"], *stats["action"])
    if need_fk:
        model.set_fk_stats(*stats["fk"])

    counts = ", ".join(
        f"{skill_id_name(sid)} {n}" for sid, n in sorted(moments["action"].counts.items())
    )
    print(f"[norm] stats from {len(dataset)} frames in {time.time() - t0:.0f}s ({counts})")
    groups = {
        "state": STATE71_GROUPS,
        "action": {"arms": ACTION16_ARMS_SLICE, "hand": ACTION16_HAND_SLICE},
        "fk": FK_FEATURE_SLICES,
    }
    names = (["state", "action"] if need_norm else []) + (["fk"] if need_fk else [])
    for name in names:
        std = stats[name][1]
        floored = np.flatnonzero(std <= std_min).tolist()
        print(
            f"[norm] {name} std (min/median/max): "
            + ", ".join(
                f"{g} {std[sl].min():.3f}/{np.median(std[sl]):.3f}/{std[sl].max():.3f}"
                for g, sl in groups[name].items()
            )
            + f" | dims at std_min={std_min:g}: {floored}"
        )
        for sid, nstd in sorted(normalized_std_by_skill(moments[name], std).items()):
            parts = [
                f"{g} {nstd[sl].min():.2f}/{np.median(nstd[sl]):.2f}/{nstd[sl].max():.2f}"
                for g, sl in groups[name].items()
            ]
            low = np.flatnonzero(nstd < 0.5).tolist() if name == "action" else []
            print(
                f"[norm] {name} normalized std (min/median/max) skill {skill_id_name(sid)}: "
                + ", ".join(parts)
                + (f" | WARN < 0.5 dims {low}" if low else "")
            )


def _set_memory_stats(model: RamenOriPolicy, dataset, std_min: float) -> None:
    """memory の表から、切り詰めの範囲 (p1 / p99) と切り詰めた後の平均・std を計算して model に入れる (Issue #141 Phase 7)。

    どちらも skill ごとに同じ重み (percentile は skill ごとに同じ数の frame を引く、平均・std は SkillBalancedMoments)。
    val の frame も含める (ほかの統計と同じ)。
    """
    from model.ramen_ori.memory_features import MEMORY_LAYOUT, MULTI_CLASSES, SINGLE_CLASSES
    from model.ramen_ori.normalization import SkillBalancedMoments, skill_balanced_percentiles
    from model.ramen_ori.skill_mapping import skill_id_name

    table, skill_ids = dataset.memory_arrays()
    low, high = skill_balanced_percentiles(table, skill_ids, q=(1.0, 99.0))
    moments = SkillBalancedMoments(table.shape[1])
    moments.update(np.clip(table, low, high), skill_ids)
    mean, std = moments.finalize(std_min)
    model.set_memory_stats(low, high, mean, std)

    names = list(MEMORY_LAYOUT)
    print(f"[memory] stats from {len(table)} frames (skill ごとに同じ重み)")
    for name in ("table_top.dangle", "table_top.dcx", "hand_left.dcx", "hand_right.dcx", "motion.tau1.left.x"):
        i = names.index(name)
        print(f"[memory] {name}: p1 {low[i]:+.4f} / p99 {high[i]:+.4f} / 切り詰めた後の std {std[i]:.4f}")
    for sid in np.unique(skill_ids):
        rows = table[skill_ids == sid]
        visible = ", ".join(f"{c} {rows[:, names.index(f'{c}.visible')].mean():.2f}" for c in SINGLE_CLASSES)
        counts = ", ".join(f"{c} {rows[:, names.index(f'{c}.count')].mean():.2f}" for c in MULTI_CLASSES)
        print(f"[memory] skill {skill_id_name(int(sid))}: 見えている割合 {visible} | 平均の個数 {counts}")


def _check_vision_trainability(model: RamenOriPolicy) -> None:
    """vision backbone の学習する param の数を log し、freeze と partial-train が合わない設定を止める (Issue #141)。

    - freeze=True なのに学習する param がある (partial-train on): forward が no_grad なので勾配が流れない
    - freeze=False なのに学習する param が無い (partial-train off): LingBot の loader が凍結して返すので
      学習されない。Phase K はこの設定で「全層を学習している」と誤解されていた
    """
    vision = model.vision
    learnable = sum(p.numel() for p in vision.backbone.parameters() if p.requires_grad)
    print(f"[model] vision backbone learnable {learnable / 1e6:.1f}M (freeze={vision.freeze})")
    if vision.freeze and learnable > 0:
        raise ValueError(
            "model.vision.freeze=true だが backbone に学習する param がある (optim.partial_train on?)。"
            "freeze=true の backbone は no_grad で動くので勾配が流れない。partial-train には freeze=false"
        )
    if not vision.freeze and learnable == 0:
        raise ValueError(
            "model.vision.freeze=false だが backbone に学習する param が無い (LingBot の loader は凍結して返す)。"
            "凍結するなら freeze=true、一部解放するなら optim.partial_train.enabled=true"
        )


def _compile_submodules(model: RamenOriPolicy, mode: str) -> list[str]:
    """重い部分をその場で compile する (Issue #141)。compile した部分の名前を返す。

    model 全体を `torch.compile(model)` で包むと、compile されるのは forward だけで、学習が呼ぶ
    `compute_loss*` は compile されない (Phase K)。さらに state_dict の key に `_orig_mod.` が付く。
    `nn.Module.compile()` は module をその場で compile し、key を変えない。
    loss の計算 (正規化・埋め草の mask・重み) は Python の分岐を含むので compile の外に残す。
    FK は loss が `features()` (forward 以外) を通るので、共通の `forward_detailed` を compile する。
    """
    names = []
    for name in ("vision", "temporal", "fusion", "action_expert"):
        module = getattr(model, name)
        if module is not None:
            module.compile(mode=mode)
            names.append(name)
    if model.fk is not None:
        model.fk.forward_detailed = torch.compile(model.fk.forward_detailed, mode=mode)
        names.append("fk.forward_detailed")
    return names


def _move_batch(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    device = cfg.device
    torch.manual_seed(cfg.training.seed)

    print("=" * 60)
    print("RAMEN-Ori Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    unknown_keys = _unknown_config_keys(cfg)
    if unknown_keys:
        raise ValueError(
            f"base.yaml に無い設定の key {unknown_keys}。読むコードが無く黙って無視されるので止める "
            "(綴りと、名前の変わった key を確かめる。新しい設定なら base.yaml に既定値を足す)"
        )

    # Data — Issue #122: augmentation_cfg を Dataset に注入 (DummyRamenOriDataset は ignore)
    aug_cfg_dict = None
    if cfg.get("augmentation") is not None:
        aug_cfg_dict = OmegaConf.to_container(cfg.augmentation, resolve=True)
    dataset = hydra.utils.instantiate(cfg.data, augmentation_cfg=aug_cfg_dict)

    # Issue #141 束 2-14: 学習と推論の約束 (contract.py)。すべての ckpt と wandb の config に入れる。
    # dummy の data (smoke) は実際の画像・state・skill を持たないので作らない
    contract = None if isinstance(dataset, DummyRamenOriDataset) else build_contract(cfg, dataset)
    if contract is not None:
        print("[contract]\n" + yaml.safe_dump(contract, allow_unicode=True, sort_keys=False))

    # Issue #122: episode 単位 val/test split (leakage 回避)。val.enabled=true で分離、
    # train_view を DataLoader source に、val_view は val_loader で eval に使用。
    val_cfg = cfg.get("val", None)
    val_loader = None
    train_source: torch.utils.data.Dataset = dataset
    train_indices: np.ndarray | None = None  # sampler weight を train subset に絞る用
    if val_cfg is not None and val_cfg.get("enabled", False):
        if not hasattr(dataset, "make_train_val_test_split"):
            raise RuntimeError(
                "val.enabled=true but dataset does not support make_train_val_test_split "
                "(dummy dataset は非対応)。real_task5_7 config を使うか val.enabled=false に。"
            )
        # Issue #122: val.split_json 指定時は external JSON 読む (GR00T と strict identical)
        split_json_cfg = val_cfg.get("split_json")
        split_json_path = str(split_json_cfg) if split_json_cfg else None
        train_view, val_view, test_view = dataset.make_train_val_test_split(
            val_ratio=val_cfg.get("val_ratio", 0.1),
            test_ratio=val_cfg.get("test_ratio", 0.1),
            seed=val_cfg.get("seed", 42),
            split_json=split_json_path,
        )
        train_source = train_view
        train_indices = np.array(train_view.indices, dtype=np.int64)
        split_source_msg = (
            f"split_json={split_json_path}" if split_json_path else f"seed={val_cfg.get('seed', 42)}"
        )
        print(
            f"[data] split enabled: train={len(train_view)} / val={len(val_view)} / "
            f"test={len(test_view)} (episode 単位、{split_source_msg})"
        )
        val_batch_size = val_cfg.get("val_batch_size") or cfg.training.batch_size
        val_loader = DataLoader(
            val_view,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            drop_last=False,
        )

    # Sampler selection priority:
    #   1. curriculum (H-3、Alt-5) — stage 制 active skill gating + active 内 uniform
    #   2. balancing (A-5) — 全 task を task_uniform で 1:1 balance
    #   3. なし → shuffle=True (DataLoader default)
    # + Alt-6 AWR: いずれの経路でも advantage weight を multiplier として注入可能。
    # Issue #122: val enabled 時、train_indices で subset に絞る。
    curriculum_cfg = cfg.training.get("curriculum", None)
    balancing_cfg = cfg.training.get("balancing", None)
    awr_cfg = cfg.training.get("awr", None)
    sampler = None
    # train loop 内で step を書き込む共有 holder (curriculum sampler が per-epoch 参照)
    step_holder = {"v": 0}

    def _subset(arr: np.ndarray) -> np.ndarray:
        return arr if train_indices is None else arr[train_indices]

    # Alt-6 H-7 AWR: enabled 時 per-frame advantage weight を計算 (mean=1 normalize 済)
    frame_adv: np.ndarray | None = None
    if awr_cfg is not None and awr_cfg.get("enabled", False):
        from model.ramen_ori.awr import broadcast_to_frames, compute_duration_advantage

        ep_meta = dataset.sample_episode_metadata()
        ep_adv = compute_duration_advantage(
            ep_meta["episode_lengths"],
            method=awr_cfg.get("method", "inverse"),
            temperature=awr_cfg.get("temperature", 1.0),
        )
        frame_adv = broadcast_to_frames(ep_adv, ep_meta["episode_lengths"])
        frame_adv = _subset(frame_adv)
        print(
            f"[data] AWR enabled: method={awr_cfg.get('method', 'inverse')}, "
            f"ep weight range [{ep_adv.min():.3f}, {ep_adv.max():.3f}] (mean=1)"
        )

    if curriculum_cfg is not None and curriculum_cfg.get("enabled", False):
        from model.ramen_ori.curriculum import CurriculumSampler, CurriculumStage

        # Issue #141 RO-1: model の入力と同じ skill_id (episode meta の source_task_index 由来)
        skill_ids = _subset(dataset.sample_skill_ids())
        stages = [
            CurriculumStage(step=int(s["step"]), skills=list(s["skills"]))
            for s in curriculum_cfg["stages"]
        ]
        sampler = CurriculumSampler(
            skill_ids_per_sample=skill_ids,
            stages=stages,
            step_ref=lambda: step_holder["v"],
            num_samples_per_epoch=len(train_source),
            advantage_weights=frame_adv,  # Alt-6: None (無効時) or per-frame adv
        )
        print(
            f"[data] curriculum enabled: {len(stages)} stages, "
            f"stage 0 active skills = {sorted(stages[0].skills)}"
        )
    elif balancing_cfg is not None and balancing_cfg.get("enabled", False):
        from torch.utils.data import WeightedRandomSampler
        from model.ramen_ori.data_lerobot import compute_task_uniform_weights

        strategy = balancing_cfg.get("strategy", "task_uniform")
        if strategy != "task_uniform":
            raise NotImplementedError(f"balancing.strategy={strategy!r} not supported yet")
        skill_ids = _subset(dataset.sample_skill_ids())
        weights = compute_task_uniform_weights(skill_ids)
        # Alt-6: AWR advantage を balancing weight に乗算 (mean=1 なので合計不変)
        if frame_adv is not None:
            weights = weights * frame_adv
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(train_source),
            replacement=True,
        )
        print(
            f"[data] balancing: task_uniform, skills {sorted(set(int(t) for t in skill_ids))}, "
            f"weight range [{weights.min():.2e}, {weights.max():.2e}]"
            + (", + AWR" if frame_adv is not None else "")
        )
    elif frame_adv is not None:
        # AWR only (curriculum も balancing も無し) — advantage weight だけの WeightedRandomSampler
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(frame_adv),
            num_samples=len(train_source),
            replacement=True,
        )
        print("[data] AWR-only sampler (curriculum/balancing disabled)")

    # Issue #122: dataloader throughput 高速化 knobs (design 影響なし)。
    # - prefetch_factor (default 2): worker が pre-fetch する batch 数、大きい方が GPU 待たない
    # - persistent_workers (default False): epoch 跨ぎで worker 再起動しない、restart cost 節約
    # - pin_memory (default False): host RAM を page-locked にし non-blocking H→D copy 可能、
    #   実測 H100 上で ~5-10% throughput 改善
    # config 未指定なら hardcode の高性能 default を使用。
    _prefetch_factor = cfg.training.get("prefetch_factor", 4) if cfg.training.num_workers > 0 else None
    _persistent_workers = (
        cfg.training.get("persistent_workers", True) if cfg.training.num_workers > 0 else False
    )
    _pin_memory = cfg.training.get("pin_memory", True)

    # Issue #122: DataLoader を関数化 (safe_shm_cleanup_on_ckpt で再構築するため)。
    def _make_loader() -> DataLoader:
        return DataLoader(
            train_source,
            batch_size=cfg.training.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=cfg.training.num_workers,
            drop_last=True,
            prefetch_factor=_prefetch_factor,
            persistent_workers=_persistent_workers,
            pin_memory=_pin_memory,
        )

    loader = _make_loader()
    print(
        f"[data] train source size: {len(train_source)}, batch_size: {cfg.training.batch_size}, "
        f"num_workers: {cfg.training.num_workers}, prefetch: {_prefetch_factor}, "
        f"persistent: {_persistent_workers}, pin_memory: {_pin_memory}"
    )

    # Model
    model = _build_model(cfg, device)
    # Issue #141 RO-4 / RO-6 / RO-16: 正規化と FK の loss の統計。resume 時は ckpt の buffer を使う (計算し直さない)
    if (model.state_normalizer is not None or model.fk_normalizer is not None) and not cfg.training.get(
        "resume_from"
    ):
        _set_training_stats(
            model, dataset, float(cfg.model.normalization.std_min), cfg.training.num_workers
        )
    # Issue #141 RO-11: 画像差分 (model.temporal) と前 frame の読み込み (data.load_prev_image) を揃える。
    # 差分なしで前 frame を読むと無駄な読み込み、差分ありで読まないと model が images_prev を求めて落ちる
    load_prev = dataset.load_prev_image
    if (model.temporal is not None) != load_prev:
        raise ValueError(
            f"model.temporal={'set' if model.temporal is not None else 'null'} と "
            f"data.load_prev_image={load_prev} が合わない (画像差分を使うときだけ前 frame を読む)"
        )
    # Issue #141 Phase 7: memory の token と data の memory の表を揃える (dummy の data は memory を持たない)
    data_memory = bool(getattr(dataset, "memory", False))
    if (model.memory is not None) != data_memory:
        raise ValueError(
            f"model.memory={'set' if model.memory is not None else 'null'} と data.memory={data_memory} が合わない"
        )
    if model.memory is not None and not cfg.training.get("resume_from"):
        _set_memory_stats(model, dataset, float(cfg.model.normalization.std_min))
    # val の手首の誤差用の FK (FK の loss が off の run でも比べられるように、無ければ別に作る)
    from model.ramen_ori.fk import G1WristFKTorch

    val_fk = model.fk if model.fk is not None else G1WristFKTorch.from_default_urdf().to(device)

    # Issue #129 Phase C (2026-08-31): partial-train + LLRD optimizer builder。
    # cfg.optim.partial_train.enabled=false なら従来通り single group AdamW、
    # true なら last N blocks unfreeze + block ごと LLRD LR + head group の複数 group。
    from model.ramen_ori.optim_utils import make_optimizer

    optim = make_optimizer(model, cfg.optim)

    # partial_train は model.vision.backbone の requires_grad を書き換えるため、
    # 学習可能パラメータ数の集計は make_optimizer 後に実施する必要がある。
    n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[model] learnable {n_learn / 1e6:.1f}M / frozen {n_frozen / 1e6:.1f}M")
    _check_vision_trainability(model)
    if len(optim.param_groups) > 1:
        # partial_train enabled: head + backbone LLRD groups の内訳を log
        for pg in optim.param_groups:
            n_params = sum(p.numel() for p in pg["params"])
            print(f"  [param_group] name={pg.get('name','?')} lr={pg['lr']:.2e} params={n_params/1e6:.2f}M")

    # Issue #122: LR schedule (warmup + cosine)
    # Issue #129 Phase C: multi-group scheduler にも対応 (end_lr_ratio で group ごと ratio 保持)
    lr_sched_cfg = cfg.get("lr_schedule", None)
    lr_scheduler = None
    if lr_sched_cfg is not None and lr_sched_cfg.get("enabled", False):
        lr_scheduler = WarmupCosineScheduler(
            optim,
            base_lr=cfg.optim.lr,
            warmup_steps=lr_sched_cfg.get("warmup_steps", 2000),
            max_steps=cfg.training.max_steps,
            end_lr=lr_sched_cfg.get("cosine_end_lr", 1e-5),
        )
        print(
            f"[optim] LR schedule: warmup {lr_scheduler.warmup_steps} + cosine to "
            f"{lr_scheduler.end_lr:.2e} (base_lr={lr_scheduler.base_lr:.2e}, "
            f"end_lr_ratio={lr_scheduler.end_lr_ratio:.4f}, {len(optim.param_groups)} groups)"
        )

    # Issue #122: EMA (learnable のみ shadow、val は EMA weight で eval)
    ema_cfg = cfg.get("ema", None)
    ema = None
    if ema_cfg is not None and ema_cfg.get("enabled", False):
        ema = EMA(model, decay=ema_cfg.get("decay", 0.9999))
        print(f"[ema] EMA enabled: decay={ema.decay}, {len(ema.shadow)} shadow tensors")
    ema_update_every = ema_cfg.get("update_every", 1) if ema_cfg else 1
    eval_use_ema = ema_cfg.get("eval_use_ema", True) if ema_cfg else False

    # Issue #122: precision (autocast、null=fp32、"bfloat16" で bf16 autocast)
    precision_cfg = cfg.get("precision", None)
    autocast_dtype = _resolve_autocast_dtype(
        precision_cfg.get("autocast_dtype") if precision_cfg else None
    )
    if autocast_dtype is not None:
        print(f"[precision] autocast: {autocast_dtype}")

    # Issue #129 Phase I-0-7 (2026-09-01): 精度影響ゼロ or 実務的に無視できる範囲の高速化 opt。
    # OOM や速度低下時に config で off に戻せる (default false = backward compat)。
    speedup_cfg = cfg.get("speedup", None)
    _torch_compile_enabled = bool(speedup_cfg.get("torch_compile", False)) if speedup_cfg else False
    if _torch_compile_enabled:
        _compile_mode = speedup_cfg.get("torch_compile_mode", "default")
        compiled = _compile_submodules(model, _compile_mode)
        print(f"[speedup] compile (mode={_compile_mode!r}): {compiled} — 初回の batch で compile (数分)")

    val_every = val_cfg.get("val_every", 1000) if val_cfg else 0
    val_max_batches = val_cfg.get("val_max_batches", 20) if val_cfg else 0
    val_ee_error_rows = val_cfg.get("ee_error_rows", 8) if val_cfg else 8

    # Wandb (opt-in)
    wandb_run = None
    if cfg.wandb.enabled:
        import wandb

        wandb_init_kwargs = dict(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            config={**OmegaConf.to_container(cfg, resolve=True), "contract": contract},
        )
        # Issue #122: crash 復旧時は既存 run に append (id + resume="allow")。
        # log step は train loop の step (=start_step 以降) をそのまま流すため、
        # start_step > 既存 run 最終 step なら plot 上は連続する。
        wandb_resume_id = cfg.wandb.get("resume_run_id", None)
        if wandb_resume_id:
            wandb_init_kwargs["id"] = wandb_resume_id
            wandb_init_kwargs["resume"] = "allow"
            print(f"[wandb] resuming existing run id={wandb_resume_id}")
        wandb_run = wandb.init(**wandb_init_kwargs)

    # Checkpoint dir
    ckpt_dir = Path(cfg.training.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Issue #122: ckpt を HF に push する (opt-in、crash 時の resume 用。/nvme は Sakura の停止で消える)。
    # Issue #141: push は別 process で流し、学習は待たない (background_push.py)。同期で push すると 10k step ごとに
    # 約 300 s 学習が止まっていた
    hf_autopush_cfg = cfg.training.get("hf_autopush", None)
    hf_pusher = None
    if hf_autopush_cfg is not None and hf_autopush_cfg.get("enabled", False):
        if hf_autopush_cfg.get("repo_id", None) is None:
            print("[hf_autopush] WARN: enabled=true だが repo_id が未設定、skip")
        else:
            hf_pusher = BackgroundCkptPusher(
                repo_id=hf_autopush_cfg.repo_id, private=hf_autopush_cfg.get("private", True)
            )

    # Issue #122: Resume from ckpt (crash 復旧用)。
    # ckpt には model / optim (Adam m/v) / ema shadow / lr_scheduler._step / step が入っており、
    # ここで load すれば cosine LR phase + Adam momentum + EMA が bit-level 連続復元される。
    # DataLoader batch shuffle 順序は resume 前後で異なるが、Adam inertia で吸収される。
    start_step = 0
    resume_from = cfg.training.get("resume_from", None)
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"training.resume_from={resume_from} not found")
        print(f"[resume] loading ckpt from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optim_state_dict"])
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if lr_scheduler is not None and "lr_scheduler_step" in ckpt:
            lr_scheduler._step = ckpt["lr_scheduler_step"]
        start_step = ckpt["step"]
        print(
            f"[resume] resumed at step={start_step}, "
            f"lr_scheduler._step={ckpt.get('lr_scheduler_step')}, "
            f"ema={'loaded' if ema is not None and 'ema_state_dict' in ckpt else 'skip'}"
        )
        del ckpt  # free CPU memory after load

    # Training loop (max_steps 到達で止まる無限 iterator パターン)
    model.train()
    t0 = time.time()

    # Issue #122: shm leak 対策 — worker 越しの shared tensor が長期実行で /dev/shm に累積
    # (Run 1 が 15.5h で 48% 消費 → allocation 失敗 crash 実測)。
    # gc.collect() で dead ref 掃除 → SharedMemory.close() → shm file unlink を誘発。
    # cfg.training.get() を loop 内 hot path で呼ぶと OmegaConf の overhead が乗るので事前に解決。
    # Issue #141: 勾配の全体の norm を clip する (null = clip しない)。c32 の 30k step で、入口の層 (state.fc1) に
    # 勾配が集中して学習が崩れた。学習する param は partial_train で変わるので optimizer を作った後に集める
    _grad_clip_norm = cfg.optim.grad_clip_norm
    _trainable_params = [p for p in model.parameters() if p.requires_grad]
    _gc_every = cfg.training.get("gc_every", 0)
    _shm_cleanup_every = cfg.training.get("shm_cleanup_every", 0)
    _shm_cleanup_on_ckpt = cfg.training.get("shm_cleanup_on_ckpt", False)
    _safe_shm_cleanup_on_ckpt = cfg.training.get("safe_shm_cleanup_on_ckpt", False)

    # Issue #122: 明示 iter() で書くと ckpt save 時に loader を再構築できる
    # (safe_shm_cleanup_on_ckpt 対応)。従来の enumerate(_infinite_batches(loader)) は
    # loader が generator 内で参照ハンドルを握り mid-loop 差替えができなかった。
    loader_iter = iter(loader)
    step = start_step - 1
    while True:
        step += 1
        if step >= cfg.training.max_steps:
            break
        try:
            batch = next(loader_iter)
        except StopIteration:
            # dataset 1 epoch 完了 (通常 batch 数 >> max_steps なので稀)、次 epoch へ
            loader_iter = iter(loader)
            batch = next(loader_iter)
        # Curriculum sampler が次 epoch の active skills 決定に読む (H-3、Alt-5)
        step_holder["v"] = step

        # Issue #122: LR schedule は step 前に反映 (0 step 目は warmup lr で開始)
        current_lr = lr_scheduler.step() if lr_scheduler is not None else cfg.optim.lr

        batch = _move_batch(batch, device)
        # SDPA の backend は PyTorch の自動選択 (Flash だけに絞る指定は Issue #141 で削除: この model では
        # fp32 だと LingBot の attention、bf16 だと fusion の attention に使える Flash の kernel が無く、最初の step で止まる)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                parts = model.compute_loss_parts(batch)
        else:
            parts = model.compute_loss_parts(batch)
        loss = parts["loss"]
        optim.zero_grad()
        loss.backward()
        if _grad_clip_norm is not None:
            # 返り値は clip する前の全体の norm (log 用、tensor のまま持って log の step だけ CPU に読む)
            grad_norm = torch.nn.utils.clip_grad_norm_(_trainable_params, _grad_clip_norm)
        optim.step()

        # Issue #122: EMA update after optim.step()
        if ema is not None and (step + 1) % ema_update_every == 0:
            ema.update(model)

        if step % cfg.training.log_every == 0:
            dt = time.time() - t0
            # Issue #122: resume 時は start_step baseline で throughput 計算 (resume 直後の
            # elapsed=数秒で step 95000+ を割ると it/s が数千 と inflated 表示される問題への fix)
            it_per_sec = (step - start_step + 1) / max(dt, 1e-9)
            # Issue #122: fd count monitor (file_descriptor sharing strategy で fd leak
            # 早期検出用、docs/infra/pytorch_shm_leak.md 参照)。ulimit -n 1048576 に対する
            # 使用率を wandb に流す。fd 取得は数 ms なので log_every 頻度で許容。
            try:
                fd_count = len(os.listdir(f"/proc/{os.getpid()}/fd"))
            except OSError:
                fd_count = -1
            # Issue #141 RO-7: loss の内訳 (BC、FK の 3 項、3 つの重み) も出す。gnorm は clip する前の勾配の norm
            part_values = {k: v.item() for k, v in parts.items() if k != "loss"}
            if _grad_clip_norm is not None:
                part_values["gnorm"] = grad_norm.item()
            print(
                f"[step {step:5d}] loss={loss.item():.4f} "
                + " ".join(f"{k}={v:.4g}" for k, v in part_values.items())
                + f" lr={current_lr:.2e} it/s={it_per_sec:.2f} elapsed={dt:.1f}s fd={fd_count}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss.item(),
                        **{f"train/{k}": v for k, v in part_values.items()},
                        "train/lr": current_lr,
                        "train/it_per_sec": it_per_sec,
                        "system/fd_count": fd_count,
                    },
                    step=step,
                )

        # Issue #122: 定期 gc で DataLoader worker 由来の shm leak を抑制
        # (Run 1 で 15.5h の間に /dev/shm 48% 消費 → shm allocation 失敗で crash 実測)。
        # gc.collect() が dead SharedMemory ref を掃除 → __del__ で shm file unlink 発火。
        if _gc_every > 0 and step > start_step and step % _gc_every == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Issue #122: gc.collect() では拾えない torch の shm leak (Run 2 実測で
        # /dev/shm/torch_* が 4251 file / 55GB 蓄積、fd も mmap も参照ゼロの完全 orphan)
        # を /proc/*/maps 走査で判定し明示 os.unlink する。gc の後段に置いて相乗効果。
        if _shm_cleanup_every > 0 and step > start_step and step % _shm_cleanup_every == 0:
            unlinked, freed_bytes = _cleanup_orphan_shm()
            if unlinked > 0:
                print(
                    f"[shm_cleanup] step={step} unlinked {unlinked} orphan files, "
                    f"freed {freed_bytes / 1024 / 1024 / 1024:.2f} GB"
                )

        # Issue #122: val eval (val_every step ごと、EMA weight で計算 if enabled)
        if (
            val_loader is not None
            and val_every > 0
            and step > 0
            and step % val_every == 0
        ):
            if ema is not None and eval_use_ema:
                with ema.applied(model):
                    val_loss, extra_metrics = _run_val_loop(
                        model, val_loader, device, val_max_batches, autocast_dtype,
                        val_fk, val_ee_error_rows,
                    )
                tag = "val/loss_ema"
            else:
                val_loss, extra_metrics = _run_val_loop(
                    model, val_loader, device, val_max_batches, autocast_dtype,
                    val_fk, val_ee_error_rows,
                )
                tag = "val/loss"
            print(f"[val] step={step} {tag}={val_loss:.4f} (over {val_max_batches} batches)")
            # Issue #129 Phase H (2026-08-31): 補助 metric (V1: EE error mm、V2: motion energy)。
            # val loss と別 key で log、5-run 型 val_loss ≠ real gap の判断材料。
            if extra_metrics:
                print(
                    "[val] extra: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in sorted(extra_metrics.items()))
                )
            if wandb_run is not None:
                log_dict = {tag: val_loss}
                # EMA 有無の tag prefix (val/ema/... or val/...) に合わせる
                metric_prefix = "val/ema" if tag == "val/loss_ema" else "val"
                for mk, mv in extra_metrics.items():
                    log_dict[f"{metric_prefix}/{mk}"] = mv
                wandb_run.log(log_dict, step=step)

        if (step + 1) % cfg.training.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"ckpt_step_{step + 1:06d}.pt"
            # Issue #137: torch.compile 後の state_dict は全 key に "_orig_mod."
            # prefix が付く。保存側で落としておくことで、下流 (inference の
            # `RamenOriPolicy.from_ckpt` 等) が compile 有無を意識せずに済む。
            ckpt_state = {
                "step": step + 1,
                "model_state_dict": {
                    EMA._canonical(k): v for k, v in model.state_dict().items()
                },
                "optim_state_dict": optim.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "contract": contract,
            }
            if ema is not None:
                ckpt_state["ema_state_dict"] = ema.state_dict()
            if lr_scheduler is not None:
                ckpt_state["lr_scheduler_step"] = lr_scheduler._step
            torch.save(ckpt_state, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

            # Issue #122: ckpt save 直後の shm cleanup (torch.save が model+optim
            # を dup してメモリ圧迫する直後、shm 側の orphan もまとめて回収)。
            if _shm_cleanup_on_ckpt:
                unlinked, freed_bytes = _cleanup_orphan_shm()
                if unlinked > 0:
                    print(
                        f"[shm_cleanup] post-ckpt step={step + 1} unlinked {unlinked} "
                        f"orphan files, freed {freed_bytes / 1024 / 1024 / 1024:.2f} GB"
                    )

            # Issue #122: ckpt save 直後の safe shm cleanup (workers 完全 shutdown → rm →
            # respawn の race-free 版、Run 4 crash pattern の根本 fix)。
            # workers を shutdown してから rm するため race window ゼロ。ただし workers
            # respawn に ~3-5sec (LingBot は主 process 側に load 済なので worker 側は軽量)。
            if _safe_shm_cleanup_on_ckpt:
                loader, unlinked, freed_bytes = _safe_shm_cleanup_and_recreate_loader(loader, _make_loader)
                loader_iter = iter(loader)
                print(
                    f"[safe_shm_cleanup] post-ckpt step={step + 1} unlinked {unlinked} "
                    f"files, freed {freed_bytes / 1024 / 1024 / 1024:.2f} GB, loader recreated"
                )

            # Issue #122: HF autopush (every_n_ckpt 個目の ckpt ごと)。Issue #141: 別 process で流し、待たない
            if hf_pusher is not None:
                ckpt_count = (step + 1) // cfg.training.ckpt_every
                if ckpt_count % hf_autopush_cfg.get("every_n_ckpt", 2) == 0:
                    hf_pusher.submit(ckpt_path, step=step + 1)

    # Issue #141: 最後の ckpt の push が終わるまで待つ (run の終わり = 全 ckpt が HF にある、を保つ)
    if hf_pusher is not None:
        hf_pusher.wait()
    total_time = time.time() - t0
    print(f"[done] {cfg.training.max_steps} steps in {total_time:.1f}s")

    resume_log_path = Path(wandb_run.dir) / "output.log" if wandb_run is not None else None
    if wandb_run is not None:
        wandb_run.finish()

    # Issue #122: auto-merge on resume — crash → resume で wandb 側 "current step" が
    # buffer 残骸で resume 時 start_step より進んでいると、resume の log は
    # step 単位で全 reject される (Run 1 で実測: 95001-95999 の 1k step 分空白)。
    # local output.log は完全に残っているので、[done] 後に crash-time log + resume log
    # を merge した新 wandb run を自動作成する (元 run は残る、新 run は連続 curve)。
    if (
        cfg.wandb.enabled
        and cfg.wandb.get("auto_merge_source_log", None)
        and resume_from is not None
        and resume_log_path is not None
    ):
        from model.ramen_ori.scripts.merge_wandb_runs import merge_from_local_logs

        source_log = Path(cfg.wandb.auto_merge_source_log)
        if not source_log.exists():
            print(
                f"[merge_wandb] WARN: auto_merge_source_log={source_log} not found, "
                f"skip merge"
            )
        else:
            merged_name = (
                f"{cfg.wandb.run_name}_merged" if cfg.wandb.run_name else "resumed_merged"
            )
            try:
                merge_from_local_logs(
                    source_log=source_log,
                    resume_log=resume_log_path,
                    start_step=start_step,
                    entity=cfg.wandb.entity,
                    project=cfg.wandb.project,
                    merged_run_name=merged_name,
                    source_run_id=cfg.wandb.get("resume_run_id", None),
                )
            except Exception as e:
                print(f"[merge_wandb] WARN: auto-merge failed ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
