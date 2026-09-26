"""Val 時の補助 metric (Issue #129 Phase H、2026-08-31)。

5-run で observed した **val_ema ≠ real success rate** の gap を縮める判断材料。
training loss / gradient に一切影響しない (no_grad で計算、log のみ)。

# V1: EE position error (mm 換算)

- Predict action chunk (`model.predict_action(batch)`) を復元 arms + hand として取り、
  teacher waist と concat → 19D → URDF FK → 両手 wrist (tool point) 3D 位置
- 推論で実際に実行する最初の n_rows 行 (execution_steps、既定 8) のうち、区間末尾の埋め草でない行で
  pred EE vs teacher EE の L2 距離の平均を mm で log (Issue #141 RO-2 / RO-17)。
  旧定義 (chunk の最後の行、埋め草を含む) は chunk 長で時刻が変わり、chunk 16 と 32 を比べられなかった
- Relative action space の場合は `reconstruct_arms_abs_from_dq_norm` で復元してから FK

# V2: Motion energy metrics

- pred / teacher chunk それぞれについて per-tick |Δq| mean、per-dim ptp、L/R 比を計算
- **"Stops moving" 症状の val 側早期検知** (Session 32 の action data 分析で発見の
  rotate awr ptp=0.17 が実機論外だった、val でこれが見えれば実機 dispatch 前に判断可)

# 使い方 (train.py val loop 内から呼ぶ想定)

    from model.ramen_ori.val_metrics import compute_ee_error_mm, compute_motion_energy

    with torch.no_grad():
        pred_action = model.predict_action(batch)   # (B, chunk, 16)
    ee = compute_ee_error_mm(
        pred_action=pred_action,
        teacher_action=batch["action"],
        teacher_waist=batch["action_waist_teacher"],
        arms_current=batch["state"][:, STATE71_ARMS_SLICE],   # relative 時
        fk=fk,
        row_mask=~batch["action_is_pad"],
        n_rows=8,
        use_relative_action=model.use_relative_action,
        rel_mean=model._relative_arms_mean,
        rel_std=model._relative_arms_std,
    )
    motion_pred = compute_motion_energy(pred_action)
    motion_teacher = compute_motion_energy(batch["action"])

Log keys (wandb):
    val/ee_error_exec_mm/left_mm  val/ee_error_exec_mm/right_mm  val/ee_error_exec_mm/avg_mm
    val/motion/pred_dq_L  val/motion/pred_dq_R  val/motion/pred_LR_dq
    val/motion/pred_ptp_L val/motion/pred_ptp_R val/motion/pred_LR_asym
    val/motion/teacher_dq_L ... teacher_LR_asym
"""

from __future__ import annotations

import torch


def _reconstruct_arms_hand_abs(
    action_chunk_16: torch.Tensor,
    arms_current: torch.Tensor | None,
    rel_mean: torch.Tensor | None,
    rel_std: torch.Tensor | None,
    use_relative_action: bool,
) -> torch.Tensor:
    """(B, chunk, 16) action を absolute (arms + hand) に復元。

    - use_relative_action=False: そのまま return
    - use_relative_action=True: arms 14 の normalized Δq を absolute q に cumsum 復元、
      hand 2 は absolute pass-through、concat して return
    """
    if not use_relative_action:
        return action_chunk_16
    if arms_current is None or rel_mean is None or rel_std is None:
        raise ValueError(
            "relative action reconstruct requires arms_current, rel_mean, rel_std"
        )
    from model.ramen_ori.relative_action import reconstruct_arms_abs_from_dq_norm

    arms_abs = reconstruct_arms_abs_from_dq_norm(
        action_chunk_16[..., :14], arms_current, rel_mean, rel_std
    )   # (B, chunk, 14)
    hand_abs = action_chunk_16[..., 14:16]
    return torch.cat([arms_abs, hand_abs], dim=-1)


def compute_ee_error_mm(
    pred_action: torch.Tensor,
    teacher_action: torch.Tensor,
    teacher_waist: torch.Tensor,
    fk: torch.nn.Module,
    arms_current: torch.Tensor | None = None,
    rel_mean: torch.Tensor | None = None,
    rel_std: torch.Tensor | None = None,
    use_relative_action: bool = False,
    row_mask: torch.Tensor | None = None,
    n_rows: int | None = None,
) -> dict[str, float]:
    """最初の n_rows 行のうち埋め草でない行で、pred EE と teacher EE の L2 距離の平均を mm で返す。

    V1 metric (Phase H): 物理距離で "手が届いてるか" を直接測る。5-run の val_loss ≠ real
    問題を緩和 (loss は per-frame MSE で 3D 位置に translate されない)。

    Args:
        pred_action:    (B, chunk_len, 16) predict_action output (absolute or relative)
        teacher_action: (B, chunk_len, 16) teacher (data から)
        teacher_waist:  (B, chunk_len, 3)  teacher waist target
        fk:             G1WristFKTorch instance
        arms_current, rel_mean, rel_std, use_relative_action: relative 時に必要
        row_mask:       (B, chunk_len) True = 埋め草でない行 (None なら全行)
        n_rows:         先頭から何行を使うか (推論の execution_steps、None なら全行)

    Returns:
        {"left_mm": float, "right_mm": float, "avg_mm": float}
        (val loop で val_max_batches に対して mean を取る想定)
    """
    from model.ramen_ori.fk import assemble_action19  # lazy

    pred_abs = _reconstruct_arms_hand_abs(
        pred_action, arms_current, rel_mean, rel_std, use_relative_action
    )
    teacher_abs = _reconstruct_arms_hand_abs(
        teacher_action, arms_current, rel_mean, rel_std, use_relative_action
    )

    rows = slice(0, n_rows)
    pred_q19 = assemble_action19(teacher_waist[:, rows], pred_abs[:, rows])       # (B, n, 19)
    teacher_q19 = assemble_action19(teacher_waist[:, rows], teacher_abs[:, rows])
    pred_left, pred_right = fk(pred_q19)                                          # (B, n, 3)
    teacher_left, teacher_right = fk(teacher_q19)
    if row_mask is None:
        m = torch.ones(pred_left.shape[:-1], dtype=pred_left.dtype, device=pred_left.device)
    else:
        m = row_mask[:, rows].to(pred_left.dtype)

    def _mm(pred: torch.Tensor, teacher: torch.Tensor) -> float:
        # L2 距離 (m) → mm (× 1000)、有効な行の平均
        return float(((pred - teacher).norm(dim=-1) * m).sum().item() / m.sum().item() * 1000.0)

    err_left_mm = _mm(pred_left, teacher_left)
    err_right_mm = _mm(pred_right, teacher_right)
    return {
        "left_mm": err_left_mm,
        "right_mm": err_right_mm,
        "avg_mm": (err_left_mm + err_right_mm) / 2.0,
    }


def compute_motion_energy(action_chunk: torch.Tensor) -> dict[str, float]:
    """Per-tick |Δq| mean、per-dim ptp、L/R 比を返す (V2 metric、Phase H)。

    "stops moving" 症状の val 側早期検知。5-run では実測 rotate awr ptp=0.17 が
    real 論外だった → val でこれが見えれば実機 dispatch 前に判断できる。

    Args:
        action_chunk: (B, chunk_len, 16) action tensor (absolute or normalized Δq、
                     どちらでも per-tick 変化として意味を持つ)

    Returns:
        {"dq_L", "dq_R", "LR_dq", "ptp_L", "ptp_R", "LR_asym"} — float。
        arms 14 layout: [0:7]=left_arm、[7:14]=right_arm、[14:16]=hand は除外。
    """
    if action_chunk.dim() != 3 or action_chunk.shape[-1] != 16:
        raise ValueError(f"action_chunk must be (B, chunk, 16), got {tuple(action_chunk.shape)}")

    # per-tick diff (step k+1 - step k)、shape (B, chunk-1, 16)
    if action_chunk.shape[1] < 2:
        # chunk_len=1 なら Δq 定義できず 0 返す
        return {"dq_L": 0.0, "dq_R": 0.0, "LR_dq": 1.0, "ptp_L": 0.0, "ptp_R": 0.0, "LR_asym": 0.0}

    diff = action_chunk[:, 1:, :] - action_chunk[:, :-1, :]   # (B, chunk-1, 16)
    abs_diff = diff.abs()
    # arms slice: [0:7]=left, [7:14]=right, [14:16]=hand (除外)
    dq_L = abs_diff[..., 0:7].mean().item()
    dq_R = abs_diff[..., 7:14].mean().item()
    LR_dq = dq_L / max(dq_R, 1e-12)

    # ptp per-dim (max - min over chunk)、shape (B, 16)
    ptp = action_chunk.max(dim=1).values - action_chunk.min(dim=1).values
    ptp_L = ptp[..., 0:7].mean().item()
    ptp_R = ptp[..., 7:14].mean().item()
    # LR asym: per-tick |ΔL - ΔR| mean
    dL = diff[..., 0:7]
    dR = diff[..., 7:14]
    LR_asym = (dL - dR).abs().mean().item()

    return {
        "dq_L": float(dq_L),
        "dq_R": float(dq_R),
        "LR_dq": float(LR_dq),
        "ptp_L": float(ptp_L),
        "ptp_R": float(ptp_R),
        "LR_asym": float(LR_asym),
    }
