"""Install / verify LeRobot patch for GR00T rotate specialist training (Issue #129)。

# Scope

Issue #129 の 4 feature を GR00T policy に注入する統合 patch:

| Env var                            | 制御 feature                                                       | default |
|------------------------------------|--------------------------------------------------------------------|---------|
| GROOT_LEFT_HAND_LOSS_WEIGHT        | T0-1: 左手 loss weight up (2.0 で enable) — Phase 1 実装           | 1.0     |
| GROOT_PER_CAM_SCALAR_ENABLE        | T0-2: per-camera 学習 scalar — Phase 2 SKIP (GR00T arch mismatch)  | false   |
| GROOT_BILATERAL_LOSS_WEIGHT        | L1: bilateral coordination loss (0.5 で enable) — Phase 3 実装      | 0.0     |
| GROOT_FK_ANCHOR_LOSS_WEIGHT        | L4: FK anchor loss (0.3 で enable) — Phase 4 実装                  | 0.0     |

# Phase 1/3/4: loss-side patch 実装済 (T0-1 + L1 + L4 combined)

- Target: `lerobot/policies/groot/groot_n1_7.py`
- SHA256 pin: `eb6ebb45...` = LeRobot 0.6.0/0.6.1 で **byte-identical** 確認済 (2026-08-30)
- Anchor: `GR00TN17ActionHead.forward` の loss 計算 2 行 (unique)
- Combined replacement で 3 feature を runtime env で on/off:
  - **T0-1**: `GROOT_LEFT_HAND_LOSS_WEIGHT`、`LEFT_HAND_INDICES = (3,4,5,6,7,8,9,17)` に weight
  - **L1**: `GROOT_BILATERAL_LOSS_WEIGHT`、`(pred_L - pred_R) - (vel_L - vel_R)` の MSE を aux term
    に加算 (velocity 空間で計算、左右 error 非独立性を penalize)
  - **L4**: `GROOT_FK_ANCHOR_LOSS_WEIGHT`、velocity から 1-step Euler で `actions_pred =
    noisy_trajectory + (1-t) * pred_actions` を復元 → G1 URDF FK で両 wrist 3D 位置
    → GT との L2 距離を aux term として加算 (「机に届かせる」を task space で直接叩く)

**Runtime env で on/off** = install 1 回で「T0-1 だけ」「L1 だけ」「T0-1+L1+L4 全部」の Run
組合せに対応、再 patch 不要。全 feature default (1.0/0.0) 時は base code と数値的完全等価。

# L4 の追加 dependency

L4 は `inference.desktop.perception.g1_urdf_fk_torch.G1WristFKTorch` を import する。
Sakura 起動時に PYTHONPATH に iros_2026_ramen root を通す必要 (Phase 5 で train_lerobot.sh
に反映)。patch_processor 実行時、L4 enabled ならこの import を先行 verify する。

# Pattern

既存 `patch_lerobot_groot_relative_eef.py` と同じ SHA256-guarded surgical patch。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path


LEROBOT_VERSION = "0.6.0"
PATCH_MARKER = "# TEAM_RAMEN_GR00T_ROTATE_SPECIALIST_LEFT_HAND_V1"


# Env var 名 (patched LeRobot code からも参照される)
ENV_LEFT_HAND_LOSS_WEIGHT = "GROOT_LEFT_HAND_LOSS_WEIGHT"
ENV_PER_CAM_SCALAR_ENABLE = "GROOT_PER_CAM_SCALAR_ENABLE"
ENV_BILATERAL_LOSS_WEIGHT = "GROOT_BILATERAL_LOSS_WEIGHT"
ENV_FK_ANCHOR_LOSS_WEIGHT = "GROOT_FK_ANCHOR_LOSS_WEIGHT"

DEFAULT_LEFT_HAND_LOSS_WEIGHT = 1.0
DEFAULT_PER_CAM_SCALAR_ENABLE = False
DEFAULT_BILATERAL_LOSS_WEIGHT = 0.0
DEFAULT_FK_ANCHOR_LOSS_WEIGHT = 0.0


# ============================================================================
# Loss-side patch (target = groot_n1_7.py)
# ============================================================================
# Combined patch replacement 内で env var branch:
#   - T0-1 (Phase 1): 左手 loss weight up
#   - L1  (Phase 3): bilateral coordination aux loss (左右腕 velocity 差の一致)
#   - L4  (Phase 4): FK anchor aux loss  ← Phase 4 で追加予定
#
# Anchor は既存 loss 計算 2 line (unique)、replacement 内で全 feature を runtime env
# gate で on/off。よって「T0-1 だけ」「L1 だけ」「T0-1+L1」等の Run 組合せは patch
# install 1 回 + env var 切替だけで済む (再 patch 不要)。

GROOT_N1_7_ORIGINAL_SHA256 = "eb6ebb45c7aa701ae46c510638fdefd6e7d1b591e2ac6b223cdfaa82c568e707"

LOSS_PATCH_ANCHOR = (
    '        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask\n'
    "        loss = action_loss.sum() / (action_mask.sum() + 1e-6)\n"
)

# LEFT_HAND_INDICES = (3, 4, 5, 6, 7, 8, 9, 17) = 左腕 7 + 左 grip 1
# BILATERAL_LEFT_INDICES = (3..9), BILATERAL_RIGHT_INDICES = (10..16) = 左右腕 7 dim ずつ
# joint_layout.py と一致必須、patch self-contained のため inline hardcode。
LOSS_PATCH_REPLACEMENT = (
    "        " + PATCH_MARKER + "\n"
    "        # Loss-side patch: T0-1 (left hand weight) + L1 (bilateral) + L4 (FK anchor)\n"
    "        import os as _tr_os\n"
    '        _tr_lhw = float(_tr_os.environ.get("GROOT_LEFT_HAND_LOSS_WEIGHT", "1.0"))\n'
    '        _tr_biw = float(_tr_os.environ.get("GROOT_BILATERAL_LOSS_WEIGHT", "0.0"))\n'
    '        _tr_fkw = float(_tr_os.environ.get("GROOT_FK_ANCHOR_LOSS_WEIGHT", "0.0"))\n'
    "        # T0-1: per-dim loss weight (env GROOT_LEFT_HAND_LOSS_WEIGHT, default 1.0)\n"
    "        if _tr_lhw != 1.0:\n"
    "            _tr_w = torch.ones(pred_actions.shape[-1], dtype=pred_actions.dtype, device=pred_actions.device)\n"
    "            for _tr_idx in (3, 4, 5, 6, 7, 8, 9, 17):  # LEFT_HAND_INDICES: arm 7 + grip 1\n"
    "                if _tr_idx < pred_actions.shape[-1]:\n"
    "                    _tr_w[_tr_idx] = _tr_lhw\n"
    '            action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask * _tr_w\n'
    "        else:\n"
    '            action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask\n'
    "        loss = action_loss.sum() / (action_mask.sum() + 1e-6)\n"
    "        # L1: bilateral coordination aux (env GROOT_BILATERAL_LOSS_WEIGHT, default 0.0)\n"
    "        # 左右腕 7 dim ずつの velocity difference を予測誤差非独立性で拘束\n"
    "        if _tr_biw > 0.0 and pred_actions.shape[-1] >= 17:\n"
    "            _tr_L_pred = pred_actions[..., 3:10]   # BILATERAL_LEFT_INDICES\n"
    "            _tr_R_pred = pred_actions[..., 10:17]  # BILATERAL_RIGHT_INDICES\n"
    "            _tr_L_true = velocity[..., 3:10]\n"
    "            _tr_R_true = velocity[..., 10:17]\n"
    "            _tr_L_mask = action_mask[..., 3:10]\n"
    "            _tr_R_mask = action_mask[..., 10:17]\n"
    "            _tr_bi_mask = _tr_L_mask * _tr_R_mask\n"
    "            _tr_bi_diff = (_tr_L_pred - _tr_R_pred) - (_tr_L_true - _tr_R_true)\n"
    "            _tr_bi_err = (_tr_bi_diff ** 2) * _tr_bi_mask\n"
    "            _tr_bi_loss = _tr_bi_err.sum() / (_tr_bi_mask.sum() + 1e-6)\n"
    "            loss = loss + _tr_biw * _tr_bi_loss\n"
    "        # L4: FK anchor aux (env GROOT_FK_ANCHOR_LOSS_WEIGHT, default 0.0)\n"
    "        # velocity から 1-step Euler で actions_pred を復元 → URDF FK で両 wrist 3D 位置一致\n"
    "        if _tr_fkw > 0.0 and pred_actions.shape[-1] >= 19:\n"
    "            _tr_actions_pred = noisy_trajectory + (1.0 - t) * pred_actions\n"
    "            _tr_q_pred_19 = _tr_actions_pred[..., :19].float()  # FK は float32 で計算\n"
    "            _tr_q_gt_19 = actions[..., :19].float()\n"
    "            if not hasattr(self, \"_tr_fk_singleton\"):\n"
    "                from inference.desktop.perception.g1_urdf_fk_torch import G1WristFKTorch as _tr_fk_cls\n"
    "                self._tr_fk_singleton = _tr_fk_cls.from_urdf(dtype=torch.float32).to(_tr_q_pred_19.device)\n"
    "            _tr_L_pred_pos, _tr_R_pred_pos = self._tr_fk_singleton(_tr_q_pred_19)\n"
    "            _tr_L_gt_pos, _tr_R_gt_pos = self._tr_fk_singleton(_tr_q_gt_19)\n"
    "            # time-level mask (dim 0 の action_mask を time gate として使う、全 dim 一貫想定)\n"
    "            _tr_time_mask = action_mask[..., 0:1].float()\n"
    "            _tr_fk_err = (\n"
    "                ((_tr_L_pred_pos - _tr_L_gt_pos) ** 2).sum(dim=-1, keepdim=True)\n"
    "                + ((_tr_R_pred_pos - _tr_R_gt_pos) ** 2).sum(dim=-1, keepdim=True)\n"
    "            ) * _tr_time_mask\n"
    "            _tr_fk_loss = _tr_fk_err.sum() / (_tr_time_mask.sum() * 2.0 + 1e-6)  # 2 wrists\n"
    "            loss = loss + _tr_fkw * _tr_fk_loss.to(loss.dtype)\n"
)


# ============================================================================
# Env parsing helpers
# ============================================================================


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("true", "1", "yes", "on")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError as exc:
        raise ValueError(f"{name}={v!r} is not a float") from exc


def read_features_from_env() -> dict[str, float | bool]:
    """4 feature の env 値を dict で返す。log 出力にも使う。"""
    return {
        ENV_LEFT_HAND_LOSS_WEIGHT: _env_float(ENV_LEFT_HAND_LOSS_WEIGHT, DEFAULT_LEFT_HAND_LOSS_WEIGHT),
        ENV_PER_CAM_SCALAR_ENABLE: _env_bool(ENV_PER_CAM_SCALAR_ENABLE, DEFAULT_PER_CAM_SCALAR_ENABLE),
        ENV_BILATERAL_LOSS_WEIGHT: _env_float(ENV_BILATERAL_LOSS_WEIGHT, DEFAULT_BILATERAL_LOSS_WEIGHT),
        ENV_FK_ANCHOR_LOSS_WEIGHT: _env_float(ENV_FK_ANCHOR_LOSS_WEIGHT, DEFAULT_FK_ANCHOR_LOSS_WEIGHT),
    }


def _default_for(env_name: str) -> float | bool:
    return {
        ENV_LEFT_HAND_LOSS_WEIGHT: DEFAULT_LEFT_HAND_LOSS_WEIGHT,
        ENV_PER_CAM_SCALAR_ENABLE: DEFAULT_PER_CAM_SCALAR_ENABLE,
        ENV_BILATERAL_LOSS_WEIGHT: DEFAULT_BILATERAL_LOSS_WEIGHT,
        ENV_FK_ANCHOR_LOSS_WEIGHT: DEFAULT_FK_ANCHOR_LOSS_WEIGHT,
    }[env_name]


def any_feature_enabled(features: dict[str, float | bool] | None = None) -> bool:
    """いずれかの feature が default から変更されてるかを判定 (T0-1/T0-2/L1/L4 の enable 判定)。"""
    f = features or read_features_from_env()
    return any(f[k] != _default_for(k) for k in f)


# ============================================================================
# Patch install
# ============================================================================


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_groot_n1_7_path() -> Path:
    """installed lerobot.policies.groot.groot_n1_7 の file path を返す。"""
    spec = importlib.util.find_spec("lerobot.policies.groot.groot_n1_7")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate lerobot.policies.groot.groot_n1_7")
    return Path(spec.origin)


def _apply_loss_side_patch(
    check_only: bool = False, target_path: Path | None = None
) -> bool:
    """Loss-side patch: `groot_n1_7.py` の loss 計算 2 line を combined replacement で置換。

    Replacement 内で T0-1 (左手 loss weight) と L1 (bilateral coordination) の 2 feature
    を runtime env で on/off。install 1 回で全 Run 組合せに対応。

    Args:
        check_only: True なら patch されてなければ RuntimeError、書換はしない
        target_path: 明示 target file (test 用)。None なら installed lerobot を find。

    Returns:
        True: patch applied、False: already patched (no-op)
    """
    path = target_path if target_path is not None else _get_groot_n1_7_path()
    source = path.read_text(encoding="utf-8")

    if PATCH_MARKER in source:
        # already patched, marker + 全 feature env の presence check で 完全性 verify
        missing = [
            v for v in (
                "GROOT_LEFT_HAND_LOSS_WEIGHT",
                "GROOT_BILATERAL_LOSS_WEIGHT",
                "GROOT_FK_ANCHOR_LOSS_WEIGHT",
            )
            if v not in source
        ]
        if missing:
            raise RuntimeError(
                f"incomplete loss-side patch in {path}: marker present but env vars missing: {missing}"
            )
        return False

    if check_only:
        raise RuntimeError(f"loss-side patch is not active in {path}")

    digest = _sha256_text(source)
    if digest != GROOT_N1_7_ORIGINAL_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected {path}: sha256={digest}, "
            f"expected {GROOT_N1_7_ORIGINAL_SHA256}. "
            f"LeRobot version drift の可能性、SHA256 pin 更新要。"
        )

    if source.count(LOSS_PATCH_ANCHOR) != 1:
        raise RuntimeError(
            f"loss-side anchor is not unique in {path}: "
            f"found {source.count(LOSS_PATCH_ANCHOR)} times, expected 1"
        )

    patched = source.replace(LOSS_PATCH_ANCHOR, LOSS_PATCH_REPLACEMENT)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(patched, encoding="utf-8")
    tmp.replace(path)
    return True


def patch_processor(check_only: bool = False) -> bool:
    """Install / verify rotate-specialist patches。

    Feature の enable 状態ごとに patch step を dispatch。
    Phase 1 = T0-1 実装済、Phase 2-4 は NotImplementedError で fail-loud。

    Returns:
        True: いずれかの patch step が applied、False: 全て no-op

    Raises:
        RuntimeError: LeRobot version mismatch or T0-1 anchor mismatch
        NotImplementedError: T0-2/L1/L4 のいずれか enabled (Phase 2-4 未実装)
    """
    version = importlib.metadata.version("lerobot")
    if version != LEROBOT_VERSION:
        raise RuntimeError(
            f"expected lerobot=={LEROBOT_VERSION}, found {version}. "
            f"LeRobot version drift → patch anchor が古い可能性、SHA256 pin 更新要。"
        )

    features = read_features_from_env()
    if not any_feature_enabled(features):
        return False

    any_applied = False

    # L4: FK anchor は G1WristFKTorch import が必要 → training 開始前に verify
    if features[ENV_FK_ANCHOR_LOSS_WEIGHT] != DEFAULT_FK_ANCHOR_LOSS_WEIGHT:
        try:
            from inference.desktop.perception.g1_urdf_fk_torch import G1WristFKTorch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"L4 FK anchor enabled but g1_urdf_fk_torch not importable: {exc}. "
                f"Set PYTHONPATH to include iros_2026_ramen root, "
                f"e.g. export PYTHONPATH=$(pwd):$PYTHONPATH from repo root."
            ) from exc

    # T0-1 / L1 / L4: loss-side patch (Phase 1/3/4 実装済、combined patch で bundle)
    if (
        features[ENV_LEFT_HAND_LOSS_WEIGHT] != DEFAULT_LEFT_HAND_LOSS_WEIGHT
        or features[ENV_BILATERAL_LOSS_WEIGHT] != DEFAULT_BILATERAL_LOSS_WEIGHT
        or features[ENV_FK_ANCHOR_LOSS_WEIGHT] != DEFAULT_FK_ANCHOR_LOSS_WEIGHT
    ):
        applied = _apply_loss_side_patch(check_only=check_only)
        any_applied = any_applied or applied

    # T0-2: Phase 2 で skip 判断 (GR00T arch mismatch、Issue #129 discussion 参照)
    if features[ENV_PER_CAM_SCALAR_ENABLE] != DEFAULT_PER_CAM_SCALAR_ENABLE:
        raise NotImplementedError(
            "T0-2 (per-cam scalar) is skipped for GR00T (per Issue #129 Phase 2)、"
            "GR00T の flat-packed VLM arch では per-cam encoder 単位が存在しないため。"
        )

    return any_applied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify patch already active (Phase 1 T0-1 のみ対応)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = read_features_from_env()
    print(f"[patch_rotate_specialist] features from env: {features}")
    changed = patch_processor(check_only=args.check)
    if changed:
        print("[patch_rotate_specialist] patch applied (T0-1 active)")
    else:
        enabled = any_feature_enabled(features)
        if enabled:
            print("[patch_rotate_specialist] features enabled but already patched (no-op)")
        else:
            print("[patch_rotate_specialist] no features enabled, patch skipped")


if __name__ == "__main__":
    main()
