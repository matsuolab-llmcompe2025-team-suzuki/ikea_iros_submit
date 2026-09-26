#!/usr/bin/env bash
# Issue #129 rotate specialist run launcher。
#
# Usage:
#   bash model/subtask_policy_training/scripts/rotate_specialist/run.sh \
#     model/subtask_policy_training/configs/rotate_specialist/run1_t01_only.yaml
#
# 動作:
#   1. YAML config を read → env として export
#   2. Rotate specialist patch を install (idempotent、feature enabled 時のみ実 patch)
#   3. train_lerobot.sh 起動 (既存の GR00T + overlay pipeline を使う)
#
# 事前準備 (operator 側で 1 回):
#   - HF dataset を /nvme/rotate_table_base に pull (handoff §2.4、~2-3 min @500Mbps)
#   - OBB_PRECOMPUTED_ROOT=/nvme/rotate_table_base/obb_yolo で export
#     (rotate_table_base_merged_v1 に overlay cache が同梱)
#   - Sakura 起動 + venv activate (train_lerobot.sh 内で自動 activate)

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <rotate_specialist_config_yaml>" >&2
    echo "  例: $0 model/subtask_policy_training/configs/rotate_specialist/run1_t01_only.yaml" >&2
    exit 2
fi

CONFIG_YAML="$1"
if [[ ! -f "$CONFIG_YAML" ]]; then
    echo "ERROR: config not found: $CONFIG_YAML" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"           # rotate_specialist/
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                            # scripts/
ROOT_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"                              # subtask_policy_training/
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"                             # iros_2026_ramen/

# PYTHONPATH に REPO_ROOT を prefix (L4 の FK import 用、train_lerobot.sh も同様に設定するが preemptive で明示)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Venv activate: config.py / patch_lerobot.py の python 呼出前に必要。
# train_lerobot.sh 内でも activate されるが、それより前に本 script が python を呼ぶため事前必須。
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
if [[ ! -d "$VENV_DIR" && -d "$ROOT_DIR/.venv_lerobot060" ]]; then
    VENV_DIR="$ROOT_DIR/.venv_lerobot060"
fi
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR (setup_env.sh で作成要)" >&2
    exit 2
fi
source "$VENV_DIR/bin/activate"

# .env を source (HF_TOKEN / WANDB_API_KEY 等の secrets を training プロセスに渡す)
# LeRobot は dataset load 時に private repo の metadata を HF Hub に問合せ、token 無いと 401
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a && . "$REPO_ROOT/.env" && set +a
    echo "[launch] sourced $REPO_ROOT/.env (HF_TOKEN / WANDB_API_KEY loaded)"
fi

echo "[launch] loading rotate specialist config: $CONFIG_YAML"
python "$SCRIPT_DIR/config.py" --config "$CONFIG_YAML" --format summary

# YAML → shell export に変換 → eval で現 shell に反映
eval "$(python "$SCRIPT_DIR/config.py" --config "$CONFIG_YAML" --format shell)"

# OBB_PRECOMPUTED_ROOT (overlay cache path) は operator 側 export 必須
if [[ "${OBB_OVERLAY_ENABLE:-false}" == "true" && -z "${OBB_PRECOMPUTED_ROOT:-}" ]]; then
    echo "ERROR: OBB_OVERLAY_ENABLE=true requires OBB_PRECOMPUTED_ROOT to be exported" >&2
    echo "  例: export OBB_PRECOMPUTED_ROOT=/nvme/rotate_table_base/obb_yolo" >&2
    exit 2
fi

# Patch install (feature 全 off で no-op、いずれか enabled で実 patch)
echo "[launch] installing rotate specialist patch (idempotent)"
python "$SCRIPT_DIR/patch_lerobot.py"

# 学習起動 (train_lerobot.sh は scripts/ にある = SCRIPTS_DIR 直下)
echo "[launch] starting train_lerobot.sh"
exec bash "$SCRIPTS_DIR/train_lerobot.sh"
