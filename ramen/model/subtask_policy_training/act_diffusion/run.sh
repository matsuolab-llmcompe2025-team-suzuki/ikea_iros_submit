#!/usr/bin/env bash
# Issue #139: ACT / Diffusion Policy 学習の起動 wrapper。
#
# Usage (module dir で pixi 経由):
#   cd model/subtask_policy_training
#   pixi run train-act-dp act_diffusion/configs/act_rotate_table_base.yaml [KEY=VALUE ...]
#
# 動作:
#   1. run 設定 YAML を検証して env に export (末尾の KEY=VALUE で上書き可)
#   2. ckpt_uploader を低優先度 (nice / ionice) で背景起動 — 完了済み ckpt を HF に定期同期
#   3. scripts/train_lerobot.sh で学習
#   4. 終了時 (正常 / 異常とも) に uploader を止め、--final で最終同期 + repo root に最新 model
#
# CKPT_UPLOAD_DRY_RUN=true で uploader は network を使わず予定だけ出力する (ローカル確認用)。

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_config_yaml> [KEY=VALUE ...]" >&2
  exit 2
fi
CONFIG_YAML="$1"
shift

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$MODULE_DIR/../.." && pwd)"
cd "$MODULE_DIR"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# HF_TOKEN / WANDB_API_KEY 等の secrets (private dataset の pull と upload に要る)
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a && . "$REPO_ROOT/.env" && set +a
fi

set_args=()
for kv in "$@"; do
  set_args+=(--set "$kv")
done
python act_diffusion/run_config.py --config "$CONFIG_YAML" "${set_args[@]}" --format summary
eval "$(python act_diffusion/run_config.py --config "$CONFIG_YAML" "${set_args[@]}" --format shell)"

# 既存 cache を使う設定なのに restore 先が無いと、frame cache patch は黙って mp4 decode に
# fallback し、GPU が dataloader 待ちで遊ぶ。起動前に落とす。
if [[ -n "${FRAME_CACHE_ROOT_OVERRIDE:-}" && ! -f "$FRAME_CACHE_ROOT_OVERRIDE/_cache_meta.json" ]]; then
  echo "ERROR: FRAME_CACHE_ROOT_OVERRIDE=$FRAME_CACHE_ROOT_OVERRIDE に _cache_meta.json が無い" >&2
  echo "  frame_cache_tars を restore してから起動すること (handoff 参照)" >&2
  exit 2
fi

# OUTPUT_DIR は train_lerobot.sh と同じく resolve_training_config から得る (module dir 相対)
eval "$(python scripts/resolve_training_config.py --config configs/subtask_training.json --format shell \
  | grep -E '^export OUTPUT_DIR=')"
CHECKPOINTS_DIR="$OUTPUT_DIR/checkpoints"

# 前 run の出力は uploader を起動する前に片付ける (train_lerobot.sh にも同じ確認があるが、それでは遅い)。
# uploader は起動時に $OUTPUT_DIR/hf_upload_state.json を読むので、後から消しても前 run の
# 「upload 済み step」を覚えたまま新 run の同名 step を skip する。止める場合も trap を張る前に
# 止めないと、--final が前 run の最新 model を repo root に戻す。
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]]; then
  if [[ "${OUTPUT_DIR_FORCE:-false}" == "true" ]]; then
    echo "[run] OUTPUT_DIR_FORCE=true → removing existing $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
  else
    echo "ERROR: $OUTPUT_DIR is non-empty (前 run の出力)。HF に上がっていることを確認してから" >&2
    echo "  OUTPUT_DIR_FORCE=true で再実行してください。" >&2
    exit 2
  fi
fi

uploader_args=(--checkpoints-dir "$CHECKPOINTS_DIR" --repo-id "$POLICY_REPO_ID")
if [[ "${CKPT_UPLOAD_DRY_RUN:-false}" == "true" ]]; then
  uploader_args+=(--dry-run)
fi

nice -n 19 ionice -c3 python -m model.subtask_policy_training.act_diffusion.ckpt_uploader \
  "${uploader_args[@]}" --interval "${CKPT_UPLOAD_INTERVAL_S:-60}" &
UPLOADER_PID=$!
# 起動直後に落ちた uploader (import 失敗など) のまま 9〜14 h 学習すると、ckpt は /nvme にしか残らない
sleep 5
if ! kill -0 "$UPLOADER_PID" 2>/dev/null; then
  echo "ERROR: ckpt_uploader が起動直後に終了した。上の traceback を確認すること" >&2
  exit 2
fi

finish() {
  local status=$?
  local uploader_status=0
  kill -TERM "$UPLOADER_PID" 2>/dev/null || true
  wait "$UPLOADER_PID" 2>/dev/null || uploader_status=$?
  if [[ $uploader_status -ne 0 ]]; then
    echo "[run] WARN: ckpt_uploader exited with status $uploader_status (途中で止まっていた可能性)" >&2
  fi
  echo "[run] final checkpoint sync (training exit status: $status)"
  python -m model.subtask_policy_training.act_diffusion.ckpt_uploader "${uploader_args[@]}" --final \
    || echo "[run] WARN: final sync failed — checkpoints は $CHECKPOINTS_DIR に残っている" >&2
  exit "$status"
}
trap finish EXIT

# ckpt は uploader が HF に上げるので、LeRobot の wandb artifact として同じ model を二重に上げない
# (上げると ~/.cache/wandb にもコピーが残り、DP は 1.17 GB × 10 で SSD を圧迫する)
bash scripts/train_lerobot.sh --wandb.disable_artifact=true
