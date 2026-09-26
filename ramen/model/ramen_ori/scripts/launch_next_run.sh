#!/bin/bash
# Issue #122: Phase 2 5-run pipeline の next-run 自動起動 dispatcher (Sakura H100 用)。
#
# Monitor から呼ばれる想定。既存 train_runN tmux が alive なら skip、DEAD かつ
# 直前 run の 100k ckpt が保存済なら次 run を tmux で起動する idempotent スクリプト。
#
# 呼び方 (Sakura 上):
#   bash ~/GitHub/iros_2026_ramen/model/ramen_ori/scripts/launch_next_run.sh
#
# 判定順序 (先に 100k ckpt があるほど下位):
#   default (Run 1) 100k → 次は g3 (Run 2)  ※ 既に手動 launch 済、この分岐は使わない
#   g3 (Run 2) 100k     → 次は meanflow (Run 3)
#   meanflow (Run 3) 100k → 次は awr (Run 4)
#   awr (Run 4) 100k     → 次は radio (Run 5)
#   radio (Run 5) 100k   → ALL DONE
set -euo pipefail

OUTPUTS=/nvme/outputs
REPO=$HOME/GitHub/iros_2026_ramen
LOG_FILE=$OUTPUTS/launch_next_run.log

# PR #127 review MEDIUM: NVMe wipe 直後の初回 boot で $OUTPUTS 未作成なら
# set -e で即死し launch も log も残らないため、log 書き込み前に確保する。
mkdir -p "$OUTPUTS"

# 既存 train_runN tmux が alive なら skip
for tag in train_run2 train_run3 train_run4 train_run5; do
  if tmux has-session -t "$tag" 2>/dev/null; then
    echo "[$(date -u +%FT%TZ)] $tag alive, skip" >> "$LOG_FILE"
    exit 0
  fi
done

BOOT="ulimit -n 1048576 && cd $REPO && export PATH=\$HOME/.pixi/bin:\$PATH && export PIXI_CACHE_DIR=\$HOME/.cache/pixi && export LEROBOT_FRAME_CACHE_ENABLE=true && set -a && . .env && set +a && cd model/ramen_ori"

launch_run3_meanflow() {
  local ts
  ts=$(date +%Y-%m-%d_%H%M)
  tmux new -d -s train_run3 "$BOOT && pixi run train --config-name=real_task5_7_meanflow \
    training.num_workers=8 \
    training.shm_cleanup_every=0 \
    training.shm_cleanup_on_ckpt=false \
    training.ckpt_dir=$OUTPUTS/ramen_ori_phase2_meanflow \
    training.hf_autopush.enabled=true \
    training.hf_autopush.repo_id=Team-RAMEN/IROS2026_RAMEN_hara_task_5_7_ramen_ori_meanflow_100k_v1 \
    training.hf_autopush.every_n_ckpt=2 \
    val.split_json=$OUTPUTS/split_task5_7.json \
    wandb.run_name=meanflow_100k_$ts \
    2>&1 | tee $OUTPUTS/train_meanflow.log"
  echo "[$(date -u +%FT%TZ)] launched train_run3 (MeanFlow) at $ts" | tee -a "$LOG_FILE"
}

launch_run4_awr() {
  local ts
  ts=$(date +%Y-%m-%d_%H%M)
  tmux new -d -s train_run4 "$BOOT && pixi run train --config-name=real_task5_7 \
    training.num_workers=8 \
    training.shm_cleanup_every=0 \
    training.shm_cleanup_on_ckpt=false \
    training.awr.enabled=true \
    training.ckpt_dir=$OUTPUTS/ramen_ori_phase2_awr \
    training.hf_autopush.enabled=true \
    training.hf_autopush.repo_id=Team-RAMEN/IROS2026_RAMEN_hara_task_5_7_ramen_ori_awr_100k_v1 \
    training.hf_autopush.every_n_ckpt=2 \
    val.split_json=$OUTPUTS/split_task5_7.json \
    wandb.run_name=awr_100k_$ts \
    2>&1 | tee $OUTPUTS/train_awr.log"
  echo "[$(date -u +%FT%TZ)] launched train_run4 (AWR) at $ts" | tee -a "$LOG_FILE"
}

launch_run5_radio() {
  local ts
  ts=$(date +%Y-%m-%d_%H%M)
  tmux new -d -s train_run5 "$BOOT && pixi run train --config-name=real_task5_7 \
    training.num_workers=8 \
    training.shm_cleanup_every=0 \
    training.shm_cleanup_on_ckpt=false \
    vision_backbone.variant=radio-b \
    training.ckpt_dir=$OUTPUTS/ramen_ori_phase2_radio \
    training.hf_autopush.enabled=true \
    training.hf_autopush.repo_id=Team-RAMEN/IROS2026_RAMEN_hara_task_5_7_ramen_ori_radio_100k_v1 \
    training.hf_autopush.every_n_ckpt=2 \
    val.split_json=$OUTPUTS/split_task5_7.json \
    wandb.run_name=radio_100k_$ts \
    2>&1 | tee $OUTPUTS/train_radio.log"
  echo "[$(date -u +%FT%TZ)] launched train_run5 (RADIO) at $ts" | tee -a "$LOG_FILE"
}

# ckpt 存在で次 run を決定 (下から順、既に完走したものはスキップ)
if [ -f "$OUTPUTS/ramen_ori_phase2_radio/ckpt_step_100000.pt" ]; then
  echo "[$(date -u +%FT%TZ)] ALL 5 RUNS COMPLETE — nothing to launch" | tee -a "$LOG_FILE"
elif [ -f "$OUTPUTS/ramen_ori_phase2_awr/ckpt_step_100000.pt" ]; then
  launch_run5_radio
elif [ -f "$OUTPUTS/ramen_ori_phase2_meanflow/ckpt_step_100000.pt" ]; then
  launch_run4_awr
elif [ -f "$OUTPUTS/ramen_ori_phase2_g3/ckpt_step_100000.pt" ]; then
  launch_run3_meanflow
else
  echo "[$(date -u +%FT%TZ)] no 100k ckpt yet (Run 2 still running or not done) — nothing to launch" | tee -a "$LOG_FILE"
fi
