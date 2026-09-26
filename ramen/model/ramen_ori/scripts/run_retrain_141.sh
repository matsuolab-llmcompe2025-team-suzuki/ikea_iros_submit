#!/bin/bash
# Issue #141 RAMEN-Ori 再学習の run を Sakura H100 で順に回す (tmux の中で起動する)。
# 手順は docs/handoff/issue_141_ramen_ori_retrain_training.md。
#
#   bash model/ramen_ori/scripts/run_retrain_141.sh smoke   [run ...]  # 各 run を 30 step (本番の前に全部が動くか)
#   bash model/ramen_ori/scripts/run_retrain_141.sh compile             # 土台 32 を compile なし / ありで 300 step ずつ
#   bash model/ramen_ori/scripts/run_retrain_141.sh full    [run ...]  # 本番 (100k step、ckpt を HF に push)
#
# run の既定は学習の順 (c32 → c16 → c32_no_fk → c32_state_dropout → c32_memory)。
# log は /nvme/train_outputs/logs/<mode>_<run>.log、全体の進み具合は logs/chain_141.log。
# 各 run の終わりに出力先へ印の file を置く: 成功 .done、失敗 .failed (失敗しても次の run に進む)。
set -uo pipefail

MODE=${1:?"usage: $0 smoke|compile|full [run ...]"}
shift
if [ $# -gt 0 ]; then RUNS=("$@"); else RUNS=(c32 c16 c32_no_fk c32_state_dropout c32_memory); fi

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=/nvme/train_outputs
LOGS=$OUT/logs
mkdir -p "$LOGS"
cd "$REPO"
ulimit -n 1048576
export PATH="$HOME/.pixi/bin:$PATH" HF_HOME=/nvme/hf_cache FRAME_CACHE_PRECOMPUTE=false HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=2
set -a; . ./.env; set +a

chain_log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/chain_141.log"; }

run_train() {  # run_train <log の名前> <出力先> <train.py の引数...>
  local name=$1 out=$2
  shift 2
  mkdir -p "$out"
  rm -f "$out/.done" "$out/.failed"
  chain_log "start $name"
  # system の libstdc++ が古いので pixi env のものを先に読む (Phase K と同じ)。
  # Hydra は --config-name より後ろの上書きしか受け付けないので、"$@" (--config-name から始まる) を先に置く
  pixi run --manifest-path model/ramen_ori/pixi.toml --frozen bash -c \
    'LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-} exec python -m model.ramen_ori.train "$@"' _ \
    "$@" hydra.run.dir="$OUT/hydra/$name" > "$LOGS/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$out/.done"; else echo "exit $rc" > "$out/.failed"; fi
  chain_log "end $name exit=$rc"
}

case "$MODE" in
  smoke)
    for run in "${RUNS[@]}"; do
      run_train "smoke_$run" "$OUT/smoke_141/$run" --config-name="retrain/$run" \
        training.max_steps=30 training.ckpt_every=30 training.log_every=5 lr_schedule.warmup_steps=5 \
        val.val_every=20 val.val_max_batches=2 \
        training.hf_autopush.enabled=false wandb.enabled=false training.ckpt_dir="$OUT/smoke_141/$run"
    done ;;
  compile)
    for compile in false true; do
      run_train "compile_${compile}_c32" "$OUT/compile_141/$compile" --config-name=retrain/c32 \
        speedup.torch_compile=$compile training.max_steps=300 training.log_every=50 training.ckpt_every=1000000 \
        lr_schedule.warmup_steps=50 \
        val.enabled=false training.hf_autopush.enabled=false wandb.enabled=false \
        training.ckpt_dir="$OUT/compile_141/$compile"
    done ;;
  full)
    for run in "${RUNS[@]}"; do
      run_train "full_$run" "$OUT/ramen_ori_141_$run" --config-name="retrain/$run"
    done ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2 ;;
esac
chain_log "chain $MODE finished (${RUNS[*]})"
