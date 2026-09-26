#!/usr/bin/env bash
#
# 会場前の確認 (VERIFY.md): 1 stage を会場と同じ起動口 (ramen-venue) で起動する。image の中
# (GB10 の instance) で実行する。PC2 の代わりに運営の mocks/mock_orin.py (頭の左右・手首 2 台・
# 状態) を立てる。既定は指令を出さない (--actuate 無し) 起動確認だけ。
#
#   run_stage.sh <stage> <出力名> [ramen-venue へ足す option ...]
#   run_stage.sh 2 stage2_all --gpu-models 2         # 例: 会場の既定 (all) を上書き
#   NOSTRACE=1 run_stage.sh 1 stage1_plain            # 起動秒だけを測る (strace は起動を遅くする)
#   ACTUATE_HOLD=60 run_stage.sh 0 stage0_actuate     # --actuate の経路 (下)
#
# ACTUATE_HOLD=<秒>: --actuate を付け、Enter 1 の問いが出たら改行を送り、go-live 待ち
# (`[go-live]`) が出てから <秒> 待って Ctrl+C (python の process にだけ SIGINT) を送る。
# 指令の経路 (boundary の publish・実測の関節の読み取り) が落ちずに動くことと、Ctrl+C の後の
# 後始末 (手を開いて腕を下ろす) までを通す。模擬の PC2 の関節は指令と関係なく sin 波で動くので、
# go-live 待ちは成立し、その先の準備動作は時間切れになる (Stage 0 は腕を下ろせず設計どおり止まる)。
# result.txt に mode=actuate を書く (summarize.py の判定)。
#
# 出力 (${RUNS_DIR:-/root/runs}/<出力名>/):
#   run.log      起動口の出力            result.txt  rc と秒
#   mem.log      MemAvailable (kB)      gpu.log     nvidia-smi の process ごとの使用量の合計 (MiB)
#   connect.log  strace の connect / execve (外向き接続の確認は summarize.py)
#
# 共有メモリの機械 (GB10・Thor) では page cache も「使用中」に見えるので、GPU の使用量は
# gpu.log、残りの余裕は mem.log の MemAvailable の最小で見る。
#
# test 用の差し替え (tests/test_gb10_tools.py): VENUE_BIN (起動口)、NO_MOCK=1 (mock_orin を立てない)。
# 空の配列は ${A[@]+"${A[@]}"} で展開する (bash 4.4 未満は set -u で空の配列を展開できない)。

set -u

N="$1"
NAME="$2"
shift 2
OUT="${RUNS_DIR:-/root/runs}/${NAME}"
VENUE_BIN="${VENUE_BIN:-/usr/local/bin/ramen-venue}"
rm -rf "${OUT}"
mkdir -p "${OUT}"
cd "${RAMEN_ROOT:-/app/ramen}"

if [[ -z "${NO_MOCK:-}" ]]; then
  pixi run --as-is -e runtime python /app/mocks/mock_orin.py --stereo-ego > "${OUT}/mock.log" 2>&1 &
  sleep 5
fi

( while true; do
    echo "$(date +%s) $(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null)"
    sleep 1
  done ) > "${OUT}/mem.log" &
MEM_PID=$!
( while true; do
    echo "$(date +%s) $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s += $1} END {print s + 0}')"
    sleep 1
  done ) > "${OUT}/gpu.log" &
GPU_PID=$!

TRACE=(strace -f -qq -e trace=connect,execve -o "${OUT}/connect.log")
if [[ -n "${NOSTRACE:-}" ]]; then
  TRACE=()
fi

# run.log に $1 (grep -E) が出るまで待つ。起動口が先に終わったか $2 秒で諦めたら 1。
wait_for_log() {
  local deadline=$(($(date +%s) + $2))
  until grep -qE "$1" "${OUT}/run.log" 2>/dev/null; do
    if ! kill -0 "${VENUE_PID}" 2>/dev/null || (($(date +%s) > deadline)); then
      return 1
    fi
    sleep 1
  done
}

T0=$(date +%s)
if [[ -z "${ACTUATE_HOLD:-}" ]]; then
  MODE=preflight
  IROS_ORIN_HOST=127.0.0.1 ${TRACE[@]+"${TRACE[@]}"} "${VENUE_BIN}" --stage "${N}" "$@" \
    > "${OUT}/run.log" 2>&1 < /dev/null
  RC=$?
else
  MODE=actuate
  FIFO="${OUT}/stdin.fifo"
  mkfifo "${FIFO}"
  # 裏で (&) 起動した process は SIGINT が無視の設定で始まり、python は Ctrl+C を受けなくなる
  # (bash は job control の無い裏の process の SIGINT を無視にする)。起動の直前に既定へ戻す。
  RESET_SIGINT=(python3 -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])')
  IROS_ORIN_HOST=127.0.0.1 "${RESET_SIGINT[@]}" ${TRACE[@]+"${TRACE[@]}"} "${VENUE_BIN}" --stage "${N}" --actuate "$@" \
    > "${OUT}/run.log" 2>&1 < "${FIFO}" &
  VENUE_PID=$!
  # 書き手を開いたままにする (閉じると起動口の input() が EOF で落ちる)
  exec 3> "${FIFO}"
  if wait_for_log "Enter starts" "${ACTUATE_ENTER_TIMEOUT:-900}"; then
    echo "[run_stage] sending Enter 1" >> "${OUT}/steps.log"
    printf '\n' >&3
    if wait_for_log '\[go-live\]' 120; then
      sleep "${ACTUATE_HOLD}"
    fi
  fi
  # Ctrl+C 相当。pixi と strace ではなく python の process にだけ送る (会場の Ctrl+C と同じ後始末を通す)。
  # -i: macOS の python は .../MacOS/Python として動く (手元の test 用。image は python)
  echo "[run_stage] sending SIGINT" >> "${OUT}/steps.log"
  pkill -INT -i -f '^[^ ]*python[0-9.]* -m inference\.desktop\.entrypoint' 2>/dev/null  # 信号は先頭 (macOS)
  # 後始末 (腕を下ろす、最大 60 s) を待つ。終わらなければ止める
  deadline=$(($(date +%s) + ${ACTUATE_EXIT_TIMEOUT:-180}))
  while kill -0 "${VENUE_PID}" 2>/dev/null && (($(date +%s) <= deadline)); do
    sleep 1
  done
  if kill -0 "${VENUE_PID}" 2>/dev/null; then
    echo "[run_stage] still running after SIGINT; SIGTERM" >> "${OUT}/steps.log"
    kill -TERM "${VENUE_PID}" 2>/dev/null
  fi
  wait "${VENUE_PID}"
  RC=$?
  exec 3>&-
  rm -f "${FIFO}"
fi
T1=$(date +%s)

kill "${GPU_PID}" "${MEM_PID}" 2>/dev/null
pkill -f mock_orin.py 2>/dev/null
wait 2>/dev/null
echo "rc=${RC} secs=$((T1 - T0)) mode=${MODE}" | tee "${OUT}/result.txt"
