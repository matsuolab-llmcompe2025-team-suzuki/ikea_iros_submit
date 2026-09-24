#!/usr/bin/env bash
#
# 会場前の確認 (VERIFY.md): 1 stage を会場と同じ起動口 (ramen-venue) で、指令を出さずに
# (--actuate 無し) 起動する。image の中 (GB10 の instance) で実行する。PC2 の代わりに運営の
# mocks/mock_orin.py (頭の左右・手首 2 台・状態) を立てる。
#
#   run_stage.sh <stage> <出力名> [ramen-venue へ足す option ...]
#   run_stage.sh 2 stage2_all --gpu-models 2         # 例: 会場の既定 (all) を上書き
#   NOSTRACE=1 run_stage.sh 1 stage1_plain            # 起動秒だけを測る (strace は起動を遅くする)
#
# 出力 (${RUNS_DIR:-/root/runs}/<出力名>/):
#   run.log      起動口の出力            result.txt  rc と秒
#   mem.log      MemAvailable (kB)      gpu.log     nvidia-smi の process ごとの使用量の合計 (MiB)
#   connect.log  strace の connect / execve (外向き接続の確認は summarize.py)
#
# 共有メモリの機械 (GB10・Thor) では page cache も「使用中」に見えるので、GPU の使用量は
# gpu.log、残りの余裕は mem.log の MemAvailable の最小で見る。

set -u

N="$1"
NAME="$2"
shift 2
OUT="${RUNS_DIR:-/root/runs}/${NAME}"
rm -rf "${OUT}"
mkdir -p "${OUT}"
cd "${RAMEN_ROOT:-/app/ramen}"

pixi run --as-is -e runtime python /app/mocks/mock_orin.py --stereo-ego > "${OUT}/mock.log" 2>&1 &
sleep 5

( while true; do
    echo "$(date +%s) $(awk '/MemAvailable/{print $2}' /proc/meminfo)"
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

T0=$(date +%s)
IROS_ORIN_HOST=127.0.0.1 "${TRACE[@]}" /usr/local/bin/ramen-venue --stage "${N}" "$@" \
  > "${OUT}/run.log" 2>&1 < /dev/null
RC=$?
T1=$(date +%s)

kill "${GPU_PID}" "${MEM_PID}" 2>/dev/null
pkill -f mock_orin.py 2>/dev/null
wait 2>/dev/null
echo "rc=${RC} secs=$((T1 - T0))" | tee "${OUT}/result.txt"
