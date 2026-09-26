#!/usr/bin/env bash
set -euo pipefail

# 提出 image (Thor) の container の中で hybrid pick の VLM を立てる (RunPod の起動確認も同じ script)。
# 会場では entrypoint が --spawn-vlm-server でこの script を run ごとに子 process として起動し、
# run の終わりに止める (vlm_server.py)。
# 問い合わせの形は開発機の run_local_vlm_server.sh と同じ (model 名・port・画像 5 枚・長さの上限)。
#   KV            : max-model-len 4096 の 1 本分 = 4096 token × 144 KiB (36 層 × KV head 8 × 128 × bf16 × K/V)
#                   ≒ 578 MiB が起動時の必要量。足りなければ起動時に止まる。余裕を見て 1 GiB。
#   util 0.01     : KV を指定すると util は起動時の空き確認 (空き ≥ 全体 × util) にしか使われない。
#                   既定 0.92 だと他の model が載った統合メモリで起動できないので実質外す。
#   host          : container は --network host なので 127.0.0.1 に限る (0.0.0.0 は会場のネットに開く)。
#                   port は vlm_server.py の VENUE_VLM_PORT と同じ (test で突き合わせる)。
#   attention     : vLLM 同梱の FlashAttention は sm_110 の機械語を持たないので FlashInfer / SDPA。
#   ネット        : 会場は実行時にネットへ出ない。下の「ネットに出ない」で vLLM 0.29.0 の経路上の出口を塞ぐ。
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # inference/desktop (vlm 環境の pixi.toml)

# 実行時の compile 結果 (Triton の小さな kernel など) を 1 か所に置く。container は run ごとに
# 作り直すので、会場では host の directory をここに mount して 2 回目以降の compile を省く。
cache_root="${RAMEN_VLM_CACHE_DIR:-${HOME}/.cache/ramen_vlm}"
export TRITON_CACHE_DIR="${cache_root}/triton"
export VLLM_CACHE_ROOT="${cache_root}/vllm"
export FLASHINFER_WORKSPACE_BASE="${cache_root}"

# 温度 0 なので答えは変わらない。sampling kernel の JIT を避ける。
export VLLM_USE_FLASHINFER_SAMPLER=0

# --- ネットに出ない (vLLM 0.29.0 の serve 経路で外へ出うる所を source で確かめて塞ぐ) ---
# HF hub: 重みは事前取得済み。offline だと vLLM が repo id を cache 内の snapshot パスに置き換え、
# 無ければ起動時に止まる。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 利用統計 (stats.vllm.ai への送信)
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# FlashInfer attention を選ぶと、機種の判定より前に NVIDIA の artifactory へ GET を 1 回送る
# (TRT-LLM kernel の取得確認)。手元の閉じた port に向けて即失敗させる (sm_110 はもともと対象外)。
export FLASHINFER_CUBINS_REPOSITORY="http://127.0.0.1:9/"
# 自分の IP を調べる処理 (8.8.8.8 へ UDP connect) は今の経路では呼ばれないが、入口を塞ぐ。
export VLLM_HOST_IP=127.0.0.1
# 保険: 見落とした HTTP 通信は閉じた手元の port へ向ける (この script = vLLM の process だけ)。
for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  export "${proxy_var}=http://127.0.0.1:9"
done
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

# vlm 環境を有効にしてから vllm に置き換わる (pixi を間に挟まないので、止める信号が vllm に直接届く)。
# --as-is: 実行時に環境の install も lock の更新もしない。
eval "$(pixi shell-hook --as-is -e vlm -s bash)"
exec vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":5,"video":0}' \
  --allowed-media-domains 127.0.0.1 \
  --kv-cache-memory-bytes 1G \
  --gpu-memory-utilization 0.01 \
  --enforce-eager \
  --attention-backend FLASHINFER \
  --mm-encoder-attn-backend TORCH_SDPA
