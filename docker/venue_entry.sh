#!/usr/bin/env bash
#
# 提出 image (Thor) の起動口。docker/Dockerfile.thor の ENTRYPOINT。
#
# 会場:  docker run … <image> --stage N --actuate
#   会場で変わらない option はここで付ける。
#     --action-sink boundary     運営の :5556 / :5557 / :5555 でつなぐ (SDK を直接使わない)
#     --synthetic-hand-state     手の指令を (T,25) の手の列に載せるのに必須。手の実測は :5557 の
#                                gripper_q があればそれを使う (BoundaryDex1StateSource)
#     --boundary-host 0.0.0.0    :5556 は自分が bind し、PC2 の運営 adapter がつないでくる
#     --spawn-vlm-server         hybrid pick の run では VLM をこの run の中で起動する
#     --gpu-models all           stage の model を全部、起動時に GPU に載せる (run の途中で読み込み
#                                待ちをしない)。GB10 (arm64・128 GB 共有メモリ) で VLM 込み最大 42 GB、
#                                共有メモリは 48 GB 残ることを確認 (2026-09-25)
#   PC2 (カメラ :5555 と状態 :5557 の配信元) は -e IROS_ORIN_HOST=<PC2 の IP> で渡す
#   (entrypoint の configure_official_endpoints が読む)。
#   後ろに付けた option は上書きになる (argparse は後勝ち)。例: --gpu-models 2
#
# それ以外: `-` で始まらない引数はそのまま実行する (bash、python3 conformance.py … など)。

set -euo pipefail

if [[ $# -gt 0 && "$1" != -* ]]; then
  exec "$@"
fi
if [[ $# -eq 0 ]]; then
  echo "usage: docker run … <image> --stage N [--actuate] [entrypoint の option]" >&2
  echo "       docker run … <image> bash" >&2
  exit 2
fi
if [[ -z "${IROS_ORIN_HOST:-}" ]]; then
  echo "error: PC2 (カメラ :5555 と状態 :5557 の配信元) の宛先を -e IROS_ORIN_HOST=<PC2 の IP> で渡す" >&2
  exit 2
fi

cd "${RAMEN_ROOT:-/app/ramen}"
# --as-is: 実行時に環境の install も lock の更新もしない (会場は実行時オフライン)
exec pixi run --as-is -e runtime python -m inference.desktop.entrypoint \
  --action-sink boundary \
  --synthetic-hand-state \
  --boundary-host 0.0.0.0 \
  --spawn-vlm-server \
  --gpu-models all \
  "$@"
