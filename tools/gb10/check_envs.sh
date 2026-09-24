#!/usr/bin/env bash
#
# 会場前の確認 (VERIFY.md): image の中の 4 環境が、この機械の GPU を使えるかを見る。
# image の中 (GB10 の instance) で実行する。環境ごとに torch の版・CUDA・compute capability・
# 共有メモリか (integrated)・build に入っている arch を出し、bf16 の行列積を 1 回する。
#
#   runtime    entrypoint・YOLO・RAMEN-Ori      desktop  GR00T 53D の worker
#   vlm        VLM (vLLM)                       pick     GR00T pick の worker (.venv)

set -u
cd "${RAMEN_ROOT:-/app/ramen}"

CHECK='
import sys, torch
props = torch.cuda.get_device_properties(0)
integrated = getattr(props, "is_integrated", "?")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
ok = bool(torch.isfinite((x @ x).float().sum()))
print(f"py {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda} "
      f"cap {torch.cuda.get_device_capability(0)} integrated {integrated} "
      f"archs {torch.cuda.get_arch_list()} matmul_ok {ok}")
'

echo "[runtime] $(pixi run --as-is -e runtime python -c "${CHECK}" 2>&1 | tail -1)"
echo "[desktop] $(pixi run --as-is --manifest-path inference/desktop/pixi.toml python -c "${CHECK}" 2>&1 | tail -1)"
echo "[vlm]     $(pixi run --as-is --manifest-path inference/desktop/pixi.toml -e vlm python -c "${CHECK}" 2>&1 | tail -1)"
echo "[pick]    $(model/subtask_policy_training/.venv/bin/python -c "${CHECK}" 2>&1 | tail -1)"
