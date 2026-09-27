#!/usr/bin/env bash
#
# 会場前の確認 (VERIFY.md): image の中の 4 環境が、この機械の GPU を使えるかを見る。
# image の中 (GB10 の instance) で実行する。環境ごとに torch の版・CUDA・compute capability・
# 共有メモリか (integrated)・build に入っている arch を出し、bf16 の行列積を 1 回する。
#
#   runtime    entrypoint・YOLO・RAMEN-Ori      desktop  GR00T 53D の worker
#   vlm        VLM (vLLM)                       pick     GR00T pick の worker (.venv)

set -euo pipefail
cd "${RAMEN_ROOT:-/app/ramen}"

CHECK='
import sys, torch
props = torch.cuda.get_device_properties(0)
integrated = getattr(props, "is_integrated", "?")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
ok = bool(torch.isfinite((x @ x).float().sum()))
assert ok, "GPU bf16 matmul returned non-finite values"
print(f"py {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda} "
      f"cap {torch.cuda.get_device_capability(0)} integrated {integrated} "
      f"archs {torch.cuda.get_arch_list()} matmul_ok {ok}")
'

check_env() {
  local name="$1" output
  shift
  if output=$("$@" -c "${CHECK}" 2>&1); then
    printf '[%s] %s\n' "${name}" "${output}"
  else
    printf '[%s] FAILED\n%s\n' "${name}" "${output}" >&2
    return 1
  fi
}

check_env runtime pixi run --as-is -e runtime python
check_env desktop pixi run --as-is --manifest-path inference/desktop/pixi.toml python
check_env vlm pixi run --as-is --manifest-path inference/desktop/pixi.toml -e vlm python
check_env pick model/subtask_policy_training/.venv/bin/python
