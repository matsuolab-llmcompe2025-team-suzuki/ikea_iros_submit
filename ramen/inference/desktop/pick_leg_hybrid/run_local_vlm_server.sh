#!/usr/bin/env bash
set -euo pipefail

# Local RTX 5090 profile: leave enough GPU memory for the isolated GR00T
# worker. Runtime requests contain reference A/B + two history frames + one
# current frame, hence the hard minimum of five images per prompt.
docker rm -f iros-pick-leg-vlm >/dev/null 2>&1 || true
docker run -d \
  --name iros-pick-leg-vlm \
  --restart unless-stopped \
  --gpus all \
  -p 127.0.0.1:8000:8000 \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.64 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":5,"video":0}'

echo "VLM container started. Wait until: curl -fsS http://127.0.0.1:8000/health"
