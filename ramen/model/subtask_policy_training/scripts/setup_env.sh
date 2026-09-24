#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uv venv .venv --python 3.12 --allow-existing
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/patch_lerobot_groot_relative_eef.py

python - <<'PY'
import lerobot
import datasets
import huggingface_hub
import mcap
import numpy
import PIL
import torch
import wandb
try:
    import transformers
except ImportError:  # pragma: no cover - setup_env should install groot extras before this runs.
    transformers = None
try:
    import decord
except ImportError:  # pragma: no cover
    decord = None

print("Environment ready")
print("lerobot", getattr(lerobot, "__version__", "unknown"))
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("datasets", datasets.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("mcap", getattr(mcap, "__version__", "unknown"))
print("numpy", numpy.__version__)
print("pillow", PIL.__version__)
print("wandb", wandb.__version__)
print("transformers", getattr(transformers, "__version__", "missing"))
print("decord", getattr(decord, "__version__", "missing"))
PY
