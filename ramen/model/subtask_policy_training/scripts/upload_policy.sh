#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "$VENV_DIR" && -d ".venv_lerobot060" ]]; then
  VENV_DIR=".venv_lerobot060"
fi
source "$VENV_DIR/bin/activate"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/subtask_training.json}"
eval "$(python scripts/resolve_training_config.py --config "$TRAIN_CONFIG" --format shell)"

model_source_args=()
if [[ -n "${MODEL_DIR:-}" ]]; then
  model_source_args+=(--model-dir "$MODEL_DIR")
else
  model_source_args+=(--output-dir "$OUTPUT_DIR")
fi

private_args=()
if [[ "$PRIVATE" == "true" ]]; then
  private_args+=(--private)
fi
dry_run_args=()
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  dry_run_args+=(--dry-run)
fi

echo "policy: $POLICY_REPO_ID"
echo "output_dir: $OUTPUT_DIR"
if [[ -n "${MODEL_DIR:-}" ]]; then
  echo "model_dir: $MODEL_DIR"
fi

python scripts/upload_policy.py \
  --repo-id "$POLICY_REPO_ID" \
  "${model_source_args[@]}" \
  "${private_args[@]}" \
  "${dry_run_args[@]}"
