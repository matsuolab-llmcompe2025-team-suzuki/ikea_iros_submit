#!/usr/bin/env bash
set -euo pipefail

FEATURE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$FEATURE_DIR/../.." && pwd)"
cd "$FEATURE_DIR"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/subtask_training.json}"
export SUBTASK="${SUBTASK:-flip_table}"
export POLICY_TYPE=flow_matching
eval "$("$PYTHON_BIN" scripts/resolve_training_config.py --config "$TRAIN_CONFIG" --format shell)"
TRAINING_VIEW_ROOT="${FLOW_TRAINING_VIEW_ROOT:-$TRAINING_VIEW_ROOT}"
OUTPUT_DIR="${FLOW_OUTPUT_DIR:-$OUTPUT_DIR}"

prepare=(
  "$PYTHON_BIN" scripts/materialize_lerobot_training_view.py
  --config "$TRAIN_CONFIG"
  --repo-id "$DATASET_REPO_ID"
  --output-root "$TRAINING_VIEW_ROOT"
  --policy-type flow_matching
)
if [[ -n "$DATASET_REVISION" ]]; then
  prepare+=(--revision "$DATASET_REVISION")
fi
if [[ -n "${SOURCE_DATASET_ROOT:-}" ]]; then
  prepare+=(--source-root "$SOURCE_DATASET_ROOT")
fi
if [[ "$TRAINING_VIEW_FORCE" == "true" ]]; then
  prepare+=(--force)
fi
"${prepare[@]}"

split_file="$TRAINING_VIEW_ROOT/meta/team_ramen_episode_split.json"
if [[ ! -f "$split_file" ]]; then
  split_file="${FLOW_SPLIT_FILE:-outputs/audits/flip_table_episode_split.json}"
fi
if [[ ! -f "$split_file" ]]; then
  echo "ERROR: episode split not found: $split_file" >&2
  exit 1
fi

if [[ "${FLOW_AUTO_RESUME:-false}" == "true" && -z "${FLOW_RESUME:-}" ]]; then
  latest_state=""
  if [[ -d "$OUTPUT_DIR" ]]; then
    latest_state="$(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type f \
      -path '*/checkpoint_*/training_state.pt' -print | sort | tail -1)"
  fi
  if [[ -n "$latest_state" ]]; then
    FLOW_RESUME="$latest_state"
    export FLOW_RESUME
    echo "[flow_matching] auto-resuming from $FLOW_RESUME"
  fi
fi

cmd=(
  "$PYTHON_BIN" -m model.subtask_policy_training.scripts.train_flow_matching
  --dataset-root "$TRAINING_VIEW_ROOT"
  --repo-id "$DATASET_REPO_ID"
  --split-file "$split_file"
  --output-dir "$OUTPUT_DIR"
  --steps "$TRAIN_STEPS"
  --batch-size "$TRAIN_BATCH_SIZE"
  --workers "${FLOW_WORKERS:-12}"
  --validation-freq "$TRAIN_EVAL_STEPS"
  --validation-samples "$TRAIN_MAX_EVAL_SAMPLES"
  --validation-action-samples "${FLOW_VALIDATION_ACTION_SAMPLES:-64}"
  --save-freq "$TRAIN_SAVE_FREQ"
  --log-freq "$TRAIN_LOG_FREQ"
  --action-horizon "$FLOW_ACTION_HORIZON"
  --n-action-steps "$FLOW_N_ACTION_STEPS"
  --flow-inference-steps "$FLOW_INFERENCE_STEPS"
  --model-dim "$FLOW_MODEL_DIM"
  --transformer-layers "$FLOW_TRANSFORMER_LAYERS"
  --transformer-heads "$FLOW_TRANSFORMER_HEADS"
  --wandb-project "$WANDB_PROJECT"
  --wandb-run-name "$JOB_NAME"
)
if [[ -n "$DATASET_REVISION" ]]; then
  cmd+=(--revision "$DATASET_REVISION")
fi
if [[ "$TRAIN_IMAGE_TRANSFORMS_ENABLE" != "true" ]]; then
  cmd+=(--no-image-augmentation)
fi
if [[ "$WANDB_ENABLE" == "true" ]]; then
  cmd+=(--wandb)
fi
if [[ -n "${FLOW_RESUME:-}" ]]; then
  cmd+=(--resume "$FLOW_RESUME")
fi
cmd+=("$@")

printf '[flow_matching] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
