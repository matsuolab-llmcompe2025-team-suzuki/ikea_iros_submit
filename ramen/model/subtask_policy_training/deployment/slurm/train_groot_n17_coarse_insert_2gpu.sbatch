#!/bin/bash
#SBATCH --job-name=groot-n17-coarse-insert
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --partition=debug

set -euo pipefail

find_repository_root() {
  local start candidate parent
  for start in "${RAMEN_REPOSITORY_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
    if [[ -z "$start" || ! -d "$start" ]]; then
      continue
    fi
    candidate="$(cd "$start" && pwd -P)"
    while true; do
      if [[ -e "$candidate/.git" \
        && -f "$candidate/model/subtask_policy_training/configs/subtask_training.json" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
      if [[ "$candidate" == "/" ]]; then
        break
      fi
      parent="$(dirname "$candidate")"
      if [[ "$parent" == "$candidate" ]]; then
        break
      fi
      candidate="$parent"
    done
  done
  return 1
}

if [[ "${RAMEN_IN_SINGULARITY:-false}" != "true" ]]; then
  if ! host_repository_root="$(find_repository_root)"; then
    echo "Could not locate the iros_2026_ramen checkout." >&2
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}" >&2
    echo "Submit from the repository or model/subtask_policy_training directory," >&2
    echo "or export RAMEN_REPOSITORY_ROOT=/absolute/path/to/iros_2026_ramen." >&2
    exit 2
  fi
  container_repository_root="/workspace/iros_2026_ramen"
  container_feature_root="$container_repository_root/model/subtask_policy_training"
  container_image="${CONTAINER_IMAGE:-$host_repository_root/gr00t-lerobot.sif}"

  if [[ ! -r "$container_image" ]]; then
    echo "Singularity image not found or unreadable: $container_image" >&2
    exit 2
  fi
  if [[ -n "${SINGULARITY_BIN:-}" ]]; then
    singularity_bin="$SINGULARITY_BIN"
  elif command -v singularity >/dev/null 2>&1; then
    singularity_bin="$(command -v singularity)"
  elif command -v apptainer >/dev/null 2>&1; then
    singularity_bin="$(command -v apptainer)"
  else
    echo "singularity/apptainer is not available on this Slurm node" >&2
    exit 2
  fi

  # Explicitly use the container runtime's environment channel for the W&B
  # credential. Avoid `--env WANDB_API_KEY=...`, which would expose the key in
  # the process command line.
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export SINGULARITYENV_WANDB_API_KEY="$WANDB_API_KEY"
    export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
    echo "wandb_api_key_forwarding: enabled"
  else
    echo "wandb_api_key_forwarding: unavailable in Slurm environment"
  fi

  echo "container_image: $container_image"
  echo "container_runtime: $singularity_bin"
  echo "container_bind: $host_repository_root -> $container_repository_root"
  exec "$singularity_bin" exec \
    --nv \
    --bind "$host_repository_root:$container_repository_root" \
    --pwd "$container_feature_root" \
    --env RAMEN_IN_SINGULARITY=true \
    "$container_image" \
    bash "$container_feature_root/deployment/slurm/train_groot_n17_coarse_insert_2gpu.sbatch" \
    "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEATURE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$FEATURE_ROOT"

# gr00t-lerobot.sif exposes the pinned LeRobot environment here. Sourcing it
# keeps the existing training wrapper unchanged while preventing host packages
# from leaking into the job.
export VENV_DIR="/opt/lerobot/.venv"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Container LeRobot environment is missing: $VENV_DIR" >&2
  exit 2
fi
source "$VENV_DIR/bin/activate"

# The source revision is pinned so a later dataset update cannot silently alter
# the training examples or the robot_q-to-SDK mapping used by this run.
export SUBTASK="coarse_insert"
export POLICY_TYPE="groot"
export DATASET_REPO_ID="Team-RAMEN/IROS2026_RAMEN_suzuki_coarse_insert_1"
export DATASET_REVISION="${DATASET_REVISION:-338c3db7f106e5108f29df5bb564a983897c0eca}"
export NUM_GPUS="${NUM_GPUS:-2}"
export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"
export ACCELERATE_DISTRIBUTED_STRATEGY="${ACCELERATE_DISTRIBUTED_STRATEGY:-fsdp}"
export ACCELERATE_MAIN_PROCESS_PORT="${ACCELERATE_MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

# LeRobot's batch size is per process. Three 480x640 camera streams make 8 or
# 16 samples exceed an A100 40GB before FSDP wrapping is applied correctly.
# Start with 4 x 2 GPUs while FULL_SHARD distributes model, gradient, and Adam
# state memory across both devices. Override it after measuring headroom.
export GROOT_BATCH_SIZE="${GROOT_BATCH_SIZE:-4}"
export GROOT_STEPS="${GROOT_STEPS:-20000}"
export GROOT_SAVE_FREQ="${GROOT_SAVE_FREQ:-5000}"
export TRAINING_VIEW_ROOT="${TRAINING_VIEW_ROOT:-outputs/training_views/groot_coarse_insert_sdk_ids_v2}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/groot_coarse_insert_2gpu}"
export JOB_NAME="${JOB_NAME:-groot_coarse_insert_2gpu}"
export POLICY_REPO_ID="${POLICY_REPO_ID:-Team-RAMEN/IROS2026_RAMEN_suzuki_coarse_insert_groot_1}"
wandb_enable_default="false"
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  wandb_enable_default="true"
fi
export WANDB_ENABLE="${WANDB_ENABLE:-$wandb_enable_default}"
export UPLOAD_AFTER_TRAIN="${UPLOAD_AFTER_TRAIN:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# `sbatch --export=WANDB_ENABLE=...,WANDB_PROJECT=...` omits WANDB_API_KEY.
# Submit with `--export=ALL,...` to carry the credential into Slurm; the outer
# branch above then explicitly forwards it into Singularity. Keep recording
# locally instead of aborting when neither the key nor a netrc login exists.
if [[ "$WANDB_ENABLE" == "true" && -z "${WANDB_MODE:-}" ]]; then
  wandb_netrc="${NETRC:-$HOME/.netrc}"
  if [[ -z "${WANDB_API_KEY:-}" ]] \
    && ! { [[ -r "$wandb_netrc" ]] && grep -q 'machine[[:space:]]\+api\.wandb\.ai' "$wandb_netrc"; }; then
    export WANDB_MODE="offline"
    echo "W&B credentials are not available in the job; recording in offline mode."
    echo "Sync later with: wandb sync <offline-run-directory>"
  fi
fi
if [[ "$WANDB_ENABLE" == "true" && -n "${WANDB_API_KEY:-}" ]]; then
  echo "wandb_api_key_in_container: set"
fi

threads_per_process=$((SLURM_CPUS_PER_TASK / NUM_GPUS))
if (( threads_per_process < 1 )); then
  threads_per_process=1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$threads_per_process}"

python scripts/verify_g1_sdk_joint_mapping.py

if [[ "${DRY_RUN:-false}" != "true" ]]; then
  python - <<'PY'
import os
import torch

expected = int(os.environ["NUM_GPUS"])
visible = torch.cuda.device_count()
if visible != expected:
    raise RuntimeError(f"Slurm allocated {visible} visible CUDA devices; expected {expected}")
for device_id in range(visible):
    props = torch.cuda.get_device_properties(device_id)
    memory_gb = props.total_memory / 10**9
    memory_gib = props.total_memory / 1024**3
    print(f"cuda:{device_id}: {props.name}, {memory_gb:.1f} GB ({memory_gib:.1f} GiB)")
    if memory_gb < 40:
        print(
            f"WARNING: cuda:{device_id} is below NVIDIA's recommended 40 GB VRAM class; "
            "reduce GROOT_BATCH_SIZE if training runs out of memory."
        )
PY
else
  echo "DRY_RUN=true: skipping CUDA device validation"
fi

./scripts/train_groot_n17.sh "$@"
