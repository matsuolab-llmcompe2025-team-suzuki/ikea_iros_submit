#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Issue #122: REPO_ROOT を導出 (`<repo>/model/subtask_policy_training` の 2 階層上)。
# MERGED_SOURCE_DIR / PYTHONPATH の repo-root anchor に使う (RAMEN-Ori config と path 統一)
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# Issue #139: env は 2 系統ある。
#   - GR00T: uv venv (`setup_env.sh` が作る .venv / .venv_lerobot060)。従来どおり source する。
#   - ACT / Diffusion Policy: pixi sub-workspace (`model/subtask_policy_training/pixi.toml`)。
#     `pixi run` 経由で呼ばれるので既に env の中におり、source する venv は無い。
# venv が無い場合は ambient python をそのまま使う (pixi env 想定)。
VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "$VENV_DIR" && -d ".venv_lerobot060" ]]; then
  VENV_DIR=".venv_lerobot060"
fi
if [[ -d "$VENV_DIR" ]]; then
  source "$VENV_DIR/bin/activate"
else
  VENV_DIR="(none: ambient python)"
fi
if ! python -c "import lerobot" >/dev/null 2>&1; then
  echo "ERROR: lerobot not importable. GR00T は scripts/setup_env.sh の venv、ACT/DP は" >&2
  echo "  cd model/subtask_policy_training && pixi run ... で起動すること" >&2
  exit 2
fi

NUM_GPUS="${NUM_GPUS:-1}"
if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 2
fi
ACCELERATE_DISTRIBUTED_STRATEGY="${ACCELERATE_DISTRIBUTED_STRATEGY:-ddp}"
if [[ ! "$ACCELERATE_DISTRIBUTED_STRATEGY" =~ ^(ddp|fsdp)$ ]]; then
  echo "ACCELERATE_DISTRIBUTED_STRATEGY must be ddp or fsdp, got: $ACCELERATE_DISTRIBUTED_STRATEGY" >&2
  exit 2
fi

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/subtask_training.json}"
TOLERANCE_S="${TOLERANCE_S:-0.001}"
eval "$(python scripts/resolve_training_config.py --config "$TRAIN_CONFIG" --format shell)"

echo "subtask: $SUBTASK"
echo "task: $TASK"
echo "dataset: $DATASET_REPO_ID"
echo "dataset_revision: ${DATASET_REVISION:-latest}"
echo "use_merged_hf_repo: $USE_MERGED_HF_REPO"
if [[ "$USE_MERGED_HF_REPO" == "true" ]]; then
  echo "merged_dataset_repo_id: $MERGED_DATASET_REPO_ID"
  echo "merged_component_repo_ids: $MERGED_COMPONENT_REPO_IDS"
fi
echo "policy: $POLICY_REPO_ID"
echo "output_dir: $OUTPUT_DIR"
echo "policy_type: $POLICY_TYPE"
echo "control_scope: $CONTROL_SCOPE"
echo "state_dim: $STATE_DIM"
echo "action_dim: $ACTION_DIM ($ACTION_SEMANTICS)"
echo "cameras: $CAMERAS"
echo "source_camera_map: $SOURCE_CAMERA_MAP"
echo "policy_view_layout: $POLICY_VIEW_LAYOUT"
echo "policy_input_features: $POLICY_INPUT_FEATURES"
echo "wandb_project: $WANDB_PROJECT"
echo "tolerance_s: $TOLERANCE_S"
echo "upload_after_train: $UPLOAD_AFTER_TRAIN"
echo "venv: $VENV_DIR"
echo "num_gpus: $NUM_GPUS"

# Issue #122 D-4: OBB overlay (C-11) hook を GR00T / lerobot-train 経由の policy に
# 効かせるための env 状態表示 + fail-fast validation。
# wrapper (lerobot_train_with_frame_cache) 内で setup_from_env が env を読んで register。
# RAMEN-Ori は自身の dataloader (data_lerobot.py) で config 経由 register するので
# ここでは無関係 (RAMEN-Ori は train.py 側から起動、本 script 経由しない)。
#
# 使用例 (GR00T + overlay):
#   OBB_OVERLAY_ENABLE=true \
#   OBB_PRECOMPUTED_ROOT=outputs/yolo_obb_cache \
#   SUBTASK=combined_task5_7 POLICY_TYPE=groot USE_MERGED_HF_REPO=true \
#     bash model/subtask_policy_training/scripts/train_lerobot.sh
#
# option env (未指定は D-3 preview 決定 default): OBB_PRECOMPUTED_HASH / OBB_OVERLAY_CAMERAS
# / OBB_OVERLAY_TOP_K / OBB_OVERLAY_CONF (0.30) / OBB_OVERLAY_THICKNESS (2) /
# OBB_OVERLAY_CLASS_FILTER (JSON list、例 "[3,4]" = hole+table_top のみ)
OBB_OVERLAY_ENABLE_VAL="${OBB_OVERLAY_ENABLE:-false}"
echo "obb_overlay_enable: $OBB_OVERLAY_ENABLE_VAL"
if [[ "$OBB_OVERLAY_ENABLE_VAL" == "true" || "$OBB_OVERLAY_ENABLE_VAL" == "1" || "$OBB_OVERLAY_ENABLE_VAL" == "yes" ]]; then
  if [[ -z "${OBB_PRECOMPUTED_ROOT:-}" ]]; then
    echo "ERROR: OBB_OVERLAY_ENABLE=true requires OBB_PRECOMPUTED_ROOT (precompute cache dir)" >&2
    echo "  例: OBB_PRECOMPUTED_ROOT=outputs/yolo_obb_cache" >&2
    exit 2
  fi
  echo "obb_precomputed_root:  $OBB_PRECOMPUTED_ROOT"
  echo "obb_precomputed_hash:  ${OBB_PRECOMPUTED_HASH:-<auto-pick>}"
  echo "obb_overlay_cameras:   ${OBB_OVERLAY_CAMERAS:-<default: cam_0, cam_1>}"
  echo "obb_overlay_conf:      ${OBB_OVERLAY_CONF:-0.30}"
  echo "obb_overlay_thickness: ${OBB_OVERLAY_THICKNESS:-2}"
  echo "obb_overlay_class_filter: ${OBB_OVERLAY_CLASS_FILTER:-<null = 全 class>}"
fi

policy_args=()
prepare_groot_base_cmd=()
GROOT_CANONICAL_BASE_MODEL_PATH=""
GROOT_RUNTIME_BASE_MODEL_PATH=""
if [[ "$POLICY_TYPE" == "groot" ]]; then
  GROOT_CANONICAL_BASE_MODEL_PATH="$GROOT_BASE_MODEL_PATH"
  GROOT_RUNTIME_BASE_MODEL_PATH="$GROOT_PROCESSOR_OVERLAY_ROOT"
  prepare_groot_base_cmd=(
    python scripts/prepare_groot_n17_real_g1_overlay.py
    --model-path "$GROOT_CANONICAL_BASE_MODEL_PATH"
    --revision "$GROOT_BASE_MODEL_REVISION"
    --output-root "$GROOT_PROCESSOR_OVERLAY_ROOT"
  )
  if [[ "${DRY_RUN:-false}" != "true" ]]; then
    GROOT_LEROBOT_SOURCE_OVERLAY_ROOT="${GROOT_LEROBOT_SOURCE_OVERLAY_ROOT:-outputs/lerobot_source_overlays/groot_relative_eef_v3}"
    python scripts/patch_lerobot_groot_relative_eef.py \
      --overlay-root "$GROOT_LEROBOT_SOURCE_OVERLAY_ROOT"
    GROOT_LEROBOT_SOURCE_OVERLAY_ROOT="$(realpath "$GROOT_LEROBOT_SOURCE_OVERLAY_ROOT")"
    export PYTHONPATH="$GROOT_LEROBOT_SOURCE_OVERLAY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    python scripts/patch_lerobot_groot_relative_eef.py --check
    GROOT_RUNTIME_BASE_MODEL_PATH="$("${prepare_groot_base_cmd[@]}")"
  fi
  echo "groot_base_model: $GROOT_CANONICAL_BASE_MODEL_PATH@$GROOT_BASE_MODEL_REVISION"
  echo "groot_runtime_overlay: $GROOT_RUNTIME_BASE_MODEL_PATH"
  echo "groot_embodiment_tag: $GROOT_EMBODIMENT_TAG"
  echo "groot_relative_eef_processor: $GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR"
  echo "groot_lerobot_source_overlay: ${GROOT_LEROBOT_SOURCE_OVERLAY_ROOT:-not prepared in dry-run}"
  echo "groot_dataset_source: shared LeRobot v3 DATASET_REPO_ID"
  policy_args+=(
    --dataset.image_transforms.enable="$GROOT_IMAGE_TRANSFORMS_ENABLE"
    --policy.base_model_path="$GROOT_RUNTIME_BASE_MODEL_PATH"
    --policy.embodiment_tag="$GROOT_EMBODIMENT_TAG"
    --policy.chunk_size="$GROOT_CHUNK_SIZE"
    --policy.n_action_steps="$GROOT_N_ACTION_STEPS"
    --policy.use_relative_actions="$GROOT_USE_RELATIVE_ACTIONS"
    --policy.relative_exclude_joints="$GROOT_RELATIVE_EXCLUDE_JOINTS"
    --policy.use_bf16="$GROOT_USE_BF16"
    --policy.max_steps="$GROOT_STEPS"
    --batch_size="$GROOT_BATCH_SIZE"
    --steps="$GROOT_STEPS"
    --save_freq="$GROOT_SAVE_FREQ"
    --env_eval_freq="$GROOT_ENV_EVAL_FREQ"
    --eval_steps="$GROOT_EVAL_STEPS"
    --max_eval_samples="$GROOT_MAX_EVAL_SAMPLES"
    --log_freq="$GROOT_LOG_FREQ"
  )
else
  policy_args+=(
    --dataset.image_transforms.enable="$TRAIN_IMAGE_TRANSFORMS_ENABLE"
    --batch_size="$TRAIN_BATCH_SIZE"
    --steps="$TRAIN_STEPS"
    --save_freq="$TRAIN_SAVE_FREQ"
    --eval_steps="$TRAIN_EVAL_STEPS"
    --max_eval_samples="$TRAIN_MAX_EVAL_SAMPLES"
    --log_freq="$TRAIN_LOG_FREQ"
  )
  if [[ "$POLICY_TYPE" == "act" ]]; then
    policy_args+=(
      --policy.chunk_size="$ACT_CHUNK_SIZE"
      --policy.n_action_steps="$ACT_N_ACTION_STEPS"
    )
  fi
  if [[ "$POLICY_TYPE" == "diffusion" ]]; then
    policy_args+=(
      --policy.horizon="$DIFFUSION_HORIZON"
      --policy.n_action_steps="$DIFFUSION_N_ACTION_STEPS"
      --policy.n_obs_steps="$DIFFUSION_N_OBS_STEPS"
      --policy.num_inference_steps="$DIFFUSION_NUM_INFERENCE_STEPS"
      --policy.drop_n_last_frames="$DIFFUSION_DROP_N_LAST_FRAMES"
    )
  fi
fi

# Issue #122: multi-repo subtask (DATASET_REPO_IDS list に 2+ 要素) は pre-merge が必要。
# DATASET_REPO_IDS が JSON list "[..]"、length を Python で判定 (jq 依存回避)。
# D-1: USE_MERGED_HF_REPO=true 時は skill-merged repo (chunk-merged と layout 同じ、名前だけ違う)
# を pull → local union する。cache dir は namespace 分離 (chunk-merged と衝突回避)。
prepare_multi_repo_cmd=()
merged_source_root=""
if [[ -n "${DATASET_REPO_IDS:-}" ]]; then
  N_REPOS=$(python3 -c "import json,sys; print(len(json.loads(sys.argv[1])))" "$DATASET_REPO_IDS")
  if [[ "$N_REPOS" -gt 1 ]]; then
    # Issue #122: MERGED_SOURCE_DIR を repo-root anchored に (RAMEN-Ori config が
    # 同じ path を指す、cache 共有成立、symlink workaround 不要)。User override
    # (MERGED_SOURCE_DIR=<absolute> or relative) は :- pattern で保持。
    if [[ "$USE_MERGED_HF_REPO" == "true" ]]; then
      MERGED_SOURCE_DIR="${MERGED_SOURCE_DIR:-${REPO_ROOT}/outputs/merged_sources/${SUBTASK}_from_hf}"
    else
      MERGED_SOURCE_DIR="${MERGED_SOURCE_DIR:-${REPO_ROOT}/outputs/merged_sources/${SUBTASK}}"
    fi
    echo "multi_repo_count: $N_REPOS"
    echo "merged_source_dir: $MERGED_SOURCE_DIR"
    prepare_multi_repo_cmd=(
      python scripts/prepare_multi_repo_source.py
      --repo-ids "$DATASET_REPO_IDS"
      --output-dir "$MERGED_SOURCE_DIR"
    )
    if [[ -n "$DATASET_REVISION" ]]; then
      prepare_multi_repo_cmd+=(--revision "$DATASET_REVISION")
    fi
    if [[ "${MERGED_SOURCE_FORCE:-false}" == "true" ]]; then
      prepare_multi_repo_cmd+=(--force)
    fi
    merged_source_root="$MERGED_SOURCE_DIR"
  fi
fi

dataset_args=(--dataset.repo_id="$DATASET_REPO_ID")
prepare_training_view_cmd=()
if [[ "$MATERIALIZE_TRAINING_VIEW" == "true" ]]; then
  echo "training_view_root: $TRAINING_VIEW_ROOT"
  echo "training_view_force: $TRAINING_VIEW_FORCE"
  prepare_training_view_cmd=(
    python scripts/materialize_lerobot_training_view.py
    --config "$TRAIN_CONFIG"
    --repo-id "$DATASET_REPO_ID"
    --output-root "$TRAINING_VIEW_ROOT"
    --policy-type "$POLICY_TYPE"
  )
  if [[ -n "$DATASET_REVISION" ]]; then
    prepare_training_view_cmd+=(--revision "$DATASET_REVISION")
  fi
  # Issue #122: multi-repo 時は pre-merge した dir を source-root として渡す
  if [[ -n "$merged_source_root" ]]; then
    prepare_training_view_cmd+=(--source-root "$merged_source_root")
    prepare_training_view_cmd+=(--allow-multi-task)
  elif [[ -n "${SOURCE_DATASET_ROOT:-}" ]]; then
    prepare_training_view_cmd+=(--source-root "$SOURCE_DATASET_ROOT")
  fi
  # Issue #122: 外部 split JSON (apples-to-apples 用、RAMEN-Ori 側 export)
  if [[ -n "${SPLIT_JSON:-}" ]]; then
    prepare_training_view_cmd+=(--split-json "$SPLIT_JSON")
  fi
  if [[ "$TRAINING_VIEW_FORCE" == "true" ]]; then
    prepare_training_view_cmd+=(--force)
  fi
  dataset_args+=(--dataset.root="$TRAINING_VIEW_ROOT")
fi

# Issue #129: 前 run の $OUTPUT_DIR 残骸 (ckpts) があると LeRobot cfg.validate() が
# "Output directory ... already exists and resume is False" で fail。fresh session から
# 同 SUBTASK を再走行する時に頻発 (前 run の ckpt が HF autopush 済で local に残ってる pattern)。
# 明示 opt-in の OUTPUT_DIR_FORCE=true でのみ rm、それ以外は fail-fast + 復旧手順を案内。
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]]; then
  if [[ "${OUTPUT_DIR_FORCE:-false}" == "true" ]]; then
    echo "[train_lerobot] OUTPUT_DIR_FORCE=true → removing existing $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
  else
    # ${BASH_SOURCE[0]} は本 script の実体 path。$0 だと nohup + kick script 経由の
    # invocation で kick script path が入り、operator が意図と違う script を叩く事故になる。
    echo "ERROR: $OUTPUT_DIR is non-empty (前 run の ckpt が残ってる可能性)。" >&2
    echo "  ckpts が HF autopush 済であることを確認してから:" >&2
    echo "    OUTPUT_DIR_FORCE=true bash ${BASH_SOURCE[0]}" >&2
    echo "  で明示 rm + 再走行してください。" >&2
    exit 2
  fi
fi

# Issue #122: JPG frame cache 経由で decode を 10-30x 高速化するため lerobot-train CLI を
# lerobot_train_with_frame_cache wrapper に差し替え。
# LEROBOT_FRAME_CACHE_ENABLE=true が未 export なら wrapper は元 decode に full fallback
# = 挙動不変。default true (無効化したいなら LEROBOT_FRAME_CACHE_ENABLE=false で override)。
export LEROBOT_FRAME_CACHE_ENABLE="${LEROBOT_FRAME_CACHE_ENABLE:-true}"
# cwd=$ROOT_DIR から module 起動できるよう repo root を追加し、GR00T processor
# overlay など既存の PYTHONPATH は保持する。
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
train_module="model.subtask_policy_training.scripts.lerobot_train_with_frame_cache"
train_launcher=(python -m "$train_module")
if (( NUM_GPUS > 1 )); then
  if [[ "$DEVICE" != cuda* ]]; then
    echo "multi-GPU training requires DEVICE=cuda, got: $DEVICE" >&2
    exit 2
  fi
  if ! command -v accelerate >/dev/null 2>&1; then
    echo "accelerate is required for NUM_GPUS=$NUM_GPUS" >&2
    exit 2
  fi
  if [[ "$POLICY_TYPE" == "groot" ]]; then
    default_mixed_precision="bf16"
  else
    default_mixed_precision="no"
  fi
  ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-$default_mixed_precision}"
  if [[ ! "$ACCELERATE_MIXED_PRECISION" =~ ^(no|fp16|bf16|fp8)$ ]]; then
    echo "invalid ACCELERATE_MIXED_PRECISION: $ACCELERATE_MIXED_PRECISION" >&2
    exit 2
  fi
  train_launcher=(accelerate launch)
  if [[ "$ACCELERATE_DISTRIBUTED_STRATEGY" == "fsdp" ]]; then
    train_launcher+=(
      --use_fsdp
      --fsdp_version=1
      --fsdp_sharding_strategy=FULL_SHARD
      --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP
      --fsdp_transformer_layer_cls_to_wrap=Qwen3VLTextDecoderLayer,Qwen3VLVisionBlock,BasicTransformerBlock
      --fsdp_use_orig_params=true
      --fsdp_offload_params=false
      --fsdp_backward_prefetch=NO_PREFETCH
      --fsdp_forward_prefetch=false
    )
  else
    train_launcher+=(--multi_gpu)
  fi
  train_launcher+=(
    --num_processes="$NUM_GPUS"
    --num_machines=1
    --mixed_precision="$ACCELERATE_MIXED_PRECISION"
  )
  if [[ -n "${ACCELERATE_MAIN_PROCESS_PORT:-}" ]]; then
    if [[ ! "$ACCELERATE_MAIN_PROCESS_PORT" =~ ^[0-9]+$ ]] \
      || (( ACCELERATE_MAIN_PROCESS_PORT < 1024 || ACCELERATE_MAIN_PROCESS_PORT > 65535 )); then
      echo "ACCELERATE_MAIN_PROCESS_PORT must be in [1024, 65535]" >&2
      exit 2
    fi
    train_launcher+=(--main_process_port="$ACCELERATE_MAIN_PROCESS_PORT")
  fi
  train_launcher+=(--module "$train_module")
  echo "accelerate_distributed_strategy: $ACCELERATE_DISTRIBUTED_STRATEGY"
  echo "accelerate_mixed_precision: $ACCELERATE_MIXED_PRECISION"
fi

wandb_args=(
  --wandb.enable="$WANDB_ENABLE"
  --wandb.project="$WANDB_PROJECT"
)
if [[ -n "${WANDB_MODE:-}" ]]; then
  wandb_args+=(--wandb.mode="$WANDB_MODE")
fi

cmd=(
  "${train_launcher[@]}"
  "${dataset_args[@]}"
  --policy.type="$POLICY_TYPE"
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --policy.device="$DEVICE"
  "${wandb_args[@]}"
  --policy.repo_id="$POLICY_REPO_ID"
  --policy.push_to_hub="$PUSH_TO_HUB"
  --policy.private="$PRIVATE"
  --policy.input_features="$POLICY_INPUT_FEATURES"
  --policy.output_features="$POLICY_OUTPUT_FEATURES"
  --tolerance_s="$TOLERANCE_S"
  "${policy_args[@]}"
  "$@"
)

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  if [[ "${#prepare_groot_base_cmd[@]}" -gt 0 ]]; then
    printf "prepare_groot_base:"
    printf " %q" "${prepare_groot_base_cmd[@]}"
    printf "\n"
  fi
  if [[ "${#prepare_multi_repo_cmd[@]}" -gt 0 ]]; then
    printf "prepare_multi_repo:"
    printf " %q" "${prepare_multi_repo_cmd[@]}"
    printf "\n"
  fi
  if [[ "${#prepare_training_view_cmd[@]}" -gt 0 ]]; then
    printf "prepare_training_view:"
    printf " %q" "${prepare_training_view_cmd[@]}"
    printf "\n"
    printf "resolve_training_split: python scripts/resolve_training_split.py --dataset-root %q --format shell\n" \
      "$TRAINING_VIEW_ROOT"
  fi
  # Issue #122: dry-run 時も frame_cache precompute command を表示 (実 run 時に走る想定)
  if [[ "${LEROBOT_FRAME_CACHE_ENABLE}" == "true" && "${FRAME_CACHE_PRECOMPUTE:-true}" == "true" ]]; then
    if [[ -n "$merged_source_root" ]]; then
      dry_frame_cache_target="$merged_source_root"
    else
      dry_frame_cache_target="$TRAINING_VIEW_ROOT"
    fi
    printf "precompute_frame_cache: python %q --lerobot-root %q\n" \
      "$ROOT_DIR/../../data/bitrobot_lerobot_subtask_datasets/scripts/precompute_frame_cache.py" \
      "$dry_frame_cache_target"
  fi
  printf "command (LEROBOT_FRAME_CACHE_ENABLE=%s, OBB_OVERLAY_ENABLE=%s):" \
    "$LEROBOT_FRAME_CACHE_ENABLE" "$OBB_OVERLAY_ENABLE_VAL"
  printf " %q" "${cmd[@]}"
  printf "\n"
  if [[ "$UPLOAD_AFTER_TRAIN" == "true" ]]; then
    upload_cmd=(
      python scripts/upload_policy.py
      --repo-id "$POLICY_REPO_ID"
      --output-dir "$OUTPUT_DIR"
      --commit-message "Upload $POLICY_TYPE $SUBTASK LeRobot checkpoint"
    )
    if [[ "$PRIVATE" == "true" ]]; then
      upload_cmd+=(--private)
    fi
    printf "post_train_upload:"
    printf " %q" "${upload_cmd[@]}"
    printf "\n"
  fi
  exit 0
fi

if [[ "${#prepare_multi_repo_cmd[@]}" -gt 0 ]]; then
  # Issue #122: multi-repo は materialize より先に snapshot_download + merge
  "${prepare_multi_repo_cmd[@]}"
fi

if [[ "${#prepare_training_view_cmd[@]}" -gt 0 ]]; then
  "${prepare_training_view_cmd[@]}"
fi

# Issue #122: JPG frame cache 事前展開 (LEROBOT_FRAME_CACHE_ENABLE=true 前提)。
# Cache は元 mp4 の場所 (multi-repo なら $merged_source_root、single-repo なら
# $TRAINING_VIEW_ROOT の symlinked videos) に置く → policy_type 越しに共有可能。
# precompute script が settings hash 検証 + per-mp4 verify + auto mp4 drop を実施。
# 24 core 並列で 200k frames × 3 cam ≒ 5-10 min。skip したい場合は
# FRAME_CACHE_PRECOMPUTE=false で明示。
if [[ "${LEROBOT_FRAME_CACHE_ENABLE}" == "true" && "${FRAME_CACHE_PRECOMPUTE:-true}" == "true" ]]; then
  # Cache target: multi-repo → merged_source (policy 越し共有可能)、single-repo → training view
  if [[ -n "$merged_source_root" ]]; then
    frame_cache_target="$merged_source_root"
  else
    frame_cache_target="$TRAINING_VIEW_ROOT"
  fi
  echo "[frame_cache] target root: $frame_cache_target (共有 cache = policy_type 越しに再利用可能)"
  precompute_cmd=(
    python "$ROOT_DIR/../../data/bitrobot_lerobot_subtask_datasets/scripts/precompute_frame_cache.py"
    --lerobot-root "$frame_cache_target"
  )
  "${precompute_cmd[@]}"
fi

eval "$(python scripts/resolve_training_split.py --dataset-root "$TRAINING_VIEW_ROOT" --format shell)"
echo "split_sha256: $SPLIT_SHA256"
echo "episodes: train=$TRAIN_EPISODE_COUNT validation=$VALIDATION_EPISODE_COUNT test=$TEST_EPISODE_COUNT"
cmd+=(
  --dataset.episodes="$DATASET_EPISODES_JSON"
  --dataset.eval_split="$DATASET_EVAL_SPLIT"
)

"${cmd[@]}"

if [[ "$POLICY_TYPE" == "groot" ]]; then
  python scripts/restore_groot_base_model_path.py \
    --output-dir "$OUTPUT_DIR" \
    --runtime-path "$GROOT_RUNTIME_BASE_MODEL_PATH" \
    --canonical-path "$GROOT_CANONICAL_BASE_MODEL_PATH"
fi

if [[ "$UPLOAD_AFTER_TRAIN" == "true" ]]; then
  upload_cmd=(
    python scripts/upload_policy.py
    --repo-id "$POLICY_REPO_ID"
    --output-dir "$OUTPUT_DIR"
    --commit-message "Upload $POLICY_TYPE $SUBTASK LeRobot checkpoint"
  )
  if [[ "$PRIVATE" == "true" ]]; then
    upload_cmd+=(--private)
  fi
  "${upload_cmd[@]}"
fi
