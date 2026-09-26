# Subtask Policy Training

This folder contains lightweight wrappers for training LeRobot policies on Team RAMEN IROS 2026 subtask datasets uploaded to Hugging Face.

The default config trains `pick_leg`, but the scripts are subtask-generic. Change `subtask` in `configs/subtask_training.json`, or pass `SUBTASK=...` at runtime.

The training environment is pinned to LeRobot `0.6.0` with the `dataset`, `training`, and `groot` extras. LeRobot `0.6.0` uses the LeRobotDataset codebase tag `v3.0` on Hugging Face datasets and integrates GR00T N1.7 through `--policy.type=groot`.

## Dataset Source

Shared LeRobotDataset v3 repos:

```text
Team-RAMEN/IROS2026_RAMEN_suzuki_{subtask}_1
```

These repos are split from `BitRobot/G1_WBT_Dex1_Building-Children-Table`
without renaming cameras or concatenating state/action. They preserve the
official BitRobot LeRobot v3 schema so the same datasets can be reused by other
models. This training folder materializes a local model-facing view immediately
before training.

Supported subtasks:

```text
move_to_work_pose
pick_leg
coarse_insert
final_insert_contact
tighten
rotate_table_base
recover_or_regrasp
flip_table
```

## Interface

The uploaded shared datasets use the official source keys:

- FPS: `30`
- RGB videos: `observation.images.cam_0`, `cam_1`, `cam_2`, `cam_3`
- IR videos: `observation.images.cam_0_ir` ... `cam_3_ir`
- State keys: `observation.state.ee_state` [12],
  `observation.state.robot_q_current` [36], `observation.state.hand_state` [2]
- Action keys: `action.ee_action` [12], `action.robot_q_desired` [36],
  `action.hand_cmd` [2]

Training uses a local view with the standard 3-camera policy layout:

```text
observation.images.head_left   <- observation.images.cam_0
observation.images.left_wrist  <- observation.images.cam_2
observation.images.right_wrist <- observation.images.cam_3
```

`cam_0` is the head stereo left eye according to `docs/dataset/dataset.md` and
`docs/inference/vit_phase1.md`. `cam_1` is the head right eye and is ignored for
the standard policy input. IR videos are also ignored for policy training.

For ACT and Diffusion Policy, the local view materializes an upper-body-only
non-walking policy interface:

- `observation.state`: 19-D upper-body robot configuration state.
- `action`: 19-D upper-body absolute robot configuration target action.

State order:

```text
0..2    waist_yaw/roll/pitch
3..9    left shoulder/elbow/wrist joints
10..16  right shoulder/elbow/wrist joints
17      left_gripper_q
18      right_gripper_q
```

Action order:

```text
0..2    waist_yaw/roll/pitch targets
3..9    left shoulder/elbow/wrist targets
10..16  right shoulder/elbow/wrist targets
17      left_gripper_q_cmd
18      right_gripper_q_cmd
```

Root pose and lower-body joints are deliberately dropped for ACT/Diffusion
training. During execution the policy should not command walking or leg joints;
the G1 lower-body/balance controller remains responsible for standing.

The action is an absolute upper-body target in the dataset. If a residual
action-chunk baseline is used, convert to residuals in the training adapter and
add the residual back to the current state/action reference at inference time.

`robot_q_current` and `robot_q_desired` follow the BitRobot LeRobot convention:
the first 7 values are root pose `(x,y,z,w,x,y,z)`, and the remaining 29 values
are robot joint positions/targets. The two gripper values are stored separately.

The two EEF fields are left pose followed by right pose. Each pose is
`[x, y, z, roll, pitch, yaw]` in the root-link frame, with Euler XYZ angles in
radians. The dataset card identifies these as root-to-EEF FK poses; the mapping
also rejects any config that does not declare this exact convention.

For GR00T N1.7, the local view uses the base checkpoint's official
`real_g1_relative_eef_relative_joints` contract:

```text
state  49D: left EEF 9, right EEF 9, left hand 7, right hand 7,
            left arm 7, right arm 7, waist 3
action 53D: the same groups, then base height 1 and navigation 3
```

Source EEF Euler poses are converted to absolute `XYZ_ROT6D` using the first
two rows of the rotation matrix. Source arm and waist targets remain absolute
in the materialized parquet. During training, after LeRobot has assembled the
complete action chunk, the GR00T processor applies the checkpoint contract:

- EEF: `T_relative = inverse(T_current) @ T_target` for every future target,
  always anchored to the chunk's current observation.
- Arms: `q_target - q_current` for every future target.
- Hands and waist: absolute targets.
- Base-height and navigation: zero because this non-walking dataset has no such
  commands, and excluded from the loss. No leg or root-pose slots are exposed
  to the policy.

The Dex1-1 open/close scalar is expanded into the fixed official 7-D G1 hand
synergy for each side. The real-robot inference adapter projects the predicted
7-D hand group back onto the same synergy with a least-squares inverse. This
keeps physical observations and training actions in one distribution; zero
padding the six remaining coordinates is not a compatible substitute. The
bridge is recorded in `meta/modality.json`. The shared HF dataset itself is
unchanged and does not need re-uploading.

## Setup

Run from this directory:

```bash
./scripts/setup_env.sh
source .venv/bin/activate
hf auth login
```

Setup applies a hash-guarded compatibility patch to exactly LeRobot `0.6.0`.
Upstream 0.6.0 can decode relative EEF actions but rejects relative EEF during
training; the patch adds the same SE(3) conversion to its training processor and
refuses to modify an unknown LeRobot source version.

## Select A Subtask

Option 1: edit `configs/subtask_training.json`:

```json
{
  "subtask": "pick_leg"
}
```

Option 2: override at runtime:

```bash
SUBTASK=tighten ./scripts/train_lerobot.sh
```

To inspect the resolved training settings:

```bash
SUBTASK=pick_leg POLICY_TYPE=act \
python scripts/resolve_training_config.py --config configs/subtask_training.json --format json
```

## Train

Before training a newly generated subtask dataset, run the structural and
numerical audit. It checks source-episode provenance, contiguous indices,
timestamps, finite state/action values, camera metadata, Hub video inventory,
nearly static episodes, and writes a leakage-safe 80/10/10 split grouped by the
original recording:

```bash
python scripts/audit_lerobot_dataset.py \
  --repo-id Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1
```

The default reports are written under `outputs/audits/`. A warning about the
missing explicit success label is expected until sampled videos have been
reviewed; it must not be interpreted as proof that all sliced demonstrations
are successful.

ACT baseline:

```bash
SUBTASK=pick_leg \
POLICY_TYPE=act \
./scripts/train_lerobot.sh
```

The ACT default is an explicit `100`-target chunk with `10` executed targets,
batch size `8`, `300000` optimizer steps, train-only image augmentation, and
offline validation every `10000` steps. Override these with the corresponding
`ACT_*` or `TRAIN_*` environment variables only after a smoke run confirms the
new memory and timing contract.

Diffusion Policy baseline:

```bash
SUBTASK=pick_leg \
POLICY_TYPE=diffusion \
./scripts/train_lerobot.sh
```

Flow Matching behavior-cloning baseline for `flip_table`:

```bash
SUBTASK=flip_table ./scripts/train_flow_matching.sh
```

This repository-native policy uses the same three-camera, 19-D state, and 19-D
absolute-target contract as ACT. The default model predicts a 24-target chunk
and executes 6 targets before replanning at the dataset's 30 Hz rate. Raw camera
frames remain 640x480 at the runtime boundary and are resized deterministically
inside the model. Training includes image augmentation, grouped train/validation
splits, periodic held-out validation, atomic checkpoints, and optional W&B
logging.

Evaluate a checkpoint only on the held-out validation or test split:

```bash
python -m model.subtask_policy_training.scripts.evaluate_flow_matching \
  --checkpoint outputs/train/flow_matching_flip_table/checkpoint_00075000 \
  --dataset-root outputs/training_views/flow_matching_flip_table \
  --split-file outputs/training_views/flow_matching_flip_table/meta/team_ramen_episode_split.json \
  --split test \
  --output outputs/evaluations/flow_matching_flip_table_test.json
```

The Flow checkpoint can be evaluated directly in simulation or used as the
frozen base policy for
`model/flip_table_reinforcement_learning/run_train_in_container.sh train_rlpd`.

GR00T N1.7 baseline through LeRobot `0.6.0`:

```bash
SUBTASK=pick_leg \
./scripts/train_groot_n17.sh
```

For the pinned `coarse_insert` dataset on one Slurm node with two GPUs:

```bash
sbatch deployment/slurm/train_groot_n17_coarse_insert_2gpu.sbatch
```

The batch resolves the checkout from Slurm's `SLURM_SUBMIT_DIR`, because Slurm
copies batch scripts into its spool directory before execution. Submit from the
repository root or this `model/subtask_policy_training` directory. When passing
W&B settings, keep the rest of the submission environment (including tokens)
with `ALL`:

```bash
sbatch \
  --export=ALL,WANDB_ENABLE=true,WANDB_PROJECT=iros2026-ramen-coarse-insert \
  deployment/slurm/train_groot_n17_coarse_insert_2gpu.sbatch
```

If `WANDB_ENABLE=true` is exported without `WANDB_API_KEY` and the submitting
user has no W&B entry in `~/.netrc`, the batch records an offline W&B run rather
than aborting. After logging in on a machine with network access, upload it with
`wandb sync <offline-run-directory>`. Use `--export=ALL,...` to pass an existing
`WANDB_API_KEY` and log online immediately.
The batch forwards the key through `SINGULARITYENV_WANDB_API_KEY` without
placing its value in the Singularity command line.

The batch requests two GPUs and 16 CPU cores on the `debug` partition. It uses
LeRobot's Accelerate integration with two processes and BF16. The default
per-process batch is 4, giving a global batch of 8. This conservative setting
targets three 480x640 camera streams on A100 40GB; larger per-GPU batches may
reach the memory limit. The job executes inside
the repository-root `gr00t-lerobot.sif`; it does not create or use a host
`.venv`. The repository is bind-mounted read/write at
`/workspace/iros_2026_ramen`, so training views and checkpoints persist on the
host. The host home directory remains available for Hugging Face caches and
credentials.

Because the SIF is read-only, the job copies the small LeRobot Python package
into `outputs/lerobot_source_overlays/groot_relative_eef_v3`, applies the
checked relative-EEF training patch there, and prepends that directory to
`PYTHONPATH`. The container installation and the SIF itself remain unchanged.
The separately gated Cosmos-Reason2 processor assets are read from its public
Qwen3-VL-2B-Instruct base; all tokenizer, chat-template, and image/video
processor files have identical Hugging Face Git blob IDs in the two repos.

Multi-GPU training uses FSDP `FULL_SHARD` with transformer-block wrapping: each
GPU processes a disjoint batch of 4, while parameters, gradients, and FP32 Adam
states are sharded across both GPUs. This gives a global batch of 8 and avoids
duplicating the optimizer memory, but does not combine two 40 GB GPUs into a
generic 80 GB memory pool. An A100 40GB may be
reported by CUDA as roughly 39.4 GiB; it is still a 40 GB-class GPU and is
accepted by the batch. GPUs below the recommended 40 GB class produce a warning
instead of being rejected.

Cluster-specific resource flags and runtime overrides can be supplied at
submission time:

```bash
sbatch \
  --partition=<gpu-partition> \
  --account=<project> \
  --gres=gpu:a100:2 \
  --export=ALL,GROOT_BATCH_SIZE=4,WANDB_ENABLE=true \
  deployment/slurm/train_groot_n17_coarse_insert_2gpu.sbatch
```

Use `CONTAINER_IMAGE` if the SIF is staged outside the repository, and
`SINGULARITY_BIN` if the cluster requires a particular Singularity/Apptainer
binary:

```bash
sbatch \
  --export=ALL,CONTAINER_IMAGE=/shared/containers/gr00t-lerobot.sif \
  deployment/slurm/train_groot_n17_coarse_insert_2gpu.sbatch
```

The job requires access to the private Team RAMEN dataset and the gated NVIDIA
backbone through the submitting user's existing Hugging Face authentication. It
does not upload the resulting policy by default; set
`--export=ALL,UPLOAD_AFTER_TRAIN=true` when the final checkpoint should be
published.

Before downloading or training, the batch runs:

```bash
python scripts/verify_g1_sdk_joint_mapping.py
```

This compares the training mapping with the runtime bridge's complete
`G1_JOINT_NAMES` table. The source vectors use
`dataset_robot_q_index = 7 + sdk_motor_id`: root pose occupies indices `0..6`,
and SDK motor IDs `0..28` occupy dataset indices `7..35`. The materializer then
selects SDK IDs `12..14` (waist), `15..21` (left arm), and `22..28` (right arm)
for the GR00T REAL_G1 slots. Legs remain available in the verified source
mapping but are intentionally excluded from this non-walking manipulation
policy. Dex1 open/close values come from the separate two-value hand fields.

The general training wrapper also accepts `NUM_GPUS`. To invoke it manually in
the same container:

```bash
cd /home/naoki.takada/workspace/iros_2026_ramen
SUBTASK=coarse_insert NUM_GPUS=2 GROOT_BATCH_SIZE=4 \
singularity exec --nv \
  --bind "$PWD:/workspace/iros_2026_ramen" \
  --pwd /workspace/iros_2026_ramen/model/subtask_policy_training \
  gr00t-lerobot.sif \
  bash -lc 'VENV_DIR=/opt/lerobot/.venv ./scripts/train_groot_n17.sh'
```

This uses the shared LeRobot v3 dataset (`DATASET_REPO_ID`) and materializes the
local REAL_G1 relative-EEF view before launching `lerobot-train`. The first run
downloads the pinned base model and builds a local sidecar overlay. The overlay
preserves the official state/action contract, statistics, and embodiment id;
only the video keys are changed to the required current-frame three-camera
layout. The wrapper injects these defaults:

```text
--policy.type=groot
--policy.base_model_path=nvidia/GR00T-N1.7-3B
base revision: 2fc962b973bccdd5d8ce4f67cc63b264d6886495
--policy.embodiment_tag=real_g1_relative_eef_relative_joints
--policy.chunk_size=16
--policy.n_action_steps=16
--policy.use_relative_actions=true
--policy.use_bf16=true
--dataset.image_transforms.enable=true
```

Chunk length and execution length can still be overridden when needed:

```bash
GROOT_CHUNK_SIZE=40 \
GROOT_N_ACTION_STEPS=25 \
./scripts/train_groot_n17.sh --steps=1000
```

`GROOT_USE_RELATIVE_ACTIONS` and
`GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR` must remain true. The resolver
fails before training if either is disabled or if a different embodiment tag is
requested. `GROOT_RELATIVE_EXCLUDE_JOINTS` is fixed to
`["hand","waist","base_height","navigate"]`; changing it would make the
training processor and simulator decoder disagree about absolute action groups.

Train another subtask by changing only `SUBTASK`:

```bash
SUBTASK=coarse_insert \
POLICY_TYPE=act \
./scripts/train_lerobot.sh
```

The defaults resolve to:

```text
DATASET_REPO_ID=Team-RAMEN/IROS2026_RAMEN_suzuki_{subtask}_1
DATASET_REVISION=  # latest revision; set a commit SHA for reproducible releases
POLICY_REPO_ID=Team-RAMEN/IROS2026_RAMEN_suzuki_{subtask}_{policy_type}_1
OUTPUT_DIR=outputs/train/{policy_type}_{subtask}
JOB_NAME={policy_type}_{subtask}
TRAINING_VIEW_ROOT=outputs/training_views/{policy_type}_{subtask}
WANDB_PROJECT=iros2026-ramen-{subtask-with-hyphens}
```

You can still override any of them explicitly:

```bash
SUBTASK=pick_leg \
POLICY_TYPE=act \
OUTPUT_DIR=outputs/train/debug_pick_leg \
POLICY_REPO_ID=Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_act_debug \
./scripts/train_lerobot.sh --steps=1000
```

Additional arguments are passed through to `lerobot-train`.

Set `DRY_RUN=true` to print the resolved `lerobot-train` command without starting training.

By default, the wrapper keeps LeRobot's internal `--policy.push_to_hub=false` and uploads the final checkpoint after successful training with `scripts/upload_policy.py`. This avoids losing the final model when the training command was launched with hub push disabled. Set `UPLOAD_AFTER_TRAIN=false` to skip the post-training upload, or set `PUSH_TO_HUB=true` if you also want LeRobot's native hub push behavior during training.

W&B project names default to `iros2026-ramen-{subtask}` with underscores converted to hyphens, for example `iros2026-ramen-flip-table`. Override with `WANDB_PROJECT=...` if a run needs to land somewhere else.

The wrapper defaults `TOLERANCE_S=0.001` to tolerate small MP4 timestamp rounding differences during video-backed training. Override it only if a run needs stricter validation:

```bash
TOLERANCE_S=0.0001 ./scripts/train_lerobot.sh
```

## Local Training View

`scripts/train_lerobot.sh` always runs `scripts/materialize_lerobot_training_view.py`
before training. The raw official schema is not directly compatible with the
policy keys, so disabling this step is rejected. The local view is written under:

```text
outputs/training_views/{policy_type}_{subtask}
```

The source HF dataset is not rewritten. The materializer downloads only
`meta/**`, `data/**`, and the selected RGB camera videos (`cam_0`, `cam_2`,
`cam_3`), then writes:

- `data/chunk-*/file-*.parquet` with canonical LeRobot policy keys:
  `observation.state` and `action`.
- `videos/observation.images.head_left`, `left_wrist`, `right_wrist` as local
  symlinks to the selected source video directories.
- `meta/info.json`, `meta/stats.json`, and `meta/episodes/**` with renamed
  camera keys and recombined state/action stats, including `q01/q99`.
- `meta/team_ramen_episode_split.json` with a deterministic 80/10/10 split by
  original recording. Test recordings are excluded from training; LeRobot's
  offline evaluator receives only the held-out validation recordings.
- For GR00T, `meta/modality.json` with the checked 49-D/53-D group order,
  source EEF convention, per-group relative/absolute action semantics, and the
  complete source `robot_q` index to Unitree SDK motor ID table.

The materializer validates LeRobot v3, 30 Hz, the exact three 640x480 RGB
cameras, all source vector dimensions, and finite mapped values. It fingerprints
the source metadata/parquets and builds into a temporary directory before
replacing an old view, so failed conversion cannot destroy a usable view.
State/action normalization statistics are recomputed from the training split
only; validation and test recordings do not influence those statistics.

Set `TRAINING_VIEW_FORCE=true` to rebuild an existing view, or
`SOURCE_DATASET_ROOT=/path/to/local/dataset` to materialize from an already
downloaded dataset tree.

Audit the source dataset before a release training run:

```bash
python scripts/audit_lerobot_dataset.py \
  --repo-id Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1 \
  --revision 10a6ec05f9993b8d59faad2957e47153b0f15f37 \
  --strict
```

The audit checks metadata, all numeric columns, local video decoding, and the
exact remote MP4 inventory. A dataset without an explicit success field reports
a warning until its demonstrations have been visually reviewed; run without
`--strict` only after recording that review decision.

To compare one ACT checkpoint on real and simulator observations, supply a
captured frame directory and the matching evaluator log explicitly:

```bash
python scripts/diagnose_act_policy_outputs.py \
  --checkpoint /path/to/pretrained_model \
  --dataset-root outputs/training_views/act_flip_table \
  --sim-frame-dir /path/to/test_0/camera_frames/frame_0010 \
  --sim-log /path/to/flip_table_eval.log
```

The diagnostic uses strict model loading and the normalizer files referenced by
the serialized processor manifests. It does not substitute missing statistics.

## Upload A Policy

```bash
SUBTASK=pick_leg \
POLICY_TYPE=act \
MODEL_DIR=outputs/train/act_pick_leg/checkpoints/last/pretrained_model \
./scripts/upload_policy.sh
```

If `MODEL_DIR` is omitted, it defaults to:

```text
outputs/train/{policy_type}_{subtask}/checkpoints/last/pretrained_model
```

The upload wrapper uses the Python Hugging Face Hub API, creates the private
model repo if needed, and validates both the core LeRobot files and every
processor state file referenced by the serialized processors. Files left by an
older checkpoint are deleted from the model repo, while `README.md` and
`.gitattributes` are preserved.

The training wrapper calls the same uploader automatically after a successful run. It resolves `outputs/train/{policy_type}_{subtask}/checkpoints/last/pretrained_model` first, then falls back to the newest numbered checkpoint.

## Download Raw Source MCAPs

This is usually unnecessary for training because the converted HF datasets are already uploaded. For debugging source data:

```bash
SUBTASK=pick_leg \
python scripts/download_source_mcaps.py \
  --config configs/subtask_training.json \
  --max-mcaps 3
```

Dataset conversion is intentionally kept out of this training folder. Production
subtask datasets are created by `data/bitrobot_lerobot_subtask_datasets`, which
splits the official BitRobot LeRobot v3 dataset by subtask and uploads the
shared HF repos without model-specific key rewriting. Model-specific layout is
handled locally by the training view materializer.

`scripts/write_groot_modality_json.py` can regenerate only the checked modality
metadata for diagnostics. New and existing training runs must use
`scripts/train_lerobot.sh`; the obsolete 43-D/47-D standalone GR00T
materializer has been removed to prevent accidental use.

## Coarse-insert inference

The coarse-insert runtime is
`inference/coarse_insert.py`; it has no dependency on the flip-table simulator.
`build_coarse_insert_state()` accepts two root-frame wrist poses, the 29 body
joints in Unitree SDK ID order, and two physical Dex1 values. The runtime
decodes only SDK motor IDs 12–28 and the two Dex1 commands. Dex1 outputs are
projected from each predicted 7-D hand group onto the fixed official G1 hand
synergy using its least-squares inverse; base-height and navigation outputs are
never sent to the robot. By default the runtime loads the checked G1 29-DoF +
Dex1-1 deployment URDF and clamps every SDK target; pass limits loaded by
`load_sdk_position_limits()` to use a custom URDF.

For held-out dataset inference and URDF motion rendering, use:

```bash
singularity exec --nv \
  --bind /home/naoki.takada/workspace/iros_2026_ramen:/workspace/iros_2026_ramen \
  /home/naoki.takada/workspace/iros_2026_ramen/gr00t-lerobot.sif \
  bash -lc 'cd /workspace/iros_2026_ramen/model/subtask_policy_training && \
    PYTHONPATH=outputs/lerobot_source_overlays/groot_relative_eef_v3:$PWD \
    /opt/lerobot/.venv/bin/python scripts/evaluate_groot_joint_motion.py \
      --checkpoint outputs/train/CHOSEN_RUN/checkpoints/020000/pretrained_model \
      --device cuda:0'
```
