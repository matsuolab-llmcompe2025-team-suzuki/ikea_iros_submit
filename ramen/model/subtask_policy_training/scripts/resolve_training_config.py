"""Validate and resolve the shared subtask training configuration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
from pathlib import Path
from typing import Any


FEATURE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = FEATURE_ROOT / "configs" / "subtask_training.json"
MAPPING_PATH = FEATURE_ROOT / "gr00t" / "g1_full_body_mapping.py"
STANDARD_POLICY_CAMERAS = ("head_left", "left_wrist", "right_wrist")
DEFAULT_IMAGE_SHAPE_HWC = (480, 640, 3)
SUPPORTED_POLICY_TYPES = ("act", "diffusion", "flow_matching", "groot")

# LeRobot DiffusionConfig の既定 down_dims=(512, 1024, 2048) = 3 段 → 2 ** 3。
DIFFUSION_UNET_DOWNSAMPLING_FACTOR = 8
GROOT_RELATIVE_EXCLUDE_JOINTS = ("hand", "waist", "base_height", "navigate")


def load_mapping_module() -> Any:
    spec = importlib.util.spec_from_file_location("g1_full_body_mapping", MAPPING_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load mapping module: {MAPPING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mapping = load_mapping_module()
GROOT_G1_STATE_DIM = _mapping.REAL_G1_RELATIVE_EEF_STATE_DIM
GROOT_G1_ACTION_DIM = _mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM
GROOT_G1_ACTION_SEMANTICS = "real_g1_relative_eef_relative_arm_absolute_hand_waist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--format", choices=("shell", "json"), default="shell")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "subtask_training_v1":
        raise ValueError(f"unsupported training config schema: {config.get('schema_version')}")
    return config


def format_template(template: str, *, subtask: str, policy_type: str) -> str:
    return template.format(
        subtask=subtask,
        subtask_kebab=subtask.replace("_", "-"),
        policy_type=policy_type,
    )


def env_bool(name: str, default: bool) -> str:
    if name not in os.environ:
        return "true" if default else "false"
    value = os.environ[name].strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "true"
    if value in {"0", "false", "no", "off"}:
        return "false"
    raise ValueError(f"{name} must be a boolean-like value, got {os.environ[name]!r}")


def env_string(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int, *, minimum: int | None = None) -> str:
    if name not in os.environ:
        value = int(default)
    else:
        try:
            value = int(os.environ[name])
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {os.environ[name]!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return str(value)


def env_json_list(name: str, default: list[str]) -> str:
    if name not in os.environ:
        values = default
    else:
        values = json.loads(os.environ[name])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{name} must be a JSON list of strings")
    return json.dumps(values, separators=(",", ":"))


def policy_camera_names(config: dict[str, Any]) -> list[str]:
    camera_names = [str(name) for name in config["cameras"]]
    if tuple(camera_names) != STANDARD_POLICY_CAMERAS:
        raise ValueError(
            f"cameras must be exactly {STANDARD_POLICY_CAMERAS} in that order, got {camera_names}"
        )
    camera_map = config.get("source_dataset", {}).get("camera_map", {})
    missing = [name for name in STANDARD_POLICY_CAMERAS if name not in camera_map]
    mapped = [camera_map.get(name) for name in STANDARD_POLICY_CAMERAS]
    if missing or len(set(mapped)) != len(mapped):
        raise ValueError(f"source camera map is incomplete or aliased: missing={missing}, values={mapped}")
    return camera_names


def training_video_map(config: dict[str, Any]) -> dict[str, str]:
    camera_map = config["source_dataset"]["camera_map"]
    return {
        f"observation.images.{camera_name}": camera_map[camera_name]
        for camera_name in policy_camera_names(config)
    }


def image_shape_hwc(config: dict[str, Any]) -> tuple[int, int, int]:
    image = config.get("image", {})
    shape = (
        int(image.get("height", DEFAULT_IMAGE_SHAPE_HWC[0])),
        int(image.get("width", DEFAULT_IMAGE_SHAPE_HWC[1])),
        int(image.get("channels", DEFAULT_IMAGE_SHAPE_HWC[2])),
    )
    if shape != DEFAULT_IMAGE_SHAPE_HWC:
        raise ValueError(
            f"real G1 policy cameras must be HWC={DEFAULT_IMAGE_SHAPE_HWC}, got {shape}"
        )
    return shape


def control_scope(config: dict[str, Any], policy_type: str) -> str:
    if policy_type == "groot":
        return "upper_body_relative_eef"
    value = os.environ.get("CONTROL_SCOPE", config.get("training", {}).get("control_scope", "upper_body"))
    value = str(value).strip().lower()
    if value not in {"upper_body", "full_robot_q"}:
        raise ValueError("CONTROL_SCOPE must be 'upper_body' or 'full_robot_q'")
    return value


def policy_dims(config: dict[str, Any], policy_type: str, scope: str) -> tuple[int, int]:
    if policy_type == "groot":
        return GROOT_G1_STATE_DIM, GROOT_G1_ACTION_DIM
    if scope == "upper_body":
        return _mapping.UPPER_BODY_STATE_DIM, _mapping.UPPER_BODY_ACTION_DIM
    return _mapping.SOURCE_STATE_DIM, _mapping.SOURCE_ACTION_DIM


def action_semantics(config: dict[str, Any], policy_type: str, scope: str) -> str:
    if policy_type == "groot":
        return GROOT_G1_ACTION_SEMANTICS
    if scope == "upper_body":
        return "upper_body_absolute_target"
    return str(config["action"]["semantics"])


def policy_view_layout(policy_type: str, scope: str) -> str:
    if policy_type == "groot":
        return "real_g1_relative_eef_relative_joints"
    if scope == "upper_body":
        return "robot_q_upper_body_19d"
    return "robot_q_current_robot_q_desired_38d"


def policy_input_features_json(
    config: dict[str, Any],
    camera_names: list[str],
    *,
    state_dim: int,
) -> str:
    height, width, channels = image_shape_hwc(config)
    features: dict[str, Any] = {
        str(config["state"]["key"]): {
            "type": "STATE",
            "shape": [state_dim],
        }
    }
    for camera_name in camera_names:
        features[f"observation.images.{camera_name}"] = {
            "type": "VISUAL",
            "shape": [channels, height, width],
        }
    return json.dumps(features, separators=(",", ":"))


def policy_output_features_json(config: dict[str, Any], *, action_dim: int) -> str:
    features = {
        str(config["action"]["key"]): {
            "type": "ACTION",
            "shape": [action_dim],
        }
    }
    return json.dumps(features, separators=(",", ":"))


def resolve(config: dict[str, Any]) -> dict[str, str]:
    training = config.get("training", {})
    if int(config.get("fps", 0)) != 30:
        raise ValueError(f"the source and real G1 controller contract requires fps=30, got {config.get('fps')!r}")
    subtask = os.environ.get("SUBTASK", config["subtask"])
    if subtask not in config["subtasks"]:
        choices = ", ".join(sorted(config["subtasks"]))
        raise ValueError(f"unknown SUBTASK={subtask!r}; choices: {choices}")

    policy_type = str(os.environ.get("POLICY_TYPE", training.get("policy_type", "act"))).strip().lower()
    if policy_type not in SUPPORTED_POLICY_TYPES:
        raise ValueError(
            f"POLICY_TYPE must be one of {SUPPORTED_POLICY_TYPES}, got {policy_type!r}"
        )
    # Issue #122: subtask 定義に `dataset_repos` (list) があれば multi-repo として export、
    # 無ければ従来通り single `dataset_repo_template` から format (backward compat)。
    subtask_entry = config["subtasks"][subtask]
    subtask_dataset_repos = subtask_entry.get("dataset_repos")
    if subtask_dataset_repos is not None:
        if not isinstance(subtask_dataset_repos, list) or not all(
            isinstance(x, str) for x in subtask_dataset_repos
        ):
            raise ValueError(
                f"subtask '{subtask}' dataset_repos must be a list of strings, got {subtask_dataset_repos!r}"
            )
        if not subtask_dataset_repos:
            raise ValueError(f"subtask '{subtask}' dataset_repos must not be empty")

    # Issue #122 D-1: skill 単位 merged HF repo mode。
    # merged_repo_template から skill-level merged repo id を解決し、
    # component_subtasks (union subtask) 対応も入れる。USE_MERGED_HF_REPO=true で
    # 学習側の DATASET_REPO_IDS を chunk list → skill-merged repo list に差替える。
    # default (false) は現行 chunk-level flow 完全維持 (5 Run pipeline regression 回避)。
    merged_repo_template = config.get("merged_repo_template")
    component_subtasks = subtask_entry.get("component_subtasks")
    if component_subtasks is not None:
        if not isinstance(component_subtasks, list) or not all(
            isinstance(x, str) for x in component_subtasks
        ):
            raise ValueError(
                f"subtask '{subtask}' component_subtasks must be a list of strings, got {component_subtasks!r}"
            )
        if not component_subtasks:
            raise ValueError(f"subtask '{subtask}' component_subtasks must not be empty")
        for component in component_subtasks:
            if component not in config["subtasks"]:
                raise ValueError(
                    f"subtask '{subtask}' references unknown component_subtask '{component}'"
                )
    if merged_repo_template is not None and not isinstance(merged_repo_template, str):
        raise ValueError(
            f"merged_repo_template must be a string, got {merged_repo_template!r}"
        )
    merged_dataset_repo_id = ""
    merged_component_repo_ids_json = "[]"
    if merged_repo_template:
        if component_subtasks:
            # union subtask (combined_task5_7 等) は self の単一 merged repo を持たない
            # — MERGED_DATASET_REPO_ID を empty にすることで orchestrator (prepare_merged_
            # skill_dataset.sh) が誤って union に対して push しようとした場合に fail させる。
            # 学習側は MERGED_COMPONENT_REPO_IDS の component list を使う。
            merged_dataset_repo_id = os.environ.get("MERGED_DATASET_REPO_ID", "")
            default_component_ids = [
                format_template(merged_repo_template, subtask=component, policy_type=policy_type)
                for component in component_subtasks
            ]
        else:
            merged_dataset_repo_id = os.environ.get(
                "MERGED_DATASET_REPO_ID",
                format_template(merged_repo_template, subtask=subtask, policy_type=policy_type),
            )
            default_component_ids = [merged_dataset_repo_id]
        merged_component_repo_ids_json = os.environ.get(
            "MERGED_COMPONENT_REPO_IDS",
            json.dumps(default_component_ids, separators=(",", ":")),
        )
        parsed_components = json.loads(merged_component_repo_ids_json)
        if not isinstance(parsed_components, list) or not all(
            isinstance(x, str) for x in parsed_components
        ):
            raise ValueError(
                "MERGED_COMPONENT_REPO_IDS must be a JSON list of strings, got "
                f"{merged_component_repo_ids_json!r}"
            )
        if not parsed_components:
            raise ValueError("MERGED_COMPONENT_REPO_IDS must not be empty")

    use_merged_hf_repo = env_bool("USE_MERGED_HF_REPO", False)

    if subtask_dataset_repos is not None:
        # 単一 DATASET_REPO_ID は 1 つ目 (後続 downstream の backward compat 用)、
        # 全 repo は DATASET_REPO_IDS で export。
        default_repo_id = subtask_dataset_repos[0]
        default_repo_ids = json.dumps(subtask_dataset_repos, separators=(",", ":"))
    else:
        default_repo_id = format_template(
            config["dataset_repo_template"], subtask=subtask, policy_type=policy_type
        )
        default_repo_ids = json.dumps([default_repo_id], separators=(",", ":"))

    if use_merged_hf_repo == "true":
        if not merged_repo_template:
            raise ValueError(
                "USE_MERGED_HF_REPO=true requires 'merged_repo_template' in the training config"
            )
        # opt-in: DATASET_REPO_IDS を skill-merged repo list に差替える。
        # env override は引き続き優先 (test/pilot 用の逃げ道を維持)。
        dataset_repo_ids = os.environ.get(
            "DATASET_REPO_IDS", merged_component_repo_ids_json
        )
        parsed_ids = json.loads(dataset_repo_ids)
        dataset_repo_id = os.environ.get("DATASET_REPO_ID", parsed_ids[0])
    else:
        dataset_repo_id = os.environ.get("DATASET_REPO_ID", default_repo_id)
        dataset_repo_ids = os.environ.get("DATASET_REPO_IDS", default_repo_ids)
    dataset_revision = os.environ.get(
        "DATASET_REVISION",
        str(config.get("source_dataset", {}).get("revision", "")),
    ).strip()
    groot_dataset_repo_id = os.environ.get(
        "GROOT_DATASET_REPO_ID",
        format_template(config["groot_dataset_repo_template"], subtask=subtask, policy_type=policy_type),
    )
    policy_repo_id = os.environ.get(
        "POLICY_REPO_ID",
        format_template(config["policy_repo_template"], subtask=subtask, policy_type=policy_type),
    )
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        format_template(
            training.get("output_dir_template", "outputs/train/{policy_type}_{subtask}"),
            subtask=subtask,
            policy_type=policy_type,
        ),
    )
    job_name = os.environ.get(
        "JOB_NAME",
        format_template(
            training.get("job_name_template", "{policy_type}_{subtask}"),
            subtask=subtask,
            policy_type=policy_type,
        ),
    )
    wandb_project = os.environ.get(
        "WANDB_PROJECT",
        format_template(
            training.get("wandb_project_template", "iros2026-ramen-{subtask_kebab}"),
            subtask=subtask,
            policy_type=policy_type,
        ),
    )
    policy_defaults = config.get("policy_defaults", {}).get(policy_type, {})
    groot_defaults = config.get("policy_defaults", {}).get("groot", {})

    camera_names = policy_camera_names(config)
    image_shape_hwc(config)
    scope = control_scope(config, policy_type)
    state_dim, action_dim = policy_dims(config, policy_type, scope)
    training_view_root = os.environ.get(
        "TRAINING_VIEW_ROOT",
        format_template(
            training.get(
                "training_view_root_template",
                "outputs/training_views/{policy_type}_{subtask}",
            ),
            subtask=subtask,
            policy_type=policy_type,
        ),
    )
    materialize_training_view = env_bool(
        "MATERIALIZE_TRAINING_VIEW", bool(training.get("materialize_training_view", True))
    )
    if materialize_training_view != "true":
        raise ValueError(
            "MATERIALIZE_TRAINING_VIEW must stay true because the official dataset uses raw "
            "cam_*/robot_q/EEF keys rather than the policy feature contract"
        )

    values = {
        "SUBTASK": subtask,
        "TASK": str(config["subtasks"][subtask]["task"]),
        "POLICY_TYPE": policy_type,
        "CONTROL_SCOPE": scope,
        "DATASET_REPO_ID": dataset_repo_id,
        "DATASET_REPO_IDS": dataset_repo_ids,  # Issue #122: JSON list of repo_ids (multi-chunk 対応)
        "DATASET_REVISION": dataset_revision,
        # Issue #122 D-1: skill 単位 merged HF repo id (upload/pull 用参照値、USE_MERGED_HF_REPO と直交)
        "MERGED_DATASET_REPO_ID": merged_dataset_repo_id,
        "MERGED_COMPONENT_REPO_IDS": merged_component_repo_ids_json,
        "USE_MERGED_HF_REPO": use_merged_hf_repo,
        "GROOT_DATASET_REPO_ID": groot_dataset_repo_id,
        "POLICY_REPO_ID": policy_repo_id,
        "OUTPUT_DIR": output_dir,
        "JOB_NAME": job_name,
        "DEVICE": os.environ.get("DEVICE", training.get("device", "cuda")),
        "WANDB_ENABLE": env_bool("WANDB_ENABLE", bool(training.get("wandb_enable", False))),
        "WANDB_PROJECT": wandb_project,
        "PUSH_TO_HUB": env_bool("PUSH_TO_HUB", bool(training.get("push_to_hub", False))),
        "UPLOAD_AFTER_TRAIN": env_bool(
            "UPLOAD_AFTER_TRAIN", bool(training.get("upload_after_train", True))
        ),
        "PRIVATE": env_bool("PRIVATE", bool(training.get("private", True))),
        "FPS": str(config["fps"]),
        "ROBOT_TYPE": str(config["robot_type"]),
        "STATE_DIM": str(state_dim),
        "ACTION_DIM": str(action_dim),
        "ACTION_SEMANTICS": action_semantics(config, policy_type, scope),
        "CAMERAS": ",".join(camera_names),
        "SOURCE_CAMERA_MAP": json.dumps(training_video_map(config), separators=(",", ":")),
        "POLICY_VIEW_LAYOUT": policy_view_layout(policy_type, scope),
        "MATERIALIZE_TRAINING_VIEW": materialize_training_view,
        "TRAINING_VIEW_ROOT": training_view_root,
        "TRAINING_VIEW_FORCE": env_bool(
            "TRAINING_VIEW_FORCE", bool(training.get("force_training_view", False))
        ),
        "POLICY_INPUT_FEATURES": policy_input_features_json(
            config,
            camera_names,
            state_dim=state_dim,
        ),
        "POLICY_OUTPUT_FEATURES": policy_output_features_json(config, action_dim=action_dim),
        "TRAIN_IMAGE_TRANSFORMS_ENABLE": env_bool(
            "TRAIN_IMAGE_TRANSFORMS_ENABLE",
            bool(policy_defaults.get("image_transforms_enable", True)),
        ),
        "TRAIN_BATCH_SIZE": env_int(
            "TRAIN_BATCH_SIZE", int(policy_defaults.get("batch_size", 8)), minimum=1
        ),
        "TRAIN_STEPS": env_int(
            "TRAIN_STEPS", int(policy_defaults.get("steps", 300000)), minimum=1
        ),
        "TRAIN_SAVE_FREQ": env_int(
            "TRAIN_SAVE_FREQ", int(policy_defaults.get("save_freq", 50000)), minimum=1
        ),
        "TRAIN_EVAL_STEPS": env_int(
            "TRAIN_EVAL_STEPS", int(policy_defaults.get("eval_steps", 10000)), minimum=1
        ),
        "TRAIN_MAX_EVAL_SAMPLES": env_int(
            "TRAIN_MAX_EVAL_SAMPLES",
            int(policy_defaults.get("max_eval_samples", 512)),
            minimum=1,
        ),
        "TRAIN_LOG_FREQ": env_int(
            "TRAIN_LOG_FREQ", int(policy_defaults.get("log_freq", 100)), minimum=1
        ),
    }
    if policy_type == "act":
        act_chunk_size = env_int(
            "ACT_CHUNK_SIZE", int(policy_defaults.get("chunk_size", 100)), minimum=1
        )
        act_action_steps = env_int(
            "ACT_N_ACTION_STEPS", int(policy_defaults.get("n_action_steps", 10)), minimum=1
        )
        if int(act_action_steps) > int(act_chunk_size):
            raise ValueError("ACT_N_ACTION_STEPS cannot exceed ACT_CHUNK_SIZE")
        values.update(
            {
                "ACT_CHUNK_SIZE": act_chunk_size,
                "ACT_N_ACTION_STEPS": act_action_steps,
            }
        )
    if policy_type == "diffusion":
        horizon = env_int("DIFFUSION_HORIZON", int(policy_defaults.get("horizon", 32)), minimum=1)
        n_action_steps = env_int(
            "DIFFUSION_N_ACTION_STEPS", int(policy_defaults.get("n_action_steps", 8)), minimum=1
        )
        n_obs_steps = env_int(
            "DIFFUSION_N_OBS_STEPS", int(policy_defaults.get("n_obs_steps", 2)), minimum=1
        )
        # LeRobot DiffusionConfig.__post_init__: horizon は U-Net の downsampling factor
        # (2 ** len(down_dims)、既定 down_dims=(512,1024,2048) で 8) の倍数でなければならない。
        if int(horizon) % DIFFUSION_UNET_DOWNSAMPLING_FACTOR:
            raise ValueError(
                f"DIFFUSION_HORIZON must be a multiple of {DIFFUSION_UNET_DOWNSAMPLING_FACTOR} "
                f"(U-Net downsampling factor); got {horizon}"
            )
        if int(n_action_steps) + int(n_obs_steps) - 1 > int(horizon):
            raise ValueError(
                "DIFFUSION_N_ACTION_STEPS + DIFFUSION_N_OBS_STEPS - 1 cannot exceed DIFFUSION_HORIZON"
            )
        # 元実装は末尾 (horizon - n_action_steps - n_obs_steps + 1) frame を sample しない。
        # horizon を既定から変えた時に追従しないと episode 末尾で padding された窓を学習する。
        drop_n_last_frames = env_int(
            "DIFFUSION_DROP_N_LAST_FRAMES",
            int(horizon) - int(n_action_steps) - int(n_obs_steps) + 1,
            minimum=0,
        )
        values.update(
            {
                "DIFFUSION_HORIZON": horizon,
                "DIFFUSION_N_ACTION_STEPS": n_action_steps,
                "DIFFUSION_N_OBS_STEPS": n_obs_steps,
                "DIFFUSION_NUM_INFERENCE_STEPS": env_int(
                    "DIFFUSION_NUM_INFERENCE_STEPS",
                    int(policy_defaults.get("num_inference_steps", 10)),
                    minimum=1,
                ),
                "DIFFUSION_DROP_N_LAST_FRAMES": drop_n_last_frames,
            }
        )
    if policy_type == "flow_matching":
        action_horizon = env_int(
            "FLOW_ACTION_HORIZON", int(policy_defaults.get("action_horizon", 24)), minimum=2
        )
        n_action_steps = env_int(
            "FLOW_N_ACTION_STEPS", int(policy_defaults.get("n_action_steps", 6)), minimum=1
        )
        if int(n_action_steps) > int(action_horizon):
            raise ValueError("FLOW_N_ACTION_STEPS cannot exceed FLOW_ACTION_HORIZON")
        values.update(
            {
                "FLOW_ACTION_HORIZON": action_horizon,
                "FLOW_N_ACTION_STEPS": n_action_steps,
                "FLOW_INFERENCE_STEPS": env_int(
                    "FLOW_INFERENCE_STEPS",
                    int(policy_defaults.get("flow_inference_steps", 10)),
                    minimum=1,
                ),
                "FLOW_MODEL_DIM": env_int(
                    "FLOW_MODEL_DIM", int(policy_defaults.get("model_dim", 384)), minimum=32
                ),
                "FLOW_TRANSFORMER_LAYERS": env_int(
                    "FLOW_TRANSFORMER_LAYERS",
                    int(policy_defaults.get("transformer_layers", 6)),
                    minimum=1,
                ),
                "FLOW_TRANSFORMER_HEADS": env_int(
                    "FLOW_TRANSFORMER_HEADS",
                    int(policy_defaults.get("transformer_heads", 8)),
                    minimum=1,
                ),
            }
        )
        if int(values["FLOW_MODEL_DIM"]) % int(values["FLOW_TRANSFORMER_HEADS"]):
            raise ValueError("FLOW_MODEL_DIM must be divisible by FLOW_TRANSFORMER_HEADS")
    if policy_type == "groot":
        defaults = dict(groot_defaults)
        defaults.update(policy_defaults)
        embodiment_tag = env_string(
            "GROOT_EMBODIMENT_TAG",
            str(defaults.get("embodiment_tag", _mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG)),
        )
        if embodiment_tag != _mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG:
            raise ValueError(
                "The GR00T training view uses the official "
                f"{_mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG!r} slot order; "
                f"GROOT_EMBODIMENT_TAG={embodiment_tag!r} is incompatible"
            )
        use_relative_actions = env_bool(
            "GROOT_USE_RELATIVE_ACTIONS", bool(defaults.get("use_relative_actions", True))
        )
        require_native_relative_eef = env_bool(
            "GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR",
            bool(defaults.get("require_native_relative_eef_processor", True)),
        )
        if require_native_relative_eef != "true":
            raise ValueError(
                "GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR must stay true for the "
                "REAL_G1 relative-EEF training layout"
            )
        if use_relative_actions != "true":
            raise ValueError(
                "GROOT_USE_RELATIVE_ACTIONS must stay true: EEF targets are converted in the "
                "N1.7 processor against the action chunk's current observation"
            )
        relative_exclude_joints = env_json_list(
            "GROOT_RELATIVE_EXCLUDE_JOINTS",
            list(defaults.get("relative_exclude_joints", GROOT_RELATIVE_EXCLUDE_JOINTS)),
        )
        if tuple(json.loads(relative_exclude_joints)) != GROOT_RELATIVE_EXCLUDE_JOINTS:
            raise ValueError(
                "GROOT_RELATIVE_EXCLUDE_JOINTS must be exactly "
                f"{GROOT_RELATIVE_EXCLUDE_JOINTS}; changing it breaks the REAL_G1 action contract"
            )
        values.update(
            {
                "GROOT_BASE_MODEL_PATH": env_string(
                    "GROOT_BASE_MODEL_PATH", str(defaults.get("base_model_path", "nvidia/GR00T-N1.7-3B"))
                ),
                "GROOT_BASE_MODEL_REVISION": env_string(
                    "GROOT_BASE_MODEL_REVISION", str(defaults.get("base_model_revision", "main"))
                ),
                "GROOT_EMBODIMENT_TAG": embodiment_tag,
                "GROOT_CHUNK_SIZE": env_int(
                    "GROOT_CHUNK_SIZE", int(defaults.get("chunk_size", 16)), minimum=1
                ),
                "GROOT_N_ACTION_STEPS": env_int(
                    "GROOT_N_ACTION_STEPS", int(defaults.get("n_action_steps", 16)), minimum=1
                ),
                "GROOT_USE_RELATIVE_ACTIONS": use_relative_actions,
                "GROOT_RELATIVE_EXCLUDE_JOINTS": relative_exclude_joints,
                "GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR": require_native_relative_eef,
                "GROOT_PROCESSOR_OVERLAY_ROOT": env_string(
                    "GROOT_PROCESSOR_OVERLAY_ROOT",
                    str(
                        defaults.get(
                            "processor_overlay_root",
                            "outputs/groot_base_overlays/real_g1_relative_eef_3cam_dex1_v2",
                        )
                    ),
                ),
                "GROOT_USE_BF16": env_bool("GROOT_USE_BF16", bool(defaults.get("use_bf16", True))),
                "GROOT_IMAGE_TRANSFORMS_ENABLE": env_bool(
                    "GROOT_IMAGE_TRANSFORMS_ENABLE", bool(defaults.get("image_transforms_enable", True))
                ),
                "GROOT_BATCH_SIZE": env_int(
                    "GROOT_BATCH_SIZE", int(defaults.get("batch_size", 64)), minimum=1
                ),
                "GROOT_STEPS": env_int(
                    "GROOT_STEPS", int(defaults.get("steps", 20000)), minimum=1
                ),
                "GROOT_SAVE_FREQ": env_int(
                    "GROOT_SAVE_FREQ", int(defaults.get("save_freq", 5000)), minimum=1
                ),
                "GROOT_ENV_EVAL_FREQ": env_int(
                    "GROOT_ENV_EVAL_FREQ", int(defaults.get("env_eval_freq", 0)), minimum=0
                ),
                "GROOT_EVAL_STEPS": env_int(
                    "GROOT_EVAL_STEPS", int(defaults.get("eval_steps", 10000)), minimum=1
                ),
                "GROOT_MAX_EVAL_SAMPLES": env_int(
                    "GROOT_MAX_EVAL_SAMPLES",
                    int(defaults.get("max_eval_samples", 512)),
                    minimum=1,
                ),
                "GROOT_LOG_FREQ": env_int(
                    "GROOT_LOG_FREQ", int(defaults.get("log_freq", 10)), minimum=1
                ),
            }
        )
        if int(values["GROOT_N_ACTION_STEPS"]) > int(values["GROOT_CHUNK_SIZE"]):
            raise ValueError("GROOT_N_ACTION_STEPS cannot exceed GROOT_CHUNK_SIZE")
    return values


def print_shell(values: dict[str, str]) -> None:
    for key, value in values.items():
        print(f"export {key}={shlex.quote(value)}")


def main() -> None:
    args = parse_args()
    values = resolve(load_config(args.config))
    if args.format == "json":
        print(json.dumps(values, indent=2, ensure_ascii=False))
    else:
        print_shell(values)


if __name__ == "__main__":
    main()
