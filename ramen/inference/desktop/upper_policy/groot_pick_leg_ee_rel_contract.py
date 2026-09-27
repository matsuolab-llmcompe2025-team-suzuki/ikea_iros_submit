"""Sealed metadata contract for the pick-leg relative-EEF GR00T checkpoint.

The model emits absolute EEF poses only *after* the official processor
decodes its relative SE(3) action against the current 56-D state. This
contract must not be routed through the 38-D joint-action pick adapter.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference.desktop.upper_policy.groot_pick_leg_contract import (
    CAMERA_KEYS,
    DATASET_REPO_ID,
    DATASET_REVISION,
    EMBODIMENT_TAG,
    MODEL_ACTION_HORIZON,
    TASK_TEXT,
    _stat_dim,
)

MODEL_REPO_ID = "Team-RAMEN/groot-n1.7-pick-legs-ee-rel-lora"
MODEL_REVISION = "ba3a20f30f5d0388c6656390e1b88228c06aa332"
MODEL_STATE_DIM = 56
MODEL_ACTION_DIM = 20


def validate_ee_relative_checkpoint_metadata(
    checkpoint: str | Path, *,
    model_repo_id: str = MODEL_REPO_ID,
    model_revision: str = MODEL_REVISION,
    task: str = TASK_TEXT,
) -> dict:
    if model_repo_id != MODEL_REPO_ID or model_revision != MODEL_REVISION:
        raise ValueError("relative-EEF pick model must use the pinned HF revision")
    if task != TASK_TEXT:
        raise ValueError(f"relative-EEF pick task must be {TASK_TEXT!r}")
    root = Path(checkpoint).expanduser().resolve()
    required = (
        "config.json", "processor_config.json", "statistics.json",
        "embodiment_id.json", "model.safetensors.index.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete relative-EEF checkpoint: {missing}")
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "Gr00tN1d7" or config.get("action_horizon") != 40:
        raise ValueError("relative-EEF checkpoint GR00T architecture changed")
    kwargs = json.loads((root / "processor_config.json").read_text())[
        "processor_kwargs"
    ]
    if kwargs.get("use_relative_action") is not True:
        raise ValueError("relative-EEF processor must decode relative actions")
    modality = kwargs["modality_configs"][EMBODIMENT_TAG]
    video, state, action = (
        modality[key] for key in ("video", "state", "action")
    )
    if video.get("modality_keys") != ["cam_0", "cam_1", "cam_2", "cam_3"]:
        raise ValueError("relative-EEF camera order changed")
    state_keys = ["left_ee", "right_ee", "hand", "robot_q"]
    action_keys = ["left_ee", "right_ee", "hand"]
    if state.get("modality_keys") != state_keys:
        raise ValueError("relative-EEF state order changed")
    if action.get("modality_keys") != action_keys:
        raise ValueError("relative-EEF action order changed")
    if action.get("delta_indices") != list(range(MODEL_ACTION_HORIZON)):
        raise ValueError("relative-EEF action horizon changed")
    configs = action.get("action_configs") or ()
    if len(configs) != 3:
        raise ValueError("relative-EEF action group count changed")
    for i in (0, 1):
        cfg = configs[i]
        if (cfg.get("rep"), cfg.get("type"), cfg.get("format"), cfg.get("state_key")) != (
            "RELATIVE", "EEF", "XYZ_ROT6D", action_keys[i],
        ):
            raise ValueError("relative-EEF arm representation changed")
    if configs[2].get("rep") != "ABSOLUTE":
        raise ValueError("relative-EEF Dex1 representation changed")
    stats = json.loads((root / "statistics.json").read_text())[EMBODIMENT_TAG]
    expected = {
        "state": {"left_ee": 9, "right_ee": 9, "hand": 2, "robot_q": 36},
        "action": {"left_ee": 9, "right_ee": 9, "hand": 2},
    }
    for group, keys in expected.items():
        for key, dim in keys.items():
            if _stat_dim(stats[group][key]) != dim:
                raise ValueError(f"relative-EEF {group}.{key} statistics changed")
    embodiment = json.loads((root / "embodiment_id.json").read_text())
    if embodiment.get(EMBODIMENT_TAG) != 10:
        raise ValueError("relative-EEF embodiment ID changed")
    return {
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "embodiment_tag": EMBODIMENT_TAG,
        "task": task,
        "state_dim": MODEL_STATE_DIM,
        "decoded_action_dim": MODEL_ACTION_DIM,
        "executable_action_dim": 16,
        "action_horizon": MODEL_ACTION_HORIZON,
        "camera_keys": list(CAMERA_KEYS),
        "lower_body_command_dimensions": 0,
    }
