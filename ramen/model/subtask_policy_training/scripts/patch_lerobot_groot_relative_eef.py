"""Install or verify the LeRobot patch required for native GR00T relative EEF actions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
from pathlib import Path


LEROBOT_VERSION = "0.6.0"
ORIGINAL_SHA256 = "e6ee08e9705d83f889a39f8c801a45100bc44e359c38438bd2587149663a027f"
PATCH_MARKER_V1 = "# TEAM_RAMEN_GR00T_RELATIVE_EEF_TRAINING_V1"
PATCH_MARKER_V2 = "# TEAM_RAMEN_GR00T_RELATIVE_EEF_TRAINING_V2"
PATCH_MARKER = "# TEAM_RAMEN_GR00T_RELATIVE_EEF_TRAINING_V3"

HELPER_ANCHOR = '''def _feature_group_key(name: str) -> str:
    base = name.removesuffix(".pos").split(".")[-1]
    return base.replace(" ", "_") or "action"
'''

HELPER_REPLACEMENT = '''# TEAM_RAMEN_GR00T_RELATIVE_EEF_TRAINING_V3
def _rot6d_rows_to_matrix_torch(rot6d: torch.Tensor) -> torch.Tensor:
    rows = rot6d.reshape(*rot6d.shape[:-1], 2, 3)
    row0 = torch.nn.functional.normalize(rows[..., 0, :], dim=-1, eps=1e-12)
    row1 = rows[..., 1, :] - (row0 * rows[..., 1, :]).sum(dim=-1, keepdim=True) * row0
    row1 = torch.nn.functional.normalize(row1, dim=-1, eps=1e-12)
    row2 = torch.linalg.cross(row0, row1, dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _absolute_eef_xyz_rot6d_to_relative_torch(
    target: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Compute inv(T_reference) @ T_target for a complete action chunk."""
    if target.shape[-1] != 9 or reference.shape[-1] != 9:
        raise ValueError(
            "Relative EEF conversion requires 9-D XYZ+ROT6D target and state groups, "
            f"got {target.shape[-1]} and {reference.shape[-1]}."
        )
    reference = reference.to(device=target.device, dtype=target.dtype)
    reference_rotation = _rot6d_rows_to_matrix_torch(reference[..., 3:]).unsqueeze(1)
    target_rotation = _rot6d_rows_to_matrix_torch(target[..., 3:])
    reference_rotation_t = reference_rotation.transpose(-1, -2)
    relative_translation = torch.matmul(
        reference_rotation_t,
        (target[..., :3] - reference[:, None, :3]).unsqueeze(-1),
    ).squeeze(-1)
    relative_rotation = torch.matmul(reference_rotation_t, target_rotation)
    relative_rot6d = relative_rotation[..., :2, :].reshape(*target.shape[:-1], 6)
    return torch.cat((relative_translation, relative_rot6d), dim=-1)


def _feature_group_key(name: str) -> str:
    base = name.removesuffix(".pos").split(".")[-1]
    return base.replace(" ", "_") or "action"
'''

CONVERSION_ANCHOR = '''            if config_value(cfg.get("rep")) == "relative":
                action_type = config_value(cfg.get("type"))
                if action_type != "non_eef":
                    raise ValueError(f"Unsupported relative N1.7 action config for '{key}': {cfg}")
                state_key = cfg.get("state_key") or key
                reference = state_groups.get(state_key)
                if reference is None:
                    raise KeyError(f"Missing raw state group '{state_key}' for relative N1.7 action '{key}'")
                if reference.shape[-1] != dim:
                    raise ValueError(
                        f"Relative N1.7 action group '{key}' has dim {dim}, but state group "
                        f"'{state_key}' has dim {reference.shape[-1]}."
                    )
                if not cloned:
                    converted = action.clone()
                    cloned = True
                converted[..., start_idx:end_idx] -= reference[:, None, :]
'''

CONVERSION_REPLACEMENT = '''            if config_value(cfg.get("rep")) == "relative":
                action_type = config_value(cfg.get("type"))
                action_format = config_value(cfg.get("format"))
                state_key = cfg.get("state_key") or key
                reference = state_groups.get(state_key)
                if reference is None:
                    raise KeyError(f"Missing raw state group '{state_key}' for relative N1.7 action '{key}'")
                if not cloned:
                    converted = action.clone()
                    cloned = True
                if action_type == "non_eef":
                    if reference.shape[-1] != dim:
                        raise ValueError(
                            f"Relative N1.7 action group '{key}' has dim {dim}, but state group "
                            f"'{state_key}' has dim {reference.shape[-1]}."
                        )
                    converted[..., start_idx:end_idx] -= reference[:, None, :]
                elif action_type == "eef" and action_format == "xyz+rot6d":
                    converted[..., start_idx:end_idx] = _absolute_eef_xyz_rot6d_to_relative_torch(
                        action[..., start_idx:end_idx], reference
                    )
                else:
                    raise ValueError(f"Unsupported relative N1.7 action config for '{key}': {cfg}")
'''

PROCESSOR_ASSET_ANCHOR = '''def _build_n1_7_processor(model_name: str = GROOT_N1_7_BACKBONE_MODEL) -> ProcessorMixin:
    require_package("transformers", extra="groot")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    video_processor = Qwen3VLVideoProcessor.from_pretrained(model_name, trust_remote_code=True)
'''

PROCESSOR_ASSET_REPLACEMENT = '''def _build_n1_7_processor(model_name: str = GROOT_N1_7_BACKBONE_MODEL) -> ProcessorMixin:
    require_package("transformers", extra="groot")
    # Cosmos-Reason2-2B is gated separately from GR00T-N1.7. Its seven
    # tokenizer/image/video processor files have the exact same Hugging Face
    # Git blob IDs as its public Qwen3-VL-2B-Instruct base model. Load only
    # those processor assets from the public source; GR00T still loads and
    # trains the Cosmos backbone weights embedded in its own checkpoint.
    processor_assets_model = (
        "Qwen/Qwen3-VL-2B-Instruct" if model_name == GROOT_N1_7_BACKBONE_MODEL else model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(processor_assets_model, trust_remote_code=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(processor_assets_model, trust_remote_code=True)
    video_processor = Qwen3VLVideoProcessor.from_pretrained(processor_assets_model, trust_remote_code=True)
'''

ACTION_MASK_ANCHOR = '''            action_mask[:, :, :valid_dim] = horizon_valid.unsqueeze(-1).to(dtype=action_mask.dtype)
            transition[TransitionKey.ACTION] = action
'''

ACTION_MASK_REPLACEMENT = '''            action_mask[:, :, :valid_dim] = horizon_valid.unsqueeze(-1).to(dtype=action_mask.dtype)
            loss_excluded_indices: list[int] = []
            if isinstance(self.modality_config, dict):
                configured_indices = self.modality_config.get("action", {}).get(
                    "loss_excluded_indices", []
                )
                if configured_indices is not None:
                    if not isinstance(configured_indices, list) or any(
                        isinstance(index, bool) or not isinstance(index, int)
                        for index in configured_indices
                    ):
                        raise ValueError(
                            "action.loss_excluded_indices must be a list of integer action indices"
                        )
                    loss_excluded_indices = configured_indices
            for index in loss_excluded_indices:
                if not 0 <= index < valid_dim:
                    raise ValueError(
                        f"action loss exclusion index {index} is outside the valid action dimension {valid_dim}"
                    )
                action_mask[:, :, index] = 0.0
            transition[TransitionKey.ACTION] = action
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify that the patch is already active")
    parser.add_argument(
        "--overlay-root",
        type=Path,
        help="copy LeRobot into this writable PYTHONPATH root before applying the patch",
    )
    return parser.parse_args()


def processor_path() -> Path:
    spec = importlib.util.find_spec("lerobot.policies.groot.processor_groot")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate lerobot.policies.groot.processor_groot")
    return Path(spec.origin)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_source_overlay(source_processor: Path, overlay_root: Path) -> Path:
    """Copy the installed LeRobot package into a writable, reusable overlay."""
    source_processor = source_processor.resolve()
    source_package = source_processor.parents[2]
    if source_package.name != "lerobot" or not (source_package / "__init__.py").is_file():
        raise RuntimeError(f"unexpected LeRobot package layout: {source_package}")

    overlay_root = overlay_root.expanduser().resolve()
    target_package = overlay_root / "lerobot"
    target_processor = target_package / "policies" / "groot" / "processor_groot.py"
    if target_package.exists():
        if not target_processor.is_file():
            raise RuntimeError(f"incomplete LeRobot source overlay: {overlay_root}")
        return target_processor

    overlay_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = overlay_root.with_name(f".{overlay_root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(source_package, temporary / "lerobot", symlinks=True)
        try:
            temporary.replace(overlay_root)
        except OSError:
            # A concurrently starting rank/job may have completed the same
            # immutable 0.6.0 package copy first.
            if not target_processor.is_file():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target_processor


def patch_processor(path: Path, *, check_only: bool = False) -> bool:
    version = importlib.metadata.version("lerobot")
    if version != LEROBOT_VERSION:
        raise RuntimeError(f"expected lerobot=={LEROBOT_VERSION}, found {version}")

    source = path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        if (
            "_absolute_eef_xyz_rot6d_to_relative_torch" not in source
            or "Qwen/Qwen3-VL-2B-Instruct" not in source
            or "loss_excluded_indices" not in source
        ):
            raise RuntimeError(f"incomplete relative-EEF patch in {path}")
        return False
    if check_only:
        raise RuntimeError(f"relative-EEF training patch is not active in {path}")

    if PATCH_MARKER_V2 in source:
        if source.count(ACTION_MASK_ANCHOR) != 1:
            raise RuntimeError(f"LeRobot action-mask patch anchor is not unique in {path}")
        patched = source.replace(PATCH_MARKER_V2, PATCH_MARKER, 1).replace(
            ACTION_MASK_ANCHOR, ACTION_MASK_REPLACEMENT
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(patched, encoding="utf-8")
        temporary.replace(path)
        return True

    if PATCH_MARKER_V1 in source:
        if source.count(PROCESSOR_ASSET_ANCHOR) != 1 or source.count(ACTION_MASK_ANCHOR) != 1:
            raise RuntimeError(f"LeRobot V1 patch anchors are not unique in {path}")
        patched = (
            source.replace(PATCH_MARKER_V1, PATCH_MARKER, 1)
            .replace(PROCESSOR_ASSET_ANCHOR, PROCESSOR_ASSET_REPLACEMENT)
            .replace(ACTION_MASK_ANCHOR, ACTION_MASK_REPLACEMENT)
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(patched, encoding="utf-8")
        temporary.replace(path)
        return True

    digest = sha256_text(source)
    if digest != ORIGINAL_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected LeRobot source {path}: sha256={digest}, "
            f"expected {ORIGINAL_SHA256}"
        )
    if (
        source.count(HELPER_ANCHOR) != 1
        or source.count(CONVERSION_ANCHOR) != 1
        or source.count(PROCESSOR_ASSET_ANCHOR) != 1
        or source.count(ACTION_MASK_ANCHOR) != 1
    ):
        raise RuntimeError(f"LeRobot patch anchors are not unique in {path}")

    patched = (
        source.replace(HELPER_ANCHOR, HELPER_REPLACEMENT)
        .replace(CONVERSION_ANCHOR, CONVERSION_REPLACEMENT)
        .replace(PROCESSOR_ASSET_ANCHOR, PROCESSOR_ASSET_REPLACEMENT)
        .replace(ACTION_MASK_ANCHOR, ACTION_MASK_REPLACEMENT)
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(patched, encoding="utf-8")
    temporary.replace(path)
    return True


def main() -> None:
    args = parse_args()
    path = processor_path()
    if args.overlay_root is not None:
        if args.check:
            raise ValueError("--check and --overlay-root cannot be used together")
        path = prepare_source_overlay(path, args.overlay_root)
    changed = patch_processor(path, check_only=args.check)
    status = "patched" if changed else "verified"
    print(f"LeRobot GR00T relative-EEF processor {status}: {path}")


if __name__ == "__main__":
    main()
