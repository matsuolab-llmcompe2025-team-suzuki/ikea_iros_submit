"""Compare an ACT checkpoint on matched real and simulator observations."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from safetensors.torch import load_file
from safetensors.torch import load_model as load_model_as_safetensor

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


FEATURE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = FEATURE_ROOT / "outputs" / "train" / "act_flip_table_upper_body"
DEFAULT_DATASET_ROOT = FEATURE_ROOT / "outputs" / "training_views" / "act_flip_table_upper_body"
CAMERA_KEYS = (
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
SIM_CAMERA_FILES = {
    "observation.images.head_left": "head_left_rgb.png",
    "observation.images.left_wrist": "left_wrist_rgb.png",
    "observation.images.right_wrist": "right_wrist_rgb.png",
}
ARM_SLICE = slice(0, 17)
HAND_SLICE = slice(17, 19)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether an ACT flip_table checkpoint produces useful arm actions "
            "on real training observations versus simulator observations."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1")
    parser.add_argument("--sim-frame-dir", type=Path, required=True)
    parser.add_argument("--sim-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=FEATURE_ROOT / "outputs" / "diagnostics" / "act_flip_table_output_diagnosis.json")
    parser.add_argument("--num-real-samples", type=int, default=18)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--tolerance-s", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()

    policy = load_policy(checkpoint, device=device)
    pre_path, pre_eps = resolve_processor_state(
        checkpoint,
        "policy_preprocessor.json",
        "normalizer_processor",
    )
    post_path, post_eps = resolve_processor_state(
        checkpoint,
        "policy_postprocessor.json",
        "unnormalizer_processor",
    )
    if pre_eps != post_eps:
        raise ValueError(f"normalizer eps mismatch: {pre_eps} != {post_eps}")
    pre_stats = load_file(str(pre_path), device=str(device))
    post_stats = load_file(str(post_path), device=str(device))

    dataset = LeRobotDataset(
        args.repo_id,
        root=dataset_root,
        download_videos=False,
        video_backend=args.video_backend,
        return_uint8=False,
        tolerance_s=args.tolerance_s,
    )
    table = load_data_table(dataset_root)
    states = fixed_list_column_to_numpy(table, "observation.state")
    actions = fixed_list_column_to_numpy(table, "action")
    episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    episodes = load_episode_rows(dataset_root)
    sample_indices = select_sample_indices(episodes, args.num_real_samples, args.chunk_size)

    real_results: list[dict[str, Any]] = []
    for index in sample_indices:
        item = dataset[int(index)]
        state = item["observation.state"].to(dtype=torch.float32)
        images = {key: item[key].to(dtype=torch.float32) for key in CAMERA_KEYS}
        pred = predict_raw_chunk(
            policy,
            pre_stats,
            post_stats,
            state,
            images,
            device=device,
            normalizer_eps=pre_eps,
        )
        ep = int(episode_index[index])
        ep_to = int(episodes[ep]["dataset_to_index"])
        gt = torch.as_tensor(actions[index : min(ep_to, index + args.chunk_size)], dtype=torch.float32)
        real_results.append(
            {
                "sample_index": int(index),
                "episode_index": ep,
                "frame_index": int(frame_index[index]),
                "metrics": chunk_metrics(pred, state, gt),
                "state": vector_summary(state),
                "image_stats": image_stats(images),
            }
        )

    selected = select_representative_real_sample(real_results)
    selected_index = int(selected["sample_index"])
    selected_item = dataset[selected_index]
    selected_real_state = selected_item["observation.state"].to(dtype=torch.float32)
    selected_real_images = {key: selected_item[key].to(dtype=torch.float32) for key in CAMERA_KEYS}

    sim_state = parse_sim_state(args.sim_log)
    sim_images = load_sim_images(args.sim_frame_dir)
    cross_checks = {
        "real_images_real_state": run_single(
            policy, pre_stats, post_stats, selected_real_state, selected_real_images, device, pre_eps
        ),
        "sim_images_sim_state": run_single(
            policy, pre_stats, post_stats, sim_state, sim_images, device, pre_eps
        ),
        "sim_images_real_state": run_single(
            policy, pre_stats, post_stats, selected_real_state, sim_images, device, pre_eps
        ),
        "real_images_sim_state": run_single(
            policy, pre_stats, post_stats, sim_state, selected_real_images, device, pre_eps
        ),
    }

    output = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "sim_frame_dir": str(args.sim_frame_dir),
        "sim_log": str(args.sim_log),
        "device": str(device),
        "num_real_samples": len(real_results),
        "selected_cross_check_sample": {
            "sample_index": selected_index,
            "episode_index": int(selected["episode_index"]),
            "frame_index": int(selected["frame_index"]),
            "selection_reason": "largest ground-truth arm range among sampled real observations",
        },
        "aggregate": aggregate_real_results(real_results),
        "real_results": real_results,
        "cross_checks": cross_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(short_report(output), indent=2))
    print(f"Wrote diagnosis to {args.output}")


def load_policy(checkpoint: Path, *, device: torch.device) -> ACTPolicy:
    raw_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    valid_fields = {field.name for field in dataclasses.fields(ACTPolicy.config_class)}
    filtered_config = {key: value for key, value in raw_config.items() if key in valid_fields}
    if "device" in valid_fields:
        filtered_config["device"] = str(device)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as config_file:
        json.dump(filtered_config, config_file)
        config_path = Path(config_file.name)
    try:
        config = draccus.parse(ACTPolicy.config_class, config_path, args=[])
    finally:
        config_path.unlink(missing_ok=True)

    policy = ACTPolicy(config)
    load_kwargs: dict[str, Any] = {"strict": True}
    try:
        load_kwargs["device"] = str(device)
        result = load_model_as_safetensor(
            policy,
            str(checkpoint / "model.safetensors"),
            **load_kwargs,
        )
    except TypeError:
        load_kwargs.pop("device", None)
        result = load_model_as_safetensor(
            policy,
            str(checkpoint / "model.safetensors"),
            **load_kwargs,
        )
    missing, unexpected = result if result is not None else ((), ())
    if missing or unexpected:
        raise RuntimeError(
            "ACT checkpoint does not exactly match its config: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    policy.to(device)
    policy.eval()
    return policy


def resolve_processor_state(
    checkpoint: Path,
    manifest_name: str,
    registry_name: str,
) -> tuple[Path, float]:
    manifest_path = checkpoint / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    steps = manifest.get("steps") if isinstance(manifest, dict) else None
    if not isinstance(steps, list):
        raise ValueError(f"processor manifest must contain a steps list: {manifest_path}")
    matches = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("registry_name") == registry_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"processor manifest must contain exactly one {registry_name!r} step: {manifest_path}"
        )
    step = matches[0]
    state_file = step.get("state_file")
    if not isinstance(state_file, str) or not state_file:
        raise ValueError(f"processor step {registry_name!r} has no state_file")
    checkpoint_root = checkpoint.resolve()
    state_path = (checkpoint / state_file).resolve()
    try:
        state_path.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError(f"processor state_file escapes checkpoint: {state_file!r}") from exc
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    config = step.get("config")
    eps_value = config.get("eps") if isinstance(config, dict) else None
    if isinstance(eps_value, bool) or not isinstance(eps_value, (int, float)):
        raise ValueError(f"processor step {registry_name!r} has invalid eps")
    eps = float(eps_value)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"processor step {registry_name!r} eps must be finite and positive")
    return state_path, eps


def load_data_table(dataset_root: Path) -> pa.Table:
    files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {dataset_root / 'data'}")
    return pa.concat_tables([pq.read_table(path) for path in files])


def fixed_list_column_to_numpy(table: pa.Table, column: str) -> np.ndarray:
    rows = table[column].to_pylist()
    return np.asarray(rows, dtype=np.float32)


def load_episode_rows(dataset_root: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        data = table.to_pylist()
        for row in data:
            rows[int(row["episode_index"])] = row
    if not rows:
        raise FileNotFoundError(f"no episode metadata under {dataset_root / 'meta' / 'episodes'}")
    return rows


def select_sample_indices(episodes: dict[int, dict[str, Any]], count: int, chunk_size: int) -> list[int]:
    candidates: list[int] = []
    episode_ids = sorted(episodes)
    if count <= 0:
        return candidates
    target_episode_ids = np.linspace(0, len(episode_ids) - 1, num=min(count, len(episode_ids)), dtype=int)
    for ordinal in target_episode_ids:
        row = episodes[episode_ids[int(ordinal)]]
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        length = end - start
        if length <= 2:
            continue
        # Avoid first/last few frames and leave room for a full action chunk when possible.
        usable_end = max(start + 1, end - min(chunk_size, max(1, length // 4)))
        rel = 0.35 if usable_end > start else 0.0
        index = int(round(start + rel * max(0, usable_end - start - 1)))
        candidates.append(min(max(index, start), end - 1))
    return candidates[:count]


def normalize_feature(
    stats: dict[str, torch.Tensor],
    key: str,
    tensor: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mean_key = f"{key}.mean"
    std_key = f"{key}.std"
    if mean_key not in stats or std_key not in stats:
        raise KeyError(f"normalizer statistics are missing {mean_key!r} or {std_key!r}")
    mean = stats[mean_key].to(device=tensor.device, dtype=tensor.dtype)
    std = stats[std_key].to(device=tensor.device, dtype=tensor.dtype)
    return (tensor - mean) / (std + eps)


def unnormalize_action(
    stats: dict[str, torch.Tensor],
    action: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mean = stats["action.mean"].to(device=action.device, dtype=action.dtype)
    std = stats["action.std"].to(device=action.device, dtype=action.dtype)
    return action * (std + eps) + mean


def predict_raw_chunk(
    policy: ACTPolicy,
    pre_stats: dict[str, torch.Tensor],
    post_stats: dict[str, torch.Tensor],
    state: torch.Tensor,
    images: dict[str, torch.Tensor],
    *,
    device: torch.device,
    normalizer_eps: float,
) -> torch.Tensor:
    state = state.to(device=device, dtype=torch.float32)
    batch: dict[str, torch.Tensor] = {
        "observation.state": normalize_feature(
            pre_stats,
            "observation.state",
            state,
            normalizer_eps,
        ).unsqueeze(0)
    }
    for key in CAMERA_KEYS:
        image = images[key].to(device=device, dtype=torch.float32)
        if image.ndim != 3:
            raise ValueError(f"{key} must be CHW image, got {tuple(image.shape)}")
        if float(image.detach().max()) > 2.0:
            image = image / 255.0
        batch[key] = normalize_feature(pre_stats, key, image, normalizer_eps).unsqueeze(0)
    with torch.inference_mode():
        normalized_chunk = policy.predict_action_chunk(batch)
        raw_chunk = unnormalize_action(post_stats, normalized_chunk, normalizer_eps)
    return raw_chunk[0].detach().cpu()


def chunk_metrics(pred: torch.Tensor, state: torch.Tensor, gt: torch.Tensor | None = None) -> dict[str, Any]:
    state = state.detach().cpu().to(dtype=torch.float32)
    pred = pred.detach().cpu().to(dtype=torch.float32)
    first = pred[0]
    arm_range = pred[:, ARM_SLICE].amax(dim=0) - pred[:, ARM_SLICE].amin(dim=0)
    hand_range = pred[:, HAND_SLICE].amax(dim=0) - pred[:, HAND_SLICE].amin(dim=0)
    metrics: dict[str, Any] = {
        "pred_first_arm_delta_norm": float((first[ARM_SLICE] - state[ARM_SLICE]).norm()),
        "pred_first_total_delta_norm": float((first - state).norm()),
        "pred_arm_chunk_max_dim_range": float(arm_range.max()),
        "pred_arm_chunk_range_norm": float(arm_range.norm()),
        "pred_hand_chunk_range": [float(value) for value in hand_range.tolist()],
        "pred_first_action": [float(value) for value in first.tolist()],
        "pred_chunk_mean": [float(value) for value in pred.mean(dim=0).tolist()],
        "pred_chunk_std": [float(value) for value in pred.std(dim=0, unbiased=False).tolist()],
    }
    if gt is not None and gt.numel() > 0:
        gt = gt.detach().cpu().to(dtype=torch.float32)
        gt_range = gt[:, ARM_SLICE].amax(dim=0) - gt[:, ARM_SLICE].amin(dim=0)
        gt_hand_range = gt[:, HAND_SLICE].amax(dim=0) - gt[:, HAND_SLICE].amin(dim=0)
        aligned = pred[: gt.shape[0]]
        metrics.update(
            {
                "gt_first_arm_delta_norm": float((gt[0, ARM_SLICE] - state[ARM_SLICE]).norm()),
                "gt_first_total_delta_norm": float((gt[0] - state).norm()),
                "gt_arm_chunk_max_dim_range": float(gt_range.max()),
                "gt_arm_chunk_range_norm": float(gt_range.norm()),
                "gt_hand_chunk_range": [float(value) for value in gt_hand_range.tolist()],
                "first_action_error_norm": float((first - gt[0]).norm()),
                "first_arm_error_norm": float((first[ARM_SLICE] - gt[0, ARM_SLICE]).norm()),
                "chunk_mse": float(torch.mean((aligned - gt[: aligned.shape[0]]) ** 2)),
                "chunk_arm_mse": float(torch.mean((aligned[:, ARM_SLICE] - gt[: aligned.shape[0], ARM_SLICE]) ** 2)),
            }
        )
    return metrics


def parse_sim_state(log_path: Path) -> torch.Tensor:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    step_match = re.search(r"\[LeRobotACTPolicy\]\[debug\] step=0(?P<body>.*?)(?:\n\[LeRobotACTPolicy\]\[debug\] step=1|\Z)", text, re.S)
    if step_match is None:
        raise ValueError(f"could not find step=0 debug block in {log_path}")
    body = step_match.group("body")
    raw = parse_values_line(body, "raw_action")
    delta = parse_values_line(body, "raw_action-state_before")
    if len(raw) != len(delta):
        raise ValueError("raw_action and raw_action-state_before vector lengths differ")
    return torch.as_tensor([r - d for r, d in zip(raw, delta, strict=True)], dtype=torch.float32)


def parse_values_line(text: str, label: str) -> list[float]:
    pattern = rf"{re.escape(label)}:[^\n]*values=\[(?P<values>[^\]]+)\]"
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"could not find values for {label}")
    return [float(part.strip()) for part in match.group("values").split(",") if part.strip()]


def load_sim_images(frame_dir: Path) -> dict[str, torch.Tensor]:
    images: dict[str, torch.Tensor] = {}
    for key, filename in SIM_CAMERA_FILES.items():
        path = frame_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        images[key] = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return images


def run_single(
    policy: ACTPolicy,
    pre_stats: dict[str, torch.Tensor],
    post_stats: dict[str, torch.Tensor],
    state: torch.Tensor,
    images: dict[str, torch.Tensor],
    device: torch.device,
    normalizer_eps: float,
) -> dict[str, Any]:
    pred = predict_raw_chunk(
        policy,
        pre_stats,
        post_stats,
        state,
        images,
        device=device,
        normalizer_eps=normalizer_eps,
    )
    return {
        "metrics": chunk_metrics(pred, state),
        "state": vector_summary(state),
        "image_stats": image_stats(images),
    }


def select_representative_real_sample(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("no real results")
    return max(results, key=lambda item: item["metrics"].get("gt_arm_chunk_range_norm", -1.0))


def aggregate_real_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "pred_first_arm_delta_norm",
        "pred_first_total_delta_norm",
        "pred_arm_chunk_max_dim_range",
        "pred_arm_chunk_range_norm",
        "gt_first_arm_delta_norm",
        "gt_first_total_delta_norm",
        "gt_arm_chunk_max_dim_range",
        "gt_arm_chunk_range_norm",
        "first_action_error_norm",
        "first_arm_error_norm",
        "chunk_mse",
        "chunk_arm_mse",
    ]
    aggregate: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([item["metrics"][key] for item in results if key in item["metrics"]], dtype=np.float64)
        if values.size == 0:
            continue
        aggregate[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return aggregate


def vector_summary(vector: torch.Tensor) -> dict[str, Any]:
    value = vector.detach().cpu().to(dtype=torch.float32)
    return {
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "arm": [float(item) for item in value[ARM_SLICE].tolist()],
        "hand": [float(item) for item in value[HAND_SLICE].tolist()],
    }


def image_stats(images: dict[str, torch.Tensor]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, image in images.items():
        value = image.detach().cpu().to(dtype=torch.float32)
        if float(value.max()) > 2.0:
            value = value / 255.0
        output[key] = {
            "shape": list(value.shape),
            "min": float(value.min()),
            "max": float(value.max()),
            "mean": [float(item) for item in value.mean(dim=(1, 2)).tolist()],
            "std": [float(item) for item in value.std(dim=(1, 2), unbiased=False).tolist()],
        }
    return output


def short_report(output: dict[str, Any]) -> dict[str, Any]:
    cross = {}
    for name, result in output["cross_checks"].items():
        metrics = result["metrics"]
        cross[name] = {
            "pred_first_arm_delta_norm": metrics["pred_first_arm_delta_norm"],
            "pred_arm_chunk_max_dim_range": metrics["pred_arm_chunk_max_dim_range"],
            "pred_arm_chunk_range_norm": metrics["pred_arm_chunk_range_norm"],
            "pred_hand_chunk_range": metrics["pred_hand_chunk_range"],
        }
    return {
        "num_real_samples": output["num_real_samples"],
        "aggregate": output["aggregate"],
        "selected_cross_check_sample": output["selected_cross_check_sample"],
        "cross_checks": cross,
    }


if __name__ == "__main__":
    main()
