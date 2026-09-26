"""LeRobot v3 dataset adapter for flow-matching action chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .config import FlowMatchingConfig, POLICY_CAMERAS


# Source MP4 presentation timestamps can differ from the 30 Hz parquet clock by
# about 0.2 ms. One millisecond covers that quantization error while remaining
# far below the 33.3 ms frame interval, so it cannot select an adjacent frame.
VIDEO_TIMESTAMP_TOLERANCE_S = 1.0e-3


class FlowMatchingDataset(Dataset):
    def __init__(
        self,
        *,
        repo_id: str,
        root: str | Path,
        episodes: list[int],
        config: FlowMatchingConfig,
        augment: bool,
        revision: str | None = None,
    ) -> None:
        delta_timestamps = {
            "action": [index / config.fps for index in range(config.action_horizon)]
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=Path(root),
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            revision=revision,
            tolerance_s=VIDEO_TIMESTAMP_TOLERANCE_S,
            download_videos=False,
            video_backend="pyav",
        )
        self.config = config
        self.transform = self._image_transform(augment)

    def _image_transform(self, augment: bool) -> v2.Transform:
        if not augment:
            return v2.Resize(
                (self.config.image_height, self.config.image_width),
                antialias=True,
            )
        return v2.Compose(
            [
                v2.RandomResizedCrop(
                    (self.config.image_height, self.config.image_width),
                    scale=(0.90, 1.0),
                    ratio=(1.25, 1.42),
                    antialias=True,
                ),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.1),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        images = torch.stack([self.transform(sample[key]) for key in POLICY_CAMERAS])
        state = sample["observation.state"].float()
        action = sample["action"].float()
        action_is_pad = sample["action_is_pad"].bool()
        if images.shape != (
            len(POLICY_CAMERAS),
            3,
            self.config.image_height,
            self.config.image_width,
        ):
            raise ValueError(f"unexpected image batch shape: {tuple(images.shape)}")
        if state.shape != (self.config.state_dim,):
            raise ValueError(f"unexpected state shape: {tuple(state.shape)}")
        if action.shape != (self.config.action_horizon, self.config.action_dim):
            raise ValueError(f"unexpected action shape: {tuple(action.shape)}")
        return {
            "images": images,
            "state": state,
            "action": action,
            "action_is_pad": action_is_pad,
        }


def load_episode_split(path: str | Path) -> dict[str, list[int]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = value.get("splits", {})
    result: dict[str, list[int]] = {}
    for name in ("train", "validation", "test"):
        entry = splits.get(name)
        indices = entry.get("episode_indices") if isinstance(entry, dict) else None
        if not isinstance(indices, list) or not indices or not all(isinstance(x, int) for x in indices):
            raise ValueError(f"split {name!r} must contain non-empty integer episode_indices")
        result[name] = indices
    if len(set().union(*map(set, result.values()))) != sum(map(len, result.values())):
        raise ValueError("episode splits overlap")
    return result


def load_dataset_stats(root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stats = json.loads((Path(root) / "meta" / "stats.json").read_text(encoding="utf-8"))
    try:
        return stats["observation.state"], stats["action"]
    except KeyError as exc:
        raise ValueError("training view stats must contain observation.state and action") from exc
