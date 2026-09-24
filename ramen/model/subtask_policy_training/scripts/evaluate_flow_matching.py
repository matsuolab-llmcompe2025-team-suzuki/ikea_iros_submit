#!/usr/bin/env python3
"""Evaluate a trained Flow Matching policy on a held-out LeRobot split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.subtask_policy_training.flow_matching import FlowMatchingPolicy
from model.subtask_policy_training.flow_matching.data import (
    FlowMatchingDataset,
    load_episode_split,
)
from model.subtask_policy_training.scripts.train_flow_matching import validate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default="Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1",
    )
    parser.add_argument("--revision")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-action-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if args.max_samples < 0 or args.max_action_samples <= 0:
        raise ValueError("sample limits must be non-negative, with action samples positive")
    if args.output.exists():
        raise FileExistsError(args.output)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    policy = FlowMatchingPolicy.from_pretrained(args.checkpoint, device=device)
    policy.requires_grad_(False)
    split = load_episode_split(args.split_file)
    dataset = FlowMatchingDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=split[args.split],
        config=policy.config,
        augment=False,
        revision=args.revision,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=False,
    )
    max_samples = len(dataset) if args.max_samples == 0 else min(args.max_samples, len(dataset))
    metrics = validate(
        policy,
        loader,
        device=device,
        max_samples=max_samples,
        max_action_samples=min(args.max_action_samples, max_samples),
    )
    model_path = args.checkpoint / "model.safetensors"
    report = {
        "schema_version": "team_ramen_flow_matching_evaluation_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "model_sha256": _sha256(model_path),
        "dataset_root": str(args.dataset_root.resolve()),
        "repo_id": args.repo_id,
        "split_file": str(args.split_file.resolve()),
        "split": args.split,
        "episode_count": len(split[args.split]),
        "available_frames": len(dataset),
        "evaluated_frames": max_samples,
        "metrics": metrics,
        "policy_inputs": [
            "head-left RGB",
            "left D405 RGB",
            "right D405 RGB",
            "19D upper-body joint state",
        ],
        "policy_output": "19D upper-body absolute joint target chunk",
        "privileged_inputs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
