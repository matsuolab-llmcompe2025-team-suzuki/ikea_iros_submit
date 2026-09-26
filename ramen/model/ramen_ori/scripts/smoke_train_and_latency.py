"""RAMEN-Ori Phase 1 M11 smoke: 100 iter train + inference latency measurement。

Design doc §7 Phase 1 Success criteria:
- pick_table_leg で BC training が完走 (loss 減少)
- Inference で 1 chunk (16 step × 19 dim) が正常に出力される
- Latency: RTX 5090 で 1 forward < 50ms (target 30Hz)

本 host は RTX 3060 Ti (8GB) しかないため、RTX 5090 の proxy として測定。
実 5090 では 2-3x 高速化見込み。

使用: `cd model/ramen_ori && pixi run python -m model.ramen_ori.scripts.smoke_train_and_latency`
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from model.ramen_ori.data import DummyRamenOriDataset
from model.ramen_ori.model import RamenOriPolicy


def main() -> None:
    device = "cuda"
    torch.manual_seed(42)

    # === Build model with Phase 1 default config ===
    from lingbot_vision import load_pretrained_backbone

    print("[setup] loading LingBot-B backbone ...")
    backbone, embed_dim = load_pretrained_backbone(
        variant="base", device=device, dtype="fp32"
    )
    model = RamenOriPolicy(vision_backbone=backbone, vision_embed_dim=embed_dim).to(
        device
    )
    n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        f"[model] learnable={n_learn / 1e6:.1f}M / frozen={n_frozen / 1e6:.1f}M "
        f"(total {(n_learn + n_frozen) / 1e6:.1f}M)"
    )

    # === Data ===
    dataset = DummyRamenOriDataset(num_samples=200)
    batch_size = 2  # 3060 Ti 8GB での実測、5090 32GB なら 8-16 可
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"[data] batch_size={batch_size}")

    # === Optim ===
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    # === Training 100 iter ===
    print("\n[training] 100 iter smoke ...")
    losses: list[float] = []
    model.train()
    torch.cuda.synchronize()
    t0 = time.time()
    for step, batch in enumerate(loader):
        if step >= 100:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model.compute_loss(batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if step % 10 == 0 or step == 99:
            elapsed = time.time() - t0
            print(f"  [step {step:3d}] loss={loss.item():.4f} elapsed={elapsed:.1f}s")
    torch.cuda.synchronize()
    total_train_time = time.time() - t0
    print(
        f"[training] 100 iter in {total_train_time:.1f}s "
        f"({total_train_time / 100 * 1000:.0f}ms/iter avg)"
    )
    loss_first = sum(losses[:5]) / 5
    loss_last = sum(losses[-5:]) / 5
    print(
        f"[loss] first 5 avg={loss_first:.4f} → last 5 avg={loss_last:.4f} "
        f"(delta={loss_last - loss_first:+.4f})"
    )

    # === Inference latency ===
    print("\n[inference] latency measurement ...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}

        # Warmup
        for _ in range(3):
            context, mask = model._encode(batch)
            x = torch.randn(batch_size, 16, 19, device=device)
            t = torch.rand(batch_size, device=device)
            _ = model.action_expert(x, t, context, mask)
        torch.cuda.synchronize()

        # Measure 1 action_expert forward (single denoise step)
        n_trials = 20
        t0 = time.time()
        for _ in range(n_trials):
            _ = model.action_expert(x, t, context, mask)
        torch.cuda.synchronize()
        per_action_forward = (time.time() - t0) / n_trials * 1000
        print(
            f"  1 action_expert.forward()      : {per_action_forward:6.2f}ms "
            f"(target < 50ms on RTX 5090)"
        )

        # Measure 1 full _encode (Vision + Delta + OBB + State + Skill + Fusion)
        t0 = time.time()
        for _ in range(n_trials):
            _ = model._encode(batch)
        torch.cuda.synchronize()
        per_encode = (time.time() - t0) / n_trials * 1000
        print(f"  1 _encode() (all path + Fusion): {per_encode:6.2f}ms")

        # Full predict_action (6-step Euler)
        t0 = time.time()
        for _ in range(n_trials):
            action = model.predict_action(batch)
        torch.cuda.synchronize()
        per_predict = (time.time() - t0) / n_trials * 1000
        print(
            f"  full predict_action (6-step)    : {per_predict:6.2f}ms "
            f"(_encode + 6 × action_expert)"
        )
        print(
            f"  → action chunk shape={tuple(action.shape)} "
            f"(chunk_len={model.chunk_len}, action_dim={model.action_dim})"
        )

    print("\n[done] M11 smoke complete")


if __name__ == "__main__":
    main()
