"""保存済み ckpt を val で推論と同じ生成にかけ、skill 別に誤差を出す (Issue #155)。

学習中の val の EE 誤差は「chunk の先頭 8 行・手先の位置だけ」しか見ていない。一方、推論の
temporal ensemble (`temporal_lambda=-0.1`、8 tick ごとに 32 行の chunk) では、1 tick の指令の
約 43% が 9 行目以降から来る。手の向きと hand の開閉も実行される。ここではそれらを ckpt ごと・
skill ごとに測る。

測るもの (いずれも `predict_action` = 推論と同じ `sample_n_steps` の生成、EMA の重み):

- pos_mm:  手先位置の誤差 (左右平均) [mm]
- rot_deg: 手の向き (wrist_yaw_link の回転) の誤差 (左右平均、測地角) [deg]
- hand:    hand の開閉の指令の絶対誤差 (左右平均、元の単位)
- 行のまとまり: 先頭 8 行 / 9〜32 行 / ensemble 相当 (行 r を exp(-0.1 r) で重み付け。
  1 tick の指令に対する行ごとの寄与の平均と同じ比率)

val は学習と同じ split と `skill_balanced_val_positions` (6 skill × 224 sample)。生成の noise と
焼き込み variant の選び方は seed を固定するので、ckpt 間で同じ条件になる。

usage (学習の env、repo root で):
    python -m model.ramen_ori.scripts.eval_ckpt_val \
        --ckpt-dir /nvme/train_outputs/ramen_ori_all6_state_dropout \
        --steps 100000 200000 --out /nvme/logs/eval_ckpt_val.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from model.ramen_ori.build import build_model
from model.ramen_ori.fk import G1WristFKTorch, assemble_action19
from model.ramen_ori.skill_mapping import skill_id_name
from model.ramen_ori.state_derive import ACTION16_HAND_SLICE
from model.ramen_ori.train import skill_balanced_val_positions

TEMPORAL_LAMBDA = -0.1   # policy_config.yaml の RAMEN-Ori slot と同じ
EXECUTION_STEPS = 8
SEED = 0


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    shadow = ckpt["ema_state_dict"]
    learnable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    missing = sorted(set(learnable) - set(shadow))
    if missing:
        raise RuntimeError(f"EMA shadow に無い学習対象の重み {missing[:5]} (計 {len(missing)})")
    with torch.no_grad():
        for n, p in learnable.items():
            p.copy_(shadow[n].to(p.device, p.dtype))
    model.eval()
    return model, cfg


def build_val_loader(cfg, batch_size: int, num_workers: int) -> DataLoader:
    aug = OmegaConf.to_container(cfg.augmentation, resolve=True) if cfg.get("augmentation") else None
    dataset = hydra.utils.instantiate(cfg.data, augmentation_cfg=aug)
    _, val_view, _ = dataset.make_train_val_test_split(
        val_ratio=cfg.val.val_ratio, test_ratio=cfg.val.test_ratio, seed=cfg.val.seed, split_json=None
    )
    val_skills = dataset.sample_skill_ids()[np.asarray(val_view.indices, dtype=np.int64)]
    pos = skill_balanced_val_positions(val_skills, batch_size, cfg.val.val_max_batches)
    gen = torch.Generator().manual_seed(SEED)   # worker の seed (= 焼き込み variant の選び方) を固定
    return DataLoader(Subset(val_view, pos.tolist()), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, generator=gen, drop_last=False)


def geodesic_deg(ra: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) の回転 2 つの間の角度 [deg]。"""
    tr = (ra.transpose(-1, -2) @ rb).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1.0, 1.0)))


@torch.no_grad()
def evaluate(model, loader, fk, device: str) -> dict:
    chunk = model.action_expert.chunk_len if hasattr(model.action_expert, "chunk_len") else None
    acc: dict[str, dict[str, list]] = {}
    torch.manual_seed(SEED)   # 生成の noise を ckpt 間でそろえる
    # worker の seed は iterator を作るたびに loader.generator から引かれる。loader は ckpt 間で
    # 使い回すので、振り直さないと 2 個目以降の ckpt が別の焼き込み variant を引く
    loader.generator.manual_seed(SEED)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model.predict_action(batch).float()          # (B, T, 16) 元の単位
        teach = batch["action"].float()
        waist = batch["action_waist_teacher"].float()
        T = pred.shape[1]
        chunk = chunk or T
        dp = fk.forward_detailed(assemble_action19(waist, pred))
        dt = fk.forward_detailed(assemble_action19(waist, teach))
        pos = ((dp["left_hand"] - dt["left_hand"]).norm(dim=-1)
               + (dp["right_hand"] - dt["right_hand"]).norm(dim=-1)) / 2 * 1000     # (B, T) mm
        rot = (geodesic_deg(dp["left_rot"], dt["left_rot"])
               + geodesic_deg(dp["right_rot"], dt["right_rot"])) / 2                  # (B, T) deg
        hand = (pred[..., ACTION16_HAND_SLICE] - teach[..., ACTION16_HAND_SLICE]).abs().mean(-1)
        valid = (~batch["action_is_pad"]).float()                                     # (B, T)
        for b in range(pred.shape[0]):
            name = skill_id_name(int(batch["skill_id"][b]))
            d = acc.setdefault(name, {"pos": [], "rot": [], "hand": [], "valid": []})
            d["pos"].append(pos[b].cpu().numpy()); d["rot"].append(rot[b].cpu().numpy())
            d["hand"].append(hand[b].cpu().numpy()); d["valid"].append(valid[b].cpu().numpy())

    w = np.exp(TEMPORAL_LAMBDA * np.arange(chunk))
    out = {}
    for name, d in acc.items():
        v = np.stack(d["valid"])
        res = {"n": int(v.shape[0])}
        for key in ("pos", "rot", "hand"):
            x = np.stack(d[key])
            for grp, rows, wt in (("head", slice(0, EXECUTION_STEPS), None),
                                  ("tail", slice(EXECUTION_STEPS, chunk), None),
                                  ("ens", slice(0, chunk), w)):
                xm, vm = x[:, rows], v[:, rows]
                ww = vm if wt is None else vm * wt[rows]
                res[f"{key}_{grp}"] = float((xm * ww).sum() / max(ww.sum(), 1e-9))
        out[name] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    fk = G1WristFKTorch.from_default_urdf(dtype=torch.float32).to(args.device)
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    loader = None
    for step in args.steps:
        if str(step) in results:
            print(f"[eval] step {step}: 既にある、skip"); continue
        t0 = time.time()
        model, cfg = load_model(args.ckpt_dir / f"ckpt_step_{step:06d}.pt", args.device)
        if loader is None:   # data は ckpt 間で共通 (同じ run の cfg)
            loader = build_val_loader(cfg, cfg.training.batch_size, args.num_workers)
        results[str(step)] = evaluate(model, loader, fk, args.device)
        args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False))   # 1 ckpt ごとに保存
        del model; torch.cuda.empty_cache()
        print(f"[eval] step {step} {time.time() - t0:.0f}s")
        for name, r in results[str(step)].items():
            print(f"  {name:24} pos head/tail/ens {r['pos_head']:6.2f}/{r['pos_tail']:6.2f}/{r['pos_ens']:6.2f} mm"
                  f"  rot {r['rot_head']:5.2f}/{r['rot_tail']:5.2f}/{r['rot_ens']:5.2f} deg"
                  f"  hand {r['hand_head']:.3f}/{r['hand_tail']:.3f}/{r['hand_ens']:.3f}")


if __name__ == "__main__":
    main()
