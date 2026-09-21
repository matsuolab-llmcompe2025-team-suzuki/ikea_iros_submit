"""RAMEN-Ori の model を config から組み立てる (Issue #141 P8-1)。

学習 (`train.py`) と推論 (`inference/desktop/lower_policy/policies/ramen_ori.py`) の
両方がここを呼ぶ。推論側に別の組み立てを書くと、学習で入れた構造 (正規化・state の
切り詰め・QK-norm・memory・FK) を渡し忘れても気付けない。実際、Phase K までの推論は
`base.yaml` + slot の `hydra_overrides` から組み立てていて、再学習の ckpt の構造とは
合わなくなっていた。

推論は `ckpt["cfg"]` (学習に使った config そのもの) を渡す。
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from model.ramen_ori.model import RamenOriPolicy


def build_model(cfg: DictConfig, device: str) -> RamenOriPolicy:
    """`cfg` (学習に使った Hydra config) から RAMEN-Ori の model を組み立てる。

    Hydra の `_target_` swap で sub-module を 1 つずつ instantiate して model に入れる。
    Vision backbone (LingBot / RADIO) は HF の download を伴うのでここで個別に読み、
    VisionEncoder に backbone + embed_dim を渡す。

    Args:
        cfg: 学習時の config。推論では `ckpt["cfg"]` をそのまま渡す (Issue #141 P8-1)。
        device: "cuda" / "cpu"。

    Returns:
        `RamenOriPolicy` (device へ移動済み)。重みは呼び出し側で読む。
    """
    # lazy: 統一 loader (A-1 LingBot / A-2 RADIO を variant で dispatch)。
    # 初回はいずれも HF DL / torch.hub clone を trigger。
    from model.ramen_ori.vision_backbone import load_vision_backbone

    backbone, embed_dim = load_vision_backbone(
        variant=cfg.vision_backbone.variant, device=device, dtype=cfg.vision_backbone.dtype
    )

    # 各 sub-module を Hydra で instantiate (_target_ swap で alternative 差替可)
    vision = hydra.utils.instantiate(cfg.model.vision, backbone=backbone, embed_dim=embed_dim)
    temporal = hydra.utils.instantiate(cfg.model.temporal)
    obb = hydra.utils.instantiate(cfg.model.obb)
    state = hydra.utils.instantiate(cfg.model.state)
    skill = hydra.utils.instantiate(cfg.model.skill)
    fusion = hydra.utils.instantiate(cfg.model.fusion)
    action_expert = hydra.utils.instantiate(cfg.model.action_expert)
    # Alt-3 I-6 Depth aux: cfg.model.aux_head が非 null (dict with _target_) なら instantiate
    aux_head = None
    if cfg.model.get("aux_head") is not None:
        aux_head = hydra.utils.instantiate(cfg.model.aux_head)

    # Issue #129 Phase B (2026-08-31): L4 FK anchor loss。
    # fk: null (default) なら l4_enabled=True 時に model 内で G1WristFKTorch.from_default_urdf() が呼ばれる。
    # fk に _target_ dict 指定なら Hydra instantiate (URDF path override 等したい時)。
    fk = None
    if cfg.model.get("fk") is not None:
        fk = hydra.utils.instantiate(cfg.model.fk)
    # Issue #141 Phase 7: memory の token (null = 使わない、Phase K と同じ構造)
    memory = hydra.utils.instantiate(cfg.model.memory) if cfg.model.get("memory") is not None else None

    # Issue #129 Phase F (2026-08-31): relative action space。data 側と model 側の
    # 両方が stats を必要とするので、model 側は data.relative_stats_path 経由で load。
    use_relative_action = bool(cfg.data.get("use_relative_action", False))
    relative_stats = None
    if use_relative_action:
        from model.ramen_ori.relative_action import load_relative_stats

        stats_path = cfg.data.get("relative_stats_path", None)
        if stats_path is None:
            raise ValueError(
                "data.use_relative_action=True requires data.relative_stats_path "
                "(precompute output from scripts/compute_relative_action_stats.py)"
            )
        relative_stats = load_relative_stats(stats_path)

    model = RamenOriPolicy(
        vision=vision,
        temporal=temporal,
        obb=obb,
        state=state,
        skill=skill,
        fusion=fusion,
        action_expert=action_expert,
        aux_head=aux_head,
        aux_weight=cfg.model.get("aux_weight", 0.1),
        fk=fk,
        l4_enabled=cfg.model.get("l4_enabled", False),
        fk_loss_ratio=cfg.model.fk_loss.ratio,
        fk_weight_max=cfg.model.fk_loss.weight_max,
        use_relative_action=use_relative_action,
        relative_stats=relative_stats,
        normalization=bool((cfg.model.get("normalization") or {}).get("enabled", False)),
        state_clip=cfg.model.normalization.state_clip,
        memory=memory,
        d_model=cfg.model.d_model,
        num_cams=cfg.model.num_cams,
        chunk_len=cfg.model.chunk_len,
        action_dim=cfg.model.action_dim,
        sample_n_steps=cfg.model.sample_n_steps,
    )
    # projection Linear など vision_backbone 外の学習可能 param を device へ移動
    # (backbone 自体は load_pretrained_backbone 側で既に device 指定済)
    return model.to(device)
