"""full orchestrator を boundary から駆動する driver (段階2)。

原さんの desktop orchestrator (vendor/desktop) を build し、DDS I/O を
orchestrator_io のアダプタに差し替え、boundary act(obs) 毎に tick を1回回して
(T,25) を返す。skill 遷移 (perception[YOLO]+dwell+is_complete) は原さんの実装のまま。

skill→variant (leg round):
  rotate_table_base = groot_overlay (53D) / pick = groot_pick_legs_v2 (38D) /
  insert = groot_insert_leg_200k (53D) / rotate_leg = groot_rotate_leg_200k (53D)

worker env: pick=RAMEN_WORKER_PYTHON (lerobot0.6.0) / 53D=RAMEN_WORKER_PYTHON_53D (0.6.1)。
YOLO weight: RAMEN_YOLO_WEIGHT (dev 既定 outputs/yolo_obb/weights/m_lowaug_v4_flat.pt)。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .g1_urdf_fk import G1WristFK
from .orchestrator_io import (
    BoundaryDex1StateSource,
    BoundaryJointStateSource,
    BoundaryWristSource,
    InterceptorActuator,
    assemble_19d,
    build_frame_data,
)
from .taskspace_adapter import groot_chunk_to_taskspace

_VENDOR_DESKTOP = str(Path(__file__).resolve().parent / "vendor" / "desktop")

# leg round の skill → (VlaSkill class 名, policy_config variant)
_STAGE_SKILLS = (
    ("rotate_table_base", "RotateTableBaseVlaSkill", "groot_overlay"),
    ("pick_table_leg", "PickTableLegVlaSkill", "groot_pick_legs_v2"),
    ("insert_table_leg", "InsertTableLegVlaSkill", "groot_insert_leg_200k"),
    ("rotate_leg_to_tighten", "RotateLegToTightenVlaSkill", "groot_rotate_leg_200k"),
)
_TRANSITIONS = {
    "rotate_table_base": ["pick_table_leg"],
    "pick_table_leg": ["insert_table_leg"],
    "insert_table_leg": ["rotate_leg_to_tighten"],
    "rotate_leg_to_tighten": [],
}


class OrchestratorDriver:
    """boundary act(obs) → orchestrator.tick → (T,25)。full skill 遷移を再利用。"""

    def __init__(
        self,
        yolo_weight: str | None = None,
        device: str | None = None,
        dex1_open_fraction: tuple[float, float] = (1.0, 1.0),
        ee_frame_transform: np.ndarray | None = None,
    ):
        if _VENDOR_DESKTOP not in sys.path:
            sys.path.insert(0, _VENDOR_DESKTOP)
        from inference.desktop.orchestrator import Orchestrator, DEFAULT_ENTER_CHECK
        from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
        from inference.desktop.lower_policy.policies.config_loader import (
            load_policy_variant, resolve_policy_class,
        )
        from inference.desktop.lower_policy.policies.deferred import DeferredPolicy
        from inference.desktop.lower_policy.skills import vla_skill as _vla
        from inference.desktop.perception.cleaner import load_cleanup_config
        from inference.desktop.perception.stream import DetectionStream
        from inference.desktop.perception.yolo_obb import YoloObbPerception

        weight = self._resolve_yolo_weight(
            yolo_weight or os.environ.get(
                "RAMEN_YOLO_WEIGHT",
                "/datadrive2/iros_2026_ramen/outputs/yolo_obb/weights/m_lowaug_v4_flat.pt",
            )
        )
        cfg_path = os.path.join(
            _VENDOR_DESKTOP, "inference/desktop/lower_policy/configs/policy_config.yaml"
        )
        self._fk = G1WristFK.from_urdf()
        self._ee_frame_transform = ee_frame_transform
        self._t = 0

        # I/O adapters
        self._joint_src = BoundaryJointStateSource()
        self._dex1_src = BoundaryDex1StateSource(dex1_open_fraction)
        self._wrist_l = BoundaryWristSource()
        self._wrist_r = BoundaryWristSource()
        self._arm = InterceptorActuator("arm")
        self._waist = InterceptorActuator("waist")
        self._hand = InterceptorActuator("hand")

        # perception (YOLO) + cleaner
        perception = YoloObbPerception(weight, device=device)
        cleaner = DetectionStream(load_cleanup_config())

        # skills (lazy DeferredPolicy + interceptor actuators)
        registry = {}
        for skill_name, cls_name, variant in _STAGE_SKILLS:
            entry = load_policy_variant(cfg_path, variant)
            policy = DeferredPolicy(
                resolve_policy_class(entry.policy_type), entry.policy_config,
                label=f"{skill_name}:{variant}",
            )
            VlaCls = getattr(_vla, cls_name)
            registry[skill_name] = VlaCls(
                policy=policy,
                waist_actuator=self._waist,
                hand_actuator=self._hand,
                fk=self._fk,
                dispatch_waist=True,
                motion_limiter=None,
            )
        dispatcher = SkillDispatchLowerPolicy(registry)
        # enter_check は「候補(遷移先)skill」で引かれるので、_TRANSITIONS の values を key に。
        _candidates = {c for cands in _TRANSITIONS.values() for c in cands}
        enter_check = {c: DEFAULT_ENTER_CHECK[c] for c in _candidates}

        self._orch = Orchestrator(
            perception, cleaner, dispatcher,
            initial_skill="rotate_table_base",
            transitions=_TRANSITIONS,
            enter_check=enter_check,
            actuator_send_fn=self._arm.send_action,
            joint_state_source=self._joint_src,
            dex1_state_source=self._dex1_src,
            wrist_left_source=self._wrist_l,
            wrist_right_source=self._wrist_r,
            head_perception_view="left",   # boundary は単一 head を packed で複製
        )

    @staticmethod
    def _resolve_yolo_weight(ref: str) -> str:
        """local .pt path ならそのまま。HF repo[@rev] なら .pt を snapshot_download。

        container では RAMEN_YOLO_WEIGHT に HF ref を渡す:
        Team-RAMEN/IROS2026_RAMEN_Hara_yoloobb_upperpolicy@<rev>。
        """
        if os.path.isfile(ref):
            return ref
        if "/" not in ref:
            return ref   # そのまま (存在しなければ後段で error)
        from huggingface_hub import snapshot_download

        repo_id, revision = ref, None
        if "@" in ref:
            repo_id, revision = ref.rsplit("@", 1)
        snap = Path(snapshot_download(
            repo_id=repo_id, revision=revision, allow_patterns=("*.pt",)
        ))
        # snapshot は repo の nested 構造を保持する (weight は runs/.../weights/best.pt に居る)。
        # Path.glob("*.pt") は非再帰で top-level しか見ず空になるので recursive glob を使う。
        pts = sorted(snap.glob("**/*.pt"))
        if not pts:
            raise FileNotFoundError(f"no .pt in YOLO repo {repo_id}")
        return str(pts[0])

    def act(self, obs: dict) -> dict:
        self._t += 1
        body_q = np.asarray(obs["body_q"], dtype=np.float64)
        self._joint_src.update(body_q, t=self._t)
        images = obs.get("images", {})
        ego = images.get("ego_view")
        head_bgr = (
            np.ascontiguousarray(np.asarray(ego, np.uint8)[:, :, ::-1])
            if ego is not None else np.zeros((480, 640, 3), np.uint8)
        )
        frame = build_frame_data(head_bgr, t=self._t, packed_stereo=True)

        def _bgr(key):
            im = images.get(key)
            return (np.ascontiguousarray(np.asarray(im, np.uint8)[:, :, ::-1])
                    if im is not None else np.zeros((480, 640, 3), np.uint8))
        self._wrist_l.update(_bgr("left_wrist"), t=self._t)
        self._wrist_r.update(_bgr("right_wrist"), t=self._t)

        self._arm.reset(); self._waist.reset(); self._hand.reset()
        result = self._orch.tick(frame)
        arms14 = result.action if (result is not None and result.action is not None) \
            else self._arm.last
        if arms14 is None:
            # buffer 充填中など: 現在姿勢保持で (T,25)
            arms14 = body_q[15:29]
        step19 = assemble_19d(self._waist.last, arms14, self._hand.last)

        body29 = np.concatenate([body_q[:12], step19[0:3], step19[3:17]])
        root = np.array([0, 0, 0.70, 1, 0, 0, 0], dtype=np.float64)
        action38 = np.concatenate([root, body29, step19[17:19]])[None, :]   # (1,38)
        actions = groot_chunk_to_taskspace(
            action38, self._fk, ee_frame_transform=self._ee_frame_transform
        )
        return {"actions": actions, "current_skill":
                getattr(result, "current_skill", None) if result else None}

    def reset(self) -> None:
        self._t = 0

    def close(self) -> None:
        for a in (self._arm, self._waist, self._hand):
            a.reset()
