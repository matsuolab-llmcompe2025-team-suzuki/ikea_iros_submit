"""full orchestrator を boundary から駆動する driver (段階2)。

原さんの desktop orchestrator (vendor/desktop) を build し、DDS I/O を
orchestrator_io のアダプタに差し替え、boundary act(obs) 毎に tick を1回回して
(T,25) を返す。skill 遷移 (perception[YOLO]+dwell+is_complete) は原さんの実装のまま。

skill→variant (leg round) の正本は **`policy_config.yaml` の
`default_variant_by_skill`**。自前経路 (entrypoint) と同じ節を読む (Issue #148)。
会場での差し替えは `RAMEN_VARIANT_<SKILL>` が勝つ。

worker env: pick=RAMEN_WORKER_PYTHON (lerobot0.6.0) / 53D=RAMEN_WORKER_PYTHON_53D (0.6.1)。
YOLO weight: RAMEN_YOLO_WEIGHT (dev 既定 outputs/yolo_obb/weights/m_lowaug_v4_flat.pt)。

会場で焼き直さずに変えられるもの (env):
  RAMEN_VARIANT_<SKILL>   expert の差し替え (例: rotate_table_base を RAMEN-Ori に)
  RAMEN_GPU_MODELS        GPU に置く model 数 (既定 2 = 今 + 次)
  RAMEN_PICK_HYBRID=1     pick を VLM/VLA/MP の hybrid にする (Issue #148、既定 off)
  RAMEN_PICK_VLM_ENDPOINT hybrid の VLM endpoint (既定は同梱 yaml の値)
  RAMEN_PICK_VLM_MODEL    hybrid の served model 名
  RAMEN_ON_TIMEOUT        時間切れの動き (advance/stop)。既定は advance =
                          YOLO が外しても先へ進む。詳細は _load_stage_timeouts
  RAMEN_START_LEG         何本目の脚から始めるか (既定 1)。自前経路の
                          --phase3-start-stage 相当
  RAMEN_END_LEG           何本目を終えたらやめるか (既定 4)。--phase3-end-stage 相当
  RAMEN_START_SKILL       脚の途中から戻すとき。通常は RAMEN_START_LEG だけでよい
  RAMEN_HEAD_MONO=1       head の左右キーを無視して mono を複製する。head カメラが
                          3840x1080 で開けていないとき用 (詳細は _ingest_images)
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .g1_urdf_fk import G1WristFK
from .orchestrator_io import (
    BoundaryDex1StateSource,
    BoundaryJointStateSource,
    BoundaryWristSource,
    InterceptorActuator,
    MeasuredDex1StateSource,
    assemble_19d,
    build_frame_data,
)
from .taskspace_adapter import groot_chunk_to_taskspace

_VENDOR_DESKTOP = str(Path(__file__).resolve().parent / "vendor" / "desktop")

# leg round の skill → VlaSkill class 名。
#
# **どの ckpt で走るかの正本は `policy_config.yaml` の `default_variant_by_skill`**
# (Issue #148)。自前経路 (`entrypoint.fill_policy_variants_from_config`) と同じ節を
# 同じ規則で読む。ここに variant を書くと二重管理になり、実際に 2026-09-21 まで
# pick が v2 (submit) と v1 (本体 config) で食い違っていた。
#
# 会場での差し替えは `RAMEN_VARIANT_<SKILL>` が勝つ (`_variant_override`)。
_STAGE_SKILLS = (
    ("rotate_table_base", "RotateTableBaseVlaSkill"),
    ("pick_table_leg", "PickTableLegVlaSkill"),
    ("insert_table_leg", "InsertTableLegVlaSkill"),
    ("rotate_leg_to_tighten", "RotateLegToTightenVlaSkill"),
)
# 脚 1 本ぶんの列を 4 回まわす。`rotate_leg_to_tighten` から `rotate_table_base` へ
# 戻すことで `SkillState.transition()` が **n_legs_completed を +1** し、
# `_build_transition_ctx` が次の脚の基準 (天板の姿勢) を取り直す。
#
# 自前経路は stage ごとに Orchestrator を作り直して脚を回す (`--phase3-full`) が、
# boundary の server は 1 本の process が動き続けるので、ここでループにする。
_TRANSITIONS = {
    "rotate_table_base": ["pick_table_leg"],
    "pick_table_leg": ["insert_table_leg"],
    "insert_table_leg": ["rotate_leg_to_tighten"],
    "rotate_leg_to_tighten": ["rotate_table_base"],
}
# **1 本目は卓を回さない。** 自前経路の `STAGE_SKILL_SEQUENCES` と同じ:
#   stage 1     pick -> insert -> rotate_leg
#   stage 2..4  rotate_table_base -> pick -> insert -> rotate_leg
# 上のループを `pick_table_leg` から始めれば、ちょうどこの順になる (回転は 3 回)。
#
# 元は `rotate_table_base` 始まりで、4 脚に対して回転が 4 回あった。学習データの
# 1 本目とは違う卓の向きから掴みにいくうえ、頭で `max_seconds_hard` の 30 秒を
# 余分に使う。#1 の 15 で「自前経路と揃える」と言いながら、開始 skill だけ
# 揃っていなかった (2026-09-21 に修正)。
_INITIAL_SKILL = "pick_table_leg"
# 1 脚の skill 数 x 4。ModelResidency に渡す順序 (先読みの範囲を決める)。
_LEGS = 4
# YOLO の enter 条件を持たない遷移先。`rotate_table_base` は自前経路では stage の
# 先頭なので DEFAULT_ENTER_CHECK に無い。脚のループでは「前の脚が終わったら入る」
# ので、advance_finished_skill (dwell/timeout) に任せる。
_YOLO_FREE_ENTRY = frozenset({"rotate_table_base"})
# GPU に置く model の既定数 (RAMEN_GPU_MODELS で上書き)。自前経路の
# entrypoint.DEFAULT_RESIDENT_MODELS と揃える。
_DEFAULT_RESIDENT_MODELS = 2
# 時間切れの既定の動き。**自前経路 (YAML) とは意図的に違う** — 理由は
# `_load_stage_timeouts` の docstring。`RAMEN_ON_TIMEOUT` で上書きできる。
_BOUNDARY_TIMEOUT_ACTION = "advance"
_TIMEOUT_ACTIONS = frozenset({"advance", "stop"})
# カメラが止まったとみなすまでの時間。自前経路と同じ値 (assembly.CAMERA_STALE_TIMEOUT_S)。
# 判定は driver の `_ingest_images` で行う (理由はそこの comment)。
_CAMERA_STALE_TIMEOUT_S = 0.50
# orchestrator 側の同じ判定は **保険**として緩めに回す。`tick()` の中では
# `dispatcher.start()` の同期ロード (起動直後 45s / 切替 8〜16s) を挟むので、
# 0.5s のままだと model を読むたびに誤発火する。
_ORCH_CAMERA_STALE_TIMEOUT_S = 120.0
# `RAMEN_PICK_HYBRID=1` のとき、Dex1 の実測が来るのを待つ tick 数。運営 client は
# 20 Hz 前後で act() を叩くので 100 tick ≒ 5 秒。bridge は 50 Hz なので、
# 正常なら 1 tick 目から入っている。
_DEX1_HYBRID_GRACE_TICKS = 100


def _variant_override(skill_name: str, default: str) -> str:
    """`RAMEN_VARIANT_<SKILL>` で variant を差し替える。

    会場で image を焼き直さずに expert を入れ替えられるようにする。例えば
    `rotate_table_base` を同じ RAMEN-Ori の別 run に振り替えたいとき:

        -e RAMEN_VARIANT_ROTATE_TABLE_BASE=rotate_table_base_ramen_ori_141_c32

    policy_type が違っても `assembly.build_vla_skill` が
    `resolve_policy_class` 経由で正しい class を選ぶので、名前を変えるだけでよい。
    state の次元も使うカメラも policy 自身が `CAMERAS` と `build_state_from_raw`
    で持っていて、VlaSkill はそれを使う。
    """
    key = f"RAMEN_VARIANT_{skill_name.upper()}"
    override = os.environ.get(key, "").strip()
    if not override or override == default:
        return default
    print(
        f"[orch-driver] {skill_name}: variant を {default} -> {override} に差し替え "
        f"({key})",
        file=sys.stderr,
    )
    return override


def _env_flag(key: str) -> bool:
    """`1` / `true` / `yes` / `on` を真とみなす (大小文字は問わない)。"""
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


class _Resume(NamedTuple):
    """どの脚から始めてどこでやめるか。"""

    start_skill: str
    legs_done: int  # 開始時点の `n_legs_completed`
    end_legs: int  # この本数に達したら以後 skill を進めない


def _resume_settings() -> _Resume:
    """`RAMEN_START_LEG` / `RAMEN_END_LEG` / `RAMEN_START_SKILL` を読む。

    自前経路の `--phase3-start-stage` / `--phase3-end-stage` に相当する。
    boundary の server は 1 本の process が走り続けるので stage を分けられないが、
    **会場で 3 本目からやり直せないと困る**ので、同じ粒度を env で持たせる。

        -e RAMEN_START_LEG=3     3 本目から (n_legs_completed=2、卓を回してから pick)
        -e RAMEN_END_LEG=3       3 本目を終えたらそこで止める
        -e RAMEN_START_SKILL=insert_table_leg
                                 脚の途中から戻すとき。**物理的な前提は運用側の責任**
                                 (insert から始めるなら既に脚を握っている必要がある)

    脚番号から開始 skill が決まるのは `STAGE_SKILL_SEQUENCES` と同じ規則:
    1 本目は卓を回さず pick から、2 本目以降は rotate_table_base から。

    `n_legs_completed` は表示用の数ではなく **判定に効く**:
      0     `enter_pick_table_leg` が Kabsch で判定
      1..3  aspect 規則に切り替わる
      >= 4  entry 系が全部 False を返す
    自前経路も `orch.state.n_legs_completed = stage - 1` と種を入れている。
    """
    names = [name for name, _cls in _STAGE_SKILLS]

    def _leg(key: str, default: int) -> int:
        raw = os.environ.get(key, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{key} must be an integer 1..{_LEGS}, got {raw!r}")
        if not 1 <= value <= _LEGS:
            raise ValueError(f"{key} must be 1..{_LEGS}, got {value}")
        return value

    start_leg = _leg("RAMEN_START_LEG", 1)
    end_leg = _leg("RAMEN_END_LEG", _LEGS)
    if end_leg < start_leg:
        raise ValueError(
            f"RAMEN_END_LEG ({end_leg}) must be >= RAMEN_START_LEG ({start_leg})"
        )

    # 1 本目は卓を回さない (STAGE_SKILL_SEQUENCES[1] と同じ)。
    start_skill = _INITIAL_SKILL if start_leg == 1 else "rotate_table_base"
    override = os.environ.get("RAMEN_START_SKILL", "").strip()
    if override:
        if override not in names:
            raise ValueError(
                f"RAMEN_START_SKILL must be one of {names}, got {override!r}"
            )
        start_skill = override

    resume = _Resume(start_skill=start_skill, legs_done=start_leg - 1, end_legs=end_leg)
    if resume != _Resume(_INITIAL_SKILL, 0, _LEGS):
        print(
            f"[orch-driver] 再開: 脚 {start_leg}..{end_leg} / 開始 skill "
            f"{start_skill} / n_legs_completed={resume.legs_done}",
            file=sys.stderr,
        )
    return resume


class _HybridPick(NamedTuple):
    """pick を hybrid に差し替えるのに要るもの一式。"""

    skill_cls: type
    extra_kwargs: dict
    #: hybrid 自身の予算 (`pick_leg_hybrid.yaml` の `runtime.hard_timeout_sec`)。
    #: skill_config の `pick_table_leg.max_seconds_hard` を **上書きする**。
    hard_timeout_sec: float


def _hybrid_pick_settings(skill_config: dict):
    """`RAMEN_PICK_HYBRID=1` のとき、pick を hybrid に差し替える材料を返す。

    返り値は `_HybridPick` か、env が無ければ `None`。
    `None` のときは今までどおり GR00T の `PickTableLegVlaSkill` が使われる
    = **既定の挙動は一切変わらない**。

    会場での切り替え:

        -e RAMEN_PICK_HYBRID=1
        -e RAMEN_PICK_VLM_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
        -e RAMEN_PICK_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

    自前経路 (`entrypoint.py` の `--pick-leg-hybrid`) と同じ構成にする:

    - `phase3_executor` は `rule_based` 固定。VLA の区間 3 には有限の完了条件が
      無く、本番では `_validate_phase3_config` も rule_based しか許さない
    - `dispatch_waist` を **False** にする。hybrid は腰を一切出さず Regular Mode に
      残す。`build_vla_skill` より先に潰さないと MotionLimiter の包絡と
      `BuiltSkill` の契約が食い違う
    - 区間 3 の終点は `insert_table_leg` の frame-0 姿勢 (正本は skill_config.yaml)
    - pick の時間切れは **hybrid 自身の予算** (30s) と `stop` で上書きする。
      skill_config の 21s は従来 pick 用で、VLM の揺らぎを見込んでいない。
      `stop` なのは hybrid が有限 FSM だから: 持ち替えを確認できないまま insert へ
      進むと、脚を落とすか腕同士がぶつかる (自前経路 entrypoint.py と同じ判断)

    **endpoint は skill を組む前に probe する。** VLM が居ないことに気付けるのが
    `pick_table_leg` に入った後だと、会場では「掴まない」としか見えない
    (hybrid は境界が立たないまま `hard_timeout_sec` で HOLD 停止する)。
    """
    if not _env_flag("RAMEN_PICK_HYBRID"):
        return None

    from inference.desktop.lower_policy.initial_pose import initial_pose_from_config
    from inference.desktop.pick_leg_hybrid.real_skill import (
        RealPickLegHybridVlaSkill,
        load_reference_images,
        probe_vlm_endpoint,
    )
    from inference.desktop.pick_leg_hybrid.config import DEFAULT_CONFIG_PATH

    config_path = os.environ.get("RAMEN_PICK_HYBRID_CONFIG", "").strip() or str(
        DEFAULT_CONFIG_PATH
    )
    endpoint = os.environ.get("RAMEN_PICK_VLM_ENDPOINT", "").strip() or None
    model = os.environ.get("RAMEN_PICK_VLM_MODEL", "").strip() or None

    cfg, references = load_reference_images(
        config_path, endpoint_override=endpoint, model_override=model
    )
    served = probe_vlm_endpoint(cfg, references)
    print(
        f"[orch-driver] pick_table_leg=hybrid (VLM {cfg.vlm.model} @ "
        f"{cfg.vlm.endpoint}, served={sorted(served)})",
        file=sys.stderr,
    )

    # hybrid は腰を出さない。`build_vla_skill` が読む前に潰しておく。
    skills = skill_config.setdefault("skills", {})
    skills.setdefault("pick_table_leg", {})["dispatch_waist"] = False

    insert_initial = initial_pose_from_config(skill_config, "insert_table_leg")
    extra = {
        "hybrid_config_path": config_path,
        "hybrid_vlm_endpoint": endpoint,
        "hybrid_vlm_model": model,
        "phase3_executor": "rule_based",
        "next_initial_arm_target": insert_initial.arm_position_rad,
        "next_initial_hand_target": insert_initial.dex1_target_rad,
    }
    return _HybridPick(
        skill_cls=RealPickLegHybridVlaSkill,
        extra_kwargs=extra,
        hard_timeout_sec=float(cfg.runtime.hard_timeout_sec),
    )


def _load_skill_config(vendor_desktop: str) -> dict:
    """`skill_config.yaml` を丸ごと読む (`assembly` が期待する形)。

    `assembly.build_vla_skill` は `skills` を含んだ **top-level の dict** を受け取り、
    `skills.<name>.wrist_tool_offset` / `teacher_joint_range` / `dispatch_waist` /
    `motion_limits` をそこから引く。節だけ渡すと全部 default に落ちる。
    """
    import yaml  # vendor の config_loader と同じく safe_load で読む (RCE 回避)

    cfg_path = os.path.join(
        vendor_desktop, "inference/desktop/lower_policy/configs/skill_config.yaml"
    )
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _stage_variants(cfg_path: str) -> dict[str, str]:
    """`policy_config.yaml` の `default_variant_by_skill` を読む (Issue #148)。

    **どの ckpt で走るかの正本はこの節。** 自前経路の
    `entrypoint.fill_policy_variants_from_config` と同じものを読む。

    無い skill があれば起動時に落とす。会場で「なぜか古い ckpt で走っていた」に
    なるより、起動しない方が安全。`RAMEN_VARIANT_<SKILL>` は後段で上書きする。
    """
    from inference.desktop.lower_policy.policies.config_loader import (
        load_default_variant_by_skill,
    )

    defaults = load_default_variant_by_skill(cfg_path)
    missing = [name for name, _cls in _STAGE_SKILLS if name not in defaults]
    if missing:
        raise ValueError(
            f"{cfg_path}: default_variant_by_skill に {missing} が無い。"
            "leg round の 4 skill は全部ここで既定を持つこと"
        )
    return defaults


def _load_stage_timeouts(vendor_desktop: str) -> tuple[dict, dict]:
    """`skill_config.yaml` の `max_seconds_hard` / `on_timeout` を読む。

    `tick()` は YOLO の `enter_check` しか見ない。自前経路 (`run_live`) はこれを
    受け皿として持っているのに、boundary 経路には渡っていなかったため、YOLO が
    落とすと `rotate_table_base` から永久に出られなかった (2026-09-21 に実 image で
    実測)。同じ YAML を同じ規約で読んで渡す。

    **秒数は YAML、action は大会経路の既定 (`advance`) を使う。**

    脚の 4 skill には他の受け皿が無い: `is_complete` は VlaSkill では常に False
    (`skills/base.py`)、`max_dwell_sec` は `move_to_table` にしか無い。つまり
    時間切れが YOLO 以外の唯一の前進手段で、`stop` にするとそのバックアップが
    消える = YOLO が外した時点でそのエピソードは何も進まないまま終わる。

    YAML 側は #148 で全 skill `stop` になった。あれは **実機 SDK 経路**の判断で、
    空の腕のまま insert へ進んで卓にぶつかるのを防ぐためのもの。妥当だが、
    大会経路は前提が違う: IK は運営 WBC がやり、episode は `reset` できて、
    止まっても安全なだけで点は増えない。なのでここだけ `advance` に倒す。

    会場で危ないと判断したら `-e RAMEN_ON_TIMEOUT=stop` で YAML 側に戻せる。
    どちらで走っているかは起動ログの `actions={...}` に出る。

    ⚠️ hybrid の `pick_table_leg` はこの後で 30s / `stop` に上書きされる
    (有限 FSM なので、持ち替えを確認できないまま進まない方が正しい)。
    """
    skills = _load_skill_config(vendor_desktop).get("skills") or {}

    override = os.environ.get("RAMEN_ON_TIMEOUT", "").strip().lower()
    if override and override not in _TIMEOUT_ACTIONS:
        raise ValueError(
            f"RAMEN_ON_TIMEOUT must be one of {sorted(_TIMEOUT_ACTIONS)}, "
            f"got {override!r}"
        )
    action = override or _BOUNDARY_TIMEOUT_ACTION

    hard: dict[str, float] = {}
    actions: dict[str, str] = {}
    for skill_name, _cls in _STAGE_SKILLS:
        section = skills.get(skill_name) or {}
        if "max_seconds_hard" not in section:
            continue
        timeout_s = float(section["max_seconds_hard"])
        if timeout_s <= 0:
            raise ValueError(f"skills.{skill_name}.max_seconds_hard must be > 0")
        hard[skill_name] = timeout_s
        actions[skill_name] = action
    yaml_actions = {
        name: str((skills.get(name) or {}).get("on_timeout", "advance"))
        for name in hard
    }
    if yaml_actions != actions:
        print(
            f"[orch-driver] on_timeout: YAML {yaml_actions} -> 大会経路 {action!r}"
            f"{'' if override else ' (既定。RAMEN_ON_TIMEOUT で変えられる)'}",
            file=sys.stderr,
        )
    return hard, actions


class OrchestratorDriver:
    """boundary act(obs) → orchestrator.tick → (T,25)。full skill 遷移を再利用。"""

    def __init__(
        self,
        yolo_weight: str | None = None,
        device: str | None = None,
        dex1_open_fraction: tuple[float, float] = (1.0, 1.0),
        ee_frame_transform: np.ndarray | None = None,
        prime_first_model: bool = True,
    ):
        if _VENDOR_DESKTOP not in sys.path:
            sys.path.insert(0, _VENDOR_DESKTOP)
        from inference.desktop.orchestrator import (
            Orchestrator,
            DEFAULT_ENTER_CHECK,
            LiveSourceSafetyError,
            enter_never,
        )
        from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
        from inference.desktop import assembly as _assembly
        from inference.desktop.lower_policy.policies.config_loader import (
            load_policy_variant,
        )
        from inference.desktop.lower_policy.skills import vla_skill as _vla
        from inference.desktop.perception.cleaner import load_cleanup_config
        from inference.desktop.perception.policy_filter import (
            PolicyDetectionFilter,
            load_policy_filter_config,
        )
        from inference.desktop.perception.stream import DetectionStream
        from inference.desktop.perception.yolo_obb import YoloObbPerception
        from inference.desktop.lower_policy.initial_pose import (
            initial_pose_from_config,
        )
        from inference.desktop.perception.dex1_state_source import (
            SyntheticDex1StateSource,
        )

        weight = self._resolve_yolo_weight(
            yolo_weight
            or os.environ.get(
                "RAMEN_YOLO_WEIGHT",
                "/datadrive2/iros_2026_ramen/outputs/yolo_obb/weights/m_lowaug_v4_flat.pt",
            )
        )
        cfg_path = os.path.join(
            _VENDOR_DESKTOP, "inference/desktop/lower_policy/configs/policy_config.yaml"
        )
        skill_cfg_raw = _load_skill_config(_VENDOR_DESKTOP)
        self._fk = G1WristFK.from_urdf()
        self._ee_frame_transform = ee_frame_transform
        self._t = 0
        self._advance_halted = False
        #: HOLD の理由 (None = 通常運転)。`reset` で解ける。
        self._hold_reason: str | None = None
        self._last_step19: np.ndarray | None = None
        #: 直前に publish した `(T,25)`。`_taskspace` が落ちたときの最後の砦。
        self._last_actions: np.ndarray | None = None
        # カメラ: 運営の obs["t"] が変わった瞬間を「届いた瞬間」として刻む。
        self._last_obs_t: object = None
        self._frame_received_ns = 0
        self._head_bgr: np.ndarray | None = None
        # mono の `ego_view` しか無い tick では左右に複製する。実ステレオが
        # 来ている tick では既に 2W 幅なので複製しない。
        self._head_duplicate_mono = True
        self._stereo_seen = False
        #: `RAMEN_HEAD_MONO=1` = 左右キーを無視して必ず mono 複製にする。
        self._head_mono = _env_flag("RAMEN_HEAD_MONO")
        self._head_generation = 0
        self._head_received_ns = 0
        self._missing_images = {"head": 0, "wrist_l": 0, "wrist_r": 0}
        # どの脚から始めてどこでやめるか。env が無ければ 1..4 を頭から。
        self._resume = _resume_settings()
        # 開始 skill の frame-0 の手の開度。hand state の seed と、手が未 dispatch
        # の tick の fallback に使う (0 埋めだと `+1.0` = 全閉になる)。
        self._initial_hand2 = tuple(
            float(v)
            for v in initial_pose_from_config(
                skill_cfg_raw, self._resume.start_skill
            ).dex1_target_rad
        )
        # vendor tree は __init__ で sys.path に入れるので module 直下では import
        # できない。act() から使う例外クラスをここで捕まえておく。
        self._LiveSourceSafetyError = LiveSourceSafetyError

        # I/O adapters
        self._joint_src = BoundaryJointStateSource()
        self._wrist_l = BoundaryWristSource()
        self._wrist_r = BoundaryWristSource()
        self._arm = InterceptorActuator("arm")
        self._waist = InterceptorActuator("waist")
        self._hand = InterceptorActuator("hand")
        # hand state は **実測優先、来なければ合成**。
        #
        # 実測は運営 bridge が `:5557` に載せている `gripper_q` (2026-09-21 版から)。
        # `boundary/states.py` が捨てるので `components/client.py` が 2 本目の SUB で
        # 拾って `obs["gripper_q"]` に入れる (`components/ramen/gripper_state.py`)。
        #
        # 合成 (`SyntheticDex1StateSource`) は **自分の hand 指令のエコー**。
        # 以前は `BoundaryDex1StateSource((1.0, 1.0))` = 常に全開 4.5 rad の定数で、
        # `update()` を一度も呼んでいなかった。`insert_table_leg` /
        # `rotate_leg_to_tighten` は **脚を握った状態が frame 0** なので、
        # 「握っているのに開いている」と毎 tick model に伝えていた (4 脚とも通る)。
        # `command_source` に hand actuator を繋ぐと、以後は指令をエコーする。
        self._dex1_synth = SyntheticDex1StateSource(
            self._initial_hand2, command_source=self._hand
        )
        self._dex1_src = MeasuredDex1StateSource(fallback=self._dex1_synth)
        #: 実測に切り替わった/戻ったログを 1 度だけ出すための直近状態。
        self._dex1_was_measured = False
        _ = dex1_open_fraction  # signature 後方互換 (source 側が値を持つ)

        # perception (YOLO) + cleaner
        perception = YoloObbPerception(weight, device=device)
        cleaner = DetectionStream(load_cleanup_config())

        # skills (lazy DeferredPolicy + interceptor actuators)
        #
        # **VlaSkill も policy も、vendor 側の assembly.build_vla_skill を通して組む。**
        # 自前で組むと 2 種類の事故が起きる:
        #   - DeferredPolicy を直接呼ぶと signature 変更に追従できない
        #     (Issue #141 の変更で TypeError になり groot_orchestrator が丸ごと
        #      起動不能だった、2026-09-20)
        #   - VlaCls を直接呼ぶと、assembly が入れる 6 つが丸ごと抜ける
        #     (2026-09-21 に自前経路と突き合わせて判明):
        #   - language_override … rotate_table_base の variant は specialist 用の
        #     'rotate table base'。渡さないと class の
        #     "rotate and move table base (combined 5+7)" で推論してしまう
        #   - dispatch_waist    … rotate_table_base は config で **False**。
        #     True 固定だと腰を出してはいけない skill で出す
        #   - motion_limiter    … 速度・加速度の包絡 + 位置の限界 (URDF − margin)
        #   - teacher_range     … 4 skill 全てに設定あり。最後の安全網より先に効く補正
        #   - fk (skill 別)     … rotate_table_base は独自 wrist_tool_offset を持つ
        #     (既定と左手で 4.4cm ずれる)
        #   - skill_id_override / progress_monitor / z_ceiling / retry (現状は全て未設定)
        # assembly を通せば、今後 skill_config に設定が増えても自動で入る。
        fk_factory = _assembly.FkFactory()
        # `RAMEN_PICK_HYBRID=1` のときだけ pick が VLM/VLA/MP に替わる (Issue #148)。
        # env が無ければ None = 既定の GR00T のまま。
        hybrid_pick = _hybrid_pick_settings(skill_cfg_raw)
        #: hybrid は Dex1 の実測が要る。実測は obs 経由なのでここでは判定できず、
        #: `act()` の `_wait_for_measured_dex1` が毎 tick 見る。
        self._hybrid_needs_measured_dex1 = hybrid_pick is not None
        stage_variants = _stage_variants(cfg_path)
        registry = {}
        policies = {}
        for skill_name, cls_name in _STAGE_SKILLS:
            variant = _variant_override(skill_name, stage_variants[skill_name])
            entry = load_policy_variant(cfg_path, variant)
            skill_cls = getattr(_vla, cls_name)
            extra_skill_kwargs = None
            if hybrid_pick is not None and skill_name == "pick_table_leg":
                skill_cls = hybrid_pick.skill_cls
                extra_skill_kwargs = hybrid_pick.extra_kwargs
            built = _assembly.build_vla_skill(
                skill_name=skill_name,
                vla_skill_cls=skill_cls,
                variant=entry,
                skill_config=skill_cfg_raw,
                waist_actuator=self._waist,
                hand_actuator=self._hand,
                fk_factory=fk_factory,
                # 先読み (ModelResidency) が読み込みと解放を握る。
                deferred=True,
                extra_skill_kwargs=extra_skill_kwargs,
            )
            registry[skill_name] = built.skill
            policies[skill_name] = built.policy
        dispatcher = SkillDispatchLowerPolicy(registry)
        hard_timeouts, timeout_actions = _load_stage_timeouts(_VENDOR_DESKTOP)
        if hybrid_pick is not None:
            # hybrid は従来 pick とは別の有限手順で、予算も別 (21s -> 30s)。
            # YAML 側を上書きしないと、hybrid が自分の予算を使い切る前に
            # skill 側の時間切れで切られる。`stop` も hybrid の設計どおり
            # (確認できない持ち替えで insert へ進まない)。
            hard_timeouts["pick_table_leg"] = hybrid_pick.hard_timeout_sec
            timeout_actions["pick_table_leg"] = "stop"
            print(
                "[orch-driver] pick_table_leg timeout="
                f"{hybrid_pick.hard_timeout_sec:g}s action=stop/HOLD (hybrid)",
                file=sys.stderr,
            )

        # policy が見る検出は planner 用とは別の、**遅れの無い** filter を通す
        # (Issue #141 D3)。渡さないと cleaner の median filter 越しの検出が
        # そのまま overlay に焼かれ、自前経路と違う画像で推論することになる。
        policy_filter = PolicyDetectionFilter(load_policy_filter_config())

        # 次の expert を **background thread で先読み**する (Issue #141 D7-2)。
        # これが無いと skill 切替のたびに act() が model load でブロックする
        # (実測: pick 24.7s / insert 92s / rotate_leg 69s)。会場では
        # その間ずっと運営へ (T,25) を返せない。
        self._residency = self._build_residency(policies)
        if self._residency is not None and prime_first_model:
            # 最初の model を **serve する前に** 読む。自前経路も stage 開始前に
            # `residency.prime()` を呼んでいる。これが無いと 1 tick 目の中で
            # 同期ロード (pick は 45 秒) が走り、その間に運営から来た frame が
            # 古くなって鮮度チェックに掛かる (2026-09-21 に pod で実測)。
            self._residency.prime()

        # JSONL ログ。手順書が「log の taskspace_25 を確認」と案内しているのに
        # boundary 経路だけ何も残らなかった。path を渡されたときだけ書く。
        log_sink = self._build_log_sink()
        # enter_check は「候補(遷移先)skill」で引かれるので、_TRANSITIONS の values を key に。
        #
        # ⚠️ `.get(c, enter_never)` にはしない。それだと **enter 条件の書き忘れ**まで
        # 静かに「YOLO では永久に発火しない」に化ける。YOLO 判定を持たない skill は
        # ここで明示し、それ以外は登録が無ければ KeyError で落とす。
        _candidates = {c for cands in _TRANSITIONS.values() for c in cands}
        enter_check = {
            c: enter_never if c in _YOLO_FREE_ENTRY else DEFAULT_ENTER_CHECK[c]
            for c in _candidates
        }

        self._orch = Orchestrator(
            perception,
            cleaner,
            dispatcher,
            initial_skill=self._resume.start_skill,
            transitions=_TRANSITIONS,
            enter_check=enter_check,
            actuator_send_fn=self._arm.send_action,
            joint_state_source=self._joint_src,
            dex1_state_source=self._dex1_src,
            wrist_left_source=self._wrist_l,
            wrist_right_source=self._wrist_r,
            head_perception_view="left",  # boundary は単一 head を packed で複製
            hard_timeout_by_skill=hard_timeouts,
            timeout_action_by_skill=timeout_actions,
            policy_filter=policy_filter,
            log_sink=log_sink,
            on_tick=self._on_tick,
            camera_stale_timeout_s=_ORCH_CAMERA_STALE_TIMEOUT_S,
        )
        self._seed_resume_state()
        print(
            f"[orch-driver] hard timeouts={hard_timeouts} actions={timeout_actions}",
            file=sys.stderr,
        )

    def _build_residency(self, policies: dict):
        """次の expert を background で先読みする管理を作る (Issue #141 D7-2)。

        GPU に置く数は `RAMEN_GPU_MODELS` (既定は `_DEFAULT_RESIDENT_MODELS` = 2 =
        今の skill + 次の 1 つ)。切替を隠すのにこれで足りる: 読み込みは約 8 秒で、
        各 skill は 21〜58 秒走るので、次の skill が始まるまでに間に合う。

        **全部 (53D×4 + pick) 載せると約 26 GiB になり、機体によっては入らない。**
        増やすときは env で明示する。

        これが無いと `act()` が model load でブロックする (実測 pick 24.7s /
        insert 92s / rotate_leg 69s)。その間 boundary へ (T,25) を返せない。
        """
        from inference.desktop.lower_policy.policies.residency import ModelResidency

        loadable = {
            name: pol for name, pol in policies.items() if hasattr(pol, "prepare")
        }
        if not loadable:
            print("[orch-driver] 先読み対象の policy が無い", file=sys.stderr)
            return None
        # 既定は「今の skill + 次の 1 つ」。切替を隠すのにこれで足りる
        # (読み込み 約 8 秒 < 各 skill の 21〜58 秒)。全部載せると 53D×4 + pick で
        # 約 26 GiB になり、機体によっては入らない。増やすときは env で明示する。
        raw = os.environ.get("RAMEN_GPU_MODELS", "").strip()
        resident = int(raw) if raw else _DEFAULT_RESIDENT_MODELS
        # ⚠️ ModelResidency は order を **直線**として扱う (`order[i:i+resident]` を
        # 保ち、外れたものを解放する)。1 脚ぶんの列だけを渡すと、脚の末尾
        # (rotate_leg_to_tighten、index 3) で keep が自分 1 つだけになり、
        # **他の 3 つを毎周回解放して次の脚で読み直す**。実測では脚ごとに
        # 8.5 秒のスパイクが出ていた (2026-09-21、pod 実測)。
        # 列を 4 脚ぶんに伸ばすと、どの skill に居ても keep が 4 つを覆う
        # (`_order.index()` は先頭の一致を返すので index は 0..3 のまま)。
        #
        # ⚠️ 列は **実際に最初に走る skill から**並べる。ModelResidency は構築時に
        # `order[0:resident]` を先読みするので、`_STAGE_SKILLS` の並び順
        # (rotate_table_base 始まり) のままだと「1 本目に使わない rotate」を
        # 読んで「次に要る insert」を読まない = 最初の切替で丸ごとブロックする。
        # `RAMEN_START_LEG` / `RAMEN_START_SKILL` で再開するときも同じ。
        names = [name for name, _cls in _STAGE_SKILLS]
        start = names.index(self._resume.start_skill)
        order = (names[start:] + names[:start]) * _LEGS
        print(
            f"[orch-driver] gpu models resident={resident} of {len(loadable)} "
            f"({', '.join(order)})",
            file=sys.stderr,
        )
        return ModelResidency(order, loadable, resident=resident)

    @staticmethod
    def _build_log_sink():
        """`RAMEN_ORCH_LOG` が指されていれば JSONL を書く sink を返す。

        手順書は症状切り分けで `taskspace_25` を読むよう案内しているが、boundary
        経路には log_sink が渡っておらず何も残らなかった。
        """
        path = os.environ.get("RAMEN_ORCH_LOG", "").strip()
        if not path:
            return None
        # orchestrator は `log_sink.write(json.dumps(...) + "\n")` を呼ぶだけなので
        # 素の file object でよい (自前経路も `log_path.open("w")` を渡している)。
        print(f"[orch-driver] orchestrator log -> {path}", file=sys.stderr)
        return open(path, "w", encoding="utf-8")

    def _on_tick(self, result, obs, skill) -> None:
        """tick ごとの hook。先読みの範囲を今の skill に合わせる。"""
        if self._residency is not None:
            self._residency.on_skill_started(getattr(skill, "name", None))

    @staticmethod
    def _resolve_yolo_weight(ref: str) -> str:
        """local .pt path ならそのまま。HF repo[@rev] なら .pt を snapshot_download。

        container では RAMEN_YOLO_WEIGHT に HF ref を渡す:
        Team-RAMEN/IROS2026_RAMEN_Hara_yoloobb_upperpolicy@<rev>。
        """
        if os.path.isfile(ref):
            return ref
        if "/" not in ref:
            return ref  # そのまま (存在しなければ後段で error)
        from huggingface_hub import snapshot_download

        repo_id, revision = ref, None
        if "@" in ref:
            repo_id, revision = ref.rsplit("@", 1)
        snap = Path(
            snapshot_download(
                repo_id=repo_id, revision=revision, allow_patterns=("*.pt",)
            )
        )
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
        self._ingest_gripper(obs)

        # 保持中は推論しない。**`tick()` を呼び続けると、時間切れした expert が
        # そのまま腕を動かし続ける。** 自前経路の `on_timeout: stop` は
        # 「最後の安全 target を保持して動きを止める」なので、ここも合わせる。
        if self._hold_reason is not None:
            return self._held_action(body_q)

        # hybrid は実測が要る (理由は `_wait_for_measured_dex1`)。取れるまで動かない。
        if self._hybrid_needs_measured_dex1 and not self._wait_for_measured_dex1():
            return self._held_action(body_q)

        self._ingest_images(obs)
        if self._hold_reason is not None:  # 取り込みでカメラが止まったと判定した
            return self._held_action(body_q)
        frame = self._head_frame()

        self._arm.reset()
        self._waist.reset()
        # ⚠️ hand は reset しない。実 publisher は最後の target を持ち続けるし、
        # `SyntheticDex1StateSource` もここを読んで state を合成する。毎 tick
        # None に戻すと (a) 未 dispatch tick で「全閉」を出し、(b) 握っている
        # のに「手は開いている」と model に伝えることになる。
        try:
            result = self._orch.tick(frame)
        except BaseException as exc:  # noqa: BLE001
            # 推論の一過性エラー / worker 死亡 / shape 不一致。ここで素通りさせると
            # transport が client 接続を切って run が終わる。自前経路は保持して
            # operator を待つので、同じく保持に倒す。
            self._enter_hold(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            return self._held_action(body_q)

        # 決めた本数まで回り切ったら止める (以後は保持)。時間切れ前進は enter_check を
        # 見ないので、放っておくと最後の skill が永久に締め続ける。
        # 既定は 4 本。`RAMEN_END_LEG` で減らせる (自前経路の --phase3-end-stage 相当)。
        end_legs = self._resume.end_legs
        if self._orch.state.n_legs_completed >= end_legs:
            self._enter_hold(f"{end_legs} 脚完了")
            return self._held_action(body_q)
        # tick() は YOLO の enter_check しか見ない。is_complete / max_seconds_hard /
        # max_dwell_sec の受け皿を自前経路と同じ method で回す。会場は学習データと
        # 違うシーンなので、YOLO が落としたときにここが無いと skill が進まない。
        try:
            self._orch.advance_finished_skill()
        except self._LiveSourceSafetyError as exc:
            # `on_timeout: stop` / dwell 上限で遷移先が無い / カメラが止まった。
            # server は運営に (T,25) を返し続ける必要があるので落とさず、
            # **最後の安全 target を保持する** (自前経路の HOLD と同じ意味)。
            self._enter_hold(str(exc))
            return self._held_action(body_q)

        arms14 = (
            result.action
            if (result is not None and result.action is not None)
            else self._arm.last
        )
        if arms14 is None:
            # buffer 充填中など: 現在姿勢保持で (T,25)
            arms14 = body_q[15:29]
        step19 = assemble_19d(
            self._waist.last,
            arms14,
            self._hand.last,
            measured_waist3=body_q[12:15],
            fallback_hand2=self._initial_hand2,
        )
        self._last_step19 = step19
        # `_taskspace` も try に入れる。ここを素通りさせると `serve_policy` が
        # traceback を client へ返して **接続を切る** (`components/transport.py`)。
        # 会場ではそれが run の終わりになる。FK / 列の組み立てで想定外が出ても、
        # 「返し続けるが新しい動きは作らない」に倒す方が安全。
        try:
            actions = self._taskspace(step19, body_q)
        except BaseException as exc:  # noqa: BLE001
            self._enter_hold(f"taskspace: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return self._held_action(body_q)
        self._last_actions = actions
        return {
            "actions": actions,
            "current_skill": getattr(result, "current_skill", None) if result else None,
        }

    # ---------------------------------------------------------------- Dex1 実測
    def _ingest_gripper(self, obs: dict) -> None:
        """`obs["gripper_q"]` (運営 `:5557` の実測) を state source へ流す。

        `components/client.py` が 2 本目の SUB で拾って入れている
        (`boundary/states.py` は 4 キーしか decode しないため)。キーが無いのは
        旧 client、値が None なのは bridge が旧版かリグが 35 スロット未満の場合。
        どちらも合成 (指令のエコー) に落ちるだけで run は止めない。
        """
        self._dex1_src.update(obs.get("gripper_q"), t=self._t, obs_t=obs.get("t"))
        measured = self._dex1_src.measured_is_fresh
        if measured == self._dex1_was_measured:
            return
        self._dex1_was_measured = measured
        if measured:
            print(
                "[orch-driver] Dex1 は実測 state を使う (:5557 の gripper_q)",
                file=sys.stderr,
            )
        else:
            print(
                "[orch-driver] Dex1 の実測が来ていない -> 合成 (自分の指令のエコー) "
                "にする。hybrid は使えない",
                file=sys.stderr,
            )

    def _wait_for_measured_dex1(self) -> bool:
        """hybrid が動いてよいか。実測が無ければ False (呼び出し側は保持)。

        hybrid の区間 1->2 は **実測 - 指令** で把持を判定する (`interlock.py`)。
        合成 state は指令のエコーなので差が恒等的に 0 になり、`is_grasping` は
        永久に False。`boundary.require_interlock: true` なので VLM 単独でも
        進めず、境界が立たないまま `hard_timeout_sec` (30s) で HOLD する。
        **会場では「掴まない」としか見えない**ので、実測が無いうちは expert を
        1 度も走らせない。

        判定を起動時ではなくここで行うのは、実測が `obs` 経由で来るから。
        driver を組む時点ではまだ 1 通も届いていない。

        猶予を過ぎても来なければ sticky な HOLD に倒す。復旧不能な設定ミス
        (bridge が旧版 / client が `gripper_q` を載せていない) を、運営スロットの
        中で延々と待ち続けないため。
        """
        if self._dex1_src.measured_is_fresh:
            return True
        if self._t <= _DEX1_HYBRID_GRACE_TICKS:
            if self._t == 1:
                print(
                    "[orch-driver] RAMEN_PICK_HYBRID=1: Dex1 の実測を待っている "
                    f"(最大 {_DEX1_HYBRID_GRACE_TICKS} tick)",
                    file=sys.stderr,
                )
            return False
        self._enter_hold(
            "RAMEN_PICK_HYBRID=1 は Dex1 の実測 state を要求するが、"
            f"{_DEX1_HYBRID_GRACE_TICKS} tick 待っても obs['gripper_q'] が来ない。"
            "把持判定 (interlock) は 実測 - 指令 の差を見るので、合成 "
            "(自分の指令のエコー) では差が恒等的に 0 になり一度も発火しない。"
            " 運営 bridge (real_orin_state.py) が 2026-09-21 版か確認するか、"
            "RAMEN_PICK_HYBRID を外して既定の GR00T pick で走らせること "
            "(docs/handoff/connection_test_20260927.md)"
        )
        return False

    # ---------------------------------------------------------------- 観測
    def _ingest_images(self, obs: dict) -> None:
        """運営の `obs["images"]` を source へ流す。

        **`obs["t"]` を捨てないこと。** 運営 client はここに `frame.received_at`
        (カメラの実受信時刻) を入れている (`components/client.py`):

            frame = self._cameras.read(timeout_ms=0) or self._cameras.latest()
            return {..., "t": frame.received_at}

        新しい frame が無ければ `latest()` が返るので **`t` は変わらない**。
        つまり「カメラが止まった」がこの値で分かる。以前はここで自前のカウンタ
        (`self._t`) を入れていたため、

          - `camera_stale_roles` の年齢が常に「今」= 鮮度チェックが発火しない
          - `detection_refreshed` が常に True = カメラが 30Hz を割っても
            median filter が「新しい frame が来た」と誤認する

        の 2 つが同時に起きていた。

        受信時刻は **こちらの monotonic** で持つ。`received_at` は Orin の
        wall clock なので、Thor と時計がずれていると年齢を直接は計算できない。
        「`t` が変わった瞬間 = 届いた瞬間」として自前の時計で刻む。

        画像が欠けた tick は **直前の画像を保持する** (以前は無言で真っ黒画像を
        入れていた。YOLO は検出 0、policy は真っ黒 wrist で推論を続けるので、
        症状が「動いているのに掴まない」になり切り分けられない)。
        """
        now_ns = time.monotonic_ns()
        obs_t = obs.get("t")
        if obs_t is not None and obs_t != self._last_obs_t:
            self._last_obs_t = obs_t
            self._frame_received_ns = now_ns
        # 運営が t を入れない構成でも止まらないよう、未設定なら今を使う。
        received_ns = self._frame_received_ns or now_ns
        generation = int(float(obs_t) * 1e9) if obs_t is not None else int(received_ns)

        # ⚠️ 鮮度は **ここ (取り込んだ瞬間) で測る。**
        # orchestrator 側の `_check_camera_freshness` は `tick()` の (d) にあり、
        # その手前の (a)/(c) で `dispatcher.start()` が model を同期ロードする。
        # 起動直後の pick は 45 秒かかるので、そこで測ると「45 秒古い frame」に
        # 見えて誤発火する (2026-09-21 に pod で実測)。運営は act() を呼び続けて
        # いるので、次の呼び出しでは新しい t が来ている = ここで測れば影響を受けない。
        age_s = (now_ns - self._frame_received_ns) / 1e9
        if self._frame_received_ns and age_s > _CAMERA_STALE_TIMEOUT_S:
            self._enter_hold(
                f"カメラが止まっている: obs['t'] が {age_s:.2f}s 変わらない "
                f"(許容 {_CAMERA_STALE_TIMEOUT_S:g}s)"
            )

        images = obs.get("images") or {}

        def _bgr(key):
            raw = images.get(key)
            if raw is None:
                return None
            return np.ascontiguousarray(np.asarray(raw, np.uint8)[:, :, ::-1])

        # head は **実ステレオがあればそれを使う。** 運営 package 2026.09.21 で
        # `ego_view_left` / `ego_view_right` が追加された。無い構成 (旧 bridge や
        # カメラ不調) では mono の `ego_view` を左右に複製する従来どおりの動き。
        #
        # 複製のままだと **pick の expert に偽の右眼を渡す**ことになる
        # (`policies/groot_pick_legs.py:70` は HEAD_LEFT/HEAD_RIGHT/両手首の 4 cam)。
        #
        # ⚠️ `RAMEN_HEAD_MONO=1` で強制的に mono 複製へ落とせる。head カメラが
        # 3840x1080 で開けなかった場合、運営 bridge は **警告を出しつつ左右キーを
        # publish し続ける**:
        #     real_orin_cameras.py:166-169
        #       WARNING: requested 3840x1080 but camera gave WxH -- eye split
        #       below assumes a side-by-side stereo frame and will be wrong if
        #       this mode isn't genuinely binocular
        # このとき左右は「モノラル画像の左半分と右半分」になる。中身は違うので
        # 運営 preflight の byte-identical 検査 (`preflight_sensors.py:163-167`) も
        # 通ってしまい、**こちらからは見分けが付かない**。bridge の起動ログで
        # `head camera live at 3840x1080` を確認し、違ったらこの env を立てる。
        left, right = _bgr("ego_view_left"), _bgr("ego_view_right")
        if self._head_mono:
            left = right = None
        if left is not None and right is not None:
            head = np.concatenate([left, right], axis=1)  # packed (H, 2W, 3)
            duplicate_mono = False  # 既に 2W 幅。複製しない
            if not self._stereo_seen:
                self._stereo_seen = True
                print(
                    "[orch-driver] head は実ステレオ (ego_view_left/right) を使う",
                    file=sys.stderr,
                )
        else:
            head = _bgr("ego_view")
            duplicate_mono = True  # mono を左右に複製する
            if head is not None and self._stereo_seen:
                self._stereo_seen = False
                print(
                    "[orch-driver] ⚠️ 実ステレオが来ていない。mono を複製する "
                    "(pick の右眼が左眼の複製になる)",
                    file=sys.stderr,
                )

        for name, slot, bgr in (
            ("ego_view", "head", head),
            ("left_wrist", "wrist_l", _bgr("left_wrist")),
            ("right_wrist", "wrist_r", _bgr("right_wrist")),
        ):
            if bgr is None:
                if self._missing_images[slot] == 0:
                    print(
                        f"[orch-driver] {name} が obs に無い。直前の画像を保持する",
                        file=sys.stderr,
                    )
                self._missing_images[slot] += 1
                continue
            self._missing_images[slot] = 0
            if slot == "head":
                self._head_bgr = bgr
                self._head_duplicate_mono = duplicate_mono
                self._head_generation = generation
                self._head_received_ns = received_ns
            else:
                source = self._wrist_l if slot == "wrist_l" else self._wrist_r
                source.update(bgr, t=generation, received_monotonic_ns=received_ns)

    def _head_frame(self):
        """head の `FrameData`。まだ 1 枚も来ていなければ真っ黒 + 受信時刻 0。

        受信時刻 0 は `camera_stale_roles` が `inf` (= 来ていない) と扱うので、
        「画像はあるが古い」と「一度も来ていない」が区別できる。
        """
        if self._head_bgr is None:
            return build_frame_data(
                np.zeros((480, 640, 3), np.uint8),
                t=0,
                packed_stereo=True,
                received_monotonic_ns=0,
            )
        return build_frame_data(
            self._head_bgr,
            t=self._head_generation,
            # 実ステレオを連結済みなら複製しない。mono しか無いときだけ複製する。
            packed_stereo=self._head_duplicate_mono,
            received_monotonic_ns=self._head_received_ns,
        )

    # ---------------------------------------------------------------- 保持
    def _enter_hold(self, reason: str) -> None:
        """以後 `tick()` を呼ばず、最後の安全 target を返し続ける。

        自前経路は `run_live` を抜けて HOLD ハンドラ → controlled release に入るが、
        boundary の server は運営へ `(T,25)` を返し続ける必要があり process を
        抜けられない。**「返し続けるが新しい動きは作らない」**がこちらの HOLD。
        `reset` で解ける。
        """
        if self._hold_reason is not None:
            return
        self._hold_reason = reason
        self._advance_halted = True
        print(f"[orch-driver] HOLD: {reason}", file=sys.stderr)
        sink = getattr(self._orch, "log_sink", None)
        if sink is not None:
            # 障害直前のログが buffer に残ったまま container が落ちるのを防ぐ。
            try:
                sink.flush()
            except Exception:  # noqa: BLE001
                pass

    def _held_action(self, body_q: np.ndarray) -> dict:
        """現在の実測姿勢 + 直近の手指令で `(T,25)` を組む (新しい動きは作らない)。

        **ここは最後の砦なので例外を出さない。** `_taskspace` が落ちたら、直前に
        publish した行をそのまま返す。それも無ければ (起動直後) 例外を上げるしか
        ないが、その場合は `act()` の 1 回目なので運営へ届く前に気づける。
        """
        hand2 = self._hand.last if self._hand.last is not None else self._initial_hand2
        step19 = assemble_19d(None, body_q[15:29], hand2, measured_waist3=body_q[12:15])
        try:
            actions = self._taskspace(step19, body_q)
        except BaseException:  # noqa: BLE001
            if self._last_actions is None:
                raise
            traceback.print_exc()
            actions = self._last_actions
        self._last_actions = actions
        return {
            "actions": actions,
            "current_skill": self._orch.state.current_skill,
        }

    def _taskspace(self, step19: np.ndarray, body_q: np.ndarray) -> np.ndarray:
        """19D (waist3 + arms14 + hand2) → 運営の `(T,25)`。"""
        body29 = np.concatenate([body_q[:12], step19[0:3], step19[3:17]])
        root = np.array([0, 0, 0.70, 1, 0, 0, 0], dtype=np.float64)
        action38 = np.concatenate([root, body29, step19[17:19]])[None, :]  # (1,38)
        return groot_chunk_to_taskspace(
            action38, self._fk, ee_frame_transform=self._ee_frame_transform
        )

    def reset(self) -> None:
        """運営が episode 間に呼ぶ (`components/transport.py` の route)。

        **`_advance_halted` と orchestrator 側の state を必ず戻す。** 戻さないと
        1 本走り切った後の 2 本目が skill を一切進めないまま終わる
        (4 脚完了で halt したまま、`n_legs_completed` も 4 のままになる)。

        ⚠️ reset 時に active だった skill の model は解放される
        (`dispatcher.stop()` が `VlaSkill._on_stop()` を通るため)。常駐から
        外れていないものは先読みが背後で読み直す。
        """
        self._t = 0
        self._advance_halted = False
        self._hold_reason = None
        self._last_step19 = None
        # Dex1 の実測は episode ごとに取り直す。warmup の dummy obs で入った値や
        # 前 episode の最後のサンプルを「新鮮な実測」として持ち越さないため。
        # ⚠️ 属性の張り替えでは駄目 (orchestrator が構築時の instance を持つ)。
        self._dex1_src.reset()
        self._dex1_was_measured = False
        self._orch.reset_episode()
        # `reset_episode()` は n_legs_completed を 0 に戻す。3 本目から再開する
        # 構成でそのままにすると、reset のたびに 1 本目の規則 (Kabsch) に落ちる。
        self._seed_resume_state()

    def _seed_resume_state(self) -> None:
        """`RAMEN_START_LEG` に対応する `n_legs_completed` を入れる。

        自前経路の `orch.state.n_legs_completed = stage - 1` と同じ。表示用の数では
        なく判定に効く (0 なら Kabsch、1..3 なら aspect、>=4 で entry が全部 False)。
        """
        legs_done = self._resume.legs_done
        if legs_done:
            self._orch.state.n_legs_completed = legs_done
            print(
                f"[orch-driver] n_legs_completed を {legs_done} で開始 "
                f"(脚 {legs_done + 1} 本目から)",
                file=sys.stderr,
            )

    def close(self) -> None:
        for a in (self._arm, self._waist, self._hand):
            a.reset()
        # 先読みの worker thread と、開いていれば JSONL を畳む。
        residency = getattr(self, "_residency", None)
        if residency is not None:
            shutdown = getattr(residency, "shutdown", None) or getattr(
                residency, "close", None
            )
            if callable(shutdown):
                shutdown()
        sink = getattr(getattr(self, "_orch", None), "log_sink", None)
        if sink is not None and hasattr(sink, "close"):
            sink.close()
