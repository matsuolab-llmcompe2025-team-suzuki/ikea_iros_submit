"""full orchestrator を boundary から駆動する driver (段階2)。

原さんの desktop orchestrator (vendor/desktop) を build し、DDS I/O を
orchestrator_io のアダプタに差し替え、boundary act(obs) 毎に tick を1回回して
(T,25) を返す。skill 遷移 (perception[YOLO]+dwell+is_complete) は原さんの実装のまま。

skill→variant (leg round):
  rotate_table_base = groot_overlay (53D) / pick = groot_pick_legs_v2 (38D) /
  insert = groot_insert_leg_200k (53D) / rotate_leg = groot_rotate_leg_200k (53D)

worker env: pick=RAMEN_WORKER_PYTHON (lerobot0.6.0) / 53D=RAMEN_WORKER_PYTHON_53D (0.6.1)。
YOLO weight: RAMEN_YOLO_WEIGHT (dev 既定 outputs/yolo_obb/weights/m_lowaug_v4_flat.pt)。

会場で焼き直さずに変えられるもの (env):
  RAMEN_VARIANT_<SKILL>   expert の差し替え (例: rotate_table_base を RAMEN-Ori に)
  RAMEN_GPU_MODELS        GPU に置く model 数 (既定 2 = 今 + 次)
  RAMEN_ON_TIMEOUT        時間切れの動き (advance/stop)。既定は advance =
                          YOLO が外しても先へ進む。詳細は _load_stage_timeouts
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


def _variant_override(skill_name: str, default: str) -> str:
    """`RAMEN_VARIANT_<SKILL>` で variant を差し替える。

    会場で image を焼き直さずに expert を入れ替えられるようにする。例えば
    `rotate_table_base` を GR00T ではなく RAMEN-Ori で回したいとき:

        -e RAMEN_VARIANT_ROTATE_TABLE_BASE=rotate_table_base_ramen_ori_141_c32

    policy_type が違っても `assembly.build_vla_skill` が
    `resolve_policy_class` 経由で正しい class を選ぶので、名前を変えるだけでよい
    (RAMEN-Ori は state 71D / 4 cam、GR00T は 49D / 3 cam。どちらも policy 自身が
    `CAMERAS` と `build_state_from_raw` を持ち、VlaSkill はそれを使う)。
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
    for skill_name, _cls, _variant in _STAGE_SKILLS:
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
        # vendor tree は __init__ で sys.path に入れるので module 直下では import
        # できない。act() から使う例外クラスをここで捕まえておく。
        self._LiveSourceSafetyError = LiveSourceSafetyError

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
        registry = {}
        policies = {}
        for skill_name, cls_name, variant in _STAGE_SKILLS:
            variant = _variant_override(skill_name, variant)
            entry = load_policy_variant(cfg_path, variant)
            built = _assembly.build_vla_skill(
                skill_name=skill_name,
                vla_skill_cls=getattr(_vla, cls_name),
                variant=entry,
                skill_config=skill_cfg_raw,
                waist_actuator=self._waist,
                hand_actuator=self._hand,
                fk_factory=fk_factory,
                # 先読み (ModelResidency) が読み込みと解放を握る。
                deferred=True,
            )
            registry[skill_name] = built.skill
            policies[skill_name] = built.policy
        dispatcher = SkillDispatchLowerPolicy(registry)
        hard_timeouts, timeout_actions = _load_stage_timeouts(_VENDOR_DESKTOP)

        # policy が見る検出は planner 用とは別の、**遅れの無い** filter を通す
        # (Issue #141 D3)。渡さないと cleaner の median filter 越しの検出が
        # そのまま overlay に焼かれ、自前経路と違う画像で推論することになる。
        policy_filter = PolicyDetectionFilter(load_policy_filter_config())

        # 次の expert を **background thread で先読み**する (Issue #141 D7-2)。
        # これが無いと skill 切替のたびに act() が model load でブロックする
        # (実測: pick 24.7s / insert 92s / rotate_leg 69s)。会場では
        # その間ずっと運営へ (T,25) を返せない。
        self._residency = self._build_residency(policies)

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
            initial_skill="rotate_table_base",
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
        )
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
        order = [name for name, _cls, _v in _STAGE_SKILLS] * _LEGS
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
        images = obs.get("images", {})
        ego = images.get("ego_view")
        head_bgr = (
            np.ascontiguousarray(np.asarray(ego, np.uint8)[:, :, ::-1])
            if ego is not None
            else np.zeros((480, 640, 3), np.uint8)
        )
        frame = build_frame_data(head_bgr, t=self._t, packed_stereo=True)

        def _bgr(key):
            im = images.get(key)
            return (
                np.ascontiguousarray(np.asarray(im, np.uint8)[:, :, ::-1])
                if im is not None
                else np.zeros((480, 640, 3), np.uint8)
            )

        self._wrist_l.update(_bgr("left_wrist"), t=self._t)
        self._wrist_r.update(_bgr("right_wrist"), t=self._t)

        self._arm.reset()
        self._waist.reset()
        self._hand.reset()
        result = self._orch.tick(frame)

        # 4 脚まわり切ったら止める。時間切れ前進は enter_check を見ないので、
        # 放っておくと `n_legs_completed >= 4` で enter_pick_table_leg が False を
        # 返しても timeout がループを回し続けてしまう。
        if not self._advance_halted and self._orch.state.n_legs_completed >= _LEGS:
            self._advance_halted = True
            print(
                f"[orch-driver] {_LEGS} 脚完了。以後は skill を進めない",
                file=sys.stderr,
            )
        # tick() は YOLO の enter_check しか見ない。is_complete / max_seconds_hard /
        # max_dwell_sec の受け皿を自前経路と同じ method で回す。会場は学習データと
        # 違うシーンなので、YOLO が落としたときにここが無いと skill が進まない。
        try:
            if not self._advance_halted:
                self._orch.advance_finished_skill()
        except self._LiveSourceSafetyError as exc:
            # server は運営に (T,25) を返し続ける必要があるので落とさない。
            # 以後は最後の skill を保持したまま進まなくなる (= `on_timeout: stop`)。
            if not self._advance_halted:
                self._advance_halted = True
                print(f"[orch-driver] advance halted: {exc}", file=sys.stderr)
        arms14 = (
            result.action
            if (result is not None and result.action is not None)
            else self._arm.last
        )
        if arms14 is None:
            # buffer 充填中など: 現在姿勢保持で (T,25)
            arms14 = body_q[15:29]
        step19 = assemble_19d(self._waist.last, arms14, self._hand.last)

        body29 = np.concatenate([body_q[:12], step19[0:3], step19[3:17]])
        root = np.array([0, 0, 0.70, 1, 0, 0, 0], dtype=np.float64)
        action38 = np.concatenate([root, body29, step19[17:19]])[None, :]  # (1,38)
        actions = groot_chunk_to_taskspace(
            action38, self._fk, ee_frame_transform=self._ee_frame_transform
        )
        return {
            "actions": actions,
            "current_skill": getattr(result, "current_skill", None) if result else None,
        }

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
        self._orch.reset_episode()

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
