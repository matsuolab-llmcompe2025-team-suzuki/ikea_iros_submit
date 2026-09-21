"""YOLO-OBB 検出 → root_link 基準の把持ポーズ (Issue #123)。

## なぜ密 depth を使わないか

dataset 実測では把持ポーズの自由度は実質 (x, y, yaw) の 3 つしか動かない:

    z     IQR 0.010 m   ほぼ定数 (テーブル面)
    pitch IQR 0.154 rad ほぼ定数 (横倒しの円筒を掴む手首角)
    x     IQR 0.045 m   OBB 中心
    y     IQR 0.077 m   OBB 中心
    yaw   IQR 0.298 rad OBB の長軸そのもの

「脚はテーブル面の上に横たわっている」という拘束を置けば、2D の OBB 4 頂点を
平面へ逆投影するだけで 6D が決まる。`docs/dataset/depth.md` が報告するとおり
白い無テクスチャのプラ面は SGBM が落ちて縁しか取れないので、脚の胴体の
点群に依存しないこの構成のほうが頑健。

## 掴む位置は「良い場所」ではなく「毎回同じ場所」

持ち替えを固定ポーズで成立させるには脚の突き出し量が一定でなければならない。
把持点は OBB の長軸上の一点で、基準端は「ロボットに近い側の端」に固定してある
(検出のたびに端の取り違えが起きないように)。

## 突き出し量は持ち替え距離から逆算する

比率を定数 (`grasp_fraction`) で決め打つと、**持ち替え距離との整合が誰にも保証されない**。
左手は右グリッパから `handover_clearance_m` だけ離れた点を掴むので、把持点より先に
その距離ぶん脚が出ていなければ左手は空を掴む:

    (1 - 比率) x 脚長 >= handover_clearance_m + handover_margin_m

`handover_clearance_m` を設定すると、比率は **検出した脚長から実行時に逆算**される。
これで 2 つの設定値が構造的に噛み合う。脚が短くて条件を満たせない場合は
その検出を捨てる (掴んでから持ち替え不能に気づくより早く落ちる)。
`handover_clearance_m` 未設定なら従来どおり `grasp_fraction` を使う。

## 校正について

`CameraExtrinsic` の default は URDF の `d435_joint` (torso_link → d435_link、
xyz [0.0576, 0.0175, 0.4299] / rpy [0, 0.8308, 0]) をそのまま入れてある。ただし
`docs/dataset/depth.md` のとおり head は D435 系ではない別 stereo モジュールで、
`docs/flip_table/2026-07-10_act_policy_root_cause_report.md` も「camera extrinsic は
まだ計測ベースの完全校正ではない」と明記している。**実機では必ず校正値で上書きすること。**
内部パラメータの default (fx=fy=510.15) は depth.md の rectified P1 実測値。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from inference.desktop.lower_policy.kinematics.types import (
    EEPose,
    Side,
    matrix_to_rpy,
    rpy_to_matrix,
)
from inference.desktop.perception.yolo_obb import OBBDetection

# ROS の光学フレーム規約: 光学 z が前方 / x 右 / y 下。
# 一方 URDF の body フレームは x 前方 / y 左 / z 上。両者の固定回転。
_BODY_TO_OPTICAL: np.ndarray = np.array(
    [[0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PinholeCamera:
    """rectify 済み単眼の内部パラメータ。default は depth.md の head stereo 実測値。"""

    fx: float = 510.15
    fy: float = 510.15
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("PinholeCamera: fx/fy must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("PinholeCamera: width/height must be > 0")

    def ray(self, uv: np.ndarray) -> np.ndarray:
        """画素 (u, v) [px] → 光学フレームの視線方向 (正規化)。(N,2) → (N,3)。"""
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        if uv.shape[-1] != 2:
            raise ValueError(f"ray: expected (...,2) pixels, got {uv.shape}")
        d = np.stack(
            [(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy,
             np.ones(len(uv))],
            axis=1,
        )
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "PinholeCamera":
        if cfg is None:
            return cls()
        if not isinstance(cfg, dict):
            raise ValueError("camera: must be a mapping")
        unknown = set(cfg) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"camera: unknown key(s) {sorted(unknown)}")
        return cls(**{k: (int(v) if k in ("width", "height") else float(v))
                      for k, v in cfg.items()})


@dataclass(frozen=True)
class CameraExtrinsic:
    """torso_link から見たカメラ body フレームの固定変換。

    `optical` を False にすると body フレームをそのまま光学フレームとして扱う
    (既に光学規約で与えられている校正値を入れる場合)。
    """

    xyz: tuple[float, float, float] = (0.0576235, 0.01753, 0.42987)
    rpy: tuple[float, float, float] = (0.0, 0.8307767239493009, 0.0)
    optical: bool = True

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """(位置, 回転) を torso_link 基準の **光学フレーム**として返す。"""
        R = rpy_to_matrix(np.asarray(self.rpy, dtype=np.float64))
        if self.optical:
            R = R @ _BODY_TO_OPTICAL
        return np.asarray(self.xyz, dtype=np.float64), R

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "CameraExtrinsic":
        if cfg is None:
            return cls()
        if not isinstance(cfg, dict):
            raise ValueError("extrinsic: must be a mapping")
        unknown = set(cfg) - {"xyz", "rpy", "optical"}
        if unknown:
            raise ValueError(f"extrinsic: unknown key(s) {sorted(unknown)}")
        out = {}
        for k in ("xyz", "rpy"):
            if k in cfg:
                v = np.asarray(cfg[k], dtype=np.float64).reshape(-1)
                if v.shape != (3,):
                    raise ValueError(f"extrinsic.{k}: must have 3 values")
                out[k] = tuple(float(x) for x in v)
        if "optical" in cfg:
            out["optical"] = bool(cfg["optical"])
        return cls(**out)


@dataclass(frozen=True)
class SupportPlane:
    """脚が載っている平面 (root_link 基準)。default は水平面。

    dataset 実測で把持 z の IQR は 0.010 m しかない = テーブル面高さは
    ほぼ定数。したがって毎 tick 平面フィットする必要はなく、校正値で足りる。
    平面推定を後から差すときは `normal` / `offset` を更新すればよい。
    """

    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    offset: float = 0.101   # dataset 実測 把持点 z の中央値 [m]

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=np.float64)
        if n.shape != (3,) or float(np.linalg.norm(n)) < 1e-9:
            raise ValueError("SupportPlane.normal: must be a non-zero 3-vector")
        object.__setattr__(self, "normal", tuple(n / np.linalg.norm(n)))

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "SupportPlane":
        if cfg is None:
            return cls()
        if not isinstance(cfg, dict):
            raise ValueError("support_plane: must be a mapping")
        unknown = set(cfg) - {"normal", "offset"}
        if unknown:
            raise ValueError(f"support_plane: unknown key(s) {sorted(unknown)}")
        out = {}
        if "normal" in cfg:
            v = np.asarray(cfg["normal"], dtype=np.float64).reshape(-1)
            if v.shape != (3,):
                raise ValueError("support_plane.normal: must have 3 values")
            out["normal"] = tuple(float(x) for x in v)
        if "offset" in cfg:
            out["offset"] = float(cfg["offset"])
        return cls(**out)


def intersect_ray_plane(
    origin: np.ndarray, direction: np.ndarray, plane: SupportPlane
) -> Optional[np.ndarray]:
    """視線と平面の交点。平面に平行 / 背面側なら `None`。"""
    n = np.asarray(plane.normal, dtype=np.float64)
    denom = float(direction @ n)
    if abs(denom) < 1e-9:
        return None
    t = (plane.offset - float(origin @ n)) / denom
    if t <= 0:
        return None
    return origin + t * direction


class ObbGraspPoseProvider:
    """`GraspPoseProvider` の実装。head の OBB 検出から把持ポーズを組む。

    Args:
        camera / extrinsic / plane: 校正。
        leg_class: 脚として扱う OBB の class 名。
        approach_pitch / approach_roll: 手首の固定角 [rad]。dataset 実測中央値。
        grasp_fraction: ロボットに近い側の端から測った、長軸上の把持位置 [0, 1]。
            `handover_clearance_m` が設定されている場合は無視され、脚長から逆算される。
        handover_clearance_m: 左手が右グリッパから離れて掴む距離 [m]。
            skill 設定の持ち替え offset の大きさと一致させること
            (test がこの一致を検証する)。
        handover_margin_m: 上記に足す余裕 [m]。
        grasp_near_end_margin_m: 右手の把持点よりロボット側にも残す脚長 [m]。
            遠端側の持ち替え距離だけで比率を決めると、短く見積もられた OBB を
            脚端ぎりぎりで掴む危険があるための fail-closed 下限。
        kinematics / ik_side / pregrasp_offset / pregrasp_rpy_offset:
            与えると **到達可能な手首姿勢を候補から選ぶ**。
            固定 roll/pitch では把持点中央値がそのまま到達域の縁に来てしまい
            (前方 / 左右いずれも余裕 0.00m)、脚が少しでも動くと IK が解けない。
            dataset のデモ者も手首角を位置に応じて変えている (roll IQR 0.279 /
            pitch IQR 0.154)。候補から選ぶことでこの自由度を取り戻す。
            `None` のときは候補選択を行わず先頭の (roll, pitch) を返す。
        roll_candidates / pitch_candidates: 探索する手首角 [rad]。先頭が第一候補。
        min_confidence: これ未満の検出は捨てる。
        search_center / search_radius: 探索マスク。root_link 基準の xy [m] で、
            この円外に落ちた検出は無視する (別の脚 / 背景の誤検出を弾く)。
            dataset 実測では把持点の 95% が中央値から 0.122 m 以内。
    """

    def __init__(
        self,
        *,
        camera: Optional[PinholeCamera] = None,
        extrinsic: Optional[CameraExtrinsic] = None,
        plane: Optional[SupportPlane] = None,
        grasp_height_offset_m: float = 0.0,
        leg_class: str = "leg",
        approach_pitch: float = 0.872,
        approach_roll: float = 0.187,
        grasp_fraction: float = 0.5,
        handover_clearance_m: Optional[float] = None,
        handover_margin_m: float = 0.03,
        grasp_near_end_margin_m: float = 0.0,
        min_confidence: float = 0.5,
        search_center: Optional[Sequence[float]] = None,
        search_radius: Optional[float] = None,
        kinematics: Optional[object] = None,
        ik_side: Side = Side.RIGHT,
        pregrasp_offset: Optional[Sequence[float]] = None,
        pregrasp_rpy_offset: Optional[Sequence[float]] = None,
        roll_candidates: Optional[Sequence[float]] = None,
        pitch_candidates: Optional[Sequence[float]] = None,
    ) -> None:
        self.camera = camera or PinholeCamera()
        self.extrinsic = extrinsic or CameraExtrinsic()
        self.plane = plane or SupportPlane()
        if not np.isfinite(grasp_height_offset_m):
            raise ValueError("grasp_height_offset_m: must be finite")
        self.grasp_height_offset_m = float(grasp_height_offset_m)
        self.leg_class = str(leg_class)
        self.approach_pitch = float(approach_pitch)
        self.approach_roll = float(approach_roll)
        if not 0.0 <= grasp_fraction <= 1.0:
            raise ValueError("grasp_fraction: must be within [0, 1]")
        self.grasp_fraction = float(grasp_fraction)
        if handover_clearance_m is not None and float(handover_clearance_m) <= 0:
            raise ValueError("handover_clearance_m: must be > 0")
        self.handover_clearance_m = (
            None if handover_clearance_m is None else float(handover_clearance_m)
        )
        if float(handover_margin_m) < 0:
            raise ValueError("handover_margin_m: must be >= 0")
        self.handover_margin_m = float(handover_margin_m)
        if float(grasp_near_end_margin_m) < 0:
            raise ValueError("grasp_near_end_margin_m: must be >= 0")
        self.grasp_near_end_margin_m = float(grasp_near_end_margin_m)
        self.min_confidence = float(min_confidence)
        self.search_center = (
            None if search_center is None
            else np.asarray(search_center, dtype=np.float64).reshape(-1)
        )
        if self.search_center is not None and self.search_center.shape != (2,):
            raise ValueError("search_center: must have 2 values (x, y)")
        self.search_radius = None if search_radius is None else float(search_radius)
        self.kinematics = kinematics
        self.ik_side = ik_side
        self.pregrasp_offset = (
            None if pregrasp_offset is None
            else np.asarray(pregrasp_offset, dtype=np.float64).reshape(-1)
        )
        self.pregrasp_rpy_offset = (
            None if pregrasp_rpy_offset is None
            else np.asarray(pregrasp_rpy_offset, dtype=np.float64).reshape(-1)
        )
        for attr in ("pregrasp_offset", "pregrasp_rpy_offset"):
            v = getattr(self, attr)
            if v is not None and v.shape != (3,):
                raise ValueError(f"{attr}: must have 3 values")
        # 先頭は default (dataset 実測中央値)。以降は到達域を広げるための代替。
        self.roll_candidates = tuple(
            float(v) for v in (roll_candidates or (approach_roll, 0.0, -0.25, 0.45))
        )
        self.pitch_candidates = tuple(
            float(v) for v in (pitch_candidates or (approach_pitch, 0.75, 0.65, 0.55))
        )
        # 直近の中間結果 (診断 / test 用)
        self.last_corners_root: Optional[np.ndarray] = None
        self.last_orientation: Optional[tuple[float, float]] = None
        self.last_reachable: bool = True
        self.last_leg_length: Optional[float] = None
        self.last_grasp_fraction: Optional[float] = None
        self.last_reject_reason: Optional[str] = None

    # ------------------------------------------------------------ 変換

    def camera_pose_in_root(
        self, waist: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """root_link 基準のカメラ光学フレーム (位置, 回転)。

        腕と同じく、カメラも torso_link 側に付くので腰 3 関節の影響を受ける。
        """
        from inference.desktop.lower_policy.kinematics.g1_arm import _WAIST, _rot_axis

        w = np.zeros(3) if waist is None else np.asarray(waist, dtype=np.float64)
        if w.shape != (3,):
            raise ValueError(f"waist: must have shape (3,), got {w.shape}")
        p = np.zeros(3)
        R = np.eye(3)
        for (xyz, rpy, axis), th in zip(_WAIST, w):
            p = p + R @ np.asarray(xyz, dtype=np.float64)
            R = R @ rpy_to_matrix(np.asarray(rpy)) @ _rot_axis(axis, float(th))
        t_cam, R_cam = self.extrinsic.matrix()
        return p + R @ t_cam, R @ R_cam

    def project_to_plane(
        self, verts_norm: np.ndarray, waist: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """正規化 OBB 頂点 (4,2) [0-1] → 平面上の root_link 座標 (4,3)。"""
        v = np.asarray(verts_norm, dtype=np.float64)
        if v.shape != (4, 2):
            raise ValueError(f"verts: must have shape (4, 2), got {v.shape}")
        px = v * np.array([self.camera.width, self.camera.height])
        origin, R_cam = self.camera_pose_in_root(waist)
        dirs = self.camera.ray(px) @ R_cam.T
        out = []
        for d in dirs:
            hit = intersect_ray_plane(origin, d, self.plane)
            if hit is None:
                return None
            out.append(hit)
        return np.array(out)

    # ------------------------------------------------------------ 把持ポーズ

    def pose_from_corners(self, corners: np.ndarray) -> EEPose:
        """平面上の 4 頂点 → 把持ポーズ。

        長辺の中心線を脚の軸とみなし、ロボットに近い側の端から
        `grasp_fraction` の位置を把持点にする。
        """
        c = np.asarray(corners, dtype=np.float64)
        if c.shape != (4, 3):
            raise ValueError(f"corners: must have shape (4, 3), got {c.shape}")
        # 隣接辺の長さを比べ、長いほうを長軸とする
        e01 = c[1] - c[0]
        e12 = c[2] - c[1]
        if np.linalg.norm(e01) >= np.linalg.norm(e12):
            end_a = (c[0] + c[3]) / 2.0
            end_b = (c[1] + c[2]) / 2.0
        else:
            end_a = (c[0] + c[1]) / 2.0
            end_b = (c[2] + c[3]) / 2.0
        # ロボット (root 原点) に近い端を基準にして端の取り違えを防ぐ
        if np.linalg.norm(end_b[:2]) < np.linalg.norm(end_a[:2]):
            end_a, end_b = end_b, end_a
        axis = end_b - end_a
        length = float(np.linalg.norm(axis))
        self.last_leg_length = length
        self.last_reject_reason = None

        fraction = self._fraction_for(length)
        if fraction is None:
            # 脚が短くて持ち替え距離を確保できない。掴む前に捨てる。
            self.last_grasp_fraction = None
            self.last_reachable = False
            self.last_reject_reason = "leg_too_short"
            fraction = self.grasp_fraction
        self.last_grasp_fraction = fraction

        point = end_a + axis * fraction
        # The OBB corners lie on the physical support surface, while the
        # commanded pose is the Dex1 base frame.  Those heights are not the
        # same: using the measured Dex1-base grasp height as the ray/plane
        # intersection height also changes the metric scale of every detected
        # leg.  Keep the projective support plane and tool-frame offset as two
        # explicit quantities.
        point = point.copy()
        point[2] += self.grasp_height_offset_m
        yaw = float(np.arctan2(axis[1], axis[0]))
        chosen = self._select_orientation(point, yaw)
        roll, pitch = chosen if chosen is not None else (
            self.roll_candidates[0], self.pitch_candidates[0]
        )
        self.last_orientation = (roll, pitch)
        if self.last_reject_reason is None:
            self.last_reachable = chosen is not None
            if chosen is None:
                self.last_reject_reason = "unreachable"
        return EEPose(position=point, rpy=np.array([roll, pitch, yaw]))

    def _fraction_for(self, leg_length: float) -> Optional[float]:
        """脚長から長軸上の把持比率を決める。

        `handover_clearance_m` 未設定なら固定値をそのまま使う。設定されている場合は
        「把持点より先に clearance + margin ぶん出る」ように逆算し、脚が短くて
        成立しないなら `None` を返す。
        """
        if self.handover_clearance_m is None:
            return self.grasp_fraction
        if leg_length <= 1e-6:
            return None
        required = self.handover_clearance_m + self.handover_margin_m
        if leg_length < required + self.grasp_near_end_margin_m:
            return None
        fraction = 1.0 - required / leg_length
        return fraction if fraction >= 0.0 else None

    def _select_orientation(
        self, point: np.ndarray, yaw: float
    ) -> Optional[tuple[float, float]]:
        """把持と pre-grasp の **両方** が IK で解ける手首角を候補から選ぶ。

        kinematics 未注入なら先頭候補をそのまま返す (幾何だけ使いたい場合)。
        全候補が解けなければ `None` = 「この脚は届かない」。呼び出し側は
        その検出を捨て、skill は「まだ見つかっていない」として待機する
        (途中で IK 失敗して abort するより安全)。
        """
        if self.kinematics is None:
            return self.roll_candidates[0], self.pitch_candidates[0]
        seed = np.zeros(7)
        for pitch in self.pitch_candidates:
            for roll in self.roll_candidates:
                grasp = EEPose(position=point, rpy=np.array([roll, pitch, yaw]))
                if not self.kinematics.ik(grasp, seed, self.ik_side).ok:
                    continue
                if self.pregrasp_offset is None and self.pregrasp_rpy_offset is None:
                    return roll, pitch
                pos = point if self.pregrasp_offset is None else point + self.pregrasp_offset
                rpy = np.array([roll, pitch, yaw])
                if self.pregrasp_rpy_offset is not None:
                    rpy = rpy + self.pregrasp_rpy_offset
                if self.kinematics.ik(EEPose(position=pos, rpy=rpy), seed, self.ik_side).ok:
                    return roll, pitch
        return None

    def grasp_pose(self, obs: dict) -> Optional[EEPose]:
        """`GraspPoseProvider` 本体。検出できなければ `None`。"""
        # Diagnostics describe this invocation, never a stale candidate from a
        # previous frame.  In particular, a later rejected OBB must not
        # overwrite the diagnostics of the accepted OBB returned below.
        self.last_corners_root = None
        self.last_orientation = None
        self.last_reachable = False
        self.last_leg_length = None
        self.last_grasp_fraction = None
        self.last_reject_reason = None
        dets = obs.get("cleaned") or []
        cands = [
            d for d in dets
            if d.class_name == self.leg_class and d.confidence >= self.min_confidence
        ]
        if not cands:
            return None
        js = obs.get("joint_state")
        waist = None if js is None else np.asarray(js.position, dtype=np.float64)[12:15]

        best: Optional[
            tuple[
                float,
                EEPose,
                np.ndarray,
                Optional[float],
                Optional[float],
                bool,
                Optional[str],
                Optional[tuple[float, float]],
            ]
        ] = None
        for det in cands:
            corners = self.project_to_plane(det.verts, waist)
            if corners is None:
                continue
            pose = self.pose_from_corners(corners)
            if self.last_reject_reason is not None:
                # 届かない / 短すぎる脚は候補から外す。ここで返してしまうと skill が
                # 動作の途中で IK 失敗、あるいは持ち替えで空振りする。
                continue
            if self.search_center is not None and self.search_radius is not None:
                if float(np.linalg.norm(pose.position[:2] - self.search_center)) > self.search_radius:
                    self.last_reject_reason = "outside_search_region"
                    continue
            # 同点候補が複数あるときは confidence が最も高いものを採る
            if best is None or det.confidence > best[0]:
                best = (
                    det.confidence,
                    pose,
                    corners,
                    self.last_leg_length,
                    self.last_grasp_fraction,
                    self.last_reachable,
                    self.last_reject_reason,
                    self.last_orientation,
                )
        if best is None:
            return None
        self.last_corners_root = best[2]
        self.last_leg_length = best[3]
        self.last_grasp_fraction = best[4]
        self.last_reachable = best[5]
        self.last_reject_reason = best[6]
        self.last_orientation = best[7]
        return best[1]

    # ------------------------------------------------------------ config

    @classmethod
    def from_config(cls, cfg: Optional[dict], **kwargs: object) -> "ObbGraspPoseProvider":
        # `or {}` は使わない: 空 list / 空文字が falsy として素通りし型誤りを見逃す。
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise ValueError("obb_grasp: must be a mapping")
        known = {
            "camera", "extrinsic", "support_plane", "grasp_height_offset_m",
            "leg_class", "approach_pitch",
            "approach_roll", "grasp_fraction", "min_confidence", "search_center",
            "search_radius", "pregrasp_offset", "pregrasp_rpy_offset",
            "roll_candidates", "pitch_candidates", "handover_clearance_m",
            "handover_margin_m", "grasp_near_end_margin_m",
        }
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"obb_grasp: unknown key(s) {sorted(unknown)}")
        return cls(
            camera=PinholeCamera.from_config(cfg.get("camera")),
            extrinsic=CameraExtrinsic.from_config(cfg.get("extrinsic")),
            plane=SupportPlane.from_config(cfg.get("support_plane")),
            grasp_height_offset_m=float(cfg.get("grasp_height_offset_m", 0.0)),
            leg_class=str(cfg.get("leg_class", "leg")),
            approach_pitch=float(cfg.get("approach_pitch", 0.872)),
            approach_roll=float(cfg.get("approach_roll", 0.187)),
            grasp_fraction=float(cfg.get("grasp_fraction", 0.5)),
            handover_clearance_m=(
                None if cfg.get("handover_clearance_m") is None
                else float(cfg["handover_clearance_m"])
            ),
            handover_margin_m=float(cfg.get("handover_margin_m", 0.03)),
            grasp_near_end_margin_m=float(
                cfg.get("grasp_near_end_margin_m", 0.0)
            ),
            min_confidence=float(cfg.get("min_confidence", 0.5)),
            search_center=cfg.get("search_center"),
            search_radius=(
                None if cfg.get("search_radius") is None
                else float(cfg["search_radius"])
            ),
            pregrasp_offset=cfg.get("pregrasp_offset"),
            pregrasp_rpy_offset=cfg.get("pregrasp_rpy_offset"),
            roll_candidates=cfg.get("roll_candidates"),
            pitch_candidates=cfg.get("pitch_candidates"),
            **kwargs,
        )


def make_obb(
    verts_norm: Sequence[Sequence[float]],
    *,
    class_name: str = "leg",
    confidence: float = 0.9,
    class_id: int = 0,
) -> OBBDetection:
    """test / bring-up 用に `OBBDetection` を組む小さな helper。"""
    return OBBDetection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        verts=np.asarray(verts_norm, dtype=np.float64),
    )
