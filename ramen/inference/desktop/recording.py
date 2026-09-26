"""tick ごとの記録 (JSONL) と 4 カメラの収録。

評価と本番の両方が使う (Issue #141 D1 #4。本番の run も評価と同じ形・同じ集計 script で
見られるようにする)。4 カメラの収録は `start_policy_capture` を呼んだときだけ動くので、
本番は JSONL の 4 file だけを書く。

Layout (`log_dir/`):
    metadata.json     : 起動時の config snapshot (skill_name, variant, initial_pose, ...)
    events.jsonl      : pre-motion 開始/完了、hand init、policy 開始/停止、
                        controlled release 等の phase transition + operator action
    states.jsonl      : per-tick joint state / hand state / ee state (30Hz)
    actions.jsonl     : per-tick VlaSkill 出力 chunk[0] (waist3 + arm14 + hand2)
    capture/          : policy区間だけのhead L/R・wrist L/R JPEG/MP4と監査report

wandb upload はここでは扱わない。upload eligible 判定と upload は
wandb_exporter.py で。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import queue
import shutil
import threading
import time
from typing import Any, Callable

import numpy as np


CAMERA_ROLES = ("head_left", "head_right", "left_wrist", "right_wrist")
CAPTURE_SCHEMA = "team_ramen_skill_evaluation_capture/v2"
MIN_CAMERA_CAPTURE_HZ = 25.0
MIN_CAMERA_COVERAGE_FRACTION = 0.95
MAX_CAMERA_FRAME_GAP_S = 0.50


@dataclass(frozen=True)
class _CameraCaptureItem:
    role: str
    frame_index: int
    generation: int
    received_monotonic_ns: int
    image_bgr: np.ndarray


class RunRecorder:
    """Telemetry/camera writer with deterministic context-manager shutdown."""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._events = open(self._log_dir / "events.jsonl", "w", encoding="utf-8")
        self._states = open(self._log_dir / "states.jsonl", "w", encoding="utf-8")
        self._actions = open(self._log_dir / "actions.jsonl", "w", encoding="utf-8")
        self.tick_count = 0
        self.policy_seconds = 0.0
        self._capture_root = self._log_dir / "capture"
        self._frames_root = self._capture_root / "frames"
        for role in CAMERA_ROLES:
            (self._frames_root / role).mkdir(parents=True, exist_ok=True)
        self._camera_queue: queue.Queue[_CameraCaptureItem | None] = queue.Queue(
            maxsize=512
        )
        self._camera_lock = threading.Lock()
        self._camera_active = False
        self._camera_started = False
        self._camera_started_ns: int | None = None
        self._camera_stopped_ns: int | None = None
        self._camera_counts = {role: 0 for role in CAMERA_ROLES}
        self._camera_first_ns: dict[str, int] = {}
        self._camera_last_ns: dict[str, int] = {}
        self._camera_received_timestamps = {
            role: [] for role in CAMERA_ROLES
        }
        self._camera_last_generation: dict[str, int] = {}
        self._camera_queue_drops = 0
        self._camera_errors: list[str] = []
        self._camera_rows: list[dict[str, Any]] = []
        self._camera_closed = False
        self._camera_sampler_thread: threading.Thread | None = None
        self._camera_sample_hz: float | None = None
        self._camera_thread = threading.Thread(
            target=self._camera_writer,
            name="skill-evaluation-camera-writer",
            # A wedged codec/filesystem must never keep the safety runner
            # process alive after arm control has already been released.
            daemon=True,
        )
        self._camera_thread.start()

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        (self._log_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

    def write_event(self, event: dict[str, Any]) -> None:
        self._events.write(json.dumps(event, default=str) + "\n")
        self._events.flush()

    def write_state(self, state: dict[str, Any]) -> None:
        self._states.write(json.dumps(state, default=str) + "\n")

    def write_action(self, action: dict[str, Any]) -> None:
        self._actions.write(json.dumps(action, default=str) + "\n")
        self.tick_count += 1

    def start_policy_capture(
        self,
        monotonic_ns: int,
        *,
        observation_supplier: Callable[[], dict[str, Any]] | None = None,
        camera_sample_hz: float = 30.0,
    ) -> None:
        """Open one policy interval and optionally sample cameras off-loop."""

        with self._camera_lock:
            if self._camera_closed:
                raise RuntimeError("camera recorder is already closed")
            if self._camera_active or self._camera_started:
                raise RuntimeError("policy camera capture was already started")
            self._camera_active = True
            self._camera_started = True
            self._camera_started_ns = int(monotonic_ns)
            if observation_supplier is not None:
                if not np.isfinite(camera_sample_hz) or camera_sample_hz <= 0.0:
                    raise ValueError("camera_sample_hz must be finite and positive")
                self._camera_sample_hz = float(camera_sample_hz)
                self._camera_sampler_thread = threading.Thread(
                    target=self._camera_sampler,
                    args=(observation_supplier,),
                    name="skill-evaluation-camera-sampler",
                    daemon=True,
                )
                self._camera_sampler_thread.start()

    def stop_policy_capture(self, monotonic_ns: int) -> bool:
        with self._camera_lock:
            if not self._camera_active:
                return False
            self._camera_active = False
            self._camera_stopped_ns = int(monotonic_ns)
            sampler = self._camera_sampler_thread
        if sampler is not None and sampler is not threading.current_thread():
            sampler.join(timeout=2.0)
            if sampler.is_alive():
                with self._camera_lock:
                    self._camera_errors.append(
                        "camera sampler did not stop within 2 seconds"
                    )
        return True

    def _camera_sampler(
        self, observation_supplier: Callable[[], dict[str, Any]]
    ) -> None:
        assert self._camera_sample_hz is not None
        period = 1.0 / self._camera_sample_hz
        deadline = time.monotonic()
        while True:
            with self._camera_lock:
                if not self._camera_active or self._camera_closed:
                    return
            try:
                self.record_camera_observation(observation_supplier())
            except Exception as exc:
                with self._camera_lock:
                    self._camera_errors.append(
                        f"camera sampler: {type(exc).__name__}: {exc}"
                    )
                return
            deadline += period
            delay = deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                # Do not burst stale polls after a delayed filesystem/CPU tick.
                deadline = time.monotonic()

    def record_camera_observation(self, observation: dict[str, Any]) -> None:
        """Queue unique camera generations without blocking policy inference."""

        frames = observation.get("_camera_frames")
        generations = observation.get("_camera_generations")
        received = observation.get("_camera_received_monotonic_ns")
        if not isinstance(frames, dict) or not isinstance(generations, dict):
            return
        if not isinstance(received, dict):
            received = {}
        with self._camera_lock:
            if not self._camera_active or self._camera_closed:
                return
            pending: list[_CameraCaptureItem] = []
            for role in CAMERA_ROLES:
                image = frames.get(role)
                generation = generations.get(role)
                if image is None or generation is None:
                    continue
                array = np.asarray(image)
                if array.ndim != 3 or array.shape[2] != 3:
                    self._camera_errors.append(
                        f"{role}: expected HWC 3-channel image, got {array.shape}"
                    )
                    continue
                if array.dtype != np.uint8 or not array.flags.c_contiguous:
                    array = np.ascontiguousarray(array, dtype=np.uint8)
                generation = int(generation)
                if self._camera_last_generation.get(role) == generation:
                    continue
                frame_index = self._camera_counts[role]
                timestamp = int(received.get(role) or observation.get("t") or 0)
                pending.append(
                    _CameraCaptureItem(
                        role=role,
                        frame_index=frame_index,
                        generation=generation,
                        received_monotonic_ns=timestamp,
                        image_bgr=array,
                    )
                )
            for item in pending:
                try:
                    self._camera_queue.put_nowait(item)
                except queue.Full:
                    self._camera_queue_drops += 1
                    continue
                self._camera_last_generation[item.role] = item.generation
                self._camera_counts[item.role] += 1
                self._camera_first_ns.setdefault(
                    item.role, item.received_monotonic_ns
                )
                self._camera_last_ns[item.role] = item.received_monotonic_ns
                self._camera_received_timestamps[item.role].append(
                    item.received_monotonic_ns
                )

    def _camera_writer(self) -> None:
        try:
            import cv2
        except Exception as exc:
            with self._camera_lock:
                self._camera_errors.append(f"{type(exc).__name__}: {exc}")
            return

        while True:
            item = self._camera_queue.get()
            try:
                if item is None:
                    return
                try:
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        item.image_bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                    )
                    if not ok:
                        raise RuntimeError("OpenCV JPEG encoder returned false")
                    relative = (
                        Path("frames") / item.role / f"{item.frame_index:08d}.jpg"
                    )
                    (self._capture_root / relative).write_bytes(encoded.tobytes())
                    row = {
                        "schema_version": CAPTURE_SCHEMA,
                        "role": item.role,
                        "frame_index": item.frame_index,
                        "generation": item.generation,
                        "received_monotonic_ns": item.received_monotonic_ns,
                        "relative_path": relative.as_posix(),
                    }
                    with self._camera_lock:
                        self._camera_rows.append(row)
                except Exception as exc:
                    with self._camera_lock:
                        self._camera_errors.append(
                            f"{item.role}[{item.frame_index}]: "
                            f"{type(exc).__name__}: {exc}"
                        )
            finally:
                self._camera_queue.task_done()

    def _write_camera_report(self) -> dict[str, Any]:
        with self._camera_lock:
            started_ns = self._camera_started_ns
            stopped_ns = self._camera_stopped_ns
            interval_s = (
                0.0
                if started_ns is None or stopped_ns is None
                else max(0.0, (stopped_ns - started_ns) / 1e9)
            )
            source_rates: dict[str, float] = {}
            effective_rates: dict[str, float] = {}
            coverage_fractions: dict[str, float] = {}
            start_delays: dict[str, float] = {}
            end_stale: dict[str, float] = {}
            maximum_gaps: dict[str, float] = {}
            validation_errors: list[str] = []
            for role, count in self._camera_counts.items():
                first = self._camera_first_ns.get(role)
                last = self._camera_last_ns.get(role)
                span_s = 0.0 if first is None or last is None else (last - first) / 1e9
                source_rate = (
                    0.0 if count < 2 or span_s <= 0.0 else (count - 1) / span_s
                )
                source_rates[role] = source_rate
                effective_rates[role] = (
                    0.0 if interval_s <= 0.0 else count / interval_s
                )
                start_delays[role] = (
                    interval_s
                    if first is None or started_ns is None
                    else max(0.0, (first - started_ns) / 1e9)
                )
                end_stale[role] = (
                    interval_s
                    if last is None or stopped_ns is None
                    else max(0.0, (stopped_ns - last) / 1e9)
                )
                role_timestamps = sorted(self._camera_received_timestamps[role])
                gaps_s = [
                    (b - a) / 1e9
                    for a, b in zip(role_timestamps, role_timestamps[1:])
                ]
                maximum_gaps[role] = max(gaps_s, default=interval_s)
                # A sequence of N samples covers N frame periods, not only the
                # distance between the first and last sample.  Adding the
                # median observed period avoids penalising the final normal
                # 33-ms tail while still exposing multi-second outages.
                nominal_period_s = (
                    float(np.median(gaps_s))
                    if gaps_s
                    else (1.0 / source_rate if source_rate > 0.0 else 0.0)
                )
                covered_s = min(interval_s, max(0.0, span_s + nominal_period_s))
                coverage_fractions[role] = (
                    0.0 if interval_s <= 0.0 else covered_s / interval_s
                )
                if count < 2:
                    validation_errors.append(f"{role}: fewer than two frames")
                if effective_rates[role] < MIN_CAMERA_CAPTURE_HZ:
                    validation_errors.append(
                        f"{role}: effective rate {effective_rates[role]:.2f} Hz "
                        f"is below {MIN_CAMERA_CAPTURE_HZ:.2f} Hz"
                    )
                if coverage_fractions[role] < MIN_CAMERA_COVERAGE_FRACTION:
                    validation_errors.append(
                        f"{role}: interval coverage "
                        f"{coverage_fractions[role]:.3f} is below "
                        f"{MIN_CAMERA_COVERAGE_FRACTION:.3f}"
                    )
                if end_stale[role] > MAX_CAMERA_FRAME_GAP_S:
                    validation_errors.append(
                        f"{role}: final frame was stale for "
                        f"{end_stale[role]:.3f}s"
                    )
                if maximum_gaps[role] > MAX_CAMERA_FRAME_GAP_S:
                    validation_errors.append(
                        f"{role}: maximum frame gap "
                        f"{maximum_gaps[role]:.3f}s exceeds "
                        f"{MAX_CAMERA_FRAME_GAP_S:.3f}s"
                    )
            written_counts = {
                role: sum(row["role"] == role for row in self._camera_rows)
                for role in CAMERA_ROLES
            }
            report = {
                "schema_version": CAPTURE_SCHEMA,
                "recording_started": self._camera_started,
                "recording_started_monotonic_ns": self._camera_started_ns,
                "recording_stopped_monotonic_ns": self._camera_stopped_ns,
                "policy_interval_s": interval_s,
                "requested_camera_sample_hz": self._camera_sample_hz,
                "camera_frame_count": dict(self._camera_counts),
                "camera_written_frame_count": written_counts,
                "camera_source_hz": source_rates,
                "camera_effective_hz": effective_rates,
                "camera_coverage_fraction": coverage_fractions,
                "camera_start_delay_s": start_delays,
                "camera_end_stale_s": end_stale,
                "camera_max_interframe_gap_s": maximum_gaps,
                "validation_thresholds": {
                    "minimum_effective_hz": MIN_CAMERA_CAPTURE_HZ,
                    "minimum_coverage_fraction": MIN_CAMERA_COVERAGE_FRACTION,
                    "maximum_frame_gap_s": MAX_CAMERA_FRAME_GAP_S,
                },
                "validation_errors": validation_errors,
                "queue_drop_count": self._camera_queue_drops,
                "errors": list(self._camera_errors),
                "complete": (
                    self._camera_started
                    and interval_s > 0.0
                    and written_counts == self._camera_counts
                    and self._camera_queue_drops == 0
                    and not self._camera_errors
                    and not validation_errors
                    and not self._camera_thread.is_alive()
                ),
            }
            rows = sorted(
                self._camera_rows,
                key=lambda row: (row["role"], row["frame_index"]),
            )
        (self._capture_root / "camera_frames.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        (self._capture_root / "capture_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report

    def _package_camera_videos(self, *, capture_complete: bool) -> dict[str, Any]:
        import cv2

        output_root = self._capture_root / "videos"
        output_root.mkdir(parents=True, exist_ok=True)
        with self._camera_lock:
            grouped = {
                role: sorted(
                    (row for row in self._camera_rows if row["role"] == role),
                    key=lambda row: row["frame_index"],
                )
                for role in CAMERA_ROLES
            }
        videos: dict[str, Any] = {}
        errors: list[str] = []
        for role, rows in grouped.items():
            if not rows:
                errors.append(f"{role}: no frames")
                continue
            timestamps = [int(row["received_monotonic_ns"]) for row in rows]
            span_s = (timestamps[-1] - timestamps[0]) / 1e9
            fps = 30.0 if span_s <= 0.0 else (len(rows) - 1) / span_s
            fps = min(60.0, max(1.0, fps))
            first = cv2.imread(str(self._capture_root / rows[0]["relative_path"]))
            if first is None:
                errors.append(f"{role}: first JPEG did not decode")
                continue
            height, width = first.shape[:2]
            path = output_root / f"{role}.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                errors.append(f"{role}: MP4 writer did not open")
                continue
            written = 0
            try:
                for row in rows:
                    frame = cv2.imread(str(self._capture_root / row["relative_path"]))
                    if frame is None or frame.shape[:2] != (height, width):
                        raise RuntimeError(
                            f"frame {row['frame_index']} failed decode/shape validation"
                        )
                    writer.write(frame)
                    written += 1
            except Exception as exc:
                errors.append(f"{role}: {type(exc).__name__}: {exc}")
            finally:
                writer.release()
            if written == len(rows) and path.exists() and path.stat().st_size > 0:
                videos[role] = {
                    "relative_path": path.relative_to(self._log_dir).as_posix(),
                    "frame_count": written,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "duration_s": span_s,
                }
        if not capture_complete:
            errors.append(
                "capture report failed full-interval rate/coverage/staleness validation"
            )
        manifest = {
            "complete": (
                capture_complete
                and len(videos) == len(CAMERA_ROLES)
                and not errors
            ),
            "videos": videos,
            "errors": errors,
        }
        (self._capture_root / "video_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if manifest["complete"]:
            shutil.rmtree(self._frames_root, ignore_errors=True)
        return manifest

    def close(self) -> None:
        if not self._camera_closed:
            self.stop_policy_capture(monotonic_ns=time.monotonic_ns())
            with self._camera_lock:
                self._camera_closed = True
            if self._camera_thread.is_alive():
                try:
                    # FIFO placement means every already accepted frame is
                    # written before the worker consumes this sentinel.
                    self._camera_queue.put(None, timeout=10.0)
                except queue.Full:
                    with self._camera_lock:
                        self._camera_errors.append(
                            "capture queue did not accept shutdown sentinel"
                        )
            self._camera_thread.join(timeout=30.0)
            if self._camera_thread.is_alive():
                with self._camera_lock:
                    self._camera_errors.append(
                        "capture writer did not stop within 30 seconds"
                    )
            capture_report = self._write_camera_report()
            try:
                self._package_camera_videos(
                    capture_complete=bool(capture_report["complete"])
                )
            except Exception as exc:
                (self._capture_root / "video_manifest.json").write_text(
                    json.dumps(
                        {
                            "complete": False,
                            "videos": {},
                            "errors": [f"{type(exc).__name__}: {exc}"],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        for f in (self._events, self._states, self._actions):
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def state_record(obs: dict) -> dict:
    """states.jsonl の 1 行を組む。

    JointStateData の position/velocity/effort と Dex1 の position_rad を、
    source 欠損 (`None`) をそのまま `null` として残す形で写す。欠損を 0 埋めすると
    「センサが 0 を返した」と「届いていない」が後から区別できなくなるため。

    Args:
        obs: `assembly.build_observation` の戻り値。

    Returns:
        recorder.write_state に渡す dict。
    """

    def as_list(value: Any) -> list[float] | None:
        return list(np.asarray(value, dtype=float)) if value is not None else None

    js = obs.get("joint_state")
    hs = obs.get("hand_state")
    return {
        "t": obs.get("t"),
        "joint_position": as_list(getattr(js, "position", None) if js else None),
        "joint_velocity": as_list(getattr(js, "velocity", None) if js else None),
        # Issue #137: tau_est = 接触/外力の proxy。Orin bridge が
        # motor_state[i].tau_est を JointState.effort に載せる経路は既にあり、
        # ここで記録しないと z 補正の設計材料が残らない。
        "joint_effort": as_list(getattr(js, "effort", None) if js else None),
        "hand_position_rad": as_list(
            getattr(hs, "position_rad", None) if hs else None
        ),
        "camera_received_monotonic_ns": dict(
            obs.get("_camera_received_monotonic_ns", {})
        ),
    }


def record_policy_tick(
    recorder: Any,
    obs: dict,
    active_skill: Any,
    *,
    learned_skills: Any,
    arm_actuator: Any = None,
    probe_state: dict | None = None,
) -> None:
    """orchestrator の tick ごとの記録 (Issue #141 D1 #4 / D6-3)。評価と本番の両方で使う。

    Args:
        learned_skills: 記録する skill 名の集合 (頭の手順の tick は記録しない)。

    states と actions は**同じ tick で対にして**書く。model を読んだ直後など
    `last_action` がまだ無い tick は、どちらも書かない (wandb の exporter と
    `timeline.py` は 2 つの file を行番号で対にしている)。
    """
    name = getattr(active_skill, "name", None)
    if name not in learned_skills:
        return
    last_pa = getattr(active_skill, "last_action", None)
    if last_pa is None:
        return
    record = state_record(obs)
    record["skill"] = name
    # Issue #137: tau_est (JointState.effort) が実際に届いているかを 1 回だけ残す。
    # bridge は motor_state[i].tau_est をそのまま流すので、SDK が 0 を返していると
    # 「配線は正しいが全ゼロ」になる。run 後の解析で気付くと slot を 1 本無駄にする。
    if probe_state is not None and not probe_state.get("effort_probe_done"):
        probe_state["effort_probe_done"] = True
        eff = record["joint_effort"]
        arr = np.asarray(eff or [], dtype=float)
        recorder.write_event({
            "kind": "joint_effort_probe",
            "present": eff is not None,
            "count": int(arr.size),
            "nonzero_count": int(np.count_nonzero(arr)),
            "max_abs_nm": float(np.max(np.abs(arr))) if arr.size else 0.0,
        })
    published_arm = None
    published_waist = None
    if arm_actuator is not None:
        published_arm_arr, published_waist_arr = (
            arm_actuator.read_last_published_targets()
        )
        if published_arm_arr is not None:
            published_arm = published_arm_arr.astype(float).tolist()
        if published_waist_arr is not None:
            published_waist = published_waist_arr.astype(float).tolist()
    recorder.write_state(record)
    recorder.write_action({
        "t": obs.get("t"),
        "skill": name,
        "chunk_0": last_pa.action_chunk[0].astype(float).tolist(),
        "dds_last_successful_arm_target": published_arm,
        "dds_last_successful_waist_target": published_waist,
        "latency_ms": last_pa.latency_ms,
        "policy_metadata": dict(last_pa.metadata),
    })
