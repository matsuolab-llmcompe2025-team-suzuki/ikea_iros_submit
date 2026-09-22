#!/usr/bin/env python3
"""Your policy client, running on the Jetson Orin NX onboard the G1.

    Orin 192.168.100.2   this file + the organizer's three endpoints
    Thor 192.168.100.1   components/server.py

Adapt this freely — it is yours. It exists to show the shape of the loop:

    boundary.CameraStream  :5555 ─┐
                                  ├─> observation ─> Thor ─> action chunk ─┐
    boundary.StateStream   :5557 ─┘                                        │
                                                boundary.ActionSink :5556 <┘

    # on the Orin
    python components/client.py --lane sonic --thor 192.168.100.1

--------------------------------------------------------------------------
Why this loop is pipelined
--------------------------------------------------------------------------
The naive loop — query, execute the whole chunk, query again — leaves the
robot executing stale actions for one full inference latency at the seam of
every chunk. At ~200 ms of round trip that is ten dead control periods, and
it shows up as a visible stutter between chunks.

So this client does two things instead:

  1. PREFETCH. The next inference is submitted on a worker thread while the
     current chunk is still playing out, so a fresh chunk is usually already
     in hand when the old one runs dry.

  2. LATENCY COMPENSATION. A chunk that took L seconds to compute describes
     the world as it was L seconds ago, so its first round(L * rate) rows
     have already been overtaken by events. They are skipped rather than
     replayed.

Both matter more the slower your model is. Tune with --prefetch-rows.

--------------------------------------------------------------------------
Safety
--------------------------------------------------------------------------
Killing or pausing this client does NOT stop the robot. The whole-body
controller keeps replaying the last command it got. The only thing that
brings the robot to a safe state is the organizer's independent e-stop,
which damps the motors directly and does not go through your code. Never
treat "my client exited" as "the robot stopped".

NETWORK: the Thor<->Orin link is a direct ethernet cable with static IPs. If
the client cannot reach the Thor, check `ip link` for the interface being
down before you go looking at code.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boundary import ActionSink, CameraStream, StateStream  # noqa: E402
from boundary.actions import ActionError  # noqa: E402
from components.ramen.gripper_state import GripperStateStream  # noqa: E402
from components.ramen.raw_camera import RawCameraStream  # noqa: E402
from components.transport import PolicyLink  # noqa: E402

LANES = ("sonic", "decoupled")
SONIC_STEP_HZ = 50.0  # gear_sonic_deploy's control cadence
DECOUPLED_CHUNK_HZ = 20.0  # task-space re-query rate

# ---------------------------------------------------------------- カメラ凍結
# 運営 bridge の publish loop は **capture と非同期**で、新しい frame が撮れたか
# どうかを一切見ずに 30Hz 固定で `_latest_frames` を再エンコードして送る
# (`reference/orin_bridge/real_orin_cameras.py: camera_publish_loop`)。しかも
# `timestamps[key] = now` は **publish 時刻**で、capture 時刻ではない。
#
# つまり capture 側が止まっても
#   - JPEG は **byte 単位で同一**のまま 30Hz で流れ続け
#   - `obs["t"]` (= 受信時刻) は正常に進む
# ので、受信時刻ベースの鮮度チェックは **原理的に発火しない**。
# head の capture thread には再起動が無く、USB が外れれば `cap.read()` は永久に
# 失敗し続ける (`if not ok: time.sleep(0.05); continue`) = 凍結は回復しない。
#
# ⚠️ **止めない。log だけ出す。**
# publish 30Hz と capture のレート差で **短い重複は正常に起きる**ので、これを
# 停止条件にすると健全なカメラで run を落とす。閾値は tick 数ではなく「秒」で
# 持つ (`observe()` の呼ばれる間隔は lane と負荷で変わる)。
#
# 目的は CONTRACT.md「Fairness standard」の立証:
#   "If a run fails, the cause must be demonstrably yours."
#   your failure の例: "a policy that runs correctly but produces unreachable
#   or task-incorrect targets"
# 凍結画像で動く policy の外見は、この「我々の失敗」と区別が付かない。運営
# bridge は凍結時に無言、運営 preflight は起動時しか見ない、こちらも画像を
# 記録していない — **この log が no-contest を申告できる唯一の材料**になる。
CAMERA_FREEZE_WARN_S = 1.0  # 30Hz なら 30 frame ぶん撮れていない
CAMERA_FREEZE_REPEAT_S = 5.0  # 凍結が続いている間の再通知間隔


class Inference:
    """Runs one inference at a time on a worker thread, so the control loop
    never blocks on the network.

    ``submit`` snapshots the observation on the calling thread — sampling it
    inside the worker would time-stamp the observation at whenever the thread
    happened to start, which is exactly the error latency compensation is
    trying to correct.
    """

    def __init__(
        self, link: PolicyLink, cameras, states, prompt, camera_keys, grippers=None
    ):
        self._link = link
        self._cameras = cameras
        self._states = states
        self._grippers = grippers
        self._prompt = prompt
        self._camera_keys = camera_keys
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._pending: Future | None = None
        # カメラ凍結の検出用 (log only)。key -> 直前の JPEG / 最後に中身が変わった
        # 時刻 / 最後に警告を出した時刻。
        self._last_jpeg: dict[str, bytes] = {}
        self._jpeg_changed_at: dict[str, float] = {}
        self._freeze_warned_at: dict[str, float] = {}

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def observe(self) -> dict | None:
        """Sample both input endpoints. None until each has produced once."""
        frame = self._cameras.read(timeout_ms=0) or self._cameras.latest()
        state = self._states.read(timeout_ms=0) or self._states.latest()
        if frame is None or state is None:
            return None
        # Only the cameras the server declared — shipping ones the model
        # ignores is pure latency.
        #
        # **JPEG のまま運ぶ。** bridge は JPEG を配っている (CONTRACT.md:78) のに、
        # 以前はここで decode 済みの RGB ndarray を送っていた。会場リグの実写では
        # 1 枚 68〜97 KB の JPEG が raw 900 KB に膨らむ (x9〜13)。5 枚で 4.61 MB、
        # Thor <-> PC2 の実効 101 MB/s (1 GbE、実測) では **転送だけで 45.6 ms** =
        # 運営 adapter の 1 周期 50 ms をほぼ使い切っていた。
        # 再エンコードはしないので server が展開したピクセルは以前と同一。
        # 展開は components/server.py が policy に渡す直前で 1 回だけ行う。
        images_jpeg = {k: frame.jpegs[k] for k in self._camera_keys if k in frame.jpegs}
        self._check_frozen(images_jpeg)
        return {
            "images_jpeg": images_jpeg,
            "body_q": state.body_q,
            "base_quat": state.base_quat,
            # Dex1-1 の実測開度。boundary/states.py は REQUIRED/OPTIONAL の 4 キーしか
            # decode しないので、同じ :5557 に張った 2 本目の SUB から拾っている
            # (components/ramen/gripper_state.py)。bridge が旧版なら None。
            # 単位は運営の生モータ角 (q=0.0 閉 / -5.30 開)。
            "gripper_q": self._grippers.poll() if self._grippers is not None else None,
            "prompt": self._prompt,
            "t": frame.received_at,
        }

    def _check_frozen(self, images_jpeg: dict) -> None:
        """同じ JPEG が届き続けていないか見る。**止めない。log だけ出す。**

        判定は decode 前の JPEG bytes 同士の比較。bridge は同じ ndarray を
        `cv2.imencode` し直すだけなので、capture が止まっていれば byte 完全一致
        になる。実写では センサノイズで必ず違う bytes になるため、一致が
        `CAMERA_FREEZE_WARN_S` 続くのは「撮れていない」を意味する。

        止めない理由と閾値の根拠は module 冒頭の `CAMERA_FREEZE_WARN_S` を参照。
        """
        now = time.monotonic()
        for key, jpeg in images_jpeg.items():
            if self._last_jpeg.get(key) != jpeg:
                # 中身が変わった = 新しい capture が来ている。凍結していたなら
                # **継続時間を残す** (申告では「いつから いつまで」が要る)。
                if self._freeze_warned_at.pop(key, None) is not None:
                    held = now - self._jpeg_changed_at.get(key, now)
                    print(
                        f"[client] camera {key}: 画像の更新が再開した "
                        f"(凍結していた時間 {held:.1f}s)",
                        file=sys.stderr,
                    )
                self._last_jpeg[key] = jpeg
                self._jpeg_changed_at[key] = now
                continue

            held = now - self._jpeg_changed_at.get(key, now)
            if held < CAMERA_FREEZE_WARN_S:
                continue  # レート差による短い重複は正常
            last_warn = self._freeze_warned_at.get(key)
            if last_warn is not None and now - last_warn < CAMERA_FREEZE_REPEAT_S:
                continue
            self._freeze_warned_at[key] = now
            print(
                f"[client] WARNING: camera {key} が {held:.1f}s 変わっていない "
                f"(JPEG が byte 単位で同一)。bridge は送り続けていて obs['t'] は "
                f"進むので鮮度チェックは発火しない。capture 側が止まっている可能性が "
                f"高い — 走行後の申告根拠になるのでこの行を残すこと",
                file=sys.stderr,
            )

        # key ごと消えた場合は別経路 (server が直前画像を保持する) の話なので、
        # 凍結判定の連続性だけ切っておく。再登場は新しい run として数える。
        for key in [k for k in self._last_jpeg if k not in images_jpeg]:
            del self._last_jpeg[key]
            self._jpeg_changed_at.pop(key, None)
            self._freeze_warned_at.pop(key, None)

    def submit(self) -> bool:
        """Kick off the next inference. False if one is already in flight or
        the endpoints have not warmed up yet."""
        if self._pending is not None:
            return False
        obs = self.observe()
        if obs is None:
            return False
        issued = time.monotonic()
        self._pending = self._pool.submit(
            lambda: (self._link.act(obs), time.monotonic() - issued)
        )
        return True

    def ready(self) -> bool:
        return self._pending is not None and self._pending.done()

    def collect(self, block: bool = True) -> tuple[dict, float] | None:
        """Take the finished chunk and its measured latency in seconds."""
        if self._pending is None:
            return None
        if not block and not self._pending.done():
            return None
        try:
            return self._pending.result()
        finally:
            self._pending = None

    def close(self):
        # cancel_futures は Python 3.9+ で追加。Orin (JetPack 5.1.1) の default Python は
        # 3.8 で未対応 → shutdown が TypeError で exit path を crash させる (運営 onboarding
        # Finding 3、upstream template の bug)。version guard で回避。
        if sys.version_info >= (3, 9):
            self._pool.shutdown(wait=False, cancel_futures=True)
        else:
            self._pool.shutdown(wait=False)


_warned_short_chunk = False


def _skip_rows(latency_s: float, rate_hz: float, chunk_length: int) -> int:
    """How many leading rows of a fresh chunk are already stale.

    Always leaves at least one row: a chunk that took longer to compute than
    it lasts is a sign your model is too slow for the cadence, not a reason
    to send nothing.
    """
    global _warned_short_chunk
    skip = min(int(round(latency_s * rate_hz)), max(chunk_length - 1, 0))
    if not _warned_short_chunk and skip * 2 > chunk_length:
        _warned_short_chunk = True
        print(
            f"[client] WARNING: {skip} of {chunk_length} rows are stale on arrival "
            f"({latency_s * 1000:.0f}ms at {rate_hz:.0f}Hz). Most of every chunk is "
            f"being discarded. Return a longer chunk (>= {int(latency_s * rate_hz * 3)} "
            f"rows) or make inference faster, or the robot will spend most of its "
            f"time replaying the tail of an old chunk.",
            file=sys.stderr,
        )
    return skip


def run_sonic(inference: Inference, sink, prefetch_rows: int):
    """Stream latent rows at 50 Hz, prefetching the next chunk mid-playback."""
    period = 1.0 / SONIC_STEP_HZ
    token = left = right = None
    index = 0
    chunks = 0

    while True:
        # Out of rows: block for the chunk that should already be in flight.
        if token is None or index >= len(token):
            if not inference.busy:
                while not inference.submit():
                    print("[client] waiting for camera/state...")
                    time.sleep(0.1)
            action, latency = inference.collect(block=True)
            token = action["motion_token"]
            left = action["left_hand_joints"]
            right = action["right_hand_joints"]
            # Validate the whole chunk before any of it reaches the robot, so
            # a bad tail cannot be half-executed.
            token, left, right = sink.validate_chunk(token, left, right)
            index = _skip_rows(latency, SONIC_STEP_HZ, len(token))
            chunks += 1
            if chunks % 10 == 1:
                print(
                    f"[client] chunk T={len(token)} latency={latency * 1000:.0f}ms "
                    f"skip={index} peak|token|={float(abs(token).max()):.3f}"
                )

        # Enough rows left to cover the round trip? Start the next inference.
        if not inference.busy and (len(token) - index) <= prefetch_rows:
            inference.submit()

        tick = time.monotonic()
        sink.send_step(
            token[index : index + 1], left[index : index + 1], right[index : index + 1]
        )
        index += 1
        remaining = period - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)


def run_decoupled(inference: Inference, sink):
    """Publish a task-space chunk per tick, with the next one already in flight."""
    period = 1.0 / DECOUPLED_CHUNK_HZ
    chunks = 0

    while not inference.submit():
        print("[client] waiting for camera/state...")
        time.sleep(0.1)

    while True:
        tick = time.monotonic()
        action, latency = inference.collect(block=True)
        chunk = sink.validate_chunk(action["actions"])

        # The adapter owns interpolation, so hand it the trajectory minus the
        # rows that inference latency already consumed.
        skip = _skip_rows(latency, DECOUPLED_CHUNK_HZ, len(chunk))
        sink.send_chunk(chunk[skip:], issued_at=time.time() - latency)

        # Submit the next one before sleeping, not after.
        inference.submit()

        chunks += 1
        if chunks % 10 == 1:
            print(
                f"[client] chunk T={len(chunk)} latency={latency * 1000:.0f}ms skip={skip}"
            )

        remaining = period - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lane", choices=LANES, default=os.environ.get("PEVAL_LANE", "sonic")
    )
    parser.add_argument("--thor", default="192.168.100.1", help="Policy server host.")
    parser.add_argument("--thor-port", type=int, default=8765)
    parser.add_argument(
        "--orin",
        default="127.0.0.1",
        help="Host of the organizer's camera/state endpoints.",
    )
    parser.add_argument(
        "--prompt",
        default="grab the bottle",
        help="Task instruction handed to the policy.",
    )
    parser.add_argument(
        "--prefetch-rows",
        type=int,
        default=8,
        help="SONIC only: rows left in the current chunk when the "
        "next inference starts. Raise it if your model is slow.",
    )
    args = parser.parse_args()

    # live 判定だけは運営の実装をそのまま使い、hot loop は JPEG のまま運ぶ方を使う。
    cameras = CameraStream(host=args.orin)
    raw_cameras = RawCameraStream(host=args.orin)
    states = StateStream(host=args.orin)
    grippers = GripperStateStream(host=args.orin)
    sink = ActionSink.for_lane(args.lane)

    print(
        f"[client] lane={args.lane} thor={args.thor}:{args.thor_port} orin={args.orin}"
    )
    print("[client] waiting for the organizer's endpoints...")
    cameras.wait_until_live()
    # live 判定が済んだら閉じる。**hot loop は raw_cameras 側しか使わない**ので、
    # 張りっぱなしにすると bridge が同じ frame を 2 回配ることになる
    # (実測 0.338 MB x 30 Hz = 10 MB/s の二重配信)。
    cameras.close()
    state = states.wait_until_live()
    print(
        f"[client] endpoints live (hand state "
        f"{'present' if state.hands_present else 'absent — Dex1-1 rig'})"
    )
    # Dex1-1 の実測が :5557 に載っているか。載っていれば hand_state が実測になり、
    # pick_leg_hybrid の interlock も本物になる。載っていなければ合成のまま走る。
    # go-live 前に気づけるよう、ここで 1 度 poll して結果を出しておく。
    print(
        f"[client] Dex1 gripper_q: "
        f"{'present' if grippers.poll() is not None else 'absent — synthesizing'}"
    )

    link = PolicyLink(f"ws://{args.thor}:{args.thor_port}")
    declared = link.metadata.get("lane")
    if declared != args.lane:
        raise SystemExit(
            f"[client] lane mismatch: server declared {declared!r}, client is "
            f"{args.lane!r}. Start both on the same lane."
        )

    link.reset()
    inference = Inference(
        link,
        raw_cameras,
        states,
        args.prompt,
        link.metadata.get("camera_keys", ["ego_view"]),
        grippers,
    )
    try:
        if args.lane == "sonic":
            run_sonic(inference, sink, args.prefetch_rows)
        else:
            run_decoupled(inference, sink)
    except KeyboardInterrupt:
        print(
            "\n[client] interrupted — THE ROBOT IS STILL HOLDING ITS LAST COMMAND. "
            "Use the e-stop to bring it to a safe state."
        )
    except ActionError as exc:
        print(f"\n[client] ACTION REJECTED: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        inference.close()
        sink.close()
        cameras.close()  # live 判定の直後に閉じてある。二重呼び出しは LINGER=0 で安全
        raw_cameras.close()
        states.close()
        grippers.close()


if __name__ == "__main__":
    main()
