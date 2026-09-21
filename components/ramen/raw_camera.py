"""`:5555` のカメラ frame を **JPEG のまま**保持する SUB。

# なぜ要るか

運営 bridge は JPEG を配る (`real_orin_cameras.py:233` の `cv2.imencode`)。
`boundary/cameras.py` はそれを受け取った側で decode して RGB の ndarray にする。
ところが我々の client はその ndarray を **websocket でそのまま Thor へ送っていた**。

```
bridge :5555  JPEG 68 KB  ->  Orin で decode  ->  RGB 900 KB  ->  Thor
```

会場リグの実写で測った実サイズ (PC2 の camera_check/):

```
ego_view_left   67.6 KB      raw 900 KB   x13.3
ego_view_right  67.8 KB      raw 900 KB   x13.3
left_wrist      75.8 KB      raw 900 KB   x11.9
right_wrist     97.1 KB      raw 900 KB   x 9.3
```

5 枚で **4.61 MB / step**。Thor <-> PC2 は 1 GbE で実効 101 MB/s (実測) なので
**転送だけで 45.6 ms**。運営 adapter の `--chunk-hz 20.0` = 1 周期 50 ms を
ほぼ使い切る。さらに `max_step = --max-joint-vel / --chunk-hz` は
**こちらが送った waypoint ごと**に掛かるので、step レートがそのまま腕の速度上限
になる (`wbc_driver.py:357,426`)。

# 何をするか

**bridge が作った JPEG のバイト列をそのまま運び、展開を Thor 側でやる。**
再エンコードしないので **model が見るピクセルは今までと同一**。decode する場所が
Orin から Thor に移るだけで、Orin 側の毎 tick 5 枚 decode も消える。

```
5 枚とも送って 0.38 MB / step  ->  転送 3.8 ms
```

`boundary/cameras.py` は改変禁止なので、`gripper_state.py` と同じく同じ endpoint に
2 本目の SUB を張る (PUB は購読者数に依存しないので運営側の挙動は変わらない)。
起動時の live 判定だけは運営の `CameraStream.wait_until_live` をそのまま使う。

wire 形式は `docs/CONTRACT.md:78` に明記されている:

    msgpack frame: {"timestamps": {key: float}, "images": {key: jpeg_bytes}}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import msgpack
import zmq

#: `boundary/cameras.py` と同じ port / 必須キー。
DEFAULT_PORT = 5555
REQUIRED_KEY = "ego_view"


@dataclass
class RawCameraFrame:
    """1 回の publish を **decode せずに**保持したもの。"""

    jpegs: dict[str, bytes] = field(default_factory=dict)
    timestamps: dict[str, float] = field(default_factory=dict)
    received_at: float = 0.0

    def nbytes(self) -> int:
        return sum(len(v) for v in self.jpegs.values())


class RawCameraStream:
    """`boundary.CameraStream` と同じ形で、JPEG を decode せずに返す。

    socket 設定も同じ (SUB + CONFLATE) なので、遅れたら最新に飛ぶだけで backlog は
    溜まらない。API を揃えてあるので client 側の差し替えは 1 行で済む。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        self.endpoint = f"tcp://{host}:{port}"
        context = zmq.Context.instance()
        self._socket = context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)
        self._latest: RawCameraFrame | None = None

    def read(self, timeout_ms: int = 0) -> RawCameraFrame | None:
        """次の frame を待つ。timeout なら None。"""
        if not self._socket.poll(timeout_ms):
            return None
        frame = self._decode(self._socket.recv())
        if frame is not None:
            self._latest = frame
        return frame

    def latest(self) -> RawCameraFrame | None:
        """直近に読んだ frame (blocking しない)。"""
        return self._latest

    def wait_until_live(self, timeout_s: float = 30.0) -> RawCameraFrame:
        """`ego_view` が来るまで待つ。来なければ TimeoutError。"""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            frame = self.read(timeout_ms=500)
            if frame is not None and REQUIRED_KEY in frame.jpegs:
                return frame
        raise TimeoutError(
            f"no '{REQUIRED_KEY}' frame from {self.endpoint} within {timeout_s}s"
        )

    @staticmethod
    def _decode(blob: bytes) -> RawCameraFrame | None:
        """msgpack だけ解いて、**JPEG は触らない**。壊れていたら None。"""
        try:
            msg = msgpack.unpackb(blob, raw=False)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(msg, dict):
            return None
        images = msg.get("images")
        if not isinstance(images, dict):
            return None
        jpegs = {
            str(k): bytes(v)
            for k, v in images.items()
            if isinstance(v, (bytes, bytearray))
        }
        if not jpegs:
            return None
        timestamps = msg.get("timestamps")
        return RawCameraFrame(
            jpegs=jpegs,
            timestamps=dict(timestamps) if isinstance(timestamps, dict) else {},
            received_at=time.time(),
        )

    def close(self) -> None:
        self._socket.close(linger=0)
