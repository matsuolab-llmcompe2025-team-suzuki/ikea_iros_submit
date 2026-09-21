"""JPEG をそのまま運ぶ経路が、旧経路と **同じピクセル**を出すことを固定する。

運営 bridge は JPEG を配る (`CONTRACT.md:78`)。以前は `boundary/cameras.py` が
Orin 側で decode し、生 RGB を websocket に流していた。会場リグの実写では 1 枚
68〜97 KB の JPEG が raw 900 KB に膨らみ、5 枚で 4.61 MB/step。Thor <-> PC2 は
1 GbE 実効 101 MB/s (実測) なので **転送だけで 45.6 ms** = 運営 adapter の
1 周期 50 ms をほぼ使い切っていた。

いまは JPEG のまま運び、`components/server.py` が policy へ渡す直前に展開する。
**再エンコードしていないので、model が見るピクセルは以前と同一**でなければならない。
ここがズレると全 policy が静かに壊れる (色反転・解像度違いは例外にならない)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import msgpack
import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_VENDOR_DESKTOP = _HERE.parent / "vendor" / "desktop"
for _p in (str(_VENDOR_DESKTOP), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from boundary.cameras import CameraStream  # noqa: E402
from components.ramen.raw_camera import RawCameraStream  # noqa: E402
from components.server import decode_obs_images  # noqa: E402


def _scene(seed: int) -> np.ndarray:
    """JPEG が素直に潰れない程度に構造のある BGR 画像。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(0, 255, 480, dtype=np.uint8)[:, None]
    img[100:300, 200:400] = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    return img


def _wire(keys_to_bgr: dict[str, np.ndarray]) -> bytes:
    """bridge が :5555 に出すのと同じ 1 通 (BGR JPEG)。"""
    images, timestamps = {}, {}
    for key, bgr in keys_to_bgr.items():
        ok, jpeg = cv2.imencode(".jpg", bgr)
        assert ok
        images[key] = jpeg.tobytes()
        timestamps[key] = 1.0
    return msgpack.packb(
        {"timestamps": timestamps, "images": images}, use_bin_type=True
    )


# ---------------------------------------------------------------- 同一性 (最重要)
def test_the_decoded_pixels_match_the_old_orin_side_decode():
    """`server.decode_obs_images` が `boundary.CameraStream._decode` と一致すること。

    **これが崩れると全 policy が静かに壊れる。** 例外にはならず、色が反転したまま、
    あるいは BGR/RGB が入れ替わったまま推論が続く。
    """
    scene = {"ego_view": _scene(1), "left_wrist": _scene(2), "right_wrist": _scene(3)}
    blob = _wire(scene)

    old = CameraStream._decode(blob).images  # 旧: Orin 側で decode
    new_frame = RawCameraStream._decode(blob)  # 新: JPEG のまま運ぶ
    new = decode_obs_images({"images_jpeg": new_frame.jpegs})["images"]

    assert sorted(old) == sorted(new)
    for key in old:
        assert old[key].shape == new[key].shape == (480, 640, 3)
        assert old[key].dtype == new[key].dtype == np.uint8
        assert np.array_equal(old[key], new[key]), f"{key} のピクセルが一致しない"


def test_the_payload_shrinks_by_an_order_of_magnitude():
    """生 RGB より 1 桁小さいこと (これが変更の理由そのもの)。"""
    blob = _wire({"ego_view": _scene(4), "left_wrist": _scene(5)})
    frame = RawCameraStream._decode(blob)

    jpeg_bytes = sum(len(v) for v in frame.jpegs.values())
    raw_bytes = 2 * 480 * 640 * 3

    assert jpeg_bytes * 5 < raw_bytes, (
        f"圧縮率が想定より悪い: jpeg {jpeg_bytes} vs raw {raw_bytes}"
    )


# ---------------------------------------------------------------- decode の頑健性
def test_a_broken_frame_is_dropped_instead_of_raising():
    assert RawCameraStream._decode(b"not msgpack at all") is None
    assert RawCameraStream._decode(msgpack.packb([1, 2, 3])) is None
    assert RawCameraStream._decode(msgpack.packb({"timestamps": {}})) is None
    assert RawCameraStream._decode(msgpack.packb({"images": {}})) is None


def test_non_binary_image_values_are_ignored():
    """JPEG でない値が混ざっても他のキーを巻き込まないこと。"""
    good = _wire({"ego_view": _scene(6)})
    msg = msgpack.unpackb(good, raw=False)
    msg["images"]["left_wrist"] = "これは JPEG ではない"
    frame = RawCameraStream._decode(msgpack.packb(msg, use_bin_type=True))

    assert sorted(frame.jpegs) == ["ego_view"]


def test_the_received_time_is_stamped_on_arrival():
    """`obs["t"]` の元になるので、frame ごとに刻まれること。"""
    import time

    before = time.time()
    frame = RawCameraStream._decode(_wire({"ego_view": _scene(7)}))
    after = time.time()

    assert before <= frame.received_at <= after


# ---------------------------------------------------------------- server 側の展開
def test_an_obs_without_jpegs_is_left_alone():
    """warmup の dummy obs や旧 client をそのまま通すこと。"""
    raw = np.zeros((480, 640, 3), np.uint8)
    obs = {"images": {"ego_view": raw}, "body_q": np.zeros(29)}

    assert decode_obs_images(obs) is obs


def test_an_explicit_raw_image_wins_over_the_jpeg():
    """同じキーが両方にあるとき、生画像を優先すること (warmup 用)。"""
    raw = np.full((480, 640, 3), 7, np.uint8)
    frame = RawCameraStream._decode(_wire({"ego_view": _scene(8)}))
    out = decode_obs_images({"images": {"ego_view": raw}, "images_jpeg": frame.jpegs})

    assert np.array_equal(out["images"]["ego_view"], raw)


def test_a_corrupt_jpeg_drops_one_key_without_killing_the_step():
    """壊れた 1 枚で run を落とさないこと (driver が直前の画像を保持する)。"""
    frame = RawCameraStream._decode(
        _wire({"ego_view": _scene(9), "left_wrist": _scene(10)})
    )
    jpegs = dict(frame.jpegs)
    jpegs["left_wrist"] = b"\xff\xd8broken"

    out = decode_obs_images({"images_jpeg": jpegs})["images"]

    assert sorted(out) == ["ego_view"]


def test_the_jpeg_key_is_removed_so_policies_see_the_old_shape():
    """policy には `images` だけを渡すこと (obs の形を増やさない)。"""
    frame = RawCameraStream._decode(_wire({"ego_view": _scene(11)}))
    out = decode_obs_images({"images_jpeg": frame.jpegs, "prompt": "x"})

    assert "images_jpeg" not in out
    assert sorted(out) == ["images", "prompt"]


def test_other_observation_keys_survive_the_decode():
    """body_q / gripper_q / t を落とさないこと。"""
    frame = RawCameraStream._decode(_wire({"ego_view": _scene(12)}))
    obs = {
        "images_jpeg": frame.jpegs,
        "body_q": np.zeros(29, np.float32),
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0], np.float32),
        "gripper_q": {"left": {"q": 0.0}, "right": {"q": -5.3}},
        "t": 12.5,
    }
    out = decode_obs_images(obs)

    assert out["gripper_q"] == obs["gripper_q"]
    assert out["t"] == pytest.approx(12.5)
    assert out["body_q"].shape == (29,)
