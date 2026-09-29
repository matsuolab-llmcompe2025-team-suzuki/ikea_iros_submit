import msgpack
import numpy as np
import pytest

from tools.gb10.wire_probe import decode_joint, decode_goto


def packet(rows, issued_at=1):
    return b"joint" + msgpack.packb({"dtype": "f32", "shape": list(rows.shape),
                                    "actions": rows.astype(np.float32).tobytes(),
                                    "issued_at": issued_at}, use_bin_type=True)


def test_joint_roundtrip():
    rows = np.zeros((2, 22), dtype=np.float32)
    rows[:, 21] = 0.74
    rows[:, 4:18] = np.arange(14) / 10
    decoded, stamp = decode_joint(packet(rows))
    np.testing.assert_array_equal(decoded, rows)
    assert stamp == 1


@pytest.mark.parametrize("column,value", [(0, 2), (4, np.nan), (21, 0)])
def test_invalid_value(column, value):
    rows = np.zeros((1, 22), dtype=np.float32)
    rows[:, 21] = 0.74
    rows[0, column] = value
    with pytest.raises(ValueError):
        decode_joint(packet(rows))


def test_wrong_lane():
    with pytest.raises(ValueError, match="topic"):
        decode_joint(b"taskspace")


def goto_packet(**changes):
    payload = dict(left_arm=[0.1] * 7, right_arm=[-0.1] * 7,
                   max_speed=0.3, issued_at=1234.0)
    payload.update(changes)
    return b"goto" + msgpack.packb(payload, use_bin_type=True)


def test_goto_preserves_hands_and_decodes_both_arms():
    arms, speed, stamp = decode_goto(goto_packet())
    np.testing.assert_allclose(arms, [0.1] * 7 + [-0.1] * 7)
    assert speed == 0.3 and stamp == 1234


@pytest.mark.parametrize("change", [
    {"max_speed": 0.31}, {"max_speed": 0}, {"issued_at": float("nan")},
    {"left_arm": [0] * 6}, {"right_arm": [float("nan")] * 7},
    {"hands": [0, 0]},
])
def test_invalid_preparation_goto_rejected(change):
    with pytest.raises(ValueError):
        decode_goto(goto_packet(**change))
