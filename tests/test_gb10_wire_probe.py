import msgpack
import numpy as np
import pytest

from tools.gb10.wire_probe import decode_joint


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
