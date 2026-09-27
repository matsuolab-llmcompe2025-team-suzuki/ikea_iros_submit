import struct

import pytest

from tools.gb10.organizer_replay import read_packets


def test_framed_capture_roundtrip(tmp_path):
    path = tmp_path / "capture.wire"
    packets = [b"joint first", b"joint second"]
    path.write_bytes(b"".join(struct.pack("!I", len(p)) + p for p in packets))
    assert list(read_packets(path)) == packets


@pytest.mark.parametrize("content", [b"x", struct.pack("!I", 3) + b"x", struct.pack("!I", 0), struct.pack("!I", 1_000_001)])
def test_corrupt_capture_fails_closed(tmp_path, content):
    path = tmp_path / "capture.wire"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        list(read_packets(path))
