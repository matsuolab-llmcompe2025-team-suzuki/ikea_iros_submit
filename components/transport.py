"""Thor <-> Orin transport for the reference implementation.

TEAM-OWNED. This link is entirely inside your submission — the organizer
never speaks it and never inspects it. Keep it, rewrite it, or replace it
with gRPC, shared memory, or whatever your stack already uses. It is here
so you have something working on day one, not because it is required.

What it is: WebSocket carrying msgpack, with a small hook so numpy arrays
survive the trip. The server announces its own contract in a metadata frame
the instant a client connects, so the client can adapt without a config file
being kept in sync across two machines.

    server.py on the Thor  ── ws://192.168.100.1:8765 ──  client.py on the Orin

Two settings that matter on real hardware:

  * ping_interval=None — a cold first inference can take longer than the
    default keepalive timeout, and we do not want that read as a dead peer.
  * max_size=None — observation frames carry three 480x640x3 images.
"""

from __future__ import annotations

import sys

import msgpack
import numpy as np
from websockets.sync.client import connect as _ws_connect
from websockets.sync.server import serve as _ws_serve

import websockets as _websockets

# websockets の sync API が ping_interval/ping_timeout を受けるのは 14+。実 Orin は
# Python 3.8 = websockets 13.x が上限で、13.x の sync connect()/serve() は未知 kwarg を
# socket.create_connection に転送して TypeError で落ちる (client 即死)。version で切替。
_WS_MAJOR = int(_websockets.__version__.split(".")[0])
_KEEPALIVE = {"ping_interval": None, "ping_timeout": None} if _WS_MAJOR >= 14 else {}

__all__ = ["PolicyLink", "serve_policy"]

_ND = "~nd"   # marker key for an encoded ndarray


def _encode(value):
    """msgpack ``default`` hook: describe an ndarray by buffer, dtype, shape."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OVc":
            raise TypeError(f"refusing to send dtype {value.dtype!r} over the wire")
        return {_ND: [np.ascontiguousarray(value).tobytes(), value.dtype.str, list(value.shape)]}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _decode(obj):
    """msgpack ``object_hook``: rebuild whatever :func:`_encode` wrote."""
    payload = obj.get(_ND)
    if payload is None:
        return obj
    buffer, dtype, shape = payload
    return np.frombuffer(buffer, dtype=np.dtype(dtype)).reshape(shape)


def _pack(obj) -> bytes:
    return msgpack.packb(obj, default=_encode, use_bin_type=True)


def _unpack(blob):
    return msgpack.unpackb(blob, object_hook=_decode, raw=False)


# ---------------------------------------------------------------------------
# Orin side
# ---------------------------------------------------------------------------


class PolicyLink:
    """Handle to the policy running on the Thor, used like a local object."""

    def __init__(self, url: str, connect_timeout_s: float = 60.0):
        import time

        deadline = time.time() + connect_timeout_s
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                self._socket = _ws_connect(
                    url, compression=None, max_size=None,
                    **_KEEPALIVE,
                )
                break
            except OSError as exc:      # server not up yet — normal at startup
                last_error = exc
                print(f"[link] {url} not answering; retrying...")
                time.sleep(2.0)
        else:
            raise TimeoutError(f"could not reach {url} in {connect_timeout_s}s") from last_error

        self.url = url
        self.metadata = _unpack(self._socket.recv())
        print(f"[link] connected to {url}; server metadata: {self.metadata}")

    def _call(self, op: str, **payload):
        self._socket.send(_pack({"op": op, **payload}))
        reply = self._socket.recv()
        if isinstance(reply, str):          # text frame == server-side failure
            raise RuntimeError(f"policy server failed:\n{reply}")
        return _unpack(reply)

    def act(self, observation: dict) -> dict:
        return self._call("act", obs=observation)

    def reset(self) -> dict:
        return self._call("reset")

    def close(self):
        self._socket.close()


# ---------------------------------------------------------------------------
# Thor side
# ---------------------------------------------------------------------------


def serve_policy(policy, host: str = "0.0.0.0", port: int = 8765):
    """Serve ``policy`` forever. Blocks.

    ``policy`` needs three members: a ``metadata`` property, ``act(obs)``,
    and ``reset()``.
    """
    import traceback

    from websockets.exceptions import ConnectionClosed

    routes = {"act": lambda req: policy.act(req["obs"]), "reset": lambda req: policy.reset()}

    def handler(socket):
        peer = socket.remote_address
        print(f"[serve] client {peer} connected")
        socket.send(_pack(policy.metadata))
        try:
            for frame in socket:
                request = _unpack(frame)
                route = routes.get(request.get("op"))
                if route is None:
                    raise ValueError(f"unknown op {request.get('op')!r}")
                socket.send(_pack(route(request)))
        except ConnectionClosed:
            pass                    # the client went away; nothing to report
        except Exception:
            # Hand the traceback back as a text frame so the operator on the
            # Orin sees the Thor-side failure without SSHing across. If the
            # client is already gone there is nobody to tell.
            print(traceback.format_exc(), file=sys.stderr)
            try:
                socket.send(traceback.format_exc())
            except ConnectionClosed:
                pass
        finally:
            print(f"[serve] client {peer} disconnected")

    with _ws_serve(
        handler, host, port,
        compression=None, max_size=None,
        **_KEEPALIVE,
    ) as server:
        print(f"[serve] policy server listening on ws://{host}:{port}")
        server.serve_forever()
