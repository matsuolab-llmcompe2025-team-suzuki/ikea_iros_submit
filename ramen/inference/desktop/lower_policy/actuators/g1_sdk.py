"""G1SDKWalkActuator: Unitree SDK 経由で実 G1 を動かす WalkActuator 実装。

`unitree_sdk2py` と `cyclonedds` は module top-level では **import しない**
(lazy import at `__init__`)。理由: main pixi env (Python 3.12) は SDK が
入っていないので、この module を top-level で import できないと import
graph が壊れる。dispatcher / skill 側から indirect import されても
問題ないよう、SDK 呼び出しは `__init__` に閉じる。

実行時は `env/g1_runtime/` env (Python 3.10 + cyclonedds + unitree_sdk2py)
が必要。使い方: `inference/desktop/lower_policy/scripts/run_g1_walk_to_table.py`
参照。

# SDK 実装との対応 (source: third_party/unitree_sdk2_python/.../g1_loco_client.py)

    set_velocity(vx,vy,vyaw, duration=None) → SetVelocity(..., duration=864000s)
        duration を省略した skill 経路は持続歩行にする。実機の単発診断では
        有限の duration を明示して、SDK 側にも停止時刻を渡す。
    stand_up()               → HighStand()
        SetStandHeight(UINT32_MAX) 相当の setpoint 命令。冪等 (既立ちでも安全)。
        Squat2StandUp() は FSM 遷移 (SetFsmId(706)) なので、既立ち時に呼ぶと
        squat → stand の再遷移で motor が一瞬 unload される可能性があり不採用。
    damp()                   → Damp()  # FSM 1 = damping state
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass


class G1SDKCommandError(RuntimeError):
    """Raised when the G1 locomotion service rejects a command."""


@dataclass(frozen=True)
class G1LocoStatus:
    """Read-only high-level locomotion status used by the safety preflight."""

    client_api_version: str
    server_api_version: str
    fsm_id: int
    fsm_mode: int


class G1SDKWalkActuator:
    """Unitree SDK LocoClient を叩く WalkActuator 実装 (実機用)。

    前提: robot が **walking FSM state (立ち姿勢)** に既に入っていること。
        Damp からの startup は operator の責任 (Unitree 標準リモコン等)、
        skill / actuator の scope 外。

    Args:
        interface: DDS を流す network interface 名 (`ip a` で確認)。
                   sim では "lo"、実機では "eth0" 等。
        domain_id: DDS domain ID (sim / 実機の設定と一致させる)。default 0。
        timeout: LocoClient RPC の timeout [s]。default 10.0。
        client: 既存の LocoClient instance を注入する場合に使う (test 用)。
                None の場合は通常経路で SDK を初期化する。

    Note:
        通常経路の `__init__` で `ChannelFactoryInitialize` を呼ぶ (プロセス
        全体の global singleton の初期化を伴う)。同一プロセスで複数の
        actuator を作らないこと。
    """

    def __init__(
        self,
        interface: str = "eth0",
        domain_id: int = 0,
        timeout: float = 10.0,
        client: object | None = None,
    ) -> None:
        if client is not None:
            # test injection: SDK 初期化を skip
            self._client = client
            return

        # lazy import: main env (SDK なし) から本 module を import しても
        # ImportError にならないよう、__init__ 内に閉じる。
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        ChannelFactoryInitialize(domain_id, interface)
        self._client = LocoClient()
        self._client.SetTimeout(timeout)
        self._client.Init()

    @staticmethod
    def _require_success(command: str, code: object) -> None:
        """Report the Loco RPC result and fail closed on a rejected command."""

        if not isinstance(code, int) or isinstance(code, bool):
            raise G1SDKCommandError(
                f"{command} returned a non-integer SDK result: {code!r}"
            )
        print(f"[g1_sdk] {command}: code={code}", file=sys.stderr, flush=True)
        if code != 0:
            raise G1SDKCommandError(f"{command} rejected by G1 Loco service: code={code}")

    def get_loco_status(self) -> G1LocoStatus:
        """Read the active sport service/FSM state without issuing motor commands."""

        version_code, server_api_version = self._client.GetServerApiVersion()
        self._require_success("GetServerApiVersion", version_code)

        fsm_code, fsm_id = self._client.GetFsmId()
        self._require_success("GetFsmId", fsm_code)

        # The current Python SDK registers API 7002 but does not expose the
        # GetFsmMode wrapper that exists in the official C++ SDK.
        fsm_mode_code, raw_fsm_mode = self._client._Call(7002, json.dumps({}))
        self._require_success("GetFsmMode", fsm_mode_code)
        try:
            fsm_mode = json.loads(raw_fsm_mode)["data"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise G1SDKCommandError(
                f"GetFsmMode returned malformed data: {raw_fsm_mode!r}"
            ) from exc

        client_api_version = self._client.GetApiVersion()
        if not isinstance(client_api_version, str) or not isinstance(
            server_api_version, str
        ):
            raise G1SDKCommandError(
                "sport API version response is malformed: "
                f"client={client_api_version!r}, server={server_api_version!r}"
            )
        if not isinstance(fsm_id, int) or isinstance(fsm_id, bool):
            raise G1SDKCommandError(f"GetFsmId returned malformed data: {fsm_id!r}")
        if not isinstance(fsm_mode, int) or isinstance(fsm_mode, bool):
            raise G1SDKCommandError(
                f"GetFsmMode returned malformed data: {fsm_mode!r}"
            )

        return G1LocoStatus(
            client_api_version=client_api_version,
            server_api_version=server_api_version,
            fsm_id=fsm_id,
            fsm_mode=fsm_mode,
        )

    def set_velocity(
        self,
        vx: float,
        vy: float,
        vyaw: float,
        *,
        duration: float | None = None,
    ) -> None:
        """Send a velocity command, optionally with a finite SDK-side timeout."""

        # LocoClient.Move() discards SetVelocity's RPC result.  Use the direct
        # service call so rejection is visible.  Normal skills retain the
        # continuous command, while the walk diagnostic passes its requested
        # finite duration through to the robot firmware.
        sdk_duration = 864000.0 if duration is None else duration
        code = self._client.SetVelocity(vx, vy, vyaw, duration=sdk_duration)
        self._require_success(f"SetVelocity(duration={sdk_duration:g}s)", code)

    def stand_up(self) -> None:
        # HighStand() is a wrapper that discards SetStandHeight's RPC result.
        # Keep the same idempotent high-stand setpoint but surface rejection.
        code = self._client.SetStandHeight((1 << 32) - 1)
        self._require_success("SetStandHeight(high)", code)

    def damp(self) -> None:
        # モータフリー。立ち状態で呼ぶと崩れ落ちるため、通常の停止経路からは
        # 呼ばない。ハーネス支持等を確認した専用手順だけで使用する。
        code = self._client.SetFsmId(1)
        self._require_success("SetFsmId(damp)", code)
