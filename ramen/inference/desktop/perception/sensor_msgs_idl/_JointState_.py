"""sensor_msgs/JointState の cyclonedds IDL 定義 (Issue #75)。

ROS2 canonical: `sensor_msgs/msg/JointState`
DDS typename: `sensor_msgs::msg::dds_::JointState_` (ROS2 mangling 準拠)

field 定義は ROS2 の msg/JointState.msg に一致:
    std_msgs/Header header
    string[] name         # joint 名 (関節毎、G1 は 29 entries)
    float64[] position    # 関節位置 [rad]
    float64[] velocity    # 関節速度 [rad/s]
    float64[] effort      # 関節トルク推定 [Nm]

`header` は `unitree_sdk2py.idl.std_msgs.msg.dds_.Header_` を流用 (SDK 既存)、
CompressedImage_ と同じ pattern。Orin 側の `real_hw_bridge_node` (Issue #65) が
rclpy `sensor_msgs.msg.JointState` で publish するのを wire 互換で subscribe する。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

if TYPE_CHECKING:
    import unitree_sdk2py.idl.std_msgs.msg.dds_  # noqa: F401

# NOTE: Do not enable ``from __future__ import annotations`` in this module.
# CycloneDDS resolves the generated-IDL-style string below as a module path;
# postponed evaluation would preserve an extra pair of quote characters and
# make the runtime resolver try to import ``'unitree_sdk2py``.


@dataclass
@annotate.final
@annotate.autoid("sequential")
class JointState_(
    idl.IdlStruct, typename="sensor_msgs::msg::dds_::JointState_"
):
    header: "unitree_sdk2py.idl.std_msgs.msg.dds_.Header_"
    name: types.sequence[str]
    position: types.sequence[types.float64]
    velocity: types.sequence[types.float64]
    effort: types.sequence[types.float64]
