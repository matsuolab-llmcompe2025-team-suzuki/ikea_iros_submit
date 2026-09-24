"""sensor_msgs/CompressedImage の cyclonedds IDL 定義 (Issue #58)。

ROS2 canonical: `sensor_msgs/msg/CompressedImage`
DDS typename: `sensor_msgs::msg::dds_::CompressedImage_` (ROS2 mangling 準拠)

field 定義は ROS2 の msg/CompressedImage.msg に一致:
    std_msgs/Header header
    string format         # 例: "jpeg", "png"
    uint8[] data          # 圧縮された画像 bytes

`header` は `unitree_sdk2py.idl.std_msgs.msg.dds_.Header_` を流用 (SDK 既存)。
これにより Header_ の typename も `std_msgs.msg.dds_.Header_` (canonical) と一致し、
rclpy publisher が流す msg と wire 互換で subscribe できる。
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
class CompressedImage_(
    idl.IdlStruct, typename="sensor_msgs::msg::dds_::CompressedImage_"
):
    header: "unitree_sdk2py.idl.std_msgs.msg.dds_.Header_"
    format: str
    data: types.sequence[types.uint8]
