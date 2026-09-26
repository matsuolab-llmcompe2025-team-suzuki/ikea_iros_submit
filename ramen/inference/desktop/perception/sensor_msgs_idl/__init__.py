"""ROS2 sensor_msgs の cyclonedds IDL 定義 (Issue #58)。

Ros2FrameSource が subscribe する ROS2 標準 msg を、rclpy を経由せず cyclonedds
で直接 wire 互換に扱うための IDL dataclass 群。ROS2 の canonical typename に一致
する形で `@idl.dataclass` を書けば、rclpy publisher が流す msg を DDS layer で
そのまま deserialize できる。

依存する `std_msgs.msg.Header_` および `builtin_interfaces.msg.Time_` は
`unitree_sdk2py.idl` に既存の SDK 定義を流用する (重複を避けつつ typename も
一致するため wire 互換)。

なぜ rclpy を使わないかの背景は Issue #58 の「Scope 決定の経緯」参照。
"""

from __future__ import annotations

from ._CompressedImage_ import CompressedImage_
from ._JointState_ import JointState_

__all__ = ["CompressedImage_", "JointState_"]
