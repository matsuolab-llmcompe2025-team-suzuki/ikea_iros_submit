#!/usr/bin/env bash
# Start the isolated, official TeleImager instance on the Orin host.
#
# This process only reads the verified head-camera V4L2 node and sends images.
# It never publishes a Unitree motor command.  USB /dev/videoN numbering can
# change after a reboot or reconnect, so the script verifies the node on every
# launch and updates the copied TeleImager configuration only when there is one
# unambiguous matching candidate.  Copy cam_config_server.avp_head_only.yaml to
# $XR_TELEIMAGER_ROOT/cam_config_server.yaml before the first launch.
set -euo pipefail

XR_TELEIMAGER_ROOT="${XR_TELEIMAGER_ROOT:-/home/unitree/xr-teleimager-official}"
XR_TELEIMAGER_VENV="${XR_TELEIMAGER_VENV:-/home/unitree/.venvs/xr-teleimager-py38}"
XR_TELEOP_CERT="${XR_TELEOP_CERT:-$HOME/.config/xr_teleoperate_avp/cert.pem}"
XR_TELEOP_KEY="${XR_TELEOP_KEY:-$HOME/.config/xr_teleoperate_avp/key.pem}"
EXPECTED_COMMIT="89d461330479ed0d71d642e092acea9e9fe71494"
CONFIG_PATH="$XR_TELEIMAGER_ROOT/cam_config_server.yaml"
MODE="run"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--check-only | --run]

  --check-only  Inspect and validate the head-camera node without starting a server.
  --run         Validate the node, update video_id if it changed, then start TeleImager.

The expected head-camera geometry comes from head_camera.image_shape in
cam_config_server.yaml.  By default, the configured node is kept when it
matches.  If it no longer matches, exactly one matching V4L2 node is selected
automatically.  Set G1_HEAD_CAMERA_VIDEO_ID=N to select a verified /dev/videoN
explicitly when there are multiple candidates.
EOF
}

case "${1:---run}" in
  --check-only) MODE="check" ;;
  --run) MODE="run" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }

export XR_TELEOP_CERT XR_TELEOP_KEY
# The launcher verifies the selected OpenCV video_id before reaching this
# point. Avoid the upstream all-device UVC scan: it can block on unrelated D405
# nodes and is not required for this head-only service.
export TELEIMAGER_SKIP_UVC_DISCOVERY=1

if [[ ! -x "$XR_TELEIMAGER_VENV/bin/python" ]]; then
  echo "ERROR: TeleImager venv is missing: $XR_TELEIMAGER_VENV" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: camera configuration is missing." >&2
  exit 1
fi
if [[ ! -r "$XR_TELEOP_CERT" || ! -r "$XR_TELEOP_KEY" ]]; then
  echo "ERROR: WebRTC TLS certificate/key are missing from ~/.config/xr_teleoperate_avp/." >&2
  exit 1
fi
if [[ "$(git -C "$XR_TELEIMAGER_ROOT" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "ERROR: unexpected TeleImager source revision; refusing to start." >&2
  exit 1
fi

if ! command -v v4l2-ctl >/dev/null; then
  echo "ERROR: v4l2-ctl is required to verify the head camera; install v4l-utils." >&2
  exit 1
fi

readarray -t camera_config < <("$XR_TELEIMAGER_VENV/bin/python" - "$CONFIG_PATH" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    head = yaml.safe_load(f)["head_camera"]

shape = head.get("image_shape")
video_id = head.get("video_id")
if not isinstance(shape, list) or len(shape) != 2 or not all(isinstance(v, int) for v in shape):
    raise SystemExit("head_camera.image_shape must be [height, width]")
if not isinstance(video_id, int):
    raise SystemExit("head_camera.video_id must be an integer")
print(video_id)
print(f"{shape[1]}x{shape[0]}")
PY
)
configured_video_id="${camera_config[0]}"
expected_geometry="${camera_config[1]}"

matching_nodes=()
for node in /dev/video[0-9]*; do
  [[ -e "$node" ]] || continue
  # Do not pipe directly to `grep -q` here: with `pipefail`, grep exits as
  # soon as it finds a match and v4l2-ctl can then receive SIGPIPE for a long
  # format list, incorrectly rejecting a valid camera.
  format_list="$(timeout 2s v4l2-ctl -d "$node" --list-formats-ext 2>/dev/null || true)"
  if grep -Eq "Size: Discrete ${expected_geometry}" <<<"$format_list"; then
    matching_nodes+=("$node")
  fi
done

configured_node="/dev/video${configured_video_id}"
requested_video_id="${G1_HEAD_CAMERA_VIDEO_ID:-auto}"
selected_node=""
if [[ "$requested_video_id" != "auto" ]]; then
  [[ "$requested_video_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: G1_HEAD_CAMERA_VIDEO_ID must be a numeric V4L2 index or auto." >&2
    exit 2
  }
  selected_node="/dev/video${requested_video_id}"
  if [[ ! " ${matching_nodes[*]} " =~ " ${selected_node} " ]]; then
    echo "ERROR: requested ${selected_node} does not advertise ${expected_geometry}." >&2
    exit 1
  fi
elif [[ " ${matching_nodes[*]} " =~ " ${configured_node} " ]]; then
  selected_node="$configured_node"
elif (( ${#matching_nodes[@]} == 1 )); then
  selected_node="${matching_nodes[0]}"
else
  echo "ERROR: cannot select the head camera unambiguously." >&2
  echo "Configured node: ${configured_node}; expected geometry: ${expected_geometry}" >&2
  echo "Matching nodes: ${matching_nodes[*]:-(none)}" >&2
  echo "Reconnect/check the head camera, then rerun with G1_HEAD_CAMERA_VIDEO_ID=N after verification." >&2
  exit 1
fi

echo "Head-camera verification: configured=${configured_node}, selected=${selected_node}, expected=${expected_geometry}"
timeout 2s v4l2-ctl -d "$selected_node" -D 2>/dev/null | sed -n '1,8p' || true

if [[ "$MODE" == "check" ]]; then
  echo "Head-camera preflight passed.  TeleImager was not started."
  exit 0
fi

selected_video_id="${selected_node#/dev/video}"
if [[ "$selected_video_id" != "$configured_video_id" ]]; then
  "$XR_TELEIMAGER_VENV/bin/python" - "$CONFIG_PATH" "$selected_video_id" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
video_id = sys.argv[2]
text = path.read_text(encoding="utf-8")
pattern = r"(head_camera:\s*.*?^\s*video_id:\s*)[^\s#]+"
updated, count = re.subn(pattern, rf"\g<1>{video_id}", text, count=1, flags=re.MULTILINE | re.DOTALL)
if count != 1:
    raise SystemExit("could not update head_camera.video_id in camera configuration")
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(updated, encoding="utf-8")
temporary.replace(path)
print(f"Updated head_camera.video_id: {video_id}")
PY
fi

if pgrep -af '[t]eleimager(-server|\.image_server)' >/dev/null; then
  echo "ERROR: a TeleImager image server is already running; do not start a second camera owner." >&2
  pgrep -af '[t]eleimager(-server|\.image_server)' >&2
  exit 1
fi
if pgrep -af '[u]sb_cam_node_exe' >/dev/null; then
  echo "ERROR: the selected head camera may be owned by ROS usb_cam." >&2
  echo "Stop the dedicated Orin camera bringup first, then retry this XR image service." >&2
  echo "This launcher will not stop another camera process automatically." >&2
  pgrep -af '[u]sb_cam_node_exe' >&2
  exit 1
fi

echo "Starting head-stereo image server only (no robot-control publisher)."
exec "$XR_TELEIMAGER_VENV/bin/python" -m teleimager.image_server --no-affinity
