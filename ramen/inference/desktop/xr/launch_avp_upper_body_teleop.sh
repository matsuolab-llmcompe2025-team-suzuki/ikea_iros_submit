#!/usr/bin/env bash
# Launch official Unitree XR teleoperation with safe fixed choices for G1.
#
# The only enabled robot command paths are the official `rt/arm_sdk` motion-mode
# arm controller and the official Dex1-1 gripper topics.  Walking/controller
# input and debug `rt/lowcmd` are excluded.
set -euo pipefail

MODE="check"
DDS_INTERFACE="${G1_DDS_INTERFACE:-enx18c2bf548e45}"
IMAGE_SERVER_IP="${G1_IMAGE_SERVER_IP:-192.168.29.159}"
AVP_DESKTOP_IP="${AVP_DESKTOP_IP:-192.168.29.175}"
XR_ROOT="${XR_TELEOP_ROOT:-/home/ubuntu/GitHub/xr_teleoperate}"
XR_ENV="${XR_TELEOP_ENV:-xr-teleop}"
CONDA_EXE="${XR_TELEOP_CONDA:-$HOME/miniconda3/condabin/conda}"
OFFICIAL_ENTRYPOINT="$XR_ROOT/teleop/teleop_hand_and_arm.py"
STEREO_CHECK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/check_avp_stereo_stream.py"
DEX1_CHECK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/check_dex1_state.py"
EXPECTED_COMMIT="4c0afdb81f2c19709f053c9c9890a98aca687b44"
XR_TELEOP_CERT="${XR_TELEOP_CERT:-$HOME/.config/xr_teleoperate_avp/cert.pem}"
XR_TELEOP_KEY="${XR_TELEOP_KEY:-$HOME/.config/xr_teleoperate_avp/key.pem}"

export XR_TELEOP_CERT XR_TELEOP_KEY AVP_DESKTOP_IP

usage() {
  cat <<EOF
Usage: $(basename "$0") [--check-only | --run]

  --check-only  Validate only; default.  Sends no Unitree command.
  --run         Launch official XR after the non-actuating preflight passes.

Fixed safety choices in --run:
  --motion --input-mode hand --arm G1_29 --ee dex1

This keeps G1 in Regular Mode and uses the official rt/arm_sdk path plus the
official Dex1-1 gripper path.  Pinching thumb and index finger controls each
gripper.  It does not enable controller-based locomotion or the Debug Mode
rt/lowcmd path.
EOF
}

case "${1:---check-only}" in
  --check-only) MODE="check" ;;
  --run) MODE="run" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }

fail=0
check() {
  local label="$1"
  shift
  if "$@"; then
    echo "OK: $label"
  else
    echo "NG: $label" >&2
    fail=1
  fi
}

check "DDS interface exists" ip link show dev "$DDS_INTERFACE"
check "DDS interface is UP" bash -c "ip -o link show dev '$DDS_INTERFACE' | grep -q 'UP'"
check "conda executable exists" test -x "$CONDA_EXE"
check "official entrypoint exists" test -f "$OFFICIAL_ENTRYPOINT"
check "official source revision is pinned" bash -c "test \"\$(git -C '$XR_ROOT' rev-parse HEAD)\" = '$EXPECTED_COMMIT'"
check "Apple Vision Pro TLS certificate exists" test -r "$XR_TELEOP_CERT"
check "Apple Vision Pro TLS key exists" test -r "$XR_TELEOP_KEY"
check "XR Python environment imports" "$CONDA_EXE" run --no-capture-output -n "$XR_ENV" python -c 'from unitree_sdk2py.core.channel import ChannelFactoryInitialize; from televuer import TeleVuerWrapper; from teleimager.image_client import ImageClient; print("imports-ok")'
# Import the selected Dex1-1 controller before launching the official program.
# This keeps a missing optional hand dependency from being discovered only after
# the arm controller has started its initial-pose sequence.
check "Dex1-1 controller imports" env "PYTHONPATH=$XR_ROOT" \
  "$CONDA_EXE" run --no-capture-output -n "$XR_ENV" python -c 'from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller; print("dex1-imports-ok")'
# Dex1-1 is served by dex1_1_gripper_server on Orin/PC2, not by G1's normal
# arm DDS service.  Verify both state topics before the official program starts
# its arm initial-pose sequence, so a stopped PC2 service cannot strand startup.
check "Dex1-1 state topics are live" \
  "$CONDA_EXE" run --no-capture-output -n "$XR_ENV" python "$DEX1_CHECK" --interface "$DDS_INTERFACE"
check "Orin image-server host is reachable" ping -c 1 -W 1 "$IMAGE_SERVER_IP"
check "Orin sends 640x480 left/right head-stereo frames" \
  "$CONDA_EXE" run --no-capture-output -n "$XR_ENV" python "$STEREO_CHECK" --host "$IMAGE_SERVER_IP"

# TeleVuer always binds TCP/8012.  Refuse to start before creating an arm or
# Dex1-1 publisher if an earlier display-only session still owns that port.
if ss -ltn | grep -q ':8012\b'; then
  echo "NG: TCP/8012 is occupied (stop the display-only AVP viewer first)." >&2
  fail=1
else
  echo "OK: TCP/8012 is free for the official XR session"
fi

conflicts="$(ps -eo pid=,args= | grep -E '[r]un_g1_walk_to_table|[i]nference\.desktop\.entrypoint|[t]eleop_hand_and_arm\.py|[g]1_loco_client_example\.py|[g]1_arm[57]_sdk_dds_example\.py|[g]1_low_level_example\.py' || true)"
if [[ -n "$conflicts" ]]; then
  echo "NG: another known robot-control process is running:" >&2
  echo "$conflicts" >&2
  fail=1
else
  echo "OK: no known walk/orchestrator/XR control process is running"
fi

if (( fail )); then
  echo "Preflight failed.  Nothing was launched and no Unitree command was sent." >&2
  exit 1
fi

echo "Preflight passed.  This check sent no Unitree command."
if [[ "$MODE" == "check" ]]; then
  exit 0
fi

cat <<EOF

About to start official XR teleoperation.
  - G1 must already be standing in Regular Mode, with harness and E-stop ready.
  - The 14 arm joints (shoulder/elbow/wrist) and the two 1-DoF Dex1-1
    grippers are teleoperated. Walking, waist, and legs are not enabled.
  - Dex1-1 follows the official thumb-index pinch mapping. The upstream
    program moves the arms and grippers toward its initial control pose during
    startup; this is intentional. Keep both grippers clear of objects.
  - The upstream program transitions the arms to its official initial
    motion-control pose on start. Match that pose with your own arms before
    pressing r, after verifying the Apple Vision Pro view.
  - q or Ctrl-C asks the official program to bring arms home and release them.

Open the Vision Pro view using this exact URL (the Desktop port is WSS, not a
standalone rendering page):
  https://${AVP_DESKTOP_IP}:8012/?ws=wss://${AVP_DESKTOP_IP}:8012&grid=False
EOF

# The upstream IK model uses paths relative to xr_teleoperate/teleop.
cd "$XR_ROOT/teleop"
exec "$CONDA_EXE" run --no-capture-output -n "$XR_ENV" python "teleop_hand_and_arm.py" \
  --motion \
  --input-mode hand \
  --display-mode immersive \
  --arm G1_29 \
  --ee dex1 \
  --img-server-ip "$IMAGE_SERVER_IP" \
  --network-interface "$DDS_INTERFACE"
