#!/usr/bin/env bash
# Start the AVP stereo display only.  This intentionally has no Unitree import.
set -euo pipefail

XR_ENV="${XR_TELEOP_ENV:-xr-teleop}"
CONDA_EXE="${XR_TELEOP_CONDA:-$HOME/miniconda3/condabin/conda}"
IMAGE_SERVER_IP="${G1_IMAGE_SERVER_IP:-192.168.29.159}"
AVP_DESKTOP_IP="${AVP_DESKTOP_IP:-192.168.29.175}"
XR_TELEOP_CERT="${XR_TELEOP_CERT:-$HOME/.config/xr_teleoperate_avp/cert.pem}"
XR_TELEOP_KEY="${XR_TELEOP_KEY:-$HOME/.config/xr_teleoperate_avp/key.pem}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -r "$XR_TELEOP_CERT" || ! -r "$XR_TELEOP_KEY" ]]; then
  echo "ERROR: Apple Vision Pro TLS certificate/key are missing." >&2
  exit 1
fi
if [[ ! -x "$CONDA_EXE" ]]; then
  echo "ERROR: conda executable is missing: $CONDA_EXE" >&2
  exit 1
fi
if ! ping -c 1 -W 1 "$IMAGE_SERVER_IP" >/dev/null; then
  echo "ERROR: Orin image server is unreachable: $IMAGE_SERVER_IP" >&2
  exit 1
fi
if ss -ltn | grep -q ':8012\b'; then
  echo "ERROR: TCP/8012 is already in use; stop the existing AVP view/control process first." >&2
  exit 1
fi

export XR_TELEOP_CERT XR_TELEOP_KEY AVP_DESKTOP_IP
exec "$CONDA_EXE" run --no-capture-output -n "$XR_ENV" python \
  "$SCRIPT_DIR/avp_stereo_view_only.py" --image-server-ip "$IMAGE_SERVER_IP"
