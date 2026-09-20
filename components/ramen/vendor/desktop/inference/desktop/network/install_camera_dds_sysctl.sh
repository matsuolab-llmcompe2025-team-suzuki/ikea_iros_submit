#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/99-iros-camera-dds.conf"
DESTINATION="/etc/sysctl.d/99-iros-camera-dds.conf"

if [[ ! -f "$SOURCE" ]]; then
  echo "ERROR: missing $SOURCE" >&2
  exit 1
fi

sudo install -m 0644 "$SOURCE" "$DESTINATION"
sudo sysctl --system >/dev/null

actual_max="$(sysctl -n net.core.rmem_max)"
actual_backlog="$(sysctl -n net.core.netdev_max_backlog)"
if (( actual_max < 16777216 || actual_backlog < 4096 )); then
  echo "ERROR: camera DDS sysctl settings were not applied" >&2
  exit 1
fi
echo "camera-dds-sysctl-ok rmem_max=$actual_max netdev_max_backlog=$actual_backlog"
