#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export POLICY_TYPE="${POLICY_TYPE:-groot}"
exec "$SCRIPT_DIR/train_lerobot.sh" "$@"
