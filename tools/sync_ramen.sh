#!/usr/bin/env bash
#
# 本体 repo (iros_2026_ramen) の推論コードを、この repo の ramen/ へコピーする。
#
# # なぜ要るか
#
# 推論コードの正本は本体。ramen/ は本体と**同じ相対パス**の写しで、手で直さない。
# 手で直すと、次のコピーで黙って消えるうえ、本体のどの commit と同じなのかが分からなくなる。
# 直したいときは本体を直して push し、この script でコピーし直す。
#
# # コピーするもの (本体と同じ相対パス。tests/ は除く)
#
#   pixi.toml pixi.lock scripts/activate_runtime.sh   runtime 環境
#   inference/__init__.py inference/desktop/            推論の本体・設定・参照画像・環境
#   model/__init__.py model/ramen_ori/                  RAMEN-Ori (Hydra が部品を名前で読むので丸ごと)
#   model/subtask_policy_training/                      GR00T の補助 module と pick の worker script
#   G1 の URDF 1 file                                   FK と重力補償が repo root 相対で読む (meshes は不要)
#
# # 止める条件
#
#   - 指定した commit が本体の origin に push されていない (image の中身を GitHub で辿れるように)
#   - 本体の inference/desktop/boundary/*.py と、この repo の boundary/*.py (運営の最新。
#     tools/update_organizer.sh で取り込む) が違う。運営が境界を変えたので、本体を直してから
#
# # 使い方
#
#   ./tools/sync_ramen.sh                    # 本体で今 checkout している commit
#   ./tools/sync_ramen.sh <commit|branch>    # 指定
#   IROS_RAMEN_REPO=~/work/iros/iros_2026_ramen ./tools/sync_ramen.sh
#
# コピー元の commit は ramen/RAMEN_SOURCE.txt に残る。コピーしたら差分を読んでから commit する。
# コピー漏れは image の build (docker/Dockerfile.thor の import 確認) で止まる。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_REPO="${IROS_RAMEN_REPO:-$(cd "${SUBMIT_ROOT}/.." && pwd)/iros_2026_ramen}"
REF="${1:-HEAD}"
DEST="${SUBMIT_ROOT}/ramen"

PATHS=(
  pixi.toml
  pixi.lock
  scripts/activate_runtime.sh
  inference/__init__.py
  inference/desktop
  model/__init__.py
  model/ramen_ori
  model/subtask_policy_training
  inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/g1_29dof_mode_15_with_dex1_1.urdf
)

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# --- コピー元の commit ---------------------------------------------------------

COMMIT="$(git -C "${SOURCE_REPO}" rev-parse --verify "${REF}^{commit}")"
SUBJECT="$(git -C "${SOURCE_REPO}" log -1 --format=%s "${COMMIT}")"
REMOTE_URL="$(git -C "${SOURCE_REPO}" remote get-url origin)"

echo "[ramen] source : ${SOURCE_REPO} (${REMOTE_URL})"
echo "[ramen] commit : ${COMMIT} ${SUBJECT}"

git -C "${SOURCE_REPO}" fetch --quiet origin
if [[ -z "$(git -C "${SOURCE_REPO}" branch -r --contains "${COMMIT}")" ]]; then
  echo "error: ${COMMIT} は本体の origin に push されていない。push してからコピーする" >&2
  exit 1
fi

# --- 取り出す (その commit の中身だけ。手元の未 commit の変更は入らない) --------

mkdir -p "${WORK}/src"
git -C "${SOURCE_REPO}" archive --format=tar "${COMMIT}" -- "${PATHS[@]}" |
  tar -x -C "${WORK}/src"
find "${WORK}/src" -type d -name tests -prune -exec rm -rf {} +

# --- 境界のコードが運営の最新と同じか ------------------------------------------

if ! diff -rq -x '__pycache__' -x '*.md' \
  "${WORK}/src/inference/desktop/boundary" "${SUBMIT_ROOT}/boundary"; then
  echo "error: 本体の inference/desktop/boundary と運営の boundary/ が違う。" >&2
  echo "       運営が境界を変えたので、本体を運営の最新に合わせて push してからコピーする" >&2
  exit 1
fi

# --- 出所の記録と置き換え ------------------------------------------------------

# 時刻は書かない (同じ commit をコピーし直したら差分が出ないように)
cat >"${WORK}/src/RAMEN_SOURCE.txt" <<EOF
# tools/sync_ramen.sh が生成。手で編集しない。ramen/ の中も手で直さない (本体を直してコピーし直す)。
repo: ${REMOTE_URL}
commit: ${COMMIT}
subject: ${SUBJECT}
EOF

# --delete で本体で消えたファイルはこちらからも消す。手元の pixi 環境の実体と
# image の中で clone する SDK は build context に入れないもの (.dockerignore) なので消さない
mkdir -p "${DEST}"
rsync -a --delete --exclude '.pixi/' --exclude 'third_party/' --exclude '__pycache__/' \
  "${WORK}/src/" "${DEST}/"

echo "[ramen] copied $(find "${DEST}" -type f -not -path '*/.pixi/*' -not -path '*/__pycache__/*' | wc -l | tr -d ' ') files into ramen/"
