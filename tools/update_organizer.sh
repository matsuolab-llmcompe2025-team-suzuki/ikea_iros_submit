#!/usr/bin/env bash
#
# 運営の repo から、運営が持つパスを取り込む。
#
# # なぜ要るか
#
# 運営は repo を予告なく更新する。運営のパスは**手で触らず、この script で
# 上書きするだけ**にする。手で直すと、次に取り込んだときに黙って消えるうえ、
# 運営の最新とどこがずれているのか分からなくなる。
#
# # 取り込むもの (運営 README の Layout 表で owner = organizer のもの)
#
#   boundary/                           ← interface package (iacevaltest/iros_g1_orin_package)
#   mocks/  conformance.py  requirements.txt ← 提出 template (iacevaltest/ikea_iros_submit)
#
# boundary/ (client の契約実装) は両方の repo にあるが、運営が更新するのは interface package の方
# (2026-09-25 の joint lane は template には入っていない)。本体の inference/desktop/boundary も
# interface package から vendor しているので、同じ所から取る (sync_ramen.sh は両者が違うと止まる)。
#
# `components/` は運営 README が「OURS」と定めている場所なので取り込まない
# (`conformance.py` が `components/server.py` と `components/client.py` を
# このパスで起動するので、名前だけは合わせて置いておく)。
#
# # 使い方
#
#   ./tools/update_organizer.sh                    # 両方とも main の最新
#   ./tools/update_organizer.sh 2ae4eeb 609f61d    # template と interface package の commit / tag / branch
#
# 取り込んだ commit は ORGANIZER_SOURCE.txt に残る。取り込んだら差分を読んでから
# commit すること (運営が契約を変えていないか)。

set -euo pipefail

TEMPLATE_REPO_URL="${TEMPLATE_REPO_URL:-https://github.com/iacevaltest/ikea_iros_submit.git}"
PACKAGE_REPO_URL="${PACKAGE_REPO_URL:-https://github.com/iacevaltest/iros_g1_orin_package.git}"
TEMPLATE_REF="${1:-main}"
PACKAGE_REF="${2:-main}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEMPLATE_DIRS=(mocks)
TEMPLATE_FILES=(conformance.py requirements.txt)
PACKAGE_DIRS=(boundary)

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# --- 運営の repo を取ってくる --------------------------------------------------

# fetch <名前> <url> <ref>: ${WORK}/<名前> に clone して ref を checkout する
fetch() {
  git clone --quiet "$2" "${WORK}/$1"
  # commit / tag はそのまま、main 以外の branch 名は origin/ 付きで解決する
  git -C "${WORK}/$1" checkout --quiet --detach "$3" 2>/dev/null ||
    git -C "${WORK}/$1" checkout --quiet --detach "origin/$3"
  echo "[organizer] $1: $2 $(git -C "${WORK}/$1" log -1 --format='%H (%cI) %s')"
}
fetch template "${TEMPLATE_REPO_URL}" "${TEMPLATE_REF}"
fetch package "${PACKAGE_REPO_URL}" "${PACKAGE_REF}"

# 運営が構成を変えていたら、こちらの物を消す前に止める
for path in "${TEMPLATE_DIRS[@]}" "${TEMPLATE_FILES[@]}"; do
  if [[ ! -e "${WORK}/template/${path}" ]]; then
    echo "error: 運営 template に ${path} が無い。構成が変わったので手で確認すること" >&2
    exit 1
  fi
done
for path in "${PACKAGE_DIRS[@]}"; do
  if [[ ! -e "${WORK}/package/${path}" ]]; then
    echo "error: 運営 interface package に ${path} が無い。構成が変わったので手で確認すること" >&2
    exit 1
  fi
done

# --- 上書き --------------------------------------------------------------------

# --delete で、運営側で消えたファイルはこちらからも消す
for dir in "${TEMPLATE_DIRS[@]}"; do
  mkdir -p "${SUBMIT_ROOT}/${dir}"
  rsync -a --delete --exclude "__pycache__/" \
    "${WORK}/template/${dir}/" "${SUBMIT_ROOT}/${dir}/"
done
for file in "${TEMPLATE_FILES[@]}"; do
  cp "${WORK}/template/${file}" "${SUBMIT_ROOT}/${file}"
done
for dir in "${PACKAGE_DIRS[@]}"; do
  mkdir -p "${SUBMIT_ROOT}/${dir}"
  rsync -a --delete --exclude "__pycache__/" \
    "${WORK}/package/${dir}/" "${SUBMIT_ROOT}/${dir}/"
done

# --- 出所の記録 ----------------------------------------------------------------

# describe <名前> <url> <ref> <paths...>: ORGANIZER_SOURCE.txt の 1 つ分
describe() {
  local name="$1" url="$2" ref="$3"
  shift 3
  cat <<EOF
[${name}]
source_repo   : ${url}
source_ref    : ${ref}
source_commit : $(git -C "${WORK}/${name}" rev-parse HEAD)
committed_at  : $(git -C "${WORK}/${name}" log -1 --format=%cI)
subject       : $(git -C "${WORK}/${name}" log -1 --format=%s)
paths         : $*
EOF
}

# 取り込んだ時刻は書かない (同じ commit を取り込み直したら差分が出ないように)
{
  echo "# tools/update_organizer.sh が生成。手で編集しない。"
  describe template "${TEMPLATE_REPO_URL}" "${TEMPLATE_REF}" \
    "${TEMPLATE_DIRS[@]/%//}" "${TEMPLATE_FILES[@]}"
  describe package "${PACKAGE_REPO_URL}" "${PACKAGE_REF}" "${PACKAGE_DIRS[@]/%//}"
} >"${SUBMIT_ROOT}/ORGANIZER_SOURCE.txt"

# --- 何が変わったか ------------------------------------------------------------

echo "[organizer] 変わったもの:"
git -C "${SUBMIT_ROOT}" status --short -- \
  "${TEMPLATE_DIRS[@]}" "${TEMPLATE_FILES[@]}" "${PACKAGE_DIRS[@]}" ORGANIZER_SOURCE.txt
