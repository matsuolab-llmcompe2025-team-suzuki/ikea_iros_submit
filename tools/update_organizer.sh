#!/usr/bin/env bash
#
# 運営の提出 template (iacevaltest/ikea_iros_submit) から、運営が持つパスを取り込む。
#
# # なぜ要るか
#
# 運営は template を予告なく更新する。運営のパスは**手で触らず、この script で
# 上書きするだけ**にする。手で直すと、次に取り込んだときに黙って消えるうえ、
# 運営の最新とどこがずれているのか分からなくなる。
#
# # 取り込むもの
#
# 運営 README の Layout 表で owner = organizer のもの:
#   boundary/  mocks/  conformance.py  requirements.txt
#
# `components/` は運営 README が「OURS」と定めている場所なので取り込まない
# (`conformance.py` が `components/server.py` と `components/client.py` を
# このパスで起動するので、名前だけは合わせて置いておく)。
#
# # 使い方
#
#   ./tools/update_organizer.sh            # 運営 main の最新
#   ./tools/update_organizer.sh 2ae4eeb    # commit / tag / branch を指定
#
# 取り込んだ commit は ORGANIZER_SOURCE.txt に残る。取り込んだら差分を読んでから
# commit すること (運営が契約を変えていないか)。

set -euo pipefail

ORGANIZER_REPO_URL="${ORGANIZER_REPO_URL:-https://github.com/iacevaltest/ikea_iros_submit.git}"
REF="${1:-main}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ORGANIZER_DIRS=(boundary mocks)
ORGANIZER_FILES=(conformance.py requirements.txt)

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# --- 運営 template を取ってくる ------------------------------------------------

git clone --quiet "${ORGANIZER_REPO_URL}" "${WORK}/organizer"
# commit / tag はそのまま、main 以外の branch 名は origin/ 付きで解決する
git -C "${WORK}/organizer" checkout --quiet --detach "${REF}" 2>/dev/null ||
  git -C "${WORK}/organizer" checkout --quiet --detach "origin/${REF}"

COMMIT="$(git -C "${WORK}/organizer" rev-parse HEAD)"
COMMITTED_AT="$(git -C "${WORK}/organizer" log -1 --format=%cI)"
SUBJECT="$(git -C "${WORK}/organizer" log -1 --format=%s)"

echo "[organizer] source : ${ORGANIZER_REPO_URL}"
echo "[organizer] commit : ${COMMIT} (${COMMITTED_AT}) ${SUBJECT}"

# 運営が構成を変えていたら、こちらの物を消す前に止める
for path in "${ORGANIZER_DIRS[@]}" "${ORGANIZER_FILES[@]}"; do
  if [[ ! -e "${WORK}/organizer/${path}" ]]; then
    echo "error: 運営 template に ${path} が無い。構成が変わったので手で確認すること" >&2
    exit 1
  fi
done

# --- 上書き --------------------------------------------------------------------

# --delete で、運営側で消えたファイルはこちらからも消す
for dir in "${ORGANIZER_DIRS[@]}"; do
  mkdir -p "${SUBMIT_ROOT}/${dir}"
  rsync -a --delete --exclude "__pycache__/" \
    "${WORK}/organizer/${dir}/" "${SUBMIT_ROOT}/${dir}/"
done
for file in "${ORGANIZER_FILES[@]}"; do
  cp "${WORK}/organizer/${file}" "${SUBMIT_ROOT}/${file}"
done

# --- 出所の記録 ----------------------------------------------------------------

# 取り込んだ時刻は書かない (同じ commit を取り込み直したら差分が出ないように)
cat >"${SUBMIT_ROOT}/ORGANIZER_SOURCE.txt" <<EOF
# tools/update_organizer.sh が生成。手で編集しない。
source_repo   : ${ORGANIZER_REPO_URL}
source_ref    : ${REF}
source_commit : ${COMMIT}
committed_at  : ${COMMITTED_AT}
subject       : ${SUBJECT}
paths         : ${ORGANIZER_DIRS[*]/%//} ${ORGANIZER_FILES[*]}
EOF

# --- 何が変わったか ------------------------------------------------------------

echo "[organizer] 変わったもの:"
git -C "${SUBMIT_ROOT}" status --short -- \
  "${ORGANIZER_DIRS[@]}" "${ORGANIZER_FILES[@]}" ORGANIZER_SOURCE.txt
