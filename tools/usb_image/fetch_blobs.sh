#!/usr/bin/env bash
#
# USB に入れる Thor の image を、GHCR から**展開せずに** blob（層・config・manifest）のまま落とす（WEIGHTS.md 3）。
# docker pull は落とした層を Docker の中に展開してから終わる（Mac の Docker で約 30 分）が、USB の tar には要らない。
# 環境の層（約 8.5 GB）は CI で作るたびに中身が変わるので、前の image があっても毎回丸ごと落とす。
#
#   GHCR_USER=<GitHub user> GHCR_PAT_FILE=<read:packages の PAT を入れた file> \
#     tools/usb_image/fetch_blobs.sh <tag> <out_dir>          # 例: gb10-test-1b34e82 ~/usb_image_1b34e82
#
# 小さい blob は JOBS 本ずつ同時に、1 GB を超える blob は Range で PARTS 区間に分けて同時に落とす。
# GHCR の blob の URL は約 7 分で切れるので、区間は「今ある大きさの続き」から新しい URL で落とし直す。
# どの blob も sha256 が digest と一致した物だけを残す。やり直すと、そろっている blob は飛ばす。
# 次は tools/usb_image/assemble_tar.py → verify_tar.py。

set -euo pipefail
TAG=$1; OUT=$2
REPO=${GHCR_REPO:-matsuolab-llmcompe2025-team-suzuki/ikea-thor}; PARTS=${PARTS:-16}; JOBS=${JOBS:-8}
: "${GHCR_USER:?GHCR_USER が要る}" "${GHCR_PAT_FILE:?GHCR_PAT_FILE が要る}"
mkdir -p "$OUT/blobs"

size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
token() {
  curl -sf -u "$GHCR_USER:$(cat "$GHCR_PAT_FILE")" "https://ghcr.io/token?scope=repository:$REPO:pull&service=ghcr.io" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])'
}
ACCEPT="application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"

T=$(token)
DIGEST=$(curl -sfI -H "Authorization: Bearer $T" -H "Accept: $ACCEPT" "https://ghcr.io/v2/$REPO/manifests/$TAG" \
  | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest" {print $2}')
[ -n "$DIGEST" ] || { echo "manifest が見つからない: $REPO:$TAG"; exit 1; }
HEX=${DIGEST#sha256:}
curl -sf -H "Authorization: Bearer $T" -H "Accept: $ACCEPT" "https://ghcr.io/v2/$REPO/manifests/$DIGEST" -o "$OUT/blobs/$HEX"
[ "$(sha "$OUT/blobs/$HEX")" = "$HEX" ] || { echo "manifest の digest が合わない"; exit 1; }
printf '%s %s %s\n' "$REPO" "$TAG" "$HEX" > "$OUT/image.txt"
echo "manifest $DIGEST"

# config と層の一覧 (同じ層が 2 回出ることがある)。1 GB を超える物は big
python3 - "$OUT/blobs/$HEX" "$OUT" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
if "layers" not in m:
    sys.exit(f"single-platform の manifest ではない (mediaType={m.get('mediaType')})")
blobs = {b["digest"][7:]: b["size"] for b in [m["config"]] + m["layers"]}
with open(f"{sys.argv[2]}/small.txt", "w") as small, open(f"{sys.argv[2]}/big.txt", "w") as big:
    for h, s in blobs.items():
        (big if s > 1 << 30 else small).write(f"{h} {s}\n")
print(f"{len(m['layers'])} layers, {len(blobs)} blobs, {sum(blobs.values()) / 1e9:.2f} GB")
PY

fetch_small() {
  local hex=$1 want=$2 f=$OUT/blobs/$1
  [ -f "$f" ] && [ "$(size "$f")" = "$want" ] && [ "$(sha "$f")" = "$hex" ] && return 0
  for _ in 1 2 3; do
    curl -sfL --retry 3 -H "Authorization: Bearer $(token)" "https://ghcr.io/v2/$REPO/blobs/sha256:$hex" -o "$f.part" \
      && [ "$(sha "$f.part")" = "$hex" ] && mv "$f.part" "$f" && { echo "ok   $hex"; return 0; }
  done
  echo "FAIL $hex"; return 1
}

fetch_big() {
  local hex=$1 want=$2 f=$OUT/blobs/$1 p=$OUT/parts_$1
  [ -f "$f" ] && [ "$(size "$f")" = "$want" ] && [ "$(sha "$f")" = "$hex" ] && { echo "have $hex"; return 0; }
  mkdir -p "$p"
  local ch=$(( (want + PARTS - 1) / PARTS )) round i a b part have left u
  for round in $(seq 1 30); do
    left=0
    u=$(curl -s -o /dev/null -w '%{redirect_url}' -H "Authorization: Bearer $(token)" "https://ghcr.io/v2/$REPO/blobs/sha256:$hex")
    for i in $(seq 0 $((PARTS - 1))); do
      a=$((i * ch)); b=$(( (i + 1) * ch - 1 )); [ $b -ge $want ] && b=$((want - 1))
      part=$p/$(printf %03d $i); have=0; [ -f "$part" ] && have=$(size "$part")
      [ $have -gt $((b - a + 1)) ] && { rm -f "$part"; have=0; }
      [ $have -eq $((b - a + 1)) ] && continue
      left=$((left + b - a + 1 - have))
      curl -sf -r $((a + have))-$b "$u" >> "$part" &     # 失敗しても次の round で続きから
    done
    echo "$hex: round $round, $((left / 1000000)) MB left"
    [ $left -eq 0 ] && break
    wait
  done
  cat "$p"/* > "$f.part"
  [ "$(size "$f.part")" = "$want" ] && [ "$(sha "$f.part")" = "$hex" ] || { echo "FAIL $hex"; return 1; }
  mv "$f.part" "$f"; rm -rf "$p"; echo "ok   $hex (big)"
}

export -f size sha token fetch_small; export REPO OUT GHCR_USER GHCR_PAT_FILE
xargs -P "$JOBS" -n 2 bash -c 'fetch_small "$0" "$1"' < "$OUT/small.txt"
while read -r hex want; do fetch_big "$hex" "$want"; done < "$OUT/big.txt"
echo "ALL BLOBS OK ($OUT)"
