#!/usr/bin/env bash
#
# 本体 repo (iros_2026_ramen) の推論コードを、この repo の
# components/ramen/vendor/desktop/ へ丸ごと同期する。
#
# # なぜ要るか
#
# 提出 image は docker/Dockerfile.thor.groot の `COPY . ./` でこの repo を丸ごと焼く。
# 推論コードの本体は iros_2026_ramen 側にあるので、vendor が古いとその古いコードが
# そのまま image に入る。2026-09-20 時点で約 3 週間ぶんドリフトしていた
# (20 files が古い / 38 files が欠落、ramen_ori.py は 914 行 → 1904 行)。
#
# 手でコピーすると同じことが必ず再発するので、同期は必ずこの script 経由で行う。
#
# # 使い方
#
#   ./tools/sync_vendor_desktop.sh
#   IROS_RAMEN_REPO=~/work/iros/iros_2026_ramen ./tools/sync_vendor_desktop.sh
#
# 同期後に tools/vendor_patches.py が走り、container 用の自作パッチを再適用する
# (上流がアンカーを変えていたら失敗して止まる = パッチが黙って消えない)。
#
# 同期したら必ず検証すること:
#   python3 conformance.py --lane decoupled
#   python3 -m pytest components/ramen/tests -q

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_REPO="${IROS_RAMEN_REPO:-$(cd "${SUBMIT_ROOT}/.." && pwd)/iros_2026_ramen}"

VENDOR_DESKTOP="${SUBMIT_ROOT}/components/ramen/vendor/desktop"

# G1 URDF は perception/g1_urdf_fk.py と lower_policy/gravity_compensation.py が
# **repo root 相対** (Path(__file__).parents[3] / parents[4]) で探す。vendor tree は
# その repo root の役目を兼ねるので、URDF も同じ相対位置に置かないと
# FileNotFoundError になる (置き忘れると 53D / orchestrator が全部落ちる)。
URDF_RELATIVE="inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/g1_29dof_mode_15_with_dex1_1.urdf"

# 本体 repo に無く vendor にだけ要る空の package marker。
# 本体は namespace package (PEP 420) で動くが、vendor は sys.path 直挿しで
# import されるので明示 package にしておく。rsync --delete の後に置き直す。
VENDOR_ONLY_INIT=(
  "inference/desktop/__init__.py"
  "inference/desktop/lower_policy/configs/__init__.py"
  "inference/desktop/perception/configs/__init__.py"
  "inference/desktop/skill_planner/configs/__init__.py"
)

# tests は image に不要。pixi 系は container に pixi が無いので入れても誤解を招くだけ。
RSYNC_EXCLUDES=(
  --exclude "tests/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude ".pytest_cache/"
  --exclude "pixi.toml"
  --exclude "pixi.lock"
)

# --- 事前確認 ---------------------------------------------------------------

if [[ ! -d "${SOURCE_REPO}/inference/desktop" ]]; then
  echo "error: 本体 repo が見つからない: ${SOURCE_REPO}" >&2
  echo "       IROS_RAMEN_REPO=<path> で指定すること。" >&2
  exit 1
fi

if [[ ! -f "${SOURCE_REPO}/${URDF_RELATIVE}" ]]; then
  echo "error: URDF が見つからない: ${SOURCE_REPO}/${URDF_RELATIVE}" >&2
  exit 1
fi

SOURCE_COMMIT="$(git -C "${SOURCE_REPO}" rev-parse HEAD)"
SOURCE_BRANCH="$(git -C "${SOURCE_REPO}" rev-parse --abbrev-ref HEAD)"
if [[ -n "$(git -C "${SOURCE_REPO}" status --porcelain)" ]]; then
  SOURCE_DIRTY="yes  ⚠️ 未コミットの変更を含む"
else
  SOURCE_DIRTY="no"
fi

echo "[sync] source : ${SOURCE_REPO}"
echo "[sync] commit : ${SOURCE_COMMIT} (${SOURCE_BRANCH}, dirty=${SOURCE_DIRTY})"
echo "[sync] target : ${VENDOR_DESKTOP}"

# --- inference/desktop -------------------------------------------------------

# --delete で上流から消えた file を vendor からも落とす (腐った残骸を残さない)。
mkdir -p "${VENDOR_DESKTOP}/inference/desktop"
rsync -a --delete "${RSYNC_EXCLUDES[@]}" \
  "${SOURCE_REPO}/inference/desktop/" \
  "${VENDOR_DESKTOP}/inference/desktop/"

# --- repo root 相対で参照される資産 -------------------------------------------

mkdir -p "${VENDOR_DESKTOP}/$(dirname "${URDF_RELATIVE}")"
cp "${SOURCE_REPO}/${URDF_RELATIVE}" "${VENDOR_DESKTOP}/${URDF_RELATIVE}"

# --- model/subtask_policy_training (GR00T の共有ヘルパ) ------------------------

rsync -a "${RSYNC_EXCLUDES[@]}" \
  "${SOURCE_REPO}/model/subtask_policy_training/gr00t/" \
  "${VENDOR_DESKTOP}/model/subtask_policy_training/gr00t/"

# --- model/ramen_ori (RAMEN-Ori の nn.Module + Hydra config) -------------------
#
# `policies/ramen_ori.py` の `from_ckpt` が
#   from model.ramen_ori.model import RamenOriPolicy
#   from model.ramen_ori.vision_backbone import load_vision_backbone
# を import し、Hydra が `configs/` を読む。推論に lerobot は要らない
# (training env の pixi.toml が fork を使うだけで、推論側は torch +
# huggingface_hub + hydra + lingbot_vision のみ)。
#
# `rotate_table_base` を GR00T ではなく RAMEN-Ori で回す選択肢を残すために入れる。
rsync -a "${RSYNC_EXCLUDES[@]}" \
  "${SOURCE_REPO}/model/ramen_ori/" \
  "${VENDOR_DESKTOP}/model/ramen_ori/"

# --- package marker の復元 ----------------------------------------------------

for marker in "${VENDOR_ONLY_INIT[@]}"; do
  mkdir -p "${VENDOR_DESKTOP}/$(dirname "${marker}")"
  : >"${VENDOR_DESKTOP}/${marker}"
done
: >"${VENDOR_DESKTOP}/inference/__init__.py"
: >"${VENDOR_DESKTOP}/model/__init__.py"
: >"${VENDOR_DESKTOP}/model/subtask_policy_training/__init__.py"
: >"${VENDOR_DESKTOP}/model/subtask_policy_training/gr00t/__init__.py"
: >"${VENDOR_DESKTOP}/model/subtask_policy_training/gr00t/assets/__init__.py"
# ramen_ori は本家に __init__.py があるので置き直さない (rsync がそのまま運ぶ)。

# --- 自作パッチの再適用 -------------------------------------------------------

python3 "${SCRIPT_DIR}/vendor_patches.py" "${VENDOR_DESKTOP}"

# --- 出所の記録 ---------------------------------------------------------------

cat >"${VENDOR_DESKTOP}/VENDOR_SOURCE.txt" <<EOF
# tools/sync_vendor_desktop.sh が生成。手で編集しない。
source_repo   : iros_2026_ramen
source_commit : ${SOURCE_COMMIT}
source_branch : ${SOURCE_BRANCH}
source_dirty  : ${SOURCE_DIRTY}
synced_at     : $(date -u +%Y-%m-%dT%H:%M:%SZ)
excluded      : tests/ __pycache__/ *.pyc .pytest_cache/ pixi.toml pixi.lock
patches       : tools/vendor_patches.py
EOF

echo "[sync] done"
echo "[sync] 次に検証すること:"
echo "         python3 conformance.py --lane decoupled"
echo "         python3 -m pytest components/ramen/tests -q"
