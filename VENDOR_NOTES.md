# Vendor notes — ikea_iros_submit (Issue #116)

## Provenance
- Upstream: `https://github.com/iacevaltest/ikea_iros_submit` (運営公式 template)
- Vendored commit: `7a4f071c8fef9ce66647f6527bc22d4a30d91d54` (2026-07-21)
- Vendored: `boundary/` `components/` `mocks/` `conformance.py` `requirements.txt` `README.md`

## 改変ポリシー
- **`boundary/` は無改変で追従する**（cameras/states/actions = 運営所有の契約）。
  upstream が更新されたら再取得して差分確認（契約は session 間で動きうる、`conformance.py`
  を毎提出前に回す）。
- 編集するのは `components/`（特に `server.py` の `Policy`）と、追加した
  `docker/` `manifest.yaml` `INSTRUCTIONS.md` のみ。

## upstream 追従 / 意図的な乖離（2026-08、再同期時 要注意）
- upstream 最新 = **`2ae4eeb`**（vendored 元 `7a4f071` から 2 commit 先）:
  - `9f770d2` stereo ego_view 追加（`boundary/cameras.py` の CAMERA_KEYS、README camera 表、`mock_orin.py --stereo-ego`）
  - `2ae4eeb` README の JetPack 訂正 + Base images 節 + NGC note
- **採用**: jetpack 訂正（docs）は **README へ手で反映済**（ハード表 Thor R39.2 / Orin R35.3.1、Base images、NGC）。
- **非採用**: stereo（`9f770d2`）。我々は **mono `ego_view` のみ** = `boundary/cameras.py`・`mocks/mock_orin.py` は
  **`7a4f071` のまま**、README camera 表も mono。stereo 採用は #115 のモデル判断（採用時は boundary 同期 +
  image 再ビルド/再 push = digest 変更が発生）。
- ⚠️ **`README.md` は first-person（RAMEN 視点）へ全面書き換え済**（upstream は second-person）。
  **plain re-vendor で README を上書きしないこと**。再同期は upstream 差分を cherry-pick する。
- ⚠️ **`components/transport.py` は自作パッチあり**（websockets の ping kwargs を `_WS_MAJOR>=14` で version-guard。
  実 Orin = Python3.8 = websockets 13.1 で client 即死するのを回避、cross-container e2e で確認）。
  **plain re-vendor で上書きしないこと**。
- ⚠️ **`components/client.py` は自作パッチあり**（`close()` の `ThreadPoolExecutor.shutdown(cancel_futures=True)` を
  `sys.version_info>=(3,9)` で version-guard。`cancel_futures` は Python3.9+ で、Orin=Python3.8 では exit path が
  TypeError で crash する upstream template bug。運営 onboarding **Finding 3** で実発火確認）。**plain re-vendor で上書きしないこと**。
- `components/server.py` は `7a4f071` のまま（`2ae4eeb` は stereo コメント追記のみ = 非採用）。

## 現状（onboarding）
- `python3 conformance.py --lane decoupled` = **PASS**（hold-still Policy、40 accepted / 0 rejected、default env の numpy/msgpack/pyzmq/opencv/websockets で実行可）。
- lane = **decoupled**（RAMEN-Ori は非 GR00T-SONIC → (T,25) task-space）。

## Base image（運営 2026-08 訂正で確定）
- Thor(server): `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`（標準 NGC CUDA、l4t ではない / JetPack 7.2 / CUDA 13 / sm_110）。代替 `nvcr.io/nvidia/pytorch:25.08-py3`。
- Orin(client): `nvcr.io/nvidia/l4t-jetpack:r35.3.1`（**JetPack 5.1.1**、6.x は誤り / CUDA 11.4 / sm_87 / Python3.8）。軽量代替 `l4t-base:r35.3.1`。
- 両 image **linux/arm64** build。nvcr.io は NGC login 必須。private registry へ push しアクセス付与。

## arm64 ビルド機構の検証（2026-08-05、済）
- x86 host で QEMU arm64 登録（`tonistiigi/binfmt --install arm64`）+ buildx docker-container builder で
  **arm64 image を build → コンテナ内で `conformance --lane decoupled` PASS**（40/0）を確認済。
- 検証は public base（`docker/Dockerfile.smoke-arm64` = python:3.11-slim、NGC 不要）で実施。
  → 提出 image への残差は「base を nvcr.io に差し替え + NGC login + private registry push」のみ。

## TODO（#116）
- NGC account + API key（`docker login nvcr.io -u '$oauthtoken' -p <key>`）で本番 base を pull。
- `docker/Dockerfile.{thor,orin}` を arm64 build（buildx+QEMU、機構は検証済）→ private Docker Hub/GHCR push
  → 運営にアクセス付与 → digest を manifest へ。
- weights_uri（RAMEN-Ori HF）/ peak GPU mem 実測。
- **action-space adapter**: RAMEN-Ori 19D upper-body joint → boundary (T,25) task-space
  （arm14→FK→EE pose / waist3→torso rpy / hand2→4 / navigate・base は 0 埋め、歩行は別 source）。
  design doc `docs/model/ramen_ori_vla_design.md` §8.6 に未反映。
  - ⚠️ **`scipy` を Thor image の deps に焼き込む**（FK step が要求。運営 onboarding **Finding 5**:
    OOJU/CuriosAI が dev host patch のみで clean run の import time に container fail した事例あり）。
    RAMEN-Ori 搭載時に `docker/Dockerfile.thor` の model deps へ `scipy` を追加すること。
- **運営 onboarding = PASS**（2026-08、`ikea_iros_submit@6a46c3d`、contract+plumbing 認証、two-box 0 reject）。
  残 gate: (1) `-e NVIDIA_DISABLE_REQUIRE=1`（Finding 1、対応済 INSTRUCTIONS/下記）、(2) GHCR access（Finding 2、public 化で解消）、
  (3) RAMEN-Ori 実搭載（real checkpoint、weighted stage 未検証）。Orin base r35.3.1 は初回正解で credit（Finding 4）。

## GR00T-pick 提出トラック（2026-08-30、`components/ramen/`）
明日の運営評価向けに、RAMEN-Ori とは別に **GR00T pick 単体**を decoupled lane へ載せる track。
- `components/ramen/policy.py`: `GrootPickTaskspacePolicy`（`RAMEN_POLICY=groot_pick_real` で
  server.py が選択）。backend = `GrootWorkerBackend`（`groot_worker.GrootPickWorker` で
  vendored worker を spawn、raw 38D → adapter で (T,25)）。`HoldPoseBackend` は GPU 無し fallback。
- `components/ramen/{taskspace_adapter,g1_urdf_fk}.py`: iros_2026_ramen から vendor した
  **pure-numpy** FK + adapter（arm14→FK→EE / hand2→4 / waist3→torso_rpy は param / navigate・base=0）。
  ⚠️ **この FK 経路は scipy 不要**（Shepperd matrix→quat、euler 非経由）= Finding 5 の scipy 罠は
  **GR00T-pick track には非該当**（scipy を焼く必要なし。RAMEN-Ori track のみ別途要判断）。
- `components/ramen/vendor/`: worker inference の最小 closure（`groot_pick_leg_contract` /
  `worker_protocol` / `real_groot_n17_worker.py`）を package path 保存で vendor。desktop policy
  stack 非依存。
- **container build**: `docker/Dockerfile.thor.groot`（RAMEN-Ori 用と別 image）。
  - base = `nvcr.io/nvidia/pytorch:25.08-py3`（arm64 variant が Grace-Blackwell 向け torch/CUDA13 同梱）。
    torch **sm_110** の現実解（repo に Thor install script 無し、標準 NGC pytorch を採る）。
  - `pip install lerobot[groot]==0.6.0`（torch 制約 >=2.7,<2.12 = NGC torch を保持）+ numpy 2.2.6 +
    submission reqs。inference は `[groot]` のみで足りる（dataset/training 不要 = torchcodec aarch64 回避）。
  - `ENV RAMEN_POLICY=groot_pick_real RAMEN_WORKER_PYTHON=python3`（同一 env で worker spawn）。
  - weights（ver2-lora, private）は runtime に HF 取得 → `docker run -e HF_TOKEN=...` 必須。
  - build: `docker buildx build --builder armbuilder --platform linux/arm64 \
    -f docker/Dockerfile.thor.groot -t <registry>/ramen-thor-groot:<tag> --push .`
  - ✅ **arm64 build 検証済（2026-08-31、QEMU+buildx、push 無し validate）**:
    NGC arm64 base pull OK、`pip install lerobot[groot]==0.6.0` DONE、
    `[build] torch 2.8.0a0+nv25.08 cuda 13.0 | numpy 2.2.6 | lerobot 0.6.0 | cv2/hf_hub import OK`。
    恐れた CUDA 拡張 compile 失敗は無し。numpy は最後に 2.2.6 固定（worker 一致）。
    pip の conflict 警告は NGC base 同梱 RAPIDS/numba/cupy/scipy の版ズレのみ = GR00T pick は
    未 import で無害。
  - ⚠️ **実行は未検証**（sm_110 GPU 必要、x86/QEMU では GPU 実行不可）: 実 Thor で
    `RAMEN_POLICY=groot_pick_real python3 components/server.py` の worker ready + conformance +
    (T,25) 出力を確認する。weights は `-e HF_TOKEN=...` で runtime 取得。
  - ✅ **GHCR push 済（2026-08-31）**: 既存 **`ikea-thor`** package の新 tag
    `ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor:20260831-groot-pick`
    index digest `sha256:207c8db9027325a50dca7ff47298aaed2ca0e3a2e9732456b5aadfa7f650c3a5`
    （arm64 platform manifest = `sha256:2c00de8edef8da824e92dd5aa24954daf3a12fafa09a5af96e21e1f6001ebe9c`）。
    → `manifest.groot.yaml`（RAMEN-Ori 用 manifest.yaml と別 track）に反映済。
    ※ 運営に伝達済の package 名 `ikea-thor` を維持（`:onboarding` tag は温存、read 権限は既存のまま）。
    ※ 初回誤って push した `ikea-thor-groot` package は堀江が別途削除予定。
- ⚠️ EE frame（pelvis/torso）は運営未確認（adapter に `ee_frame_transform` 穴あり）。
