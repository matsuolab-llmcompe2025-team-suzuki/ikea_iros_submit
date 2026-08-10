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
