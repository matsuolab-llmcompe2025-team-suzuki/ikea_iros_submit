# Run Instructions — IKEA IROS submission (Issue #116)

運営が各コンテナをどう起動する想定かを明示する（Participant Guide §9 + 2026-08 訂正メール）。

## 前提
- 2 コンテナは **別々にビルド**（GPU 世代が別: Thor sm_110 / Orin sm_87、1 image は両機で動かない）。
- 両 image とも **linux/arm64 (aarch64)** で build（x86 host からは buildx + QEMU で cross-build）。
- `boundary/` は**無改変**（ローカル改変はテストを通して実機で落ちる）。
- weights は image に焼き込まない（`manifest.yaml` の `weights_uri` から取得）。
- base image（運営訂正で確定）:
  - Thor: `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`（標準 NGC CUDA、l4t ではない。代替 `nvcr.io/nvidia/pytorch:25.08-py3`）
  - Orin: `nvcr.io/nvidia/l4t-jetpack:r35.3.1`（**JetPack 5.1.1**、device L4T に完全一致必須。client は推論しないので `l4t-base:r35.3.1` でも可）

## NGC login（nvcr.io は public でも必須）
```bash
# ngc.nvidia.com で無料アカウント作成 → API key 発行
docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>
```

## ビルド（arm64）
```bash
cd <repo-root>

# Thor (policy server)
docker buildx build --platform linux/arm64 -f docker/Dockerfile.thor \
  -t <registry>/ramen-thor:<tag> --push .

# Orin (policy client)
docker buildx build --platform linux/arm64 -f docker/Dockerfile.orin \
  -t <registry>/ramen-orin:<tag> --push .
```
push で `@sha256:…` digest が生成される（chicken-and-egg 解消）。digest を `manifest.yaml`
の各 `images.*.digest` に記入。**image は private Docker Hub / GHCR に push し、運営にアクセス付与**。

## 起動（運営が実行する想定）
Thor `192.168.100.1` / Orin `192.168.100.2`、両者は ethernet 直結。

```bash
# on the Thor (policy server)
docker run --rm --runtime nvidia --network host \
  -e HF_TOKEN=<token-if-gated> \
  <registry>/ramen-thor@sha256:<digest>
# → components/server.py --lane decoupled --host 0.0.0.0 --port 8765

# on the Orin (policy client)
docker run --rm --runtime nvidia --network host \
  <registry>/ramen-orin@sha256:<digest>
# → components/client.py --lane decoupled --thor 192.168.100.1 --orin 192.168.100.2
```
- `--network host`: boundary の ZeroMQ 3 endpoint（cameras:5555 / state:5557 / actions:5556、Orin 上）と Thor↔Orin WebSocket:8765 のため。
- `--runtime nvidia`: GPU アクセス。Orin は CUDA/driver userspace が host mount。

## 提出時に添えるもの（運営チェックリスト、2026-08 訂正）
1. Git repo link（無改変 `boundary/` + 各コンテナの Dockerfile）
2. 両 image の **registry digest**（`:latest` ではなく `@sha256:…`）
3. 各 image が load する **exact repo:tag**
4. 両方 **linux/arm64 (aarch64)** build である旨の確認
5. Model weights の HF link（image に焼き込まない）
6. `manifest.yaml`（lane / image refs / base image / entrypoint / Thor↔Orin port / weights URI / Thor peak GPU mem）
7. `python conformance.py --lane decoupled` の PASS ターミナル出力
8. 各コンテナの exact `docker run` コマンド（`--runtime nvidia` / mount / env 含む）
9. bench slot 中に連絡が取れる contact 1名

## onboarding での確認（実機不要）
```bash
python3 conformance.py --lane decoupled   # PASS を確認してから提出
```
現状は同梱の **hold-still Policy** で PASS 済み。RAMEN-Ori（#115）を載せる際は
`Policy.act()` を差し替え、19D joint→(T,25) task-space adapter を通す。
