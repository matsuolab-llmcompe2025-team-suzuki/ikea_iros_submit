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
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e HF_TOKEN=<token-if-gated> \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  <registry>/ramen-thor@sha256:<digest>
# → components/server.py --lane decoupled --host 0.0.0.0 --port 8765

# on the Orin (policy client)
docker run --rm --runtime nvidia --network host \
  <registry>/ramen-orin@sha256:<digest>
# → components/client.py --lane decoupled --thor 192.168.100.1 --orin 192.168.100.2
```
- `--network host`: boundary の ZeroMQ 3 endpoint（cameras:5555 / state:5557 / actions:5556、Orin 上）と Thor↔Orin WebSocket:8765 のため。
- `--runtime nvidia`: GPU アクセス。Orin は CUDA/driver userspace が host mount。
- **`-e NVIDIA_DISABLE_REQUIRE=1`(Thor のみ、必須)**: base の `cuda:13.0.0-devel-ubuntu24.04` は driver-compat gate を焼き込んでおり、運営 Thor(driver 595.78)では `--runtime nvidia` だけだと GPU が全く渡らない(`nvidia-smi` が container 内で失敗)。このフラグで解消(運営 onboarding Finding 1 で確認済)。無いと RAMEN-Ori が CPU-only load or crash する。
- **`-v ~/.cache/huggingface:/root/.cache/huggingface`(Thor、強く推奨)**: weights は image に焼かず runtime に `huggingface_hub.snapshot_download` で取る設計のため、mount が無いと**毎回空キャッシュから 8.90 GB を落とし直す**。準備時間は 1 スロット 20 分しかない。事前に host 側へ pull しておけば起動が即時になる。

## 自前経路（same image、大会経路とは別プロセス）
グリッパは `(T,25)` の `[0:2]`/`[2:4]` を運営 adapter が relay する大会経路でしか動かないが、
歩行（Stage 0）と腕は `rt/arm_sdk` 直の自前経路の方が確実（運営 IK を通らないので EE frame の
不確定性を受けない）。同じ image から起動できる:

```bash
docker run --rm --runtime nvidia --network host \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e HF_TOKEN=<token-if-gated> \
  -e PYTHONPATH=/app/components/ramen/vendor/desktop \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  <registry>/ramen-thor@sha256:<digest> \
  python3 -m inference.desktop.entrypoint \
    --head-source zmq --synthetic-hand-state --action-sink boundary
```
- `--head-source zmq`: 会場は ROS2 カメラを publish しない（boundary の `:5555` から取る）。
- `--synthetic-hand-state`: 会場は hand state を publish しない（Dex1-1、`boundary/states.py` に「usually absent, synthesize whatever your model expects」と明記）。
- `--action-sink boundary`: `(T,25)` を運営 adapter へ出す。外すと `rt/arm_sdk` へ直接出す（DDS 経路、CycloneDDS + `unitree_sdk2py` を image に同梱済み）。

## pick の hybrid（任意。既定は GR00T のまま）

`pick_table_leg` は既定で GR00T expert（`groot_pick_legs_v2`）で走る。**何もしなければ
これまでどおり**。VLM で区間境界を判定する hybrid（本体 repo Issue #148）に切り替えたい
場合だけ、VLM サーバを 1 つ足して env を渡す。

```bash
# Thor 上、policy server より先に起動しておく（8B の load に数分かかる）
docker run -d --name ramen-vlm --runtime nvidia --network host \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai@sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a \
  Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 4096 \
  --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":5,"video":0}' \
  --gpu-memory-utilization 0.20

curl -fsS http://127.0.0.1:8000/health    # これが通ってから policy server を起動する

# policy server 側に足す env
#   -e RAMEN_PICK_HYBRID=1
```
- digest は `vllm/vllm-openai:v0.29.0` の **linux/arm64**。`TORCH_CUDA_ARCH_LIST` に
  `11.0`（sm_110 = Thor）を含み、CUDA 13.0.2 / `NVARCH=sbsa` でこちらの thor image と同系統。
- `--gpu-memory-utilization 0.20`: Thor は 128 GB unified なので約 26 GB。`--max-model-len 4096`
  かつ `--max-num-seqs 1` なので KV cache はごく小さく、weights（bf16 8B ≒ 16 GB）が主。
  GR00T の常駐 2 つ（約 12 GiB）+ YOLO と同居させる前提の値。**開発機（RTX 5090）用の
  `run_local_vlm_server.sh` は 0.64 なので、そのまま Thor に持ち込まないこと。**
- `--host 127.0.0.1`: このサーバは同じ Thor の policy server からしか呼ばない。外に出さない。
- **事前に HF cache へ pull しておくこと。** 準備は 1 スロット 20 分しかなく、
  Qwen3-VL-8B を空キャッシュから落とすと間に合わない。
- 別ホストに立てた場合は `-e RAMEN_PICK_VLM_ENDPOINT=http://<host>:8000/v1/chat/completions`。
- **VLM に繋がらないと policy server は起動時に落ちる**（endpoint を skill を組む前に
  probe している）。`pick_table_leg` に入ってから気付く形にすると、会場では
  「掴まない」としか見えないため。hybrid をやめるなら `RAMEN_PICK_HYBRID` を外すだけでよい。

## 会場で image を焼き直さずに変えられるもの（Thor、policy server の env）

| env | 既定 | 何が変わるか |
|---|---|---|
| `RAMEN_POLICY` | `groot_pick_real` | `groot_orchestrator`（推奨、全 skill 自動遷移）/ `groot_53d_real` |
| `RAMEN_VARIANT_<SKILL>` | policy_config.yaml | expert の差し替え（例 `RAMEN_VARIANT_ROTATE_TABLE_BASE=rotate_table_base_ramen_ori_141_c32`） |
| `RAMEN_GPU_MODELS` | `2` | GPU に載せる model 数（今 + 次）。苦しければ `1` |
| `RAMEN_PICK_HYBRID` | 未設定（GR00T） | `1` で pick を VLM/VLA/MP hybrid に |
| `RAMEN_ON_TIMEOUT` | `advance` | `stop` で「時間切れなら止める」に。既定は YOLO が外しても先へ進む |
| `RAMEN_ORCH_LOG` | 未設定 | `(T,25)` を JSONL に残す |

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
