# 会場前の確認（Vast.ai の GB10）— Team RAMEN

提出 image を会場に持っていく前に、**Thor に近い機械で image そのものを起動して**、ネット無しで
全 stage が立ち上がるかを確かめる手順。image を焼き直したら毎回やる（30 分・$0.3〜0.5 ほど）。

会場の手順は `INSTRUCTIONS.md`、重みは `WEIGHTS.md`。

## 1. なぜ GB10 か

- 提出 image は **arm64**（Thor 用）。RunPod の GPU は全部 x86 なので、image が動かない（2026-09-24 に確認）。
- Vast.ai の **GB10**（NVIDIA DGX Spark）は arm64・Blackwell・CPU と GPU で約 122 GB を共有するメモリで、
  Thor（arm64・Blackwell・128 GB 共有）に近い。image の土台は Jetson 専用ではない汎用の arm64 CUDA 13
  （`nvidia/cuda:13.0.3-base-ubuntu24.04`）なので、**GHCR から pull してそのまま起動できる**。

## 2. 確かめること / 確かめられないこと

| 確かめること | 確かめられないこと（09-27 に Thor で） |
|---|---|
| image の起動口・4 環境（runtime / desktop / vlm / pick）が GPU を使えるか | sm_110（Thor）専用の kernel（GB10 は sm_121） |
| 重みの事前取得と、ネット無しの `--check` | 実機・実カメラ・運営の WBC / adapter |
| Stage 0〜5 の起動（VLM・YOLO・全 model の読み込み）が**ネット無しで**通るか | Thor の disk の速さ（重みの読み込み秒は変わる） |
| **外へ一度も接続しないか**（strace で全 process の `connect()`） | |
| `--gpu-models all` と VLM を合わせた GPU の使用量・共有メモリの余裕 | |
| conformance | |

`--actuate` を付けないので、ロボットへの指令は 1 通も出ない（起動確認だけで終わる）。

## 3. 秘密の値

| 値 | 使う所 | 渡し方 | 終わったら |
|---|---|---|---|
| Vast の API key | instance の検索・作成・削除 | **個人の account** の key（`~/.config/vastai/vast_api_key` はチーム「RAMEN」account の key。合意なしに使わない） | — |
| GitHub の PAT | Vast が GHCR から image を pull する | **classic**、scope は `read:packages` だけ、期限 1 日（fine-grained は GHCR が受け付けない）。instance 作成の `image_login` にだけ入れる | pull が終わったら revoke |
| HF の token | 重みの取得 | SSH の標準入力で container の `~/.cache/huggingface/token` に書く。**instance の env や API の body に入れない** | 重みを取ったらすぐ消す |

⚠️ 値を送る前に、それが何か（`ghp_` / `hf_` / 64 桁の 16 進）を確かめる。2026-09-25 に、クリップボードの
HF token を Vast の key だと思って Vast の API に送ってしまった（その token は作り直す）。

## 4. 手順

### 4-1 image を GHCR に上げる（試験用 tag）

`.github/ci-build-request.env` を `TAG=gb10-test-<本体の commit>`・`PUSH=true` にして push する（CI が
arm64 で焼いて push、20〜30 分）。本番の tag とは分ける。確認が済んだら tag を消す。

### 4-2 GB10 を借りる

```bash
VAST_KEY=<個人 account の key>
# 借りられる GB10（検証済み・信頼度 0.95 以上・disk 300 GB 以上・回線が速いものを選ぶ）
curl -s -G -H "Authorization: Bearer $VAST_KEY" https://console.vast.ai/api/v0/bundles/ \
  --data-urlencode 'q={"cpu_arch":{"eq":"arm64"},"rentable":{"eq":true},"gpu_name":{"eq":"GB10"},"disk_space":{"gte":300}}'

# 作る（create.json は権限 600 で作り、作ったらすぐ消す。PAT が入っている）
#   {"client_id":"me","image":"ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor:gb10-test-<commit>",
#    "image_login":"-u <GitHub user> -p <PAT> ghcr.io","env":{},"disk":250,
#    "runtype":"ssh_direc ssh_proxy","label":"ramen-gb10-check","cancel_unavail":true}
curl -s -X PUT -H "Authorization: Bearer $VAST_KEY" -H "Content-Type: application/json" \
  --data @create.json https://console.vast.ai/api/v0/asks/<offer id>/

# 起動待ち（image の pull に 10 分ほど）。running になったら SSH の宛先を読む
curl -s -H "Authorization: Bearer $VAST_KEY" https://console.vast.ai/api/v0/instances/<instance id>/
#   actual_status=running / public_ipaddr / ports["22/tcp"][0].HostPort
```

`running` になったら PAT は要らない（revoke してよい）。SSH の鍵は、Vast の account に登録した公開鍵の対になる秘密鍵。

### 4-3 道具を置き、環境の GPU を確かめる

bash で実行する（zsh は `$SSH` を単語に分けないので動かない）。

```bash
SSH="ssh -i <秘密鍵> -p <port> root@<ip>"
scp -i <秘密鍵> -P <port> tools/gb10/* root@<ip>:/root/
$SSH 'apt-get update -qq && apt-get install -y -qq strace'   # 使い捨ての container に OS の道具を入れるだけ
$SSH 'bash /root/check_envs.sh'
```

### 4-4 重みを取り、token を消し、ネット無しで確かめる

```bash
$SSH 'umask 077; mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token' <<< "$HF_TOKEN"
$SSH 'cd /app/ramen && HF_HUB_OFFLINE=0 pixi run --as-is -e runtime python /app/tools/prefetch_weights.py'
$SSH 'rm -f ~/.cache/huggingface/token ~/.cache/huggingface/stored_tokens'
$SSH 'cd /app/ramen && pixi run --as-is -e runtime python /app/tools/prefetch_weights.py --check'
```

### 4-5 各 stage を起動する

container では `unshare`（ネットを断った空間）が許されないので、strace で全 process の `connect()` を記録して
外向きが 0 件かを見る。1 本ずつ裏で流す（SSH は `-n` と `setsid nohup … < /dev/null &` で切り離す）。

```bash
$SSH -n 'setsid nohup bash -c "for s in 0 1 2 5; do /root/run_stage.sh \$s stage\$s; done; \
  NOSTRACE=1 /root/run_stage.sh 1 stage1_plain" > /dev/null 2>&1 < /dev/null &'
$SSH 'python3 /root/summarize.py /root/runs/stage*'     # 終わったら。exit 0 = 合格
$SSH 'cd /app/ramen && pixi run --as-is -e runtime python /app/conformance.py --lane decoupled'
```

strace の下は起動が遅く出る（VLM で 1.3 倍ほど）。起動秒は `stage1_plain`（strace 無し）で見る。

### 4-6 片付け

```bash
curl -s -X DELETE -H "Authorization: Bearer $VAST_KEY" https://console.vast.ai/api/v0/instances/<instance id>/
```

試験用 tag を GHCR から消し、PAT を revoke する。

## 5. 合格の基準

- `summarize.py` が exit 0: 全 stage が `rc=0`（`[preflight] … validation passed; NO command sent`）、**外向き connect 0 件**
- `prefetch_weights.py --check` が `all present`、conformance が `PASS`
- GPU の使用量が基準値から大きく増えていない（増えたら model か設定の変更を疑う）

## 6. 基準値（2026-09-25、image `gb10-test-d029549` に下の 2 つの直しを当てて計測）

| 項目 | 値 |
|---|---|
| 4 環境の GPU | runtime torch 2.12.1 / desktop 2.11.0 / vlm 2.13.0 / pick 2.11.0、全部 cu130、`cap (12, 1)`・integrated |
| 重みの取得 | 10 個 約 86 GB を 160 秒（回線 4.8 Gbps の host） |
| Stage 1（strace 無し） | 全体 267 秒。VLM の起動 196 秒（うち重み 16.3 GiB の読み込み 125 秒）、慣らし 1.1 秒、`vlm_latency` 0.70 秒、model 3 つ 約 70 秒 |
| Stage 5 / Stage 0 | 48 秒 / 12 秒（strace 下） |
| GPU の使用量の最大 | Stage 1: 28.3 GB（1 つずつ読む場合）。**`--gpu-models all` の Stage 2: 41.9 GB、MemAvailable は 48 GB 残る** |
| 空きの判断 | cudaMemGetInfo は page cache を使用中と数えて空き 10.7 GB と出る。MemAvailable（74.6 GB）で判断しているので読み込める |
| 外向き connect | 0 件（VLM・カメラ・状態の loopback と Unix socket だけ） |

この確認で見つけて直したもの:

1. **GR00T 53D（insert・締め付け・flip）がネット無しで読めなかった**。worker の環境の huggingface_hub 1.28 は、
   commit hash 指定でも cache に file 一覧の記録が無いとネットへ取りに行く。事前取得は runtime 環境
   （1.20.1）で行うので記録が無い。→ 本体 `5fac481`（オフラインなら `local_files_only=True`）。
   image の中の 4 環境で huggingface_hub の版が違う（1.20.1 / 1.28.0 / 1.32.0 / 1.22.0）ことに注意。
2. **YOLO（ultralytics 8.4.80）が外に出ていた**。import 時に DNS でネットの有無を調べ、推論の開始時に
   Google Analytics へ利用統計を送る。→ image の ENV に `YOLO_OFFLINE=true`（submit `6c0c6e6`）。

## 7. 費用

GB10 は 1 時間 $0.3〜0.7（disk 250 GB の保存料金込みで $0.4 前後）。1 回の確認は image の pull 10 分・
重み 3 分・stage の起動 20〜30 分で、合わせて $0.5 ほど。2026-09-25 の初回は $0.90（不具合の調査を含む）。
