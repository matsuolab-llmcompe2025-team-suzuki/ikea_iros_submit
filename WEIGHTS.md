# 重みの一覧と USB の持ち込み — Team RAMEN

会場は**実行時にネットに出ない**（`HF_HUB_OFFLINE=1`）。重みは Thor の host の HF cache を
読み取り専用で mount して読む（`INSTRUCTIONS.md` の Step 4）。1 つでも無いと、その stage は起動時に止まる。

- 一覧の正本は `tools/prefetch_weights.py`（`policy_config.yaml` の既定の model・YOLO と hybrid の VLM から
  組み立てる。model を差し替えたら script は追従し、この表と `manifest.yaml` は test で食い違いを止める）。
- 取るときも確かめるときも、**image の中で** script を動かす（runtime と同じ huggingface_hub で、cache の形が実行時と一致する）。

## 1. 一覧（2026-09-24、合計 約 86 GB）

| # | 使う所 | repo | revision | 取る範囲 | 大きさ |
|---|---|---|---|---|---|
| 1 | 台を回す（Stage 2〜4） | `Team-RAMEN/IROS2026_RAMEN_hara_ramen_ori_141_c32_state_dropout` | `b19c777` | `ckpt_step_100000.pt` だけ | 3.5 GB |
| 2 | 1 の画像 backbone | `robbyant/lingbot-vision-vit-base` | main | 全部 | 0.3 GB |
| 3 | pick（hybrid の区間 1、Stage 1〜4） | `Team-RAMEN/groot-n1.7-pick-legs-ver1` | `b63d9c4` | `checkpoint-40000/` の実行時の file（optimizer 13 GB は取らない） | 12.6 GB |
| 4 | insert（Stage 1〜4） | `Team-RAMEN/IROS2026_RAMEN_takada_insert_leg_optimal_gr00t_200k` | `0f0927b` | 全部 | 13.8 GB |
| 5 | 締め付け（Stage 1〜4） | `Team-RAMEN/IROS2026_RAMEN_takada_rotate_leg_to_tighten_optimal_gr00t_200k` | `51e306f` | 全部 | 13.8 GB |
| 6 | flip（Stage 5） | `Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_groot_n17_2_baseline_checkpoints` | `1a408d8` | `checkpoints/020000/pretrained_model/` だけ（無いと約 102 GB） | 12.6 GB |
| 7 | 4〜6 の base model | `nvidia/GR00T-N1.7-3B` | `2fc962b` | 全部 | 6.9 GB |
| 8 | GR00T の backbone（tokenizer・前処理） | `nvidia/Cosmos-Reason2-2B` | main | 全部。**gated** | 4.9 GB |
| 9 | YOLO（overlay） | `Team-RAMEN/IROS2026_RAMEN_Hara_yoloobb_upperpolicy` | `8221d0a` | `runs/m_lowaug_v11b/weights/best_20260818.pt` | 0.04 GB |
| 10 | VLM（hybrid pick の区間 1→2、Stage 1〜4） | `Qwen/Qwen3-VL-8B-Instruct` | main | 全部 | 17.5 GB |

- Stage 0（歩く）は学習済み model を使わない。`ramen_ori_default` は `--phase1-11-arm-only` 専用なので入れない。
- 7 の revision は 4〜6 の ckpt の `config.json` の `base_model_revision`（今は 3 つとも既定の `2fc962b`）。
- main で取るもの（2・8・10）は、実行時も main を引く（ネット無しで引くために `refs/main` も一緒に入る）。

## 2. 取り方（ネットのある所で）

token は `.env` の `HF_TOKEN` を使う（値はコマンドに書かない）。必要な権限は
**Team-RAMEN の private repo を読めること**と、**その account で 8（Cosmos、gated）の license を承認済み**のこと。
script は取り始める前に全 repo を読めるかを確かめ、読めなければ repo 名と理由を出して何も取らずに止まる。

```bash
set -a; . ./.env; set +a                 # HF_TOKEN を環境変数に（.env は編集・commit しない）
mkdir -p $RAMEN_HOST_DIR/hf_cache
docker run --rm -e HF_HUB_OFFLINE=0 -e HF_TOKEN \
  -v $RAMEN_HOST_DIR/hf_cache:/root/.cache/huggingface \
  <IMAGE> pixi run --as-is -e runtime python /app/tools/prefetch_weights.py
```

## 3. USB に入れる物（チェックリスト）

**exFAT**（4 GB を超える file があるので FAT32 は不可）。重み 約 86 GB + image（無圧縮）。
2026-09-24 に 250 GB の USB へ重みと `SHA256SUMS` を入れ、USB の上で `--check` が `all present` になることを確かめた。

- [ ] **重み**（約 86 GB）。USB の上に HF cache を**直接**作る（tar は要らない）
  ```bash
  HF_HOME=<USB>/hf_cache HF_HUB_OFFLINE=0 HF_HUB_DISABLE_SYMLINKS=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    python tools/prefetch_weights.py        # HF_TOKEN は 2 と同じく環境変数で。huggingface_hub は runtime と同じ 1.20.1
  ```
  **`HF_HUB_DISABLE_SYMLINKS=1` は必須**。HF cache は普段 symbolic link を使うが、macOS は exFAT の上の
  symbolic link を独自形式（XSym）の普通の file として書くので、Thor（Linux）からは壊れた file に見える。
  この設定だと `snapshots/` に本物の file が置かれる（新しい file は移動なので、容量は倍にならない）。
  Linux で取るときは 2 の `docker run` に `-e HF_HUB_DISABLE_SYMLINKS=1` を足し、`-v` を `<USB>/hf_cache` にする。
- [ ] **`SHA256SUMS`**（path は `hf_cache` からの相対。USB の上でも Thor に copy した先でも確かめられる）
  ```bash
  cd <USB>/hf_cache && find ./hub -type f ! -path './hub/.locks/*' -print0 | sort -z | xargs -0 shasum -a 256 > ../SHA256SUMS
  ```
- [ ] **Thor の image**（無圧縮。Thor に zstd が無くても `docker load` だけで入る）
  ```bash
  docker save -o <USB>/ikea-thor_<TAG>.tar <IMAGE>
  cd <USB> && shasum -a 256 ikea-thor_<TAG>.tar > SHA256SUMS.image
  ```
- [ ] （任意）運営 package（`iacevaltest/iros_g1_orin_package`、手順の根拠）

## 4. 会場での確認（**既にある物は入れない**）

運営が image を事前に取っていたり、前の session の重みが Thor に残っていたりする。足りない物だけを USB から入れる。
USB は Thor に mount して使う（exFAT は Linux 5.7 以降なら標準で読める）。

1. **image**
   ```bash
   docker images --digests | grep ikea-thor                 # manifest.yaml の images.thor の digest があれば load 不要
   cd <USB> && sha256sum -c SHA256SUMS.image                # 無いときだけ。USB の中身が壊れていないか
   docker load -i <USB>/ikea-thor_<TAG>.tar
   ```
2. **重み**（image を使って、ネット無しで確かめる）
   ```bash
   docker run --rm -v $RAMEN_HOST_DIR/hf_cache:/root/.cache/huggingface:ro \
     <IMAGE> pixi run --as-is -e runtime python /app/tools/prefetch_weights.py --check
   ```
   1 行ずつ `OK`（大きさ）/ `MISSING`（何が・なぜ）が出て、最後に `all present` か `N missing`。
3. **足りなければ**
   ```bash
   mkdir -p $RAMEN_HOST_DIR/hf_cache
   cp -a <USB>/hf_cache/. $RAMEN_HOST_DIR/hf_cache/                            # 既にある物に重ねる (中身は同じ)
   cd $RAMEN_HOST_DIR/hf_cache && sha256sum -c --quiet <USB>/SHA256SUMS         # copy した後の中身が壊れていないか
   ```
   もう一度 2 を回して `all present` を確かめる。copy 先は `-v` で mount する `$RAMEN_HOST_DIR/hf_cache` と同じにする
   （別の場所に copy すると container から見えない）。
