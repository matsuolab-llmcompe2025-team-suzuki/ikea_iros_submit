# 会場前の確認（Vast.ai の GB10）— Team RAMEN

提出 image を会場に持っていく前に、**Thor に近い機械で image そのものを起動して**、ネット無しで
全 stage が立ち上がるかを確かめる手順。image を焼き直したら毎回やる（準備込みで 45 分〜1 時間、$1〜3 ほど。§7）。

会場の手順は `INSTRUCTIONS.md`、重みは `WEIGHTS.md`。

**更新状況（Issue #12）:** 本体 `e3a4187` の `20260927-rebuild5` を GHCR へ公開済み。
ARM64 [CI run 36264063035](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/actions/runs/36264063035)
が build/import・重力補償 probe を通過した（build commit `56c69c5`）。digest は `manifest.yaml` に固定。
手元の提出用 test 33 件、本体の boundary・Stage・操作・重力補償などの回帰 test 311 件が成功。
この PC の GHCR 読取権限が不足しているため再 pull は未確認。公開の根拠は上記 CI の push 成功と digest 出力。
§6 の測定結果は旧 `rebuild4` のもの。新 image の GB10 全 stage 起動・Thor 接続試験は未実施であり、
CI の build/import 検査と手元の mock test だけでは実機動作確認済みと扱わない。

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
| 会場で切り替える候補（`policy_config.yaml` の `variant_sets`）と DP が、ネット無しで読めるか | 候補の model の動きの良し悪し |
| `--actuate` の経路: Enter 1 → go-live 待ち → 安全停止の判断待ち / Ctrl+C → 後始末 | 実機が指令へ追従すること（mock の sin 波で go-live が誤成立する場合がある） |
| conformance | |

4-6 の run 以外は `--actuate` を付けないので、指令は 1 通も出ない（起動確認だけで終わる）。4-6 の run も、
指令の宛先（Thor が bind する `:5556`）には誰もつながっていない。

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

作業 branch（`issue/*`）で `.github/ci-build-request.env` を `TAG=gb10-test-<本体の commit>`・`PUSH=true` にして
push する（CI が arm64 で焼いて push、12〜30 分）。または main の workflow を手で呼ぶ:
`gh workflow run build-thor-image.yml --ref <branch> -f tag=gb10-test-<commit> -f push=true`。
本番の tag とは分ける。合格したら、焼き直さずにその版へ本番の tag を足す（試験用 tag は同じ版なので残る）。

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

- `running` でも SSH がすぐつながるとは限らない。host によっては Vast が SSH を入れる途中で失敗し続けて、
  つながらないままになる（2026-09-25 のカナダの host。container の log は `PUT /api/v0/instances/request_logs/<id>/`
  で読める）。10 分つながらなければ消して別の host を借りる（そのとき PAT がもう一度要る）。
- 待つ script を zsh で書かない（`set -- $VAR` が単語に分かれず、つながっているのに気づかない）。

### 4-3 道具を置き、環境の GPU を確かめる

bash で実行する（zsh は `$SSH` を単語に分けないので動かない）。

```bash
SSH="ssh -i <秘密鍵> -p <port> root@<ip>"
# scp は使えない (Vast の SSH は sftp を通さない)。標準入力で送る
for f in tools/gb10/*; do $SSH "cat > /root/$(basename $f) && chmod +x /root/$(basename $f)" < $f; done
$SSH 'apt-get update -qq && apt-get install -y -qq strace'   # 使い捨ての container に OS の道具を入れるだけ
$SSH 'bash /root/check_envs.sh'
```

### 4-4 重みを取り、token を消し、ネット無しで確かめる

```bash
$SSH 'umask 077; mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token' <<< "$HF_TOKEN"
# 既定と variant_sets の全部に加えて、会場の候補には入れていない rotate の DP (DP が image の中で動くかを見る)
DP=rotate_table_base_diffusion
$SSH "cd /app/ramen && HF_HUB_OFFLINE=0 pixi run --as-is -e runtime python /app/tools/prefetch_weights.py --variant $DP"
$SSH 'rm -f ~/.cache/huggingface/token ~/.cache/huggingface/stored_tokens'
$SSH "cd /app/ramen && pixi run --as-is -e runtime python /app/tools/prefetch_weights.py --check --variant $DP"
```

### 4-5 各 stage を起動する

container では `unshare`（ネットを断った空間）が許されないので、strace で全 process の `connect()` を記録して
外向きが 0 件かを見る。1 本ずつ裏で流す（SSH は `-n` と `setsid nohup … < /dev/null &` で切り離す）。

既定（Stage 0/1/2/5）に続けて、会場の候補（`all6_400k`）と DP でも起動する。候補の stage は、切り替わる
model が載るもの（Stage 2 = 台を回す・insert・締め付け、Stage 5 = flip。flip は GR00T と違い overlay を使うので YOLO も要る）。

```bash
$SSH -n 'setsid nohup bash -c "for s in 0 1 2 5; do /root/run_stage.sh \$s stage\$s; done; \
  NOSTRACE=1 /root/run_stage.sh 1 stage1_plain; \
  /root/run_stage.sh 2 stage2_all6 --policy-variant-set all6_400k; \
  /root/run_stage.sh 5 stage5_all6 --policy-variant-set all6_400k; \
  /root/run_stage.sh 2 stage2_dp --policy-variant-rotate-table-base rotate_table_base_diffusion; \
  /root/run_stage.sh 2 stage2_pose --boundary-lane pose" \
  > /dev/null 2>&1 < /dev/null &'
$SSH 'python3 /root/summarize.py /root/runs/stage*'     # 終わったら。exit 0 = 合格
$SSH 'cd /app/ramen && pixi run --as-is -e runtime python /app/conformance.py --lane decoupled'
```

strace の下は起動が遅く出る（VLM で 1.3 倍ほど）。起動秒は `stage1_plain`（strace 無し）で見る。
起動 log の `[init] policy variants: … (set:all6_400k)` で、候補に切り替わったことを確かめる。
既定（本体 `9965c90` 以降）は joint lane で `[boundary] JointSink (joint lane) bound on …`。pose lane（`stage2_pose`）は
`[boundary] DecoupledSink (pose lane) bound on …` と `[boundary] wrist_roll clamp: off …` が出ること。

### 4-6 `--actuate` の経路

`ACTUATE_HOLD=<秒>` で `--actuate` を付けて起動し、Enter 1 の問いに改行を送り、go-live 待ち（`[go-live]`）が
出てからその秒数だけ待つ。安全停止で「判断待ち」なら Enter で戻し動作を承認し、それ以外は
自分の擬似端末へ Ctrl+C を送る。外向きの接続は 4-5 で見たので strace は付けない。
**この自動承認は mock 専用。実機で run_stage.sh を使わない。**

模擬の PC2 の関節は指令と関係なく sin 波で動くので、go-live 待ちは「ついてきた」と成立してしまい、その先の
準備動作（腕を動かす）は指令に従わないので時間切れになる。見るのは、そこまでに指令の経路（実測の関節の読み取り・
到達の誤差（joint lane は関節角の `joint_error`、pose lane は運営 IK と同じ URDF での手先の `ee_pos_error`）・
publish・後始末）がコードの誤り無しに動くこと。
- Stage 1〜5: 開始姿勢（時間切れ）→ 安全停止・判断待ち → Enter 承認 → 後始末 → rc=2。
  `[safety-stop] operator transition … did not reach its target` と判断待ち・承認の両 log があること。
- Stage 0: 腕を下ろす準備動作が時間切れ → 判断待ち → Enter 承認 → 後始末 → **設計どおりの停止**（腕を下ろせないまま歩かない。
  `RuntimeError: lowering the arms before the walk failed: … did not converge`、rc=1）
- `actuate5_pose_clamp`: `[boundary] wrist_roll clamp: on …`。`actuate0_converged`: `[init] walk latch check = converged (cli)`
  （模擬の PC2 では下ろす動きが収束しないので、終わり方は Stage 0 と同じ設計どおりの停止）

```bash
$SSH -n 'setsid nohup bash -c "for s in 0 5; do NOSTRACE=1 ACTUATE_HOLD=60 /root/run_stage.sh \$s actuate\$s; \
  NOSTRACE=1 ACTUATE_HOLD=60 /root/run_stage.sh \$s actuate\${s}_pose --boundary-lane pose; done; \
  NOSTRACE=1 ACTUATE_HOLD=60 /root/run_stage.sh 5 actuate5_pose_clamp --boundary-lane pose --wrist-roll-clamp on; \
  NOSTRACE=1 ACTUATE_HOLD=60 /root/run_stage.sh 0 actuate0_converged --walk-lowering-check converged" \
  > /dev/null 2>&1 < /dev/null &'
$SSH 'python3 /root/summarize.py /root/runs/actuate*'   # 終わったら。exit 0 = 合格
```

2026-09-25 の本番 image（`20260925-rebuild`）は、ここで落ちる不具合（`main()` の numpy の import 漏れ、
本体 #164）を持っていた。4-5 は `--actuate` 無しなので通っていた。

### 4-7 片付け

```bash
curl -s -X DELETE -H "Authorization: Bearer $VAST_KEY" https://console.vast.ai/api/v0/instances/<instance id>/
```

試験用 tag を GHCR から消し、PAT を revoke する。

## 5. 合格の基準

- `summarize.py` が exit 0: 全 stage が `rc=0`（`[preflight] … validation passed; NO command sent`）、**外向き connect 0 件**。
  候補（`stage*_all6`）と DP（`stage2_dp`）も同じ
- `--actuate` の run（`actuate*`）: Enter 1 → `[go-live]` まで進み、log にコードの誤り（`NameError` など。
  後始末は例外を握って `[return] failed: …` と出すので、名前で見る）が無い。終わり方は Ctrl+C（rc 0 か 130）か
  4-6 の設計どおりの停止（Stage 0 の rc=1、Stage 1〜5 の安全停止 rc=2）。
  安全停止では判断待ちと操作者の応答が必須。それ以外の例外は不合格
- `prefetch_weights.py --check` が `all present`、conformance が `PASS`
- GPU の使用量が基準値から大きく増えていない（増えたら model か設定の変更を疑う）

## 6. 基準値（2026-09-26 昼、image `gb10-test-9965c90` = `sha256:0cf460af…` をそのまま起動 = 本番 `20260926-rebuild4`）

既定の `--gpu-models all`（起動口が付ける）で、stage の model を全部載せた状態。host はハンガリー（offer 52373900、
前回 `gb10-test-5fac481` と同じ host）。この image から既定の送り方は **joint lane**（本体 #170）。時間の都合で DP は流していない
（DP の読み込みは本体・image とも前回 `20260925-rebuild3` から変わっていない）。

| 項目 | 値 |
|---|---|
| 4 環境の GPU | runtime torch 2.12.1 / desktop 2.11.0 / vlm 2.13.0 / pick 2.11.0、全部 cu130、`cap (12, 1)`・integrated |
| 重み | 既定・set の全部で `--check` は all present（DP の `--variant` は付けていない） |
| Stage 1（strace 無し） | 全体 198 秒。VLM の起動 160 秒、`vlm_latency` 0.75 秒。その後 model 3 つ |
| strace 下の各 stage（既定 = joint lane） | Stage 0: 8 秒 / Stage 1: 267 秒 / Stage 2: 247 秒 / Stage 5: 22 秒、すべて rc=0。起動 log に `[boundary] JointSink (joint lane)` |
| 候補 `all6_400k` | Stage 2: 223 秒・GPU 30.6 GB（台を回す・insert・締め付けが all6）/ Stage 5: 9 秒（flip の RAMEN-Ori と YOLO） |
| pose lane（`--boundary-lane pose`） | Stage 2: 235 秒・GPU 42.7 GB。`[boundary] DecoupledSink (pose lane)` と `wrist_roll clamp: off` |
| GPU の使用量の最大（既定） | Stage 1: 41.7 GB / **Stage 2: 41.9 GB（VLM と 4 model）** / Stage 5: 7.1 GB |
| MemAvailable の最小 | Stage 2 で 44.6 GB 残る（Thor の 128 GB でも余裕） |
| 空きの判断 | cudaMemGetInfo は page cache を使用中と数えて空き 7〜9 GB と出る。MemAvailable（55〜65 GB）で判断しているので読み込める |
| 外向き connect | **strace を付けた全 run で 0 件**（VLM `:8000`・カメラ `:5555`・状態 `:5557`・FlashInfer の閉じた `:9` と Unix socket だけ） |
| `--actuate`（4-6） | joint・pose とも Stage 5: rc=0（88 秒）、Enter 2 の問いは `NOT reached`（joint は `worst=R.wrist_yaw error=0.831rad`、pose は `ee_pos_error=0.0684m`）/ Stage 0: 設計どおりの停止（64 秒）。`--wrist-roll-clamp on` で `clamp: on`、`--walk-lowering-check converged` で `walk latch check = converged (cli)`。コードの誤り 0 件 |
| conformance | PASS |

起動秒は host の disk の速さで変わる（前回のスペインの host は strace 下の Stage 1 が 478 秒）。GPU の使用量はどれも同じ。
費用は instance 1 台・約 45 分で約 $3.0（この host は回線の従量料金が高い。重み 85 GB と image 9 GB の取得が大半とみられる）。

**`20260925-rebuild3` から入った直しで、この image で確かめたもの**: Stage 1〜5 の開始姿勢への準備動作が時間切れで
先へ進んだとき、Enter 2 の問いが「reached」ではなく `[gate] WARNING: … NOT reached (…)` と出る（本体 `ed28f38`）。
rebuild3 までは「initial arm/hand pose is reached and actively held」と出ていた（模擬の PC2 の Stage 5 で、準備動作が
15 秒で時間切れ → 手の姿勢 → 保持 → この問い）。進む決まり（完了か時間切れで次へ）は変わらない。

この確認で見つけて直したもの（1 回目、image `gb10-test-d029549`。2 回目で直った image を確認）:

1. **GR00T 53D（insert・締め付け・flip）がネット無しで読めなかった**。worker の環境の huggingface_hub 1.28 は、
   commit hash 指定でも cache に file 一覧の記録が無いとネットへ取りに行く。事前取得は runtime 環境
   （1.20.1）で行うので記録が無い。→ 本体 `5fac481`（オフラインなら `local_files_only=True`）。
   image の中の 4 環境で huggingface_hub の版が違う（1.20.1 / 1.28.0 / 1.32.0 / 1.22.0）ことに注意。
2. **YOLO（ultralytics 8.4.80）が外に出ていた**。import 時に DNS でネットの有無を調べ、推論の開始時に
   Google Analytics へ利用統計を送る。→ image の ENV に `YOLO_OFFLINE=true`（submit `6c0c6e6`）。

2 回目の image（`20260925-rebuild`）の後に見つけて直したもの（3 回目の上の表で確認）:

3. **`--actuate` を付けると起動直後に止まった**。`main()` の会場の処理が `np` を使うのに import が無かった
   （本体 #164 が見つけて直した）。確認が `--actuate` 無しだったので通っていた → 4-6 を追加。
4. **DP（act_diffusion）がネット無しで起動できなかった**（読んで見つけた）。worker を `--as-is` 無しで起動していた、
   cache だけを見る指定が無かった（1. と同じ）、ckpt の `pretrained_backbone_weights`（ImageNet の ResNet18）を
   torchvision が model を作る時点でネットから取りに行く。→ 本体 `b591049`。

## 7. 費用

GB10 は 1 時間 $0.3〜0.7（disk 250 GB の保存料金込みで $0.4 前後）。1 回の確認は image の pull 10 分・
重み 3 分・stage の起動 30〜40 分（4-5・4-6 の全部）で、合わせて 45 分〜1 時間。host によっては回線の従量料金が
時間料金より大きい（2026-09-26 のハンガリーの host は約 45 分で $3.0。スペインの host は約 1.1 時間で $0.99）。
借りる前に offer の `inet_down_cost`（1 GB あたり）を見る。2026-09-25 の初回は $0.90（不具合の調査を含む）。
