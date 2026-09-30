# 会場前の確認（Vast.ai の GB10）— Team RAMEN

提出 image を会場に持っていく前に、**Thor に近い機械で image そのものを起動して**、ネット無しで
全 stage が立ち上がるかを確かめる手順。image を焼き直したら毎回やる。
起動確認だけと、全Stageの操作・故障注入・連続推論を含む拡張検証では所要時間が異なる（§7）。

会場の手順は `INSTRUCTIONS.md`、重みは `WEIGHTS.md`。

**更新状況（Issue #20）:** 本体 `1cd5fdf` の `gb10-guard-1cd5fdf` を GHCR へ公開済み。
ARM64 [CI run 36567638842](https://github.com/matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit/actions/runs/36567638842)
が成功した（build commit `55c1a55`）。digest は `manifest.yaml` に固定。
このimageの検証結果・完了状況は[GB10_STATE_GUARD_REPORT.md](GB10_STATE_GUARD_REPORT.md)を参照。
§6は旧`rebuild4`、§8は旧`rebuild5`、[準備動作の記録](GB10_PREPARATION_REPORT.md)は旧`10b8d73`の履歴であり、新imageの検証を代替しない。
GB10/mockでの成功を、Thor/G1の実機動作やタスク成功の保証とは扱わない。

## 1. なぜ GB10 か

- 提出 image は **arm64**（Thor 用）。RunPod の GPU は全部 x86 なので、image が動かない（2026-09-24 に確認）。
- Vast.ai の **GB10**（NVIDIA DGX Spark）は arm64・Blackwell・CPU と GPU で約 122 GB を共有するメモリで、
  Thor（arm64・Blackwell・128 GB 共有）に近い。image の土台は Jetson 専用ではない汎用の arm64 CUDA 13
  （`nvidia/cuda:13.0.3-base-ubuntu24.04`）なので、**GHCR から pull してそのまま起動できる**。

## 2. 確かめること / 確かめられないこと

| 確かめること | GB10だけでは確かめられないこと |
|---|---|
| image の起動口・4 環境（runtime / desktop / vlm / pick）が GPU を使えるか | sm_110（Thor）専用の kernel（GB10 は sm_121） |
| 重みの事前取得と、ネット無しの `--check` | 実機・実カメラ・運営の WBC / adapter |
| Stage 0〜5 の起動（VLM・YOLO・全 model の読み込み）が**ネット無しで**通るか | Thor の disk の速さ（重みの読み込み秒は変わる） |
| **外へ一度も接続しないか**（strace で全 process の `connect()`） | |
| `--gpu-models plan`（#188 の既定）の読み込み: 各 model の秒数・VLM の起動時間・空きメモリ・読み込み中の制御の周期（orch log の `model_load`）、R の候補も含めた GPU の使用量・共有メモリの余裕 | |
| 会場で切り替える候補（`policy_config.yaml` の `variant_sets`）と DP が、ネット無しで読めるか | 候補の model の動きの良し悪し |
| `--actuate` の経路: 準備goto・到達待ち・Enter/N/R・安全停止の判断待ち・後始末 | 実機の追従・干渉・接触（遅れ付き模擬追従でも物理応答は証明できない） |
| conformance | |

4-6 の run 以外は `--actuate` を付けないので、指令は 1 通も出ない（起動確認だけで終わる）。
4-6 はloopbackの模擬adapterだけが `:5556` を購読する。会場への接続や実機指令は行わない。

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
#    "image_login":"-u <GitHub user> -p <PAT> ghcr.io","env":{},"disk":350,
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
# strace は /install の確認・承認後、使い捨てGB10だけへ固定versionを導入する。
# 検証時: 6.8-0ubuntu2。導入直前にも apt-get -s で追加依存を確認する。
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

現行joint準備の検証には `operator_probe.py` を使う。これは運営adapterの
`_handle_joint` / `_handle_goto` と遅れ付き模擬関節を使い、実際のimageの
`ramen-venue --actuate` を擬似端末から操作する。接続先はloopbackに固定する。
**キーの自動承認はこの隔離試験専用。会場・実機では実行しない。**

`fetch_validation_assets.py` で固定revisionの運営コードとWBC資産を用意する。
公式adapter用のPython依存は、`GB10_PREPARATION_REPORT.md`記載の承認済みversionを
`/install`手順で独立venvへ入れる。本番のPython環境や運営コードを書き換えない。

```bash
# GB10内で、他のprobeが停止し :5555/:5556/:5557 が空いているときだけ実行
export PYTHONPATH=/root/gb10-validation/wbc:/app/ramen:/app:/root
export RAMEN_TEST_ORGANIZER=/root/gb10-validation/organizer
export OPENBLAS_NUM_THREADS=1
PY=/root/gb10-validation/.venv/bin/python
"$PY" /root/operator_probe.py --case stage --stage 1 --dwell-seconds 30 \
  --trace --output /root/runs/stage-1
"$PY" /root/operator_probe.py --case full --dwell-seconds 30 \
  --trace --output /root/runs/full
"$PY" /root/operator_probe.py --case gate-camera --trace --output /root/runs/gate-camera
"$PY" /root/operator_probe.py --case retry-camera --trace --output /root/runs/retry-camera
"$PY" /root/operator_probe.py --case soak --dwell-seconds 600 \
  --trace --output /root/runs/soak
```

同じ方法で個別Stage 0〜5、`retry` / `next` / `camera` / `state` / `worker` を逐次確認する。
出力先は毎回新しいdirectoryにする。各`result.json`の`passed`、wireと外向きconnectを確認する。
Enterが速度・鮮度条件で拒否された場合、probeは拒否ログを確認した後だけ新しいEnterで再確認する。
到達判定自体は変更しない。安全停止を検出した場合は通常のPolicy開始を再試行しない。

`preparation_adapter_probe.py`は受信欠落・再試行・clamp・中断・遅延到達を別途検査する。
未到達timeoutは終了ではなく`holding`、到達後は`ready`となり新しいEnterを待つ。
古い開始待ちやR待ちが、安全停止画面のEnterを消費しないことも確認する。
旧`run_stage.sh`のsin波mockはpreflight用であり、準備の追従・到達の証拠にはしない。

2026-09-25のimageでは`--actuate`なしの検査を通っても、実行経路にimport漏れがあった。
このため、preflightだけをもって合格としない。

### 4-7 片付け

```bash
curl -s -X DELETE -H "Authorization: Bearer $VAST_KEY" https://console.vast.ai/api/v0/instances/<instance id>/
```

ログを退避してhashを確認してから、今回借りたinstanceだけを削除する。
digest固定で提出する検証済みimageは消さない。不要な試験tagと一時認証を整理する。
他作業が使う共用認証を勝手にrevokeしない。

## 5. 合格の基準

- `summarize.py` が exit 0: 全 stage が `rc=0`（`[preflight] … validation passed; NO command sent`）、**外向き connect 0 件**。
  候補（`stage*_all6`）と DP（`stage2_dp`）も同じ
- `operator_probe.py`の個別・連続Stage、N/R、故障注入、soakが全て`passed=true`。
  通常ケースは実際のPolicy開始・所定時間の推論・正常終了まで確認する。
  故障ケースは安全停止と新しいEnterによる戻し承認を確認する。
  `[return] failed`や未回収worker、予期しない例外を合格扱いにしない。
- `preparation_adapter_probe.py`の全ケースが合格し、wire再生が運営adapterに受理される。
- 全model構成で実forwardを実施する。起動だけ・即追従mockだけでは合格にしない。
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

GB10 は 1 時間 $0.3〜0.7程度（disk容量・hostに依存）。旧来の起動確認は image の pull 10 分・
重み 3 分・stage の起動 30〜40 分（4-5・4-6 の全部）で、合わせて 45 分〜1 時間。host によっては回線の従量料金が
時間料金より大きい（2026-09-26 のハンガリーの host は約 45 分で $3.0。スペインの host は約 1.1 時間で $0.99）。
借りる前に offer の `inet_down_cost`（1 GB あたり）を見る。2026-09-25 の初回は $0.90（不具合の調査を含む）。

Issue #16の拡張検証は、全Stage個別・連続実行、操作・故障注入、600秒soak、
990回forward、公式adapterへのwire再生、候補別preflightを逐次行う。
上記45分〜1時間はこの全検証の見積りではない。実行時間と結果は
[GB10_PREPARATION_REPORT.md](GB10_PREPARATION_REPORT.md)に記録する。

## 8. rebuild5 の拡張検証と提出判断

rebuild5 の実測記録は [GB10_REBUILD5_REPORT.md](GB10_REBUILD5_REPORT.md) を参照。
実機なしのソフトウェア検証を完了し、2026-09-27にユーザーからmainへのマージ承認を受けた。
通信断時の歩行速度保持（Issue #14）はユーザー指示により今回の修正・マージ対象外とする。
未解決の既知事項と会場確認項目は残す。マージ承認を全故障ケースの合格や実機安全性の保証として扱わない。

| 項目 | 必須証拠 | rebuild5 状態 |
|---|---|---|
| image 同一性 | manifest digest、image ID、RAMEN_SOURCE.txt、CPU/GPU/driver/空き容量 | 確認済み。vendor 300 file の SHA-256 一致 |
| 4 Python 環境 | check_envs.sh の成功終了、CUDA bf16 演算の有限値、各環境の版 | 4 環境で成功 |
| 重みと offline | 全既定・候補・DP の cache 検査、実行時の外向き接続 0 件 | cache検査成功。全Stage連続実行と11構成990callのstraceで外向きconnect 0件 |
| 全 Stage | **0/1/2/3/4/5 を省略せず**既定の joint lane で起動。4 RGB・関節・Dex1 の入力記録 | 全 Stage の明示的 preflight 完了を確認 |
| 実モデル forward | RAMEN-Ori、GR00T 53D、pick worker、DP、YOLO、VLM の実推論・有限出力・所要時間。validate_load_and_release だけでは合格にしない | 11 構成 x 90 call、VLM の起動時 self-check 成功 |
| 代替経路 | all6_400k、pose lane、wrist clamp、walk-lowering option を個別記録 | Stage 2 all6/DP、Stage 5 all6/pose/clamp、Stage 0 converged の preflight 成功 |
| joint の送信契約 | 実 socket で chunk を受信し、shape、有限値、hand 範囲、base height、joint 順序、時刻・周期を検査 | 模擬 socket の 16 行、nav=0、height=0.74、時刻単調性を確認。関節順序は回帰 test。PC2 の実時計は未確認 |
| 操作遷移 | 対話端末で Enter/N/R、retry、保持、Ctrl+C、再起動を検査 | retryに加え、Stage 0→5の16区間を各30秒実行し、Enter/N/保持/正常終了を確認。各試験で独立再起動 |
| 運営adapter | 指定revisionの公式回帰と提出packetの非作動再生 | 公式84件成功。全Stageの記録5,298packetの行を再生し、拒否・clamp 0、腕/Dex1/nav/高さの対応一致 |
| client/transport消失 | 非ゼロnav送信後、全clientが消えた場合の停止挙動 | **自動停止を確認できない**。運営adapterが最後のnavを保持することを再現。Issue #14で運営確認が必要 |
| 故障注入 | camera/state 途絶、worker 異常、未到達を区別し、停止時の nav=0、最後の arm/hand target 保持、判断待ちを記録 | 4 ケースで判断待ちを確認。未到達 mock は戻しも非収束。実 WBC の保持・歩行からの制動は未確認 |
| 長時間実行 | 同じ方策を継続実行し、異常・メモリ増加・正常停止を検査 | flipを600秒実行して正常終了。元packet 5,306件の運営adapter再生も成功 |
| 負荷と後始末 | 最大 GPU 使用量、最小 MemAvailable、終了後の worker/VLM/socket 残留なし | 全Stage連続で最大49.46 GiB、最小空き31.99 GiB。10分soakはGPU 7.12 GiBで安定。残留なし、証拠回収・検証instance削除済み |

### 検証データの扱い

- 各 run に image/source の識別子、コマンド、開始・終了時刻、終了コード、全ログ、合否理由を保存する。
- 合成画像や指令追従 mock は配線・実行系の検査に限る。タスク成功や本物の WBC の安定性の証拠にしない。
- 関節を sin 波で動かす標準 mock の go-live 成立は追従確認の証拠ではない。model load、実推論、操作遷移を別々に判定する。
- 外部ホストへロボット指令は送らない。自動 Enter/actuate は実機と隔離した検証用環境のみで行う。
- 問題を修正して image 入力が変わった場合は再ビルドし、新 digest で必要試験を再実施する。
- GB10 の sm_121 では Thor の sm_110、実カメラ、DDS、WBC、実機接触を保証できない。未確認事項を明記して接続試験へ渡す。

手元の検証ツール回帰: `tests/test_gb10_envs.py` の 5 件が成功。
従来の check_envs.sh は command substitution の失敗を echo が隠していたが、各環境の失敗を非ゼロ終了として返すよう修正済み。
これは **GB10 での GPU 実測ではない**。

### 追加の検証ツール

`tools/gb10/` は image に含めず、隔離した GB10 container へ別途転送する。
`forward_matrix.py` は `policy_config.yaml` の既定・variant set と DP を順番に実推論する。
`operator_probe.py` は Enter/R/N と camera/state/worker 故障を、loopback の
`following_mock.py` と `wire_probe.py` で検査する。後者は物理 simulator ではない。
`--case full`はStage 0→5の全16 policy区間を操作者のNで進める配線試験であり、
VLMによる把持判定や物理的な組立完了を証明するものではない。

```bash
# GB10 container の /app/ramen で実行。helper 一式を /root へ転送済みであること。
pixi run --as-is -e runtime python /root/forward_matrix.py --output /root/runs/forwards
pixi run --as-is -e runtime python /root/operator_probe.py --case retry --output /root/runs/retry
pixi run --as-is -e runtime python /root/operator_probe.py --case full --trace --dwell-seconds 30 --output /root/runs/full
pixi run --as-is -e runtime python /root/operator_probe.py --case soak --trace --dwell-seconds 600 --output /root/runs/soak
# 他の case: next / camera / state / worker。port/GPUを共有するので同時起動しない。
```

model forward は合成画像での実行可能性検査であり、タスク成功率の評価ではない。
`summarize.py` の `外向き通信: 未検査` は通信ゼロの証拠に数えない。
Stage結果をmerge判定へ使うときは`--require-trace`を指定し、trace不在・空の記録を不合格にする。
`strace`等の追加導入は`/install`の確認・承認後に隔離検証環境だけで行う。
対話試験のtraceには`--seccomp-bpf`を使い、CUDAの無関係なsyscallごとのptrace停止を避ける。
追跡対象は引き続き全子processの`connect,execve`。通常traceは計測対象自体を遅くするため、
鮮度停止が出た場合は非trace/filtered traceと比較する。安全閾値を緩めて合格させない。

`wire_probe.py`は元packetを長さ付き`.wire`と、解析用`.npz`へ保存する。
`organizer_replay.py --organizer <checkout> --capture <wire.wire> --output <result.json>`は
固定revisionの運営公式fixtureを使用してWBC/Dex1のfake backendへ再入力する。ソケットは作らない。
運営テスト依存とWBCの固定commit・LFS資産が必要（版は検証レポート参照）。
実測姿勢をpacket先頭targetに設定し時刻だけ更新するため、実機追従誤差や実ホスト間の時計精度は検証しない。
旧記録用`--rows-capture <wire.npz>`ではenvelopeを再構成するので、元wireそのものの試験とは区別する。

## 9. 任意の PC2 state guard を含む image を焼き直したとき

本体 [#184](https://github.com/matsuolab-llmcompe2025-team-suzuki/iros_2026_ramen/issues/184) の guard は任意の経路で、
既定の経路（Thor が `:5557` を直接読む）は変えない。`gb10-preparation-10b8d73` には入っていない。焼き直したら、上の手順に加えて:

1. `ramen/` は push 済みの commit から通常の `tools/sync_ramen.sh <commit>` で写す（`--worktree` の写しは Dockerfile・CI が拒む）。
   `tests/test_manifest.py` は `ramen/RAMEN_SOURCE.txt` の commit と manifest の tag が合うまで通らない。
2. **既定の経路**（option 無し）で、これまでと同じ Stage の確認を通す（guard の追加で既定の経路が変わっていないこと）。
3. **guard の経路**: 模擬の camera/state に加えて、同じ loopback に guard（`venue_state_guard.serve` と模擬 DDS）を立て、
   `--boundary-state-guard --boundary-state-port 5558` で Stage を通す。DDS の停止・bridge だけの停止・tick の逆行・guard の再起動・
   往復の遅れで、古い state で準備が進まず、安全停止に入ることを確かめる。実機への故障注入はしない。
   `operator_probe.py --state-guard` は両側を自動設定する。`following_mock.py` は別 thread の本番 `serve()` に模擬 lowstate を渡す。
   `state` は bridge だけ停止、`guard-dds` は bridge の再配信を続けたまま DDS だけ停止する。通常の経路は option 無し。
   tick 逆行・guard 再起動は `tests/test_gb10_guard.py`、通信遅延と静止時間は本体の `test_venue_state_guard.py` / `test_boundary_preparation.py` でも確認する。
4. PC2 用 bundle（`tools/package_pc2_guard.py`）を同じ commit から作り、bundle の SHA256・image の digest・本体 commit・結果を記録する。
5. 実機の DDS の負荷・タイミングは GB10 では確かめられない。運営の許可を得て、会場で `--check-only` と短い試運転から使う。

```bash
# GB10 only, /app/ramen; helpers are under /root. All endpoints are loopback.
PY=/root/gb10-validation/.venv/bin/python
"$PY" /root/operator_probe.py --state-guard --case full --trace --output /root/runs/guard-full
"$PY" /root/operator_probe.py --state-guard --case state --trace --output /root/runs/guard-bridge-stop
"$PY" /root/operator_probe.py --state-guard --case guard-dds --trace --output /root/runs/guard-dds-stop
```

会場の override を使う試験では各 command に `--skill-config /root/submission-validation/venue/skill_config_venue.yaml`
を追加する（同じ commit の `venue/` を事前転送）。結果 JSON に採用した path を記録する。
