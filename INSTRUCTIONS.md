# 会場での動かし方 — Team RAMEN（Thor の image 1 つ）

会場では、運営の RUNBOOK（`iacevaltest/iros_g1_orin_package` の `docs/RUNBOOK.md`）どおり、
**全部を私たちが起動する**。PC2 では運営のプログラムだけを起動し、私たちの物は **Thor の container 1 つ**。
PC2 用の container は無い（運営 template の「PC2 の client」は使わない。理由は下の「全体の形」）。

## 0. 全体の形

```mermaid
flowchart LR
  subgraph PC2["PC2（ロボット搭載 = Orin）: 運営のプログラム"]
    BR["bridge<br/>:5555 カメラ / :5557 状態"]
    WBC["WBC<br/>run_wbc_with_dex1.py"]
    AD["adapter<br/>wbc_driver.py --actions-host THOR"]
  end
  subgraph THOR["Thor: 私たちの container"]
    EP["entrypoint<br/>:5556 を bind"]
    VLM["VLM server<br/>127.0.0.1:8000（Stage 1〜4 だけ）"]
  end
  BR -- ":5555 / :5557 を直接購読" --> EP
  AD -- ":5556 に接続" --> EP
  AD --> WBC
  EP -.-> VLM
```

- Thor の entrypoint が PC2 の `:5555` / `:5557` を**直接**読み、`:5556` は **Thor が bind** する。
  PC2 の adapter は `--actions-host <THOR_IP>` で Thor につなぐ（adapter の `--actions-host` は
  「:5556 を bind した側の host」、既定は 127.0.0.1 = PC2 自身）。
- VLM（hybrid pick の区間 1→2 の判定）は、Stage 1〜4 の run の中で container が自分で起動し、run の終わりに止める。
- container は run ごとに作り直す。重みは host の HF cache を読み取り専用で mount し、**実行中はネットに出ない**。

| Stage | 中身 | Enter |
|---|---|---|
| 0 | 準備（go-live 後に腕を下ろす → 台まで歩く → pick の開始姿勢） | 1 回（安全確認） |
| 1〜4 | 脚 1 本ずつ: 台を回す → pick（VLM + GR00T + IK + 持ち替え）→ insert → 締め付け | 2 回（安全確認 / policy 開始） |
| 5 | 台を裏返す（flip） | 2 回 |

## 1. 事前準備（会場の前に Thor で 1 回）

```bash
# 置き場所（例）。以下の手順はこの 3 つを使う
export RAMEN_HOST_DIR=~/ramen
mkdir -p $RAMEN_HOST_DIR/{hf_cache,outputs,vlm_cache}

# image（tag 20260925-rebuild3。digest は manifest.yaml の images.thor と同じ。GB10 で確認済み = VERIFY.md）
docker pull ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor@sha256:3064ee8bc0348bae4a31b7e7b98309850843eababbd2071cfe69b2cabcbaa8b8

# 重みの事前取得（ネットのある所で。会場の実行中は取りに行かない）と、ネット無しの確認
#   → WEIGHTS.md（一覧・取り方・USB に入れる物・会場での確認）
```

- `vlm_cache` は VLM の compile 結果の置き場。1 回目の run だけ小さな kernel の compile が走り、2 回目以降は再利用する。
- `outputs` に run ごとの log（`orch_logs/orch_*.jsonl`、VLM の `orch_logs/vlm_*.log`）が残る。

## 2. 1 run の手順（**順番が大事**）

```mermaid
sequenceDiagram
  participant P as PC2（人が SSH）
  participant T as Thor（container）
  P->>P: Step 0-1 環境・bridge（:5555 / :5557）
  P->>P: Step 2 WBC（run_wbc_with_dex1.py）
  P->>P: Step 3 adapter 試運転（--actions-host THOR、--live 無し）
  T->>T: Step 4 docker run … --stage N --actuate（model・VLM の読み込み）
  T->>T: Enter 1（安全確認）→ go-live 待ち（肩を少し動かし、実測がついてくるまで待つ）
  P->>T: Step 5 人が go-live（--live --engage-policy）
  T->>T: 開始姿勢・保持 → Enter 2 → policy（Stage 0 は Enter 2 無し）
  T->>T: 終わり: 手を開き、腕を下ろして container が終了
```

### Step 0〜1 [PC2] 環境とカメラ・状態の配信

RUNBOOK の Step 0（環境変数・`rt/lowcmd` を掴んでいる process が無いこと）と Step 1（`real_orin_cameras.py` と
`real_orin_state.py` の 2 本）をそのまま行う。

### Step 2 [PC2] WBC — **`run_wbc_with_dex1.py` で起動する**

```bash
conda activate g1_wbc
cd ~/GR00T-WholeBodyControl
python <運営 package の tools>/run_wbc_with_dex1.py \
  --interface real --no-with-hands --keyboard_dispatcher_type ros --no-enable-onscreen \
  --dex1-max-speed 4.2
```

- RUNBOOK の `run_g1_control_loop.py` **ではない**。それだとグリッパ（Dex1-1）を動かすものが居ない
  （`run_wbc_with_dex1.py` は同じ引数を受ける差し替えで、グリッパの指令を同じ `rt/lowcmd` に載せる）。
- `--dex1-max-speed 4.2`: 学習データのグリッパの速さ（運営の既定は 2.0）。
- `run_wbc_with_dex1.py` の PC2 上の置き場所は会場で確かめる（運営 package の `tools/` にある。運営 RUNBOOK
  （2026-09-25 版）の例は `~/wbc_adapter/deploy/run_wbc_with_dex1.py --interface eth0`。`--interface` は `real` でも
  インターフェース名でもよい）。
- **WBC の起動時の腕**: 運営 package が `f31952c`（2026-09-25）より古い wrapper だと、起動した瞬間に
  （adapter も私たちのコードもつながる前に）腕を肩 roll ±0.2・他 0 の姿勢へ 2 秒・フル剛性で動かす。新しい
  wrapper は実測の関節から始めて保持する（`--seed-from-measured` が既定。取れないと `[seed] WARNING: falling back
  to STOCK start-up` と出て古い動きになる）。どちらでも、**腕が台や物に届かない所で起動し**、止まってから stage の
  位置に置く。
- 安定した保持状態になってから次へ（`ros2 topic hz /G1Env/env_state_act`）。

### Step 3 [PC2] adapter の試運転（まだ `--live` を付けない）

```bash
conda activate g1_wbc
cd ~/wbc_adapter
python wbc_driver.py --lane decoupled --actions-host <THOR_IP> --state-source boundary
```

### Step 4 [Thor] 私たちの container

```bash
docker run -it --rm --runtime nvidia --gpus all -e NVIDIA_DISABLE_REQUIRE=1 --network host \
  -e IROS_ORIN_HOST=<PC2_IP> \
  -v $RAMEN_HOST_DIR/hf_cache:/root/.cache/huggingface:ro \
  -v $RAMEN_HOST_DIR/outputs:/app/ramen/outputs \
  -v $RAMEN_HOST_DIR/vlm_cache:/cache \
  ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor@sha256:3064ee8bc0348bae4a31b7e7b98309850843eababbd2071cfe69b2cabcbaa8b8 \
  --stage N --actuate
```

- `-it` 必須（Enter を押すため）。`<PC2_IP>` は通常 `192.168.123.164`（会場で確認）。
- 会場で変わらない option（boundary 経路・`:5556` の bind・VLM の起動・`--gpu-models all` など）は image の起動口
  （`docker/venue_entry.sh`）が付ける。打つのは `--stage N --actuate` だけ。後ろに足した option は上書きになる
  （例: `--gpu-models 2`）。
- 起動すると model と（Stage 1〜4 では）VLM を読み込む。**読み込みは時間制限なしで待つ**（10 秒ごとに経過が出る）。
  目安（GB10 = Thor に近い arm64・128 GB 共有メモリで実測、2026-09-25）: Stage 1〜4 は Enter 1 まで 4〜5 分
  （VLM の起動 約 3.3 分 + model の読み込み）、Stage 5 は 1 分弱、Stage 0 は十数秒。GPU は VLM 込みで最大 42 GB。
- `Enter 1`: ハーネス・E-stop・周りの空きを確かめてから押す。
- 使う model は起動 log の `[init] policy variants: insert=… (config)` で分かる（`config` = 既定、`set:…` / `cli` = 切り替え）。
- go-live 待ち: 両肩を少し（−0.05 rad）動かす指令を出し、実測がついてくる（0.02 rad）まで待つ。時間制限なし。

#### model を切り替えるとき（焼き直さない。既定は変わらない）

既定の model は image の中の `policy_config.yaml`（`default_variant_by_skill`）。`--stage N --actuate` の後ろに
足すだけで、その run だけ切り替わる。pick は hybrid（`groot_pick_legs_v1`）のまま。

| 足す option | 切り替わるもの |
|---|---|
| `--policy-variant-set all6_400k` | pick 以外の全部（台を回す・insert・締め付け・flip）を 6 skill 統合 RAMEN-Ori（400k）に |
| `--policy-variant-insert insert_table_leg_ramen_ori_all6_400k` | insert だけ。他は `--policy-variant-rotate-table-base` / `--policy-variant-rotate-leg` / `--policy-variant-flip` に `<skill>_ramen_ori_all6_400k` |

- 切り替え先は **`WEIGHTS.md` の一覧にある重みを使う slot だけ**（会場はネット無し。`policy_config.yaml` の
  `variant_sets` に書いた組み合わせは事前取得に入っている）。一覧に無い slot を指定すると、起動時に重みが無くて止まる。
- 新しい model（学習中の insert の DP など）は、本体で slot と set を足して image を焼き直してから使う。

#### 指令の送り方を切り替えるとき（joint lane、焼き直さない）

`--stage N --actuate` の後ろに `--boundary-lane joint` を足すと、その run だけ腕の関節角をそのまま送る
（運営が 2026-09-25 に足した joint lane。運営 IK を通らない）。既定は `pose`（手先の姿勢を送り、運営 IK が関節角に戻す）。

| | pose（既定） | joint |
|---|---|---|
| 送るもの | 手先の位置と向き (T,25) | 腕の関節角 (T,22)。手・歩行・骨盤高さの列は pose と同じ値 |
| 「腕が着いたか」 | 手先で比べる | 関節角で比べる（0.10 rad） |
| 前提 | — | PC2 の運営 package が `609f61d` 以降（`wbc_driver.py` の `--joint-lane on` が既定）。古いと指令が無視され、go-live 待ちのまま動かない |

- 起動 log の `[boundary] JointSink (joint lane) bound on …` で joint になったことが分かる（pose なら `DecoupledSink (pose lane)`）。
- 運営はまだ実機で試していない（「組み込んでよいが、本番で頼るのはまだ」）。09-27 に試し（6 の項目 11）、使うと決めたら
  Step 4 の `docker run` の行に `--boundary-lane joint` を入れて、それを会場の既定にする。

### Step 5 [PC2] go-live — **人がキーボードで打つ**（script や agent から実行しない）

```bash
python wbc_driver.py --lane decoupled --actions-host <THOR_IP> --live --engage-policy
```

- **`--state-source boundary` を付けない**（既定の `wbc` のまま）。decoupled で `boundary` と `--live` を
  一緒にすると adapter が `LAUNCH DENIED` で起動を拒否する（RUNBOOK の Step 6 の書き方はこの点でコードと違う）。
- `--engage-policy` 必須（無いと WBC の下半身が学習済みの制御に入らないまま、他は全部正常に見える）。
- E-stop 担当が付いてから。

### その後

- Stage 1〜5: 開始姿勢に移って保持 → `Enter 2` で policy が始まる。stage の中の model の切り替えは
  自動（腕を次の開始姿勢へ → 保持 → 次の model）。次へ進むのは model の完了か時間切れだけ（YOLO では進まない）。
- ⚠️ `Enter 2` の問いは、開始姿勢への準備動作が時間切れで先へ進んだときも「initial arm/hand pose is reached」と出る
  （未修正、`VERIFY.md` §6）。**押す前に、腕が開始姿勢にあるかを目で確かめる**。直前の `[orch] … timed out short of
  its target` の行が出ていたら、届いていない。
- Stage 0（Enter 2 無し）: go-live の直後に、WBC の既定の姿勢（前腕が前に出た HOME）から腕を下ろし
  （肩 roll ±0.2・肘 0.9）、台まで歩き、止まってから手を開いて pick の開始姿勢へ移る。
- 終わると手を開き、腕を下ろして（同じ姿勢）container が終わる。

### なぜこの順番か

- PC2 の配信・WBC・adapter（試運転）を先に立て、Thor を最後に起動する（RUNBOOK と同じ）。
- Thor を go-live より先に起動するのは、指令を 1 通も受けていない adapter は WBC に何も送らず、
  WBC 自身の 1 秒の見張りが速度制限を通らない保持の目標を差し込むため。Thor が go-live 待ちで指令を
  出している状態で go-live する。

## 3. 次の run

- **Thor の `docker run` だけをやり直す**（`--stage` を変える）。
- **adapter は止めない。** `--engage-policy` は押すたびに切り替わる。adapter を起動し直すなら WBC から起動し直す。

## 4. よく出る表示

| 表示 | 意味・すること |
|---|---|
| `error: PC2 … -e IROS_ORIN_HOST=<PC2 の IP> で渡す` | `docker run` に `-e IROS_ORIN_HOST=…` を付け忘れた |
| `[vlm] loading... Ns (no time limit; …)` | VLM を読み込み中。待つ |
| `[vlm] server ready` → `[vlm] warm-up done` → `[hybrid] … preflight passed … vlm_latency=…` | VLM の準備完了。`vlm_latency` は本番と同じ 5 枚の問い合わせ 1 回の秒数（5 秒以内に答えないと、ロボットが動く前の確認で止まる） |
| `the VLM server exited while loading` | VLM が起動に失敗。末尾の log と `outputs/orch_logs/vlm_*.log` を見る |
| `…:8000 is already in use` | 前の VLM が残っている。`docker ps` で古い container を確かめる |
| `[groot] waiting for the GR00T worker to load... Ns` | GR00T の model を読み込み中。待つ |
| `[groot] integrated GPU: load headroom from MemAvailable=…` | 情報。GR00T を読む前の空きメモリ |
| `sender clock offset ~ ±x.xxxs` | 情報。PC2 と Thor の時計の差（指令の送信時刻をこの分だけ直している） |
| 重みが cache に無い（`LocalEntryNotFoundError` など） | 事前取得の漏れ。`WEIGHTS.md` の 4（`prefetch_weights.py --check`） |

## 5. conformance（運営の適合試験）

```bash
# Thor（本番の run を動かしていないとき。:5555-5557 を 127.0.0.1 で使う）
docker run --rm --network host <IMAGE> pixi run --as-is -e runtime python /app/conformance.py --lane decoupled
# 手元（submit repo の直下）
python conformance.py --lane decoupled
```

`components/server.py` は conformance 専用（本番と同じ受け口・送り口で、実測の姿勢を保つ指令を送る）。
会場の run はこの file を通らない。`components/client.py` は conformance が起動するための置き物。

## 6. 09-27 の接続テストで確かめること

1. 運営 README（「You do not run any of this」）と RUNBOOK（「you run the whole pipeline yourself」）のどちらが正か。
   manifest に PC2 用の image が無くてよいか
2. bridge の起動 log が `head camera live at 1280x480` か（学習データと自前実機はこの 1280x480 モード）。運営は
   2026-09-25 に直した（`9e32910`: 1280x480 で開き、それ以外なら頭の画像を出さずに ERROR）。`HEAD_ALLOW_RESIZE_FALLBACK=1`
   が付いていると古い潰れた画（片目 1920x1080 を 640x480 へ、横 3/4）に戻るので、付いていないこと。**こちらでは補正しない**
3. `[groot] integrated GPU: …` の MemAvailable と cudaMemGetInfo の値
4. model・VLM の読み込み秒、`vlm_latency`、定常の周期
5. preflight の `--require-stereo` で落ちたら外してよい（単眼に落ちる）
6. PC2 の運営 package が `609f61d`（2026-09-25）以降か。これより古いと、頭の画像が潰れる（2）・WBC の起動時に腕が
   動く（Step 2）・joint lane が無い
7. go-live 前の揺れ / go-live から pick 開始までの時間 / 関節の到達判定（0.10 rad）/ :5557 のレート / 開 4.5 の `gripper_q`
8. `sender clock offset` と、adapter の `[stats]` で stale が 0 か
9. 準備動作の診断行の `speed=`（止まっているのに 0.08 を超えるなら受信時刻のゆらぎ）
10. PC2 の `md5sum ~/g1_bridge/robot/assets/g1_urdf/g1_29dof_with_hand.urdf` が `093c36ba3284c6cce5f2b62041626b79` か
    （PC2 を読むので運営に断ってから）。publish する手先の位置は、運営 IK と同じこの URDF の運動学で計算している
    （image に同梱、本体 #164）。違えば運営 IK の運動学が変わっているので、動かす前に運営に確かめる
11. joint lane（Step 4 の「指令の送り方を切り替えるとき」）: PC2 の package が `609f61d` 以降なら、`--boundary-lane joint` で
    1 run 試す。go-live 後に腕が開始姿勢へ動くか、adapter の `[stats]` に joint の受信が数えられ、clamp の log が多すぎないか。
    よければ会場の既定にする。pose のままなら、手首 roll の clamp（±0.88、本体 #164）を外すかもここで決める
    （運営 IK の手首 roll の上限は `a1af470` で無くなった。古い IK なら外すと腕が止まる）
12. Stage 0（pose lane）で、go-live の後に `[setup] measured lowered arm pose latched` が出て歩き出すか。腕を下ろす動きは
    手先で「着いた」と判定するが、その後の関節の範囲の検査（肘 1.00 + 0.05 など）で止まる可能性がある（運営 IK が同じ手先を
    別の関節角で作るため。PR #167 のレビュー）。`lowering the arms before the walk failed` や `not in the lowered walk
    envelope` で止まったら、どの関節が範囲を出たかを記録する（次の焼き直しで直す）
