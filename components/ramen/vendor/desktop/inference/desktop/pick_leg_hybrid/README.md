# pick_leg_hybrid — VLM planner + VLA / Motion planning ハイブリッド

`pick_table_leg` を 3 区間に分け、VLA の流れの中に Motion planning を 1 区間だけ
差し込む構成 (Issue #136)。

設計判断の経緯は [`tmp/execution-plan/determinded.md`](../../../tmp/execution-plan/determinded.md)、
未決事項は [`planning.md`](../../../tmp/execution-plan/planning.md)、
残作業は [`todo.md`](../../../tmp/execution-plan/todo.md)。

## 区間

| # | 区間 | 実行系 | 境界の判定 |
|---|---|---|---|
| 1 | 脚に近づいて右手で掴む | VLA | 把持インターロック **AND** VLM |
| 2 | 右手で脚を左手の位置まで運ぶ | Motion planning | MP 自身の到達判定 (実測 EE pose) |
| 3 | 左右で持ち替え、insertのframe-0姿勢へ接続 | VLA（既定）またはrule-based FSM | — (最終区間) |

境界は 2 箇所。**VLM を使うのは 1→2 だけ**で、2→3 は MP が自分のゴールに着いた
ことを自分で判定できるので知覚を使わない。

## module

| file | 役割 |
|---|---|
| `phases.py` | 3 区間と**後戻りしない**状態機械。単調性はここで保証する |
| `interlock.py` | Dex1 の把持インターロック (追従誤差 AND 値の帯) |
| `vlm.py` | Qwen3-VL クライアント、プロンプト、JSON パース |
| `boundary.py` | 1→2 の境界検出 (インターロック AND VLM、連続確認、呼び出し間引き) |
| `phase2.py` | 区間 2 の補間と boundary `(25,)` 行の組み立て |
| `overlay_input.py` | YOLO-OBB overlay を描いて base64 JPEG にする |
| `controller.py` | 3 区間を管理して実行系に振り分ける本体 |
| `config.py` | `configs/pick_leg_hybrid.yaml` の読み込み |
| `phase3_rule.py` | 右手再把持からinsert frame-0へ片腕ずつ接続するfail-closed FSM |
| `real_skill.py` | 実機用adapter。Cartesian区間をG1 IK、insert接続をarm14へ変換し既存limiterへ渡す |

Phase 1–3の数値は`configs/pick_leg_hybrid.yaml`、次skillへの接続目標は正本の
`lower_policy/configs/skill_config.yaml`。Python側に運用値を置かない
(CLAUDE.mdの「Skill numerics live in YAML」)。

## 使い方

`PickLegHybridController.step(obs)` を毎 tick 呼ぶ。VLA 本体と FK は注入する。

```python
from inference.desktop.pick_leg_hybrid.config import load_yaml
from inference.desktop.pick_leg_hybrid.controller import PickLegHybridController

cfg = load_yaml()
controller = PickLegHybridController(
    cfg,
    vla_step=my_groot_step,   # obs -> (T, 25)
    ee_probe=my_ee_probe,     # obs -> (左 EePose, 右 EePose)
)
result = controller.step(obs)   # result.action = (T, 25)
```

`vla_step` を注入にしているのは、GR00T pick_legs が Python 3.12 の別プロセス
worker 経由で動き、生成コストが大きいため。**worker は区間ごとに起動しない**
(determinded.md D-1 制約 7)。

## 競技boundaryと実機G1の違い

提出は boundary の `decoupled` lane で、腕は**手先の位置と姿勢**を送る
(`(T,25)` の `[4:18]`)。関節角への変換は運営側の IK が行う。こちらが持つのは
FK だけで、それは `perception/g1_urdf_fk.G1WristFK` にある。

競技boundaryではそのためIKは不要。一方、`run_skill` の実機経路はarm14の絶対
関節角を要求する。`RealPickLegHybridVlaSkill` は同じCartesian補間を
`G1ArmKinematics`で左右7関節ずつへ変換する。片腕だけIKに失敗した場合も部分指令を
送らず、直前の安全な両腕targetを保持する。腰と脚は常にRegular Mode所有のまま。

## 実機runner

VLM endpointを起動し、次のread-only preflightを先に行う。参照画像、endpoint、
served model、4 camera、joint/Dex1、モデルforwardを検査するが指令は送らない。

```bash
pixi run -e runtime python -m evaluate.model_evaluation.runners.run_skill \
  --skill pick_table_leg \
  --variant groot_pick_legs_v1 \
  --interface enx58278cbf8be0 \
  --max-seconds 30 \
  --pick-leg-hybrid \
  --pick-leg-vlm-endpoint http://THOR_HOST:8000/v1/chat/completions \
  --pick-leg-vlm-model Qwen/Qwen3-VL-8B-Instruct
```

実機では上記に `--actuate --use-real-hand --no-dispatch-waist` を追加する。
`--pick-leg-hybrid`を付けない既存pick VLAと他skillの経路は変更されない。

Phase 3もルールベースにする比較版は、さらに次を追加する。

```bash
--pick-leg-phase3-executor rule_based
```

この版はPhase 2終点から、左把持、右解放、左での提示、右再把持、左解放、
左腕をinsert初期姿勢へ退避、脚を保持する右腕をinsert初期姿勢へ移動、Dex1を
insert初期指令へ整える順に実行する。insert目標は値を複製せず、runnerが使用中の
`skill_config.yaml:skills.insert_table_leg.initial_pose`から読む。把持はDex1の
実測state-command差、解放は実測開度、Cartesian腕stageは実測FK、insert接続は
実測関節角で終了判定する。両手保持中に腕を動かさず、腕を動かすstageも常に
片腕だけである。insert接続では終点を共通`MotionLimiter`へ直接渡し、速度・加速度・
制動を一層だけで生成する（二重補間しない）。insert初期姿勢の到達には、既存の
実機pre-motionと同じ0.10 radの実測許容に加え、最終指令との差0.02 rad以下、
実測速度0.05 rad/s以下を連続確認する。したがってLimiter途中や動作中を到達扱い
せず、arm_sdkの定常的な重力・コンプライアンス偏差だけを許容する。
全閉指令は接触検出だけに使い、
把持確認後は実測開度から`grasp_preload_rad`だけ閉じたtargetへ切り替える。これにより
state-command interlockを維持しつつ、物体へ全閉指令を連続してDex1を過熱させない。
実機評価中にstage安全判定が失敗した場合はarm_sdkを即時解放せず、最後の安全targetを
保持して操作者が物体を支えたことをEnterで確認してからcontrolled releaseする。
単体`run_skill`でexecutorを省略した場合の`vla`は従来挙動を保持する。本番の
stage実行は有限完了契約を必須にするため`rule_based`を明示する (既定値がこれ)。

## 本番 Stage 1〜4 への差し替え

本番のstage実行は`inference.desktop.entrypoint`が入口で、`--pick-leg-hybrid`を
付けるとStage 1〜4の`pick_table_leg`が上記の有限`rule_based` Phase 3版になる。
pick完了時には`skill_config.yaml:skills.insert_table_leg.initial_pose`の腕14Dと
Dex1 2Dへ実測到達し、その後だけ`insert_table_leg`へ遷移する。hybrid全体が30秒を
超える、または任意の実測安全条件に失敗した場合は次skillへ進まず、最後の安全target
を保持する。操作者が物体と腕を支えてEnterを押した後にだけarm_sdkを
controlled releaseする。

```bash
# Stage 1〜4 を 1 process で通す (途中から始めるときは --phase3-start-stage を変える)
pixi run -e runtime python -m inference.desktop.entrypoint \
  --interface enx58278cbf8be0 \
  --phase3-full --phase3-start-stage 1 --phase3-end-stage 4 \
  --actuate --use-real-waist --use-real-hand \
  --pick-leg-hybrid
```

policy variantは`policy_config.yaml:default_variant_by_skill`から自動で入るので
書かなくてよい。別のckptで試すときだけ`--policy-variant-*`で上書きする
(CLIが勝つ)。旧learned-only pickを再現するときは`--pick-leg-hybrid`を外す。

### 単一RTX 5090でVLMとGR00Tを共存させる場合

30B-A3B BF16/FP8はGR00Tと同じ32GB GPUへ同時常駐できない。別GPU hostが無い
場合は、公式`Qwen/Qwen3-VL-8B-Instruct`をvLLMでVRAM 64%に制限して配信し、
runnerへ`--pick-leg-vlm-model Qwen/Qwen3-VL-8B-Instruct`を明示する。8Bは30Bと
同一モデルではないため、実験記録には必ずserved model idを残す。2026-09-20の
RTX 5090実測ではVLM約19.5GB、GR00T同時ロード込みpreflightが成功した。

ローカルserverは`run_local_vlm_server.sh`で起動する。production requestは参照2枚、
履歴2枚、現在1枚の合計5枚なので、vLLMの画像上限を4以下にしてはいけない。
preflightも同じ5枚を実際に送信し、この設定不整合を指令前に検出する。

## 数値の出所

`configs/pick_leg_hybrid.yaml` の値は `Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1`
(2114 episode / 975,291 frame) から算出した。再算出:

```bash
python evaluate/pick_leg_hybrid/derive_thresholds.py
```

主な実測:

- 指令帯ごとの追従誤差 — 開 / 中間は ±0.013 で追従、HOLD 帯だけ +0.35。
  `follow_error_min = 0.2` が両者を明確に分ける
- **追従誤差だけでは足りない** — 閉じ始めの過渡でも追従誤差は出る。
  `err > 0.2` が最初に立つ瞬間の `hand_state` は median 4.46 (ほぼ全開) で、
  HOLD 帯に入るのは 0.6% のみ。値の帯との AND が要る
- 区間 3 入口の EE pose (2094 episode の median + quaternion 平均)
- 区間 2 の長さ median 81 frame = 2.70 秒 → `duration_sec`

## 評価

```bash
# 基準線 (VLM 不要、numpy + pyarrow だけで走る)
python evaluate/pick_leg_hybrid/eval_boundary.py --mode interlock --limit 300
```

実測 (300 episode): 検出漏れ 3、**早まり 0 件**、遅れ median +30 frame = 1.00 秒。
遅れは `confirm_count = 2` × `min_interval_sec = 1.0` のコストそのもの。
早まりは偽陽性で回復不能、遅れはただの遅延なので、この非対称は狙いどおり。

VLM 込みの評価 (`--mode overlay` / `--raw`) は **未実装**。VLM endpoint と
映像 frame の decode が要る (todo.md 参照)。

## EE frame

`configs/pick_leg_hybrid.yaml` の `phase2.goal_*` は **`G1WristFK` 定義**で
書いてある (`ee_frame: g1_wrist_fk`)。データセット記録の `ee_state` とは
**左 99 mm / 右 129 mm** ずれる。

競技boundaryの外部IKが期待するframeは未確定のまま。ただし実機adapterは外部IKを
使用せず、目標値を導出したものと同じ`G1WristFK` tool offsetを自前G1 IKにも設定する。
固定終点は左右とも位置誤差1 mm未満で解けることをunit testで確認する。

## test

```bash
pixi run -e runtime python -m pytest inference/desktop/pick_leg_hybrid/tests/ -v
```

実 VLM はunit testでは呼ばない (クライアントを差し替えて戻り値を制御する)。
