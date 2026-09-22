# GR00T on the decoupled boundary — deploy guide (Team RAMEN)

RAMEN-Ori (`README.md` / `manifest.yaml`) とは別トラックで、**GR00T 系 policy を decoupled
lane の `(T,25)` task-space に載せて** 運営評価に出すための実装と手順。

- 実装: `components/ramen/`
- 提出 manifest: `manifest.yaml`（2026-09-22 に `manifest.groot.yaml` から統合）
- image: `ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor:20260920-groot-orch-eeorigin`
- 技術詳細 / 判断ログ: `VENDOR_NOTES.md` §「GR00T-pick 提出トラック」「53D skills」ほか

---

## 1. できること (run modes)

image は1つで、`RAMEN_POLICY` env で挙動を切替える。全て decoupled lane、`(T,25)` 出力。

| `RAMEN_POLICY` | 内容 | 追加 env |
|---|---|---|
| `groot_orchestrator` (**推奨**) | 原さんの orchestrator を full 駆動。perception(YOLO)+dwell+is_complete で **5 skill を自動遷移** (「一通り全部」) | `HF_TOKEN` |
| `groot_53d_real` | 53D LeRobot GR00T 単体 | `HF_TOKEN`, `RAMEN_VARIANT=<skill>` |
| `groot_pick_real` | pick (38D Isaac native) 単体 (image 既定) | `HF_TOKEN` |
| `stub` / 未設定 | 参照 hold-still Policy (conformance 既定) | — |

`RAMEN_VARIANT`: `groot_rotate_leg_200k` / `groot_insert_leg_200k` / `groot_overlay` /
`groot_flip_table_n17_2_baseline`。

### 運営 run 例
```bash
docker run --rm --runtime nvidia --network host \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e RAMEN_POLICY=groot_orchestrator \
  -e HF_TOKEN=<Team-RAMEN HF read token> \
  ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor:20260920-groot-orch-eeorigin
```
- `NVIDIA_DISABLE_REQUIRE=1` は Thor で GPU passthrough に必須 (onboarding Finding 1)。
- weights (GR00T / YOLO) は image に焼かず **runtime に HF から取得** → `HF_TOKEN` 必須。

---

## 2. アーキテクチャ

```
organizer WBC ──(T,25)── boundary/actions.py(運営所有) ──┐
organizer cams/state ── boundary/{cameras,states}.py ──┤
                                                        ▼
                             components/server.py Policy.act(obs)
                                                        │
     RAMEN_POLICY=groot_orchestrator ─────────────────┐ │
                                                       ▼ ▼
        OrchestratorTaskspacePolicy → OrchestratorDriver
          obs → I/O adapter (orchestrator_io) 注入
             ├ BoundaryJointStateSource  (body_q → joint_state)
             ├ BoundaryDex1StateSource
             ├ BoundaryWristSource ×2
             └ InterceptorActuator ×3 (waist/arm/hand の action 捕捉)
          → vendored 原's Orchestrator.tick(FrameData)
             perception(YOLO-OBB) → cleaner → transition(enter_check)
             → dispatcher.step → active VlaSkill(GR00T policy) → 19D
          → assemble_19d → body29 → taskspace_adapter → (T,25)
```

**adapter (`taskspace_adapter.py`)**: 19D/38D → `(T,25)`。arm14 → FK(`g1_urdf_fk.G1WristFK`,
pure-numpy) → EE pos+quat / hand → `[-1 open,+1 closed]` / waist3 → torso_rpy(param) /
navigate・base = 0。**scipy 不要** (matrix→quat は Shepperd 法)。

**2 lerobot env** (checkpoint 形式が違うため):
- pick (38D Isaac native) = image main の lerobot **0.6.0** (`RAMEN_WORKER_PYTHON=python3`)
- 53D (REAL_G1_RELATIVE_EEF) = `/opt/venv-groot53` の lerobot **0.6.1**
  (`RAMEN_WORKER_PYTHON_53D`)。takada/suzuki checkpoint は custom load (raw_config +
  streaming shards + tied-embedding 復元) が必要 = vendored `groot.py` が担う。

**skill → model** (leg round)。**正本は `policy_config.yaml` の
`default_variant_by_skill`** — driver はそこを読む (Issue #148)。下表は転記なので、
食い違ったら config が正しい。

| skill | policy_config variant | 形式 |
|---|---|---|
| rotate_table_base | `rotate_table_base_ramen_ori_141_c32_state_dropout` | RAMEN-Ori |
| pick_table_leg | `groot_pick_legs_v1` (subdir `checkpoint-40000`) | 38D |
| insert_table_leg | `groot_insert_leg_200k` | 53D |
| rotate_leg_to_tighten | `groot_rotate_leg_200k` | 53D |
| flip_table | `groot_flip_table_n17_2_baseline` | 53D |

⚠️ `RAMEN_POLICY=groot_pick_real` (単体 pick の debug 経路) は別実装
(`components/ramen/groot_worker.py`) で、contract に焼かれた **ver2-lora** を使う。
大会経路 (`groot_orchestrator`) とは別の ckpt なので、比較するときは注意。

---

## 3. ファイル構成 (`components/ramen/`)

| file | 役割 |
|---|---|
| `policy.py` | boundary Policy 群 (GrootPickTaskspacePolicy / Groot53Backend / OrchestratorTaskspacePolicy / HoldPoseBackend …) |
| `taskspace_adapter.py` | 19D/38D → `(T,25)` (FK+quat+hand+下半身)、pure-numpy |
| `g1_urdf_fk.py` | G1 wrist FK (vendored、pure-numpy) |
| `groot_worker.py` | pick worker client (self-contained、直接 worker protocol、raw 38D) |

| `orchestrator_io.py` | boundary↔orchestrator の I/O adapter (state/wrist 注入、interceptor、19D 再構成) |
| `orchestrator_driver.py` | full orchestrator driver (perception+skills+遷移 build、act→tick→(T,25)) |
| `smoke_*.py` | GPU smoke (pick / 53d / orchestrator) |
| `vendor/` | 依存を self-contained 化した複製 (下記) |

**`vendor/`**:
- `vendor/inference/desktop/upper_policy/` … pick worker の contract / protocol / worker script
- `vendor/desktop/` … **desktop inference runtime 全体** (orchestrator / dispatcher / 5 VlaSkill /
  actuators / perception[cleaner/stream/yolo_obb] / skill_planner / policies + configs)。
  53D と orchestrator が使う。`groot.py` に spawn patch (pixi→`RAMEN_WORKER_PYTHON_53D`)。
- ~~`vendor/groot53/`, `vendor/dex1/`~~ … 53D self-contained 版の WIP。**2026-09-22 に削除**
  (誰からも起動されず、生きている `Groot53Backend` は vendored の `build_state_from_raw`
   を使っている。残すと「直せば動く」と誤解される / Issue #7)。

> vendored code は iros_2026_ramen からの複製。上流更新時は同期が要る (各 file 冒頭に出典)。

---

## 4. build & push (arm64)

x86 host から QEMU cross-build (機構検証済)。実行は sm_110 GPU (Thor) が必要。

```bash
# NGC (base pull) と GHCR (push) に login (自分の端末、token は会話に貼らない)
docker login nvcr.io -u '$oauthtoken'          # NGC API key
docker login ghcr.io -u <github-user>          # GitHub PAT (write:packages)

docker buildx build --builder armbuilder --platform linux/arm64 \
  -f docker/Dockerfile.thor.groot \
  -t ghcr.io/matsuolab-llmcompe2025-team-suzuki/ikea-thor:<tag> \
  --push .
```

- base = `nvcr.io/nvidia/pytorch:25.12-py3` (numpy 2.x ABI torch sm_110 / CUDA13。25.08 は numpy 1.x ABI で from_numpy が壊れるため不可)。
- image 内訳: image main = lerobot 0.6.0 + ultralytics + pyyaml、`/opt/venv-groot53` = lerobot 0.6.1。
- 新規 package を作らず **既存 `ikea-thor` の tag** に push (運営に伝達済の名前を維持、
  `:onboarding` tag は温存)。push 後 digest を `manifest.yaml` に反映（本体 repo の venue_runbook / offline_bundle も同時に）。

---

## 5. 検証状況

| 項目 | 状態 |
|---|---|
| adapter (19D/38D→(T,25)) 単体 | ✅ unit test |
| pick 実推論 → (T,25) | ✅ 実 GPU (self-contained) |
| 53D (rotate_leg / flip) 実推論 → (T,25) | ✅ 実 GPU |
| full orchestrator driving → (T,25) | ✅ 実 GPU (skill=rotate_table_base の実推論が tick 毎に走る) |
| arm64 image build (2 lerobot env + YOLO) | ✅ QEMU build+push |
| conformance decoupled (stub) | ✅ (onboarding) |
| **Thor 実機で ready + conformance + (T,25)** | ❌ **未** (sm_110 GPU 必須、x86/QEMU は build のみ) |
| 実画像で 4-skill 遷移 (rotate→pick→insert→rotate_leg) | ❌ 未 (zeros 画像では YOLO 未検出=遷移せず=正常) |

---

## 6. 既知の gap / 要対応

1. **Thor 実機検証**: `RAMEN_POLICY=groot_orchestrator python3 components/server.py` の
   worker ready + conformance + (T,25) を実 Thor で確認する (最重要ゲート)。
2. **Q1: EE frame**: organizer の decoupled IK が期待する EE 基準 frame (pelvis? torso?
   tool offset 有無) が vendored template に明示なし。現状 pelvis 前提。ずれると motion が
   狂う → 運営確認。`taskspace_adapter` に `ee_frame_transform` の後付け穴あり。
3. **HF_TOKEN**: GR00T / YOLO weights は private Team-RAMEN repo → `-e HF_TOKEN=<read token>` 必須。
4. **GHCR read 権限**: `ikea-thor` package を運営が pull できる状態か (onboarding Finding 2)。
5. **NVIDIA_DISABLE_REQUIRE=1**: Thor docker run に必須 (Finding 1)。
6. manifest は `manifest.yaml` 1 本（トラック分岐は 2026-09-22 に解消）。
   運営にどちらを評価してもらうか要調整。

---

## 7. 開発機での smoke (dev)

```bash
cd /datadrive2/iros_2026_ramen   # worker venv / YOLO weight があるリポ
RAMEN_WORKER_PYTHON=<.../model/subtask_policy_training/.venv/bin/python> \
RAMEN_WORKER_PYTHON_53D=<.../inference/desktop/.pixi/envs/default/bin/python> \
pixi run -e runtime python /datadrive2/ikea_iros_submit/components/ramen/smoke_orchestrator.py
```
pick / 53d 単体は `smoke_groot_pick.py` / `smoke_groot_53d.py`。
