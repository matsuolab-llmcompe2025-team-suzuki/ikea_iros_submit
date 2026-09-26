# `boundary/` — 運営公式の契約パッケージ（vendor、無改変）

**このディレクトリの `.py` は 1 行も編集しないこと。** パッケージ自身がそう明記している:

> DO NOT MODIFY THIS PACKAGE. The bench runs against these exact formats; a local
> edit here will pass your tests and fail on the robot.
> — `__init__.py`

## 出所

| | |
|---|---|
| upstream | `https://github.com/iacevaltest/iros_g1_orin_package`（運営公式 interface package） |
| package revision | `497f3ab93e5baa706311daebd31c7a9798258450`（2026-09-25、head calibration YAML対応） |
| boundary revision | `609f61d1224a1e952eb5f265664bf491b1b1f77b`（package内の`boundary/`はここから不変） |
| vendor 日 | 2026-09-26（4 Pythonファイルをpackage最新版とSHA-256比較済み） |
| ファイル | `__init__.py` / `actions.py` / `cameras.py` / `states.py`（4 ファイル・888 行、bit 単位で一致） |

提出リポジトリ（`ikea_iros_submit`）の `boundary/` も同じ upstream から `tools/update_organizer.sh` で
取り込む（出所は `ORGANIZER_SOURCE.txt`）。`tools/sync_ramen.sh` は両者が一致しないとコピーを止める。
**両経路が同一の契約実装を通る**ようにするため、こちらにも同じ形で置く。

## なぜ必要か

会場では、我々の action は **`:5556` に `(T,25)` task-space chunk として publish** し、
運営の `wbc_driver.py` がそれを購読して IK → WBC → ロボットに流す。この経路だけが
グリッパを動かせる（`rt/dex1/*` はラボ専用、詳細は
[`docs/setup/thor_pc2_environment.md`](../../../docs/setup/thor_pc2_environment.md) §9.6）。

その publish 側の実装（bind / 検証 / フレーミング）が `actions.py:DecoupledSink` に
既にある。**自前で同等品を書くと「テストは通るが実機で落ちる」** と upstream が警告
している当のものになるので、運営の実装をそのまま使う。

## このリポジトリで使うもの / 使わないもの

| | 使うか | 理由 |
|---|---|---|
| **`ActionSink` / `DecoupledSink`** | ✅ **使う** | `(T,25)` の出口（pose lane、会場の既定） |
| `JointSink` | 追加（2026-09-25） | `(T,22)` 関節角の出口（joint lane。運営 IK を通らない）。使い方は本体の送り口で決める |
| `CameraStream` | ❌ 使わない | JPEG を decode して **RGB に flip** する。我々は BGR のまま扱う必要があるため `perception/frame_source.py:ZmqFrameSource` を使う（同じ `:5555` を読むが、再 flip を避ける） |
| `StateStream` | ✅ 会場で使う | 公式契約は `:5557` を唯一のstate入口とする。labのSDK経路だけ`rt/lowstate`を使う |

ただし `cameras.py` / `states.py` は**仕様書として読む価値が高い**（wire format・
`g1_debug` prefix・「hand state は通常来ない」など）。消さずに残す。

## 更新するとき

upstream の差分を確認してから4つのPythonファイルを一式で入れ替える。
採用可否は提出側（`VENDOR_NOTES.md`）と揃えること。**片方だけ更新すると契約がずれる。**
