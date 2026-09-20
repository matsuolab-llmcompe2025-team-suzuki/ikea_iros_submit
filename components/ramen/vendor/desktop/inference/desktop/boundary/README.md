# `boundary/` — 運営公式の契約パッケージ（vendor、無改変）

**このディレクトリの `.py` は 1 行も編集しないこと。** パッケージ自身がそう明記している:

> DO NOT MODIFY THIS PACKAGE. The bench runs against these exact formats; a local
> edit here will pass your tests and fail on the robot.
> — `__init__.py`

## 出所

| | |
|---|---|
| upstream | `https://github.com/iacevaltest/ikea_iros_submit`（運営公式 template） |
| 経由 | `matsuolab-llmcompe2025-team-suzuki/ikea_iros_submit` @ `a024ab3`（2026-09-12） |
| vendor 日 | 2026-09-20 |
| ファイル | `__init__.py` / `actions.py` / `cameras.py` / `states.py`（4 ファイル・636 行、bit 単位で一致） |

提出リポジトリ側も同じものを upstream から vendor している（`VENDOR_NOTES.md` 参照）。
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
| **`ActionSink` / `DecoupledSink`** | ✅ **使う** | `(T,25)` の出口。これが目的 |
| `CameraStream` | ❌ 使わない | JPEG を decode して **RGB に flip** する。我々は BGR のまま扱う必要があるため `perception/frame_source.py:ZmqFrameSource` を使う（同じ `:5555` を読むが、再 flip を避ける） |
| `StateStream` | ❌ 使わない | `:5557` は 50 Hz。`perception/lowstate_joint_source.py` の `rt/lowstate` 直読み（188 Hz 実測）の方が新鮮 |

ただし `cameras.py` / `states.py` は**仕様書として読む価値が高い**（wire format・
`g1_debug` prefix・「hand state は通常来ない」など）。消さずに残す。

## 更新するとき

upstream の差分を確認してから入れ替える。2026-09-20 時点で upstream HEAD は `2ae4eeb`
で、我々が持つ `7a4f071` との差は:

- `9f770d2` "Added ego_view_left and ego_view_right"（stereo キー追加）— **提出側は 8 月に非採用としたが、会場の camera server が publish していたのはこの 5 キー**だった
- `2ae4eeb` README の JetPack 訂正（本パッケージには影響なし）

採用可否は提出側（`VENDOR_NOTES.md`）と揃えること。**片方だけ更新すると契約がずれる。**
