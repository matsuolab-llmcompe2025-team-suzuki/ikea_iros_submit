# Team RAMEN — IKEA IROS 提出物

## 運営の責任範囲

運営が用意するもの。**ここに書いたもの以外は、すべて Team RAMEN のもの。**

### 会場（ロボットに載っている PC2 の上）

| もの | やること |
|---|---|
| カメラ bridge | `:5555` に head と手首カメラの JPEG を publish する |
| 状態 bridge | `:5557` に `body_q`・`base_quat` を 50 Hz で publish する（2026-09-21 以降の bridge は `gripper_q` も載せる。CONTRACT には無い項目） |
| WBC adapter | 我々が bind した `:5556` に接続し、`(T,25)` を IK で関節の目標にして WBC に渡す。手の列は Dex1 へ relay する |
| WBC 本体 | 全身の制御 |
| e-stop | 我々のコードを通らずにモータを止める |

これらを誰が起動するかは、運営の README（"You do not run any of this"）と
RUNBOOK（"you run the whole pipeline yourself"）で食い違っている。

### この repo の中（運営の template から取り込む。手で変更しない）

`tools/update_organizer.sh` で運営の template から上書きする。取り込んだ commit は
`ORGANIZER_SOURCE.txt` に残る。

| path | 役割 |
|---|---|
| `boundary/` | 3 socket（`:5555` / `:5557` / `:5556`）の契約の実装 |
| `mocks/` | ロボット無しで試すための偽 PC2（`mock_orin.py`）と偽 WBC（`mock_wbc.py`） |
| `conformance.py` | 提出前の配線確認（提出の条件）。`components/server.py` と `components/client.py` をこの名前で起動する |
| `requirements.txt` | template の依存 |

### 正本

- template: https://github.com/iacevaltest/ikea_iros_submit （取り込んだ commit は `ORGANIZER_SOURCE.txt`）
- interface package: https://github.com/iacevaltest/iros_g1_orin_package @ `47f4e1b`
  （`docs/CONTRACT.md`。doc とコードが食い違ったらコードが正）
