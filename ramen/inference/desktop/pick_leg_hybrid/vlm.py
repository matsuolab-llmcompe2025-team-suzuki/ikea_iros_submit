"""Qwen3-VL による区間境界の二択判定 (Issue #136、determinded.md D-5 / D-6)。

# 何を訊くか

区間列は VLA → MP → VLA に固定されているので、VLM は順序を決めない。
**「次の区間に入ったか」の 1 ビットだけ**を答える。使うのは 1→2 の境界のみ
(2→3 は MP 自身が到達を判定できるので VLM を使わない)。

# 参照画像との強制二択

「これに近いか」の絶対判定ではなく、**遷移前と遷移後の参照画像を両方見せて
どちらに近いかを選ばせる**。VLM は絶対判定より強制二択の方が安定する。
参照画像はデータセットから取り、現在画像と**同じ overlay 描画**をかける。

# overlay

YOLO-OBB の枠を画像に描き込んでから渡す (D-7)。脚と両手がどれかを VLM が
自力で当てる必要がなくなる。色は固定なので、対応をプロンプトで教える。

# 未検証

repo の GR00T overlay は overlay 済み画像で **fine-tune した**モデルだが、
ここでは overlay を見たことのない Qwen3-VL にゼロショットで渡す。
効くかどうかは `evaluate/pick_leg_hybrid/` のオフライン評価で測る (planning.md P-11)。

# 実行先

Thor コンテナ上の vLLM (OpenAI 互換 endpoint) を想定。`requests` は runtime
依存なので lazy import する。
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

# overlay の色 (policies/ramen_ori.py:CLASS_COLORS_BGR と一致)。
# プロンプトで VLM に教えるための表記。
OVERLAY_LEGEND: dict[str, str] = {
    "leg": "緑",
    "leg_tip": "黄",
    "hand_right": "オレンジ",
    "hand_left": "マゼンタ",
}

#: overlay 対象の class_id (yolo_obb dataset の names 準拠)。
#: 0 workspace / 1 leg / 2 leg_tip / 3 hole / 4 table_top / 5 hand_right / 6 hand_left
OVERLAY_CLASS_IDS: frozenset[int] = frozenset({1, 5, 6})

#: 既定のシステム指示。**「まだ」寄りに倒してある** (D-5 の非対称性)。
DEFAULT_SYSTEM_PROMPT: str = (
    "あなたはロボットの作業を監視する判定器です。"
    "画像には物体検出の枠が色分けで描かれています: "
    "テーブルの脚=緑、脚の先端=黄、右手=オレンジ、左手=マゼンタ。\n"
    "質問には必ず JSON だけで答えてください: {\"answer\": 0 または 1}\n"
    "確信が持てない場合、迷った場合、判断材料が足りない場合は必ず 0 を返してください。"
    "0 は「まだ起きていない」を意味します。"
)

#: 1→2 の質問文。対象は IKEA の**テーブルの脚** (椅子ではない)。
#: データセットの言語プロンプト "pick table leg" に揃えてある。
GRASP_QUESTION: str = (
    "参考画像 A は、ロボットがまだテーブルの脚を掴んでいない状態です。\n"
    "参考画像 B は、ロボットが右手でテーブルの脚を掴んだ状態です。\n"
    "最後の画像が現在の状態です。その前の画像は少し前の状態です。\n\n"
    "現在の状態は A と B のどちらに近いですか。"
    "B に近い（右手がテーブルの脚を掴んでいる）なら 1、"
    "A に近い（まだ掴んでいない）なら 0 を返してください。"
)

#: 区間 1 (1→2 を訊く区間) の説明文。本番の問い合わせと、起動時の自己確認・慣らしで同じものを渡す。
GRASP_PHASE_TEXT: str = "右手でテーブル脚へ接近し把持する"


@dataclass(frozen=True)
class VlmConfig:
    """VLM 呼び出しの設定。

    Attributes:
        endpoint: OpenAI 互換の chat completions URL。
        model: モデル名 (vLLM に渡す識別子)。
        timeout_sec: 1 回の呼び出しの上限。超えたら「まだ」扱い。
        max_tokens: 出力は JSON 1 行なので小さくてよい。
        temperature: 同じ入力に対する揺れを抑えるため 0。
        history_frames: 現在画像より前に何枚渡すか。
        jpeg_quality: 画像を JPEG にするときの品質。
    """

    endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    timeout_sec: float = 5.0
    max_tokens: int = 32
    temperature: float = 0.0
    history_frames: int = 2
    jpeg_quality: int = 85
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if self.timeout_sec <= 0.0:
            raise ValueError(f"timeout_sec must be > 0, got {self.timeout_sec}")
        if self.history_frames < 0:
            raise ValueError(
                f"history_frames must be >= 0, got {self.history_frames}"
            )
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be 1..100, got {self.jpeg_quality}")


@dataclass
class VlmAnswer:
    """1 回の呼び出しの結果。

    `value` が None = 判定不能 (通信失敗 / timeout / 壊れた出力)。
    呼び出し側は None を「まだ」として扱う (安全側)。
    """

    value: Optional[int]
    raw: str = ""
    latency_sec: float = 0.0
    error: str = ""

    @property
    def is_yes(self) -> bool:
        """「次の区間に入った」と答えたか。判定不能なら False。"""
        return self.value == 1


def parse_answer(text: str) -> Optional[int]:
    """モデル出力から 0 / 1 を取り出す。取れなければ None。

    JSON を素直に読むのを第一とし、前後に説明が付いた場合に備えて
    ``{...}`` を抜き出す fallback、それも駄目なら裸の 0/1 を拾う。

    Args:
        text: モデルの出力文字列。

    Returns:
        0 / 1、または解釈できなければ None。
    """
    if not text:
        return None
    candidates = [text.strip()]
    brace = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and "answer" in obj:
            return _as_bit(obj["answer"])
    stripped = text.strip()
    if stripped in ("0", "1"):
        return int(stripped)
    return None


def _as_bit(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    return None


def encode_jpeg_base64(frame_bgr, quality: int = 85) -> str:
    """BGR uint8 frame → base64 JPEG (data URI の中身部分)。

    Args:
        frame_bgr: (H, W, 3) uint8 BGR。
        quality: JPEG 品質。

    Returns:
        base64 文字列。

    Raises:
        ValueError: encode に失敗した場合。
    """
    # lazy: cv2 は runtime env にしか無い
    import cv2

    ok, buf = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise ValueError("failed to JPEG-encode frame for the VLM request")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_messages(
    question: str,
    images_b64: Sequence[str],
    *,
    current_phase_text: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict]:
    """OpenAI 互換 chat messages を組み立てる。

    画像は渡された順に並べる。呼び出し側の並びは
    ``[参考A, 参考B, 過去…, 現在]`` を想定 (質問文がその順を前提にしている)。

    Args:
        question: ユーザ側の質問文。
        images_b64: base64 JPEG の並び。
        current_phase_text: 現在フェーズの説明 (文脈として渡す。単調性の保証には
            使わない — 保証は状態機械側、D-9)。
        system_prompt: システム指示。

    Returns:
        chat completions の messages。
    """
    content: list[dict] = []
    count = len(images_b64)
    for index, b64 in enumerate(images_b64):
        if count == 1:
            label = "現在画像"
        elif index == 0:
            label = "参考画像 A（遷移前）"
        elif index == 1:
            label = "参考画像 B（遷移後）"
        elif index == count - 1:
            label = "現在画像（判定対象）"
        else:
            label = f"少し前の画像 {index - 1}"
        # Do not rely on positional image ordering alone. Explicit labels are
        # important for OpenAI-compatible VLM servers, which otherwise receive
        # an undifferentiated run of image tokens before the question.
        content.append({"type": "text", "text": label})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    text = question
    if current_phase_text:
        text = f"現在の区間: {current_phase_text}\n\n{question}"
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


class VlmBoundaryClient:
    """OpenAI 互換 endpoint 越しに二択を訊くクライアント。

    通信失敗・timeout・壊れた出力はすべて `VlmAnswer(value=None)` にする。
    **例外は呼び出し側に投げない** — 制御ループを VLM の障害で止めないため。
    呼び出し側は None を「まだ」として扱う (安全側、D-5)。
    """

    def __init__(self, cfg: Optional[VlmConfig] = None, _session=None) -> None:
        self.cfg = cfg or VlmConfig()
        self._session = _session  # test から差し替える口

    def ask(
        self,
        question: str,
        images_b64: Sequence[str],
        *,
        current_phase_text: str = "",
    ) -> VlmAnswer:
        """二択を 1 回訊く。

        Args:
            question: 質問文。
            images_b64: base64 JPEG の並び。
            current_phase_text: 現在フェーズの説明。

        Returns:
            `VlmAnswer`。判定不能なら `value=None`。
        """
        import time

        messages = build_messages(
            question,
            images_b64,
            current_phase_text=current_phase_text,
            system_prompt=self.cfg.system_prompt,
        )
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        t0 = time.monotonic()
        try:
            text = self._post(payload)
        except Exception as exc:  # noqa: BLE001 — 制御ループを止めない
            return VlmAnswer(
                value=None,
                latency_sec=time.monotonic() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
        return VlmAnswer(
            value=parse_answer(text),
            raw=text,
            latency_sec=time.monotonic() - t0,
        )

    def _post(self, payload: dict) -> str:
        session = self._session
        if session is None:
            # lazy: requests は runtime env にしか無い
            import requests

            session = requests
        resp = session.post(
            self.cfg.endpoint, json=payload, timeout=self.cfg.timeout_sec
        )
        try:
            resp.raise_for_status()
        except Exception as exc:
            body = str(getattr(resp, "text", "")).strip()
            detail = body[:500] if body else str(exc)
            raise RuntimeError(f"VLM HTTP request failed: {detail}") from exc
        body = resp.json()
        return str(body["choices"][0]["message"]["content"])
