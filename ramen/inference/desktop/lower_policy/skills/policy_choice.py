"""R の後の待ちで、skill の model を既定と候補から選ぶ (Issue #188 ② 段 3)。

会場の 1 Stage の run (`--gpu-models plan`) で、R → `腕保持・ハンド全開` の画面に既定と候補を並べる。

- R = いまの model のまま初期姿勢へ (今までと同じ)
- 0 = 既定、1・2… = 候補 (`policy_config.yaml` の `alternatives_by_skill` の順)。
  切り替えてから初期姿勢へ戻る (その後はいつもの Enter で始める)
- 読み込み中・読み込みに失敗した候補は選べない (押しても待ち続ける)
- 選んだ model は、次に変えるまで R をくり返しても同じまま。各 skill の 1 回目は既定

skill の名前はそのままで、dispatcher の中身だけを差し替える (`replace_fn`)。候補の skill は
既定と同じ開始姿勢でなければならない (entrypoint が起動時に確かめる)。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

#: 読み込みの状態の表示 (`LoadPlan.model_state`)。
_STATE_TEXT = {
    "loaded": "読み込み済み",
    "loading": "読み込み中",
    "failed": "読み込み失敗",
    "released": "解放済み",
}


@dataclass(frozen=True)
class ChoiceOption:
    """選べる model 1 つ。

    Attributes:
        variant: `policy_config.yaml` の variant 名。
        skill: 組み立て済みの skill (名前は既定の skill と同じ)。
        note: 同じ variant を別のやり方で使うときの添え書き (pick の「hybrid」「hybrid なし」)。
        requires: この候補を選ぶのに、ほかに読めている必要がある物 (`state_fn` に渡す名前。
            既定が hybrid でない run の hybrid の候補なら "vlm")。
    """

    variant: str
    skill: Any
    note: str = ""
    requires: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.variant}（{self.note}）" if self.note else self.variant


class PolicyChoice:
    """1 skill の既定と候補。R の後の待ちの関所 (`OperatorConfirmationHoldSkill`) が使う。

    Args:
        skill_name: skill 名。
        options: 先頭が既定、残りが候補。

    entrypoint が後から付けるもの:
        replace_fn: 選んだ skill を dispatcher に差し替える関数。
        state_fn: variant 名 → 読み込みの状態 ("loaded" / "loading" / "failed" / "released")。
    """

    def __init__(self, skill_name: str, options: Sequence[ChoiceOption]) -> None:
        if len(options) < 2:
            raise ValueError(
                f"{skill_name}: a choice needs the default and at least one candidate"
            )
        if len(options) > 10:
            raise ValueError(f"{skill_name}: at most 9 candidates fit the keys 1-9")
        labels = [option.label for option in options]
        if len(set(labels)) != len(labels):
            raise ValueError(f"{skill_name}: a variant is listed twice: {labels}")
        for option in options:
            if getattr(option.skill, "name", None) != skill_name:
                raise ValueError(
                    f"{skill_name}: the skill for {option.variant} is named "
                    f"{getattr(option.skill, 'name', None)!r}"
                )
        self.skill_name = skill_name
        self._options = list(options)
        self._current = 0
        self._lock = threading.Lock()
        self.replace_fn: Optional[Callable[[Any], None]] = None
        self.state_fn: Optional[Callable[[str], str]] = None

    @property
    def options(self) -> tuple[ChoiceOption, ...]:
        return tuple(self._options)

    @property
    def keys(self) -> tuple[str, ...]:
        """選ぶキー (0 = 既定、1… = 候補)。"""
        return tuple(str(index) for index in range(len(self._options)))

    @property
    def current_variant(self) -> str:
        with self._lock:
            return self._options[self._current].variant

    @property
    def current_label(self) -> str:
        with self._lock:
            return self._options[self._current].label

    def labels(self) -> dict[str, str]:
        """画面の「操作」に出す名前 (R と数字)。"""
        labels = {"r": f"R 初期姿勢へ（{self.current_label} のまま）"}
        for index, option in enumerate(self._options):
            if index == 0:
                note = "・".join(part for part in ("既定", option.note) if part)
                labels["0"] = f"0 {option.variant}（{note}）"
            else:
                labels[str(index)] = f"{index} {option.label}"
        return labels

    def status(self) -> str:
        """今の model と、それぞれの読み込みの状態 (1 行)。"""
        current = self.current_label
        states = " ｜ ".join(
            f"{index} {_STATE_TEXT.get(self._option_state(option), self._option_state(option))}"
            for index, option in enumerate(self._options)
        )
        return f"[model] {self.skill_name}: いま {current} ｜ {states}"

    def choose(self, key: str) -> tuple[bool, str]:
        """数字キーで選ぶ。(受け付けたか, 端末に出す 1 行) を返す。

        受け付けたら関所は R と同じく初期姿勢へ進む。受け付けなければ待ち続ける。
        """
        index = int(key)
        option = self._options[index]
        with self._lock:
            previous = self._options[self._current]
            if index == self._current:
                return True, f"[model] {self.skill_name}: {option.label} のまま"
        state = self._option_state(option)
        if state != "loaded":
            reason = {
                "loading": "まだ読み込み中（読むのは腕が止まっている間だけ。少し待ってからもう一度押す）",
                "failed": "読み込みに失敗したので選べない",
                "released": "もう解放したので選べない",
            }.get(state, f"選べない（{state}）")
            return (
                False,
                f"[model] {self.skill_name}: {index} {option.label} は{reason}",
            )
        if self.replace_fn is None:
            raise RuntimeError(f"{self.skill_name}: no replace_fn is bound")
        self.replace_fn(option.skill)
        with self._lock:
            self._current = index
        return True, (
            f"[model] {self.skill_name}: {previous.label} → {option.label} に切り替えた"
            "（次の試行から）"
        )

    def _state(self, variant: str) -> str:
        return "loaded" if self.state_fn is None else self.state_fn(variant)

    def _option_state(self, option: ChoiceOption) -> str:
        """その候補の model と requires の、いちばん悪い状態。"""
        states = [self._state(option.variant)] + [self._state(name) for name in option.requires]
        for state in ("failed", "released", "loading"):
            if state in states:
                return state
        return "loaded" if all(state == "loaded" for state in states) else states[0]
