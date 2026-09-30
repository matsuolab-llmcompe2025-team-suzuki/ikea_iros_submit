"""1 Stage の run で、model を決めた順番に 1 本ずつ読む (Issue #188 ② 段 2、`--gpu-models plan`)。

# なぜ要るか

会場の `--gpu-models all` は stage の model を起動時に全部読み、読み終わるまで開始の
Enter を受け付けない。R で選ぶ候補 (`policy_config.yaml` の `alternatives_by_skill`)
を足すと、起動の待ちはさらに伸びる。

# 読む順番

1. 起動時 (`prime`) は最初の skill の main だけを、呼び出し元の thread で読む。
2. 残りは 1 本の worker thread で、並べた順に 1 本ずつ読む (並列に読むとページ
   キャッシュが膨らむ)。最初の skill の候補 → `preload_through_skill` までの skill の
   main → 候補。
3. それより後ろの skill (insert・締め付け) は、その skill の前の保持
   (`hold_transition_*`) に入ったときに main → 候補の順で積む。保持が待つのは main
   だけ (`is_ready`)。候補は後ろで読む。
4. 保持に入ったら、それより前の skill の model (main・候補) を解放する。1 Stage の run
   の中で前の skill には戻らない。

# 読むのは腕が止まっている間だけ

policy が動いている間・腕や手を動かしている間は、新しく読み始めない (解放もしない)。
RAMEN-Ori は制御ループと同じ process で読むので処理を取り合い、別 process の
GR00T・DP・VLM も GPU とディスクを取り合う。止まっている間 = `STATIONARY_PREFIXES`
(開始待ち・R の後の待ち・保持)。そこから動き出す関所 (開始待ちの Enter・R の後の待ち)
は `may_leave` で読みかけの 1 本が終わるのを待つ (途中では止められない)。例外は
`load_while_policy_runs` に書いた種類だけ (既定は空 = 全部止まっている間だけ)。

止まっている区間を出るのは、その区間が終わったとき (Enter・保持の時間) だけ。N・R は
policy が動いている間しか効かない (`Orchestrator._operator_command`)。

# 記録

1 本読む・解放するたびに `event_fn` へ 1 件渡す (`"event": "model_load"`、entrypoint が
orch log に書く): かかった秒数・始めたときの skill・空きメモリとキャッシュ (前・後)・
落としたページキャッシュ・読んでいる間の制御の周期 (p95・最大)。worker thread からは
書かず、tick の thread (`on_skill_started`) でまとめて渡す (orch log を書く thread を 1 つに保つ)。
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from inference.desktop.lower_policy.policies.page_cache import drop_page_cache

#: 止まっている区間から動き出す関所 (Enter・R を待つ `OperatorConfirmationHoldSkill`)。
#: entrypoint がここに `may_leave` を付ける (`set_leave_barrier`)。
EXIT_GATE_PREFIXES: tuple[str, ...] = (
    "operator_gate_for_",  # → policy
    "retry_wait_",  # → retry_arm_ (腕を開始姿勢へ)
    "retry_hand_gate_",  # → retry_hand_ (手を開始の開度へ)
)

#: 腕が止まっている skill の名前の頭。出口は `EXIT_GATE_PREFIXES` の関所か、次も止まって
#: いる区間 (`retry_hold_`・`hold_transition_`・`hold_pose_for_` → 開始待ちの関所)。
#: `wait_for_go_live_*` は入れない (go-live で自動的に抜けて腕の pre-motion が始まるので、
#: 読みかけの 1 本を待つ関所が無い)。
#: 開始待ち (`operator_gate_for_*`) で U / D を押すと腕が 0.2 rad/s で動くが、止まっている区間の
#: まま扱う (動きは小さく、1 tick の動きも最大 0.02 rad に抑えてある。`start_lift.StartLiftAdjuster.ramp`)。
STATIONARY_PREFIXES: tuple[str, ...] = EXIT_GATE_PREFIXES + (
    "retry_hold_",
    "hold_transition_",
    "hold_pose_for_",
)

#: `load_while_policy_runs` に書ける種類 (`policy_type` と VLM)。
LOAD_KINDS: frozenset[str] = frozenset(
    {
        "groot",
        "groot_pick_legs",
        "groot_pick_legs_ee_rel",
        "act_diffusion",
        "ramen_ori",
        "vlm",
    }
)


@dataclass(frozen=True)
class PlannedModel:
    """順番に並べる model 1 本。

    Attributes:
        skill: どの skill の model か。
        label: log に出す名前。
        kind: `policy_type` ("ramen_ori" / "groot" / …) か "vlm"。
        policy: `prepare()` と `close()` を持つもの (`release_from_residency()` が
            あればそちらで解放する)。
        required: True = main (保持はこれを待つ)。False = 候補。
        variant: `policy_config.yaml` の variant 名 (VLM は空)。
        weight_files: 重みの file を返す関数。読み終わったらそのページキャッシュを落とす
            (`page_cache.drop_page_cache`)。None なら落とさない。
    """

    skill: str
    label: str
    kind: str
    policy: Any
    required: bool = True
    variant: str = ""
    weight_files: Optional[Callable[[], Sequence[Path]]] = None


class _Entry:
    """1 本の状態。idle → queued → loading → loaded → releasing → released (失敗は failed)。"""

    __slots__ = ("model", "state", "error", "release_after_load")

    def __init__(self, model: PlannedModel) -> None:
        self.model = model
        self.state = "idle"
        self.error: Optional[str] = None
        self.release_after_load = False


class LoadPlan:
    """1 Stage の skill の列に沿って、model を決めた順番に 1 本ずつ読む。

    `ModelResidency` と同じ呼ばれ方をする (`prime` / `on_skill_started` / `is_ready` /
    `error_for` / `close`)。

    Args:
        order: stage の skill 名の列 (頭の手順・切り替えの手順を含む)。
        models: 読む model。skill ごとに main (`required=True`) を先、候補を後に、
            読みたい順に並べる。
        preload_through_skill: 起動の後に先に読んでおく最後の skill (この stage に無ければ
            最初の skill だけ)。
        load_while_policy_runs: 動いている間にも読んでよい種類 (`LOAD_KINDS`)。
        event_fn: 1 本ごとの記録 (dict) を受け取る関数 (tick の thread から呼ぶ)。
        clock: 秒の時計 (test 用)。
        meminfo: 空きメモリとキャッシュ [GB] を返す関数 (既定は `/proc/meminfo`、test 用)。
    """

    def __init__(
        self,
        order: list[str],
        models: Iterable[PlannedModel],
        *,
        preload_through_skill: Optional[str] = None,
        load_while_policy_runs: Iterable[str] = (),
        event_fn: Optional[Callable[[dict], None]] = None,
        clock: Callable[[], float] = time.monotonic,
        meminfo: Optional[Callable[[], Optional[dict]]] = None,
    ) -> None:
        self._timeline = list(order)
        self._entries = [_Entry(model) for model in models]
        owners = {entry.model.skill for entry in self._entries}
        unknown = sorted(owners - set(self._timeline))
        if unknown:
            raise ValueError(
                f"load plan has models for skills not in the stage: {unknown}"
            )
        self._skills = [
            name for name in dict.fromkeys(self._timeline) if name in owners
        ]
        for skill in self._skills:
            if not self._entries_of(skill, required=True):
                raise ValueError(f"load plan has no main model for {skill!r}")
        allowed = frozenset(load_while_policy_runs)
        if allowed - LOAD_KINDS:
            raise ValueError(
                f"load_while_policy_runs has unknown kinds {sorted(allowed - LOAD_KINDS)} "
                f"(valid: {sorted(LOAD_KINDS)})"
            )
        self._allowed_while_moving = allowed
        through = (
            self._skills.index(preload_through_skill)
            if preload_through_skill in self._skills
            else 0
        )
        self._preload = self._skills[: through + 1]
        self._cond = threading.Condition()
        self._queue: list[_Entry] = []
        self._releases: list[_Entry] = []
        self._busy: Optional[_Entry] = None
        self._active: Optional[str] = None
        self._stationary = False
        self._leaving = False
        self._closing = False
        self._worker: Optional[threading.Thread] = None
        self._event_fn = event_fn
        self._clock = clock
        self._meminfo = meminfo or read_meminfo_gb
        self._events: list[dict] = []
        self._last_tick: Optional[float] = None
        self._busy_ticks: list[float] = []
        for entry in self._entries:
            setter = getattr(entry.model.policy, "set_residency_managed", None)
            if callable(setter):
                setter(True)

    # ------------------------------------------------------------------ 呼ばれ方

    def prime(self) -> None:
        """最初の skill の main をこの thread で読み、残りを順番に積む。

        stage を始める前 (開始待ちより前) に呼ぶ。失敗は例外のまま返す (動かす前に止める)。
        """
        if not self._skills:
            return
        first = self._skills[0]
        for entry in self._entries_of(first, required=True):
            print(
                f"[load-plan] loading {entry.model.label} before the stage starts",
                file=sys.stderr,
            )
            with self._cond:
                entry.state = "loading"
                self._busy_ticks = []
            measure = self._begin(stationary=True, during="before the stage")
            try:
                entry.model.policy.prepare()
            except Exception as exc:
                with self._cond:
                    entry.state = "failed"
                    entry.error = repr(exc)
                self._finish(entry, "load", measure, exc, 0)
                raise
            with self._cond:
                entry.state = "loaded"
            self._finish(entry, "load", measure, None, self._drop_page_cache(entry))
        with self._cond:
            self._schedule(self._entries_of(first, required=False))
            for skill in self._preload[1:]:
                self._schedule(self._entries_of(skill))
            queued = [entry.model.label for entry in self._queue]
        if queued:
            print(
                "[load-plan] loading next while the arms are still: "
                + " → ".join(queued),
                file=sys.stderr,
            )
        self._ensure_worker()

    def on_skill_started(self, skill_name: Optional[str]) -> None:
        """tick ごとに今の skill 名を渡す。変わったときだけ仕事をする。"""
        if skill_name is None:
            return
        now = self._clock()
        with self._cond:
            # 読んでいる間の制御の周期 (tick の間隔) を測る
            if self._busy is not None and self._last_tick is not None:
                self._busy_ticks.append(now - self._last_tick)
            self._last_tick = now
            events, self._events = self._events, []
            changed = skill_name != self._active
            if changed:
                self._active = skill_name
                self._stationary = skill_name.startswith(STATIONARY_PREFIXES)
                self._leaving = False
                if (
                    skill_name.startswith("hold_transition_")
                    and skill_name in self._timeline
                ):
                    upcoming = self._next_skill_after(skill_name)
                    if upcoming is not None:
                        self._enter(upcoming)
                self._cond.notify_all()
        self._emit(events)
        if changed:
            self._ensure_worker()

    def stay(self) -> None:
        """動き出すのをやめた (関所が Enter の確認を取り消した)。止まっている間なら読み込みを続ける。"""
        with self._cond:
            if self._stationary and self._leaving:
                self._leaving = False
                self._cond.notify_all()

    def may_leave(self) -> bool:
        """止まっている区間から動き出してよいか (関所が Enter の後に毎 tick 聞く)。

        聞かれた後は新しく読み始めない。読みかけの 1 本が終わったら True。
        """
        with self._cond:
            self._leaving = True
            busy = self._busy
            return busy is None or busy.model.kind in self._allowed_while_moving

    def is_ready(self, skill_name: str) -> bool:
        """その skill の main を読み終わったか (model の無い skill は True)。"""
        entries = self._entries_of(skill_name, required=True)
        with self._cond:
            states = [entry.state for entry in entries]
        return all(state == "loaded" for state in states) and all(
            _policy_is_loaded(entry.model.policy) for entry in entries
        )

    def model_state(self, skill_name: str, variant: str) -> str:
        """その skill の variant の読み込みの状態 (R の後に model を選ぶ画面用、Issue #188 段 3)。

        "loaded" / "loading" (積んだ・読みかけ・まだ積んでいない) / "failed" / "released"。
        main は main と一緒に読む物 (hybrid pick の VLM) も揃って "loaded"。
        """
        entries = [e for e in self._entries_of(skill_name) if e.model.variant == variant]
        if not entries:
            raise KeyError(f"{skill_name!r} has no model {variant!r} in the load plan")
        if entries[0].model.required:
            entries = self._entries_of(skill_name, required=True)
        with self._cond:
            states = [entry.state for entry in entries]
        if "failed" in states:
            return "failed"
        if any(state in ("released", "releasing") for state in states):
            return "released"
        if all(state == "loaded" for state in states) and all(
            _policy_is_loaded(entry.model.policy) for entry in entries
        ):
            return "loaded"
        return "loading"

    def error_for(self, skill_name: str) -> Optional[str]:
        """その skill の main の読み込みが失敗していれば理由。"""
        with self._cond:
            for entry in self._entries_of(skill_name, required=True):
                if entry.state == "failed":
                    return f"{entry.model.label}: {entry.error}"
        return None

    def check_each(self, *, skip_ids: Iterable[int] = ()) -> list[str]:
        """動かさない確かめ (`--actuate` 無し) 用: 並べた物を 1 本ずつ読んで解放する。

        候補や VLM は skill の policy ではないので、entrypoint の main の確かめでは読まれない。
        会場はオフラインなので、重みが事前取得から漏れていないかをここで見る。
        ``skip_ids`` (`id(policy)`) は確かめ済みの物。確かめた物の label を返す。
        """
        skipped = set(skip_ids)
        checked = []
        for entry in self._entries:
            policy = entry.model.policy
            if id(policy) in skipped:
                continue
            print(f"[load-plan] checking {entry.model.label}", file=sys.stderr)
            validate = getattr(policy, "validate_load_and_release", None)
            if callable(validate):
                validate()
            else:
                policy.prepare()
                self._release(entry)
            skipped.add(id(policy))
            checked.append(entry.model.label)
        return checked

    @property
    def loaded(self) -> tuple[str, ...]:
        with self._cond:
            return tuple(
                entry.model.label for entry in self._entries if entry.state == "loaded"
            )

    def close(self) -> None:
        """worker を止め、読んだ model を全部解放する。"""
        with self._cond:
            self._closing = True
            busy = self._busy
            self._cond.notify_all()
        if busy is not None and busy.model.kind == "vlm":
            # VLM の起動待ち (wait_until_ready) は時間制限が無い。先に止めると待ちが抜けるので、
            # worker を待つ前に止める (GR00T などは読み込みの lock を握るので、ここでは待つしかない)
            self._release(busy)
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=30.0)
        self._worker = None
        for entry in self._entries:
            if entry.state in ("loading", "loaded", "releasing"):
                self._release(entry)
                entry.state = "released"
            setter = getattr(entry.model.policy, "set_residency_managed", None)
            if callable(setter):
                setter(False)
        with self._cond:
            events, self._events = self._events, []
        self._emit(events)

    # ------------------------------------------------------------------ 中身

    def _entries_of(
        self, skill: str, *, required: Optional[bool] = None
    ) -> list[_Entry]:
        return [
            entry
            for entry in self._entries
            if entry.model.skill == skill
            and (required is None or entry.model.required == required)
        ]

    def _next_skill_after(self, name: str) -> Optional[str]:
        index = self._timeline.index(name)
        return next(
            (skill for skill in self._timeline[index + 1 :] if skill in self._skills),
            None,
        )

    def _schedule(self, entries: Iterable[_Entry]) -> None:
        """lock を持って呼ぶ。まだ積んでいない物だけを後ろに足す。"""
        for entry in entries:
            if entry.state == "idle":
                entry.state = "queued"
                self._queue.append(entry)

    def _enter(self, skill: str) -> None:
        """lock を持って呼ぶ。その skill の保持に入った: 前の skill を解放し、main → 候補を積む。"""
        index = self._skills.index(skill)
        for earlier in self._skills[:index]:
            for entry in self._entries_of(earlier):
                self._drop(entry)
        self._schedule(self._entries_of(skill))

    def _drop(self, entry: _Entry) -> None:
        """lock を持って呼ぶ。もう使わない model を積みから外すか、解放に回す。"""
        if entry.state == "queued":
            self._queue.remove(entry)
            entry.state = "released"
        elif entry.state == "loading":
            entry.release_after_load = True
        elif entry.state == "loaded":
            entry.state = "releasing"
            self._releases.append(entry)
        elif entry.state == "idle":
            entry.state = "released"

    def _may_run(self, entry: _Entry) -> bool:
        return self._stationary or entry.model.kind in self._allowed_while_moving

    def _next_job(self) -> Optional[tuple[str, _Entry]]:
        """lock を持って呼ぶ。今やってよい次の仕事 (解放が先)。"""
        if self._leaving:
            return None
        for operation, jobs in (("release", self._releases), ("load", self._queue)):
            for entry in jobs:
                if self._may_run(entry):
                    jobs.remove(entry)
                    return operation, entry
        return None

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        with self._cond:
            if self._closing:
                return
        self._worker = threading.Thread(target=self._run, name="load-plan", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            with self._cond:
                while True:
                    if self._closing:
                        return
                    job = self._next_job()
                    if job is not None:
                        break
                    self._cond.wait()
                operation, entry = job
                self._busy = entry
                self._busy_ticks = []
                entry.state = "loading" if operation == "load" else "releasing"
                stationary, during = self._stationary, self._active
            label = entry.model.label
            error: Optional[Exception] = None
            dropped = 0
            measure = self._begin(stationary=stationary, during=during)
            try:
                if operation == "load":
                    print(f"[load-plan] loading {label}", file=sys.stderr)
                    entry.model.policy.prepare()
                    dropped = self._drop_page_cache(entry)
                else:
                    self._release(entry)
            # 読み込みの失敗で run を落とさない (main なら保持が止める)
            except Exception as exc:
                error = exc
            with self._cond:
                self._busy = None
                if operation == "release":
                    entry.state = "released"
                elif error is not None:
                    entry.state = "failed"
                    entry.error = repr(error)
                elif entry.release_after_load:
                    entry.release_after_load = False
                    entry.state = "releasing"
                    self._releases.append(entry)
                else:
                    entry.state = "loaded"
                self._cond.notify_all()
            self._finish(entry, operation, measure, error, dropped)
            if error is not None:
                print(
                    f"[load-plan] 🔴 {operation} failed for {label}: {error!r}"
                    + (
                        " (main: the hold before it stops the run)"
                        if entry.model.required and operation == "load"
                        else ""
                    ),
                    file=sys.stderr,
                )

    @staticmethod
    def _drop_page_cache(entry: _Entry) -> int:
        """読み終わった重みの file をページキャッシュから落とす (失敗しても読み込みは成功のまま)。

        落とした file の大きさの合計 [byte] を返す。
        """
        files = entry.model.weight_files
        if files is None:
            return 0
        try:
            count, size = drop_page_cache(files())
        except Exception as exc:  # noqa: BLE001 - 落とせなくても model は使える
            print(
                f"[load-plan] could not drop the page cache of {entry.model.label}: {exc!r}",
                file=sys.stderr,
            )
            return 0
        if count:
            print(
                f"[load-plan] dropped the page cache of {entry.model.label}: "
                f"{count} files, {size / 1e9:.1f} GB",
                file=sys.stderr,
            )
        return size

    def _begin(self, *, stationary: bool, during: Optional[str]) -> dict:
        return {
            "started": self._clock(),
            "stationary": stationary,
            "during": during,
            "memory": self._read_memory(),
        }

    def _finish(
        self,
        entry: _Entry,
        operation: str,
        measure: dict,
        error: Optional[BaseException],
        dropped_bytes: int,
    ) -> None:
        """1 本の記録を作って積み (tick の thread が渡す)、1 行で知らせる。"""
        seconds = self._clock() - measure["started"]
        memory = self._read_memory()
        with self._cond:
            ticks = list(self._busy_ticks)
        period = _period_stats_ms(ticks)
        before, after = measure["memory"] or {}, memory or {}
        record = {
            "event": "model_load",
            "monotonic_ns": time.monotonic_ns(),
            "operation": operation,
            "label": entry.model.label,
            "skill": entry.model.skill,
            "variant": entry.model.variant,
            "kind": entry.model.kind,
            "role": "main" if entry.model.required else "candidate",
            "ok": error is None,
            "error": None if error is None else repr(error),
            "seconds": round(seconds, 3),
            "during": measure["during"],
            "while_moving": not measure["stationary"],
            "mem_available_gb": [before.get("MemAvailable"), after.get("MemAvailable")],
            "cached_gb": [before.get("Cached"), after.get("Cached")],
            "page_cache_dropped_gb": round(dropped_bytes / 1e9, 3),
            "tick_period_ms": period,
        }
        with self._cond:
            self._events.append(record)
        if error is None:
            memory_text = (
                f"; free {after['MemAvailable']:.1f} GB, cache {after['Cached']:.1f} GB"
                if "MemAvailable" in after and "Cached" in after
                else ""
            )
            period_text = (
                f"; control period while loading p95 {period['p95']:.0f} ms, "
                f"max {period['max']:.0f} ms"
                if period is not None
                else ""
            )
            verb = "loaded" if operation == "load" else "released"
            print(
                f"[load-plan] {verb} {entry.model.label} in {seconds:.1f}s"
                f"{memory_text}{period_text}",
                file=sys.stderr,
            )

    def _read_memory(self) -> Optional[dict]:
        try:
            return self._meminfo()
        except Exception:  # noqa: BLE001 - 記録のためだけ
            return None

    def _emit(self, events: list[dict]) -> None:
        if self._event_fn is None:
            return
        for record in events:
            try:
                self._event_fn(record)
            except Exception as exc:  # noqa: BLE001 - 記録の失敗で run を止めない
                print(f"[load-plan] could not record a load event: {exc!r}", file=sys.stderr)

    @staticmethod
    def _release(entry: _Entry) -> None:
        policy = entry.model.policy
        close = getattr(policy, "release_from_residency", None)
        if not callable(close):
            close = getattr(policy, "close", None)
        if callable(close):
            print(f"[load-plan] releasing {entry.model.label}", file=sys.stderr)
            close()


def read_meminfo_gb(path: Path = Path("/proc/meminfo")) -> Optional[dict[str, float]]:
    """`/proc/meminfo` の MemAvailable と Cached [GB]。読めない OS (macOS) は None。"""
    try:
        text = path.read_text(encoding="ascii")
    except OSError:
        return None
    values: dict[str, float] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemAvailable", "Cached") and rest.split():
            values[key] = round(float(rest.split()[0]) * 1024 / 1e9, 3)
    return values or None


def _period_stats_ms(ticks: Sequence[float]) -> Optional[dict]:
    """tick の間隔 [s] の数・p95・最大 [ms]。tick が無ければ None。"""
    if not ticks:
        return None
    ordered = sorted(ticks)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "n": len(ordered),
        "p95": round(p95 * 1e3, 1),
        "max": round(ordered[-1] * 1e3, 1),
    }


def _policy_is_loaded(policy: Any) -> bool:
    """policy が `is_loaded` を持つならそれを、無ければ True (帳簿を信じる)。"""
    loaded = getattr(policy, "is_loaded", None)
    return bool(loaded) if isinstance(loaded, bool) else True
