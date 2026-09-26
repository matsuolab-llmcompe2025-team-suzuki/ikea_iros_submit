"""One terminal reader for the production Stage controls.

The reader owns stdin for the entire actuated run.  Gates wait on its queue while
the control loop continues to publish holds and check observation freshness.
"""

from __future__ import annotations

import os
import queue
import select
import sys
import termios
import threading
import time
import tty

#: 安全停止の後、戻す動作の前に操作者の判断を待つ画面のフェーズ名 (Issue #172)。
SAFETY_STOP_PHASE = "安全停止"


class OperatorConsole:
    def __init__(self) -> None:
        self._events: queue.Queue[tuple[int, str]] = queue.Queue()
        self._lock = threading.Lock()
        self._generation = 0
        self._view: tuple[int, str, tuple[str, ...]] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._original_termios: list | None = None
        self._fd: int | None = None
        self._last_key: str | None = None
        self._last_key_at = 0.0
        self._accept_after = 0.0

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("operator controls require an interactive TTY")
        self._fd = sys.stdin.fileno()
        self._original_termios = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        termios.tcflush(self._fd, termios.TCIFLUSH)
        self._thread = threading.Thread(target=self._read, name="operator-keys", daemon=True)
        self._thread.start()

    @property
    def running(self) -> bool:
        """キーを読む thread が動いていて、端末が切れていないか。"""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop.is_set()
        )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)
        if self._fd is not None and self._original_termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._original_termios)
            except termios.error:
                # A disconnected terminal must not interrupt robot cleanup.
                pass
            self._original_termios = None

    def _read(self) -> None:
        assert self._fd is not None
        while not self._stop.is_set():
            try:
                ready = select.select([self._fd], [], [], 0.1)[0]
            except (OSError, ValueError):
                self._stop.set()
                return
            if not ready:
                continue
            try:
                raw = os.read(self._fd, 1)
            except OSError:
                self._stop.set()
                return
            if not raw:
                self._stop.set()
                return
            key = "enter" if raw in (b"\r", b"\n") else raw.decode("ascii", "ignore").lower()
            now = time.monotonic()
            with self._lock:
                repeated = key == self._last_key and now - self._last_key_at < 0.35
                self._last_key, self._last_key_at = key, now
                if (
                    self._view is not None
                    and key in self._view[2]
                    and now >= self._accept_after
                    and not repeated
                ):
                    self._events.put((self._generation, key))

    def show(self, stage: int, phase: str, allowed: tuple[str, ...]) -> None:
        view = (stage, phase, allowed)
        with self._lock:
            if view == self._view:
                return
            self._generation += 1
            self._view = view
            self._accept_after = time.monotonic() + 0.15
            while not self._events.empty():
                try:
                    self._events.get_nowait()
                except queue.Empty:
                    break
            if self._fd is not None:
                termios.tcflush(self._fd, termios.TCIFLUSH)
        labels = {"n": "N 次へ", "r": "R やり直し", "enter": "Enter 開始"}
        stop_label = "Ctrl+C 終了"
        if phase.endswith("腕保持・ハンド全開"):
            labels["r"] = "R 初期姿勢へ"
        if phase.endswith("脚配置・ハンド初期幅待ち"):
            labels["enter"] = "Enter ハンド初期幅へ"
        if phase.startswith(SAFETY_STOP_PHASE):
            labels["enter"] = "Enter 戻す（初期姿勢→ハンド全開→腕下ろし）"
            stop_label = "Ctrl+C その場で終了"
        controls = " ｜ ".join([*(labels[key] for key in allowed), stop_label])
        print(f"Stage {stage}\nフェーズ：{phase}\n操作：{controls}", file=sys.stderr)

    def poll(self) -> str | None:
        if self._stop.is_set():
            raise EOFError("operator terminal disconnected")
        try:
            generation, key = self._events.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            return key if generation == self._generation else None

    def has_pending(self) -> bool:
        """True while an accepted key is waiting for the control loop."""
        with self._lock:
            if self._stop.is_set():
                return True
            with self._events.mutex:
                return any(
                    generation == self._generation
                    for generation, _ in self._events.queue
                )

    def wait_for(
        self, key: str, *, stage: int, phase: str, detail: str | None = None
    ) -> str:
        self.show(stage, phase, (key,))
        if detail:
            # 例: 開始姿勢に届いたか・一番ずれた関節 (gate が実測から作る 1 行)
            print(detail, file=sys.stderr)
        while not self._stop.is_set():
            try:
                generation, received = self._events.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                if generation == self._generation and received == key:
                    return received
        raise EOFError("operator terminal closed")
