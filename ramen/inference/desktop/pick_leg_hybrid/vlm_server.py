"""会場で hybrid pick の VLM server を run ごとに立てる (Issue #161)。

開発機は今までどおり外で立てた server (``run_local_vlm_server.sh``) を使う。会場 (提出 image) と
RunPod は entrypoint に ``--spawn-vlm-server`` を付け、この run の子 process として
``run_venue_vlm_server.sh`` を起動する。run が終われば VLM も止まり、メモリは次の run まで残らない。

1. 起動前に port が空いていることを確かめる。container は ``--network host`` なので、同じ port に
   別の server がいると、この run の問い合わせがそちらに行ってしまう。
2. 読み込みは時間制限なしで待つ (``/health`` が 200 を返すまで)。子 process が終われば故障として止める。
3. 慣らしに本番と同じ 5 枚の問い合わせを 1 回、時間制限なしで送る。起動直後の kernel の compile を
   読み込み側に入れ、その後の本番の確認 (5 s 以内) を定常の速さで測る。
4. 止めるときは process group ごと止める (vLLM は engine の process を別に作る)。
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import IO, Optional, Sequence

from inference.desktop.pick_leg_hybrid.config import PickLegHybridConfig
from inference.desktop.pick_leg_hybrid.vlm import (
    GRASP_PHASE_TEXT,
    GRASP_QUESTION,
    build_messages,
)

#: 会場で起動する script。host / port はこの script と同じ (test で突き合わせる)。
VENUE_VLM_SCRIPT: Path = Path(__file__).resolve().parent / "run_venue_vlm_server.sh"
VENUE_VLM_HOST = "127.0.0.1"
VENUE_VLM_PORT = 8000

_LOG_TAIL_LINES = 40


def check_venue_endpoint(endpoint: str) -> None:
    """起動する server と client の宛先が同じかを確かめる。

    Raises:
        ValueError: endpoint が ``http://127.0.0.1:8000/...`` でない場合。
    """
    parsed = urllib.parse.urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != VENUE_VLM_HOST
        or parsed.port != VENUE_VLM_PORT
    ):
        raise ValueError(
            f"--spawn-vlm-server starts the VLM at http://{VENUE_VLM_HOST}:{VENUE_VLM_PORT}, "
            f"but the VLM endpoint is {endpoint!r}"
        )


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _health_ok(url: str, timeout_s: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.status == 200
    except OSError:
        # 接続拒否・読み込み中の timeout・HTTP エラー (URLError / HTTPError は OSError)
        return False


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class VenueVlmServer:
    """VLM server をこの run の子 process として起動・待機・停止する。

    entrypoint は ``_register_policy_resource`` で登録し、終了時 (例外・Ctrl-C を含む) の
    ``close()`` で止める。
    """

    def __init__(
        self,
        log_path: Path,
        *,
        command: Sequence[str] = ("bash", str(VENUE_VLM_SCRIPT)),
        host: str = VENUE_VLM_HOST,
        port: int = VENUE_VLM_PORT,
        poll_interval_s: float = 1.0,
        progress_interval_s: float = 10.0,
        stop_timeout_s: float = 10.0,
    ) -> None:
        self.log_path = Path(log_path)
        self.command = list(command)
        self.host = host
        self.port = port
        self.poll_interval_s = poll_interval_s
        self.progress_interval_s = progress_interval_s
        self.stop_timeout_s = stop_timeout_s
        self._process: Optional[subprocess.Popen] = None
        self._log: Optional[IO[bytes]] = None

    def start(self) -> None:
        """port が空いていることを確かめてから起動する。

        Raises:
            RuntimeError: port が既に使われている場合。
        """
        if _port_is_open(self.host, self.port):
            raise RuntimeError(
                f"{self.host}:{self.port} is already in use; not starting the VLM server "
                "(this run's requests would go to whatever is listening there)"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("wb")
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(
            f"[vlm] starting the VLM server (pid={self._process.pid}); log -> {self.log_path}",
            file=sys.stderr,
        )

    def wait_until_ready(self) -> float:
        """``/health`` が 200 を返すまで待つ。時間制限なし。

        Returns:
            待った秒数。

        Raises:
            RuntimeError: 起動前に呼んだ場合、または読み込み中に子 process が終わった場合。
        """
        process = self._process
        if process is None:
            raise RuntimeError("VenueVlmServer.start() must be called first")
        url = f"http://{self.host}:{self.port}/health"
        started = time.monotonic()
        next_progress = started + self.progress_interval_s
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"the VLM server exited while loading (exit_code={exit_code}); "
                    f"last lines of {self.log_path}:\n{self._log_tail()}"
                )
            if _health_ok(url, timeout_s=self.poll_interval_s):
                elapsed = time.monotonic() - started
                print(f"[vlm] server ready after {elapsed:.1f}s", file=sys.stderr)
                return elapsed
            now = time.monotonic()
            if now >= next_progress:
                print(
                    f"[vlm] loading... {now - started:.0f}s "
                    "(no time limit; Ctrl-C stops the run)",
                    file=sys.stderr,
                )
                next_progress = now + self.progress_interval_s
            time.sleep(self.poll_interval_s)

    def close(self) -> None:
        """process group ごと止める。何度呼んでもよい。"""
        process = self._process
        if process is not None:
            self._process = None
            # 起動した script は vllm に置き換わり、vllm は engine の process を別に作る。
            # 先頭の process が既に終わっていても、残った engine まで止めるため group に送る。
            _signal_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                pass
            _signal_group(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        if self._log is not None:
            self._log.close()
            self._log = None

    def _log_tail(self) -> str:
        if self._log is not None:
            self._log.flush()
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError as exc:
            return f"(log unreadable: {exc})"
        return "\n".join(lines[-_LOG_TAIL_LINES:])


def warm_up_vlm(
    cfg: PickLegHybridConfig,
    images_b64: Sequence[str],
    *,
    progress_interval_s: float = 10.0,
) -> float:
    """本番と同じ形の問い合わせを 1 回、時間制限なしで送る。答えは見ない。

    待っている間は ``progress_interval_s`` ごとに経過を出す (時間制限が無いので、詰まったときに
    操作者が「待つか止めるか」を決められるように。読み込みの待ちと同じ)。

    Args:
        cfg: hybrid の設定 (endpoint・model・出力の長さ・システム指示)。
        images_b64: 本番と同じ枚数の画像 (``build_vlm_self_check_images``)。
        progress_interval_s: 経過を出す間隔 [s]。

    Returns:
        かかった秒数。

    Raises:
        RuntimeError: HTTP エラーや接続の失敗 (読み込みは済んでいるので故障として止める)。
    """
    # VlmBoundaryClient.ask と同じ形。ask は cfg.vlm.timeout_sec で打ち切るので使わない
    payload = {
        "model": cfg.vlm.model,
        "messages": build_messages(
            GRASP_QUESTION,
            images_b64,
            current_phase_text=GRASP_PHASE_TEXT,
            system_prompt=cfg.vlm.system_prompt,
        ),
        "max_tokens": cfg.vlm.max_tokens,
        "temperature": cfg.vlm.temperature,
    }
    request = urllib.request.Request(
        cfg.vlm.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(
        f"[vlm] warm-up request ({len(images_b64)} images, no time limit)...",
        file=sys.stderr,
    )
    started = time.monotonic()
    done = threading.Event()

    def _report_progress() -> None:
        while not done.wait(progress_interval_s):
            print(
                f"[vlm] warm-up still running... {time.monotonic() - started:.0f}s "
                "(no time limit; Ctrl-C stops the run)",
                file=sys.stderr,
            )

    threading.Thread(
        target=_report_progress, name="vlm-warm-up-progress", daemon=True
    ).start()
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            response.read()
    except OSError as exc:
        detail = (
            exc.read().decode("utf-8", "replace")[:500] if hasattr(exc, "read") else ""
        )
        raise RuntimeError(
            f"VLM warm-up request failed at {cfg.vlm.endpoint}: {exc} {detail}".rstrip()
        ) from exc
    finally:
        done.set()
    elapsed = time.monotonic() - started
    print(f"[vlm] warm-up done in {elapsed:.1f}s", file=sys.stderr)
    return elapsed
