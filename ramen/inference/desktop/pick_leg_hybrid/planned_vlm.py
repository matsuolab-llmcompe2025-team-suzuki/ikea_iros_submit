"""hybrid pick の VLM を `--gpu-models plan` の読む順番の 1 本として起動する (Issue #188 ② 段 2)。

`--gpu-models plan` では VLM を起動時に待たず、読む順番の中 (pick の GR00T の前) で起動する
(`policies/load_plan.LoadPlan`)。やることは今の起動時の手順と同じ:
起動 → `/health` を待つ → 慣らし → 本番と同じ問い合わせで確かめる (`probe_vlm_endpoint`)。

VLM が起動しなければ pick の main が揃わないので、pick の前の保持が止める (安全停止)。
Stage 1 (pick から始める) は最初の skill なので、今までどおり動かす前に待つ。
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Optional, Sequence


class PlannedVlm:
    """VLM server を LoadPlan の 1 本として扱う (`prepare` で起動、`close` で止める)。

    Args:
        server: `vlm_server.VenueVlmServer` (起動前のもの)。
        cfg: hybrid の設定。
        references: 参照画像 (`load_reference_images`)。
        warm_up / probe / self_check_images: test 用の差し替え口 (既定は本番の関数)。
    """

    def __init__(
        self,
        server: Any,
        cfg: Any,
        references: Sequence[Any],
        *,
        warm_up: Optional[Callable[..., Any]] = None,
        probe: Optional[Callable[..., dict]] = None,
        self_check_images: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._server = server
        self._cfg = cfg
        self._references = references
        self._warm_up = warm_up
        self._probe = probe
        self._self_check_images = self_check_images
        self._ready = False
        #: 本番と同じ問い合わせの結果 (model・latency など)。起動が済むまで None。
        self.endpoint: Optional[dict] = None

    @property
    def is_loaded(self) -> bool:
        return self._ready

    def prepare(self) -> None:
        """起動 → `/health` → 慣らし → 確かめ。途中で失敗したら止めてから例外を返す。"""
        if self._ready:
            return
        warm_up, probe, self_check_images = self._functions()
        try:
            self._server.start()
            self._server.wait_until_ready()
            warm_up(self._cfg, self_check_images(self._cfg, self._references))
            endpoint = probe(self._cfg, self._references)
        except BaseException:
            self._server.close()
            raise
        self.endpoint = dict(endpoint)
        self._ready = True
        print(
            f"[hybrid] VLM ready: model={endpoint.get('model')} "
            f"references={len(self._references)} "
            f"vlm_latency={endpoint.get('multimodal_latency_sec', float('nan')):.2f}s",
            file=sys.stderr,
        )

    def close(self) -> None:
        """止める。何度呼んでもよい。"""
        self._ready = False
        self._server.close()

    def _functions(self):
        warm_up, probe, self_check_images = (
            self._warm_up,
            self._probe,
            self._self_check_images,
        )
        if warm_up is None:
            from inference.desktop.pick_leg_hybrid.vlm_server import (
                warm_up_vlm as warm_up,
            )
        if probe is None or self_check_images is None:
            from inference.desktop.pick_leg_hybrid import real_skill

            probe = probe or real_skill.probe_vlm_endpoint
            self_check_images = (
                self_check_images or real_skill.build_vlm_self_check_images
            )
        return warm_up, probe, self_check_images
