"""vLLM 0.29.0 の Conv3dLayer に、Thor (sm_110) 向けの cuBLAS → 畳み込みの fallback を入れる。

Qwen3-VL の画像エンコーダの入口 (Qwen3VisionPatchEmbed の Conv3dLayer) は、torch 2.9 以上だと
畳み込みを行列の掛け算 (F.linear = cuBLAS) に置き換えて計算する。Thor (sm_110) では、その
cuBLAS の準備 (cublasLtCreate) が CUBLAS_STATUS_NOT_INITIALIZED で落ちる報告がある。
報告元の直し方は「sm_110 では常に F.conv3d (cuDNN) を使う」
(ms1design/thorllm の patches/patch_sm110.py、Patch 6。vLLM 0.18.0 向けの書き方)。

ここでは、まず今までどおり行列の掛け算を試し、**cuBLAS の例外のときだけ**畳み込みで計算し直す。
以後はずっと畳み込みを使い、切り替えたことを log に残す。cuBLAS が動けば develop と同じ計算の
まま。環境変数 RAMEN_VLLM_CONV3D=conv なら最初から畳み込み (会場で判定に拾えない失敗が
出たときに、image を焼き直さずに全面的に切り替える逃げ道)。

- 失敗は cuBLAS の準備の段階で例外になる (壊れた計算が以後に残る種類のエラーではない)
  ので、同じ process のまま畳み込みで計算し直せる
- vLLM は起動時に画像を最大枚数入れた試しの計算をするので、切り替えは server の起動時に起きる
- --enforce-eager で動かす前提 (CUDA graph の記録中に例外が起きる状況を作らない)

使い方 (vlm 環境の python で): python patch_vllm_conv3d_sm110.py
完全一致のアンカーで置換し、見つからなければ失敗で止まる (版が変わって当たらないまま焼かない)。
既に当たっていれば何もしない。
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

VLLM_VERSION = "0.29.0"
MARKER = "# TEAM_RAMEN_THOR_CONV3D_FALLBACK"

# class の前に、切り替えの判定と log 用の名前を置く
CLASS_ANCHOR = '''# --8<-- [start:conv3d]
@CustomOp.register("conv3d")
class Conv3dLayer(ConvLayerBase):
    """Conv layer with Conv3d."""

    # --8<-- [end:conv3d]

    num_dim = 3
'''
CLASS_REPLACEMENT = '''# TEAM_RAMEN_THOR_CONV3D_FALLBACK (ikea_iros_submit tools/patch_vllm_conv3d_sm110.py)
import logging as _ramen_logging  # noqa: E402
import os as _ramen_os  # noqa: E402

_RAMEN_CONV3D_LOG = _ramen_logging.getLogger("vllm.ramen.conv3d")


# --8<-- [start:conv3d]
@CustomOp.register("conv3d")
class Conv3dLayer(ConvLayerBase):
    """Conv layer with Conv3d."""

    # --8<-- [end:conv3d]

    num_dim = 3

    # Thor (sm_110) では F.linear の cuBLAS の準備が落ちる報告がある (thorllm の Patch 6)。
    # 行列の掛け算を試し、cuBLAS の例外のときだけ畳み込みに切り替えて、以後は畳み込みを使う。
    # RAMEN_VLLM_CONV3D=conv なら最初から畳み込み。
    _ramen_use_conv = (
        _ramen_os.environ.get("RAMEN_VLLM_CONV3D", "").strip().lower() == "conv"
    )

    def _forward_mulmat_or_conv(self, x: torch.Tensor) -> torch.Tensor:
        if Conv3dLayer._ramen_use_conv:
            return self._forward_conv(x)
        try:
            return self._forward_mulmat(x)
        except RuntimeError as exc:
            if "CUBLAS" not in str(exc).upper():
                raise
            Conv3dLayer._ramen_use_conv = True
            _RAMEN_CONV3D_LOG.warning(
                "Conv3dLayer: F.linear (cuBLAS) failed (%s); falling back to F.conv3d "
                "(cuDNN) from now on (Thor sm_110 workaround)",
                exc,
            )
            return self._forward_conv(x)
'''

NATIVE_ANCHOR = '''    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Expected input shape: (batch_size, in_channels, time, height, width)"""
        if self.enable_linear:
            return self._forward_mulmat(x)
        else:
            return self._forward_conv(x)
'''
NATIVE_REPLACEMENT = '''    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Expected input shape: (batch_size, in_channels, time, height, width)"""
        if self.enable_linear:
            return self._forward_mulmat_or_conv(x)
        else:
            return self._forward_conv(x)
'''

CUDA_ANCHOR = """        if self.enable_linear and is_torch_equal_or_newer("2.9.0"):
            return self._forward_mulmat(x)
        return self._forward_conv(x)
"""
CUDA_REPLACEMENT = """        if self.enable_linear and is_torch_equal_or_newer("2.9.0"):
            return self._forward_mulmat_or_conv(x)
        return self._forward_conv(x)
"""


def main() -> int:
    got = importlib.metadata.version("vllm")
    if got != VLLM_VERSION:
        print(
            f"[patch] FAILED: vllm {got} (このパッチは {VLLM_VERSION} 用)",
            file=sys.stderr,
        )
        return 1
    spec = importlib.util.find_spec("vllm")
    target = Path(spec.origin).parent / "model_executor" / "layers" / "conv.py"
    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"[patch] already-applied: {target}")
        return 0
    for name, anchor in (
        ("class", CLASS_ANCHOR),
        ("forward_native", NATIVE_ANCHOR),
        ("forward_cuda", CUDA_ANCHOR),
    ):
        hits = source.count(anchor)
        if hits != 1:
            print(
                f"[patch] FAILED: {name} のアンカーが {hits} 箇所 (1 箇所であるべき)",
                file=sys.stderr,
            )
            return 1
    source = (
        source.replace(CLASS_ANCHOR, CLASS_REPLACEMENT)
        .replace(NATIVE_ANCHOR, NATIVE_REPLACEMENT)
        .replace(CUDA_ANCHOR, CUDA_REPLACEMENT)
    )
    target.write_text(source, encoding="utf-8")
    print(f"[patch] applied: Conv3dLayer cuBLAS -> conv3d fallback ({target})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
