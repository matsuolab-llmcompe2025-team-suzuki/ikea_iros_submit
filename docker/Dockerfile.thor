# syntax=docker/dockerfile:1
#
# 提出 image (Thor 側) — Team RAMEN (#10)。
#   Jetson AGX Thor: aarch64 / sm_110 (Blackwell) / JetPack 7.x / CUDA 13
#
# 本体 iros_2026_ramen の推論をそのまま動かす。Python の環境は本体の pixi (ramen/ に同じ
# 相対パスでコピー) で作り、この Dockerfile には Python 以外 (apt / CycloneDDS の C ライブラリ /
# unitree SDK のソース / pixi 本体) だけを書く。
#
# 本体コードは ramen/ (tools/sync_ramen.sh が本体の 1 commit からコピー、手で直さない) を
# 環境の層の後に入れる。起動口は docker/venue_entry.sh (会場は `… <image> --stage N --actuate`)。
# ⚠️ linux/arm64。.github/workflows/build-thor-image.yml が ubuntu-24.04-arm でネイティブに焼く。

# torch は pixi が cu130 の wheel で入れる (CUDA のランタイムも wheel に同梱。sm_110 を含むのは
# cu130 の aarch64 build だけ)。driver は Thor の host から nvidia container runtime が入れる
# ので、base は CUDA の最小構成で足りる (NGC の pytorch image は使わない)。
# tag ではなく index digest で固定する (同じ tag の再 push で中身が変わらないように)。
#   tag   : nvidia/cuda:13.0.3-base-ubuntu24.04 (NVIDIA 公式、2026-04-14 公開)
#   arm64 : sha256:56d9d8183e2181a20be6b0d3801d1f056a0e75c17706df939ba207b126e1cb9c
ARG BASE=nvidia/cuda@sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1
FROM ${BASE}

# NVIDIA_DISABLE_REQUIRE: 運営 onboarding の Finding 1。以前は docker run の -e で
# 渡してもらっていた。image に焼いて渡し忘れで起動を拒否されないようにする。
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    NVIDIA_DISABLE_REQUIRE=1

# --- 1) apt -----------------------------------------------------------------
#   git / cmake / build-essential : CycloneDDS と cyclonedds の python binding の build
#   ca-certificates / curl        : pixi の取得
#   libgl1 / libglib2.0-0         : opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git cmake build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CUDA 13 の compiler 部品 (base image に入っている NVIDIA の apt source から)。VLM server 用:
#   cuda-nvcc-13-0       : ptxas。Triton が同梱する ptxas は CUDA 12.8 版で sm_110a を知らない
#                          (check-thor-vlm で確認) ので、TRITON_PTXAS_PATH でこちらを使わせる。
#                          nvcc は FlashInfer の事前 compile 済み kernel に無い形が来たときの予備
#   cuda-cudart-dev-13-0 : その予備の compile に要る header
#   cuda-cuobjdump-13-0  : build 時の確認 (事前 compile 済み kernel に sm_110a があるか)
RUN apt-get update && apt-get install -y --no-install-recommends \
      cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cuobjdump-13-0 \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas

# --- 2) CycloneDDS の C ライブラリ --------------------------------------------
# python binding の cyclonedds 0.10.2 は linux/aarch64 の wheel が無く、pixi install が
# CYCLONEDDS_HOME を見て sdist から build する。binding と同じ版を tag で取り、commit も確かめる
# (releases/0.10.x の先端は 0.10.5 で、binding とずれる。tag は付け替えられうるので、
# 他の依存と同じく commit で固定する。PR #11 のレビュー)。
ARG CYCLONEDDS_REF=0.10.2
ARG CYCLONEDDS_COMMIT=9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd
RUN git clone --depth 1 --branch "${CYCLONEDDS_REF}" \
      https://github.com/eclipse-cyclonedds/cyclonedds /tmp/cyclonedds \
    && test "$(git -C /tmp/cyclonedds rev-parse HEAD)" = "${CYCLONEDDS_COMMIT}" \
    && cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
         -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_IDLC=ON \
    && cmake --build /tmp/cyclonedds/build --target install -j"$(nproc)" \
    && ldconfig \
    && rm -rf /tmp/cyclonedds
# build 時 (binding の link) と run 時 (import 時の .so 探索) の両方で要る。
ENV CYCLONEDDS_HOME=/usr/local

# --- 3) pixi (版と checksum を固定) --------------------------------------------
# #152 (実機評価) の方と同じ版。lock の書き方 (空の run_exports 等) がそろう。
ARG PIXI_VERSION=v0.73.0
ARG PIXI_SHA256=0788f47eb37e0706de209c3ce81cc804ee128f7dd4ec09a686754a67170ddb64
RUN curl -fsSL -o /tmp/pixi.tar.gz \
      "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-aarch64-unknown-linux-musl.tar.gz" \
    && echo "${PIXI_SHA256}  /tmp/pixi.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/pixi.tar.gz -C /usr/local/bin \
    && rm /tmp/pixi.tar.gz \
    && pixi --version

WORKDIR /app/ramen

# --- 4) unitree SDK のソース ---------------------------------------------------
# root の runtime env が third_party/unitree_sdk2_python を editable で参照する (本体と同じ置き場所)。
# ref は固定する。運営 vendor 版 (GR00T-WholeBodyControl/external_dependencies) は
# LOCO_SERVICE_NAME="loco" で robot と話せず、全 RPC が 3102 になる (下の assert で確かめる)。
ARG UNITREE_SDK_REF=65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python \
      third_party/unitree_sdk2_python \
    && git -C third_party/unitree_sdk2_python checkout --quiet "${UNITREE_SDK_REF}" \
    && rm -rf third_party/unitree_sdk2_python/.git

# --- 5) pixi の環境 (Python のものは全部ここ) ------------------------------------
# lock どおりに入れ、解き直さない (--frozen)。定義だけを先に COPY して、コードの変更で
# この重い層を作り直さないようにする。
#   runtime                          : entrypoint 本体 / YOLO / RAMEN-Ori (py3.10)
#   inference/desktop の default      : GR00T 53D の worker (py3.12、lerobot 0.6.1)
#   inference/desktop の groot-pick   : GR00T pick の worker (py3.12.11、lerobot 0.6.0、#152 の .venv と同じ版)
#   inference/desktop の vlm          : pick hybrid の VLM server (vLLM 0.29.0 + Qwen3-VL-8B)
# 1 つの RUN で入れる: 同じ wheel (torch 2.11.0 は default と groot-pick の両方) を cache からの
# hardlink で共有させ、image を小さくする。
COPY ramen/pixi.toml ramen/pixi.lock ./
COPY ramen/scripts/ scripts/
COPY ramen/inference/desktop/pixi.toml ramen/inference/desktop/pixi.lock inference/desktop/
RUN export PIXI_CACHE_DIR=/tmp/pixi-cache UV_CACHE_DIR=/tmp/uv-cache \
    && pixi install --frozen -e runtime \
    && pixi install --frozen --manifest-path inference/desktop/pixi.toml \
    && pixi install --frozen --manifest-path inference/desktop/pixi.toml -e groot-pick \
    && pixi install --frozen --manifest-path inference/desktop/pixi.toml -e vlm \
    && rm -rf /tmp/pixi-cache /tmp/uv-cache

# GR00T pick の worker は groot_pick_legs.py が repo root (/app/ramen) からのパス
# model/subtask_policy_training/.venv/bin/python で起動する (開発機の uv の .venv)。コードは変えず、
# そこに groot-pick 環境の python を置く。
RUN mkdir -p model/subtask_policy_training/.venv/bin \
    && ln -s /app/ramen/inference/desktop/.pixi/envs/groot-pick/bin/python \
         model/subtask_policy_training/.venv/bin/python

# unitree SDK の interface 指定の config から <Tracing> を落とす。cyclonedds 0.10.2 はこの
# block を含む config で Domain を作ると glibc の _FORTIFY_SOURCE に引っかかって core dump し、
# ChannelFactoryInitialize(0, <interface>) が全滅する (理由は tools/patch_unitree_tracing.py)。
# runtime env は SDK をソースのまま (editable) 参照するので、install の後に当てても効く。
COPY tools/patch_unitree_tracing.py /tmp/patch_unitree_tracing.py
RUN pixi run --frozen -e runtime python /tmp/patch_unitree_tracing.py \
      third_party/unitree_sdk2_python/unitree_sdk2py/core/channel_config.py \
    && rm /tmp/patch_unitree_tracing.py

# vLLM の Conv3dLayer (Qwen3-VL の画像エンコーダの入口) に、Thor (sm_110) で F.linear の cuBLAS が
# 落ちたときだけ畳み込み (F.conv3d、cuDNN) に切り替える fallback を入れる。報告元の直し方
# (ms1design/thorllm の Patch 6) を、落ちたときだけ効く形にしたもの。理由と動きは
# tools/patch_vllm_conv3d_sm110.py。RAMEN_VLLM_CONV3D=conv で最初から畳み込み。
COPY tools/patch_vllm_conv3d_sm110.py /tmp/patch_vllm_conv3d_sm110.py
RUN pixi run --frozen --manifest-path inference/desktop/pixi.toml -e vlm \
      python /tmp/patch_vllm_conv3d_sm110.py \
    && rm /tmp/patch_vllm_conv3d_sm110.py

# --- 6) build 時の確認 ---------------------------------------------------------
# GPU は無いので推論はできないが、「焼けたのに会場で import から落ちる」類はここで全部出す。
# torch <-> numpy の受け渡しは NGC 25.08 で全推論が落ちた件の再発防止 (build は緑なのに
# 実推論だけ壊れる形だった)。runtime は numpy 1.26.4、53D は numpy 2.2.6。
RUN cat > /tmp/probe_runtime.py <<'PY'
import numpy as np
import torch

assert torch.__version__.startswith("2.12.1+cu130"), torch.__version__
assert "sm_110" in torch._C._cuda_getArchFlags(), torch._C._cuda_getArchFlags()
torch.from_numpy(np.zeros(3, dtype=np.float32)).numpy()

# torch / cv2 の後に pinocchio (libstdc++ の読み込み順、scripts/activate_runtime.sh)
import cv2, ultralytics, pinocchio, hydra, omegaconf, lingbot_vision  # noqa: E401,F401
import zmq, msgpack, websockets  # noqa: E401,F401

from importlib.metadata import version
from cyclonedds.domain import DomainParticipant  # noqa: F401
from unitree_sdk2py.g1.loco.g1_loco_api import LOCO_SERVICE_NAME
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

assert LOCO_SERVICE_NAME == "sport", LOCO_SERVICE_NAME  # "loco" 版だと全 RPC が 3102
assert version("cyclonedds") == "0.10.2", version("cyclonedds")
CRC()  # crc_aarch64.so が無いとここで落ちる
ChannelFactoryInitialize(0, "lo")  # <Tracing> が残っていると core dump する
print("[build] runtime env OK:", torch.__version__, "numpy", np.__version__, "sm_110",
      "| cyclonedds", version("cyclonedds"), "| unitree SDK (sport, CRC, iface init)")
PY
RUN pixi run --frozen -e runtime python /tmp/probe_runtime.py && rm /tmp/probe_runtime.py

RUN cat > /tmp/probe_desktop.py <<'PY'
import numpy as np
import torch

assert torch.__version__.startswith("2.11.0+cu130"), torch.__version__
assert "sm_110" in torch._C._cuda_getArchFlags(), torch._C._cuda_getArchFlags()
torch.from_numpy(np.zeros(3, dtype=np.float32)).numpy()

import lerobot
from lerobot.policies.groot import modeling_groot  # noqa: F401
import transformers, hydra, omegaconf, lingbot_vision, yaml  # noqa: E401,F401

assert lerobot.__version__ == "0.6.1", lerobot.__version__
print("[build] inference/desktop env OK:", torch.__version__, "numpy", np.__version__,
      "sm_110 | lerobot", lerobot.__version__, "| transformers", transformers.__version__)
PY
RUN pixi run --frozen --manifest-path inference/desktop/pixi.toml python /tmp/probe_desktop.py \
    && rm /tmp/probe_desktop.py

# GR00T pick の worker: 起動されるのと同じパス (.venv/bin/python) の python で確かめる。
RUN cat > /tmp/probe_pick.py <<'PY'
import importlib.metadata as md
import sys

import numpy as np
import torch

assert sys.version.startswith("3.12.11"), sys.version  # #152 の .venv と同じ
assert torch.__version__.startswith("2.11.0+cu130"), torch.__version__
assert "sm_110" in torch._C._cuda_getArchFlags(), torch._C._cuda_getArchFlags()
torch.from_numpy(np.zeros(3, dtype=np.float32)).numpy()
assert md.version("lerobot") == "0.6.0", md.version("lerobot")
# real_groot_n17_worker.py が使うもの (GR00T N1.7 と公式の前処理・後処理、遅延 import の 2 つ、cv2)
from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config  # noqa: F401
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors_from_pretrained  # noqa: F401
import accelerate, safetensors, cv2  # noqa: E401,F401
print("[build] groot-pick env OK:", sys.version.split()[0], torch.__version__, "sm_110 | lerobot",
      md.version("lerobot"), "| accelerate", accelerate.__version__)
PY
RUN model/subtask_policy_training/.venv/bin/python /tmp/probe_pick.py && rm /tmp/probe_pick.py

# VLM server: vLLM と、FlashInfer の事前 compile 済み kernel に Qwen3-VL-8B の言語部分
# (bf16 / head_dim 128) の prefill・decode があり、sm_110a の機械語が入っていること。
# Triton は TRITON_PTXAS_PATH の ptxas (CUDA 13) で sm_110a に compile できること。
# vLLM の GPU の部品 (_C_stable_libtorch) は driver (libcuda.so.1) が要るので build 中は import しない。
RUN cat > /tmp/probe_vlm.py <<'PY'
import os, pathlib, subprocess, tempfile

import torch
import flashinfer, flashinfer_jit_cache, vllm

assert vllm.__version__ == "0.29.0", vllm.__version__
assert torch.__version__.startswith("2.13.0+cu130"), torch.__version__
assert "sm_110" in torch._C._cuda_getArchFlags(), torch._C._cuda_getArchFlags()
cache = pathlib.Path(flashinfer_jit_cache.__file__).parent / "jit_cache"
common = "dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False"
for name in (f"batch_prefill_with_kv_cache_{common}_f16qk_False", f"batch_decode_with_kv_cache_{common}"):
    so = cache / name / f"{name}.so"
    assert so.is_file(), so
    elf = subprocess.run(["cuobjdump", "--list-elf", str(so)], capture_output=True, text=True).stdout
    assert "sm_110" in elf, f"{so.name} に sm_110 の機械語が無い"
ptxas = os.environ["TRITON_PTXAS_PATH"]
with tempfile.TemporaryDirectory() as d:
    src = pathlib.Path(d, "k.ptx")
    src.write_text(".version 9.0\n.target sm_110a\n.address_size 64\n.visible .entry k() { ret; }\n")
    subprocess.run([ptxas, "-arch=sm_110a", str(src), "-o", str(pathlib.Path(d, "k.cubin"))], check=True)
print("[build] vlm env OK: vllm", vllm.__version__, "| torch", torch.__version__, "sm_110 | flashinfer",
      flashinfer.__version__, "jit-cache (Qwen3-VL prefill/decode sm_110) | TRITON_PTXAS_PATH sm_110a")
PY
RUN pixi run --frozen --manifest-path inference/desktop/pixi.toml -e vlm python /tmp/probe_vlm.py \
    && rm /tmp/probe_vlm.py

# Conv3dLayer の fallback の動き (GPU 無しで、計算の中身を差し替えて確かめる):
#   普段は行列の掛け算 (develop と同じ) / cuBLAS の例外なら畳み込みに切り替わり以後もそのまま /
#   cuBLAS 以外の例外は握りつぶさない / RAMEN_VLLM_CONV3D=conv なら最初から畳み込み
RUN cat > /tmp/probe_conv3d.py <<'PY'
import os
import subprocess
import sys
from types import SimpleNamespace

from vllm.model_executor.layers.conv import Conv3dLayer as C

CUBLAS = "CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasLtCreate(&handle)`"


def layer(fail_with=None):
    calls = []

    def mulmat(x):
        calls.append("mulmat")
        if fail_with:
            raise RuntimeError(fail_with)
        return "mulmat"

    def conv(x):
        calls.append("conv")
        return "conv"

    ns = SimpleNamespace(enable_linear=True, calls=calls, _forward_mulmat=mulmat, _forward_conv=conv)
    ns._forward_mulmat_or_conv = lambda x: C._forward_mulmat_or_conv(ns, x)
    return ns


C._ramen_use_conv = False
assert C.forward_cuda(layer(), None) == "mulmat"
assert C.forward_native(layer(), None) == "mulmat"
failing = layer(CUBLAS)
assert C.forward_cuda(failing, None) == "conv" and failing.calls == ["mulmat", "conv"], failing.calls
assert C._ramen_use_conv is True
after = layer()
assert C.forward_cuda(after, None) == "conv" and after.calls == ["conv"], after.calls
C._ramen_use_conv = False
try:
    C.forward_cuda(layer("some other error"), None)
    sys.exit("cuBLAS 以外の例外を握りつぶした")
except RuntimeError as exc:
    assert "some other error" in str(exc)
code = "from vllm.model_executor.layers.conv import Conv3dLayer as C; assert C._ramen_use_conv is True"
subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ, RAMEN_VLLM_CONV3D="conv"))
print("[build] vLLM Conv3dLayer fallback OK (normal=mulmat / cuBLAS error -> conv, sticky / "
      "other errors raised / RAMEN_VLLM_CONV3D=conv)")
PY
RUN pixi run --frozen --manifest-path inference/desktop/pixi.toml -e vlm python /tmp/probe_conv3d.py \
    && rm /tmp/probe_conv3d.py

# --- 7) 本体のコード ------------------------------------------------------------
# 環境の層の後に置き、コードを直しても環境を入れ直さないようにする。
COPY ramen/ ./
COPY docker/venue_entry.sh /usr/local/bin/ramen-venue

# 運営の conformance 一式 (template と同じ並び)。image の環境のまま回せるように /app に置く:
#   docker run --rm <image> pixi run --as-is -e runtime python /app/conformance.py --lane decoupled
# components/server.py は /app/ramen の本番と同じ受け口・送り口を使う (conformance 専用)。
COPY conformance.py requirements.txt /app/
COPY boundary/ /app/boundary/
COPY mocks/ /app/mocks/
COPY components/ /app/components/
# 重みの事前取得とネット無しの確認 (WEIGHTS.md)。/app/ramen の本物の解決関数を呼ぶ
COPY tools/prefetch_weights.py /app/tools/prefetch_weights.py

# 会場は実行時オフライン: 重みは HF の cache (読み取り専用で mount) から読み、取りに行かない。
# YOLO_OFFLINE: ultralytics は import 時に DNS でネットの有無を調べ、推論の開始時に Google Analytics
# へ利用統計を送る (GB10 で strace、2026-09-25)。true で両方止まる。
# VLM の compile 結果の置き場は /cache (container は run ごとに作り直すので host の directory を mount)。
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    YOLO_OFFLINE=true \
    RAMEN_VLM_CACHE_DIR=/cache

# コピー漏れをここで止める: 会場で動く入口を、それぞれの環境で import する。
#   runtime    : entrypoint (と --help で CLI の定義)、VLM の起動と待ち
#   desktop    : GR00T 53D の worker
#   groot-pick : GR00T pick の worker の script (依存は上の probe_pick で確認済み)
#   vlm        : VLM の起動 script
RUN pixi run --as-is -e runtime python -c \
      "import inference.desktop.entrypoint, inference.desktop.pick_leg_hybrid.vlm_server" \
    && pixi run --as-is -e runtime python -m inference.desktop.entrypoint --help > /dev/null \
    && pixi run --as-is --manifest-path inference/desktop/pixi.toml python -c \
      "import inference.desktop.lower_policy.policies.groot_worker" \
    && model/subtask_policy_training/.venv/bin/python -m py_compile \
      model/subtask_policy_training/deployment/real_groot_n17_worker.py \
    && bash -n inference/desktop/pick_leg_hybrid/run_venue_vlm_server.sh \
    && echo "[build] ramen code OK (entrypoint / VLM server / GR00T 53D worker / pick worker)"

# --- 8) 起動口 ------------------------------------------------------------------
# 会場: docker run … <image> --stage N --actuate。`-` で始まらない引数はそのまま実行する。
ENTRYPOINT ["/usr/local/bin/ramen-venue"]
CMD []
