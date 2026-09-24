# syntax=docker/dockerfile:1
#
# 提出 image (Thor 側) — Team RAMEN (#10)。
#   Jetson AGX Thor: aarch64 / sm_110 (Blackwell) / JetPack 7.x / CUDA 13
#
# 本体 iros_2026_ramen の推論をそのまま動かす。Python の環境は本体の pixi (ramen/ に同じ
# 相対パスでコピー) で作り、この Dockerfile には Python 以外 (apt / CycloneDDS の C ライブラリ /
# unitree SDK のソース / pixi 本体) だけを書く。
#
# ⚠️ 今は環境の部分だけ。本体コードの COPY と起動コマンドは、submit と inference の
#    つなぎを決めてから足す (それまでこの image は競技には使えない)。
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

# --- 2) CycloneDDS の C ライブラリ --------------------------------------------
# python binding の cyclonedds 0.10.2 は linux/aarch64 の wheel が無く、pixi install が
# CYCLONEDDS_HOME を見て sdist から build する。binding と同じ版を tag で固定する
# (releases/0.10.x の先端は 0.10.5 で、binding とずれる)。
ARG CYCLONEDDS_REF=0.10.2
RUN git clone --depth 1 --branch "${CYCLONEDDS_REF}" \
      https://github.com/eclipse-cyclonedds/cyclonedds /tmp/cyclonedds \
    && cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
         -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_IDLC=ON \
    && cmake --build /tmp/cyclonedds/build --target install -j"$(nproc)" \
    && ldconfig \
    && rm -rf /tmp/cyclonedds
# build 時 (binding の link) と run 時 (import 時の .so 探索) の両方で要る。
ENV CYCLONEDDS_HOME=/usr/local

# --- 3) pixi (版と checksum を固定) --------------------------------------------
ARG PIXI_VERSION=v0.72.0
ARG PIXI_SHA256=8b48fd8b315552ee48d340e89d654a177d1f001810ab741f51f7dcdd7e00e1c1
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
#   runtime                     : entrypoint 本体 / YOLO / RAMEN-Ori (py3.10)
#   inference/desktop の default : GR00T 53D の worker (py3.12、lerobot 0.6.1)
#   GR00T pick の worker         : #152 の .venv の版を確かめてから足す
COPY ramen/pixi.toml ramen/pixi.lock ./
COPY ramen/scripts/ scripts/
COPY ramen/inference/desktop/pixi.toml ramen/inference/desktop/pixi.lock inference/desktop/
RUN PIXI_CACHE_DIR=/tmp/pixi-cache UV_CACHE_DIR=/tmp/uv-cache pixi install --frozen -e runtime \
    && PIXI_CACHE_DIR=/tmp/pixi-cache UV_CACHE_DIR=/tmp/uv-cache \
       pixi install --frozen --manifest-path inference/desktop/pixi.toml \
    && rm -rf /tmp/pixi-cache /tmp/uv-cache

# unitree SDK の interface 指定の config から <Tracing> を落とす。cyclonedds 0.10.2 はこの
# block を含む config で Domain を作ると glibc の _FORTIFY_SOURCE に引っかかって core dump し、
# ChannelFactoryInitialize(0, <interface>) が全滅する (理由は tools/patch_unitree_tracing.py)。
# runtime env は SDK をソースのまま (editable) 参照するので、install の後に当てても効く。
COPY tools/patch_unitree_tracing.py /tmp/patch_unitree_tracing.py
RUN pixi run --frozen -e runtime python /tmp/patch_unitree_tracing.py \
      third_party/unitree_sdk2_python/unitree_sdk2py/core/channel_config.py \
    && rm /tmp/patch_unitree_tracing.py

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

# 起動コマンドは、submit と inference のつなぎを決めてから入れる。
CMD ["bash"]
