# syntax=docker/dockerfile:1
#
# Policy SERVER container — runs on the Jetson AGX Thor.
#   sm_110 (Blackwell) / JetPack 7.2 / L4T R39.2 / CUDA 13 / aarch64 / 128 GB unified
#
# Base image (運営 2026-08 訂正): JetPack 7 は unified Arm CUDA なので **標準 NGC CUDA
# image**（l4t-* ではない）。framework から始めるなら nvcr.io/nvidia/pytorch:25.08-py3 も可。
#
#   ⚠️ image は **linux/arm64 (aarch64)** でビルドすること。x86 host からは
#      docker buildx --platform linux/arm64（QEMU）で cross-build。
#   ⚠️ nvcr.io は public でも NGC login 必須:
#      docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>
#
# Build (context = repo root):
#   docker buildx build --platform linux/arm64 -f docker/Dockerfile.thor \
#     -t <registry>/ramen-thor:<tag> --push .

ARG BASE=nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
# 代替: ARG BASE=nvcr.io/nvidia/pytorch:25.08-py3
FROM ${BASE}

WORKDIR /app

# --- template deps のみ (onboarding: hold-still Policy) ---
# cuda:*-ubuntu24.04 base は pip 未同梱 + Ubuntu 24.04 は PEP 668 (externally-managed)
# なので、python3-pip を入れ --break-system-packages で system へ install する。
# (RAMEN-Ori 段階で venv 化してもよいが、hold-still onboarding はこれで十分)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
      && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# >>> RAMEN-Ori (#115) の model deps はここに追加 (torch は sm_110/CUDA13 build) <<<
#     素の pip torch は dGPU wheel で sm_110 非対応。Thor install path を使う。
#     model weights は image に焼き込まない = manifest の weights_uri から取得。

# --- submission code (boundary は無改変) ---
COPY . ./

# onboarding は同梱の hold-still Policy。RAMEN-Ori adapter が入ったら差し替える。
CMD ["python3", "-u", "components/server.py", \
     "--lane", "decoupled", "--host", "0.0.0.0", "--port", "8765"]
