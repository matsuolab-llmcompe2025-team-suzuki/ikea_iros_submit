#!/usr/bin/env bash
# Sourced by Pixi for the runtime environment only. Preserve CUDA/other paths;
# do not change the system linker, user shell configuration, or default env.
: "${CONDA_PREFIX:?Pixi runtime activation requires CONDA_PREFIX}"
case "${LD_LIBRARY_PATH:-}" in
    "${CONDA_PREFIX}/lib"|"${CONDA_PREFIX}/lib":*) ;;
    *) export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
