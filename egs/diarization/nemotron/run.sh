#!/usr/bin/env bash
# Offline Nemotron-3-Diarization over a directory, inside Docker.
#
#   egs/diarization/nemotron/run.sh <wav-dir> <out-dir> [infer.py args...]
#
# Writes <out-dir>/<rel>.npz (speaker probabilities) and <rel>.json (segments);
# convert them with to_rttm.py.  The image is built on first use; the model is
# cached in $HF_CACHE (default ~/.cache/huggingface).  HF_TOKEN is passed through.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,8p' "$0"
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN_DIR="$(realpath "$1")"
OUT_DIR="$(realpath -m "$2")"
shift 2

IMAGE="${IMAGE:-nemotron-diar:latest}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
GPUS="${GPUS:-all}"

if [[ -z "$(docker images -q "$IMAGE")" ]]; then
    docker build -t "$IMAGE" "$HERE"
fi

mkdir -p "$OUT_DIR" "$HF_CACHE"

docker run --rm \
    --gpus "$GPUS" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e NUMBA_CACHE_DIR=/tmp/numba \
    -e HF_TOKEN \
    -v "$HF_CACHE":/hf-cache \
    -v "$HERE/infer.py":/app/infer.py:ro \
    -v "$IN_DIR":/data/in:ro \
    -v "$OUT_DIR":/data/out \
    "$IMAGE" --in-dir /data/in --out-dir /data/out "$@"
