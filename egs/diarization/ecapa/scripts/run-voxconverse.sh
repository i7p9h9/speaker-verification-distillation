#!/usr/bin/env bash
# VoxConverse (16 kHz: test; dev and test with --root of the original layout):
# ECAPA diarization: extract -> diarize -> DER/VAD scores.
#
#   run-voxconverse.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]
#
# e.g.
#   run-voxconverse.sh --exp-dir /media/raid/diarization/results/ecapa/exp-003 \
#       --arch ECAPA_TDNN_VAD --ckpt /media/raid/diarization/exps/ecapa-tdnn-003/last.ckpt \
#       --batch-size 32 --onset 1.5 --offset 0.5

source "$(dirname "$0")/common.sh"
run_dataset voxconverse "$@"
