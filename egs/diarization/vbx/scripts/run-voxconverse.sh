#!/usr/bin/env bash
# VoxConverse (16 kHz: test; dev and test with --root of the original layout):
# VBx diarization: Silero VAD -> x-vectors -> AHC + VB-HMM (PLDA) -> DER/VAD scores.
#
#   run-voxconverse.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]
#
# e.g.
#   run-voxconverse.sh --exp-dir /media/raid/diarization/results/vbx/exp-001 --model redimnet \
#       --vad-silero-prob --onset 0.99 --offset 0.67 --merge-gap 0.8 --min-dur 0

source "$(dirname "$0")/common.sh"
run_dataset voxconverse "$@"
