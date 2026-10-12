#!/usr/bin/env bash
# VoxConverse test at 8 kHz (references from meta.csv):
# VBx diarization: Silero VAD -> x-vectors -> AHC + VB-HMM (PLDA) -> DER/VAD scores.
#
#   run-voxconverse8k.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset voxconverse8k "$@"
