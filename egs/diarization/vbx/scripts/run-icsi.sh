#!/usr/bin/env bash
# ICSI (8 kHz, 10-minute pieces of the meetings):
# VBx diarization: Silero VAD -> x-vectors -> AHC + VB-HMM (PLDA) -> DER/VAD scores.
#
#   run-icsi.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset icsi "$@"
