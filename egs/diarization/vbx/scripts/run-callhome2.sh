#!/usr/bin/env bash
# Callhome_v2 (8 kHz, the Kaldi callhome_diarization key; sets part1, part2, all):
# VBx diarization: Silero VAD -> x-vectors -> AHC + VB-HMM (PLDA) -> DER/VAD scores.
#
#   run-callhome2.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset callhome2 "$@"
