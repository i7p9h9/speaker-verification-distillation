#!/usr/bin/env bash
# AISHELL-4 test (channel 0 of the array, 16 kHz):
# VBx diarization: Silero VAD -> x-vectors -> AHC + VB-HMM (PLDA) -> DER/VAD scores.
#
#   run-aishell4.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset aishell4 "$@"
