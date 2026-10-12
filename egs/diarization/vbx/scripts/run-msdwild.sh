#!/usr/bin/env bash
# MSDWild (16 kHz; the sets with audio: few_val):
# VBx diarization: Silero VAD -> x-vectors -> AHC + VB-HMM (PLDA) -> DER/VAD scores.
#
#   run-msdwild.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset msdwild "$@"
