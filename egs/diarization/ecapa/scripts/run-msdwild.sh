#!/usr/bin/env bash
# MSDWild (16 kHz; the sets with audio: few_val):
# ECAPA diarization: extract -> diarize -> DER/VAD scores.
#
#   run-msdwild.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset msdwild "$@"
