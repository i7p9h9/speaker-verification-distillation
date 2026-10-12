#!/usr/bin/env bash
# Callhome (8 kHz, references from meta.csv):
# ECAPA diarization: extract -> diarize -> DER/VAD scores.
#
#   run-callhome.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset callhome "$@"
