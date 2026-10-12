#!/usr/bin/env bash
# AMI (Mix-Headset, 16 kHz; the meetings with audio: test):
# VBx diarization: Silero VAD -> x-vectors -> AHC + VB-HMM (PLDA) -> DER/VAD scores.
#
#   run-ami.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset ami "$@"
