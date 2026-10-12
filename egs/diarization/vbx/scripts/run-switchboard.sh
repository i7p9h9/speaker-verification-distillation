#!/usr/bin/env bash
# Switchboard / NIST 2000 Disk6 (8 kHz):
# VBx diarization: Silero VAD -> x-vectors -> AHC + VB-HMM (PLDA) -> DER/VAD scores.
#
#   run-switchboard.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset switchboard "$@"
