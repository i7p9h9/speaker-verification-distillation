#!/usr/bin/env bash
# Switchboard / NIST 2000 Disk6 (8 kHz):
# ECAPA diarization: extract -> diarize -> DER/VAD scores.
#
#   run-switchboard.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset switchboard "$@"
