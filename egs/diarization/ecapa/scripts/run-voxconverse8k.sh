#!/usr/bin/env bash
# VoxConverse test at 8 kHz (references from meta.csv):
# ECAPA diarization: extract -> diarize -> DER/VAD scores.
#
#   run-voxconverse8k.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset voxconverse8k "$@"
