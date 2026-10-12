#!/usr/bin/env bash
# Robusta (8 kHz, simulated from LibriSpeech):
# ECAPA diarization: extract -> diarize -> DER/VAD scores.
#
#   run-robusta.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset robusta "$@"
