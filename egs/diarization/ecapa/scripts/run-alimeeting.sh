#!/usr/bin/env bash
# AliMeeting test (channel 0 of the far-field array, 16 kHz):
# ECAPA diarization: extract -> diarize -> DER/VAD scores.
#
#   run-alimeeting.sh --exp-dir DIR [--sets A,B] [--root DIR] [options]

source "$(dirname "$0")/common.sh"
run_dataset alimeeting "$@"
