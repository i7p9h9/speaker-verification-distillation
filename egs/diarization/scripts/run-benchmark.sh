#!/usr/bin/env bash
# The whole benchmark (benchmark.py, see there): Nemotron; Silero VAD tuned per
# dataset, ecapa and vbx with these thresholds; their mean, ecapa and vbx with it;
# the summary tables (aggregate.py).  Finished steps are skipped on a re-run.
#
#   scripts/run-benchmark.sh [--bench-dir DIR] [--datasets a,b] [benchmark.py options]
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
py() { uv run --project "$HERE/../../.." python "$@"; }

BENCH_ARGS=()
for ((i = 1; i <= $#; i++)); do
    [[ ${!i} == --bench-dir ]] && { j=$((i + 1)); BENCH_ARGS=(--bench-dir "${!j}"); }
    [[ ${!i} == --bench-dir=* ]] && BENCH_ARGS=("${!i}")
done

py "$HERE/benchmark.py" nemotron "$@"
py "$HERE/benchmark.py" silero tune ecapa vbx --vad tuned "$@"
py "$HERE/benchmark.py" average ecapa vbx --vad average "$@"
py "$HERE/aggregate.py" "${BENCH_ARGS[@]}"
