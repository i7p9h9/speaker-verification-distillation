# Shared part of the run-*.sh of the recipes (ecapa/scripts, vbx/scripts),
# sourced by their common.sh after setting SCRIPTS_DIR (the recipe's scripts
# dir, with its pipeline.py), RECIPE_DOC (file whose header documents the
# recipe's options) and defining recipe_args (adds the recipe's per-dataset
# options to ARGS, given RATE, the dataset's sample rate).
#
# A run-<dataset>.sh is `run_dataset <key> "$@"`: the dataset under $DATA_DIR
# (or --root) is brought into the common layout <root>/prepared/<set>/{wav,rttm}
# by prepare_data.py the first time (marked by <root>/prepared/.ready; run
# prepare_data.py --force to redo it), then its sets are run by pipeline.py.

set -euo pipefail

TOOLS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$TOOLS_DIR/../../.." && pwd)
DATA_DIR=${DIAR_DATA_DIR:-/media/raid/diarization/data}

# scripts/datasets.tsv: key, results name, root under DATA_DIR, sample rate,
# default sets (* = all prepared ones), the set to tune on (benchmark.py)
declare -A DATASETS=()
while IFS=$'\t' read -r key rest; do
    [[ $key == \#* || -z $key ]] || DATASETS[$key]=$rest
done <"$TOOLS_DIR/datasets.tsv"

die() { echo "$(basename "$0"): $*" >&2; exit 1; }
warn() { echo "$(basename "$0"): warning: $*" >&2; }

usage() {
    sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"
    sed -n '3,/^$/{s/^# \{0,1\}//;p}' "$RECIPE_DOC"
}

py() { uv run --project "$REPO_DIR" python "$@"; }

# parse_args DEFAULT_SETS DEFAULT_ROOT "$@"  ->  SETS (array), ROOT, ARGS (array, the rest)
parse_args() {
    local sets=$1
    ROOT=$2
    shift 2
    ARGS=()
    while (($#)); do
        case $1 in
            --sets) sets=$2; shift 2 ;;
            --sets=*) sets=${1#*=}; shift ;;
            --root) ROOT=$2; shift 2 ;;
            --root=*) ROOT=${1#*=}; shift ;;
            -h | --help) usage; exit 0 ;;
            *) ARGS+=("$1"); shift ;;
        esac
    done
    ROOT=${ROOT%/}
    IFS=, read -ra SETS <<<"$sets"
    [[ -d $ROOT ]] || die "no dataset at $ROOT"
}

# add_set NAME WAV_DIR [REF_DIR]: queue a set, skipped with a warning if it has no audio
SET_ARGS=()
add_set() {
    local name=$1 wav=$2
    if ! compgen -G "$wav/*.wav" >/dev/null; then
        warn "set $name: no wav files in $wav, skipped"
        return
    fi
    [[ -z ${3:-} || -d $3 ]] || die "set $name: no reference dir $3"
    SET_ARGS+=(--set "$@")
}

# prepared_sets: the sets of $ROOT/prepared
prepared_sets() {
    local d
    for d in "$ROOT"/prepared/*/; do
        d=$(basename "$d")
        [[ $d == audio ]] || echo "$d"
    done
}

# run_dataset KEY "$@": prepare the dataset if needed, queue its sets, run the pipeline
run_dataset() {
    local key=$1 name rel sets set
    shift
    [[ -n ${DATASETS[$key]:-} ]] || die "unknown dataset $key (${!DATASETS[*]})"
    IFS=$'\t' read -r name rel RATE sets _ <<<"${DATASETS[$key]}"
    parse_args "$sets" "$DATA_DIR/$rel" "$@"
    if [[ ! -f $ROOT/prepared/.ready ]]; then
        py "$TOOLS_DIR/prepare_data.py" "$key" "$ROOT" || die "could not prepare $ROOT"
    fi
    [[ ${SETS[*]} == "*" ]] && mapfile -t SETS < <(prepared_sets)
    for set in "${SETS[@]}"; do
        [[ -d $ROOT/prepared/$set/wav ]] || die "unknown set $set ($(prepared_sets | paste -sd,))"
        add_set "$set" "$ROOT/prepared/$set/wav" "$ROOT/prepared/$set/rttm"
    done
    ((${#SET_ARGS[@]})) || die "nothing to run"
    recipe_args
    py "$SCRIPTS_DIR/pipeline.py" --dataset "$name" "${SET_ARGS[@]}" "${ARGS[@]}"
}
