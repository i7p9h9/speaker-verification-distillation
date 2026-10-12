# Shared part of the ecapa run-*.sh, sourced by them.
#
# Every run-*.sh takes
#   --exp-dir DIR      results root: DIR/<dataset>/<set>/{npz,diar-<hash>}, DIR/<dataset>/results.tsv
#   --sets a,b         subset of the dataset's sets (default: see below)
#   --root DIR         dataset location (default: see below)
#   --stages LIST      extract,diarize,score (default: all)
#   --force            re-run the selected stages even if up to date
#   --collars C ...    DER collars (default: 0 0.25)
#   --skip-overlap     score without the reference overlap regions
#   --oracle-vad       diarize with the reference speech regions
#   --oracle-speakers  diarize with the reference number of speakers
#   --silero           Silero VAD segments instead of v_t (computed once per set;
#                      + --vad-silero-prob to threshold its probabilities with --onset ...)
# and any option of extract.py (--ckpt, --arch, --win-ms, --batch-size, ...) or
# diarize.py (--onset, --offset, --num-speakers, --vad-lda, ...) directly.
# Stages whose parameters did not change are skipped (checkpoints are compared
# by md5); see pipeline.py.
#
# Datasets (run-<key>.sh; default --root under $DIAR_DATA_DIR, /media/raid/diarization/data):
#   ami (AMI Mix-Headset, 16k: test), callhome (8k: all), callhome2 (Callhome_v2 with the
#   Kaldi key, 8k: part1, part2, all), voxconverse (16k: test; dev too with the original
#   layout, e.g. --root /media/raid/voice/VoxConverse), voxconverse8k (8k: test),
#   aishell4 (channel 0, 16k: test), alimeeting (channel 0, 16k: test), msdwild (16k: few_val),
#   icsi, robusta, switchboard (8k: all); default sets: callhome2 part1,part2, aishell4 test,
#   msdwild few_val, others all their sets.  The first run prepares the dataset in
#   <root>/prepared (../../scripts/prepare_data.py: symlinks, references from the metadata).

SCRIPTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DOC=${BASH_SOURCE[0]}
source "$SCRIPTS_DIR/../../scripts/common.sh"

# audio of another rate than the model's (--sample-rate, 16000 by default) is resampled
recipe_args() {
    local i model_rate=16000
    for ((i = 0; i < ${#ARGS[@]}; i++)); do
        case ${ARGS[i]} in
            --sample-rate) model_rate=${ARGS[i + 1]} ;;
            --sample-rate=*) model_rate=${ARGS[i]#*=} ;;
        esac
    done
    if ((RATE != model_rate)); then ARGS=(--force-resample "${ARGS[@]}"); fi
}
