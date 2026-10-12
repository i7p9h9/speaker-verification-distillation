# Shared part of the vbx run-*.sh, sourced by them.
#
# Every run-*.sh takes
#   --exp-dir DIR      results root: DIR/<dataset>/<set>/{silero,xvec-<hash>/diar-<hash>},
#                      DIR/<dataset>/results.tsv
#   --sets a,b         subset of the dataset's sets (default: see below)
#   --root DIR         dataset location (default: see below)
#   --stages LIST      extract,diarize,score (default: all)
#   --force            re-run the selected stages even if up to date
#   --collars C ...    DER collars (default: 0 0.25)
#   --skip-overlap     score without the reference overlap regions
#   --model NAME       x-vector extractor preset (models.py): resnet101-8k, resnet101-16k,
#                      redimnet, redimnet-tel, redimnet-tel-small (default: resnet101-8k/16k by the rate)
#   --oracle-vad       reference speech instead of Silero VAD
#   --silero-root DIR  reuse the Silero caches DIR/<dataset>/<set>/silero (e.g. of an ecapa run)
# and any option of extract.py (--vad-silero-prob --onset .. --offset .. --merge-gap ..
# --min-dur .., --batch-size, ...) or diarize.py (--threshold, --Fa, --Fb, --loopP,
# --init-smoothing, --plda, --lda-dim, ...) directly.  Stages whose parameters did
# not change are skipped (model files are compared by md5); see pipeline.py.
#
# Tuning (tune.py, on a share of a set; prints the options to pass here):
#   tune.py --target vad --silero-dir <exp>/<dataset>/dev/silero --ref-dir <root>/prepared/dev/rttm
#   tune.py --target vbx --xvec-dir <exp>/<dataset>/dev/xvec-<hash> --ref-dir <root>/prepared/dev/rttm --model NAME
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

# ResNet101 of the dataset's rate unless --model; extract.py resamples the audio to the
# rate of the chosen model (e.g. --model resnet101-16k on an 8 kHz dataset)
recipe_args() { ARGS=(--default-model "resnet101-$((RATE / 1000))k" "${ARGS[@]}"); }
