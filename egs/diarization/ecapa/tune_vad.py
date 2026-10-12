"""Optuna search of the VAD post-processing parameters of ``diarize.py``.

Tunes ``--onset``, ``--offset`` (<= onset), ``--merge-gap`` and ``--min-dur``
on a random ``--fraction`` of the files of an ``extract.py`` output directory
(10% by default) against reference RTTMs.  The objective is the total DER at
``--collar`` (``--metric der``, full diarization per trial) or the VAD error
MS + FA in % of reference speech (``--metric vad``, no clustering, much faster).

The first trial is the starting point given by the usual ``diarize.py``
options (``--onset 1.5 --offset 0 ...``, otherwise its defaults); every other
``diarize.py`` option (clustering, ``--vad-lda``, ``--vad-silero DIR
--vad-silero-prob``, ``--num-speakers``) is passed through and kept fixed.  The onset range defaults to the 1st..99th percentile of
the scores on the subset, since logit scales differ between models.

Prints the best parameters as ``diarize.py`` / ``run-*.sh`` options and writes
them with the baseline and the subset to ``--out`` (json).  ``--eval-all``
also scores the baseline and the best parameters on all files, to see how
well the subset result transfers.

Usage:
    python egs/diarization/ecapa/tune_vad.py --npz-dir exp/VoxConverse/dev/npz \
        --ref-dir /media/raid/voice/VoxConverse/voxconverse/dev \
        --onset 1.5 --offset 0 --merge-gap 0.1 --min-dur 0.5 [--n-trials 200] [--eval-all]
"""

from __future__ import annotations

import argparse
import json
import math
import typing as tp
from pathlib import Path

import numpy as np
import optuna

try:
    import diarize
    from rttm import read_rttm_dir
    from score_der import Score, score_file
except ImportError:
    from . import diarize
    from .rttm import read_rttm_dir
    from .score_der import Score, score_file

PARAMS = ("onset", "offset", "merge_gap", "min_dur")


def parse_args() -> tp.Tuple[argparse.Namespace, argparse.Namespace]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    p.add_argument("--npz-dir", type=Path, required=True)
    p.add_argument("--ref-dir", type=Path, required=True)
    p.add_argument("--fraction", type=float, default=0.1, help="share of the files to tune on")
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--metric", choices=["der", "vad"], default="der")
    p.add_argument("--collar", type=float, default=0.25)
    p.add_argument("--skip-overlap", action="store_true")
    p.add_argument("--oracle-speakers", action="store_true", help="number of speakers from the reference")
    p.add_argument("--onset-range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    p.add_argument("--max-merge-gap", type=float, default=1.0, help="seconds")
    p.add_argument("--max-min-dur", type=float, default=1.0, help="seconds")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--sampler-seed", type=int, default=0)
    p.add_argument("--storage", default=None, help="optuna storage to resume, e.g. sqlite:///tune.db")
    p.add_argument("--study-name", default="vad")
    p.add_argument("--eval-all", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="json with the result (default: <npz-dir>/../tune_vad.json)")
    args, rest = p.parse_known_args()

    dargs = diarize.build_parser().parse_args(["--npz-dir", str(args.npz_dir), "--out-dir", "-", *rest])
    if dargs.oracle_vad_dir is not None or dargs.oracle_speakers_dir is not None:
        p.error("oracle dirs are not supported, use --oracle-speakers")
    if dargs.vad_silero is not None and not dargs.vad_silero_prob:
        p.error("--vad-silero segments have nothing to tune, add --vad-silero-prob")
    if not 0 < args.fraction <= 1:
        p.error("--fraction must be in (0, 1]")
    return args, dargs


class File(tp.NamedTuple):
    file_id: str
    data: tp.Dict[str, np.ndarray]
    ref: tp.List[tp.Tuple[float, float, str]]


def load_files(paths: tp.List[Path], refs: dict, dargs: argparse.Namespace, sources: tuple) -> tp.List[File]:
    files = []
    for path in paths:
        with np.load(path) as npz:
            data = {k: npz[k] for k in diarize.NPZ_KEYS}
            data["vad"] = diarize.frame_scores(npz, path.stem, dargs, *sources)
        files.append(File(path.stem, data, refs[path.stem]))
    return files


def cached_spectral_cluster() -> tp.Callable:
    """``diarize.spectral_cluster`` memoized on its input.  Clustering depends on
    the VAD parameters only through the set of kept windows, which repeats a lot
    between trials."""
    cluster = diarize.spectral_cluster
    cache: tp.Dict[tuple, np.ndarray] = {}

    def cached(emb: np.ndarray, *args: tp.Any) -> np.ndarray:
        key = (emb.shape, hash(emb.tobytes()), args)
        if key not in cache:
            cache[key] = cluster(emb, *args)
        return cache[key]

    return cached


def evaluate(files: tp.List[File], dargs: argparse.Namespace, args: argparse.Namespace) -> Score:
    total = Score()
    for f in files:
        if args.metric == "vad":
            shift = float(f.data["frame_shift"])
            segments = diarize.mask_to_segments(diarize.hysteresis(f.data["vad"], dargs.onset, dargs.offset), shift)
            segments = diarize.merge_and_filter(segments, dargs.merge_gap, dargs.min_dur)
            hyp = [(s, e, "speech") for s, e in segments]
        else:
            num_speakers = dargs.num_speakers
            if args.oracle_speakers:
                num_speakers = len({spk for _, _, spk in f.ref}) or None
            hyp = diarize.diarize(f.data, dargs, None, num_speakers)
        total += score_file(f.ref, hyp, args.collar, args.skip_overlap, 0.01)
    return total


def objective_value(score: Score, metric: str) -> float:
    return score.der if metric == "der" else score.vad_rate(score.vad_miss + score.vad_fa)


def with_params(dargs: argparse.Namespace, params: tp.Mapping[str, float]) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(dargs), **params})


def describe(score: Score, metric: str) -> str:
    vad = (
        f"VAD {score.vad_rate(score.vad_miss + score.vad_fa):.2f} "
        f"(MS {score.vad_rate(score.vad_miss):.2f} FA {score.vad_rate(score.vad_fa):.2f})"
    )
    if metric == "vad":
        return vad
    return (
        f"DER {score.der:.2f} (MS {score.rate(score.miss):.2f} FA {score.rate(score.fa):.2f} "
        f"SC {score.rate(score.conf):.2f}), {vad}"
    )


def as_options(params: tp.Mapping[str, float]) -> str:
    return " ".join(f"--{name.replace('_', '-')} {params[name]:.4g}" for name in PARAMS)


def main() -> None:
    args, dargs = parse_args()
    refs = read_rttm_dir(args.ref_dir)
    paths = [p for p in sorted(args.npz_dir.rglob("*.npz")) if p.stem in refs]
    if not paths:
        raise SystemExit(f"no .npz in {args.npz_dir} with a reference in {args.ref_dir}")
    rng = np.random.default_rng(args.subset_seed)
    n = max(1, math.ceil(args.fraction * len(paths)))
    subset = sorted(rng.choice(len(paths), n, replace=False))
    sources = diarize.load_vad_sources(dargs)
    files = load_files([paths[i] for i in subset], refs, dargs, sources)
    hours = sum(float(f.data["frame_shift"]) * len(f.data["vad"]) for f in files) / 3600
    print(f"tuning on {n}/{len(paths)} files, {hours:.2f} h, metric {args.metric} at collar {args.collar:g}")

    diarize.spectral_cluster = cached_spectral_cluster()
    baseline = {name: float(getattr(dargs, name)) for name in PARAMS}
    if args.onset_range is not None:
        lo, hi = args.onset_range
    else:
        lo, hi = np.percentile(np.concatenate([f.data["vad"] for f in files]), [1, 99])
    lo, hi = min(lo, baseline["offset"], baseline["onset"]), max(hi, baseline["onset"])

    def objective(trial: optuna.Trial) -> float:
        onset = trial.suggest_float("onset", lo, hi)
        params = {
            "onset": onset,
            "offset": trial.suggest_float("offset", lo, onset),
            "merge_gap": trial.suggest_float("merge_gap", 0.0, max(args.max_merge_gap, baseline["merge_gap"])),
            "min_dur": trial.suggest_float("min_dur", 0.0, max(args.max_min_dur, baseline["min_dur"])),
        }
        score = evaluate(files, with_params(dargs, params), args)
        trial.set_user_attr("score", describe(score, args.metric))
        return objective_value(score, args.metric)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.storage is not None,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )
    if not study.trials:
        study.enqueue_trial(baseline)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    base_trial = study.trials[0]
    print(f"\nbaseline {as_options(base_trial.params)}\n  {base_trial.user_attrs['score']}")
    print(f"best (trial {best.number}) {as_options(best.params)}\n  {best.user_attrs['score']}")
    result = {
        "metric": args.metric,
        "collar": args.collar,
        "baseline": {"params": base_trial.params, "value": base_trial.value},
        "best": {"params": best.params, "value": best.value, "trial": best.number},
        "options": as_options(best.params),
        "subset": [f.file_id for f in files],
        "fixed": {k: str(v) for k, v in vars(dargs).items() if k not in PARAMS and k not in ("npz_dir", "out_dir")},
    }

    if args.eval_all:
        all_files = load_files(paths, refs, dargs, sources)
        for name, params in (("baseline", base_trial.params), ("best", best.params)):
            score = evaluate(all_files, with_params(dargs, params), args)
            result[name]["all_files"] = objective_value(score, args.metric)
            print(f"all {len(all_files)} files, {name}: {describe(score, args.metric)}")

    out = args.out or args.npz_dir.parent / "tune_vad.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n{out}\nuse: {as_options(best.params)}")


if __name__ == "__main__":
    main()
