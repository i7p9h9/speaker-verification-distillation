"""Optuna search of the VBx thresholds on a random ``--fraction`` of a set (10%).

``--target vad``: the Silero probability thresholds of ``extract.py
--vad-silero-prob`` (``--onset``, ``--offset`` <= onset, ``--merge-gap``,
``--min-dur``) against the speech of the reference RTTMs; the objective is the
VAD error MS + FA in % of the reference speech at ``--collar``.  Needs only
the Silero cache, no x-vectors.  (The VAD changes the x-vectors, so tuning it
by DER would need an extraction per trial.)

    python egs/diarization/vbx/tune.py --target vad --silero-dir EXP/VoxConverse/dev/silero \
        --ref-dir /media/raid/voice/VoxConverse/voxconverse/dev

``--target vbx``: the clustering parameters of ``diarize.py`` (``--params``,
default ``threshold,Fa,Fb,loopP,init_smoothing``) on the x-vectors of an
``extract.py`` output; the objective is the DER at ``--collar``.  The AHC
dendrogram of each file is built once, a trial cuts it and runs the VB-HMM.

    python egs/diarization/vbx/tune.py --target vbx --xvec-dir EXP/VoxConverse/dev/xvec-1234abcd \
        --ref-dir /media/raid/voice/VoxConverse/voxconverse/dev --model resnet101-16k

The first trial is the starting point given by the usual options (``--onset
..`` for vad, ``--threshold --Fa ..`` for vbx, otherwise their defaults);
``--range NAME LO HI`` changes a search range (extended to contain the start).
Prints the best parameters as options of ``extract.py`` / ``diarize.py`` /
``run-*.sh`` and writes them with the baseline and the subset to ``--out``
(json).  ``--eval-all`` also scores the baseline and the best on all files.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import typing as tp
from pathlib import Path

import numpy as np
import optuna
from threadpoolctl import threadpool_limits

try:
    import diarize
    import vad
    from vbhmm import AHCTree, VBxParams, cluster, merge_adjacent_labels
except ImportError:
    from . import diarize, vad
    from .vbhmm import AHCTree, VBxParams, cluster, merge_adjacent_labels

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ecapa.rttm import read_rttm_dir  # noqa: E402
from ecapa.score_der import Score, score_file  # noqa: E402
from ecapa.silero_vad import SileroCache  # noqa: E402

VBX_PARAMS = ("threshold", "Fa", "Fb", "loopP", "init_smoothing")
# (low, high, log scale)
RANGES: tp.Dict[str, tp.Tuple[float, float, bool]] = {
    "threshold": (-0.5, 0.5, False),
    "Fa": (0.05, 1.0, True),
    "Fb": (1.0, 100.0, True),
    "loopP": (0.3, 0.999, False),
    "init_smoothing": (1.0, 10.0, False),
    "onset": (0.01, 0.99, False),
    "offset": (0.01, 0.99, False),
    "merge_gap": (0.0, 1.0, False),
    "min_dur": (0.0, 1.0, False),
}


def parse_args() -> tp.Tuple[argparse.Namespace, argparse.Namespace]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    p.add_argument("--target", choices=["vad", "vbx"], required=True)
    p.add_argument("--ref-dir", type=Path, required=True)
    p.add_argument("--silero-dir", type=Path, default=None, help="vad: silero_vad.py output")
    p.add_argument("--xvec-dir", type=Path, default=None, help="vbx: extract.py output")
    p.add_argument("--params", default=",".join(VBX_PARAMS), help="vbx: parameters to tune")
    p.add_argument("--range", nargs=3, action="append", default=[], metavar=("NAME", "LO", "HI"))
    p.add_argument("--fraction", type=float, default=0.1, help="share of the files to tune on")
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--collar", type=float, default=0.25)
    p.add_argument("--skip-overlap", action="store_true")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--sampler-seed", type=int, default=0)
    p.add_argument("--storage", default=None, help="optuna storage to resume, e.g. sqlite:///tune.db")
    p.add_argument("--study-name", default=None)
    p.add_argument("--eval-all", action="store_true")
    p.add_argument("--workers", type=int, default=8, help="vbx: files evaluated in parallel")
    p.add_argument("--out", type=Path, default=None, help="json (default: <silero-dir>/../tune_vad.json, <xvec-dir>/tune_vbx.json)")
    args, rest = p.parse_known_args()

    if not 0 < args.fraction <= 1:
        p.error("--fraction must be in (0, 1]")
    for name, lo, hi in args.range:
        if name not in RANGES:
            p.error(f"--range: unknown parameter {name}")
        RANGES[name] = (float(lo), float(hi), RANGES[name][2])

    if args.target == "vad":
        if args.silero_dir is None:
            p.error("--target vad needs --silero-dir")
        sub = argparse.ArgumentParser(allow_abbrev=False)
        vad.add_arguments(sub)
        opts = sub.parse_args(rest)
        args.params = list(vad.PROB_PARAMS)
    else:
        if args.xvec_dir is None:
            p.error("--target vbx needs --xvec-dir")
        parser = diarize.build_parser()
        opts = diarize.resolve(parser, parser.parse_args(["--xvec-dir", str(args.xvec_dir), "--out-dir", "-", *rest]))
        args.params = [s for s in args.params.split(",") if s]
        if unknown := set(args.params) - set(VBX_PARAMS):
            p.error(f"--params: unknown {sorted(unknown)}")
    return args, opts


# ---------------------------------------------------------------------------
# Files and objectives
# ---------------------------------------------------------------------------
class File(tp.NamedTuple):
    file_id: str
    ref: tp.List[tp.Tuple[float, float, str]]
    data: tp.Any  # vad: (prob, shift); vbx: (npz fields, AHC tree)


def vad_score(f: File, params: tp.Mapping[str, float], collar: float, skip_overlap: bool) -> Score:
    prob, shift = f.data
    segments = vad.prob_segments(prob, shift, *(params[n] for n in vad.PROB_PARAMS))
    return score_file(f.ref, [(s, e, "speech") for s, e in segments], collar, skip_overlap, 0.01)


_FILES: tp.List[File] = []
_BACKEND: diarize.Backend | None = None


def _vbx_init(threads: int) -> None:
    threadpool_limits(threads)


def _tree(path: Path) -> tp.Tuple[dict, AHCTree]:
    data = diarize.load_npz(path)
    x = data["embeddings"]
    if _BACKEND.transform is not None and len(x):
        x = _BACKEND.transform(x)
    return data, AHCTree.build(x.astype(np.float64)) if len(x) else AHCTree(None, 0.0, 0.0)


def _vbx_file(job: tp.Tuple[int, VBxParams, float, bool]) -> Score:
    index, params, collar, skip_overlap = job
    f = _FILES[index]
    data, tree = f.data
    hyp: tp.List[tp.Tuple[float, float, str]] = []
    if len(data["embeddings"]):
        labels = cluster(data["embeddings"], _BACKEND.plda, params, _BACKEND.transform, tree)
        starts, ends, labels = merge_adjacent_labels(data["starts"].copy(), data["ends"].copy(), labels)
        hyp = [(float(s), float(e), str(label)) for s, e, label in zip(starts, ends, labels)]
    return score_file(f.ref, hyp, collar, skip_overlap, 0.01)


class Evaluator:
    """Total score of a parameter set over a list of files."""

    def __init__(self, args: argparse.Namespace, opts: argparse.Namespace, pool: tp.Any) -> None:
        self.args, self.opts, self.pool = args, opts, pool

    def __call__(self, files: tp.Sequence[int], params: tp.Mapping[str, float]) -> Score:
        total = Score()
        if self.args.target == "vad":
            for i in files:
                total += vad_score(_FILES[i], params, self.args.collar, self.args.skip_overlap)
            return total
        vbx = diarize.vbx_params(argparse.Namespace(**{**vars(self.opts), **params}))
        jobs = [(i, vbx, self.args.collar, self.args.skip_overlap) for i in files]
        for score in self.pool.imap_unordered(_vbx_file, jobs) if self.pool else map(_vbx_file, jobs):
            total += score
        return total


def objective_value(score: Score, target: str) -> float:
    return score.der if target == "vbx" else score.vad_rate(score.vad_miss + score.vad_fa)


def describe(score: Score, target: str) -> str:
    vad_err = (
        f"VAD {score.vad_rate(score.vad_miss + score.vad_fa):.2f} "
        f"(MS {score.vad_rate(score.vad_miss):.2f} FA {score.vad_rate(score.vad_fa):.2f})"
    )
    if target == "vad":
        return vad_err
    return (
        f"DER {score.der:.2f} (MS {score.rate(score.miss):.2f} FA {score.rate(score.fa):.2f} "
        f"SC {score.rate(score.conf):.2f}), {vad_err}"
    )


def as_options(params: tp.Mapping[str, float], target: str) -> str:
    opts = " ".join(f"--{name.replace('_', '-')} {params[name]:.4g}" for name in params)
    return f"--vad-silero-prob {opts}" if target == "vad" else opts


# ---------------------------------------------------------------------------
def load(args: argparse.Namespace, refs: dict) -> tp.List[tp.Tuple[str, Path]]:
    if args.target == "vad":
        paths = SileroCache(args.silero_dir).paths
    else:
        paths = {p.stem: p for p in sorted(args.xvec_dir.rglob("*.npz"))}
    ids = sorted(set(paths) & set(refs))
    if not ids:
        raise SystemExit(f"no files with a reference in {args.ref_dir}")
    return [(i, paths[i]) for i in ids]


def add_files(items: tp.List[tp.Tuple[str, Path]], refs: dict, args: argparse.Namespace, pool: tp.Any) -> tp.List[int]:
    """Load ``items`` into ``_FILES`` (vbx: with their AHC trees), return their indices."""
    if args.target == "vad":
        cache = SileroCache(args.silero_dir)
        data = []
        for file_id, _ in items:
            npz = cache._load(file_id)
            data.append((npz["prob"], float(npz["prob_shift"])))
    else:
        paths = [path for _, path in items]
        data = list(pool.imap(_tree, paths)) if pool else [_tree(p) for p in paths]
    start = len(_FILES)
    _FILES.extend(File(file_id, refs[file_id], d) for (file_id, _), d in zip(items, data))
    return list(range(start, len(_FILES)))


def make_pool(args: argparse.Namespace) -> tp.Any:
    if args.target == "vad" or args.workers <= 1:
        threadpool_limits(diarize.blas_threads(1))
        return None
    # forked after _FILES is filled, so the workers see the files
    return mp.get_context("fork").Pool(args.workers, initializer=_vbx_init, initargs=(diarize.blas_threads(args.workers),))


def main() -> None:
    global _BACKEND
    args, opts = parse_args()
    refs = read_rttm_dir(args.ref_dir)
    items = load(args, refs)
    rng = np.random.default_rng(args.subset_seed)
    n = max(1, math.ceil(args.fraction * len(items)))
    subset_items = [items[i] for i in sorted(rng.choice(len(items), n, replace=False))]
    if args.target == "vbx":
        _BACKEND = diarize.Backend.load(opts)

    tree_pool = make_pool(args)
    subset = add_files(subset_items, refs, args, tree_pool)
    if tree_pool is not None:
        tree_pool.close()
    pool = make_pool(args)
    evaluate = Evaluator(args, opts, pool)
    print(f"tuning {', '.join(args.params)} on {n}/{len(items)} files, {args.target} at collar {args.collar:g}")

    baseline = {name: float(getattr(opts, name)) for name in args.params}

    def suggest(trial: optuna.Trial, name: str, high: float | None = None) -> float:
        lo, hi, log = RANGES[name]
        lo, hi = min(lo, baseline[name]), max(hi, baseline[name])
        return trial.suggest_float(name, lo, hi if high is None else min(hi, high), log=log)

    def objective(trial: optuna.Trial) -> float:
        params: tp.Dict[str, float] = {}
        for name in args.params:
            params[name] = suggest(trial, name, params["onset"] if name == "offset" else None)
        score = evaluate(subset, params)
        trial.set_user_attr("score", describe(score, args.target))
        return objective_value(score, args.target)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.study_name or args.target,
        storage=args.storage,
        load_if_exists=args.storage is not None,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )
    if not study.trials:
        study.enqueue_trial(baseline)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best, base_trial = study.best_trial, study.trials[0]
    print(f"\nbaseline {as_options(base_trial.params, args.target)}\n  {base_trial.user_attrs['score']}")
    print(f"best (trial {best.number}) {as_options(best.params, args.target)}\n  {best.user_attrs['score']}")
    result = {
        "target": args.target,
        "collar": args.collar,
        "baseline": {"params": base_trial.params, "value": base_trial.value},
        "best": {"params": best.params, "value": best.value, "trial": best.number},
        "options": as_options(best.params, args.target),
        "subset": [_FILES[i].file_id for i in subset],
        "fixed": {k: str(v) for k, v in vars(opts).items() if k not in args.params},
    }

    if args.eval_all:
        if pool is not None:
            pool.close()
        rest = [item for item in items if item not in subset_items]
        tree_pool = make_pool(args)
        everything = subset + add_files(rest, refs, args, tree_pool)
        if tree_pool is not None:
            tree_pool.close()
        evaluate.pool = pool = make_pool(args)
        for name, params in (("baseline", base_trial.params), ("best", best.params)):
            score = evaluate(everything, params)
            result[name]["all_files"] = objective_value(score, args.target)
            print(f"all {len(everything)} files, {name}: {describe(score, args.target)}")
    if pool is not None:
        pool.close()

    default = args.silero_dir.parent / "tune_vad.json" if args.target == "vad" else args.xvec_dir / "tune_vbx.json"
    out = args.out or default
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n{out}\nuse: {as_options(best.params, args.target)}")


if __name__ == "__main__":
    main()
