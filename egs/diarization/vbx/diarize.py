"""VBx diarization (AHC + VB-HMM with PLDA) of ``extract.py`` outputs.

    python egs/diarization/vbx/diarize.py --xvec-dir XVEC --out-dir RTTM --model resnet101-16k \
        [--threshold -0.015 --Fa 0.3 --Fb 17 --loopP 0.99 --init-smoothing 5 --lda-dim 128]

Writes ``<out-dir>/<rel>.rttm`` per ``<xvec-dir>/<rel>.npz``.  ``--model``
gives the PLDA, the x-vector transform and the default AHC threshold of a
preset (``models.py``); ``--plda`` / ``--xvec-transform`` / ``--threshold``
override them.  The PLDA is a Kaldi file (``--lda-dim`` defaults to 128, as
in VBx) or a directory with ``mean.npy``, ``transform.npy``, ``psi.npy`` (all
dimensions by default).  Other defaults are those of the reference
``run_example.py``; ``tune.py`` searches them.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import typing as tp
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

try:
    from models import PRESETS, preset
    from vbhmm import PLDA, VBxParams, XvecTransform, cluster, load_plda, merge_adjacent_labels
except ImportError:
    from .models import PRESETS, preset
    from .vbhmm import PLDA, VBxParams, XvecTransform, cluster, load_plda, merge_adjacent_labels

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ecapa.rttm import Turn, write_rttm  # noqa: E402

NONE = "none"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    p.add_argument("--xvec-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    add_vbx_arguments(p)
    p.add_argument("--num-workers", type=int, default=4)
    return p


def add_vbx_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", choices=sorted(PRESETS), default=None, help="preset for the defaults below")
    p.add_argument("--plda", type=Path, default=None, help="Kaldi PLDA file or dir (default: the preset's)")
    p.add_argument(
        "--xvec-transform",
        default=None,
        help=f"transform.h5 applied before clustering, '{NONE}' for none (default: the preset's)",
    )
    p.add_argument("--lda-dim", type=int, default=None, help="PLDA dims (default: 128 Kaldi file, all dir)")
    p.add_argument("--threshold", type=float, default=None, help="AHC threshold (default: the preset's or -0.015)")
    p.add_argument("--Fa", type=float, default=0.3)
    p.add_argument("--Fb", type=float, default=17.0)
    p.add_argument("--loopP", type=float, default=0.99)
    p.add_argument("--init-smoothing", type=float, default=5.0)
    p.add_argument("--max-iters", type=int, default=40)


def resolve(p: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """Fill the preset defaults in, as paths (``xvec_transform`` None if none)."""
    cfg = preset(args.model) if args.model is not None else {}
    if args.plda is None:
        if "plda" not in cfg:
            p.error("give --model or --plda")
        args.plda = cfg["plda"]
    if args.xvec_transform is None:
        args.xvec_transform = cfg.get("xvec_transform")
    elif args.xvec_transform.lower() == NONE:
        args.xvec_transform = None
    if args.xvec_transform is not None:
        args.xvec_transform = Path(args.xvec_transform)
    if args.threshold is None:
        args.threshold = cfg.get("threshold", -0.015)
    if not 0 <= args.loopP <= 1:
        p.error("--loopP must be in [0, 1]")
    return args


def parse_args(argv: tp.List[str] | None = None) -> argparse.Namespace:
    p = build_parser()
    return resolve(p, p.parse_args(argv))


def vbx_params(args: argparse.Namespace) -> VBxParams:
    return VBxParams(
        threshold=args.threshold,
        Fa=args.Fa,
        Fb=args.Fb,
        loopP=args.loopP,
        init_smoothing=args.init_smoothing,
        max_iters=args.max_iters,
    )


class Backend(tp.NamedTuple):
    plda: PLDA
    transform: XvecTransform | None

    @classmethod
    def load(cls, args: argparse.Namespace) -> "Backend":
        transform = XvecTransform(args.xvec_transform) if args.xvec_transform is not None else None
        return cls(load_plda(args.plda, args.lda_dim), transform)


def diarize(data: tp.Mapping[str, np.ndarray], backend: Backend, params: VBxParams) -> tp.List[Turn]:
    """``data``: the npz fields of ``extract.py``."""
    emb, starts, ends = data["embeddings"], data["starts"].copy(), data["ends"].copy()
    if len(emb) == 0:
        return []
    labels = cluster(emb, backend.plda, params, backend.transform)
    starts, ends, labels = merge_adjacent_labels(starts, ends, labels)
    return [(float(s), float(e), str(label + 1)) for s, e, label in zip(starts, ends, labels)]


def load_npz(path: Path) -> tp.Dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {k: npz[k] for k in ("embeddings", "starts", "ends", "speech")}


_backend: Backend | None = None
_params: VBxParams | None = None


def _init(backend: Backend, params: VBxParams, threads: int) -> None:
    global _backend, _params
    _backend, _params = backend, params
    # OpenBLAS with all cores is ~50x slower on the (T, S) x (T, D) products here
    threadpool_limits(threads)


def blas_threads(num_workers: int) -> int:
    return max(1, min(4, (os.cpu_count() or 1) // max(num_workers, 1)))


def _job(paths: tp.Tuple[Path, Path]) -> None:
    src, out = paths
    write_rttm(out, src.stem, diarize(load_npz(src), _backend, _params))


def main() -> None:
    args = parse_args()
    backend, params = Backend.load(args), vbx_params(args)
    pairs = [
        (p, (args.out_dir / p.relative_to(args.xvec_dir)).with_suffix(".rttm"))
        for p in sorted(args.xvec_dir.rglob("*.npz"))
    ]
    if not pairs:
        raise SystemExit(f"no .npz in {args.xvec_dir}")
    # longest first, for the load balance
    pairs.sort(key=lambda pair: -pair[0].stat().st_size)
    if args.num_workers > 1:
        init = (backend, params, blas_threads(args.num_workers))
        with mp.get_context("fork").Pool(args.num_workers, initializer=_init, initargs=init) as pool:
            for _ in tqdm(pool.imap_unordered(_job, pairs), total=len(pairs), desc="vbx"):
                pass
    else:
        _init(backend, params, blas_threads(1))
        for pair in tqdm(pairs, desc="vbx"):
            _job(pair)


if __name__ == "__main__":
    main()
