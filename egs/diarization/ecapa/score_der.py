"""Diarization error rate for directories of reference and hypothesis RTTMs.

DER = (missed speech + false alarm + speaker confusion) / reference speech.
Reference and hypothesis speakers are mapped one-to-one by the Hungarian
algorithm on overlap duration, as in NIST md-eval.  Overlapping speech is
scored: a frame with N_ref reference and N_sys hypothesis speakers contributes
``max(N_ref - N_sys, 0)`` miss, ``max(N_sys - N_ref, 0)`` false alarm and
``min(N_ref, N_sys) - N_correct`` confusion.

``--collars`` follows the md-eval / dscore convention: ``collar=c`` excludes
``+-c`` seconds around every reference segment boundary (pyannote's ``collar``
is the total width, i.e. ``2c``).  Without UEM the scored region is
``[0, max(end of reference, end of hypothesis)]``.

Files are matched by the RTTM file id.  Totals are computed over the summed
durations of all files, not by averaging per-file DER.

Usage:
    python egs/diarization/ecapa/score_der.py --ref-dir data/ref --hyp-dir data/rttm \
        --collars 0 0.25 [--skip-overlap] [--csv der.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rttm import Turn, read_rttm_dir
from scipy.optimize import linear_sum_assignment


@dataclass
class Score:
    total: float = 0.0
    miss: float = 0.0
    fa: float = 0.0
    conf: float = 0.0

    def __iadd__(self, other: "Score") -> "Score":
        self.total += other.total
        self.miss += other.miss
        self.fa += other.fa
        self.conf += other.conf
        return self

    def rate(self, value: float) -> float:
        return 100.0 * value / self.total if self.total > 0 else float("nan")

    @property
    def der(self) -> float:
        return self.rate(self.miss + self.fa + self.conf)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-dir", type=Path, required=True)
    p.add_argument("--hyp-dir", type=Path, required=True)
    p.add_argument("--collars", type=float, nargs="+", default=[0.0, 0.25], help="seconds, +- around boundaries")
    p.add_argument("--skip-overlap", action="store_true", help="exclude reference overlap regions from scoring")
    p.add_argument("--resolution", type=float, default=0.01, help="frame step, seconds")
    p.add_argument("--csv", type=Path, default=None, help="also write the table as csv")
    return p.parse_args()


def to_activity(turns: tp.List[Turn], num_frames: int, step: float) -> np.ndarray:
    """``(num_frames, num_speakers)`` boolean speaker activity."""
    speakers = sorted({spk for _, _, spk in turns})
    index = {spk: i for i, spk in enumerate(speakers)}
    activity = np.zeros((num_frames, len(speakers)), dtype=bool)
    for start, end, spk in turns:
        activity[int(round(start / step)) : int(round(end / step)), index[spk]] = True
    return activity


def score_file(
    ref: tp.List[Turn],
    hyp: tp.List[Turn],
    collar: float,
    skip_overlap: bool,
    step: float,
) -> Score:
    end = max([e for _, e, _ in ref + hyp], default=0.0)
    num_frames = int(np.ceil(end / step)) + 1
    ref_act = to_activity(ref, num_frames, step)
    hyp_act = to_activity(hyp, num_frames, step)

    scored = np.ones(num_frames, dtype=bool)
    if collar > 0:
        for start, stop, _ in ref:
            for boundary in (start, stop):
                lo = max(0, int(round((boundary - collar) / step)))
                hi = int(round((boundary + collar) / step))
                scored[lo:hi] = False
    n_ref = ref_act.sum(axis=1)
    if skip_overlap:
        scored &= n_ref <= 1

    ref_act, hyp_act, n_ref = ref_act[scored], hyp_act[scored], n_ref[scored]
    n_hyp = hyp_act.sum(axis=1)

    correct = 0
    if ref_act.shape[1] and hyp_act.shape[1]:
        overlap = ref_act.T.astype(np.int64) @ hyp_act.astype(np.int64)
        rows, cols = linear_sum_assignment(-overlap)
        correct = int(overlap[rows, cols].sum())

    return Score(
        total=float(n_ref.sum()) * step,
        miss=float(np.maximum(n_ref - n_hyp, 0).sum()) * step,
        fa=float(np.maximum(n_hyp - n_ref, 0).sum()) * step,
        conf=float(np.minimum(n_ref, n_hyp).sum() - correct) * step,
    )


def main() -> None:
    args = parse_args()
    refs = read_rttm_dir(args.ref_dir)
    hyps = read_rttm_dir(args.hyp_dir)
    if not refs:
        raise SystemExit(f"no reference RTTMs in {args.ref_dir}")

    missing = sorted(set(refs) - set(hyps))
    extra = sorted(set(hyps) - set(refs))
    if missing:
        print(f"warning: {len(missing)} reference files without hypothesis (scored as missed)", file=sys.stderr)
    if extra:
        print(f"warning: {len(extra)} hypothesis files without reference (ignored)", file=sys.stderr)

    header = ["file"]
    for c in args.collars:
        header += [f"DER[collar={c:g}]", f"MS[collar={c:g}]", f"FA[collar={c:g}]", f"SC[collar={c:g}]"]
    rows: tp.List[tp.List[str]] = []
    totals = {c: Score() for c in args.collars}
    for file_id in sorted(refs):
        row = [file_id]
        for c in args.collars:
            score = score_file(refs[file_id], hyps.get(file_id, []), c, args.skip_overlap, args.resolution)
            totals[c] += score
            row += [f"{score.der:.2f}", f"{score.rate(score.miss):.2f}", f"{score.rate(score.fa):.2f}"]
            row += [f"{score.rate(score.conf):.2f}"]
        rows.append(row)
    total_row = ["*TOTAL*"]
    for c in args.collars:
        s = totals[c]
        total_row += [f"{s.der:.2f}", f"{s.rate(s.miss):.2f}", f"{s.rate(s.fa):.2f}", f"{s.rate(s.conf):.2f}"]
    rows.append(total_row)

    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(r, widths))))

    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            csv.writer(f).writerows([header] + rows)


if __name__ == "__main__":
    main()
