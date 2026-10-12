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

VAD metrics compare the speech regions of any speaker: VAD_MS and VAD_FA are
missed and falsely detected speech in % of reference speech (same collars),
VAD is their sum.

Files are matched by the RTTM file id.  Totals are computed over the summed
durations of all files, not by averaging per-file DER.

Usage:
    python egs/diarization/ecapa/score_der.py --ref-dir data/ref --hyp-dir data/rttm \
        --collars 0 0.25 [--skip-overlap] [--ignore-missing] [--csv der.csv] [--detailed]

Prints the totals per collar; ``--detailed`` also the per-file table.
"""

from __future__ import annotations

import argparse
import csv
import sys
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from rttm import Turn, read_rttm_dir
except ImportError:
    from .rttm import Turn, read_rttm_dir
from scipy.optimize import linear_sum_assignment


@dataclass
class Score:
    total: float = 0.0  # reference speaker time (overlap counted per speaker)
    miss: float = 0.0
    fa: float = 0.0
    conf: float = 0.0
    speech: float = 0.0  # reference speech time (any speaker)
    vad_miss: float = 0.0
    vad_fa: float = 0.0

    def __iadd__(self, other: "Score") -> "Score":
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    def rate(self, value: float) -> float:
        return 100.0 * value / self.total if self.total > 0 else float("nan")

    def vad_rate(self, value: float) -> float:
        return 100.0 * value / self.speech if self.speech > 0 else float("nan")

    @property
    def der(self) -> float:
        return self.rate(self.miss + self.fa + self.conf)

    def row(self) -> tp.List[str]:
        values = [self.der, self.rate(self.miss), self.rate(self.fa), self.rate(self.conf)]
        values += [self.vad_rate(self.vad_miss + self.vad_fa), self.vad_rate(self.vad_miss), self.vad_rate(self.vad_fa)]
        return [f"{v:.2f}" for v in values]


COLUMNS = ["DER", "MS", "FA", "SC", "VAD", "VAD_MS", "VAD_FA"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-dir", type=Path, required=True)
    p.add_argument("--hyp-dir", type=Path, required=True)
    p.add_argument("--collars", type=float, nargs="+", default=[0.0, 0.25], help="seconds, +- around boundaries")
    p.add_argument("--skip-overlap", action="store_true", help="exclude reference overlap regions from scoring")
    p.add_argument("--resolution", type=float, default=0.01, help="frame step, seconds")
    p.add_argument(
        "--ignore-missing",
        action="store_true",
        help="score only reference files that have a hypothesis RTTM (file id or file name)",
    )
    p.add_argument("--csv", type=Path, default=None, help="also write the table as csv")
    p.add_argument("--detailed", action="store_true", help="print the per-file table, not only the totals")
    return p


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

    ref_speech, hyp_speech = n_ref > 0, n_hyp > 0
    return Score(
        total=float(n_ref.sum()) * step,
        miss=float(np.maximum(n_ref - n_hyp, 0).sum()) * step,
        fa=float(np.maximum(n_hyp - n_ref, 0).sum()) * step,
        conf=float(np.minimum(n_ref, n_hyp).sum() - correct) * step,
        speech=float(ref_speech.sum()) * step,
        vad_miss=float((ref_speech & ~hyp_speech).sum()) * step,
        vad_fa=float((hyp_speech & ~ref_speech).sum()) * step,
    )


def score_dirs(
    ref_dir: Path,
    hyp_dir: Path,
    collars: tp.Sequence[float] = (0.0, 0.25),
    skip_overlap: bool = False,
    resolution: float = 0.01,
    ignore_missing: bool = False,
) -> tp.Tuple[tp.List[str], tp.List[tp.List[str]]]:
    """``(header, rows)`` with one row per reference file and a final ``*TOTAL*`` row.

    Per collar: DER and its parts (in % of reference speaker time) and the
    speech / non-speech error VAD = VAD_MS + VAD_FA (in % of reference speech).
    """
    refs = read_rttm_dir(ref_dir)
    hyps = read_rttm_dir(hyp_dir)
    if not refs:
        raise SystemExit(f"no reference RTTMs in {ref_dir}")
    if ignore_missing:
        # an RTTM of a file without detected speech has no lines, so match by file name too
        present = set(hyps) | {path.stem for path in Path(hyp_dir).rglob("*.rttm")}
        refs = {file_id: turns for file_id, turns in refs.items() if file_id in present}
        if not refs:
            raise SystemExit(f"no reference file of {ref_dir} has a hypothesis in {hyp_dir}")

    missing = sorted(set(refs) - set(hyps))
    extra = sorted(set(hyps) - set(refs))
    if missing and not ignore_missing:
        print(f"warning: {len(missing)} reference files without hypothesis (scored as missed)", file=sys.stderr)
    if extra:
        print(f"warning: {len(extra)} hypothesis files without reference (ignored)", file=sys.stderr)

    header = ["file"] + [f"{name}[collar={c:g}]" for c in collars for name in COLUMNS]
    rows: tp.List[tp.List[str]] = []
    totals = {c: Score() for c in collars}
    for file_id in sorted(refs):
        row = [file_id]
        for c in collars:
            score = score_file(refs[file_id], hyps.get(file_id, []), c, skip_overlap, resolution)
            totals[c] += score
            row += score.row()
        rows.append(row)
    rows.append(["*TOTAL*"] + [v for c in collars for v in totals[c].row()])
    return header, rows


def format_table(header: tp.List[str], rows: tp.List[tp.List[str]]) -> str:
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    return "\n".join(
        "  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(r, widths)))
        for r in [header] + rows
    )


def summary(header: tp.List[str], rows: tp.List[tp.List[str]], collars: tp.Sequence[float]) -> str:
    """The ``*TOTAL*`` row of :func:`score_dirs` as one line per collar."""
    total = dict(zip(header, rows[-1]))
    files = str(len(rows) - 1)
    table = [[f"{c:g}", files, *[total[f"{n}[collar={c:g}]"] for n in COLUMNS]] for c in collars]
    return format_table(["collar", "files", *COLUMNS], table)


def main() -> None:
    args = build_parser().parse_args()
    header, rows = score_dirs(
        args.ref_dir, args.hyp_dir, args.collars, args.skip_overlap, args.resolution, args.ignore_missing
    )
    if args.detailed:
        print(format_table(header, rows))
        print()
    print(summary(header, rows, args.collars) + ("  (overlap skipped)" if args.skip_overlap else ""))
    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            csv.writer(f).writerows([header] + rows)


if __name__ == "__main__":
    main()
