"""Speech regions for the x-vector extraction: Silero VAD (segments, or
thresholded probabilities), reference RTTMs or VBx ``.lab`` files."""

from __future__ import annotations

import argparse
import sys
import typing as tp
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecapa.diarize import hysteresis, mask_to_segments, merge_and_filter  # noqa: E402
from ecapa.rttm import read_rttm_dir  # noqa: E402
from ecapa.silero_vad import SileroCache  # noqa: E402

Segment = tp.Tuple[float, float]
PROB_PARAMS = ("onset", "offset", "merge_gap", "min_dur")


def add_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("VAD (one source)")
    g.add_argument("--vad-silero", type=Path, default=None, help="silero_vad.py output dir: its speech segments")
    g.add_argument(
        "--vad-silero-prob",
        action="store_true",
        help="with --vad-silero: threshold its probabilities with --onset/--offset/--merge-gap/--min-dur",
    )
    g.add_argument("--onset", type=float, default=0.5, help="hysteresis start threshold")
    g.add_argument("--offset", type=float, default=0.35, help="hysteresis stop threshold")
    g.add_argument("--merge-gap", type=float, default=0.1, help="merge speech segments closer than this, s")
    g.add_argument("--min-dur", type=float, default=0.25, help="drop speech segments shorter than this, s")
    g.add_argument("--vad-rttm-dir", type=Path, default=None, help="reference RTTM dir: oracle speech")
    g.add_argument("--vad-lab-dir", type=Path, default=None, help="dir of VBx <file>.lab (start end, s)")


def check_arguments(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    sources = [args.vad_silero, args.vad_rttm_dir, args.vad_lab_dir]
    if sum(s is not None for s in sources) != 1:
        p.error("give one of --vad-silero, --vad-rttm-dir, --vad-lab-dir")
    if args.vad_silero_prob and args.vad_silero is None:
        p.error("--vad-silero-prob needs --vad-silero")


def merge(segments: tp.Iterable[Segment]) -> tp.List[Segment]:
    return merge_and_filter(sorted(segments), 0.0, 0.0)


def prob_segments(prob: np.ndarray, shift: float, onset: float, offset: float, merge_gap: float, min_dur: float):
    segments = mask_to_segments(hysteresis(prob, onset, offset), shift)
    return merge_and_filter(segments, merge_gap, min_dur)


class Speech:
    """``speech(file_id) -> [(start, end)]`` in seconds, sorted and disjoint."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.silero = SileroCache(args.vad_silero) if args.vad_silero is not None else None
        self.rttm = read_rttm_dir(args.vad_rttm_dir) if args.vad_rttm_dir is not None else None

    def has(self, file_id: str) -> bool:
        if self.silero is not None:
            return file_id in self.silero.paths
        if self.rttm is not None:
            return file_id in self.rttm
        return (self.args.vad_lab_dir / f"{file_id}.lab").exists()

    def __call__(self, file_id: str) -> tp.List[Segment]:
        args = self.args
        if self.silero is not None and args.vad_silero_prob:
            data = self.silero._load(file_id)
            params = [getattr(args, name) for name in PROB_PARAMS]
            return prob_segments(data["prob"], float(data["prob_shift"]), *params)
        if self.silero is not None:
            return merge(self.silero.segments(file_id))
        if self.rttm is not None:
            return merge((s, e) for s, e, _ in self.rttm.get(file_id, []))
        path = args.vad_lab_dir / f"{file_id}.lab"
        if not path.exists():
            raise SystemExit(f"no {path}")
        labs = np.atleast_2d(np.loadtxt(path, usecols=(0, 1)))
        return [(float(s), float(e)) for s, e in labs.reshape(-1, 2)]
