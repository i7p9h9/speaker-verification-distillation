"""Convert Nemotron-3-Diarization outputs of ``infer.py`` to RTTM.

By default the segments are rebuilt from the saved frame probabilities: a frame
is speech of a speaker when its probability exceeds ``--threshold``.  With the
default threshold of 0.5 this matches ``processor.extract_speaker_dict``.
Optionally, segments shorter than ``--min-dur`` are dropped after gaps shorter
than ``--merge-gap`` within a speaker are closed.  ``--from-json`` uses the
saved ``extract_speaker_dict`` segments instead.

The RTTM files can be scored together with other systems by
``egs/diarization/ecapa/score_der.py``.  The RTTM file id is the audio file
stem, as in ``egs/diarization/ecapa``.

Usage:
    python egs/diarization/nemotron/to_rttm.py --in-dir data/nemotron --out-dir data/rttm-nemotron
"""

from __future__ import annotations

import argparse
import json
import typing as tp
from pathlib import Path

import numpy as np

Turn = tp.Tuple[float, float, str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, required=True, help="output dir of infer.py")
    p.add_argument("--out-dir", type=Path, required=True, help="one <rel>.rttm per input file")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--merge-gap", type=float, default=0.0, help="close same-speaker gaps shorter than this, s")
    p.add_argument("--min-dur", type=float, default=0.0, help="drop segments shorter than this, s")
    p.add_argument("--from-json", action="store_true", help="use the saved extract_speaker_dict segments")
    return p.parse_args()


def turns_from_probs(probs: np.ndarray, frame_shift: float, args: argparse.Namespace) -> tp.List[Turn]:
    turns: tp.List[Turn] = []
    for speaker in range(probs.shape[1]):
        active = probs[:, speaker] > args.threshold
        edges = np.diff(np.concatenate(([0], active.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        segments: tp.List[tp.List[float]] = []
        for start, end in zip(starts * frame_shift, ends * frame_shift):
            if segments and start - segments[-1][1] < args.merge_gap:
                segments[-1][1] = end
            else:
                segments.append([start, end])
        turns.extend((s, e, f"spk{speaker}") for s, e in segments if e - s >= args.min_dur)
    return turns


def write_rttm(path: Path, file_id: str, turns: tp.Iterable[Turn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for start, end, spk in sorted(turns):
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} <NA> <NA> {spk} <NA> <NA>\n")


def main() -> None:
    args = parse_args()
    pattern = "*.json" if args.from_json else "*.npz"
    files = sorted(args.in_dir.rglob(pattern))
    if not files:
        raise SystemExit(f"no {pattern} files in {args.in_dir}")

    for path in files:
        if args.from_json:
            with open(path) as f:
                segments = json.load(f)["segments"]
            turns = [(s["Start"], s["End"], f"spk{s['Speaker']}") for s in segments]
        else:
            with np.load(path) as data:
                turns = turns_from_probs(data["probs"], float(data["frame_shift"]), args)
        out_path = (args.out_dir / path.relative_to(args.in_dir)).with_suffix(".rttm")
        write_rttm(out_path, path.stem, turns)


if __name__ == "__main__":
    main()
