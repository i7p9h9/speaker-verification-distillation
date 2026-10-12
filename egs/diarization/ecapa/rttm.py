"""RTTM reading/writing shared by the diarization scripts."""

from __future__ import annotations

import typing as tp
from collections import defaultdict
from pathlib import Path

import numpy as np

# (start_sec, end_sec, speaker)
Turn = tp.Tuple[float, float, str]


def read_rttm(path: Path) -> tp.Dict[str, tp.List[Turn]]:
    """Return ``{file_id: [(start, end, speaker), ...]}`` for SPEAKER lines."""
    turns: tp.Dict[str, tp.List[Turn]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "SPEAKER":
                continue
            file_id, start, dur, spk = parts[1], float(parts[3]), float(parts[4]), parts[7]
            turns[file_id].append((start, start + dur, spk))
    return dict(turns)


def read_rttm_dir(directory: Path) -> tp.Dict[str, tp.List[Turn]]:
    """Merge all ``*.rttm`` files of a directory (recursively) by file id."""
    turns: tp.Dict[str, tp.List[Turn]] = {}
    for path in sorted(Path(directory).rglob("*.rttm")):
        for file_id, file_turns in read_rttm(path).items():
            turns.setdefault(file_id, []).extend(file_turns)
    return turns


def write_rttm(path: Path, file_id: str, turns: tp.Iterable[Turn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for start, end, spk in sorted(turns):
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} <NA> <NA> {spk} <NA> <NA>\n")


def rttm_to_vad(
    turns: tp.Iterable[Turn],
    sample_rate: float,
    num_samples: int | None = None,
    is_multispeaker: bool = False,
) -> np.ndarray:
    """Rasterize turns into a boolean activity mask.

    ``sample_rate`` is the rate of the mask: the audio sample rate for a
    sample-level mask, or e.g. 100 for 10 ms frames.  A turn covers samples
    ``round(start * sample_rate) .. round(end * sample_rate) - 1``.

    ``num_samples`` fixes the mask length: the mask is padded with zeros up to
    it, and turns past it are cut.  By default the mask ends with the last turn.

    Returns ``(num_samples,)`` speech activity of any speaker, or with
    ``is_multispeaker`` ``(num_speakers, num_samples)`` with one row per speaker,
    rows in ``sorted()`` order of the speaker labels.
    """
    turns = list(turns)
    speakers = sorted({spk for _, _, spk in turns})
    bounds = [(round(start * sample_rate), round(end * sample_rate), spk) for start, end, spk in turns]
    if num_samples is None:
        num_samples = max((end for _, end, _ in bounds), default=0)

    row = {spk: i for i, spk in enumerate(speakers)}
    vad = np.zeros((len(speakers) if is_multispeaker else 1, num_samples), dtype=bool)
    for start, end, spk in bounds:
        vad[row[spk] if is_multispeaker else 0, max(start, 0) : max(min(end, num_samples), 0)] = True
    return vad if is_multispeaker else vad[0]


def read_rttm_as_vad(
    path: Path,
    sample_rate: float,
    num_samples: int | tp.Mapping[str, int] | None = None,
    is_multispeaker: bool = False,
) -> tp.Dict[str, np.ndarray]:
    """``{file_id: mask}`` for an RTTM file, see :func:`rttm_to_vad`.

    ``num_samples`` is either one length for all files or a ``{file_id: length}``
    mapping; files missing from the mapping end with their last turn.
    """
    masks = {}
    for file_id, turns in read_rttm(path).items():
        length = num_samples.get(file_id) if isinstance(num_samples, tp.Mapping) else num_samples
        masks[file_id] = rttm_to_vad(turns, sample_rate, length, is_multispeaker)
    return masks
