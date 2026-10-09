"""RTTM reading/writing shared by the diarization scripts."""

from __future__ import annotations

import typing as tp
from collections import defaultdict
from pathlib import Path

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
