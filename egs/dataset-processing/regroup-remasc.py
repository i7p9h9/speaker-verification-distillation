#!/usr/bin/env python3
"""
Regroup the ReMASC replay-attack corpus into ``<label>/<environment>/`` folders and write a readable
``meta.csv`` that ``validation/utils/get_spoof_datasets.get_remasc`` understands.

Input: a tree (any depth) with ``<file_id>.wav`` files and the original headerless ``meta.csv``
(9 columns, see below), e.g. ``1010101,1,1,1,-1,1,-1,-1,3.385300e+00``.

Original ``meta.csv`` columns:

    1  file id                      (= audio file name, ``<id>.wav``)
    2  speech type                  1 = source, 2 = genuine, 3 = replayed
    3  speaker id                   0-50, 0 = TTS speaker
    4  environment                  1 = outdoor, 2 = indoor-1, 3 = indoor-2, 4 = vehicle
    5  position id                  meaning depends on the environment (kept as is)
    6  source recorder              -1 = N/A, 1 = DR_05, 2 = iPod, 3 = TTS
    7  playback device              -1 = N/A, 1 = ATH-AD700X headphone, 2 = iPod Touch, 3 = Sony SRS-X11,
                                    4 = Sony SRS-X5, 5 = vehicle built-in speaker
    8  recording device             -1 = N/A, 1 = AIY, 2 = Respeaker 4-Linear, 3 = Respeaker Core V2,
                                    4 = Amlogic A113X1
    9  recording length, seconds

Output (audio files are MOVED):

    <output>/live/<env>/<id>.wav            genuine: a real speaker recorded directly      (bonafide)
    <output>/spoof/replay/<env>/<id>.wav    replayed: source played back and re-recorded   (presentation attack)
    <output>/source/<env>/<id>.wav          source: the originals that are played back; neither an access
                                            nor an attack, NOT used for validation
    <output>/meta.csv                       path,type,spk,env,position,source_recording_device,
                                            playback_device,recording_device,file_length
                                            (type = live | replay | source; path relative to <output>)

--exclude-meta takes another original meta (e.g. the validation part of ReMASC): the files listed there are
left where they are, not moved and not written to the new ``meta.csv``. Use it to keep a training copy
disjoint from a validation set.

If the original ``meta.csv`` sits in the output dir it is first renamed to ``original_meta.csv``;
a re-run reads ``original_meta.csv`` (the script is idempotent: files already in place are skipped).
Use --dry-run first to see the counts.

Example:
  python regroup-remasc.py --input-dir /data/ReMASC --dry-run
  python regroup-remasc.py --input-dir /data/ReMASC --output-dir /data/ReMASC-grouped
  python regroup-remasc.py --input-dir /data/ReMASC/data --meta /data/ReMASC/meta.csv \\
      --output-dir /data/ReMASC-train --exclude-meta /data/ReMASC-val/original_meta.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AUDIO_SUFFIX = ".wav"
ORIGINAL_META = "original_meta.csv"
OUT_META = "meta.csv"

# speech type id -> (meta type, folder prefix in the output dir)
TYPES = {
    1: ("source", "source"),
    2: ("live", "live"),
    3: ("replay", "spoof/replay"),
}
ENVIRONMENTS = {1: "outdoor", 2: "indoor-1", 3: "indoor-2", 4: "vehicle"}
SOURCE_RECORDERS = {1: "DR_05", 2: "iPod", 3: "TTS"}
PLAYBACK_DEVICES = {1: "ATH-AD700X", 2: "iPod_Touch", 3: "Sony_SRS_X11", 4: "Sony_SRS_X5", 5: "vehicle_speaker"}
RECORDING_DEVICES = {1: "AIY", 2: "Respeaker_4_Linear", 3: "Respeaker_Core_V2", 4: "Amlogic_A113X1"}
NA = "N/A"
OUT_COLUMNS = [
    "path",
    "type",
    "spk",
    "env",
    "position",
    "source_recording_device",
    "playback_device",
    "recording_device",
    "file_length",
]


def _name(table: Dict[int, str], raw: str, what: str) -> str:
    """-1 -> N/A, a known id -> its name; raises ValueError on an unknown id."""
    value = int(raw)
    if value == -1:
        return NA
    if value not in table:
        raise ValueError(f"unknown {what} id: {value}")
    return table[value]


def parse_row(fields: List[str]) -> Dict[str, str]:
    """One original meta row -> decoded fields (``file_id``, ``type``, ``prefix``, ``env`` + the output columns)."""
    if len(fields) < 9:
        raise ValueError(f"expected 9 columns, got {len(fields)}: {fields}")
    f = [x.strip() for x in fields[:9]]
    file_id = f[0]
    type_id, env_id = int(f[1]), int(f[3])
    if type_id not in TYPES:
        raise ValueError(f"unknown speech type id: {type_id}")
    if env_id not in ENVIRONMENTS:
        raise ValueError(f"unknown environment id: {env_id}")
    kind, prefix = TYPES[type_id]
    return {
        "file_id": file_id,
        "type": kind,
        "prefix": prefix,
        "env": ENVIRONMENTS[env_id],
        "spk": str(int(f[2])),
        "position": f[4],
        "source_recording_device": _name(SOURCE_RECORDERS, f[5], "source recorder"),
        "playback_device": _name(PLAYBACK_DEVICES, f[6], "playback device"),
        "recording_device": _name(RECORDING_DEVICES, f[7], "recording device"),
        "file_length": f"{float(f[8]):.4f}",
    }


def read_meta(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Returns (parsed rows, error messages). The original meta is headerless; blank lines are skipped."""
    rows, errors = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for line_no, fields in enumerate(csv.reader(f), start=1):
            if not fields or not any(x.strip() for x in fields):
                continue
            try:
                rows.append(parse_row(fields))
            except Exception as e:
                errors.append(f"{path}:{line_no}: {type(e).__name__}: {e}")
    return rows, errors


def _process(job: Tuple[Dict[str, str], Optional[Path], Path, bool]) -> Tuple[str, str]:
    """Move one audio file. Returns (status, message); status in ok|skip|err."""
    row, src, output_root, dry_run = job
    dst = output_root / row["prefix"] / row["env"] / f"{row['file_id']}{AUDIO_SUFFIX}"
    try:
        if src is None:
            return "err", f"audio file not found for id {row['file_id']}"
        if src.resolve() == dst.resolve():
            return "skip", row["file_id"]
        if dst.exists():
            return "err", f"{src}: destination already exists: {dst}"
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        return "ok", row["file_id"]
    except Exception as e:  # keep going, report at the end
        return "err", f"{src}: {type(e).__name__}: {e}"


def _write_meta(rows: List[Dict[str, str]], output_root: Path) -> Path:
    out = output_root / OUT_META
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {"path": f"{r['prefix']}/{r['env']}/{r['file_id']}{AUDIO_SUFFIX}", **{k: r[k] for k in OUT_COLUMNS[1:]}}
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="dir with <id>.wav files (any depth)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to create live/, spoof/, source/ and meta.csv (default: --input-dir, i.e. regroup in place)",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help=f"original headerless meta (default: <input-dir>/{ORIGINAL_META} if it exists, else <input-dir>/meta.csv)",
    )
    parser.add_argument(
        "--exclude-meta",
        type=Path,
        action="append",
        default=[],
        help="original meta whose file ids are NOT regrouped (repeatable), e.g. the validation part",
    )
    parser.add_argument("--workers", type=int, default=16, help="I/O threads (default: 16)")
    parser.add_argument("--dry-run", action="store_true", help="only count what would be moved, write nothing")
    args = parser.parse_args()

    input_root = args.input_dir.resolve()
    output_root = (args.output_dir or args.input_dir).resolve()
    if not input_root.is_dir():
        print(f"ERROR: input-dir does not exist or is not a directory: {input_root}", file=sys.stderr)
        return 2

    meta_path = args.meta or (input_root / ORIGINAL_META if (input_root / ORIGINAL_META).is_file() else input_root / OUT_META)
    if not meta_path.is_file():
        print(f"ERROR: meta file not found: {meta_path}", file=sys.stderr)
        return 2

    rows, errors = read_meta(meta_path)
    if not rows:
        print(f"ERROR: no valid rows in {meta_path}", file=sys.stderr)
        for msg in errors[:20]:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 2

    if args.exclude_meta:
        excluded = set()
        for ex_path in args.exclude_meta:
            if not ex_path.is_file():
                print(f"ERROR: exclude-meta not found: {ex_path}", file=sys.stderr)
                return 2
            ex_rows, _ = read_meta(ex_path)
            excluded.update(r["file_id"] for r in ex_rows)
        n_before = len(rows)
        rows = [r for r in rows if r["file_id"] not in excluded]
        print(f"Excluded {n_before - len(rows)} of {n_before} rows listed in {len(args.exclude_meta)} exclude-meta file(s)")
        if not rows:
            print("ERROR: nothing left after --exclude-meta", file=sys.stderr)
            return 2

    # Materialise the audio index first: with in-place regrouping the tree changes while we walk it.
    audio: Dict[str, Path] = {}
    for p in sorted(input_root.rglob(f"*{AUDIO_SUFFIX}")):
        audio.setdefault(p.stem, p)
    print(
        f"Meta: {meta_path} ({len(rows)} rows). Found {len(audio)} {AUDIO_SUFFIX} files in {input_root}. "
        f"Output: {output_root}. {'DRY RUN' if args.dry_run else 'MOVING files'}"
    )

    jobs = [(r, audio.get(r["file_id"]), output_root, args.dry_run) for r in rows]
    status: Counter = Counter()
    groups: Counter = Counter()
    ok_rows: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for row, (st, msg) in zip(rows, pool.map(_process, jobs, chunksize=64)):
            status[st] += 1
            if st in ("ok", "skip"):
                groups[f"{row['prefix']}/{row['env']}"] += 1
                ok_rows.append(row)
            else:
                errors.append(msg)

    print("\nPer folder (moved + already in place):")
    for g in sorted(groups):
        print(f"  {g:<28} {groups[g]:>9}")

    if not args.dry_run and ok_rows:
        original = output_root / OUT_META
        if meta_path.resolve() == original.resolve():  # do not overwrite the raw meta
            backup = output_root / ORIGINAL_META
            if backup.exists():
                errors.append(f"cannot back up {original}: {backup} already exists; meta.csv not written")
                ok_rows = []
            else:
                original.rename(backup)
        if ok_rows:
            print(f"\nWrote {_write_meta(ok_rows, output_root)} ({len(ok_rows)} rows)")

    print(
        f"\nDone. moved={status['ok']}, already_in_place={status['skip']}, errors={len(errors)}"
        + (" (dry run: nothing was moved or written)" if args.dry_run else "")
    )
    for msg in errors[:20]:
        print(f"ERROR: {msg}", file=sys.stderr)
    if len(errors) > 20:
        print(f"... and {len(errors) - 20} more errors", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
