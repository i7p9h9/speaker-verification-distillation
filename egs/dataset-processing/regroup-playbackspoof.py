#!/usr/bin/env python3
"""
Regroup an extracted PlaybackSpoof dataset (https://huggingface.co/datasets/mariozenmedina/playbackspoof)
into ``<label>[/<attack type>]/<condition>/`` folders, driven by the per-sample JSON metadata.

Input: a tree (any depth) where every sample is a pair ``<sample_id>.json`` + ``<sample_id>.flac``
(the extracted WebDataset shards, e.g. ``audio/0a00...13.flac`` next to ``0a00...13.json``).

Output (files are MOVED, JSON and FLAC together):

    <output>/live/clean/<id>.{flac,json}                        CH: human voice, clean (ASVspoof 5 bonafide)
    <output>/spoof/deep-fake/clean/<id>.{...}                   CS: TTS/VC, no playback
    <output>/spoof/replay/<HH|HL|LH|LL>/<id>.{...}              PH: human voice played back and re-recorded
    <output>/spoof/replay-deep-fake/<HH|HL|LH|LL>/<id>.{...}    PS: TTS/VC played back and re-recorded

Label rules (``content_channel_category`` in the meta):

    CH  human,     clean source reference  -> live
    CS  spoof,     clean source reference  -> spoof / deep-fake
    PH  human,     playback and recapture  -> spoof / replay            (presentation attack)
    PS  spoof,     playback and recapture  -> spoof / replay-deep-fake  (presentation attack)

"Difficulty" = ``playback_condition`` (playback device quality / recorder quality, distance 30/20/15/10 cm).
Clean (non-playback) samples have no condition and go to ``clean``.

Note: CH is the unchanged ASVspoof 5 bonafide reference, not a new live microphone recording.

The script is idempotent: re-running it on an already regrouped tree is a no-op (files are
already at their destination). Use --dry-run first to see the counts.

Example:
  python regroup-playbackspoof.py --input-dir /data/playbackspoof --dry-run
  python regroup-playbackspoof.py --input-dir /data/playbackspoof --output-dir /data/playbackspoof-grouped
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

# content_channel_category -> (folder prefix inside the output dir, is_playback)
CATEGORIES = {
    "CH": ("live", False),
    "CS": ("spoof/deep-fake", False),
    "PH": ("spoof/replay", True),
    "PS": ("spoof/replay-deep-fake", True),
}
CONDITIONS = ("HH", "HL", "LH", "LL")  # playback device quality, recorder quality
CLEAN_DIR = "clean"
AUDIO_SUFFIX = ".flac"


def _category(meta: dict) -> Optional[str]:
    """CH/CS/PH/PS from the meta; falls back to origin + channel labels if the field is absent."""
    cat = meta.get("content_channel_category")
    if cat:
        return str(cat).strip().upper()
    origin = str(meta.get("content_origin_label", "")).strip().lower()
    playback = str(meta.get("channel_label", "")).strip().lower() == "playback"
    if origin in ("human", "spoof"):
        return ("P" if playback else "C") + ("H" if origin == "human" else "S")
    return None


def _target(meta: dict) -> Tuple[str, str]:
    """Returns (folder prefix, condition folder) for a sample; raises ValueError on unknown / inconsistent meta."""
    cat = _category(meta)
    if cat not in CATEGORIES:
        raise ValueError(f"unknown content_channel_category: {cat!r}")
    prefix, is_playback = CATEGORIES[cat]
    if not is_playback:
        return prefix, CLEAN_DIR
    cond = str(meta.get("playback_condition", "")).strip().upper()
    if cond not in CONDITIONS:
        raise ValueError(f"{cat}: unknown playback_condition {cond!r}")
    return prefix, cond


def _process(job: Tuple[Path, Path, bool]) -> Tuple[str, str, str]:
    """Returns (status, group, message); status in ok|skip|notsample|err."""
    json_src, output_root, dry_run = job
    try:
        with open(json_src, encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict) or ("content_channel_category" not in meta and "content_origin_label" not in meta):
            return "notsample", "", str(json_src)

        prefix, sub = _target(meta)
        group = f"{prefix}/{sub}"
        dst_dir = output_root / prefix / sub
        stem = json_src.stem
        audio_src = json_src.with_suffix(AUDIO_SUFFIX)
        json_dst = dst_dir / json_src.name
        audio_dst = dst_dir / audio_src.name

        if not audio_src.is_file():
            return "err", group, f"{json_src}: audio file not found ({audio_src.name})"
        if json_src.resolve() == json_dst.resolve() and audio_src.resolve() == audio_dst.resolve():
            return "skip", group, stem
        for dst in (json_dst, audio_dst):
            if dst.exists():
                return "err", group, f"{json_src}: destination already exists: {dst}"

        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            # audio first: if it fails the JSON stays next to its audio and the sample can be retried
            shutil.move(str(audio_src), str(audio_dst))
            shutil.move(str(json_src), str(json_dst))
        return "ok", group, stem
    except Exception as e:  # keep going, report at the end
        return "err", "", f"{json_src}: {type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="dir with <id>.json + <id>.flac pairs (any depth)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to create live/ and spoof/ (default: --input-dir, i.e. regroup in place)",
    )
    parser.add_argument("--workers", type=int, default=16, help="I/O threads (default: 16)")
    parser.add_argument("--dry-run", action="store_true", help="only count what would be moved")
    args = parser.parse_args()

    input_root = args.input_dir.resolve()
    output_root = (args.output_dir or args.input_dir).resolve()
    if not input_root.is_dir():
        print(f"ERROR: input-dir does not exist or is not a directory: {input_root}", file=sys.stderr)
        return 2

    # Materialise the list first: with in-place regrouping the tree changes while we walk it.
    json_files = sorted(input_root.rglob("*.json"))
    print(f"Found {len(json_files)} json files in {input_root}. Output: {output_root}. "
          f"{'DRY RUN' if args.dry_run else 'MOVING files'}")

    status = Counter()
    groups: Counter = Counter()
    errors = []
    jobs = ((p, output_root, args.dry_run) for p in json_files)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for st, group, msg in pool.map(_process, jobs, chunksize=64):
            status[st] += 1
            if st in ("ok", "skip"):
                groups[group] += 1
            elif st == "err":
                errors.append(msg)

    print("\nPer folder (moved + already in place):")
    for g in sorted(groups):
        print(f"  {g:<32} {groups[g]:>9}")
    print(f"\nDone. moved={status['ok']}, already_in_place={status['skip']}, "
          f"not_a_sample_json={status['notsample']}, errors={status['err']}"
          + (" (dry run: nothing was moved)" if args.dry_run else ""))
    for msg in errors[:20]:
        print(f"ERROR: {msg}", file=sys.stderr)
    if len(errors) > 20:
        print(f"... and {len(errors) - 20} more errors", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
