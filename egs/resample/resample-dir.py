#!/usr/bin/env python3
"""
Recursively copy a folder tree to an output directory:
- Resample all .m4a to 16kHz mono WAV (same relative path, .wav extension)
- Copy all other files as-is
- Multiprocessing by default: workers = max(1, ncpu - 1)

Requires:
  pip install torch torchaudio
And torchaudio must be built with sox (or have a working sox backend).

Example:
  python convert_tree_m4a_to_wav16k.py --input-dir /path/in --output-dir /path/out
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torchaudio


TARGET_SR = 16_000


@dataclass(frozen=True)
class Job:
    src: Path
    dst: Path
    is_m4a: bool
    is_wav: bool


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _rel_dst_path(src: Path, input_root: Path, output_root: Path) -> Path:
    rel = src.relative_to(input_root)
    return output_root / rel


def _make_jobs(input_root: Path, output_root: Path) -> list[Job]:
    jobs: list[Job] = []
    for p in input_root.rglob("*"):
        if p.is_dir():
            continue
        dst = _rel_dst_path(p, input_root, output_root)
        is_m4a = p.suffix.lower() == ".m4a"
        is_flac = p.suffix.lower() == ".flac"
        if is_m4a or is_flac:
            dst = dst.with_suffix(".wav")
        is_wav = p.suffix.lower() == ".wav"

        jobs.append(Job(src=p, dst=dst, is_m4a=is_m4a or is_flac, is_wav=is_wav))
    return jobs


def _resample_to_16k_mono(waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int]:
    """
    waveform: [channels, time]
    Output: mono [1, time'], sr=TARGET_SR
    """
    # Convert to mono if needed
    if waveform.dim() != 2:
        raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.size(0) == 1:
        pass
    else:
        raise ValueError("Waveform has zero channels.")

    # Resample if needed
    if sr != TARGET_SR:
        # Use torchaudio resampler (works with sox backend for I/O; resampler itself is torch)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    return waveform, TARGET_SR


def _process_one(job: Job) -> Tuple[str, str]:
    """
    Returns: ("ok"|"skip"|"err", message)
    """
    try:
        _ensure_parent_dir(job.dst)

        # If destination exists and seems up-to-date, skip (basic check)
        if job.dst.exists():
            # If src is newer, redo; else skip
            try:
                if job.dst.stat().st_mtime >= job.src.stat().st_mtime:
                    return "skip", str(job.dst)
            except Exception:
                # If stat fails, just redo
                pass

        if job.is_m4a:
            # Load audio (torchaudio I/O uses backend; user wants sox)
            # Note: some m4a decoding depends on your ffmpeg/gstreamer setup.
            waveform, sr = torchaudio.load(str(job.src))  # [C, T], sr
            waveform, out_sr = _resample_to_16k_mono(waveform, sr)

            # Save as PCM 16-bit WAV
            torchaudio.save(
                str(job.dst),
                waveform,
                out_sr,
                bits_per_sample=16,
            )
            return "ok", f"m4a->wav16k {job.src} -> {job.dst}"
        elif job.is_wav:
            waveform, sr = torchaudio.load(str(job.src))  # [C, T], sr
            waveform, out_sr = _resample_to_16k_mono(waveform, sr)

            # Save as PCM 16-bit WAV
            torchaudio.save(
                str(job.dst),
                waveform,
                out_sr,
                bits_per_sample=16,
            )
            return "ok", f"wav-{sr}->wav-16k {job.src} -> {job.dst}"
        else:
            shutil.copy2(job.src, job.dst)
            return "ok", f"copy {job.src} -> {job.dst}"

    except Exception as e:
        tb = traceback.format_exc()
        return "err", f"{job.src} -> {job.dst}\n{e}\n{tb}"


def _init_torchaudio_backend(preferred: str = "sox_io") -> None:
    """
    Try to force sox backend if available. Newer torchaudio versions may not expose
    set_audio_backend; we handle both cases.
    """
    # Best-effort: only set if supported
    try:
        if hasattr(torchaudio, "set_audio_backend"):
            available = torchaudio.list_audio_backends()
            if preferred in available:
                torchaudio.set_audio_backend(preferred)
            # else: keep default
    except Exception:
        # If backend selection fails, continue; torchaudio may still work.
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--overwrite", action="store_true", help="Always overwrite outputs")
    parser.add_argument("--backend", type=str, default="sox_io", help="Preferred torchaudio backend (default: sox_io)")
    args = parser.parse_args()

    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"ERROR: input-dir does not exist or is not a directory: {input_root}", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)

    _init_torchaudio_backend(args.backend)

    jobs = _make_jobs(input_root, output_root)
    if not jobs:
        print("No files found.", file=sys.stderr)
        return 0

    # Optionally ignore mtime skipping
    if args.overwrite:
        # Wrap jobs so _process_one won't skip based on mtime: easiest is to delete dst beforehand if exists
        # (Still safe in multiprocessing: each dst is unique)
        for j in jobs:
            try:
                if j.dst.exists():
                    j.dst.unlink()
            except Exception:
                pass

    total = len(jobs)
    print(f"Found {total} files. Workers={args.workers}. Target SR={TARGET_SR}. Backend preference={args.backend}")

    ok = skip = err = 0
    # Use chunksize to reduce overhead for large trees
    chunksize = 8 if total < 10_000 else 32

    with Pool(processes=args.workers) as pool:
        for status, msg in pool.imap_unordered(_process_one, jobs, chunksize=chunksize):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print("\n--- ERROR ---")
                print(msg, file=sys.stderr)

    print(f"Done. ok={ok}, skip={skip}, err={err}")
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())

