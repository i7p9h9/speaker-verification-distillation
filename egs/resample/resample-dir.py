#!/usr/bin/env python3
"""
Recursively copy a folder tree to an output directory:
- Resample all .m4a/.flac/.wav to 16kHz mono (same relative path); output container is
  chosen with --format: wav (PCM 16-bit, default) or ogg (Vorbis, fixed compression, see OGG_*)
- Copy all other files as-is
- Multiprocessing by default: workers = max(1, ncpu - 1)

Requires:
  pip install torch torchaudio
And torchaudio must be built with sox (or have a working sox backend).

Example:
  python resample-dir.py --input-dir /path/in --output-dir /path/out [--format ogg]
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

import soundfile as sf
import torch
import torchaudio


TARGET_SR = 16_000

# Fixed OGG settings (--format ogg): Vorbis, VBR. libsndfile maps compression_level to
# quality = 1 - level, i.e. 0.4 -> quality 0.6 (higher level = smaller file / lower quality).
OGG_FORMAT = "OGG"
OGG_SUBTYPE = "VORBIS"
OGG_COMPRESSION_LEVEL = 0.4


@dataclass(frozen=True)
class Job:
    src: Path
    dst: Path
    is_m4a: bool
    is_wav: bool
    fmt: str = "wav"


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _rel_dst_path(src: Path, input_root: Path, output_root: Path) -> Path:
    rel = src.relative_to(input_root)
    return output_root / rel


def _make_jobs(input_root: Path, output_root: Path, fmt: str = "wav") -> list[Job]:
    out_suffix = f".{fmt}"
    jobs: list[Job] = []
    for p in input_root.rglob("*"):
        if p.is_dir():
            continue
        dst = _rel_dst_path(p, input_root, output_root)
        suffix = p.suffix.lower()
        is_m4a = suffix in (".m4a", ".flac")
        is_wav = suffix == ".wav"
        if is_m4a or is_wav:
            dst = dst.with_suffix(out_suffix)

        jobs.append(Job(src=p, dst=dst, is_m4a=is_m4a, is_wav=is_wav, fmt=fmt))
    return jobs


def _save(waveform: torch.Tensor, sr: int, dst: Path, fmt: str) -> None:
    """Write mono [1, T] waveform as 16-bit PCM WAV or fixed-quality OGG Vorbis."""
    if fmt == "ogg":
        sf.write(
            str(dst),
            waveform.squeeze(0).numpy(),
            sr,
            format=OGG_FORMAT,
            subtype=OGG_SUBTYPE,
            compression_level=OGG_COMPRESSION_LEVEL,
        )
    else:
        torchaudio.save(str(dst), waveform, sr, bits_per_sample=16)


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


class EmptyAudioError(Exception):
    pass


def _load(path: Path) -> Tuple[torch.Tensor, int]:
    """Load as [C, T] float32. soundfile first (robust for wav/flac/ogg), torchaudio/torchcodec as fallback (m4a etc.)."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
        if data.shape[0] == 0:
            raise EmptyAudioError(f"empty audio (0 samples): {path}")
        return torch.from_numpy(data.T.copy()), sr
    except EmptyAudioError:
        raise
    except Exception:
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[-1] == 0:
            raise EmptyAudioError(f"empty audio (0 samples): {path}")
        return waveform, sr


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
            waveform, sr = _load(job.src)  # [C, T], sr
            waveform, out_sr = _resample_to_16k_mono(waveform, sr)

            _save(waveform, out_sr, job.dst, job.fmt)
            return "ok", f"m4a/flac->{job.fmt}16k {job.src} -> {job.dst}"
        elif job.is_wav:
            waveform, sr = _load(job.src)  # [C, T], sr
            waveform, out_sr = _resample_to_16k_mono(waveform, sr)

            _save(waveform, out_sr, job.dst, job.fmt)
            return "ok", f"wav-{sr}->{job.fmt}-16k {job.src} -> {job.dst}"
        else:
            shutil.copy2(job.src, job.dst)
            return "ok", f"copy {job.src} -> {job.dst}"

    except EmptyAudioError as e:
        return "err", f"{job.src} -> {job.dst}\n{e}"
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
    parser.add_argument(
        "--format",
        choices=["wav", "ogg"],
        default="wav",
        help="Output format for audio files: wav = PCM 16-bit, ogg = Vorbis with fixed compression (default: wav)",
    )
    parser.add_argument("--backend", type=str, default="sox_io", help="Preferred torchaudio backend (default: sox_io)")
    args = parser.parse_args()

    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"ERROR: input-dir does not exist or is not a directory: {input_root}", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)

    _init_torchaudio_backend(args.backend)

    jobs = _make_jobs(input_root, output_root, args.format)
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
    print(f"Found {total} files. Workers={args.workers}. Target SR={TARGET_SR}. Format={args.format}. Backend preference={args.backend}")

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

