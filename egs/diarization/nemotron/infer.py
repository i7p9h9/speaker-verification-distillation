"""Offline Nemotron-3-Diarization over a directory of audio files.

Runs ``nvidia/Nemotron-3-Diarization`` through Hugging Face Transformers in
offline mode: the whole recording goes into one forward, which chunks it
internally (340 encoder frames + 40 look-ahead, i.e. the 30.4 s offline
buffer of the model card), so file length is not limited.

For every ``<in-dir>/<rel>.<ext>`` writes

    <out-dir>/<rel>.npz   probs (num_frames, 8) float32 speaker probabilities,
                          frame_shift (s), duration (s)
    <out-dir>/<rel>.json  processor.extract_speaker_dict segments at --threshold

``to_rttm.py`` converts these to RTTM.  Meant to run inside the Docker image,
see ``run.sh``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.audio_utils import load_audio

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".opus", ".ogg", ".m4a")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=Path("/data/in"))
    p.add_argument("--out-dir", type=Path, default=Path("/data/out"))
    p.add_argument("--model-id", default="nvidia/Nemotron-3-Diarization")
    p.add_argument("--ext", nargs="+", default=list(AUDIO_EXTS))
    p.add_argument("--threshold", type=float, default=0.5, help="speaker probability threshold for the .json")
    p.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exts = {e if e.startswith(".") else f".{e}" for e in args.ext}
    files = sorted(p for p in args.in_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"no audio files with extensions {sorted(exts)} in {args.in_dir}")

    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForAudioFrameClassification.from_pretrained(args.model_id, dtype=dtype).to(args.device).eval()
    sampling_rate = processor.feature_extractor.sampling_rate
    frame_shift = processor.feature_extractor.hop_length / sampling_rate

    for path in tqdm(files, desc="nemotron"):
        rel = path.relative_to(args.in_dir)
        npz_path = (args.out_dir / rel).with_suffix(".npz")
        json_path = npz_path.with_suffix(".json")
        if npz_path.exists() and json_path.exists() and not args.overwrite:
            continue

        audio = load_audio(str(path), sampling_rate=sampling_rate)
        inputs = processor(audio, sampling_rate=sampling_rate).to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            logits = model(**inputs).logits
        segments = processor.extract_speaker_dict(logits, inputs.attention_mask, threshold=args.threshold)[0]

        num_frames = int(inputs.attention_mask[0].sum())
        probs = logits[0, :num_frames].float().sigmoid().cpu().numpy()

        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            probs=probs.astype(np.float32),
            frame_shift=np.float32(frame_shift),
            duration=np.float32(len(audio) / sampling_rate),
        )
        with open(json_path, "w") as f:
            json.dump({"file": str(rel), "threshold": args.threshold, "segments": segments}, f, indent=1)


if __name__ == "__main__":
    main()
