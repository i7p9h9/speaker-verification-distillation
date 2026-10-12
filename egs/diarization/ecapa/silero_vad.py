"""Silero VAD for a directory of audio files, computed once and cached.

    python egs/diarization/ecapa/silero_vad.py --wav-dir data/wav --out-dir data/silero [--num-workers 4]

For every ``<wav-dir>/<rel>.wav`` writes ``<out-dir>/<rel>.npz`` with

    segments    (K, 2) float32  speech segments of ``get_speech_timestamps``
                                (default parameters), seconds
    prob        (M,)   float32  frame speech probabilities (``audio_forward``)
    prob_shift  ()     float32  seconds per probability (512 samples at 16 kHz,
                                256 at 8 kHz, i.e. 32 ms)

Existing outputs are kept, so re-running only processes new files.  Audio at
8 or 16 kHz is used as is, other rates are resampled to 16 kHz.  The model is
``snakers4/silero-vad`` from torch.hub (downloaded on first use).

``diarize.py --vad-silero <out-dir>`` uses the segments as speech, with
``--vad-silero-prob`` it thresholds the probabilities with ``--onset`` /
``--offset`` / ``--merge-gap`` / ``--min-dur`` instead (so ``tune_vad.py``
can tune them).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import typing as tp
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

_model: tp.Any = None
_get_speech_timestamps: tp.Callable | None = None


def load_silero() -> tp.Tuple[tp.Any, tp.Callable]:
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False, trust_repo=True, verbose=False)
    return model, utils[0]


def _init_worker() -> None:
    global _model, _get_speech_timestamps
    torch.set_num_threads(1)
    _model, _get_speech_timestamps = load_silero()


def silero_file(wav: Path, out: Path) -> None:
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    x = torch.from_numpy(audio.mean(axis=1))
    if sr not in (8000, 16000):
        import torchaudio.functional as AF

        x, sr = AF.resample(x, sr, 16000), 16000
    with torch.no_grad():
        prob = _model.audio_forward(x[None], sr)[0].numpy()
        _model.reset_states()
        stamps = _get_speech_timestamps(x, _model, sampling_rate=sr)
    segments = np.array([(s["start"] / sr, s["end"] / sr) for s in stamps], dtype=np.float32).reshape(-1, 2)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, segments=segments, prob=prob.astype(np.float32), prob_shift=np.float32(512 / 16000))
    tmp.replace(out)


def _job(paths: tp.Tuple[Path, Path]) -> None:
    silero_file(*paths)


def pending(wav_dir: Path, out_dir: Path, ext: str = ".wav") -> tp.List[tp.Tuple[Path, Path]]:
    """``(wav, out)`` pairs without a cached result."""
    pairs = [(w, (out_dir / w.relative_to(wav_dir)).with_suffix(".npz")) for w in sorted(wav_dir.rglob(f"*{ext}"))]
    return [(w, o) for w, o in pairs if not o.exists()]


def compute(wav_dir: Path, out_dir: Path, ext: str = ".wav", num_workers: int = 4) -> None:
    todo = pending(wav_dir, out_dir, ext)
    if not todo:
        return
    with mp.get_context("spawn").Pool(max(num_workers, 1), initializer=_init_worker) as pool:
        for _ in tqdm(pool.imap_unordered(_job, todo), total=len(todo), desc="silero"):
            pass


class SileroCache:
    """Read access to an ``--out-dir`` of this script, by file id (stem)."""

    def __init__(self, directory: Path) -> None:
        self.paths = {p.stem: p for p in sorted(Path(directory).rglob("*.npz"))}
        if not self.paths:
            raise SystemExit(f"no silero results in {directory}, run silero_vad.py")

    def _load(self, file_id: str) -> tp.Dict[str, np.ndarray]:
        if file_id not in self.paths:
            raise SystemExit(f"no silero result for {file_id}, run silero_vad.py")
        with np.load(self.paths[file_id]) as data:
            return dict(data)

    def segments(self, file_id: str) -> tp.List[tp.Tuple[float, float]]:
        return [(float(s), float(e)) for s, e in self._load(file_id)["segments"]]

    def frame_prob(self, file_id: str, num_frames: int, frame_shift: float) -> np.ndarray:
        """Probabilities resampled (zero order hold) to the frame grid of the npz."""
        data = self._load(file_id)
        prob, shift = data["prob"], float(data["prob_shift"])
        index = np.floor((np.arange(num_frames) + 0.5) * frame_shift / shift).astype(int)
        return prob[np.clip(index, 0, len(prob) - 1)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ext", default=".wav")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    compute(args.wav_dir, args.out_dir, args.ext, args.num_workers)


if __name__ == "__main__":
    main()
