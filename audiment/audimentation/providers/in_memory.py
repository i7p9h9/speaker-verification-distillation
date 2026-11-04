"""In-memory audio provider: all files loaded at init time."""

import random
import typing as tp
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from audimentation.utils import crop_or_pad


class InMemoryAudioProvider:
    """Load all audio files into memory at construction time, sample from RAM.

    Files are stored at their native sample rate as ``(tensor, native_sr)`` pairs.
    Resampling is done lazily at ``sample()`` time.  Thread-safe because the pool
    is read-only after ``__init__``.

    Args:
        paths: List of file paths to preload.
        sample_mode: Sampling strategy. Currently only ``'random'`` is supported.
    """

    def __init__(
        self,
        paths: tp.List[tp.Union[str, Path]],
        sample_mode: str = "random",
    ) -> None:
        if not paths:
            raise ValueError("InMemoryAudioProvider requires at least one path.")
        self.sample_mode = sample_mode
        self._pool: tp.List[tp.Tuple[torch.Tensor, int]] = []
        for path in paths:
            data, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
            tensor = torch.from_numpy(data.T.copy())  # (C, T)
            self._pool.append((tensor, native_sr))

    def sample(self, duration: float, sample_rate: int) -> torch.Tensor:
        """Pick a random in-memory item, resample if needed, and crop/pad.

        Args:
            duration: Desired duration in seconds.
            sample_rate: Target sample rate in Hz.

        Returns:
            Tensor of shape ``(C, T)`` with ``T = round(duration * sample_rate)``.
        """
        tensor, native_sr = random.choice(self._pool)
        if native_sr != sample_rate:
            tensor = torchaudio.functional.resample(tensor, native_sr, sample_rate)
        target_length = round(duration * sample_rate)
        return crop_or_pad(tensor, target_length)
