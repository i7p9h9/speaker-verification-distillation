"""File-list backed audio provider: reads from disk on every call."""

import random
import typing as tp
from pathlib import Path

import torch

from audimentation.utils import crop_or_pad, load_audio


class FileListAudioProvider:
    """Load a random audio file from a list of paths on each ``sample()`` call.

    Each call is stateless: it picks a random file, loads it from disk,
    resamples to the target sample rate, and crops/pads to the requested
    duration.  Thread-safe because reads are fully stateless.

    Args:
        paths: List of file paths to sample from.
        sample_mode: Sampling strategy. Currently only ``'random'`` is supported.
    """

    def __init__(
        self,
        paths: tp.List[tp.Union[str, Path]],
        sample_mode: str = "random",
    ) -> None:
        if not paths:
            raise ValueError("FileListAudioProvider requires at least one path.")
        self._paths = [Path(p) for p in paths]
        self.sample_mode = sample_mode

    def sample(self, duration: float, sample_rate: int) -> torch.Tensor:
        """Pick a random file, load, resample, and crop/pad to *duration* seconds.

        Args:
            duration: Desired duration in seconds.
            sample_rate: Target sample rate in Hz.

        Returns:
            Tensor of shape ``(C, T)`` with ``T = round(duration * sample_rate)``.
        """
        path = random.choice(self._paths)
        tensor, _ = load_audio(path, sample_rate)
        target_length = round(duration * sample_rate)
        return crop_or_pad(tensor, target_length)
