"""Utility helpers: audio I/O, tensor manipulation, signal metrics."""

import typing as tp
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


def load_audio(
    path: tp.Union[str, Path],
    target_sr: int,
) -> tp.Tuple[torch.Tensor, int]:
    """Load an audio file and resample to *target_sr* if needed.

    Args:
        path: Path to the audio file (any format supported by soundfile).
        target_sr: Desired output sample rate in Hz.

    Returns:
        Tuple of ``(tensor, target_sr)`` where tensor has shape ``(C, T)``.
    """
    data, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile returns (T, C); transpose to (C, T)
    tensor = torch.from_numpy(data.T.copy())
    if native_sr != target_sr:
        tensor = torchaudio.functional.resample(tensor, native_sr, target_sr)
    return tensor, target_sr


def crop_or_pad(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    """Crop or zero-pad tensor along the **last** dimension to *target_length*.

    Args:
        tensor: Input tensor of shape ``(..., T)``.
        target_length: Desired length along the last dimension.

    Returns:
        Tensor with last dimension equal to *target_length*.
    """
    length = tensor.shape[-1]
    if length >= target_length:
        return tensor[..., :target_length]
    pad_size = target_length - length
    return F.pad(tensor, (0, pad_size))


def rms(
    x: torch.Tensor,
    dim=None,
    keepdim: bool = False,
    eps: float = 1e-12,
    unbiased: bool = False
) -> torch.Tensor:
    """
    RMS after removing DC component.
    Equivalent to sqrt(mean((x - mean)^2)).
    """
    x = x.float()
    return torch.std(
        x,
        dim=dim,
        unbiased=unbiased,
        keepdim=keepdim,
    ) + eps
