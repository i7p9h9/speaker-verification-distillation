"""ONNX-friendly USAD front-end: Kaldi fbank built from plain conv/matmul layers (no ``torchaudio.compliance.kaldi``).

``UsadFbank`` reproduces ``usad_model.wav_to_fbank`` for a full-length batch (no padding):

    wav - mean(wav)  ->  Kaldi fbank (hann, 25/10 ms, pre-emphasis 0.97, per-frame DC removal, 20 Hz .. Nyquist,
    log with float32-eps floor)  ->  drop the odd last frame  ->  (x - norm_mean) / (2 * norm_std)

It is built from ``voicesdk.nn.feat.features_kaldi.FBank`` and is traceable by ``torch.onnx.export``.
``UsadConvertible`` = ``UsadFbank`` + the Conformer encoder, takes either a waveform (``include_feats``)
or already normalised mel features ``(B, T, mel_dim)``.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from voicesdk.nn.feat.features_kaldi import FBank

from .usad_model import UsadModel

_FLOAT_EPS = torch.finfo(torch.float32).eps  # Kaldi's log floor


def _hann_nonperiodic(n: int) -> torch.Tensor:
    return torch.hann_window(n, periodic=False)  # Kaldi "hanning"


class UsadFbank(nn.Module):
    def __init__(
        self,
        mel_dim: int = 128,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        sample_rate: int = 16000,
        frame_length_ms: int = 25,
        frame_shift_ms: int = 10,
        low_freq: float = 20.0,
    ) -> None:
        super().__init__()
        self.norm_mean, self.norm_std = norm_mean, norm_std
        self.fbank = FBank(
            sr=sample_rate,
            winlen=sample_rate * frame_length_ms // 1000,
            winstep=sample_rate * frame_shift_ms // 1000,
            mel_min=low_freq,
            mel_max=sample_rate / 2,
            num_mels=mel_dim,
            premph_k=0.97,
            winfunc=_hann_nonperiodic,
            remove_dc=True,
            do_log=False,
        )

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:  # (B, L) -> (B, T_even, mel_dim)
        x = wavs - wavs.mean(dim=-1, keepdim=True)
        feats = torch.log(torch.clamp(self.fbank(x), min=_FLOAT_EPS))
        feats = feats[:, : feats.shape[1] - feats.shape[1] % 2]
        return (feats - self.norm_mean) / (self.norm_std * 2)


class UsadConvertible(nn.Module):
    """USAD encoder with the fbank inside. ``forward`` returns the list of per-layer hidden states."""

    def __init__(self, usad: UsadModel, include_feats: bool = True) -> None:
        super().__init__()
        self.encoder = usad.encoder  # same parameter names as UsadModel.encoder
        self.num_layers: int = usad.num_layers
        self.encoder_dim: int = usad.encoder_dim
        self.fbank = UsadFbank(mel_dim=usad.mel_dim) if include_feats else None

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mel = self.fbank(x) if self.fbank is not None else x
        _, _, layer_results = self.encoder(inputs=mel, return_hidden=True)
        return layer_results["hidden_states"]
