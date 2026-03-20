'''
Matched with torchaudio implementation
'''
import typing as tp

import torch
import torch.nn as nn
from torchaudio.transforms import MelScale

from .stft import STFT


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: tp.Optional[int] = None,
        hop_length: tp.Optional[int] = None,
        f_min: float = 0.0,
        f_max: tp.Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: tp.Callable[..., torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: tp.Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: tp.Optional[bool] = None,
        norm: tp.Optional[str] = None,
        mel_scale: str = "htk",
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pad = pad
        self.n_mels = n_mels
        self.window_fn = window_fn
        self.power = power
        self.normalized = normalized
        self.wkwargs = wkwargs
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

        self.spectrogram = STFT(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window='hann',
            fft_window=torch.hann_window(400)
        )
        self.mel_scale = MelScale(
            self.n_mels,
            self.sample_rate, 
            self.f_min,
            self.f_max,
            self.n_fft // 2 + 1,
            norm,
            mel_scale
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        specgram = self.spectrogram(waveform)
        specgram = specgram[0] ** 2 + specgram[1] ** 2
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram


class logFbankCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super(logFbankCal, self).__init__()

        self.fbankCal = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=int(win_length*sample_rate),
            hop_length=int(hop_length*sample_rate),
            n_mels=n_mels)

    def forward(self, x):
        out = self.fbankCal(x)
        out = torch.log(out + 1e-6)
        # out = out - out.mean(axis=2).unsqueeze(dim=2)
        out = out - out.mean(axis=-1, keepdim=True)
        return out
