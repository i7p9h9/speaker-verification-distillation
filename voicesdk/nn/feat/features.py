"""
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

"""

import torch
import torchaudio
from torch import nn

from .preproc import NormalizeAudio, PreEmphasis


class FbankAug(nn.Module):
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10), freq_start_bin=0):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.freq_start_bin = freq_start_bin
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(
            self.freq_start_bin, max(1, D - mask_len.max()), (batch, 1), device=x.device
        ).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class MelBanks(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        f_min=20,
        f_max=7600,
        n_mels=80,
        do_spec_aug=False,
        norm_signal=False,
        do_preemph=True,
        spec_norm="mn",
        freq_start_bin=0,
        num_apply_spec_aug=1,
        freq_mask_width=(0, 8),
        time_mask_width=(0, 10),
    ):
        super(MelBanks, self).__init__()
        self.num_apply_spec_aug = num_apply_spec_aug
        self.torchfbank = torch.nn.Sequential(
            NormalizeAudio() if norm_signal else nn.Identity(),
            PreEmphasis() if do_preemph else nn.Identity(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
                window_fn=torch.hamming_window,
            ),
        )
        self.spec_norm = spec_norm
        if spec_norm == "mn":
            self.spec_norm = lambda x: x - torch.mean(x, dim=-1, keepdim=True)
        elif spec_norm == "mvn":
            self.spec_norm = lambda x: (
                (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-8)
            )
        elif spec_norm == "bn":
            self.spec_norm = nn.BatchNorm1d(n_mels, affine=False, momentum=0.01)
        else:
            pass
        if do_spec_aug:
            self.specaug = FbankAug(
                freq_start_bin=freq_start_bin, freq_mask_width=freq_mask_width, time_mask_width=time_mask_width
            )  # Spec augmentation
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x) + 1e-6
                x = x.log()
                x = self.spec_norm(x)
                if self.training:
                    for _ in range(self.num_apply_spec_aug):
                        x = self.specaug(x)
        return x.to(xdtype)


def cos_sin_phase(stft):
    """
    stft: Complex STFT tensor of shape (B, F, T).
    Returns: (B, 3, F, T): [magnitude, cos(phase), sin(phase)].
    """
    phase = torch.angle(stft)  # (B, F, T)
    cos_phase = torch.cos(phase)  # (B, F, T)
    sin_phase = torch.sin(phase)  # (B, F, T)
    magnitude = torch.abs(stft)  # (B, F, T)
    return torch.stack([magnitude, cos_phase, sin_phase], dim=1)


class STFT(nn.Module):
    def __init__(self, n_fft, win_length, hop_length, eps=1e-6, with_phase=True):
        # --- For manual STFT (complex) if phase is requested ---
        super(STFT, self).__init__()
        self.stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.hamming_window,
            power=None,  # <--- returns complex STFT
            normalized=False,
            center=True,
            pad_mode="reflect",
        )
        self.with_phase = with_phase
        self.eps = eps

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            # (a) compute complex STFT (linear frequency)
            stft_cplx = self.stft_transform(x)  # shape (B, freq, frames), complex

            # (b) get [magnitude, cos(phase), sin(phase)] in linear freq
            csp = cos_sin_phase(stft_cplx)  # (B, 3, freq, frames)
            mag_lin = csp[:, 0]  # (B, freq, T)
            cos_lin = csp[:, 1]  # (B, freq, T)
            sin_lin = csp[:, 2]  # (B, freq, T)
            # For magnitude, do log scaling
            mag_lin = torch.log(mag_lin + self.eps)

            if self.with_phase:
                # (d) Stack => (B, 3, n_mels, T)
                out_3ch = torch.stack([mag_lin, cos_lin, sin_lin], dim=1)
                return out_3ch.to(xdtype)
            else:
                out_1ch = mag_lin.unsqueeze(1)
                return out_1ch.to(xdtype)
