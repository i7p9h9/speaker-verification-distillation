"""Kaldi-like log Mel filter bank of the BUT ResNet101 x-vector extractors.

Copied from VBx (``VBx/features.py``, Copyright Brno University of Technology,
Apache License 2.0), only the functions used by the extraction.
"""

from __future__ import annotations

import numpy as np


def framing(a: np.ndarray, window: int, shift: int = 1) -> np.ndarray:
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0] * shift, a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def mel_inv(x):
    return (np.exp(x / 1127.0) - 1.0) * 700.0


def mel(x):
    return 1127.0 * np.log(1.0 + x / 700.0)


def preemphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return x - np.c_[x[..., :1], x[..., :-1]] * coef


def mel_fbank_mx(winlen_nfft, fs, NUMCHANS=20, LOFREQ=0.0, HIFREQ=None, warp_fn=mel, inv_warp_fn=mel_inv, htk_bug=True):
    """Mel filterbank as an array (NFFT/2+1 x NUMCHANS), NFFT is ``winlen_nfft``
    rounded up to a power of two (or ``-winlen_nfft`` if negative)."""
    HIFREQ = 0.5 * fs if not HIFREQ else HIFREQ
    nfft = 2 ** int(np.ceil(np.log2(winlen_nfft))) if winlen_nfft > 0 else -int(winlen_nfft)

    fbin_mel = warp_fn(np.arange(nfft / 2 + 1, dtype=float) * fs / nfft)
    cbin_mel = np.linspace(warp_fn(LOFREQ), warp_fn(HIFREQ), NUMCHANS + 2)
    cind = np.floor(inv_warp_fn(cbin_mel) / fs * nfft).astype(int) + 1
    mfb = np.zeros((len(fbin_mel), NUMCHANS))
    for i in range(NUMCHANS):
        mfb[cind[i] : cind[i + 1], i] = (cbin_mel[i] - fbin_mel[cind[i] : cind[i + 1]]) / (cbin_mel[i] - cbin_mel[i + 1])
        mfb[cind[i + 1] : cind[i + 2], i] = (cbin_mel[i + 2] - fbin_mel[cind[i + 1] : cind[i + 2]]) / (
            cbin_mel[i + 2] - cbin_mel[i + 1]
        )
    if LOFREQ > 0.0 and float(LOFREQ) / fs * nfft + 0.5 > cind[0] and htk_bug:
        mfb[cind[0], :] = 0.0  # Just to be HTK compatible
    return mfb


def fbank_htk(x, window, noverlap, fbank_mx, nfft=None, USEPOWER=False, PREEMCOEF=0.97, ZMEANSOURCE=False):
    """Log Mel filter bank outputs (frames x NUMCHANS), HTK style (no energy)."""
    if type(USEPOWER) == bool:
        USEPOWER += 1
    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(window.size)))
    x = framing(x.astype("float"), window.size, window.size - noverlap).copy()
    if ZMEANSOURCE:
        x -= x.mean(axis=1)[:, np.newaxis]
    if PREEMCOEF is not None:
        x = preemphasis(x, PREEMCOEF)
    x *= window
    x = np.fft.rfft(x, nfft)
    x = x.real**2 + x.imag**2
    if USEPOWER != 2:
        x **= 0.5 * USEPOWER
    return np.log(np.maximum(1.0, np.dot(x, fbank_mx)))


def povey_window(winlen: int) -> np.ndarray:
    return np.power(0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, winlen)), 0.85)


def add_dither(x: np.ndarray, level: float = 8) -> np.ndarray:
    return x + level * (np.random.rand(*x.shape) * 2 - 1)


def cmvn_floating_kaldi(x: np.ndarray, LC: int, RC: int, norm_vars: bool = True) -> np.ndarray:
    """Mean (and variance) normalization over a floating window of ``LC`` + 1 +
    ``RC`` frames, shifted to fit at the edges (Kaldi style); global if shorter."""
    N, dim = x.shape
    win_len = min(len(x), LC + RC + 1)
    win_start = np.maximum(np.minimum(np.arange(-LC, N - LC), N - win_len), 0)
    f = np.r_[np.zeros((1, dim)), np.cumsum(x, 0)]
    x = x - (f[win_start + win_len] - f[win_start]) / win_len
    if norm_vars:
        f = np.r_[np.zeros((1, dim)), np.cumsum(x**2, 0)]
        x /= np.sqrt((f[win_start + win_len] - f[win_start]) / win_len)
    return x


class Fbank:
    """The front end of ``extract_xvectors_melbank``: 64 log Mel bands, 25 ms
    window, 10 ms shift, sliding CMN over 3 s."""

    def __init__(self, sample_rate: int) -> None:
        if sample_rate == 8000:
            self.noverlap, self.winlen, hifreq = 120, 200, 3700
        elif sample_rate == 16000:
            self.noverlap, self.winlen, hifreq = 240, 400, 7600
        else:
            raise ValueError(f"Only 8kHz and 16kHz are supported. Got {sample_rate} instead.")
        self.window = povey_window(self.winlen)
        self.fbank_mx = mel_fbank_mx(self.winlen, sample_rate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=hifreq, htk_bug=False)

    def __call__(self, seg: np.ndarray) -> np.ndarray:
        # mirror noverlap//2 initial and final samples
        seg = np.r_[seg[self.noverlap // 2 - 1 :: -1], seg, seg[-1 : -self.winlen // 2 - 1 : -1]]
        fea = fbank_htk(seg, self.window, self.noverlap, self.fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        return cmvn_floating_kaldi(fea, 150, 149, norm_vars=False).astype(np.float32)
