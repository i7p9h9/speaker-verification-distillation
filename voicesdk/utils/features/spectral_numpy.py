"""
NumPy procedural implementation of the FFT stage of SpectralFeaturesTF.

Output shape: [nfft//2, T]  — identical axis order to the torch class
              (freq bins × time frames, same as x[:, :nfft//2, :] after conv1d).

Precision: float64 internally, so the result matches the float32 torch class
           to within float32 rounding (~1e-6 relative), well inside e-7 atol
           for unit-normalised signals.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Window
# ---------------------------------------------------------------------------

def make_window(frame_length: int, window: str) -> np.ndarray:
    """
    Build an analysis window of length *frame_length*.

    Matches scipy.signal.windows.<name>(frame_length, sym=True) exactly:

      hann    w[n] = 0.5 * (1 - cos(2π·n / (M-1)))          n=0..M-1
      hamming w[n] = 0.54 - 0.46·cos(2π·n / (M-1))          n=0..M-1
      cosine  w[n] = sin(π·(n+0.5) / M)                      n=0..M-1
      rect    w[n] = 1

    Parameters
    ----------
    frame_length : number of samples in the analysis window
    window       : 'hann' | 'hanning' | 'hamming' | 'cosine' | anything else → rect

    Returns
    -------
    w : float64 array of shape (frame_length,)
    """
    M = frame_length
    n = np.arange(M, dtype=np.float64)
    w = window.lower() if window else ""

    if w in ("hann", "hanning"):
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (M - 1)))

    if w == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (M - 1))

    if w == "cosine":
        return np.sin(np.pi * (n + 0.5) / M)

    return np.ones(M, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. Padding
# ---------------------------------------------------------------------------

def pad_signal(signal: np.ndarray, frame_length: int, mode: str = "none") -> np.ndarray:
    """
    Zero-pad *signal* according to *mode* before framing.

    mode
    ----
    "none"   No padding.  Frames start at sample 0; the last full frame ends
             at or before the last sample.  Matches SpectralFeaturesTF exactly.
             num_frames = (N - L) // S + 1

    "center" Pad L//2 zeros on each side (librosa center=True convention).
             The first frame is centred on sample 0 of the original signal,
             so every input sample contributes to at least one frame centre.
             num_frames ≈ N // S + 1  (exact: (N + 2*(L//2) - L) // S + 1)

    "right"  Pad L-1 zeros on the right.  The last input sample is included
             in a complete frame.
             num_frames = (N + L - 1 - L) // S + 1 = (N - 1) // S + 1

    "left"   Pad L-1 zeros on the left (fully causal / look-back framing).
             The first output frame is built entirely from the look-back zeros
             ending at input sample 0.
             num_frames = (N - 1) // S + 1

    Parameters
    ----------
    signal       : 1-D float64 array of length N
    frame_length : L — analysis window size
    mode         : one of the strings above

    Returns
    -------
    padded_signal : 1-D float64 array (new allocation for padded modes, view for "none")
    """
    mode = mode.lower()
    L = frame_length

    if mode == "none":
        return signal

    if mode == "center":
        pad = L // 2
        return np.pad(signal, (pad, pad))

    if mode == "right":
        return np.pad(signal, (0, L - 1))

    if mode == "left":
        return np.pad(signal, (L - 1, 0))

    raise ValueError(f"Unknown padding mode '{mode}'. Choose: none | center | right | left")


# ---------------------------------------------------------------------------
# 3. Framing
# ---------------------------------------------------------------------------

def count_frames(signal_len: int, frame_length: int, frame_step: int) -> int:
    """
    Number of frames produced by stride-frame_step sliding over signal_len samples.

    Formula: floor((signal_len - frame_length) / frame_step) + 1
    (same as PyTorch conv1d with padding=0)
    """
    if signal_len < frame_length:
        return 0
    return (signal_len - frame_length) // frame_step + 1


def extract_frames(signal: np.ndarray, frame_length: int, frame_step: int) -> np.ndarray:
    """
    Slice signal into overlapping frames via stride tricks (zero-copy view).

    Parameters
    ----------
    signal       : 1-D contiguous float64 array of length N
    frame_length : samples per frame  (L)
    frame_step   : hop size           (S)

    Returns
    -------
    frames : shape (T, L)  — read-only view into signal
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    T = count_frames(len(signal), frame_length, frame_step)
    shape   = (T, frame_length)
    strides = (signal.strides[0] * frame_step, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)


# ---------------------------------------------------------------------------
# 4. Window application + FFT zero-pad
# ---------------------------------------------------------------------------

def apply_window_and_pad(
    frames: np.ndarray,
    win: np.ndarray,
    fft_length: int,
) -> np.ndarray:
    """
    Multiply each frame by the window and zero-pad to fft_length.

    Parameters
    ----------
    frames     : (T, frame_length)
    win        : (frame_length,)
    fft_length : N >= frame_length

    Returns
    -------
    padded : (T, fft_length) float64
    """
    T, L = frames.shape
    padded = np.zeros((T, fft_length), dtype=np.float64)
    padded[:, :L] = frames * win[np.newaxis, :]
    return padded


# ---------------------------------------------------------------------------
# 5. Spectrum magnitude
# ---------------------------------------------------------------------------

def rfft_magnitude(frames_padded: np.ndarray) -> np.ndarray:
    """
    One-sided amplitude spectrum via numpy rfft.

    Equivalent to the torch conv1d DFT:
        xr[k] = Re(X[k]),  xi[k] = -Im(X[k])  →  sqrt(xr²+xi²) = |X[k]|

    Drops the Nyquist bin (same as torch's x[:, :nfft//2, :]).

    Parameters
    ----------
    frames_padded : (T, N)

    Returns
    -------
    mag : (T, N//2)
    """
    N = frames_padded.shape[1]
    spectrum = np.fft.rfft(frames_padded, axis=1)   # (T, N//2+1)
    return np.abs(spectrum[:, : N // 2])             # drop Nyquist, (T, N//2)


# ---------------------------------------------------------------------------
# 6. Signal normalization
# ---------------------------------------------------------------------------

def _normalize_signal(signal: np.ndarray, mode: str, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize a 1-D signal before STFT, matching SpectralFeaturesTF.forward().

    mode
    ----
    "mean_max"  subtract mean, divide by max(|x|), clip denominator to eps.
                Matches normalize_mean_max() in spectra-features.py.

    "mean_std"  subtract mean, divide by std (ddof=1, Bessel-corrected to match
                torch.Tensor.std() default), clip denominator to eps.
                Matches normalize_mean_std() in spectra-features.py.

    None / ""   no normalization (default, pass-through).

    Parameters
    ----------
    signal : 1-D float64 array
    mode   : 'mean_max' | 'mean_std' | None
    eps    : clip floor for the denominator                    (default 1e-8)

    Returns
    -------
    normalized : 1-D float64 array (new allocation)
    """
    if not mode:
        return signal

    x = signal - signal.mean()

    if mode == "mean_max":
        denom = np.abs(x).max()
        denom = max(denom, eps)
        return x / denom

    if mode == "mean_std":
        # ddof=1 matches torch.Tensor.std() which uses Bessel's correction
        denom = x.std(ddof=1)
        denom = max(denom, eps)
        return x / denom

    raise ValueError(f"Unknown normalize mode '{mode}'. Choose: mean_max | mean_std | None")


# ---------------------------------------------------------------------------
# 7. Pre-emphasis
# ---------------------------------------------------------------------------

def apply_pre_emphasis(signal: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    High-pass pre-emphasis filter matching PreEmphasis in spectra-features.py.

    Uses reflect padding on the left (F.pad(x, (1,0), 'reflect')), then
    convolves with [-coef, 1]:

      y[0] = x[0] - coef * x[1]   (reflect: x[-1] = x[1])
      y[n] = x[n] - coef * x[n-1]  for n >= 1

    Parameters
    ----------
    signal : 1-D array (any dtype)
    coef   : pre-emphasis coefficient (default 0.97)

    Returns
    -------
    out : same shape and dtype as signal
    """
    if len(signal) < 2:
        return signal.copy()
    out = np.empty_like(signal)
    out[0] = signal[0] - coef * signal[1]      # reflect boundary
    out[1:] = signal[1:] - coef * signal[:-1]
    return out


# ---------------------------------------------------------------------------
# 8. Mel filterbank
# ---------------------------------------------------------------------------

def _hz2mel(hz: np.ndarray) -> np.ndarray:
    """2595 * log10(1 + hz/700)  — matches spectra-features.py hz2mel."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def make_filterbank(
    fft_length: int    = 512,
    num_bins: int      = 80,
    sample_rate: int   = 16000,
    low_freq: float    = 20.0,
    high_freq: float   = 7600.0,
) -> np.ndarray:
    """
    Triangular mel filterbank matching get_filterbanks() in spectra-features.py.

    The torch class calls get_filterbanks(nfft=self.nfft // 2, ...), i.e. it
    passes *half* the FFT size.  We accept the full fft_length and handle the
    halving internally so the caller doesn't need to think about it.

    Parameters
    ----------
    fft_length  : full FFT size (e.g. 512)
    num_bins    : number of mel filters                   (default 80)
    sample_rate : audio sample rate in Hz                  (default 16000)
    low_freq    : lowest filter centre in Hz               (default 20)
    high_freq   : highest filter centre in Hz              (default 7600)

    Returns
    -------
    fb : float64 array, shape (fft_length//2, num_bins)
         Row k holds the weight of FFT bin k for each mel filter.
         Row 0 (DC) is always zero (matches the vstack in get_filterbanks).
    """
    K = fft_length // 2          # number of one-sided FFT bins

    lowmel  = _hz2mel(np.float64(low_freq))
    highmel = _hz2mel(np.float64(high_freq))
    melpoints = np.linspace(lowmel, highmel, num_bins + 2)

    lower_edge = melpoints[:-2].reshape(1, -1)   # (1, num_bins)
    center     = melpoints[1:-1].reshape(1, -1)  # (1, num_bins)
    upper_edge = melpoints[2:].reshape(1, -1)    # (1, num_bins)

    # FFT bin frequencies in mel — matches np.linspace(0, sr//2, K)[1:] in get_filterbanks
    bin_hz  = np.linspace(0, sample_rate // 2, K)          # (K,)
    bin_mel = _hz2mel(bin_hz)[1:].reshape(-1, 1)            # (K-1, 1)  drop DC

    lower_slopes = (bin_mel - lower_edge) / (center - lower_edge)
    upper_slopes = (upper_edge - bin_mel) / (upper_edge - center)

    weights = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))  # (K-1, num_bins)
    fb = np.vstack([np.zeros((1, num_bins)), weights])                  # (K, num_bins)
    return fb.astype(np.float64)


def apply_melbanks(
    stft: np.ndarray,
    filterbank: np.ndarray,
    log: bool   = False,
    eps: float  = 1e-6,
) -> np.ndarray:
    """
    Apply mel filterbank to a magnitude spectrogram.

    Matches calc_melbanks() in SpectralFeaturesTF:
        conv1d(stft, melbanks_kernel)  ≡  filterbank.T @ stft  (per time step)
        clip(result, eps)
        optional log

    Parameters
    ----------
    stft       : (fft_length//2, T) — output of compute_spectrogram(mode='fft')
    filterbank : (fft_length//2, num_bins) — output of make_filterbank()
    log        : if True, apply natural log after clipping
    eps        : clip floor

    Returns
    -------
    mel : (num_bins, T)
    """
    mel = filterbank.T @ stft          # (num_bins, T)
    mel = np.clip(mel, eps, None)
    if log:
        mel = np.log(mel)
    return mel


# ---------------------------------------------------------------------------
# 9. Top-level
# ---------------------------------------------------------------------------

def compute_spectrogram(
    signal: np.ndarray,
    frame_length: int = 400,
    frame_step: int = 160,
    fft_length: int = 512,
    window: str = "hann",
    padding: str = "none",
    eps: float = 1e-6,
    sqrt_mag: bool = True,
    mode: str = "fft",
    num_bins: int = 80,
    sample_rate: int = 16000,
    low_freq: float = 20.0,
    high_freq: float = 7600.0,
    normalize_signal: str = None,
    pre_emphasis_coef: float = None,
    pad_input: bool = False,
    normalize_spectrogram: bool = False,
    normalize_spectrogram_std: bool = False,
    shift_value: float = 0.0,
) -> np.ndarray:
    """
    Magnitude / mel spectrogram matching SpectralFeaturesTF.

    Parameters
    ----------
    signal       : 1-D float array, length N
    frame_length : analysis window length in samples                (default 400)
    frame_step   : hop size in samples                              (default 160)
    fft_length   : FFT size, must be >= frame_length                (default 512)
    window       : 'hann' | 'hanning' | 'hamming' | 'cosine' | 'rect'
    padding      : 'none'   — no padding, matches SpectralFeaturesTF (default)
                   'center' — L//2 zeros each side  (librosa center=True)
                   'right'  — L-1 zeros on the right (all samples covered)
                   'left'   — L-1 zeros on the left  (causal look-back)
    eps          : clip floor                                       (default 1e-6)
    sqrt_mag     : True  → sqrt(re²+im²)   [SpectralFeaturesTF default]
                   False → (re²+im²) / fft_length
    mode             : 'fft'        — raw magnitude spectrum (fft_length//2, T)
                       'melbanks'   — mel filterbank applied  (num_bins, T)
                       'logmelbanks'— mel filterbank + log    (num_bins, T)
    num_bins         : number of mel filters (used when mode != 'fft')  (default 80)
    sample_rate      : audio sample rate in Hz                          (default 16000)
    low_freq         : lowest mel filter edge in Hz                     (default 20)
    high_freq        : highest mel filter edge in Hz                    (default 7600)
    normalize_signal     : None       — no normalization (default)
                           'mean_max' — subtract mean, divide by peak absolute value
                           'mean_std' — subtract mean, divide by std (ddof=1)
    pre_emphasis_coef    : None       — disabled (default)
                           float      — apply pre-emphasis y[n]=x[n]-c·x[n-1] after
                                        signal normalization, with reflect boundary
    normalize_spectrogram: if True, subtract per-bin mean over time axis
                           (matches SpectralFeaturesTF normalize_spectrogram=True)
                           x -= mean(x, axis=time)   →  zero temporal mean per bin
    shift_value          : scalar added to the final output              (default 0.0)
                           matches SpectralFeaturesTF shift_value

    Returns
    -------
    spec : float64 array
           mode='fft'         → shape (fft_length//2, T)
           mode='melbanks'    → shape (num_bins, T)
           mode='logmelbanks' → shape (num_bins, T)
    """
    assert fft_length >= frame_length, "fft_length must be >= frame_length"
    assert mode in ("fft", "melbanks", "logmelbanks"), \
        f"mode must be 'fft' | 'melbanks' | 'logmelbanks', got '{mode}'"

    # Normalize in float32 to match torch (which runs entirely in float32).
    # Converting to float64 first would give a slightly different normalization
    # factor, amplifying the error for large-valued signals.
    sig = np.asarray(signal, dtype=np.float32).ravel()
    sig = _normalize_signal(sig, normalize_signal)                   # float32 (N,)
    if pad_input:
        sig = np.pad(sig, (frame_step // 2, frame_step // 2))

    if pre_emphasis_coef is not None:
        sig = apply_pre_emphasis(sig, pre_emphasis_coef)             # float32 (N,)
    sig = sig.astype(np.float64)                                     # → float64 for FFT
    sig = pad_signal(sig, frame_length, mode=padding)             # (N',)
    win = make_window(frame_length, window)                       # (L,)
    frm = extract_frames(sig, frame_length, frame_step)           # (T, L)
    padded = apply_window_and_pad(frm, win, fft_length)              # (T, N)

    if sqrt_mag:
        mag = rfft_magnitude(padded)                                 # (T, N//2)
    else:
        N = fft_length
        spectrum = np.fft.rfft(padded, axis=1)
        mag = (spectrum.real ** 2 + spectrum.imag ** 2)[:, : N // 2] / N

    mag = np.clip(mag, eps, None)
    stft = mag.T                                                     # (N//2, T)

    if mode != "fft":
        fb   = make_filterbank(fft_length, num_bins, sample_rate, low_freq, high_freq)
        stft = apply_melbanks(stft, fb, log=(mode == "logmelbanks"), eps=eps)

    # Output normalizations — applied after mel/log, matching forward() order in torch
    if normalize_spectrogram:
        # subtract per-freq-bin temporal mean  (torch: x - mean(x, dim=2, keepdim=True))
        stft = stft - stft.mean(axis=1, keepdims=True)   # axis-1 = time
    elif normalize_spectrogram_std:
        mean = stft.mean()
        std  = stft.std(ddof=0) + eps   # torch std(dim=(1,2)) использует ddof=0
        stft = (stft - mean) / std

    if shift_value:
        stft = stft + shift_value

    return stft
