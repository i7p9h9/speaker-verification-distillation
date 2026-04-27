import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import windows
from torch import Tensor


def pad_dim_zeros_convertable(x, padding, dim=-1, pad_end=True):
    last_dim = x.ndim - 1
    if dim == -1:
        dim = last_dim

    if dim != last_dim:
        x = x.transpose(dim, last_dim)

    if pad_end:
        x = torch.cat([x, torch.zeros_like(x)[..., :padding]], dim=last_dim)
    else:
        x = torch.cat([torch.zeros_like(x)[..., :padding], x], dim=last_dim)

    if dim != last_dim:
        x = x.transpose(dim, last_dim)
    return x


def normalize_mean_max(signal, dim=-1, eps=1e-8):
    signal = signal - signal.mean(dim=dim, keepdims=True)
    max_value = signal.abs().max(dim=dim, keepdims=True).values
    max_value = max_value.clip(1e-8, None)
    return signal / max_value


def normalize_mean_std(signal, dim=-1, eps=1e-8):
    signal = signal - signal.mean(dim=dim, keepdims=True)
    dynamic_range = signal.std(dim=dim, keepdims=True)
    # return signal / (dynamic_range+eps)
    dynamic_range = dynamic_range.clip(eps, None)
    return signal / dynamic_range


def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filterbanks(
    low_freq: int = 20, high_freq: int = 7600, nfilt: int = 80, nfft: int = 512, samplerate: int = 16000
):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param low_freq: lowest band edge of mel filters, default 0 Hz
    :param high_freq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """

    # compute points evenly spaced in mels
    lowmel = hz2mel(low_freq)
    highmel = hz2mel(high_freq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)

    lower_edge_mel = melpoints[:-2].reshape(1, -1)
    center_mel = melpoints[1:-1].reshape(1, -1)
    upper_edge_mel = melpoints[2:].reshape(1, -1)

    spectrogram_bins_mel = hz2mel(np.linspace(0, samplerate // 2, nfft))[1:].reshape(-1, 1)

    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)

    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.vstack([np.zeros((1, nfilt)), mel_weights_matrix])[:, :].astype("float32")


class SpectralFeaturesTF(nn.Module):
    def __init__(
        self,
        frame_length=400,
        frame_step=160,
        fft_length=512,
        sample_rate=16000,
        window="hann",
        normalize_spectrogram=False,
        normalize_vectors=None,
        power_vectors=None,
        cut_fft=None,
        normalize_signal=None,
        eps=1e-6,
        trainable=False,
        mode="melbanks",
        low_freq=20,
        high_freq=7600,
        num_bins=80,
        num_ceps=30,
        fft_mode="abs",
        sqrt_mag=True,
        return_image=True,
        shift_value=0.0,
    ):
        """
        Requirements
        ------------
        input shape must meet the conditions: mod((input.shape[0] - length), shift) == 0
        nfft >= length

        Parameters
        ------------
        :param length: Length of each segment.
        :param shift: Number of points to step for segments
        :param nfft: number of dft points, if None => nfft == length
        :param mode: "abs" - amplitude spectrum; "real" - only real part, "imag" - only imag part,
        "complex" - concatenate real and imag part.
        :param kwargs: unuse

        Input
        -----
        input mut have shape: [n_batch, signal_length, 1]

        Returns
        -------
        A keras model that has output shape of
        (None, nfft / 2, n_time) (if type == "abs" || "real" || "imag") or
        (None, nfft / 2, n_frame, 2) (if type = "abs" & `img_dim_ordering() == 'tf').
        (None, nfft / 2, n_frame, 2) (if type = "complex" & `img_dim_ordering() == 'tf').

        number of time point of output spectrogram: n_time = (input.shape[0] - length) / shift + 1
        """
        super(SpectralFeaturesTF, self).__init__()

        assert mode in ["fft", "melbanks", "logmelbanks", "mfcc"]

        self.convert_mode_on = False

        self.trainable = trainable
        self.length = frame_length if isinstance(frame_length, int) else int(frame_length * sample_rate)
        self.shift = frame_step if isinstance(frame_step, int) else int(frame_step * sample_rate)
        self.normalize_spectrogram = normalize_spectrogram
        self.normalize_vectors = normalize_vectors
        self.power_vectors = power_vectors
        self.normalize_signal = normalize_signal
        self.cut_fft = cut_fft
        self.shift_value = shift_value
        self.sqrt_mag = sqrt_mag
        self.return_image = return_image
        self.window = window
        self.eps = eps
        if fft_length is None:
            self.nfft = frame_length
        else:
            self.nfft = fft_length

        self.samplerate = sample_rate
        self.features = mode
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_bins = num_bins
        self.num_ceps = num_ceps

        if mode in ["melbanks", "logmelbanks", "mfcc"]:
            fft_mode = "abs"
        self.fft_mode = fft_mode
        self.build()

    def build(self):
        assert self.nfft >= self.length

        if self.window:
            if self.window == "hamming":
                window_fn = windows.hamming
            elif self.window in ["hann", "hanning"]:
                window_fn = windows.hann
            elif self.window == "cosine":
                window_fn = windows.cosine
            else:
                window_fn = np.ones
            self.window = np.pad(
                window_fn(self.length), (0, self.nfft - self.length), mode="constant", constant_values=0
            )

        # kernels
        real_kernel = (
            np.asarray([np.cos(2 * np.pi * np.arange(0, self.nfft) * n / self.nfft) for n in range(self.nfft)])
            .astype("float32")
            .T
        )
        if self.window is not None:
            real_kernel *= self.window[:, None]
        real_kernel[self.length :, :] = 0.0
        self.real_kernel = real_kernel[:, np.newaxis, : self.nfft]

        image_kernel = (
            np.asarray([np.sin(2 * np.pi * np.arange(0, self.nfft) * n / self.nfft) for n in range(self.nfft)])
            .astype("float32")
            .T
        )
        if self.window is not None:
            image_kernel *= self.window[:, None]
        image_kernel[self.length :, :] = 0.0
        self.image_kernel = image_kernel[:, np.newaxis, : self.nfft]

        real_part_keras_w = self.real_kernel
        imag_part_keras_w = self.image_kernel
        # linear_to_mel_weight_matrix_keras_w = tf.signal.linear_to_mel_weight_matrix(
        #     num_mel_bins=self.num_bins,
        #     num_spectrogram_bins=self.nfft // 2,
        #     sample_rate=self.samplerate,
        #     lower_edge_hertz=self.low_freq,
        #     upper_edge_hertz=self.high_freq
        # ).numpy() # [num_spectrogram_bins, num_mel_bins]
        linear_to_mel_weight_matrix_keras_w = get_filterbanks(
            nfilt=self.num_bins,
            nfft=self.nfft // 2,
            samplerate=self.samplerate,
            low_freq=self.low_freq,
            high_freq=self.high_freq,
        )
        linear_to_mel_weight_matrix_keras_w = linear_to_mel_weight_matrix_keras_w[:, :, None]

        dct_kernel = np.asarray(
            [
                2 * np.cos(np.pi * n * np.arange(1, 2 * self.num_bins, 2) / (2 * self.num_bins))
                for n in range(self.num_bins)
            ]
        )
        dct_kernel_keras_w = dct_kernel[:, :, None].T

        n = np.arange(self.num_ceps)
        lift = 1 + (self.num_ceps / 2.0) * np.sin(np.pi * n / self.num_ceps)

        stft_dct_kernel = np.asarray(
            [2 * np.cos(np.pi * n * np.arange(1, self.nfft, 2) / (2 * self.num_bins)) for n in range(self.nfft)]
        )
        stft_dct_kernel_w = stft_dct_kernel[:, :, None].T

        # [out_channels, in_channels, filter_width] : torch like conv1d kernel shape
        # [filter_width, in_channels, out_channels] : tf like conv1d kernel shape
        PERSISTENT_BUFFERS = True

        self.register_buffer(
            name="real_part_kernel",
            tensor=torch.from_numpy(real_part_keras_w).transpose(0, 2).float(),
            persistent=PERSISTENT_BUFFERS,
        )
        self.register_buffer(
            name="imag_part_kernel",
            tensor=torch.from_numpy(imag_part_keras_w).transpose(0, 2).float(),
            persistent=PERSISTENT_BUFFERS,
        )
        self.register_buffer(
            name="melbanks_kernel",
            tensor=torch.from_numpy(linear_to_mel_weight_matrix_keras_w).permute(1, 0, 2).float(),
            persistent=PERSISTENT_BUFFERS,
        )
        self.register_buffer(
            name="mfcc_dct_kernel",
            tensor=torch.from_numpy(dct_kernel_keras_w).transpose(0, 2).float(),
            persistent=PERSISTENT_BUFFERS,
        )
        self.register_buffer(
            name="stft_dct_kernel_w",
            tensor=torch.from_numpy(stft_dct_kernel_w).transpose(0, 2).float(),
            persistent=PERSISTENT_BUFFERS,
        )
        self.register_buffer(
            name="lifter", tensor=torch.from_numpy(lift)[None, :, None].float(), persistent=PERSISTENT_BUFFERS
        )

    def forward(self, x: Tensor):
        # Input is channel first: [bs,1,num_samples]
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = x.to(self.real_part_kernel.dtype)  # TODO: Remove this dirty fix for mixed-precision training!
        if self.normalize_signal == "mean_max":
            x = normalize_mean_max(x, dim=2, eps=self.eps)
        elif self.normalize_signal == "mean_std":
            x = normalize_mean_std(x, dim=2, eps=self.eps)

        if self.length < self.nfft:
            if self.convert_mode_on:
                print("CONVERTING F.pad -> torch.cat")
                x = pad_dim_zeros_convertable(x, self.nfft - self.length, dim=2, pad_end=True)
                # x = torch.cat([x,torch.zeros_like(x)[:,:,:(self.nfft - self.length)]],dim=2)
            else:
                x = F.pad(x, (0, self.nfft - self.length), "constant", 0)

        xr = torch.nn.functional.conv1d(
            x, self.real_part_kernel, bias=None, stride=self.shift, padding=0, dilation=1, groups=1
        )
        xi = torch.nn.functional.conv1d(
            x, self.imag_part_kernel, bias=None, stride=self.shift, padding=0, dilation=1, groups=1
        )
        # xi, xr size : [bs,nfft,T]

        x = torch.square(xr) + torch.square(xi)
        if self.sqrt_mag:
            # x = torch.sqrt(torch.clip(x, 1e-10, torch.finfo(x.dtype).max))
            x = torch.sqrt(x)
        else:
            x = x / self.nfft
        x = torch.clip(x, self.eps, torch.finfo(x.dtype).max)

        if self.fft_mode == "log":
            x = torch.log(x)

        # x.shape = [bs,nfft,T]
        x = x[:, : self.nfft // 2, :]  # x.shape = [bs,nfft//2,T]
        if self.cut_fft is not None:
            x = x[:, : self.cut_fft, :]
        if self.features in ["logmelbanks", "melbanks", "mfcc"]:
            if self.features == "logmelbanks":
                x = self.calc_melbanks(x, True)
            elif self.features == "melbanks":
                x = self.calc_melbanks(x, False)
            if self.features == "mfcc":
                x = self.calc_melbanks(x, True)
                x = self.calc_mfcc(x)

        # if self.normalize_vectors is not None:
        #     x = F.normalize(x,dim=1,p=self.normalize_vectors)
        # if self.power_vectors is not None:
        #     x = torch.pow(x,self.power_vectors)
        if self.normalize_spectrogram:
            x = x - torch.mean(x, dim=2, keepdims=True)
        if self.return_image:
            x = x[:, None, :, :]  # [bs,1,nfft//2,T]
        x = x + self.shift_value
        return x

    def calc_melbanks(self, stft, log=True):
        # x - STFT
        # x.shape = [bs,nfft//2,T]
        x = stft
        x = torch.nn.functional.conv1d(x, self.melbanks_kernel, bias=None, stride=1, padding=0, dilation=1, groups=1)
        x = torch.clip(x, self.eps, torch.finfo(x.dtype).max)
        if log:
            x = torch.log(x)
        return x

    def calc_mfcc(self, melbanks):
        x = melbanks
        mfccs = torch.nn.functional.conv1d(
            x, self.mfcc_dct_kernel, bias=None, stride=1, padding=0, dilation=1, groups=1
        )[:, : self.num_ceps, ...]
        return mfccs * self.lifter

    def calc_stft_cc(self, stft):
        x = stft
        return torch.nn.functional.conv1d(
            x, self.stft_dct_kernel_w, bias=None, stride=1, padding=0, dilation=1, groups=1
        )
