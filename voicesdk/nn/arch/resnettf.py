import math
import typing as tp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import windows
from torch import Tensor
from torch.nn.modules.utils import _pair

from voicesdk.nn.layers import pooling as pooling_layers


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


# --------------------------------
#     'same'-padding layers
# --------------------------------


# https://stackoverflow.com/questions/48491728/what-is-the-behavior-of-same-padding-when-stride-is-greater-than-1
def find_same_padding_v2(i, k, s):
    o = math.ceil(i / s)
    p_min = (o - 1) * s - i + k  # i.e., when the floor is removed from the previous equation
    # p_max = o * s - i + k - 1  # i.e., when the numerator of the floor % s is s-1
    return p_min


class AvgPool2dSame(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(AvgPool2dSame, self).__init__()
        self.kernel_size = _pair(kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = _pair(stride)

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor, padding=None) -> torch.Tensor:
        self.padding = (0, 0)
        ih, iw = x.size()[-2:]

        pad_h = find_same_padding_v2(i=ih, k=self.kernel_size[0], s=self.stride[0])
        pad_w = find_same_padding_v2(i=iw, k=self.kernel_size[1], s=self.stride[1])
        # pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        # pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if (pad_h > 0 or pad_w > 0) or (padding is not None):
            if padding is not None:
                x = F.pad(x, padding)
            else:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.avg_pool2d(x, self.kernel_size, stride=self.stride)


class Conv2dSame(nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor, padding=None) -> torch.Tensor:
        self.padding = (0, 0)
        ih, iw = x.size()[-2:]

        pad_h = find_same_padding_v2(i=ih, k=self.kernel_size[0], s=self.stride[0])
        pad_w = find_same_padding_v2(i=iw, k=self.kernel_size[1], s=self.stride[1])
        # pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        # pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if (pad_h > 0 or pad_w > 0) or (padding is not None):
            if padding is not None:
                x = F.pad(x, padding)
            else:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
                # print([pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SeparableConv2dSame(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tp.Tuple[int, ...],
        stride: tp.Tuple[int, ...],
        dilation: tp.Tuple[int, ...],
        bias: bool,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super(SeparableConv2dSame, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        dilation_ = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size_
        self.stride = stride_
        self.dilation = dilation_

        self.depthwise = nn.parameter.Parameter(torch.empty((in_channels, 1, *self.kernel_size), **factory_kwargs))
        self.pointwise = nn.parameter.Parameter(torch.empty((out_channels, in_channels, 1, 1), **factory_kwargs))

        self.kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_,
            stride=stride_,
            dilation=dilation_,
            bias=bias,
        )

        if bias:
            self.bias = nn.parameter.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.depthwise, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.pointwise)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor, padding=None) -> torch.Tensor:
        self.padding = (0, 0)
        ih, iw = x.size()[-2:]

        pad_h = find_same_padding_v2(i=ih, k=self.kernel_size[0], s=self.stride[0])
        pad_w = find_same_padding_v2(i=iw, k=self.kernel_size[1], s=self.stride[1])

        if (pad_h > 0 or pad_w > 0) or (padding is not None):
            if padding is not None:
                x = F.pad(x, padding)
            else:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        x = F.conv2d(
            x,
            weight=self.depthwise,
            bias=None,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.in_channels,
        )
        x = F.conv2d(x, weight=self.pointwise, bias=self.bias, stride=1, padding=0, dilation=1, groups=1)
        return x

    def extra_repr(self) -> str:
        return ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])


CONV2D = {"base": Conv2dSame, "separable": SeparableConv2dSame}

# --------------------------------
#     ResNet blocks code
# --------------------------------


class FWSEBlock(nn.Module):
    def __init__(self, num_features: int = 128, num_out_features: int = 80, activation=nn.ReLU(inplace=True)):
        super(FWSEBlock, self).__init__()
        self.num_features = num_features
        self.num_out_features = num_out_features
        self.squeeze = nn.Linear(num_out_features, num_features, bias=True)
        self.activation = activation
        self.exitation = nn.Linear(num_features, num_out_features, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [B, C, F, T]
        # Squeeze along (C, T):
        x = inputs.mean(dim=[1, 3])  # shape: [B, F]
        x = self.squeeze(x)
        x = self.activation(x)
        x = self.exitation(x)
        x = torch.sigmoid(x)

        # Expand back for broadcast
        x = x.unsqueeze(1).unsqueeze(3)  # [B, 1, F, 1]
        x = inputs * x
        return x


class ResBlockTF(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_out: int = 64,
        strides: int = 1,
        strided_skip_kernel_size: int = 2,
        conv_type: str = "base",
        se_num_features: int = None,
        se_num_out_features: int = None,
    ):
        super(ResBlockTF, self).__init__()
        self.se_num_features = se_num_features
        self.channel_out = channel_out
        self.strides = strides
        assert conv_type in CONV2D
        Conv = CONV2D[conv_type]

        self.activation = nn.ReLU(inplace=True)

        self.conv1 = Conv(
            in_channels=channel_in,
            out_channels=channel_out,
            kernel_size=3,
            stride=strides,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(channel_out, eps=1e-3)

        self.conv2 = Conv(
            in_channels=channel_out, out_channels=channel_out, kernel_size=3, stride=1, dilation=1, groups=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-3)

        # For the skip
        self.conv_transform = Conv(
            in_channels=channel_in,
            out_channels=channel_out,
            kernel_size=strided_skip_kernel_size if strides > 1 else strides,
            stride=strides,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.bn_transform = nn.BatchNorm2d(channel_out, eps=1e-3)

        if self.se_num_features is not None:
            # print(f"Creating FWSE block : ({se_num_features},{se_num_out_features})")
            self.se_block = FWSEBlock(
                num_features=se_num_features,
                num_out_features=se_num_out_features,
            )

    def forward(self, inputs):
        # first conv
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation(x)

        # second conv
        x = self.conv2(x)
        x = self.bn2(x)

        # optional Squeeze-Excitation
        if self.se_num_features is not None:
            x = self.se_block(x)

        # skip if shapes differ
        apply_skip = (
            (inputs.size()[1] != x.size()[1])
            or (inputs.size()[-1] != x.size()[-1])
            or (inputs.size()[-2] != x.size()[-2])
        )
        if apply_skip:
            inputs = self.conv_transform(inputs)
            inputs = self.bn_transform(inputs)

        # add & final activation
        x = inputs + x
        x = self.activation(x)
        return x


# --------------------------------
#     ResNetTF backbone code
# --------------------------------


class ResNetTFBackbone(nn.Module):
    def __init__(
        self,
        init_conv_params=dict(
            in_channels=1,
            out_channels=128,
            stride=1,
            kernel_size=3,
            padding=1,  # For initial conv, direct param to nn.Conv2d
        ),
        block_setup=[
            # (filters, num_blocks, strides)
            (128, 3, 1),
            (128, 4, 2),
            (256, 6, 2),
            (256, 3, 2),
        ],
        strided_skip_kernel_size: int = 2,
        num_nodes_last_layer: int = None,
        conv_type: str = "base",
        return_all_outputs: bool = False,
    ):
        super(ResNetTFBackbone, self).__init__()

        # initial conv/bn/relu
        self.conv0 = nn.Conv2d(
            in_channels=init_conv_params["in_channels"],
            out_channels=init_conv_params["out_channels"],
            kernel_size=init_conv_params["kernel_size"],
            stride=init_conv_params["stride"],
            padding=init_conv_params["padding"],
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(init_conv_params["out_channels"], eps=1e-3)

        self.inplanes = init_conv_params["out_channels"]
        self.return_all_outputs = return_all_outputs
        self.strided_skip_kernel_size = strided_skip_kernel_size
        self.conv_type = conv_type
        self.num_stages = len(block_setup)

        # Build each stage
        for stage_ind, stage_setup in enumerate(block_setup):
            if len(stage_setup) == 3:
                num_filters, num_blocks, strides = stage_setup
                setattr(self, f"stage{stage_ind}", self._make_stage(ResBlockTF, num_filters, num_blocks, strides))
            elif len(stage_setup) == 5:
                num_filters, num_blocks, strides, se_num_features, se_num_out_features = stage_setup
                setattr(
                    self,
                    f"stage{stage_ind}",
                    self._make_stage(
                        ResBlockTF, num_filters, num_blocks, strides, se_num_features, se_num_out_features
                    ),
                )

        # optional last 1x1 conv
        if num_nodes_last_layer is not None:
            self.conv_last = nn.Conv2d(
                in_channels=self.inplanes,
                out_channels=num_nodes_last_layer,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.bn_last = nn.BatchNorm2d(num_nodes_last_layer, eps=1e-3)
            self.act_last = self.relu
        else:
            self.conv_last = nn.Identity()
            self.bn_last = nn.Identity()
            self.act_last = nn.Identity()

    def _make_stage(
        self, Block, planes, blocks, strides=1, se_num_features: int = None, se_num_out_features: int = None
    ):
        layers = [
            Block(
                self.inplanes,
                planes,
                strides,
                strided_skip_kernel_size=self.strided_skip_kernel_size,
                conv_type=self.conv_type,
                se_num_features=se_num_features,
                se_num_out_features=se_num_out_features,
            )
        ]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                Block(
                    self.inplanes,
                    planes,
                    strides=1,
                    strided_skip_kernel_size=self.strided_skip_kernel_size,
                    conv_type=self.conv_type,
                    se_num_features=se_num_features,
                    se_num_out_features=se_num_out_features,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []

        # initial layers
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        outputs.append([x])

        # stages
        for stage_ind in range(self.num_stages):
            stage_outputs = []
            stage = getattr(self, f"stage{stage_ind}")
            for block in stage:
                x = block(x)
                stage_outputs.append(x)
            outputs.append(stage_outputs)

        # last conv
        x = self.act_last(self.bn_last(self.conv_last(x)))
        outputs.append([x])

        if self.return_all_outputs:
            return outputs
        else:
            return x


class StackFreqDim(nn.Module):
    def __init__(self, permute_chan_freq=False):
        super(StackFreqDim, self).__init__()
        self.permute_chan_freq = permute_chan_freq
        self.convert_mode_on = False

    def forward(self, x):
        s = x.size()
        if self.convert_mode_on:
            print(f"RESHAPE -> SPLIT+CONCAT")
            x = torch.cat(torch.split(x, 1, dim=1), dim=2)[:, 0, :, :]
        else:
            if self.permute_chan_freq:
                x = x.permute((0, 2, 1, 3))
            x = torch.reshape(x, (int(s[0]), int(s[1] * s[2]), int(s[3])))
        return x


# --------------------------------
# Ported from tf pooling layers
# --------------------------------


class StatsPoolingTF(nn.Module):
    def __init__(self, mode="var"):
        super(StatsPoolingTF, self).__init__()
        assert mode in ["var"]
        self.mode = mode

    def forward(self, x):
        mean = x.mean(dim=2, keepdims=True)
        var = torch.mean(torch.square(x - mean), dim=2)
        return torch.cat([mean[:, :, 0], var], dim=1)


class SelfAttentionTF(nn.Module):
    """
    Self-attention pooling. Returns concatenated mean and std statistics along the time axis with self-attention.
    This is a PyTorch implementation closely matching the given TensorFlow code.
    """

    def __init__(
        self,
        add_statistics: bool = True,
        weight_l2_regularizer: float = 1e-3,
        num_features: int = 2560,
        reduction_ratio: int = 4,
        reduction_mode: str = "sum",
        add_context: bool = False,
        var2std_eps: float = 1e-12,
        pooling_dtype: str = "float32",
    ):
        super(SelfAttentionTF, self).__init__()
        assert reduction_mode in ["mean", "sum"], "Only 'mean' and 'sum' reduction modes are supported."

        self.add_statistics = add_statistics
        self.weight_l2_regularizer = weight_l2_regularizer
        self.num_features = num_features
        self.reduction_ratio = reduction_ratio
        self.add_context = add_context
        self.var2std_eps = var2std_eps
        self.pooling_dtype = pooling_dtype

        # Set reduction function
        self.reduction_mode = reduction_mode

        # compression_conv (conv1)
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features // reduction_ratio,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # bn_pooling (bn1)
        # TF's momentum=0.99 ~ PyTorch momentum=0.01
        self.bn1 = nn.BatchNorm1d(num_features // reduction_ratio, momentum=0.01, eps=0.001)

        # expansion_conv (conv2)
        self.conv2 = nn.Conv1d(
            in_channels=num_features // reduction_ratio,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, inputs):
        original_x = inputs
        # handle add_context
        if self.add_context:
            mean = original_x.mean(dim=2, keepdim=True)
            var = (original_x - mean).pow(2).mean(dim=2, keepdim=True)
            var = torch.max(var, torch.tensor(self.var2std_eps, dtype=original_x.dtype, device=original_x.device))
            std = torch.sqrt(var)
            inputs_for_stats = torch.cat((original_x, mean, std), dim=2)  # [B, F, T+2]
        else:
            inputs_for_stats = original_x

        x = self.conv1(inputs_for_stats)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)

        weights = F.softmax(x, dim=2)  # [B, F, T or T+2 depending on context]
        # print(weights)
        if self.add_statistics:
            if self.reduction_mode == "mean":
                mu = (inputs_for_stats * weights).mean(dim=2)  # [B, F]
                sg = ((inputs_for_stats**2) * weights).mean(dim=2) - mu**2
            else:  # 'sum'
                # mu = (inputs_for_stats * weights).sum(dim=2) / sum_w.squeeze(-1)
                # sg = ((inputs_for_stats ** 2) * weights).sum(dim=2) / sum_w.squeeze(-1) - mu**2
                mu = (inputs_for_stats * weights).sum(dim=2)
                sg = ((inputs_for_stats**2) * weights).sum(dim=2) - mu**2

            sg = torch.clamp(sg, min=1e-6, max=None)
            sg = torch.sqrt(sg)
            output = torch.cat([mu, sg], dim=1)  # [B, 2*F]
        else:
            if self.reduction_mode == "mean":
                output = (inputs_for_stats * weights).mean(dim=2)
            else:
                output = (inputs_for_stats * weights).sum(dim=2)

        return output


class ResNetTF(nn.Module):
    def __init__(
        self,
        features_cfg={
            "normalize_signal": True,
            "frame_length": 400,
            "frame_step": 160,
            "fft_length": 512,
            "sample_rate": 16000,
            "window": "hann",
            "eps": 1e-06,
            "trainable": False,
            "mode": "logmelbanks",
            "low_freq": 20,
            "high_freq": 7600,
            "num_bins": 96,
            "num_ceps": 30,
            "fft_mode": "abs",
            "shift_value": 0.0,
            "normalize_spectrogram": True,
            "return_img": True,
        },
        backbone_cfg={
            "init_conv_params": {"in_channels": 1, "out_channels": 128, "stride": 1, "kernel_size": 3, "padding": 1},
            "conv_type": "base",
            "num_nodes_last_layer": None,
            "block_setup": [(128, 6, 1, 128, 96), (128, 16, 2, 128, 48), (256, 24, 2, 128, 24), (256, 3, 2, 128, 12)],
        },
        pooling_type="SelfAttentionTF",
        pooling_cfg={
            "num_features": 3072,
            "add_statistics": True,
            "reduction_ratio": 12,
            "reduction_mode": "sum",
            "add_context": True,
            "var2std_eps": 1e-12,
        },
        post_pool_dims=3072 * 2,
        embedding_size=256,
        bn_eps=1e-3,
        use_feats=True
    ):
        super(ResNetTF, self).__init__()
        if use_feats:
            self.features = SpectralFeaturesTF(**features_cfg)
        else:
            self.features = None
        self.backbone = ResNetTFBackbone(**backbone_cfg)
        self.pre_pool = StackFreqDim(permute_chan_freq=True)
        self.pooling = eval(pooling_type)(**pooling_cfg)
        self.head = nn.Sequential(nn.Linear(post_pool_dims, embedding_size), nn.BatchNorm1d(embedding_size, eps=bn_eps))

    def forward(self, x):
        if self.features is not None:
            h = self.features(x)
        else:
            h = x
        h = self.backbone(h)
        h = self.pre_pool(h)
        h = self.pooling(h)
        emb = self.head(h)
        return emb


# --------------------------------
#        Subnets stuff
# --------------------------------


def make_conv_1x1(in_planes, out_planes):
    return nn.Sequential(
        *[
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        ]
    )


def make_conv_3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        *[
            AvgPool2dSame(stride),
            Conv2dSame(in_planes, out_planes, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        ]
    )


def make_sep_conv_3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        *[
            AvgPool2dSame(stride),
            SeparableConv2dSame(in_planes, out_planes, dilation=1, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        ]
    )


def make_bayes_conv_3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        *[
            BayesConv2d(in_planes, out_planes, padding_mode="same", kernel_size=3, stride=stride, bias=True),
            BayesBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        ]
    )


def make_block_3x3(in_planes, out_planes, stride=1):
    return ResBlockTF(channel_in=in_planes, channel_out=out_planes, strides=stride)


def make_linear(in_planes, out_planes):
    return nn.Sequential(*[nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True)])


class LinearWeigthedReduction(nn.Module):
    def __init__(self, num: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((num,)))

    def forward(self, xs):
        # print([x.size() for x in xs])
        # print(len(xs))
        assert len(xs) == self.weight.size(0)
        w = self.weight
        w = F.softmax(w, dim=0)
        x = torch.cat([x.unsqueeze(1) for x in xs], dim=1)
        w = w.unsqueeze(0)
        for _ in range(2, x.ndim):
            w = w.unsqueeze(-1)
        return (x * w).sum(dim=1)


class LastBlockReduction(nn.Module):
    def __init__(self, num: int = None):
        super().__init__()

    def forward(self, xs):
        return xs[-1]


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def resolve_module_conf(module_config):
    if isinstance(module_config, dict):
        module = eval(module_config["type"])(*module_config.get("args", []), **module_config.get("kwargs", {}))
        trainable = module_config.get("trainable", True)
        if trainable is not None:
            for param in module.parameters():
                param.requires_grad = trainable
        return module
    else:
        raise NotImplemented()


class SequentialModel(nn.Module):
    def __init__(self, submodules=[]):
        super(SequentialModel, self).__init__()
        print(f"submodules : {submodules}")
        self.submodules = nn.Sequential(*[resolve_module_conf(sm) for sm in submodules])

    def forward(self, x):
        return self.submodules(x)


class ResNetTFSubNetS(nn.Module):
    def __init__(
        self,
        backbone_exp_dir_ep: tuple | None = None,
        backbone: nn.Module | None = None,
        no_grad_backbone: bool = True,
        return_embedding: bool = False,
        subnets_configs: dict = {
            "vcd": dict(
                # run_num_times
                downsample_ratio=2,
                type_3x3="conv",
                head_config=None,
            )
        },
    ):
        super(ResNetTFSubNetS, self).__init__()
        self.no_grad_backbone = no_grad_backbone

        if backbone_exp_dir_ep:
            from wespeaker.utils.utils import load_experiment_or_asset_model
            self.backbone_model = freeze_model(load_experiment_or_asset_model(*backbone_exp_dir_ep).eval())
        else:
            assert backbone is not None
            self.backbone_model = backbone

        self.backbone_model.backbone.return_all_outputs = True

        bb = self.backbone_model.backbone
        convs = [bb.conv0] + [getattr(bb, f"stage{stage_ind}")[0].conv1 for stage_ind in range(bb.num_stages)]
        strides = [conv.stride for conv in convs[2:]] + [1]
        out_channels = [conv.out_channels for conv in convs]

        # Build subnetwork for each config:
        subnets = {}
        for subnet_name, subnet_config in subnets_configs.items():
            subnets[subnet_name] = self.build_subnet(strides=strides, out_channels=out_channels, **subnet_config)
        self.subnets = nn.ModuleDict(subnets)
        self.return_embedding = return_embedding

    def build_subnet(
        self,
        strides: tp.List[int],
        out_channels: tp.List[int],
        downsample_ratio: int = 4,
        type_3x3: str = "conv",
        pre_head_config: dict = None,
        head_config: dict = None,
        reduce_blocks: str = "last",
        bn_kwargs: dict = None,
        **kwargs,
    ):
        assert reduce_blocks in ["last", "linear_weighted"]

        downsampled_channels = [oc // downsample_ratio for oc in out_channels]
        out_channels_sampled = out_channels
        subnet_1x1_layers = nn.ModuleList(
            [make_conv_1x1(oc, dc) for oc, dc in zip(out_channels_sampled, downsampled_channels)]
        )
        cat_channels = [
            prev_chan + next_chan for prev_chan, next_chan in zip(downsampled_channels[:-1], downsampled_channels[1:])
        ]
        make_3x3 = {
            "conv": make_conv_3x3,
            "sep_conv": make_sep_conv_3x3,
            "block": make_block_3x3,
            "bayes_conv": make_bayes_conv_3x3,
        }[type_3x3]
        print(f"{type_3x3}, {make_3x3}")
        subnet_3x3_layers = nn.ModuleList(
            [make_3x3(cc, nc, s) for cc, nc, s in zip(cat_channels, downsampled_channels[1:], strides)]
        )

        backbone = self.backbone_model.backbone
        blocks_per_stage = [len(getattr(backbone, f"stage{sind}")) for sind in range(backbone.num_stages)]
        BlockReduction = {"last": LastBlockReduction, "linear_weighted": LinearWeigthedReduction}[reduce_blocks]
        stage_blocks_reducers = nn.ModuleList(
            [LastBlockReduction()] + [BlockReduction(num_blocks) for num_blocks in blocks_per_stage]
        )

        # Custom head on top of subnetwork output
        if head_config is not None:
            head = resolve_module_conf(head_config)
        else:
            head = nn.Identity()

        if pre_head_config is not None:
            pre_head = resolve_module_conf(pre_head_config)
        else:
            pre_head = nn.Identity()

        subnet = nn.ModuleDict(
            dict(
                head=head,
                pre_head=pre_head,
                subnet_1x1_layers=subnet_1x1_layers,
                subnet_3x3_layers=subnet_3x3_layers,
                stage_blocks_reducers=stage_blocks_reducers,
            )
        )
        if bn_kwargs is not None:
            set_batchnorms_momentum(subnet, **bn_kwargs)

        return subnet

    def backbone_forward(self, x):
        # h = self.features(x)
        # h = self.backbone(h)
        # h = self.pre_pool(h)
        # h = self.pooling(h)
        # emb = self.head(h)
        x = self.backbone_model.features(x)
        all_outputs = self.backbone_model.backbone(x)
        x = self.backbone_model.pre_pool(all_outputs[-1][-1])
        x = self.backbone_model.pooling(x)
        x = self.backbone_model.head(x)
        embeddings = x
        return embeddings, all_outputs

    def forward_subnet(
        self,
        subnet: nn.Module,
        embeddings: torch.Tensor,
        all_outputs: tp.List[tp.List[torch.Tensor]],
        y: torch.Tensor = None,
    ):
        downsampled_outputs = []
        for stage_ind in range(self.backbone_model.backbone.num_stages):
            stage_outputs = all_outputs[stage_ind]
            output = subnet["stage_blocks_reducers"][stage_ind](stage_outputs)
            downsampled_outputs.append(subnet["subnet_1x1_layers"][stage_ind](output))

        x = torch.cat([downsampled_outputs[0], downsampled_outputs[1]], dim=1)
        x = subnet["subnet_3x3_layers"][0](x)
        for ds_ind in range(2, len(downsampled_outputs)):
            x = torch.cat([x, downsampled_outputs[ds_ind]], dim=1)
            x = subnet["subnet_3x3_layers"][ds_ind - 1](x)
        x = subnet["pre_head"](x)
        if y is not None:
            return subnet["head"](x, y)
        else:
            return subnet["head"](x)

    def backbone_forward_wrapped(self, x):
        if self.no_grad_backbone:
            self.backbone_model.eval()
            with torch.no_grad():
                embeddings, all_outputs = self.backbone_forward(x)
        else:
            embeddings, all_outputs = self.backbone_forward(x)
        return embeddings, all_outputs

    def forward(self, x, y=None):
        # print(f"forward_base")
        embeddings, all_outputs = self.backbone_forward_wrapped(x)

        outputs = {}
        for subnet_name, subnet in self.subnets.items():
            outputs[subnet_name] = self.forward_subnet(subnet, embeddings, all_outputs, y)

        if self.return_embedding:
            outputs["embedding"] = embeddings

        if len(outputs) == 1:
            return outputs[list(outputs.keys())[0]]
        else:
            return outputs
