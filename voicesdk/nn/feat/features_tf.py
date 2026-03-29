import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import windows

from .preproc import NormalizeAudio, PreEmphasis


def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.)


def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filterbanks(low_freq: int = 20,
                    high_freq: int = 7600,
                    nfilt: int = 80,
                    nfft: int = 512,
                    samplerate: int = 16000):
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

    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel)

    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.vstack([np.zeros((1, nfilt)), mel_weights_matrix])[:, :].astype('float32')


class SpectralFeaturesTF(nn.Module):
    def __init__(self,
                 frame_length: int = 400,
                 frame_step: int = 160,
                 fft_length: int = 512,
                 sample_rate: int = 16000,
                 window: str = 'hann',
                 normalize_spectrogram: bool = False,
                 normalize_signal: bool = False,
                 eps: float = 1e-8,
                 mode: str = 'melbanks',
                 low_freq: int = 20,
                 high_freq: int = 7600,
                 num_bins: int = 80,
                 log_mels: bool = True,
                 fft_mode: str = 'abs',
                 sqrt_real_imag: bool = False,
                 return_img: bool = False,
                 **kwargs):
        """
        Requirements
        ------------
        input shape must meet the conditions: mod((input.shape[0] - length), shift) == 0
        fft_length >= frame_length

        Parameters
        ------------
        :param frame_length: Length of each segment in # of samples
        :param frame_step: Shift between segments in # of samples
        :param fft_length: number of dft points, if None => fft_length == frame_length
        :param fft_mode: "abs" - amplitude spectrum; "real" - only real part, "imag" - only imag part,
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
        super().__init__()

        assert mode in ['fft', 'melbanks', 'mfcc', 'complex']
        assert isinstance(frame_length, int) and isinstance(frame_step, int) and isinstance(fft_length, int)

        self.length = frame_length
        self.shift = frame_step
        self.sqrt_real_imag = sqrt_real_imag
        self.normalize_spectrogram = normalize_spectrogram
        self.normalize_signal = normalize_signal
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

        self.return_img = return_img

        if mode in ['melbanks', 'mfcc']:
            fft_mode = 'abs'
        self.fft_mode = fft_mode
        self.log_mels = log_mels
        self.build()

    def build(self):
        assert self.nfft >= self.length

        if self.window:
            if self.window == 'hamming':
                self.window = windows.hamming(self.length)
            elif self.window in ['hann', 'hanning']:
                self.window = np.array([0.5 - 0.5 * (np.cos((2 * np.pi * l) / (self.length - 1) )) \
                                        for l in range(self.length)])
            elif self.window == 'sqrt_hann':
                self.window = np.array([0.5 - 0.5 * (np.cos((2 * np.pi * l) / (self.length - 1) )) \
                                        for l in range(self.length)]) ** 0.5
            elif self.window == 'kaiser':
                self.window = windows.kaiser(self.length)
            else:
                self.window = np.ones(self.length)
        self.window = self.window.astype("float32")

        # real kernel
        real_kernel = np.asarray([np.cos(2 * np.pi * np.arange(0, self.nfft) * n / self.nfft)
                                        for n in range(self.nfft)]).astype("float32").T
        self.real_kernel = real_kernel[:self.length, :self.nfft // 2]
        if self.window is not None:
            self.real_kernel *= self.window[:, None]
        self.real_kernel = self.real_kernel[:,None,:]

        # imag kernel
        image_kernel = np.asarray([np.sin(2 * np.pi * np.arange(0, self.nfft) * n / self.nfft)
                                        for n in range(self.nfft)]).astype("float32").T
        self.image_kernel = image_kernel[:self.length, :self.nfft // 2]
        if self.window is not None:
            self.image_kernel *= self.window[:, None]
        self.image_kernel = self.image_kernel[:,None,:]

        self.register_buffer('real_kernel_pt', 
                             torch.from_numpy(self.real_kernel).permute(2,1,0).float())
        self.register_buffer('image_kernel_pt', 
                             torch.from_numpy(self.image_kernel).permute(2,1,0).float())

        if self.features in ['melbanks']:
            linear_to_mel_weight_matrix = get_filterbanks(
                                            nfilt=self.num_bins,
                                            nfft=self.nfft // 2,
                                            samplerate=self.samplerate,
                                            low_freq=self.low_freq,
                                            high_freq=self.high_freq)
            linear_to_mel_weight_matrix = linear_to_mel_weight_matrix[:,:,None]
            self.register_buffer('melbanks_pt', 
                                 torch.from_numpy(linear_to_mel_weight_matrix).permute(1,0,2).float())

    def forward(self, inputs):
        # inputs.size() : (bs,1,T)
        dtype = inputs.dtype
        inputs = inputs.float()
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)

        if self.normalize_signal:
            inputs = (inputs - inputs.mean(dim=2, keepdims=True)) /\
                     (inputs.std(dim=2, keepdims=True, unbiased=False) + self.eps)

        real_part = F.conv1d(inputs, self.real_kernel_pt, stride=self.shift, padding=self.shift//2)
        imag_part = F.conv1d(inputs, self.image_kernel_pt, stride=self.shift, padding=self.shift//2)

        if self.features == 'complex':
            return [real_part, imag_part]

        fft = torch.square(real_part) + torch.square(imag_part)
        if self.sqrt_real_imag:
            fft = torch.sqrt(fft)

        feat = fft.clip(self.eps, 1/self.eps)

        if self.fft_mode == 'log':
            feat = torch.log(feat)

        if self.features in ['melbanks']:
            mel_spectrograms = F.conv1d(feat, self.melbanks_pt, stride=1, padding=0)
            mel_spectrograms = mel_spectrograms.clip(self.eps, 1/self.eps)
            if self.log_mels:
                feat = torch.log(mel_spectrograms)
            else:
                feat = mel_spectrograms

        if self.normalize_spectrogram:
            feat = (feat - feat.mean(dim=(1, 2), keepdims=True)) /\
                    (feat.std(dim=(1, 2), keepdims=True, unbiased=False) + self.eps)
        if self.return_img:
            feat = feat[:,None,:,:]
        return feat.to(dtype)


class LogSpec(nn.Module):
    def __init__(self,eps:float=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self,x):
        return x.clip(self.eps,1e+8).log()


class FbankAug(nn.Module):
    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10), freq_start_bin=0):
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
        mask_pos = torch.randint(self.freq_start_bin, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
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


class TFMelBanks(nn.Module):
    def __init__(self, 
        sample_rate=16000, 
        n_fft=512, 
        win_length=400, 
        hop_length=160,
        f_min = 20, 
        f_max = 7600, 
        n_mels = 80, 
        do_spec_aug=False,
        norm_signal=False,
        do_preemph=True,
        freq_start_bin = 0,
        freq_mask_width = (0, 8), 
        time_mask_width = (0, 10),
        eps = 1e-8
    ):
        super(TFMelBanks, self).__init__()
        self.torchfbank = torch.nn.Sequential(
            NormalizeAudio(eps, squeeze=True) if norm_signal else nn.Identity(),
            PreEmphasis() if do_preemph else nn.Identity(),

            SpectralFeaturesTF(
                frame_length = win_length,
                frame_step = hop_length,
                fft_length = n_fft,
                sample_rate = sample_rate,
                window = 'hamming',
                normalize_spectrogram = False,
                normalize_signal = False,
                eps = eps,
                mode = 'melbanks',
                low_freq = f_min,
                high_freq = f_max,
                num_bins = n_mels,
                log_mels = False,
                fft_mode = 'abs',
                sqrt_real_imag = False,
                return_img = False,
            )
        )
        self.eps = eps
        if do_spec_aug:
            self.specaug = FbankAug(
                freq_start_bin=freq_start_bin,
                freq_mask_width=freq_mask_width,
                time_mask_width=time_mask_width) # Spec augmentation
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                x = self.torchfbank(x)+self.eps
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if self.training:
                    x = self.specaug(x)
        return x.to(xdtype)


class SlidingInstanceNormAudio(nn.Module):
    """
    Approximates an 'instance norm' in the time domain using a sliding/local
    window per channel. If input is [B, C, T], it normalizes within each
    channel's local window across time, using avg_pool1d with stride=1
    and reflective padding.

    - window_size: number of frames in the sliding window (forced to be odd).
    - eps: small constant to avoid dividing by zero.

    For an odd window_size K, we do:
      1) Reflect-pad (K-1)//2 samples on each side.
      2) avg_pool1d(..., kernel_size=K, stride=1, padding=0).
    This yields an output of length T, the same as the unpadded input,
    but with more natural boundary handling than zero-padding.
    """

    def __init__(self, window_size=400, eps=1e-5, scale_var=True, lookahead_ratio=0.5, pad_mode='reflect'):
        super().__init__()
        # Force window_size to be odd
        if window_size % 2 == 0:
            window_size += 1
        self.pad_mode = pad_mode
        self.scale_var = scale_var
        self.window_size = window_size
        self.lookahead_ratio = lookahead_ratio
        self.eps = eps

    def extra_repr(self) -> str:
        return f"window_size={self.window_size}, eps={self.eps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape [B, T] or [B, C, T]
           If 2D, we interpret as batch_size x time (1 channel).
        Returns: x_norm with shape [B, C, T].
        """
        # If input is [B, T], treat as [B, 1, T]
        squeeze_back=False
        if x.ndim == 2:
            squeeze_back=True
            x = x.unsqueeze(1)  # => [B, 1, T]

        B, C, T = x.shape

        # We'll do a 'sliding' average over time with stride=1,
        # but first apply reflective padding manually.
        pad_tot = (self.window_size - 1)
        pad_right = int(pad_tot * self.lookahead_ratio)
        pad_left = pad_tot - pad_right


        # Reflectively pad both sides of the time dimension
        # PyTorch's F.pad expects the pad tuple in (left, right) order for 1D.
        # We'll flatten B,C in a single dimension so we can reflect-pad over time.
        x_padded = F.pad(x, (pad_left, pad_right), mode=self.pad_mode)

        # local_mean => [B, C, T]
        local_mean = F.avg_pool1d(
            x_padded,
            kernel_size=self.window_size,
            stride=1,
            padding=0,
            count_include_pad=False
        )

        # print(f"local_mean : {local_mean.size()}")
        x_norm = (x - local_mean)
        if self.scale_var:
            # local_mean_sq => [B, C, T]
            x_sq = x * x
            x_sq_padded = F.pad(x_sq, (pad_left, pad_right), mode=self.pad_mode)
            local_mean_sq = F.avg_pool1d(
                x_sq_padded,
                kernel_size=self.window_size,
                stride=1,
                padding=0,
                count_include_pad=False
            )

            local_var = local_mean_sq - local_mean * local_mean
            local_std = torch.sqrt(torch.clamp(local_var, min=0.) + self.eps)
            # print(f"local_var : {local_std.size()}")
            x_norm = x_norm / local_std
        if squeeze_back:
            x_norm = x_norm.squeeze(1)
        return x_norm


class ChunkedInstanceNormAudio(nn.Module):
    """
    Normalizes an audio signal in non-overlapping chunks.

    This layer splits the time dimension into chunks (using non-overlapping avg_pool1d),
    computes the per-chunk mean and variance, and then interpolates the computed
    statistics back to the original temporal resolution. It then normalizes the input
    by subtracting the chunked mean and (optionally) dividing by the chunked standard deviation.

    Parameters:
      - chunk_size: Number of frames in each chunk (must be >=1).
      - eps: Small constant to avoid dividing by zero.
      - scale_var: Whether to scale by the local standard deviation.
      - lookahead_ratio: Ratio used to determine how many frames to pad on the right
        relative to the total pad amount (chunk_size - 1).
      - pad_mode: Padding mode to use (e.g. 'reflect').
  
    Example:
      >>> signal = torch.arange(20)[None, None, :].float()
      >>> norm_layer = ChunkedInstanceNormAudio(chunk_size=10, lookahead_ratio=0.2, eps=1e-10)
      >>> norm_signal = norm_layer(signal)
    """
    def __init__(self, chunk_size=8000, eps=1e-10, scale_var=True, lookahead_ratio=0.2, pad_mode='reflect'):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self.chunk_size = chunk_size
        self.eps = eps
        self.scale_var = scale_var
        self.lookahead_ratio = lookahead_ratio
        self.pad_mode = pad_mode

    def extra_repr(self) -> str:
        return ", ".join([
            f"chunk_size={self.chunk_size}",
            f"eps={self.eps}",
            f"scale_var={self.scale_var}",
            f"lookahead_ratio={self.lookahead_ratio}"
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, T] or [B, C, T].
           If 2D, it is interpreted as [B, 1, T].
   
        Returns:
           Normalized tensor of shape [B, C, T].
        """
        squeeze_back = False
        if x.ndim == 2:
            squeeze_back = True
            x = x.unsqueeze(1)  # Convert to shape [B, 1, T]

        B, C, T = x.shape

        # Determine padding amounts based on chunk_size and lookahead_ratio.
        pad_tot = self.chunk_size - 1
        pad_right = int(pad_tot * self.lookahead_ratio)
        pad_left = pad_tot - pad_right

        # Reflectively pad the time dimension.
        # F.pad expects padding in the form (pad_left, pad_right) for 1D.
        x_p = F.pad(x, (pad_left, pad_right), mode=self.pad_mode)

        # Compute chunked mean and mean of squared values over non-overlapping chunks.
        chunked_mean = F.avg_pool1d(x_p, kernel_size=self.chunk_size, stride=self.chunk_size,
                                    padding=0, ceil_mode=False, count_include_pad=True)
        chunked_mean_sq = F.avg_pool1d(x_p**2, kernel_size=self.chunk_size, stride=self.chunk_size,
                                       padding=0, ceil_mode=False, count_include_pad=True)

        # Interpolate the computed statistics to the original temporal length T using nearest neighbor.
        chunked_mean = F.interpolate(chunked_mean, size=T, mode='nearest')
        chunked_mean_sq = F.interpolate(chunked_mean_sq, size=T, mode='nearest')

        # Compute variance and standard deviation.
        chunked_var = chunked_mean_sq - chunked_mean * chunked_mean
        chunked_std = torch.sqrt(torch.clamp(chunked_var, min=0.) + self.eps)

        # Normalize: subtract mean and (optionally) scale by std.
        x_norm = x - chunked_mean
        if self.scale_var:
            x_norm = x_norm / chunked_std

        if squeeze_back:
            x_norm = x_norm.squeeze(1)
        return x_norm


class GlobalNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, update_steps=None, sync=True):
        """
        Custom BatchNorm1d module that accumulates statistics over multiple steps.

        Args:
            num_features (int): Number of features or channels.
            eps (float): A small value added to the denominator for numerical stability.
            update_steps (int, optional): The number of steps over which to accumulate statistics.
                                          If None, update indefinitely.
        """
        super(GlobalNorm1d, self).__init__()
        self.update_steps = update_steps
        self.sync = sync
        self.eps = eps

        # Running mean and variance are not learnable parameters
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.register_buffer('weight', torch.tensor(0.0))
        # Step counter to track updates.
        self.step_count = 0

    def forward(self, x):
        # Determine the dimensions over which to compute statistics:
        # for 2D inputs ([N, C]), we average over dimension 0 (the batch).
        # for 3D inputs ([N, C, L]), we average over dimensions 0 and 2.
        if x.dim() == 2:
            dims = (0,)
            view_shape = (1, -1)
        elif x.dim() == 3:
            dims = (0, 2)
            view_shape = (1, -1, 1)
        else:
            raise ValueError("Input dimension not supported: expected 2D or 3D input.")

        batch_mean = x.mean(dim=dims, keepdim=True)
        batch_var = x.var(dim=dims, unbiased=False, keepdim=True)

        # Update running statistics if training and within update_steps.
        if self.training and (self.update_steps is None or self.step_count < self.update_steps):
            # Compute the number of elements used for the statistics
            weight = 1
            for d in dims:
                weight *= x.size(d)

            # Optionally synchronize the current batch statistics across GPUs.
            if self.sync and dist.is_initialized():
                # Aggregate weight.
                print(f"Synced GlobalNorm1d")
                local_weight = torch.tensor([weight], device=x.device, dtype=self.running_mean.dtype)
                dist.all_reduce(local_weight, op=dist.ReduceOp.SUM)
                global_weight = local_weight.item()

                # Aggregate batch mean: scale local mean by local weight, then sum over GPUs.
                batch_mean_local = batch_mean.view(-1) * weight
                batch_var_local = batch_var.view(-1) * weight
                dist.all_reduce(batch_mean_local, op=dist.ReduceOp.SUM)
                dist.all_reduce(batch_var_local, op=dist.ReduceOp.SUM)
                effective_batch_mean = batch_mean_local / global_weight
                effective_batch_var = batch_var_local / global_weight
                effective_weight = global_weight
            else:
                effective_batch_mean = batch_mean.view(-1)
                effective_batch_var = batch_var.view(-1)
                effective_weight = weight

            new_weight = self.weight + effective_weight

            # Update running mean and variance using the accumulated weighted average.
            self.running_mean.data = (
                self.running_mean * self.weight + effective_batch_mean * effective_weight
            ) / new_weight
            self.running_var.data = (
                self.running_var * self.weight + effective_batch_var * effective_weight
            ) / new_weight
            self.weight.data = new_weight
            self.step_count += 1

        mean = self.running_mean.view(*view_shape)
        var = self.running_var.view(*view_shape)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat


class GlobalNormSignal(nn.Module):
    def __init__(self, eps=1e-5, update_steps=None, sync=True):
        super(GlobalNormSignal, self).__init__()
        self.norm = GlobalNorm1d(1, eps, update_steps, sync)

    def forward(self, x):
        squeeze_back=False
        if x.ndim == 2:
            squeeze_back=True
            x = x.unsqueeze(1)  # => [B, 1, T]
        x = self.norm(x)
        if squeeze_back:
            x = x.squeeze(1)
        return x


class TFMelBanksV2(nn.Module):
    def __init__(self, 
        sample_rate=16000, 
        n_fft=512, 
        win_length=400, 
        hop_length=160,
        norm_win_len = 8000,
        f_min = 20, 
        f_max = 7600, 
        n_mels = 80, 
        do_spec_aug=False,
        norm_signal=False,
        do_preemph=True,
        freq_start_bin = 0,
        lookahead_ratio = 0.5,
        norm_update_steps = None,
        signal_norm_type = 'sliding_mean_var',
        spec_norm_type = 'sliding_mean',
        freq_mask_width = (0, 8), 
        time_mask_width = (0, 10),
        eps = 1e-8
    ):
        super(TFMelBanksV2, self).__init__()

        if signal_norm_type == 'none':
            signal_norm = nn.Identity()
        elif signal_norm_type == 'sliding_mean_var':
            signal_norm = SlidingInstanceNormAudio(window_size=norm_win_len, lookahead_ratio=lookahead_ratio,
                                     eps=1e-5, scale_var=True)
        elif signal_norm_type == 'glob_norm':
            signal_norm = GlobalNormSignal(update_steps=norm_update_steps)
        elif signal_norm_type == 'chunked_mean_var':
            signal_norm = ChunkedInstanceNormAudio(
                chunk_size=norm_win_len, lookahead_ratio=lookahead_ratio,
                eps=1e-5, scale_var=True
            )
        else:
            raise NotImplementedError()

        self.torchfbank = torch.nn.Sequential(
            signal_norm,
            PreEmphasis() if do_preemph else nn.Identity(),
            SpectralFeaturesTF(
                frame_length = win_length,
                frame_step = hop_length,
                fft_length = n_fft,
                sample_rate = sample_rate,
                window = 'hamming',
                normalize_spectrogram = False,
                normalize_signal = False,
                eps = eps,
                mode = 'melbanks',
                low_freq = f_min,
                high_freq = f_max,
                num_bins = n_mels,
                log_mels = False,
                fft_mode = 'abs',
                sqrt_real_imag = False,
                return_img = False,
            )
        )
        self.eps = eps
        if spec_norm_type == 'sliding_mean':
            self.spec_norm = SlidingInstanceNormAudio(window_size=norm_win_len//hop_length, 
                                         eps=1e-5, scale_var=False, lookahead_ratio=lookahead_ratio)
        elif spec_norm_type == 'glob_norm':
            self.spec_norm = GlobalNorm1d(n_mels,update_steps=norm_update_steps)
        else:
            raise NotImplementedError()

        # self.spec_norm = nn.Identity()
        if do_spec_aug:
            self.specaug = FbankAug(
                freq_start_bin=freq_start_bin,
                freq_mask_width=freq_mask_width,
                time_mask_width=time_mask_width) # Spec augmentation
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                x = self.torchfbank(x)+self.eps
                x = x.log()
                x = self.spec_norm(x)
                if self.training:
                    x = self.specaug(x)
        return x.to(xdtype)


class TFSpectrogram(nn.Module):
    def __init__(self, 
        sample_rate=16000, 
        n_fft=512, 
        win_length=400, 
        hop_length=160,
        f_min = 20, 
        f_max = 7600, 
        n_mels = 80, 
        window = 'hamming',
        normalize_spectrogram = False,
        normalize_signal = False,
        mode = 'fft',
        fft_mode = 'abs',
        pool_freqs = (2,1),
 
        do_spec_aug=False,
        norm_signal=False,
        do_preemph=True,
 
        freq_start_bin = 0,
        num_apply_spec_aug = 1,
        freq_mask_width = (0, 8), 
        time_mask_width = (0, 10),
        eps = 1e-8
    ):
        super(TFSpectrogram, self).__init__()
        self.num_apply_spec_aug = num_apply_spec_aug
        self.spectrogram = torch.nn.Sequential(
            NormalizeAudio(squeeze=True) if norm_signal else nn.Identity(),
            PreEmphasis() if do_preemph else nn.Identity(),

            SpectralFeaturesTF(
                frame_length = win_length,
                frame_step = hop_length,
                fft_length = n_fft,
                sample_rate = sample_rate,
                window = window,
                eps = eps,
                mode = mode,
                low_freq = f_min,
                high_freq = f_max,
                num_bins = n_mels,

                normalize_spectrogram = False,
                normalize_signal = False,

                fft_mode = 'abs',
                log_mels = False,
                sqrt_real_imag = False,
                return_img = False,
            )
        )

        if pool_freqs is not None:
            self.pool_freq = nn.AvgPool2d(pool_freqs, stride=pool_freqs)
        else:
            self.pool_freq = nn.Identity()

        self.eps = eps
        if do_spec_aug:
            self.specaug = FbankAug(
                freq_start_bin=freq_start_bin,
                freq_mask_width=freq_mask_width,
                time_mask_width=time_mask_width) # Spec augmentation
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                x = self.spectrogram(x)+self.eps
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if self.training:
                    for _ in range(self.num_apply_spec_aug):
                        x = self.specaug(x)
                x = self.pool_freq(x.unsqueeze(1))
        return x.to(xdtype)
