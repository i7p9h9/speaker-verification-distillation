import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import resampy
import soundfile as sf

from ._types import AudioSegments


class SincWindowConfig(tp.TypedDict, total=False):
    window:     tp.Callable[[np.ndarray], np.ndarray]  # e.g. np.hanning, np.hamming, np.blackman
    num_zeros:  int                                     # number of zero-crossings (default 64)
    precision:  int                                     # table precision (default 9)


_DEFAULT_SINC_CONFIGS: tp.Tuple[SincWindowConfig, ...] = (
    SincWindowConfig(window=np.hanning,  num_zeros=64, precision=9),
    SincWindowConfig(window=np.hamming,  num_zeros=64, precision=9),
    SincWindowConfig(window=np.blackman, num_zeros=64, precision=9),
)

FilterSpec = tp.Union[str, SincWindowConfig]


class AudioReaderBase(ABC):
    """Base class responsible for reading and preprocessing audio files."""

    def __init__(
        self,
        norm_type: str,
        sample_rate: int = 16000,
        force_resample: bool = False
    ):
        """
        Args:
            norm_type: Normalization type ('std' for z-score, anything else for peak normalization)
            sample_rate: Expected sample rate of audio files
        """
        self._force_resample = force_resample
        self._sinc_pool: tp.Tuple[SincWindowConfig, ...] = _DEFAULT_SINC_CONFIGS
        self._rng = np.random.default_rng()

        self.norm_type = norm_type
        self.sample_rate = sample_rate

    @property
    def filter_spec(self):
        return _DEFAULT_SINC_CONFIGS[0]

    def _resolve_filter(self) -> FilterSpec:
        """Return the filter spec to use for the current call."""
        if self.filter_spec is not None:
            return self.filter_spec
        # Random choice from the sinc_window pool
        idx = int(self._rng.integers(0, len(self._sinc_pool)))
        return self._sinc_pool[idx]

    def _resample(
        self,
        signal: np.ndarray,
        sr_in: int,
        sr_out: int,
        spec: FilterSpec,
    ) -> np.ndarray:
        """Single resampy call, dispatched on filter spec type."""
        if isinstance(spec, str):
            return resampy.resample(signal, sr_in, sr_out, filter=spec)

        # SincWindowConfig dict — unpack as sinc_window kwargs
        return resampy.resample(
            signal,
            sr_in,
            sr_out,
            filter="sinc_window",
            **spec,
        )

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load(self, filename: str) -> np.ndarray:
        """Load a wav file and return a mono float32 signal."""
        signal, sr = sf.read(filename, dtype="float32")
        if self._force_resample:
            spec = self._resolve_filter()
            signal = self._resample(signal, sr_in=sr, sr_out=self.sample_rate, spec=spec)
            sr = self.sample_rate

        assert sr == self.sample_rate, (
            f"Expected sample rate {self.sample_rate}, got {sr} in '{filename}'"
        )
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        return signal

    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalize audio signal in-place according to norm_type."""
        if np.std(signal) == 0:
            return signal
        if self.norm_type == "std":
            signal = (signal - np.mean(signal)) / np.std(signal)
        else:
            signal = signal / (np.abs(signal).max() + 1e-4)
        return signal

    # ------------------------------------------------------------------
    # Unit conversion helpers
    # ------------------------------------------------------------------

    def _ms_to_samples(self, ms: int) -> int:
        return int(ms * self.sample_rate / 1000)

    def _samples_to_ms(self, samples: int) -> float:
        return 1000.0 * samples / self.sample_rate

    # ------------------------------------------------------------------
    # Padding helper (shared by subclasses)
    # ------------------------------------------------------------------

    def _pad_to_length(self, segment: np.ndarray, target_length: int) -> np.ndarray:
        """Pad segment to target_length by tiling."""
        current_length = len(segment)
        if current_length >= target_length:
            return segment[:target_length]
        num_repeats = (target_length + current_length - 1) // current_length
        return np.tile(segment, num_repeats)[:target_length]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def read(self, filename: str) -> AudioSegments:
        """
        Load, normalize, and split/crop audio from *filename*.

        Returns:
            AudioSegments ready for downstream processing
        """

    def __call__(self, filename: str) -> AudioSegments:
        return self.read(filename)


# ---------------------------------------------------------------------------
# Concrete readers
# ---------------------------------------------------------------------------


class AudioReaderFull(AudioReaderBase):
    """
    Reads the entire file and splits it into overlapping fixed-length segments.
    Intended for inference / evaluation.
    """

    def __init__(
        self,
        norm_type: str,
        length_segment_ms: tp.Optional[int] = None,
        segments_step_ms: tp.Optional[int] = None,
        sample_rate: int = 16000,
        force_resample: bool = False
    ):
        """
        Args:
            norm_type: Normalization type
            length_segment_ms: Segment length in ms (None = return the whole file as one segment)
            segments_step_ms: Hop between segments in ms (None = equals length_segment_ms, i.e. no overlap)
            sample_rate: Expected sample rate
        """
        super().__init__(norm_type=norm_type, sample_rate=sample_rate, force_resample=force_resample)
        self.length_segment_ms = length_segment_ms
        self.segments_step_ms = (
            segments_step_ms if segments_step_ms is not None else length_segment_ms
        )

    def read(self, filename: str) -> AudioSegments:
        signal = self.normalize(self.load(filename))
        return self._split_into_segments(signal)

    def _split_into_segments(self, signal: np.ndarray) -> AudioSegments:
        total_length = len(signal)
        total_duration_ms = self._samples_to_ms(total_length)

        # No segmentation requested — return the whole signal as a single segment
        if self.length_segment_ms is None:
            return AudioSegments(
                segments=signal[np.newaxis, :],
                segments_duration=[total_duration_ms],
                segments_weights=[1.0],
                total_duration=total_duration_ms,
            )

        segment_length = self._ms_to_samples(self.length_segment_ms)

        # Signal is shorter than one segment — pad and return
        if total_duration_ms < self.length_segment_ms:
            padded = self._pad_to_length(signal, segment_length)
            return AudioSegments(
                segments=padded[np.newaxis, :],
                segments_duration=[total_duration_ms],
                segments_weights=[1.0],
                total_duration=total_duration_ms,
            )

        step_length = self._ms_to_samples(self.segments_step_ms)

        segments: tp.List[np.ndarray] = []
        actual_durations: tp.List[float] = []

        start = 0
        while start < total_length:
            end = start + segment_length
            if end <= total_length:
                segment = signal[start:end]
                actual_length = segment_length
            else:
                # Last partial segment — record real length, pad to full
                actual_length = total_length - start
                segment = self._pad_to_length(signal[start:], segment_length)

            segments.append(segment)
            actual_durations.append(self._samples_to_ms(actual_length))

            if end >= total_length:
                break
            start += step_length

        weights = (
            [1.0]
            if len(segments) == 1
            else [d / sum(actual_durations) for d in actual_durations]
        )

        return AudioSegments(
            segments=np.stack(segments, axis=0),
            segments_duration=actual_durations,
            segments_weights=weights,
            total_duration=total_duration_ms,
        )


class AudioReaderRandom(AudioReaderBase):
    """
    Reads a file and returns a single randomly cropped segment of fixed length.
    Intended for training — each call may return a different crop.
    If the file is shorter than the requested segment, the signal is padded by tiling.
    """

    def __init__(
        self,
        norm_type: str,
        length_segment_ms: int,
        sample_rate: int = 16000,
        rng: tp.Optional[np.random.Generator] = None,
    ):
        """
        Args:
            norm_type: Normalization type
            length_segment_ms: Length of the crop in milliseconds
            sample_rate: Expected sample rate
            rng: Optional numpy random Generator for reproducibility
        """
        super().__init__(norm_type=norm_type, sample_rate=sample_rate)
        self.length_segment_ms = length_segment_ms
        self._rng = rng if rng is not None else np.random.default_rng()

    def read(self, filename: str) -> AudioSegments:
        signal = self.normalize(self.load(filename))
        return self._random_crop(signal)

    def _get_random_start(self, max_start: int) -> int:
        start = int(self._rng.integers(0, max_start + 1))
        return start

    def _random_crop(self, signal: np.ndarray) -> AudioSegments:
        total_length = len(signal)
        total_duration_ms = self._samples_to_ms(total_length)
        segment_length = self._ms_to_samples(self.length_segment_ms)

        if total_length <= segment_length:
            # Pad short signal and return as-is (actual duration < segment length)
            segment = self._pad_to_length(signal, segment_length)
            return AudioSegments(
                segments=segment[np.newaxis, :],
                segments_duration=[total_duration_ms],
                segments_weights=[1.0],
                total_duration=total_duration_ms,
            )

        # Pick a random start position
        max_start = total_length - segment_length
        start = self._get_random_start(max_start)
        segment = signal[start : start + segment_length]

        return AudioSegments(
            segments=segment[np.newaxis, :],
            segments_duration=[self._samples_to_ms(segment_length)],
            segments_weights=[1.0],
            total_duration=total_duration_ms,
        )


class AudioReaderBegin(AudioReaderRandom):
    def _get_random_start(self, *_):
        return 0


class AudioReaderTelSimulated(AudioReaderBase):
    """
    Random-crop reader with optional telephone-channel simulation.

    With probability *p_tel* the cropped segment is downsampled to 8 kHz
    and upsampled back to *sample_rate* via resampy, mimicking the bandwidth
    limitation of a telephone codec.

    Supported filter specs
    ----------------------
    - ``"kaiser_best"``  — high-quality Kaiser window preset
    - ``"kaiser_fast"``  — faster Kaiser window preset
    - :class:`SincWindowConfig` dict, e.g.::

          {"window": np.hanning, "num_zeros": 64, "precision": 9}

    Passing ``filter_spec=None`` picks one of the built-in configs at random
    on every ``read()`` call (good for training augmentation diversity).
    """

    TEL_SAMPLE_RATE: int = 8_000

    _NAMED_FILTERS: tp.Tuple[str, ...] = ("kaiser_best", "kaiser_fast")

    def __init__(
        self,
        norm_type: str,
        length_segment_ms: int,
        p_tel: float = 0.5,
        sample_rate: int = 16_000,
        filter_spec: tp.Optional[FilterSpec] = None,
        sinc_configs: tp.Optional[tp.Tuple[SincWindowConfig, ...]] = None,
        rng: tp.Optional[np.random.Generator] = None,
    ):
        """
        Args:
            norm_type: Normalization type ('std' or peak).
            length_segment_ms: Length of the random crop in milliseconds.
            p_tel: Probability of applying telephone-channel simulation [0, 1].
            sample_rate: Target sample rate of audio files.
            filter_spec: Filter to use for resampling.
                - ``str``  → named resampy preset ('kaiser_best' | 'kaiser_fast').
                - ``SincWindowConfig`` → passed as kwargs to resampy's sinc_window.
                - ``None`` → sampled at random from *sinc_configs* on every call.
            sinc_configs: Pool of SincWindowConfig dicts to sample from when
                          *filter_spec* is None.  Defaults to hanning / hamming /
                          blackman configs with num_zeros=64, precision=9.
            rng: Optional numpy random Generator for reproducibility.
        """
        assert 0.0 <= p_tel <= 1.0, "p_tel must be in [0, 1]"
        if isinstance(filter_spec, str) and filter_spec not in self._NAMED_FILTERS:
            raise ValueError(
                f"Unknown named filter '{filter_spec}'. "
                f"Choose from {self._NAMED_FILTERS} or pass a SincWindowConfig dict."
            )

        super().__init__(norm_type=norm_type, sample_rate=sample_rate)
        self.length_segment_ms = length_segment_ms
        self.p_tel = p_tel
        self._filter_spec = filter_spec
        self._sinc_pool: tp.Tuple[SincWindowConfig, ...] = (
            sinc_configs if sinc_configs is not None else _DEFAULT_SINC_CONFIGS
        )
        self._rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(self, filename: str) -> AudioSegments:
        signal = self.normalize(self.load(filename))
        segment = self._random_crop(signal)
        if self._rng.random() < self.p_tel:
            segment = self._tel_simulate(segment)
        return self._wrap(segment, total_samples=len(signal))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_crop(self, signal: np.ndarray) -> np.ndarray:
        """Return a randomly cropped (or padded) fixed-length segment."""
        total_length = len(signal)
        segment_length = self._ms_to_samples(self.length_segment_ms)

        if total_length <= segment_length:
            return self._pad_to_length(signal, segment_length)

        max_start = total_length - segment_length
        start = int(self._rng.integers(0, max_start + 1))
        return signal[start : start + segment_length]

    def _resolve_filter(self) -> FilterSpec:
        """Return the filter spec to use for the current call."""
        if self._filter_spec is not None:
            return self._filter_spec
        # Random choice from the sinc_window pool
        idx = int(self._rng.integers(0, len(self._sinc_pool)))
        return self._sinc_pool[idx]

    def _resample(
        self,
        signal: np.ndarray,
        sr_in: int,
        sr_out: int,
        spec: FilterSpec,
    ) -> np.ndarray:
        """Single resampy call, dispatched on filter spec type."""
        if isinstance(spec, str):
            return resampy.resample(signal, sr_in, sr_out, filter=spec)

        # SincWindowConfig dict — unpack as sinc_window kwargs
        return resampy.resample(
            signal,
            sr_in,
            sr_out,
            filter="sinc_window",
            **spec,
        )

    def _tel_simulate(self, segment: np.ndarray) -> np.ndarray:
        """Downsample to 8 kHz then upsample back — pure numpy via resampy."""
        spec = self._resolve_filter()

        degraded = self._resample(segment, self.sample_rate, self.TEL_SAMPLE_RATE, spec)
        restored = self._resample(degraded, self.TEL_SAMPLE_RATE, self.sample_rate, spec)

        # Resampling may produce ±1 sample difference — fix it
        target_len = self._ms_to_samples(self.length_segment_ms)
        if len(restored) < target_len:
            restored = self._pad_to_length(restored, target_len)
        else:
            restored = restored[:target_len]

        return restored

    def _wrap(self, segment: np.ndarray, total_samples: int) -> AudioSegments:
        """Package a single segment into an AudioSegments container."""
        return AudioSegments(
            segments=segment[np.newaxis, :],
            segments_duration=[self._samples_to_ms(len(segment))],
            segments_weights=[1.0],
            total_duration=self._samples_to_ms(total_samples),
        )
