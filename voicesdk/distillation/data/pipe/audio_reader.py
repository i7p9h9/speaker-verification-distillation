import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf

from ._types import AudioSegments


class AudioReaderBase(ABC):
    """Base class responsible for reading and preprocessing audio files."""

    def __init__(
        self,
        norm_type: str,
        sample_rate: int = 16000,
    ):
        """
        Args:
            norm_type: Normalization type ('std' for z-score, anything else for peak normalization)
            sample_rate: Expected sample rate of audio files
        """
        self.norm_type = norm_type
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load(self, filename: str) -> np.ndarray:
        """Load a wav file and return a mono float32 signal."""
        signal, sr = sf.read(filename, dtype="float32")
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
    ):
        """
        Args:
            norm_type: Normalization type
            length_segment_ms: Segment length in ms (None = return the whole file as one segment)
            segments_step_ms: Hop between segments in ms (None = equals length_segment_ms, i.e. no overlap)
            sample_rate: Expected sample rate
        """
        super().__init__(norm_type=norm_type, sample_rate=sample_rate)
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
        start = int(self._rng.integers(0, max_start + 1))
        segment = signal[start : start + segment_length]

        return AudioSegments(
            segments=segment[np.newaxis, :],
            segments_duration=[self._samples_to_ms(segment_length)],
            segments_weights=[1.0],
            total_duration=total_duration_ms,
        )
