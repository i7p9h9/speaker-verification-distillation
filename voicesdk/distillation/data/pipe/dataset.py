import typing as tp

import numpy as np
import soundfile as sf

from ._types import AudioSegments


class WavDataset:
    def __init__(
        self,
        wav_scp: tp.List[str],
        norm_type: str,
        length_segment_ms: tp.Optional[int] = None,
        segments_step_ms: tp.Optional[int] = None,
        sample_rate: int = 16000,
    ):
        """
        Args:
            wav_scp: List of paths to wav files
            norm_type: Normalization type ('std' or other for max normalization)
            length_segment_ms: Segment length in milliseconds (None = return entire file)
            segments_step_ms: Step between segments in ms (None = equals length_segment_ms)
            sample_rate: Sampling rate (default 16000)
        """
        self.wav_list = wav_scp
        self.norm_type = norm_type
        self.length_segment_ms = length_segment_ms
        self.segments_step_ms = segments_step_ms if segments_step_ms is not None else length_segment_ms
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.wav_list)

    def _load_data(self, filename: str) -> np.ndarray:
        """Loads wav file using soundfile."""
        signal, sr = sf.read(filename, dtype="float32")
        assert sr == self.sample_rate, f"Expected sample rate {self.sample_rate}, got {sr}"

        # Convert stereo to mono if needed
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        return signal

    def _norm_speech(self, signal: np.ndarray) -> np.ndarray:
        """Normalizes audio signal."""
        if np.std(signal) == 0:
            return signal

        if self.norm_type == "std":
            signal = (signal - np.mean(signal)) / np.std(signal)
        else:
            signal = signal / (np.abs(signal).max() + 1e-4)

        return signal

    def _ms_to_samples(self, ms: int) -> int:
        """Converts milliseconds to number of samples."""
        return int(ms * self.sample_rate / 1000)

    def _samples_to_ms(self, samples: int) -> float:
        """Converts samples to ms."""
        return 1000 * samples / self.sample_rate

    def _samples_to_seconds(self, samples: int) -> float:
        """Converts samples to seconds."""
        return samples / self.sample_rate

    def _pad_segment(self, segment: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pads segment to target length by tiling.

        Args:
            segment: Source segment
            target_length: Target length

        Returns:
            Padded segment
        """
        current_length = len(segment)

        if current_length >= target_length:
            return segment[:target_length]

        num_repeats = (target_length + current_length - 1) // current_length
        padded = np.tile(segment, num_repeats)[:target_length]

        return padded

    def _split_into_segments(self, signal: np.ndarray) -> AudioSegments:
        """
        Splits signal into segments of specified length.

        Args:
            signal: Input audio signal

        Returns:
            AudioSegments with segmented data
        """
        total_length = len(signal)
        total_duration_ms = self._samples_to_ms(total_length)

        # If no segmentation needed
        if self.length_segment_ms is None:
            return AudioSegments(
                segments=signal[np.newaxis, :],  # [1, length]
                segments_duration=[total_duration_ms],
                segments_weights=[1.0],
                total_duration=total_duration_ms,
            )

        segment_length = self._ms_to_samples(self.length_segment_ms)

        # If current signal is less than segment length
        if total_duration_ms < self.length_segment_ms:
            padded_signal = self._pad_segment(signal, segment_length)
            return AudioSegments(
                segments=padded_signal[np.newaxis, :],  # [1, length]
                segments_duration=[total_duration_ms],
                segments_weights=[1.0],
                total_duration=total_duration_ms,
            )

        step_length = self._ms_to_samples(self.segments_step_ms)

        segments = []
        actual_durations = []

        start = -step_length
        end = start + segment_length

        while end < total_length:
            start += step_length
            end = start + segment_length

            # Extract segment
            if end <= total_length:
                segment = signal[start:end]
                actual_length = len(segment)
            else:
                # Last segment might be shorter - take last segment_length samples
                segment = signal[start:]
                actual_length = len(segment)
                segment = signal[-segment_length:]

            segments.append(segment)
            actual_duration = self._samples_to_ms(actual_length)
            actual_durations.append(actual_duration)

        # Calculate segment weights
        if len(segments) == 1:
            weights = [1.0]
        else:
            weights = [_duration / sum(actual_durations) for _duration in actual_durations]

        # Stack segments into array
        segments_array = np.stack(segments, axis=0)

        return AudioSegments(
            segments=segments_array,
            segments_duration=actual_durations,
            segments_weights=weights,
            total_duration=total_duration_ms,
        )

    def __getitem__(self, idx: int) -> AudioSegments:
        """
        Returns audio segments for given index.

        Args:
            idx: File index in the list

        Returns:
            AudioSegments with processed data
        """
        filename = self.wav_list[idx]
        signal = self._load_data(filename)
        signal = self._norm_speech(signal)

        return self._split_into_segments(signal)
