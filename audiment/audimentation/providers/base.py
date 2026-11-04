"""Provider protocols for audio material sources (noise, RIRs, etc.)."""

import typing as tp

import torch


class AudioProvider(tp.Protocol):
    """Protocol for any source of audio tensors.

    Implementations must be **thread-safe**: concurrent calls to ``sample()``
    from multiple threads must not corrupt internal state.
    """

    def sample(self, duration: float, sample_rate: int) -> torch.Tensor:
        """Return an audio tensor of the requested duration.

        Args:
            duration: Desired duration in seconds.
            sample_rate: Target sample rate in Hz.

        Returns:
            Tensor of shape ``(C, T)`` where ``T = round(duration * sample_rate)``.
        """
        ...


# Type aliases — semantically distinct provider roles sharing the same protocol
NoiseProvider = AudioProvider
RIRProvider = AudioProvider
