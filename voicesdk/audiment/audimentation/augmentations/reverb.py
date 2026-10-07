"""Reverb augmentation: convolve signal with a room impulse response."""

import dataclasses
import random
import typing as tp
import uuid

import torch
import torchaudio

from audimentation.base import AugConfig, AugmentationBase, DataSample
from audimentation.providers.base import RIRProvider


@dataclasses.dataclass
class ReverbConfig(AugConfig):
    """Sampled parameters for one Reverb application.

    Attributes:
        wet_dry: Mix ratio in ``[0, 1]``.  ``1.0`` is fully wet (reverberant only).
        rir_id: Opaque identifier for the RIR sample (UUID string).
        rir_tensor: Transient RIR tensor used by ``apply()``.
            Set by ``sample()``; not persisted by default.
    """

    wet_dry: float
    rir_id: str
    rir_tensor: tp.Optional[torch.Tensor] = dataclasses.field(
        default=None, repr=False
    )


class Reverb(AugmentationBase):
    """Convolve a signal with a room impulse response at a random wet/dry ratio.

    Args:
        rir_provider: Source of RIR tensors.
        name: Stage identifier.
        wet_dry_range: ``(min, max)`` range for uniform wet/dry sampling.
        p: Application probability.
        persist: Persist both config and signal.
        persist_config: Persist config only.
        persist_signal: Persist signal only.
    """

    # Fixed RIR duration used when fetching from the provider (seconds).
    _RIR_DURATION: float = 1.0

    def __init__(
        self,
        rir_provider: RIRProvider,
        name: str = "reverb",
        wet_dry_range: tp.Tuple[float, float] = (0.3, 1.0),
        p: float = 1.0,
        persist: bool = False,
        persist_config: bool = False,
        persist_signal: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            p=p,
            persist=persist,
            persist_config=persist_config,
            persist_signal=persist_signal,
        )
        self.rir_provider = rir_provider
        self.wet_dry_range = wet_dry_range

    def sample(self, data: DataSample) -> ReverbConfig:
        """Draw wet/dry ratio and fetch an RIR tensor from the provider.

        Args:
            data: Input sample used to determine sample rate.

        Returns:
            ``ReverbConfig`` with ``wet_dry``, ``rir_id``, and ``rir_tensor``.
        """
        wet_dry = random.uniform(self.wet_dry_range[0], self.wet_dry_range[1])
        rir = self.rir_provider.sample(self._RIR_DURATION, data.sample_rate)
        rir_id = str(uuid.uuid4())
        return ReverbConfig(wet_dry=wet_dry, rir_id=rir_id, rir_tensor=rir)

    def apply(self, data: DataSample, config: ReverbConfig) -> DataSample:
        """Convolve signal with RIR and blend wet/dry.

        The output is trimmed to the original signal length.

        Args:
            data: Input sample.
            config: Sampled config with ``rir_tensor`` populated.

        Returns:
            New ``DataSample`` with reverberated signal, same length as input.

        Raises:
            ValueError: If ``config.rir_tensor`` is None.
        """
        if config.rir_tensor is None:
            raise ValueError(
                "ReverbConfig.rir_tensor is None. "
                "config must be produced by Reverb.sample() or have rir_tensor set."
            )

        signal = data.signal  # (C, T)
        rir = config.rir_tensor.to(signal.device)  # (C_rir, T_rir)
        original_length = signal.shape[-1]

        # Normalize RIR to unit energy
        rir = rir / (rir.norm() + 1e-8)

        # Match RIR channel count to signal channel count.
        # Strategy: truncate extra RIR channels, or collapse to mono then broadcast.
        if rir.shape[0] != signal.shape[0]:
            if rir.shape[0] > signal.shape[0]:
                rir = rir[: signal.shape[0]]
            else:
                # rir has fewer channels (including stereo vs. large batch dim):
                # collapse to mono via first channel, then broadcast
                rir = rir[:1].expand(signal.shape[0], -1)

        # torchaudio.functional.fftconvolve operates on the last dimension.
        # For (C, T) input it convolves each channel independently.
        wet = torchaudio.functional.fftconvolve(signal, rir)  # (C, T + T_rir - 1)
        wet = wet[..., :original_length]  # trim to original length

        # Wet/dry mix: wet_dry=1.0 → fully reverberant; 0.0 → dry only
        mixed = config.wet_dry * wet + (1.0 - config.wet_dry) * signal

        return dataclasses.replace(
            data,
            signal=mixed,
            output=mixed,
            persist=dict(data.persist),
        )
