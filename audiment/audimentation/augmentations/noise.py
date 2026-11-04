"""AddNoise augmentation: add background noise at a target SNR."""

import dataclasses
import random
import typing as tp
import uuid

import torch

from audimentation.base import AugConfig, AugmentationBase, DataSample
from audimentation.providers.base import NoiseProvider
from audimentation.utils import rms


@dataclasses.dataclass
class AddNoiseConfig(AugConfig):
    """Sampled parameters for one AddNoise application.

    Attributes:
        snr_db: Target signal-to-noise ratio in dB.
        noise_id: Opaque identifier for the noise sample (UUID string).
            Stored for replay identification; no raw tensor is persisted by default.
        noise_tensor: Transient noise tensor used by ``apply()``.
            Set by ``sample()``; not persisted unless ``store_noise_tensor=True``.
    """

    snr_db: float
    noise_id: str
    noise_tensor: tp.Optional[torch.Tensor] = dataclasses.field(
        default=None, repr=False
    )


class AddNoise(AugmentationBase):
    """Add background noise to a signal at a randomly sampled SNR.

    The noise level is set so that the output achieves the sampled ``snr_db``
    relative to the clean signal.

    Args:
        noise_provider: Source of noise tensors.
        name: Stage identifier.
        snr_range: ``(min_db, max_db)`` range for uniform SNR sampling.
        store_noise_tensor: If ``True``, the noise tensor is kept in the
            persisted ``AddNoiseConfig``.  Default ``False``.
        p: Application probability.
        persist: Persist both config and signal.
        persist_config: Persist config only.
        persist_signal: Persist signal only.
    """

    def __init__(
        self,
        noise_provider: NoiseProvider,
        name: str = "add_noise",
        snr_range: tp.Tuple[float, float] = (5.0, 30.0),
        store_noise_tensor: bool = False,
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
        self.noise_provider = noise_provider
        self.snr_range = snr_range
        self.store_noise_tensor = store_noise_tensor

    def sample(self, data: DataSample) -> AddNoiseConfig:
        """Draw SNR and fetch a noise tensor from the provider.

        Args:
            data: Input sample used to determine duration and sample rate.

        Returns:
            ``AddNoiseConfig`` with ``snr_db``, ``noise_id``, and ``noise_tensor``.
        """
        snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
        duration = data.signal.shape[-1] / data.sample_rate
        noise = self.noise_provider.sample(duration, data.sample_rate)
        noise_id = str(uuid.uuid4())
        return AddNoiseConfig(snr_db=snr_db, noise_id=noise_id, noise_tensor=noise)

    def apply(self, data: DataSample, config: AddNoiseConfig) -> DataSample:
        """Mix noise into ``data.signal`` at the target SNR.

        Args:
            data: Input sample.
            config: Sampled config with ``noise_tensor`` populated.

        Returns:
            New ``DataSample`` with augmented signal.

        Raises:
            ValueError: If ``config.noise_tensor`` is None.
        """
        if config.noise_tensor is None:
            raise ValueError(
                "AddNoiseConfig.noise_tensor is None. "
                "config must be produced by AddNoise.sample() or have noise_tensor set."
            )

        signal = data.signal
        noise = config.noise_tensor.to(signal.device)

        # Broadcast mono noise to multi-channel signal
        if noise.shape[0] == 1 and signal.shape[0] > 1:
            noise = noise.expand_as(signal)
        elif noise.shape[0] != signal.shape[0]:
            noise = noise[: signal.shape[0]]

        # Align length
        if noise.shape[-1] != signal.shape[-1]:
            from audimentation.utils import crop_or_pad

            noise = crop_or_pad(noise, signal.shape[-1])

        scale = self._compute_snr_scale(signal, noise, config.snr_db)
        mixed = signal + scale * noise

        # Normalise output power to match the input signal power so that
        # loudness is preserved regardless of the noise level.
        rms_in = rms(signal)
        rms_out = rms(mixed)
        gain = rms_in / (rms_out + 1e-10)
        augmented = mixed * gain

        # Optionally strip the tensor from config before returning
        if not self.store_noise_tensor:
            config = dataclasses.replace(config, noise_tensor=None)

        return dataclasses.replace(
            data,
            signal=augmented,
            output=augmented,
            persist=dict(data.persist),
        )

    @staticmethod
    def _compute_snr_scale(
        signal: torch.Tensor,
        noise: torch.Tensor,
        target_snr_db: float,
    ) -> float:
        """Compute the noise scale factor that achieves *target_snr_db*.

        SNR_dB = 20 * log10(rms_signal / rms_noise_scaled)
        => scale = rms_signal / (rms_noise * 10^(target_snr_db / 20))

        Args:
            signal: Clean signal tensor.
            noise: Noise tensor (same shape as signal).
            target_snr_db: Desired SNR in dB.

        Returns:
            Float scale factor to multiply noise by before adding to signal.
        """
        rms_signal = rms(signal).item()
        rms_noise = rms(noise).item()
        if rms_noise < 1e-10:
            return 0.0
        target_rms_noise = rms_signal / (10.0 ** (target_snr_db / 20.0))
        return target_rms_noise / rms_noise
