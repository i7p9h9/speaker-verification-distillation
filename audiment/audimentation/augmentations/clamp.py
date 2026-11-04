"""Clamp augmentation: clip signal amplitude at configurable quantile thresholds."""

import dataclasses
import typing as tp

import torch

from audimentation.base import AugConfig, AugmentationBase, DataSample


@dataclasses.dataclass
class ClampConfig(AugConfig):
    """Computed clamp thresholds for one application.

    Attributes:
        low: Lower bound; all signal values below this are clipped up.
        high: Upper bound; all signal values above this are clipped down.
    """

    low: float
    high: float


class Clamp(AugmentationBase):
    """Clip signal amplitude at per-call quantile thresholds.

    Thresholds are computed **from the signal itself** on each call,
    so they adapt to the actual amplitude distribution of each sample.

    Default behaviour (``quantile=0.05``, ``symmetric=True``):
    compute the 5th and 95th percentiles of the signal, take the larger
    absolute value as the symmetric bound, and clamp to ``[-bound, +bound]``.
    This effectively removes the top/bottom 5 % of amplitude excursions.

    Fine-tuning:
        - ``low_ratio`` — fraction from the **bottom** (default: ``quantile``).
          Lower bound is set to the ``low_ratio``-th percentile.
        - ``high_ratio`` — fraction from the **top** (default: ``quantile``).
          Upper bound is set to the ``(1 - high_ratio)``-th percentile.
        - ``symmetric=False`` — use asymmetric bounds computed independently.

    Example — tighter clip on the top, looser on the bottom::

        Clamp(name="clip", quantile=0.05, high_ratio=0.02, symmetric=False)

    Args:
        name: Stage identifier.
        quantile: Default fraction clipped from each side. Range ``(0, 0.5)``.
        low_ratio: Override fraction clipped from the **bottom**.
            Defaults to ``quantile`` when ``None``.
        high_ratio: Override fraction clipped from the **top**.
            Defaults to ``quantile`` when ``None``.
        symmetric: If ``True`` (default), mirror both thresholds around zero
            using ``bound = max(|low_val|, |high_val|)`` so the clamp is
            centred at zero.  If ``False``, use the raw asymmetric percentiles.
        p: Application probability.
        persist: Persist both config and signal.
        persist_config: Persist config only.
        persist_signal: Persist signal only.
    """

    def __init__(
        self,
        name: str = "clamp",
        quantile: float = 0.05,
        low_ratio: tp.Optional[float] = None,
        high_ratio: tp.Optional[float] = None,
        symmetric: bool = True,
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
        if not 0.0 < quantile < 0.5:
            raise ValueError(f"quantile must be in (0, 0.5), got {quantile}")
        self.quantile = quantile
        self._low_ratio: float = low_ratio if low_ratio is not None else quantile
        self._high_ratio: float = high_ratio if high_ratio is not None else quantile
        self.symmetric = symmetric

    def sample(self, data: DataSample) -> ClampConfig:
        """Compute clamp thresholds from the current signal distribution.

        The thresholds are derived from the actual percentiles of ``data.signal``
        and stored in ``ClampConfig`` for deterministic ``apply()``.

        Args:
            data: Input sample whose signal defines the amplitude distribution.

        Returns:
            ``ClampConfig`` with ``low`` and ``high`` thresholds.
        """
        flat = data.signal.float().flatten()

        low_val = torch.quantile(flat, self._low_ratio).item()
        high_val = torch.quantile(flat, 1.0 - self._high_ratio).item()

        if self.symmetric:
            bound = max(abs(low_val), abs(high_val))
            low_val = -bound
            high_val = bound

        return ClampConfig(low=low_val, high=high_val)

    def apply(self, data: DataSample, config: ClampConfig) -> DataSample:
        """Clip signal values to ``[config.low, config.high]``.

        Args:
            data: Input sample.
            config: Thresholds computed by ``sample()``.

        Returns:
            New ``DataSample`` with clamped signal.
        """
        clamped = data.signal.clamp(min=config.low, max=config.high)
        return dataclasses.replace(
            data,
            signal=clamped,
            output=clamped,
            persist=dict(data.persist),
        )
