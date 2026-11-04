"""Mix strategies for ParallelCompose output combination."""

import typing as tp

import torch


class MixStrategy(tp.Protocol):
    """Protocol for combining a list of signals into one.

    All implementations must be stateless and thread-safe.
    """

    def mix(
        self,
        signals: tp.List[torch.Tensor],
        targets: tp.Optional[tp.List[tp.Any]],
    ) -> torch.Tensor:
        """Combine *signals* into a single tensor.

        Args:
            signals: List of tensors with identical shapes.
            targets: Optional list of labels (one per signal). May be ignored.

        Returns:
            Combined tensor with the same shape as each input signal.
        """
        ...


class SumMix:
    """Element-wise sum of all signals."""

    def mix(
        self,
        signals: tp.List[torch.Tensor],
        targets: tp.Optional[tp.List[tp.Any]] = None,
    ) -> torch.Tensor:
        """Return the element-wise sum of *signals*."""
        return torch.stack(signals).sum(dim=0)


class MeanMix:
    """Element-wise mean of all signals."""

    def mix(
        self,
        signals: tp.List[torch.Tensor],
        targets: tp.Optional[tp.List[tp.Any]] = None,
    ) -> torch.Tensor:
        """Return the element-wise mean of *signals*."""
        return torch.stack(signals).mean(dim=0)


class MixUpMix:
    """MixUp blending strategy — reserved for future implementation.

    Raises:
        NotImplementedError: Always. This class is an architecture stub only.
            Use ``SumMix`` or ``MeanMix`` instead.
    """

    def mix(
        self,
        signals: tp.List[torch.Tensor],
        targets: tp.Optional[tp.List[tp.Any]] = None,
    ) -> torch.Tensor:
        """Not implemented. Raises NotImplementedError."""
        raise NotImplementedError(
            "MixUpMix is reserved for future implementation and is not yet available. "
            "Use SumMix or MeanMix instead."
        )
