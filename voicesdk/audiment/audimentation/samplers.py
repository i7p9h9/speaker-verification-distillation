"""Batch sampling strategies that control how augmentations are applied across a batch."""

import dataclasses
import typing as tp

from audimentation.base import AugConfig, AugmentationBase, BatchSample, DataSample


class BatchWiseSampler(AugmentationBase):
    """Sample once for the whole batch; apply per element with the same config.

    ``augmentation.sample()`` is called once using the full ``BatchSample``
    (or the single ``DataSample``). ``augmentation.apply()`` is called once
    per element with the same sampled config.

    Args:
        augmentation: Inner augmentation to wrap.
        name: Stage identifier. Default ``'batch_wise_sampler'``.
        p: Probability of running the sampler. Default ``1.0``.
        persist: Persist flags forwarded to the base class.
    """

    def __init__(
        self,
        augmentation: AugmentationBase,
        name: str = "batch_wise_sampler",
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
        self.augmentation = augmentation

    def sample(self, data: DataSample) -> AugConfig:
        """Delegate to the wrapped augmentation's sample()."""
        return self.augmentation.sample(data)

    def apply(self, data: DataSample, config: AugConfig) -> DataSample:
        """Apply wrapped augmentation to a single DataSample with given config."""
        return self.augmentation.apply(data, config)

    def __call__(
        self,
        data: DataSample,
        config: tp.Optional[AugConfig] = None,
    ) -> DataSample:
        """One sample(), then apply() per element with the same config."""
        if config is None:
            if __import__("torch").rand(1).item() > self.p:
                return data
            config = self.augmentation.sample(data)

        if not isinstance(data, BatchSample):
            return self.augmentation.apply(data, config)

        results = [self.augmentation.apply(data[i], config) for i in range(len(data))]
        return BatchSample.from_samples(results)


class SampleWiseSampler(AugmentationBase):
    """Sample independently for each element; apply per element with its own config.

    ``augmentation.sample()`` and ``augmentation.apply()`` are both called
    once per batch element.

    Args:
        augmentation: Inner augmentation to wrap.
        name: Stage identifier. Default ``'sample_wise_sampler'``.
        p: Probability of running the sampler. Default ``1.0``.
        persist: Persist flags forwarded to the base class.
    """

    def __init__(
        self,
        augmentation: AugmentationBase,
        name: str = "sample_wise_sampler",
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
        self.augmentation = augmentation

    def sample(self, data: DataSample) -> AugConfig:
        """Delegate to the wrapped augmentation's sample()."""
        return self.augmentation.sample(data)

    def apply(self, data: DataSample, config: AugConfig) -> DataSample:
        """Apply wrapped augmentation to a single DataSample with given config."""
        return self.augmentation.apply(data, config)

    def __call__(
        self,
        data: DataSample,
        config: tp.Optional[AugConfig] = None,
    ) -> DataSample:
        """One sample() and apply() per element."""
        if config is None:
            if __import__("torch").rand(1).item() > self.p:
                return data

        if not isinstance(data, BatchSample):
            cfg = self.augmentation.sample(data) if config is None else config
            return self.augmentation.apply(data, cfg)

        results = []
        for i in range(len(data)):
            element = data[i]
            cfg = self.augmentation.sample(element) if config is None else config
            results.append(self.augmentation.apply(element, cfg))
        return BatchSample.from_samples(results)


class IdWiseSampler(AugmentationBase):
    """Sample once per unique target id; all elements sharing a target get the same config.

    Requires ``data.target`` to be set.  Raises ``ValueError`` if it is None.

    Args:
        augmentation: Inner augmentation to wrap.
        name: Stage identifier. Default ``'id_wise_sampler'``.
        p: Probability of running the sampler. Default ``1.0``.
        persist: Persist flags forwarded to the base class.
    """

    def __init__(
        self,
        augmentation: AugmentationBase,
        name: str = "id_wise_sampler",
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
        self.augmentation = augmentation

    def sample(self, data: DataSample) -> AugConfig:
        """Delegate to the wrapped augmentation's sample()."""
        return self.augmentation.sample(data)

    def apply(self, data: DataSample, config: AugConfig) -> DataSample:
        """Apply wrapped augmentation to a single DataSample with given config."""
        return self.augmentation.apply(data, config)

    def __call__(
        self,
        data: DataSample,
        config: tp.Optional[AugConfig] = None,
    ) -> DataSample:
        """One sample() per unique target; apply() to each element with its group's config."""
        if config is None:
            if __import__("torch").rand(1).item() > self.p:
                return data

        if not isinstance(data, BatchSample):
            if data.target is None:
                raise ValueError(
                    "IdWiseSampler requires target to be set on the DataSample."
                )
            cfg = self.augmentation.sample(data) if config is None else config
            return self.augmentation.apply(data, cfg)

        if data.target is None:
            raise ValueError(
                "IdWiseSampler requires target to be set on the BatchSample."
            )

        # Convert targets to hashable Python scalars for grouping
        import torch as _torch

        raw = data.target
        if isinstance(raw, _torch.Tensor):
            target_list: tp.List[tp.Any] = raw.tolist()
        elif isinstance(raw, (list, tuple)):
            target_list = list(raw)
        else:
            target_list = [raw] * len(data)

        # One sample() per unique target value (order-preserving)
        seen: tp.Dict[tp.Any, AugConfig] = {}
        for idx, tgt in enumerate(target_list):
            if tgt not in seen:
                cfg = (
                    self.augmentation.sample(data[idx])
                    if config is None
                    else config
                )
                seen[tgt] = cfg

        results = [
            self.augmentation.apply(data[i], seen[tgt])
            for i, tgt in enumerate(target_list)
        ]
        return BatchSample.from_samples(results)
