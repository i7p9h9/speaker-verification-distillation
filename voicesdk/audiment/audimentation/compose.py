"""Compose classes for building sequential and parallel augmentation pipelines."""

import dataclasses
import random
import typing as tp

from audimentation.base import AugConfig, AugmentationBase, DataSample
from audimentation.mix_strategy import MixStrategy


@dataclasses.dataclass
class OneOfConfig(AugConfig):
    """Sampled parameters for one OneOf application.

    Attributes:
        chosen_index: Index of the stage selected for this call.
    """

    chosen_index: int


class OneOf(AugmentationBase):
    """Apply exactly one randomly chosen stage from the list.

    The stage is selected once per call using weighted random sampling.
    Persist entries from the chosen inner stage are accumulated on the result.

    Args:
        stages: List of candidate augmentation stages.
        weights: Optional sampling weights (same length as ``stages``).
            If ``None``, all stages are equally probable.
        name: Stage identifier. Default ``'one_of'``.
        p: Probability of running the compose at all. Default ``1.0``.
        persist: If True, save config + signal for this compose node itself.
        persist_config: If True, save only config for this compose node.
        persist_signal: If True, save only signal for this compose node.
    """

    def __init__(
        self,
        stages: tp.List[AugmentationBase],
        weights: tp.Optional[tp.List[float]] = None,
        name: str = "one_of",
        p: float = 1.0,
        persist: bool = False,
        persist_config: bool = False,
        persist_signal: bool = False,
    ) -> None:
        if not stages:
            raise ValueError("OneOf requires at least one stage.")
        if weights is not None and len(weights) != len(stages):
            raise ValueError("weights must have the same length as stages.")
        super().__init__(
            name=name,
            p=p,
            persist=persist,
            persist_config=persist_config,
            persist_signal=persist_signal,
        )
        self.stages = stages
        self.weights = weights

    def sample(self, data: DataSample) -> OneOfConfig:
        """Choose one stage index proportionally to weights."""
        chosen = random.choices(range(len(self.stages)), weights=self.weights, k=1)[0]
        return OneOfConfig(chosen_index=chosen)

    def apply(self, data: DataSample, config: OneOfConfig) -> DataSample:
        """Run the selected stage and return its result."""
        return self.stages[config.chosen_index](data)


class SequentialCompose(AugmentationBase):
    """Apply a list of augmentation stages in order.

    The output of each stage becomes the input to the next.
    Persist entries from all inner stages are accumulated in the returned sample.

    Args:
        stages: Ordered list of augmentation stages.
        name: Stage identifier. Default ``'sequential_compose'``.
        p: Probability of running the entire compose. Default ``1.0``.
        persist: If True, save config + signal for this compose node itself.
        persist_config: If True, save only config for this compose node.
        persist_signal: If True, save only signal for this compose node.
    """

    def __init__(
        self,
        stages: tp.List[AugmentationBase],
        name: str = "sequential_compose",
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
        self.stages = stages

    def sample(self, data: DataSample) -> AugConfig:
        """Return an empty config — SequentialCompose has no own parameters."""
        return AugConfig()

    def apply(self, data: DataSample, config: AugConfig) -> DataSample:
        """Run all stages in order, passing each result as the next input.

        Inner stage persist entries are accumulated on the returned sample.
        """
        current = data
        for stage in self.stages:
            current = stage(current)
        return current


class ParallelCompose(AugmentationBase):
    """Apply all stages to the **same** input and combine outputs via a mix strategy.

    Persist entries from all inner stages are collected and merged into the
    returned sample.

    Args:
        stages: List of augmentation stages, all receiving identical input.
        mix_strategy: Strategy for combining stage outputs.
        name: Stage identifier. Default ``'parallel_compose'``.
        p: Probability of running the entire compose. Default ``1.0``.
        persist: If True, save config + signal for this compose node itself.
        persist_config: If True, save only config for this compose node.
        persist_signal: If True, save only signal for this compose node.
    """

    def __init__(
        self,
        stages: tp.List[AugmentationBase],
        mix_strategy: MixStrategy,
        name: str = "parallel_compose",
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
        self.stages = stages
        self.mix_strategy = mix_strategy

    def sample(self, data: DataSample) -> AugConfig:
        """Return an empty config — ParallelCompose has no own parameters."""
        return AugConfig()

    def apply(self, data: DataSample, config: AugConfig) -> DataSample:
        """Run all stages on the same input, then mix outputs.

        Inner stage persist entries are merged (later stages win on key collision).
        """
        outputs: tp.List[tp.Any] = []
        merged_persist = dict(data.persist)

        for stage in self.stages:
            result = stage(data)
            outputs.append(result.output)
            merged_persist.update(result.persist)

        mixed = self.mix_strategy.mix(outputs, [data.target] * len(outputs))
        return dataclasses.replace(
            data,
            signal=mixed,
            output=mixed,
            persist=merged_persist,
        )
