from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from torch.utils.data import Dataset

if tp.TYPE_CHECKING:
    from .aggregated import AggregatedDataset


@dataclass
class WeightedDataset:
    dataset: Dataset
    weight: tp.Optional[float] = None
    name: tp.Optional[str] = None


@dataclass
class LabeledSample:
    """
    Single item returned by LabeledAggregatedDataset.__getitem__.

    Attributes
    ----------
    sample:
        The raw item from the inner dataset.
    label:
        Integer class label of the source that produced the sample.
    dataset_name:
        Full hierarchical path of the dataset that produced this sample,
        using '/' as the level separator.
        Example: ``"my_root/vocals/clean_speech"``
    """

    sample: tp.Any
    label: int
    dataset_name: str


@dataclass
class LabeledSource:
    """
    A labeled data source for use with LabeledAggregatedDataset.

    Parameters
    ----------
    dataset:
        Either a ready-made AggregatedDataset or a list of WeightedDataset
        entries that will be wrapped into one automatically.
    label:
        Integer class label returned alongside every sample from this source.
    weight:
        Sampling weight for this source relative to others.
        - All None  -> uniform (1/N each).
        - All set   -> normalized to sum=1.
        Mixed None / non-None is not allowed.
    name:
        Unique name for this source.  Auto-generated if not provided.
    """

    dataset: tp.Union[AggregatedDataset, tp.List[WeightedDataset]]
    label: int
    weight: tp.Optional[float] = None
    name: tp.Optional[str] = None
