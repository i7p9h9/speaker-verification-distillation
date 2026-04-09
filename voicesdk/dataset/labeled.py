from __future__ import annotations

import typing as tp

import torch

from ._registry import _NameRegistry
from ._shared_weights import SharedWeights
from ._type import LabeledSample, LabeledSource, WeightedDataset
from .aggregated import AggregatedDataset


def _sample_with_path(ds: AggregatedDataset) -> tp.Tuple[tp.Any, str]:
    """
    Sample one item from *ds* and return ``(sample, relative_path)``.

    The relative path is the '/' joined sequence of child node names
    selected at each level, *excluding* the name of *ds* itself (the
    caller is responsible for prepending it).

    If a child is itself an AggregatedDataset, we recurse into it so
    the full leaf path is captured.
    """
    r = torch.rand(1).item()
    idx = int(torch.searchsorted(ds.cum_probs, torch.tensor(r)).clamp(0, len(ds._datasets) - 1))
    wd = ds._weighted[idx]
    child_name = wd.name or ""
    child_ds = ds._datasets[idx]

    if isinstance(child_ds, AggregatedDataset):
        sample, sub_path = _sample_with_path(child_ds)
        path = f"{child_name}/{sub_path}" if sub_path else child_name
    else:
        inner_idx = int(torch.randint(0, len(child_ds), (1,)).item())  # type: ignore[arg-type]
        sample = child_ds[inner_idx]
        path = child_name

    return sample, path


# ---------------------------------------------------------------------------
# LabeledAggregatedDataset
# ---------------------------------------------------------------------------


class LabeledAggregatedDataset(AggregatedDataset):
    """
    Dataset that samples from multiple named, labeled sources and returns
    :class:`LabeledSample` instances carrying the full dataset path.

    Each source is a :class:`LabeledSource` that bundles:
      - a pool of data (AggregatedDataset or list of WeightedDataset),
      - an integer label,
      - an optional sampling weight,
      - an optional unique name.

    Sampling procedure
    ------------------
    1. Draw a source proportionally to its resolved weight.
    2. Draw one sample from that source's inner AggregatedDataset.
    3. Return :class:`LabeledSample` with ``(sample, label, dataset_name)``.

    The ``dataset_name`` field is the full '/' separated path, e.g.
    ``"root/vocals/clean_speech"``.

    Weight API
    ----------
    Inherits :meth:`get_weights` and :meth:`set_weights` from
    :class:`AggregatedDataset`.  Weights are accessible and modifiable at
    every level of the hierarchy using path strings.

    Parameters
    ----------
    sources:
        Non-empty list of :class:`LabeledSource` instances.
    name:
        Unique name for this top-level dataset. Auto-generated if not provided.
    """

    def __init__(
        self,
        sources: tp.List[LabeledSource],
        name: tp.Optional[str] = None,
    ) -> None:
        assert len(sources) > 0, "At least one LabeledSource must be provided"
        self._validate_labeled_sources(sources)

        self._inner_datasets = [self._ensure_aggregated(s) for s in sources]
        self._labels = [s.label for s in sources]

        wrapped = [
            WeightedDataset(dataset=inner, weight=src.weight, name=inner.name)
            for src, inner in zip(sources, self._inner_datasets)
        ]
        super().__init__(sources=wrapped, name=name)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_aggregated(src: LabeledSource) -> AggregatedDataset:
        """
        Return src.dataset as AggregatedDataset.
        If it is a list of WeightedDataset, wrap it first.
        The source name is forwarded to the inner dataset.
        """
        if isinstance(src.dataset, AggregatedDataset):
            return src.dataset
        return AggregatedDataset(src.dataset, name=src.name)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_labeled_sources(sources: tp.List[LabeledSource]) -> None:
        weights_none = [s.weight is None for s in sources]
        assert all(weights_none) or not any(weights_none), (
            "Either all LabeledSource weights must be None or all must be set"
        )
        if not any(weights_none):
            assert all(s.weight > 0 for s in sources), (  # type: ignore[operator]
                "All LabeledSource weights must be positive"
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------


    def __getitem__(self, index: int) -> LabeledSample:
        r = torch.rand(1).item()
        dataset_idx = int(
            torch.searchsorted(
                self.cum_probs,
                torch.tensor(r),
            ).clamp(0, len(self._datasets) - 1)
        )
        inner_ds = self._inner_datasets[dataset_idx]
        source_name = self._weighted[dataset_idx].name or inner_ds.name
        label = self._labels[dataset_idx]

        sample, leaf_path = _sample_with_path(inner_ds)
        dataset_name = f"{self._name}/{source_name}/{leaf_path}" if leaf_path else f"{self._name}/{source_name}"
        return LabeledSample(sample=sample, label=label, dataset_name=dataset_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def labels(self) -> tp.List[int]:
        """Integer labels aligned with :py:attr:`datasets`."""
        return self._labels

    def __repr__(self) -> str:
        parts = [
            f"  [{i}] label={self._labels[i]}  {wd.name}(len={self._lengths[i]}, p={wd.weight:.4f})"
            for i, wd in enumerate(self._weighted)
        ]
        return f"LabeledAggregatedDataset(name={self._name!r},\n" + "\n".join(parts) + "\n)"
