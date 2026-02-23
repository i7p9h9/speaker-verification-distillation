import typing as tp
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class WeightedDataset:
    dataset: Dataset
    weight: tp.Optional[float] = None


class AggregatedDataset(Dataset):
    """
    Aggregator dataset that samples from multiple datasets
    with configurable sampling probabilities.

    Accepts either:
    - A list of Dataset objects -> equal probability (1 / N) per dataset
    - A list of WeightedDataset dataclasses where:
        - All weights are None  -> weights proportional to len(dataset)
        - All weights are set (positive floats) -> weights used directly (normalized)

    In all cases weights are resolved into a unified list of WeightedDataset
    with explicit normalized floats before sampling.
    """

    def __init__(self, sources: tp.Union[tp.List[Dataset], tp.List[WeightedDataset]]) -> None:
        assert len(sources) > 0, "At least one dataset must be provided"

        self._weighted = self._normalize(sources)
        self._datasets = [s.dataset for s in self._weighted]
        self._lengths = [len(ds) for ds in self._datasets]  # type: ignore[arg-type]
        self._total = sum(self._lengths)
        self._cum_probs = torch.tensor([s.weight for s in self._weighted]).cumsum(dim=0)

    # ------------------------------------------------------------------
    # Normalization pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(
        sources: tp.Union[tp.List[Dataset], tp.List[WeightedDataset]]
    ) -> tp.List[WeightedDataset]:
        """
        Convert any accepted input format into a list of WeightedDataset
        with explicit normalized weights that sum to 1.
        """
        wrapped = AggregatedDataset._wrap(sources)
        return AggregatedDataset._resolve_weights(wrapped)

    @staticmethod
    def _wrap(
        sources: tp.Union[tp.List[Dataset], tp.List[WeightedDataset]]
    ) -> tp.List[WeightedDataset]:
        """
        Wrap plain Dataset list into WeightedDataset list (weights=None).
        Validate WeightedDataset list if already provided.
        """
        if not isinstance(sources[0], WeightedDataset):
            return [WeightedDataset(dataset=ds, weight=None) for ds in sources]  # type: ignore[arg-type]

        weighted: tp.List[WeightedDataset] = sources  # type: ignore[assignment]
        weights_none = [s.weight is None for s in weighted]
        assert all(weights_none) or not any(weights_none), (
            "Either all weights must be None or all must be set"
        )
        if not any(weights_none):
            assert all(s.weight > 0 for s in weighted), (  # type: ignore[operator]
                "All weights must be positive"
            )
        return weighted

    @staticmethod
    def _resolve_weights(sources: tp.List[WeightedDataset]) -> tp.List[WeightedDataset]:
        """
        Resolve weights to explicit normalized floats.

        Rules:
          - all weights None -> proportional to len(dataset)
          - all weights set  -> normalize to sum=1
        """
        if all(s.weight is None for s in sources):
            lengths = [len(s.dataset) for s in sources]  # type: ignore[arg-type]
            total = sum(lengths)
            resolved = [ln / total for ln in lengths]
        else:
            total_w = sum(s.weight for s in sources)  # type: ignore[misc]
            resolved = [s.weight / total_w for s in sources]  # type: ignore[operator]

        return [WeightedDataset(dataset=s.dataset, weight=w) for s, w in zip(sources, resolved)]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int) -> tp.Any:
        """
        Select a dataset proportionally to its weight,
        then pick a random item from it.
        """
        r = torch.rand(1).item()
        dataset_idx = int(
            torch.searchsorted(self._cum_probs, r).clamp(0, len(self._datasets) - 1)
        )
        inner_idx = int(torch.randint(0, self._lengths[dataset_idx], (1,)).item())
        return self._datasets[dataset_idx][inner_idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def datasets(self) -> tp.List[Dataset]:
        return self._datasets

    @property
    def weighted(self) -> tp.List[WeightedDataset]:
        """Resolved WeightedDataset list with normalized weights."""
        return self._weighted

    @property
    def probabilities(self) -> tp.List[float]:
        return [s.weight for s in self._weighted]  # type: ignore[misc]

    def __repr__(self) -> str:
        parts = [
            f"  [{i}] {type(ds).__name__}(len={self._lengths[i]}, p={self._weighted[i].weight:.4f})"
            for i, ds in enumerate(self._datasets)
        ]
        return "AggregatedDataset(\n" + "\n".join(parts) + "\n)"
