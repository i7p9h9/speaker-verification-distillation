from __future__ import annotations

import typing as tp

import torch
from torch.utils.data import Dataset

from ._registry import _NameRegistry
from ._type import WeightedDataset


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

    Parameters
    ----------
    sources:
        Non-empty list of Dataset or WeightedDataset objects.
    name:
        Optional unique name for this dataset. Auto-generated if not provided.
    """

    def __init__(
        self,
        sources: tp.Union[tp.List[Dataset], tp.List[WeightedDataset]],
        name: tp.Optional[str] = None,
    ) -> None:
        assert len(sources) > 0, "At least one dataset must be provided"

        self._name: str = _NameRegistry.register(name, type(self).__name__)
        self._weighted: tp.List[WeightedDataset] = self._normalize(sources)
        self._datasets: tp.List[Dataset] = [s.dataset for s in self._weighted]
        self._lengths: tp.List[int] = [len(ds) for ds in self._datasets]  # type: ignore[arg-type]
        self._total: int = sum(self._lengths)
        self._rebuild_cum_probs()

    # ------------------------------------------------------------------
    # Name
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Normalization pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(
        sources: tp.Union[tp.List[Dataset], tp.List[WeightedDataset]]
    ) -> tp.List[WeightedDataset]:
        wrapped = AggregatedDataset._wrap(sources)
        return AggregatedDataset._resolve_weights(wrapped)

    @staticmethod
    def _wrap(
        sources: tp.Union[tp.List[Dataset], tp.List[WeightedDataset]]
    ) -> tp.List[WeightedDataset]:
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
        if all(s.weight is None for s in sources):
            lengths = [len(s.dataset) for s in sources]  # type: ignore[arg-type]
            total = sum(lengths)
            resolved = [ln / total for ln in lengths]
        else:
            total_w = sum(s.weight for s in sources)  # type: ignore[misc]
            resolved = [s.weight / total_w for s in sources]  # type: ignore[operator]

        # Resolve child names so each WeightedDataset carries a final name
        result: tp.List[WeightedDataset] = []
        for s, w in zip(sources, resolved):
            child_name = (
                s.name
                if s.name is not None
                else getattr(s.dataset, "name", None)
            )
            if child_name is None:
                child_name = _NameRegistry.register(
                    None, type(s.dataset).__name__
                )
            result.append(WeightedDataset(dataset=s.dataset, weight=w, name=child_name))
        return result

    def _rebuild_cum_probs(self) -> None:
        self._cum_probs = torch.tensor(
            [s.weight for s in self._weighted], dtype=torch.float64
        ).cumsum(dim=0)

    # ------------------------------------------------------------------
    # Weight API
    # ------------------------------------------------------------------

    def get_weights(self) -> tp.Dict[str, float]:
        """
        Return a flat dict of all (normalized) weights keyed by path.

        Paths use '/' as separator, e.g. ``"AggregatedDataset_0/TensorDataset_1"``.
        Inner AggregatedDataset children are expanded recursively.
        """
        out: tp.Dict[str, float] = {}
        self._collect_weights(prefix=self._name, out=out)
        return out

    def _collect_weights(self, prefix: str, out: tp.Dict[str, float]) -> None:
        for wd in self._weighted:
            child_name = wd.name or ""
            path = f"{prefix}/{child_name}"
            if isinstance(wd.dataset, AggregatedDataset):
                # Distribute this node's weight down into its children
                child_weights = wd.dataset.get_weights()
                for sub_path, sub_w in child_weights.items():
                    # sub_path starts with wd.dataset.name — strip that prefix
                    rel = sub_path[len(wd.dataset.name):]
                    out[f"{path}{rel}"] = float(wd.weight) * sub_w  # type: ignore[operator]
            else:
                out[path] = float(wd.weight)  # type: ignore[arg-type]

    def set_weights(self, weights: tp.Dict[str, float]) -> None:
        """
        Update weights from a flat path-keyed dict.

        All values must be positive floats (not None).
        All keys in *weights* must correspond to existing paths.
        Weights at each level are re-normalized independently after update.

        Example
        -------
        ::

            ds.set_weights({
                "root/source_a/sub_1": 2.0,
                "root/source_a/sub_2": 1.0,
                "root/source_b":       3.0,
            })
        """
        assert all(v is not None and v > 0 for v in weights.values()), (
            "All weight values must be positive floats (not None, not <= 0)"
        )
        self._apply_weights(prefix=self._name, weights=weights)

    def _apply_weights(self, prefix: str, weights: tp.Dict[str, float]) -> None:
        """
        Recursively apply *weights* starting from *prefix*.

        For each child we check whether the incoming dict addresses:
          (a) a direct child key  -> update weight of that child
          (b) grandchild keys     -> recurse into child AggregatedDataset
        After applying raw values, renormalize the affected level to sum=1.
        """
        raw: tp.Dict[int, float] = {}  # index -> new unnormalized weight

        for i, wd in enumerate(self._weighted):
            child_name = wd.name or ""
            direct_key = f"{prefix}/{child_name}"

            # Check direct child hit
            if direct_key in weights:
                raw[i] = weights[direct_key]

            # Check grandchild hits (recurse)
            if isinstance(wd.dataset, AggregatedDataset):
                child_prefix = direct_key
                child_keys = {
                    k: v for k, v in weights.items()
                    if k.startswith(child_prefix + "/")
                }
                if child_keys:
                    wd.dataset._apply_weights(child_prefix, child_keys)

        if not raw:
            return  # nothing changed at this level

        # Build new weight list: update addressed indices, keep others
        current = [float(wd.weight) for wd in self._weighted]  # type: ignore[arg-type]
        for i, v in raw.items():
            current[i] = v

        assert all(w > 0 for w in current), (
            "After update all weights must remain positive"
        )
        total = sum(current)
        for i, wd in enumerate(self._weighted):
            self._weighted[i] = WeightedDataset(
                dataset=wd.dataset,
                weight=current[i] / total,
                name=wd.name,
            )
        self._rebuild_cum_probs()

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
            torch.searchsorted(self._cum_probs, torch.tensor(r)).clamp(
                0, len(self._datasets) - 1
            )
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
        return self._weighted

    @property
    def probabilities(self) -> tp.List[float]:
        return [s.weight for s in self._weighted]  # type: ignore[misc]

    def __repr__(self) -> str:
        parts = [
            f"  [{i}] {wd.name}(len={self._lengths[i]}, p={wd.weight:.4f})"
            for i, wd in enumerate(self._weighted)
        ]
        return f"{type(self).__name__}(name={self._name!r},\n" + "\n".join(parts) + "\n)"
